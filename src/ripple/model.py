from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, cat, nn

from src.ripple.conformer_encoder import ConformerEncoding, MultiConformerEncoder
from src.ripple.interaction import (
    DrugModalFusion,
    InteractionEncoding,
    InteractionSourceFusion,
    MaskedLatentResampler,
    PocketConditionedBiIntention,
    ProteinStructureFusion,
)
from src.ripple.pocket_encoder import PocketEncoding, ResiduePocketEncoder
from src.ripple.pose_confidence import (
    PocketConformerMatch,
    PocketConformerMatcher,
)
from src.ripple.propagation import (
    DifferentialResiduePropagation,
    PropagationOutput,
)
from src.ripple.structure_response import GeometryQueries, StructureResponseHead
from src.ripple.task_heads import RIPPLETaskHeads


@dataclass
class RIPPLEBatch:
    atom_features: Tensor
    atom_coords: Tensor
    atom_mask: Tensor
    conformer_mask: Tensor
    conformer_energy: Tensor
    drug_1d_tokens: Tensor
    drug_1d_mask: Tensor
    drug_2d_tokens: Tensor
    drug_2d_mask: Tensor
    residue_features: Tensor
    residue_coords: Tensor
    residue_mask: Tensor
    edge_index: Tensor
    edge_mask: Tensor
    pocket_indices: Tensor
    pocket_mask: Tensor
    pocket_valid_mask: Tensor
    pocket_score: Tensor
    edge_features: Optional[Tensor] = None
    protein_sequence_tokens: Optional[Tensor] = None
    function_mask: Optional[Tensor] = None
    endpoint_type: Optional[Tensor] = None
    geometry_queries: Optional[GeometryQueries] = None


@dataclass
class GeometryBatch:
    """Protein-only batch; nodes identify a chain copy and canonical position."""

    residue_features: Tensor
    residue_coords: Tensor
    residue_mask: Tensor
    edge_index: Tensor
    edge_mask: Tensor
    geometry_queries: GeometryQueries
    edge_features: Optional[Tensor] = None
    protein_sequence_tokens: Optional[Tensor] = None


@dataclass
class GeometryOutput:
    predictions: Dict[str, Tensor]
    apo_tokens: Tensor
    structure_gate: Tensor


@dataclass
class RIPPLEOutput:
    predictions: Dict[str, Tensor]
    conformers: ConformerEncoding
    pockets: PocketEncoding
    match: PocketConformerMatch
    interaction: InteractionEncoding
    propagation: PropagationOutput
    apo_tokens: Tensor
    structure_gate: Tensor
    fused_source: Tensor


class RIPPLEDTI(nn.Module):
    """Pocket-aware DTI with ligand-conditioned pocket-to-function propagation.

    Candidate pockets always condition the interaction representation. Pose
    confidence is a soft auxiliary condition: it modulates that representation
    but the DTI path stays intact when no pose is available.
    """

    def __init__(
        self,
        atom_input_dim: int,
        residue_input_dim: int,
        dim: int = 128,
        edge_dim: int = 0,
        conformer_layers: int = 3,
        conformer_neighbors: int = 16,
        residue_layers: int = 3,
        propagation_steps: int = 5,
        num_heads: int = 8,
        biintention_layers: int = 1,
        drug_latents: int = 32,
        protein_latents: int = 32,
        num_affinity_endpoints: int = 4,
        num_effect_classes: int = 2,
        num_mechanism_classes: int = 6,
        dropout: float = 0.1,
        matcher_temperature: float = 1.0,
        propagation_restart: float = 0.2,
        structure_gate_bias: float = -1.0,
        local_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.dim = dim
        self.conformer_encoder = MultiConformerEncoder(
            atom_input_dim=atom_input_dim,
            dim=dim,
            num_layers=conformer_layers,
            num_nearest_neighbors=conformer_neighbors,
            dropout=dropout,
        )
        self.residue_encoder = ResiduePocketEncoder(
            residue_input_dim=residue_input_dim,
            dim=dim,
            edge_dim=edge_dim,
            num_layers=residue_layers,
            dropout=dropout,
        )
        self.protein_fusion = ProteinStructureFusion(
            residue_input_dim=residue_input_dim,
            dim=dim,
            dropout=dropout,
            structure_gate_bias=structure_gate_bias,
        )
        self.matcher = PocketConformerMatcher(
            dim=dim,
            dropout=dropout,
            temperature=matcher_temperature,
        )
        self.drug_fusion = DrugModalFusion(
            dim=dim,
            num_latents=drug_latents,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.protein_resampler = MaskedLatentResampler(
            dim=dim,
            num_latents=protein_latents,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.interaction = PocketConditionedBiIntention(
            dim=dim,
            num_heads=num_heads,
            num_layers=biintention_layers,
            dropout=dropout,
        )
        self.source_fusion = InteractionSourceFusion(dim=dim, dropout=dropout)
        self.propagation = DifferentialResiduePropagation(
            dim=dim,
            edge_dim=edge_dim,
            num_steps=propagation_steps,
            restart=propagation_restart,
            dropout=dropout,
        )
        self.task_heads = RIPPLETaskHeads(
            dim=dim,
            num_affinity_endpoints=num_affinity_endpoints,
            num_effect_classes=num_effect_classes,
            num_mechanism_classes=num_mechanism_classes,
            dropout=dropout,
            local_gate_bias=local_gate_bias,
        )
        self.structure_response = StructureResponseHead(dim, dropout)
        self.gating_head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 3))

    def encode_protein(self, batch: RIPPLEBatch | GeometryBatch):
        mask = batch.residue_mask.bool()
        if not mask.any(-1).all():
            raise ValueError("Every sample requires mapped residues with coordinates")
        if not torch.isfinite(batch.residue_coords[mask]).all():
            raise ValueError("Valid residue coordinates must be finite")
        if not torch.isfinite(batch.residue_features[mask]).all():
            raise ValueError("Valid residue features must be finite")
        # Missing coordinates may be NaN in the cache; sanitize before graph arithmetic.
        features = torch.where(mask[..., None], batch.residue_features, 0.0)
        coords = torch.where(mask[..., None], batch.residue_coords, 0.0)
        structural_tokens = self.residue_encoder.encode_residues(
            residue_features=features,
            residue_coords=coords,
            residue_mask=mask,
            edge_index=batch.edge_index,
            edge_mask=batch.edge_mask,
            edge_features=batch.edge_features,
        )
        sequence_tokens = batch.protein_sequence_tokens
        if sequence_tokens is not None:
            sequence_tokens = torch.where(mask[..., None], sequence_tokens, 0.0)
        return self.protein_fusion(
            residue_features=features,
            structural_tokens=structural_tokens,
            residue_mask=mask,
            sequence_tokens=sequence_tokens,
        )

    def forward(self, batch: RIPPLEBatch | GeometryBatch) -> RIPPLEOutput | GeometryOutput:
        apo_tokens, structure_gate = self.encode_protein(batch)
        if isinstance(batch, GeometryBatch):
            predictions = self.structure_response.snapshot(
                apo_tokens, batch.residue_coords, batch.residue_mask, batch.geometry_queries,
            )
            return GeometryOutput(predictions, apo_tokens, structure_gate)

        conformers = self.conformer_encoder(
            batch.atom_features, batch.atom_coords, batch.atom_mask,
            batch.conformer_mask, batch.conformer_energy,
        )
        pockets = self.residue_encoder.extract_pockets(
            residue_tokens=apo_tokens,
            residue_mask=batch.residue_mask,
            pocket_indices=batch.pocket_indices,
            pocket_mask=batch.pocket_mask,
            pocket_valid_mask=batch.pocket_valid_mask,
        )

        chemical_context = self.drug_fusion.chemical_context(
            one_d_tokens=batch.drug_1d_tokens,
            one_d_mask=batch.drug_1d_mask,
            two_d_tokens=batch.drug_2d_tokens,
            two_d_mask=batch.drug_2d_mask,
        )
        match = self.matcher(
            conformers=conformers,
            pockets=pockets,
            conformer_energy=batch.conformer_energy,
            pocket_score=batch.pocket_score,
            chemical_context=chemical_context,
        )

        drug_tokens = self.drug_fusion(
            one_d_tokens=batch.drug_1d_tokens,
            one_d_mask=batch.drug_1d_mask,
            two_d_tokens=batch.drug_2d_tokens,
            two_d_mask=batch.drug_2d_mask,
            conformer_tokens=conformers.tokens,
            atom_mask=conformers.atom_mask,
            conformer_weight=match.conformer_weight,
        )
        protein_tokens = self.protein_resampler(apo_tokens, pockets.residue_mask)
        interaction = self.interaction(
            drug_tokens=drug_tokens,
            protein_tokens=protein_tokens,
            pair_context=match.pair_context,
            pose_confidence=match.pose_confidence,
        )

        fused_source, propagation_context = self.source_fusion(
            local_source=match.residue_source,
            pair_context=match.pair_context,
            global_context=interaction.global_context,
        )
        propagation = self.propagation(
            residue_tokens=apo_tokens,
            residue_coords=torch.where(batch.residue_mask[..., None], batch.residue_coords, 0.0),
            residue_mask=pockets.residue_mask,
            edge_index=batch.edge_index,
            edge_mask=batch.edge_mask,
            residue_source=fused_source,
            pair_context=propagation_context,
            function_mask=batch.function_mask,
            edge_features=batch.edge_features,
        )
        predictions = self.task_heads(
            global_context=interaction.global_context,
            pair_context=match.pair_context,
            function_state=propagation.function_state,
            pocket_global=pockets.pocket_global,
            pocket_valid_mask=pockets.pocket_valid_mask,
            endpoint_type=batch.endpoint_type,
        )
        predictions["joint_pocket_conformer_logits"] = match.joint_logits
        # Direction classes: promotes opening, promotes closing, no gating change.
        # Pore blockade and generic inhibition are not mapped into these classes.
        predictions["gating_logits"] = self.gating_head(
            cat([propagation.function_state, chemical_context], dim=-1)
        )
        if batch.geometry_queries is not None:
            predictions.update(
                self.structure_response(
                    apo_tokens, propagation.delta_tokens, chemical_context,
                    batch.residue_coords, pockets.residue_mask, batch.geometry_queries,
                )
            )

        return RIPPLEOutput(
            predictions=predictions,
            conformers=conformers,
            pockets=pockets,
            match=match,
            interaction=interaction,
            propagation=propagation,
            apo_tokens=apo_tokens,
            structure_gate=structure_gate,
            fused_source=fused_source,
        )
