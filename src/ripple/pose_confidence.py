import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.ripple.conformer_encoder import ConformerEncoding
from src.ripple.pocket_encoder import PocketEncoding
from src.ripple.utils import masked_softmax, scatter_pocket_nodes


@dataclass
class PocketConformerMatch:
    joint_logits: Tensor
    joint_weight: Tensor
    pocket_weight: Tensor
    conformer_weight: Tensor
    pair_context: Tensor
    residue_source: Tensor
    residue_source_strength: Tensor
    atom_attention: Tensor
    pose_confidence: Tensor
    pose_quality_logits: Tensor


class PocketConformerMatcher(nn.Module):
    """Score pocket-pose pairs and build the ligand-conditioned residue source.

    The joint distribution is a soft condition, not a gate. Pocket marginals
    drive pocket-aware DTI even when every pose is invalid; pose confidence
    only says how much the selected pose should be trusted downstream.
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.dim = dim
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.chemical_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(dim * 5 + 2),
            nn.Linear(dim * 5 + 2, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, 1),
        )
        self.pocket_query = nn.Linear(dim, dim, bias=False)
        self.atom_key = nn.Linear(dim, dim, bias=False)
        self.atom_value = nn.Linear(dim, dim, bias=False)
        self.source_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.pair_context_projection = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Pose quality is an auxiliary estimate used to re-rank poses. It is
        # trained only where an RMSD-to-crystal label exists.
        self.pose_quality_head = nn.Sequential(
            nn.LayerNorm(dim * 5 + 2),
            nn.Linear(dim * 5 + 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.confidence_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(
        self,
        conformers: ConformerEncoding,
        pockets: PocketEncoding,
        conformer_energy: Tensor,
        pocket_score: Tensor,
        chemical_context: Optional[Tensor] = None,
    ) -> PocketConformerMatch:
        pocket_global = pockets.pocket_global
        conformer_global = conformers.global_repr
        batch_size, pocket_count, dim = pocket_global.shape
        conformer_count = conformer_global.size(1)
        if dim != self.dim or conformer_global.size(-1) != self.dim:
            raise ValueError("matcher input embeddings do not match configured dim")
        if conformer_energy.shape != (batch_size, conformer_count):
            raise ValueError("conformer_energy must have shape [B, Kc]")
        if pocket_score.shape != (batch_size, pocket_count):
            raise ValueError("pocket_score must have shape [B, Kp]")

        if chemical_context is None:
            chemical_context = pocket_global.new_zeros(batch_size, dim)
        if chemical_context.shape != (batch_size, dim):
            raise ValueError("chemical_context must have shape [B, D]")
        chemical_context = self.chemical_projection(chemical_context)

        pocket_pair = pocket_global.unsqueeze(2).expand(-1, -1, conformer_count, -1)
        conformer_pair = conformer_global.unsqueeze(1).expand(-1, pocket_count, -1, -1)
        chemical_pair = chemical_context[:, None, None, :].expand(
            -1, pocket_count, conformer_count, -1
        )
        energy_pair = conformer_energy[:, None, :, None].expand(
            -1, pocket_count, -1, -1
        )
        pocket_score_pair = pocket_score[:, :, None, None].expand(
            -1, -1, conformer_count, -1
        )
        score_features = torch.cat(
            [
                pocket_pair,
                conformer_pair,
                pocket_pair * conformer_pair,
                torch.abs(pocket_pair - conformer_pair),
                chemical_pair,
                energy_pair.to(pocket_global.dtype),
                pocket_score_pair.to(pocket_global.dtype),
            ],
            dim=-1,
        )
        joint_logits = self.pair_scorer(score_features).squeeze(-1)
        pose_quality_logits = self.pose_quality_head(score_features).squeeze(-1)
        pair_mask = pockets.pocket_valid_mask.unsqueeze(
            -1
        ) & conformers.conformer_mask.unsqueeze(1)
        pose_available = conformers.conformer_mask.any(dim=-1)
        # Pocket selection must not depend on a pose existing. When a sample has
        # no valid pose, fall back to the pocket detector confidence renormalized
        # over the candidate pockets, so pocket-aware DTI still runs.
        temperature = self.log_temperature.exp().clamp(min=0.05, max=10.0)
        flat_weight = masked_softmax(
            (joint_logits / temperature).reshape(batch_size, -1),
            pair_mask.reshape(batch_size, -1),
            dim=-1,
        )
        joint_weight = flat_weight.reshape(batch_size, pocket_count, conformer_count)
        joint_pocket_weight = joint_weight.sum(dim=-1)
        pocket_prior = masked_softmax(
            pocket_score.to(pocket_global.dtype),
            pockets.pocket_valid_mask,
            dim=-1,
        )
        pocket_weight = torch.where(
            pose_available.unsqueeze(-1), joint_pocket_weight, pocket_prior
        )
        conformer_weight = joint_weight.sum(dim=-2)

        pocket_query = self.pocket_query(pockets.pocket_tokens)
        atom_key = self.atom_key(conformers.tokens)
        atom_value = self.atom_value(conformers.tokens)
        atom_logits = torch.einsum("bpnd,bcad->bpcna", pocket_query, atom_key)
        atom_logits = atom_logits / math.sqrt(self.dim)
        attention_mask = (
            pockets.pocket_mask[:, :, None, :, None]
            & conformers.atom_mask[:, None, :, None, :]
            & pair_mask[:, :, :, None, None]
        )
        atom_attention = masked_softmax(atom_logits, attention_mask, dim=-1)
        atom_context = torch.einsum("bpcna,bcad->bpcnd", atom_attention, atom_value)

        pocket_token_pair = pockets.pocket_tokens.unsqueeze(2).expand(
            -1, -1, conformer_count, -1, -1
        )
        source_features = torch.cat(
            [
                pocket_token_pair,
                atom_context,
                pocket_token_pair * atom_context,
                torch.abs(pocket_token_pair - atom_context),
            ],
            dim=-1,
        )
        pair_source = self.source_mlp(source_features)
        pair_source = pair_source * attention_mask.any(dim=-1).unsqueeze(-1).to(
            pair_source.dtype
        )
        pocket_source = torch.sum(
            joint_weight[:, :, :, None, None] * pair_source,
            dim=2,
        )
        pocket_source = pocket_source * pockets.pocket_mask.unsqueeze(-1).to(
            pocket_source.dtype
        )
        residue_source = scatter_pocket_nodes(
            pocket_source,
            pockets.pocket_indices,
            pockets.pocket_mask,
            pockets.residue_tokens.size(1),
        )
        residue_source = residue_source * pockets.residue_mask.unsqueeze(-1).to(
            residue_source.dtype
        )
        residue_source_strength = torch.linalg.vector_norm(residue_source, dim=-1)

        selected_pocket = torch.sum(
            joint_weight.unsqueeze(-1) * pocket_pair,
            dim=(1, 2),
        )
        selected_conformer = torch.sum(
            joint_weight.unsqueeze(-1) * conformer_pair,
            dim=(1, 2),
        )
        pair_context = self.pair_context_projection(
            torch.cat(
                [
                    selected_pocket,
                    selected_conformer,
                    selected_pocket * selected_conformer,
                    torch.abs(selected_pocket - selected_conformer),
                ],
                dim=-1,
            )
        )
        # Where no pose exists, drop the pose-dependent context but keep a
        # pocket-only context, so conditioning degrades instead of vanishing.
        pose_confidence = torch.sigmoid(self.confidence_head(pair_context)).squeeze(-1)
        pose_confidence = pose_confidence * pose_available.to(pose_confidence.dtype)
        pocket_only_context = self.pair_context_projection(
            torch.cat(
                [
                    selected_pocket,
                    selected_pocket.new_zeros(batch_size, dim),
                    selected_pocket.new_zeros(batch_size, dim),
                    selected_pocket.new_zeros(batch_size, dim),
                ],
                dim=-1,
            )
        )
        pair_context = torch.where(
            pose_available.unsqueeze(-1), pair_context, pocket_only_context
        )

        return PocketConformerMatch(
            joint_logits=joint_logits,
            joint_weight=joint_weight,
            pocket_weight=pocket_weight,
            conformer_weight=conformer_weight,
            pair_context=pair_context,
            residue_source=residue_source,
            residue_source_strength=residue_source_strength,
            atom_attention=atom_attention,
            pose_confidence=pose_confidence,
            pose_quality_logits=pose_quality_logits,
        )
