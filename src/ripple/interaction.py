from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from src.ripple.biintention import BiIntention
from src.ripple.utils import ensure_bool_mask, masked_mean


class ProteinStructureFusion(nn.Module):
    """Create the single residue state used by pockets and propagation."""

    def __init__(
        self,
        residue_input_dim: int,
        dim: int = 128,
        dropout: float = 0.1,
        structure_gate_bias: float = -1.0,
    ):
        super().__init__()
        self.dim = dim
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(residue_input_dim),
            nn.Linear(residue_input_dim, dim),
        )
        self.structure_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        nn.init.constant_(self.structure_gate[-1].bias, structure_gate_bias)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        residue_features: Tensor,
        structural_tokens: Tensor,
        residue_mask: Tensor,
        sequence_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if sequence_tokens is None:
            sequence_tokens = self.sequence_projection(residue_features)
        if sequence_tokens.shape != structural_tokens.shape:
            raise ValueError(
                "sequence_tokens and structural_tokens must both have shape [B, L, D]"
            )
        residue_mask = ensure_bool_mask(residue_mask, structural_tokens.device)
        gate = torch.sigmoid(
            self.structure_gate(torch.cat([sequence_tokens, structural_tokens], dim=-1))
        )
        fused = self.output_norm(sequence_tokens + gate * structural_tokens)
        fused = fused * residue_mask.unsqueeze(-1).to(fused.dtype)
        return fused, gate


class MaskedLatentResampler(nn.Module):
    """Compress variable-length tokens into a fixed latent set."""

    def __init__(
        self,
        dim: int = 128,
        num_latents: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_latents < 1:
            raise ValueError("num_latents must be positive")
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens and mask must have shapes [B, N, D] and [B, N]")
        mask = ensure_bool_mask(mask, tokens.device)
        empty = ~mask.any(dim=-1)
        safe_mask = mask.clone()
        safe_tokens = tokens
        if empty.any():
            safe_mask[empty, 0] = True
            safe_tokens = tokens.clone()
            safe_tokens[empty, 0] = 0.0

        queries = self.latents.unsqueeze(0).expand(tokens.size(0), -1, -1)
        attended, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(safe_tokens),
            safe_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        output = queries + attended
        output = output + self.feed_forward(output)
        return output * (~empty).view(-1, 1, 1).to(output.dtype)


class DrugModalFusion(nn.Module):
    """Fuse 1D and 2D drug tokens with the pose-confidence-weighted conformer."""

    def __init__(
        self,
        dim: int = 128,
        num_latents: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.modality_embedding = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.chemical_fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.resampler = MaskedLatentResampler(
            dim=dim,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout,
        )

    def chemical_context(
        self,
        one_d_tokens: Tensor,
        one_d_mask: Tensor,
        two_d_tokens: Tensor,
        two_d_mask: Tensor,
    ) -> Tensor:
        if one_d_tokens.size(-1) != self.dim or two_d_tokens.size(-1) != self.dim:
            raise ValueError("drug 1D and 2D tokens must use model dim")
        one_d = masked_mean(one_d_tokens, one_d_mask, dim=1)
        two_d = masked_mean(two_d_tokens, two_d_mask, dim=1)
        return self.chemical_fusion(
            torch.cat([one_d, two_d, one_d * two_d, torch.abs(one_d - two_d)], dim=-1)
        )

    def forward(
        self,
        one_d_tokens: Tensor,
        one_d_mask: Tensor,
        two_d_tokens: Tensor,
        two_d_mask: Tensor,
        conformer_tokens: Tensor,
        atom_mask: Tensor,
        conformer_weight: Tensor,
    ) -> Tensor:
        selected_3d = torch.sum(
            conformer_weight[:, :, None, None] * conformer_tokens,
            dim=1,
        )
        selected_atom_weight = torch.sum(
            conformer_weight[:, :, None] * atom_mask.to(conformer_weight.dtype),
            dim=1,
        )
        selected_atom_mask = selected_atom_weight > 0
        tokens = torch.cat(
            [
                one_d_tokens + self.modality_embedding[0],
                two_d_tokens + self.modality_embedding[1],
                selected_3d + self.modality_embedding[2],
            ],
            dim=1,
        )
        mask = torch.cat([one_d_mask, two_d_mask, selected_atom_mask], dim=1)
        return self.resampler(tokens, mask)


@dataclass
class InteractionEncoding:
    global_context: Tensor
    drug_tokens: Tensor
    protein_tokens: Tensor
    drug_film: Tensor
    protein_film: Tensor


class PocketConditionedBiIntention(nn.Module):
    """Condition both BiIntention inputs on the selected local interaction."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.protein_film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        nn.init.zeros_(self.drug_film[-1].weight)
        nn.init.zeros_(self.drug_film[-1].bias)
        nn.init.zeros_(self.protein_film[-1].weight)
        nn.init.zeros_(self.protein_film[-1].bias)
        self.dropout = nn.Dropout(dropout)
        self.biintention = BiIntention(
            embed_dim=dim,
            layer=num_layers,
            num_head=num_heads,
            device="cpu",
        )

    @staticmethod
    def _apply_film(tokens: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.chunk(2, dim=-1)
        return tokens * (
            1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)
        ) + 0.1 * shift.unsqueeze(1)

    def forward(
        self,
        drug_tokens: Tensor,
        protein_tokens: Tensor,
        pair_context: Tensor,
        pose_confidence: Tensor,
    ) -> InteractionEncoding:
        # Pose confidence softly scales how much the selected pose shifts the
        # interaction, it never gates it. At zero confidence the FiLM terms
        # vanish and both inputs pass through unchanged, so the pocket-aware
        # DTI path still produces a finite prediction without any pose.
        pose_confidence = pose_confidence.to(pair_context.dtype).view(-1, 1)
        drug_film = self.drug_film(pair_context) * pose_confidence
        protein_film = self.protein_film(pair_context) * pose_confidence
        conditioned_drug = self.dropout(self._apply_film(drug_tokens, drug_film))
        conditioned_protein = self.dropout(
            self._apply_film(protein_tokens, protein_film)
        )
        global_context, drug_output, protein_output, _ = self.biintention(
            conditioned_drug,
            conditioned_protein,
        )
        return InteractionEncoding(
            global_context=global_context,
            drug_tokens=drug_output,
            protein_tokens=protein_output,
            drug_film=drug_film,
            protein_film=protein_film,
        )


class InteractionSourceFusion(nn.Module):
    """Use global interaction context to modulate the local pocket source."""

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim)
        )
        self.source_gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.context_fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        local_source: Tensor,
        pair_context: Tensor,
        global_context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        global_state = self.global_projection(global_context)
        expanded_global = global_state.unsqueeze(1).expand(-1, local_source.size(1), -1)
        expanded_pair = pair_context.unsqueeze(1).expand_as(expanded_global)
        gate = self.source_gate(
            torch.cat([local_source, expanded_pair, expanded_global], dim=-1)
        )
        fused_source = local_source * (0.5 + gate)
        propagation_context = self.context_fusion(
            torch.cat(
                [
                    pair_context,
                    global_state,
                    pair_context * global_state,
                    torch.abs(pair_context - global_state),
                ],
                dim=-1,
            )
        )
        return fused_source, propagation_context
