from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.ripple.utils import ensure_bool_mask, masked_softmax


@dataclass
class PropagationOutput:
    ligand_on_tokens: Tensor
    delta_tokens: Tensor
    pocket_state: Tensor
    function_state: Tensor
    sink_weight: Tensor
    sink_logits: Tensor
    edge_flux: Tensor


class DifferentialResiduePropagation(nn.Module):
    """Personalized sparse diffusion from ligand-conditioned pocket sources."""

    def __init__(
        self,
        dim: int = 128,
        edge_dim: int = 0,
        num_steps: int = 5,
        num_rbf: int = 16,
        restart: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < restart <= 1.0:
            raise ValueError("restart must be in (0, 1]")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.dim = dim
        self.edge_dim = edge_dim
        self.num_steps = num_steps
        self.restart = restart
        edge_input_dim = dim * 3 + edge_dim + num_rbf
        self.source_projection = nn.Linear(dim, dim, bias=False)
        self.message_projection = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_steps)]
        )
        self.edge_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(edge_input_dim),
                    nn.Linear(edge_input_dim, dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_steps)
            ]
        )
        self.delta_projection = nn.Linear(dim, dim, bias=False)
        self.delta_gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.sink_score = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, 20.0, num_rbf))
        self.rbf_gamma = 0.25

    def _prepare_edges(
        self,
        residue_tokens: Tensor,
        residue_coords: Tensor,
        residue_mask: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        pair_context: Tensor,
        edge_features: Optional[Tensor],
    ):
        batch_size, residue_count, _ = residue_tokens.shape
        edge_count = edge_index.size(-1)
        batch_index = torch.arange(batch_size, device=residue_tokens.device).unsqueeze(
            -1
        )
        raw_source = edge_index[:, 0]
        raw_target = edge_index[:, 1]
        source = raw_source.clamp(min=0, max=residue_count - 1)
        target = raw_target.clamp(min=0, max=residue_count - 1)
        edge_mask = ensure_bool_mask(edge_mask, residue_tokens.device)
        edge_mask = edge_mask & (raw_source >= 0) & (raw_source < residue_count)
        edge_mask = edge_mask & (raw_target >= 0) & (raw_target < residue_count)
        edge_mask = (
            edge_mask
            & residue_mask[batch_index, source]
            & residue_mask[batch_index, target]
        )

        source_nodes = residue_tokens[batch_index, source]
        target_nodes = residue_tokens[batch_index, target]
        distance = torch.linalg.vector_norm(
            residue_coords[batch_index, source] - residue_coords[batch_index, target],
            dim=-1,
        )
        centers = self.rbf_centers.to(
            dtype=residue_tokens.dtype, device=residue_tokens.device
        )
        rbf = torch.exp(-self.rbf_gamma * (distance.unsqueeze(-1) - centers) ** 2)
        context = pair_context.unsqueeze(1).expand(-1, edge_count, -1)
        parts = [source_nodes, target_nodes, context, rbf]
        if self.edge_dim:
            if edge_features is None:
                edge_features = residue_tokens.new_zeros(
                    batch_size, edge_count, self.edge_dim
                )
            if edge_features.shape != (batch_size, edge_count, self.edge_dim):
                raise ValueError("edge_features does not match configured edge_dim")
            parts.append(edge_features)
        return source, target, edge_mask, torch.cat(parts, dim=-1)

    def forward(
        self,
        residue_tokens: Tensor,
        residue_coords: Tensor,
        residue_mask: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        residue_source: Tensor,
        pair_context: Tensor,
        function_mask: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
    ) -> PropagationOutput:
        # function_mask is accepted and ignored. Functional annotations are a
        # training target (see RIPPLEMultiTaskLoss), never an input: feeding them
        # in would let the model read the answer instead of propagating to it.
        del function_mask
        if residue_tokens.shape != residue_source.shape:
            raise ValueError(
                "residue_tokens and residue_source must have the same shape"
            )
        if edge_index.ndim != 3 or edge_index.size(1) != 2:
            raise ValueError("edge_index must have shape [B, 2, E]")
        batch_size, residue_count, dim = residue_tokens.shape
        if dim != self.dim or pair_context.shape != (batch_size, dim):
            raise ValueError("propagation embeddings do not match configured dim")

        residue_mask = ensure_bool_mask(residue_mask, residue_tokens.device)
        source_index, target_index, valid_edge, edge_input = self._prepare_edges(
            residue_tokens,
            residue_coords,
            residue_mask,
            edge_index,
            edge_mask,
            pair_context,
            edge_features,
        )
        batch_index = torch.arange(batch_size, device=residue_tokens.device).unsqueeze(
            -1
        )
        source_state = torch.tanh(self.source_projection(residue_source))
        source_state = source_state * residue_mask.unsqueeze(-1).to(source_state.dtype)
        state = source_state
        flux_history = []

        for message_projection, edge_gate in zip(
            self.message_projection, self.edge_gates
        ):
            gate = edge_gate(edge_input).squeeze(-1)
            gate = gate * valid_edge.to(gate.dtype)
            raw_message = message_projection(state[batch_index, source_index])
            message = raw_message * gate.unsqueeze(-1)

            aggregated = state.new_zeros(batch_size, residue_count, dim)
            aggregated.scatter_add_(
                1,
                target_index.unsqueeze(-1).expand(-1, -1, dim),
                message,
            )
            normalizer = state.new_zeros(batch_size, residue_count, 1)
            normalizer.scatter_add_(1, target_index.unsqueeze(-1), gate.unsqueeze(-1))
            aggregated = aggregated / normalizer.clamp_min(1e-8)
            state = torch.tanh(
                self.restart * source_state + (1.0 - self.restart) * aggregated
            )
            state = state * residue_mask.unsqueeze(-1).to(state.dtype)
            flux_history.append(gate * torch.linalg.vector_norm(raw_message, dim=-1))

        context_per_residue = pair_context.unsqueeze(1).expand(-1, residue_count, -1)
        delta_gate = self.delta_gate(
            torch.cat([residue_tokens, state, context_per_residue], dim=-1)
        )
        delta_tokens = delta_gate * self.delta_projection(state)
        delta_tokens = delta_tokens * residue_mask.unsqueeze(-1).to(delta_tokens.dtype)
        ligand_on_tokens = residue_tokens + delta_tokens

        source_strength = torch.linalg.vector_norm(residue_source, dim=-1)
        source_present = (source_strength > 0) & residue_mask
        pocket_weight = masked_softmax(source_strength, source_present, dim=-1)
        pocket_state = torch.sum(pocket_weight.unsqueeze(-1) * delta_tokens, dim=1)

        # The sink is predicted from the propagated state, so functional-site
        # supervision can be applied without leaking the annotation into the
        # forward pass. Annotating it as an input would make the prediction
        # trivial and is intentionally not supported.
        sink_logits = self.sink_score(
            torch.cat([ligand_on_tokens, context_per_residue], dim=-1)
        ).squeeze(-1)
        sink_weight = masked_softmax(sink_logits, residue_mask, dim=-1)
        function_state = torch.sum(sink_weight.unsqueeze(-1) * delta_tokens, dim=1)

        return PropagationOutput(
            ligand_on_tokens=ligand_on_tokens,
            delta_tokens=delta_tokens,
            pocket_state=pocket_state,
            function_state=function_state,
            sink_weight=sink_weight,
            sink_logits=sink_logits,
            edge_flux=torch.stack(flux_history, dim=1),
        )
