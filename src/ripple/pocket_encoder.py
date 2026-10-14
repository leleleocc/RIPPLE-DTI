from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.ripple.utils import (
    ensure_bool_mask,
    gather_nodes,
    masked_softmax,
)


@dataclass
class PocketEncoding:
    residue_tokens: Tensor
    pocket_tokens: Tensor
    pocket_global: Tensor
    residue_mask: Tensor
    pocket_mask: Tensor
    pocket_valid_mask: Tensor
    pocket_indices: Tensor


class SparseResidueLayer(nn.Module):
    """Sparse geometry-aware message passing over a padded edge list."""

    def __init__(
        self, dim: int, edge_dim: int = 0, num_rbf: int = 16, dropout: float = 0.1
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_rbf = num_rbf
        edge_input_dim = dim * 2 + edge_dim + num_rbf
        self.message = nn.Sequential(
            nn.Linear(edge_input_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_input_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("rbf_centers", torch.linspace(0.0, 20.0, num_rbf))
        self.rbf_gamma = 0.25

    def _edge_geometry(self, coords: Tensor, edge_index: Tensor) -> Tensor:
        batch_size = edge_index.size(0)
        batch_index = torch.arange(batch_size, device=coords.device).unsqueeze(-1)
        source = edge_index[:, 0].clamp(min=0, max=coords.size(1) - 1)
        target = edge_index[:, 1].clamp(min=0, max=coords.size(1) - 1)
        distance = torch.linalg.vector_norm(
            coords[batch_index, source] - coords[batch_index, target],
            dim=-1,
        )
        centers = self.rbf_centers.to(dtype=coords.dtype, device=coords.device)
        return torch.exp(-self.rbf_gamma * (distance.unsqueeze(-1) - centers) ** 2)

    def forward(
        self,
        nodes: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        node_mask: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index.ndim != 3 or edge_index.size(1) != 2:
            raise ValueError("edge_index must have shape [B, 2, E]")
        batch_size, residue_count, dim = nodes.shape
        edge_count = edge_index.size(-1)
        batch_index = torch.arange(batch_size, device=nodes.device).unsqueeze(-1)
        raw_source = edge_index[:, 0]
        raw_target = edge_index[:, 1]
        source = raw_source.clamp(min=0, max=residue_count - 1)
        target = raw_target.clamp(min=0, max=residue_count - 1)

        node_mask = ensure_bool_mask(node_mask, nodes.device)
        edge_mask = ensure_bool_mask(edge_mask, nodes.device)
        edge_mask = edge_mask & (raw_source >= 0) & (raw_source < residue_count)
        edge_mask = edge_mask & (raw_target >= 0) & (raw_target < residue_count)
        edge_mask = (
            edge_mask & node_mask[batch_index, source] & node_mask[batch_index, target]
        )

        source_nodes = nodes[batch_index, source]
        target_nodes = nodes[batch_index, target]
        parts = [source_nodes, target_nodes, self._edge_geometry(coords, edge_index)]
        if self.edge_dim:
            if edge_features is None:
                edge_features = nodes.new_zeros(batch_size, edge_count, self.edge_dim)
            if edge_features.shape != (batch_size, edge_count, self.edge_dim):
                raise ValueError("edge_features does not match configured edge_dim")
            parts.append(edge_features)
        edge_input = torch.cat(parts, dim=-1)

        messages = self.message(edge_input) * self.edge_gate(edge_input)
        messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)
        aggregated = nodes.new_zeros(batch_size, residue_count, dim)
        aggregated.scatter_add_(1, target.unsqueeze(-1).expand(-1, -1, dim), messages)

        degree = nodes.new_zeros(batch_size, residue_count, 1)
        degree.scatter_add_(
            1,
            target.unsqueeze(-1),
            edge_mask.to(dtype=nodes.dtype).unsqueeze(-1),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        updated = self.norm(nodes + self.update(torch.cat([nodes, aggregated], dim=-1)))
        return updated * node_mask.unsqueeze(-1).to(updated.dtype)


class ResiduePocketEncoder(nn.Module):
    def __init__(
        self,
        residue_input_dim: int,
        dim: int = 128,
        edge_dim: int = 0,
        num_layers: int = 3,
        num_rbf: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.residue_projection = nn.Linear(residue_input_dim, dim)
        self.layers = nn.ModuleList(
            [
                SparseResidueLayer(dim, edge_dim, num_rbf, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.pocket_pool_score = nn.Linear(dim, 1)

    def encode_residues(
        self,
        residue_features: Tensor,
        residue_coords: Tensor,
        residue_mask: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        if residue_features.ndim != 3:
            raise ValueError("residue_features must have shape [B, L, F]")
        if residue_coords.shape != residue_features.shape[:-1] + (3,):
            raise ValueError("residue_coords must have shape [B, L, 3]")

        residue_mask = ensure_bool_mask(residue_mask, residue_features.device)
        nodes = self.residue_projection(residue_features)
        nodes = nodes * residue_mask.unsqueeze(-1).to(nodes.dtype)
        for layer in self.layers:
            nodes = layer(
                nodes,
                residue_coords,
                edge_index,
                edge_mask,
                residue_mask,
                edge_features,
            )
        nodes = self.output_norm(nodes)
        nodes = nodes * residue_mask.unsqueeze(-1).to(nodes.dtype)
        return nodes

    def extract_pockets(
        self,
        residue_tokens: Tensor,
        residue_mask: Tensor,
        pocket_indices: Tensor,
        pocket_mask: Tensor,
        pocket_valid_mask: Tensor,
    ) -> PocketEncoding:
        if residue_tokens.ndim != 3:
            raise ValueError("residue_tokens must have shape [B, L, D]")
        if pocket_indices.shape != pocket_mask.shape:
            raise ValueError("pocket_indices and pocket_mask must have the same shape")

        residue_mask = ensure_bool_mask(residue_mask, residue_tokens.device)
        pocket_mask = ensure_bool_mask(pocket_mask, residue_tokens.device)
        pocket_valid_mask = ensure_bool_mask(pocket_valid_mask, residue_tokens.device)
        pocket_mask = (
            pocket_mask
            & (pocket_indices >= 0)
            & (pocket_indices < residue_tokens.size(1))
        )

        pocket_tokens = gather_nodes(residue_tokens, pocket_indices)
        gathered_residue_mask = (
            gather_nodes(
                residue_mask.unsqueeze(-1).to(residue_tokens.dtype), pocket_indices
            )
            .squeeze(-1)
            .bool()
        )
        pocket_mask = (
            pocket_mask & gathered_residue_mask & pocket_valid_mask.unsqueeze(-1)
        )
        pocket_tokens = pocket_tokens * pocket_mask.unsqueeze(-1).to(
            residue_tokens.dtype
        )
        pocket_weight = masked_softmax(
            self.pocket_pool_score(pocket_tokens).squeeze(-1),
            pocket_mask,
            dim=-1,
        )
        pocket_global = torch.sum(pocket_weight.unsqueeze(-1) * pocket_tokens, dim=-2)
        pocket_valid_mask = pocket_valid_mask & pocket_mask.any(dim=-1)
        pocket_global = pocket_global * pocket_valid_mask.unsqueeze(-1).to(
            residue_tokens.dtype
        )

        return PocketEncoding(
            residue_tokens=residue_tokens,
            pocket_tokens=pocket_tokens,
            pocket_global=pocket_global,
            residue_mask=residue_mask,
            pocket_mask=pocket_mask,
            pocket_valid_mask=pocket_valid_mask,
            pocket_indices=pocket_indices,
        )

    def forward(
        self,
        residue_features: Tensor,
        residue_coords: Tensor,
        residue_mask: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        pocket_indices: Tensor,
        pocket_mask: Tensor,
        pocket_valid_mask: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> PocketEncoding:
        residue_tokens = self.encode_residues(
            residue_features=residue_features,
            residue_coords=residue_coords,
            residue_mask=residue_mask,
            edge_index=edge_index,
            edge_mask=edge_mask,
            edge_features=edge_features,
        )
        return self.extract_pockets(
            residue_tokens=residue_tokens,
            residue_mask=residue_mask,
            pocket_indices=pocket_indices,
            pocket_mask=pocket_mask,
            pocket_valid_mask=pocket_valid_mask,
        )
