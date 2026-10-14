from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.ripple.egnn import EGNN
from src.ripple.utils import ensure_bool_mask, masked_softmax


@dataclass
class ConformerEncoding:
    tokens: Tensor
    coords: Tensor
    global_repr: Tensor
    atom_mask: Tensor
    conformer_mask: Tensor


class MultiConformerEncoder(nn.Module):
    """Encode every conformer without selecting or averaging conformers."""

    def __init__(
        self,
        atom_input_dim: int,
        dim: int = 128,
        num_layers: int = 3,
        num_nearest_neighbors: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.atom_projection = nn.Linear(atom_input_dim, dim)
        self.layers = nn.ModuleList(
            [
                EGNN(
                    dim=dim,
                    num_nearest_neighbors=num_nearest_neighbors,
                    dropout=dropout,
                    norm_feats=True,
                    norm_coors=True,
                    update_coors=True,
                    m_pool_method="mean",
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.atom_pool_score = nn.Linear(dim, 1)
        self.energy_projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        atom_features: Tensor,
        atom_coords: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
        energy: Tensor,
    ) -> ConformerEncoding:
        if atom_features.ndim != 4:
            raise ValueError("atom_features must have shape [B, Kc, Na, F]")
        if atom_coords.shape != atom_features.shape[:-1] + (3,):
            raise ValueError("atom_coords must have shape [B, Kc, Na, 3]")
        if atom_mask.shape != atom_features.shape[:-1]:
            raise ValueError("atom_mask must have shape [B, Kc, Na]")
        if conformer_mask.shape != atom_features.shape[:2]:
            raise ValueError("conformer_mask must have shape [B, Kc]")
        if energy.shape != atom_features.shape[:2]:
            raise ValueError("energy must have shape [B, Kc]")

        batch_size, conformer_count, atom_count, _ = atom_features.shape
        conformer_mask = ensure_bool_mask(conformer_mask, atom_features.device)
        atom_mask = ensure_bool_mask(atom_mask, atom_features.device)
        atom_mask = atom_mask & conformer_mask.unsqueeze(-1)

        tokens = self.atom_projection(atom_features)
        padded_atom_count = max(atom_count, self.num_nearest_neighbors)
        pad_count = padded_atom_count - atom_count
        if pad_count:
            tokens = F.pad(tokens, (0, 0, 0, pad_count))
            atom_coords = F.pad(atom_coords, (0, 0, 0, pad_count))
            atom_mask = F.pad(atom_mask, (0, pad_count), value=False)
        tokens = tokens.reshape(batch_size * conformer_count, padded_atom_count, -1)
        coords = atom_coords.reshape(batch_size * conformer_count, padded_atom_count, 3)
        flat_mask = atom_mask.reshape(batch_size * conformer_count, padded_atom_count)

        for layer in self.layers:
            tokens, coords = layer(tokens, coords, mask=flat_mask)

        tokens = self.output_norm(tokens)
        tokens = tokens * flat_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        pool_weight = masked_softmax(
            self.atom_pool_score(tokens).squeeze(-1),
            flat_mask,
            dim=-1,
        )
        global_repr = torch.sum(pool_weight.unsqueeze(-1) * tokens, dim=1)
        global_repr = global_repr.reshape(batch_size, conformer_count, -1)
        global_repr = global_repr + self.energy_projection(
            energy.unsqueeze(-1).to(global_repr.dtype)
        )
        global_repr = global_repr * conformer_mask.unsqueeze(-1).to(global_repr.dtype)

        return ConformerEncoding(
            tokens=tokens.reshape(batch_size, conformer_count, padded_atom_count, -1)[
                :, :, :atom_count
            ],
            coords=coords.reshape(batch_size, conformer_count, padded_atom_count, 3)[
                :, :, :atom_count
            ],
            global_repr=global_repr,
            atom_mask=atom_mask[:, :, :atom_count],
            conformer_mask=conformer_mask,
        )
