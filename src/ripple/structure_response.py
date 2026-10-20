"""Invariant geometry readouts and shared-weight comparison of protein states."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.ripple.task_heads import PredictionMLP

if TYPE_CHECKING:
    from src.ripple.model import RIPPLEDTI, RIPPLEBatch, RIPPLEOutput


@dataclass
class GeometryQueries:
    """Each metric averages a set of reference-residue pairs, in Angstrom.

    indices: [B, M, P, 2], pair_mask: [B, M, P]. Indices refer to the
    reference structure only. Target coordinates and state labels are absent.
    Averaging all cross-subunit pairs avoids arbitrary homomer chain matching.
    """

    indices: Tensor
    pair_mask: Tensor


class StructureResponseHead(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.readout = PredictionMLP(dim * 5 + 1, dim * 2, 1, dropout)
        self.snapshot_readout = PredictionMLP(dim * 2, dim, 1, dropout)

    @staticmethod
    def _pairs(coords: Tensor, residue_mask: Tensor, queries: GeometryQueries):
        indices, requested = queries.indices, queries.pair_mask
        if indices.ndim != 4 or indices.size(-1) != 2 or indices.dtype != torch.long:
            raise ValueError("geometry indices must be int64 [B, M, P, 2]")
        if requested.shape != indices.shape[:-1] or indices.size(0) != coords.size(0):
            raise ValueError("geometry masks/batch dimensions do not match")
        if coords.size(1) == 0:
            raise ValueError("geometry queries need a nonempty reference structure")
        in_range = ((indices >= 0) & (indices < coords.size(1))).all(dim=-1)
        safe = indices.clamp(0, coords.size(1) - 1)
        batch = torch.arange(coords.size(0), device=coords.device)[:, None, None]
        left, right = safe.unbind(dim=-1)
        valid = requested.bool() & in_range & (left != right)
        valid = valid & residue_mask[batch, left] & residue_mask[batch, right]
        xyz_left, xyz_right = coords[batch, left], coords[batch, right]
        valid = valid & torch.isfinite(xyz_left).all(-1) & torch.isfinite(xyz_right).all(-1)
        distance = torch.linalg.vector_norm(
            torch.where(valid[..., None], xyz_left - xyz_right, 0.0), dim=-1,
        )
        return batch, left, right, valid, distance

    def snapshot(self, apo_tokens, coords, residue_mask, queries):
        batch, left, right, valid, distance = self._pairs(coords, residue_mask, queries)
        a, b = apo_tokens[batch, left], apo_tokens[batch, right]
        features = torch.where(valid[..., None], torch.cat([a + b, (a - b).abs()], -1), 0.0)
        predicted = torch.nn.functional.softplus(self.snapshot_readout(features).squeeze(-1))
        denominator = valid.sum(-1).clamp_min(1)
        return {
            "geometry_snapshot": torch.where(valid, predicted, 0.0).sum(-1) / denominator,
            "geometry_reference": distance.sum(-1) / denominator,
            "geometry_valid_mask": valid.any(-1),
        }

    def forward(
        self,
        apo_tokens: Tensor,
        delta_tokens: Tensor,
        chemical_context: Tensor,
        coords: Tensor,
        residue_mask: Tensor,
        queries: GeometryQueries,
    ) -> dict[str, Tensor]:
        batch, left, right, valid, distance = self._pairs(coords, residue_mask, queries)
        a, b = apo_tokens[batch, left], apo_tokens[batch, right]
        da, db = delta_tokens[batch, left], delta_tokens[batch, right]
        context = chemical_context[:, None, None, :].expand_as(a)
        features = torch.cat(
            [
                a + b,
                (a - b).abs(),
                da + db,
                (da - db).abs(),
                context,
                distance[..., None] / 10.0,
            ],
            dim=-1,
        )
        features = torch.where(valid[..., None], features, 0.0)
        raw_delta = self.readout(features).squeeze(-1)
        snapshot_features = torch.where(
            valid[..., None], torch.cat([a + b, (a - b).abs()], -1), 0.0
        )
        snapshot = torch.nn.functional.softplus(
            self.snapshot_readout(snapshot_features).squeeze(-1)
        )
        predicted_distance = (distance + raw_delta).clamp_min(0.0)
        denominator = valid.sum(-1).clamp_min(1)

        def average(values):
            return torch.where(valid, values, 0.0).sum(-1) / denominator

        reference = average(distance)
        target = average(predicted_distance)
        return {
            "geometry_reference": reference,
            "geometry_target": target,
            "geometry_delta": target - reference,
            "geometry_snapshot": average(snapshot),
            "geometry_valid_mask": valid.any(-1),
        }


@dataclass
class MultiStateOutput:
    states: Sequence["RIPPLEOutput"]
    scores: Tensor
    valid_mask: Tensor
    preference_logits: Tensor
    preference_probability: Tensor


class RIPPLEMultiState(nn.Module):
    """Compare candidate conformations with shared RIPPLE weights.

    Scores are learned preference scores, not binding free energies. All states
    must represent the same protein and ligand. The data layer validates identity.
    """

    def __init__(self, core: "RIPPLEDTI"):
        super().__init__()
        self.core = core
        self.state_score = PredictionMLP(core.dim * 2, core.dim, 1, 0.0)

    def forward(
        self, batches: Sequence["RIPPLEBatch"], state_mask: Tensor, identities
    ) -> MultiStateOutput:
        from src.ripple.data.channels import validate_state_group

        if not batches or state_mask.shape != (
            batches[0].residue_mask.size(0),
            len(batches),
        ):
            raise ValueError(
                "state_mask must have shape [B, number of candidate states]"
            )
        if len(identities) != state_mask.size(0) or any(
            len(group) != len(batches) for group in identities
        ):
            raise ValueError(
                "identities must have shape [B, number of candidate states]"
            )
        for group in identities:
            validate_state_group(group)
        # Same ligand inputs prevent an apparent state preference caused by a
        # different compound or conformer collection in one candidate.
        for candidate in batches[1:]:
            for name in (
                "atom_features",
                "atom_coords",
                "atom_mask",
                "conformer_mask",
                "conformer_energy",
                "drug_1d_tokens",
                "drug_1d_mask",
                "drug_2d_tokens",
                "drug_2d_mask",
            ):
                if not torch.equal(getattr(batches[0], name), getattr(candidate, name)):
                    raise ValueError(
                        f"candidate states must share ligand inputs: {name}"
                    )
        outputs = [self.core(batch) for batch in batches]
        scores = torch.stack(
            [
                self.state_score(
                    torch.cat(
                        [
                            out.predictions["binding_state"],
                            out.propagation.function_state,
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)
                for out in outputs
            ],
            dim=1,
        )
        valid = state_mask.to(device=scores.device, dtype=torch.bool)
        valid = valid & torch.stack(
            [batch.residue_mask.any(-1) for batch in batches], dim=1
        )
        masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probabilities = masked.softmax(-1) * valid
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
            1e-8
        )
        return MultiStateOutput(outputs, scores, valid, masked, probabilities)
