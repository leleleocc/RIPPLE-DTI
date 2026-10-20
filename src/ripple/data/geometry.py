"""Freeze reproducible local and cross-chain distance queries after splitting."""

import torch

from src.ripple.data.manifest import SPLITS
from src.ripple.structure_response import GeometryQueries


def sample_geometry_queries(
    coords, residue_mask, chain_ids, edge_index, *, split, seed, max_pairs=64
):
    """Return [1, M, 1, 2] queries and distances; persist them in the manifest."""
    if split not in SPLITS:
        raise ValueError("Assign a frozen split before generating geometry queries")
    if not 1 <= max_pairs <= 64:
        raise ValueError("max_pairs must be between 1 and 64")
    if coords.shape != (len(chain_ids), 3) or residue_mask.shape != (len(chain_ids),):
        raise ValueError("Coordinates, masks and chain IDs must align")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    coords, residue_mask = coords.cpu(), residue_mask.bool().cpu()
    valid = residue_mask & torch.isfinite(coords).all(-1)
    nodes = valid.nonzero().flatten()
    if nodes.numel() < 2:
        raise ValueError("At least two mapped residues are required")
    generator = torch.Generator().manual_seed(seed)
    local, cross = set(), set()
    for left, right in edge_index.cpu().T.tolist():
        if (
            0 <= left < len(valid)
            and 0 <= right < len(valid)
            and left != right
            and valid[left]
            and valid[right]
        ):
            pair = tuple(sorted((left, right)))
            (local if chain_ids[left] == chain_ids[right] else cross).add(pair)
    # Add inter-chain candidates without materializing the quadratic distance matrix.
    draws = torch.randint(nodes.numel(), (max_pairs * 32, 2), generator=generator)
    for left, right in nodes[draws].tolist():
        if left != right and chain_ids[left] != chain_ids[right]:
            cross.add(tuple(sorted((left, right))))

    def shuffled(values):
        values = sorted(values)
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[index] for index in order]

    local, cross = shuffled(local), shuffled(cross)
    selected = local[: max_pairs // 2] + cross[: max_pairs - max_pairs // 2]
    remainder = [pair for pair in local + cross if pair not in selected]
    selected += remainder[: max_pairs - len(selected)]
    if not selected:
        raise ValueError("No local or cross-chain query candidates")
    indices = torch.tensor(selected, dtype=torch.long)
    distances = torch.linalg.vector_norm(
        coords[indices[:, 0]] - coords[indices[:, 1]], dim=-1
    )
    return GeometryQueries(
        indices[None, :, None, :], torch.ones(1, len(selected), 1, dtype=torch.bool)
    ), distances[None]
