import torch
from torch import Tensor


def ensure_bool_mask(mask: Tensor, device=None) -> Tensor:
    """Return a boolean mask on the requested device."""
    return mask.to(device=device, dtype=torch.bool)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax with an all-masked-row fallback to zeros."""
    mask = ensure_bool_mask(mask, logits.device)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=dim)
    probabilities = probabilities * mask.to(dtype=logits.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    mask = ensure_bool_mask(mask, values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def gather_nodes(nodes: Tensor, indices: Tensor) -> Tensor:
    """Gather padded node indices from [B, N, D] into [B, ..., D]."""
    if nodes.ndim != 3 or indices.ndim < 2:
        raise ValueError("nodes must be [B, N, D] and indices must start with B")
    if nodes.size(0) != indices.size(0):
        raise ValueError("nodes and indices must have the same batch size")

    batch_shape = [nodes.size(0)] + [1] * (indices.ndim - 1)
    batch_index = torch.arange(nodes.size(0), device=nodes.device).view(*batch_shape)
    batch_index = batch_index.expand_as(indices)
    return nodes[batch_index, indices.clamp(min=0, max=nodes.size(1) - 1)]


def scatter_pocket_nodes(
    pocket_values: Tensor,
    pocket_indices: Tensor,
    pocket_mask: Tensor,
    residue_count: int,
) -> Tensor:
    """Average overlapping pocket values back onto full residue positions."""
    batch_size, _, _, dim = pocket_values.shape
    indices = pocket_indices.clamp(min=0, max=residue_count - 1)
    valid = ensure_bool_mask(pocket_mask, pocket_values.device)

    result = pocket_values.new_zeros(batch_size, residue_count, dim)
    counts = pocket_values.new_zeros(batch_size, residue_count, 1)
    result.scatter_add_(
        1,
        indices.reshape(batch_size, -1, 1).expand(-1, -1, dim),
        (pocket_values * valid.unsqueeze(-1)).reshape(batch_size, -1, dim),
    )
    counts.scatter_add_(
        1,
        indices.reshape(batch_size, -1, 1),
        valid.to(dtype=pocket_values.dtype).reshape(batch_size, -1, 1),
    )
    return result / counts.clamp_min(1.0)
