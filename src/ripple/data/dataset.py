"""Assembles cached features and per-source labels into a RIPPLEBatch.

Pocket localization and propagation need residue-level features, coordinates,
and a residue graph, so the cache stores proteins unpooled at their real length.
"""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset

from src.ripple.data.schema import LabelMasks, SourceTable
from src.ripple.model import RIPPLEBatch
from src.ripple.training import RIPPLELabels


@dataclass
class RIPPLESample:
    """One (compound, protein) pair plus whichever labels its source provides."""

    batch: RIPPLEBatch
    labels: RIPPLELabels
    source: SourceTable
    masks: LabelMasks


class RIPPLEDataset(Dataset):
    """Reads the residue-level cache and returns model-ready samples.

    Implementations must leave absent labels as None. Filling them with zero
    would turn "no annotation" into a negative and corrupt every masked loss.
    """

    def __init__(self, cache_dir: str, split: str):
        self.cache_dir = cache_dir
        self.split = split
        raise NotImplementedError(
            "cache reader lands once scripts/ripple/build_cache.py defines the "
            "residue-level feature format"
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> RIPPLESample:
        raise NotImplementedError


def collate_ripple(samples: list[RIPPLESample]) -> tuple[RIPPLEBatch, RIPPLELabels]:
    """Pad variable-length residues, pockets, and poses into one batch.

    Residue graphs need their edge indices shifted by the batch offset, and
    every label stays None when no sample in the batch carries it.
    """
    raise NotImplementedError


def pocket_residue_indices(contacts: Tensor, cutoff: float = 4.0) -> Tensor:
    """Native pocket labels from ligand-residue contacts. Inference never calls this.

    Candidate pockets at inference time come from an apo or predicted structure,
    so test ligand coordinates cannot define the pocket the model conditions on.
    """
    raise NotImplementedError
