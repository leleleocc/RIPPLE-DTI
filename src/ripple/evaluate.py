"""Evaluation, reported per contribution rather than as one score.

The four numbers answer different claims and are not comparable to each other:

- dti: AUROC and AUPRC under a cold split. Affinity RMSE is reported per
  endpoint, never pooled across Kd, Ki, IC50, and EC50.
- pocket: top-1 recovery of the native pocket among the candidates.
- pose: ranking quality on the pose-labelled subset only.
- function: residue-level AUPRC for annotated functional sites.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SplitMetrics:
    split: str
    dti_auroc: Optional[float] = None
    dti_auprc: Optional[float] = None
    pocket_top1: Optional[float] = None
    pose_ranking_auc: Optional[float] = None
    function_auprc: Optional[float] = None


def evaluate(checkpoint: str, cache: str, split: str) -> SplitMetrics:
    """Score one checkpoint on one cold split."""
    raise NotImplementedError(
        f"evaluation loop waits on the residue-level cache reader: "
        f"checkpoint={checkpoint} cache={cache} split={split}"
    )
