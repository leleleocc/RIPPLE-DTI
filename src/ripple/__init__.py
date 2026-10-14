"""RIPPLE-DTI innovations.

Two primary contributions and one auxiliary module:

- pocket-aware DTI, where candidate pockets condition the interaction state
  (:mod:`src.ripple.pocket_encoder`, :mod:`src.ripple.interaction`);
- ligand-conditioned pocket-to-function propagation
  (:mod:`src.ripple.propagation`);
- pocket-conditioned pose confidence and re-ranking, which modulates the
  interaction state but never gates it (:mod:`src.ripple.pose_confidence`).

:class:`RIPPLEDTI` assembles them and :class:`RIPPLEMultiTaskLoss` supervises
each from its own label mask.
"""

from src.ripple.model import RIPPLEBatch, RIPPLEDTI, RIPPLEOutput
from src.ripple.training import (
    RIPPLELabels,
    RIPPLELossWeights,
    RIPPLEMultiTaskLoss,
    TrainingStage,
    build_optimizer_param_groups,
    configure_training_stage,
    stage_loss_weights,
)

__all__ = [
    "RIPPLEBatch",
    "RIPPLEDTI",
    "RIPPLELabels",
    "RIPPLELossWeights",
    "RIPPLEMultiTaskLoss",
    "RIPPLEOutput",
    "TrainingStage",
    "build_optimizer_param_groups",
    "configure_training_stage",
    "stage_loss_weights",
]
