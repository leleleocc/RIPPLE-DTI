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

The ion-channel extension adds invariant geometry targets, ligand-conditioned
structural response, and shared-weight comparison of the same protein's states.
"""

from src.ripple.model import (
    RIPPLEDTI,
    GeometryBatch,
    GeometryOutput,
    RIPPLEBatch,
    RIPPLEOutput,
)
from src.ripple.structure_response import (
    GeometryQueries,
    MultiStateOutput,
    RIPPLEMultiState,
)
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
    "RIPPLEDTI",
    "GeometryBatch",
    "GeometryOutput",
    "GeometryQueries",
    "MultiStateOutput",
    "RIPPLEBatch",
    "RIPPLELabels",
    "RIPPLELossWeights",
    "RIPPLEMultiState",
    "RIPPLEMultiTaskLoss",
    "RIPPLEOutput",
    "TrainingStage",
    "build_optimizer_param_groups",
    "configure_training_stage",
    "stage_loss_weights",
]
