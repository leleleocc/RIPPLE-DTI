from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.ripple.model import RIPPLEDTI, RIPPLEOutput


class TrainingStage(str, Enum):
    POCKET = "pocket"
    DTI = "dti"
    PROPAGATION = "propagation"
    PHARMACOLOGY = "pharmacology"
    JOINT = "joint"


@dataclass
class RIPPLELabels:
    interaction: Optional[Tensor] = None
    interaction_mask: Optional[Tensor] = None
    affinity: Optional[Tensor] = None
    affinity_mask: Optional[Tensor] = None
    effect: Optional[Tensor] = None
    effect_mask: Optional[Tensor] = None
    mechanism: Optional[Tensor] = None
    mechanism_mask: Optional[Tensor] = None
    native_pocket_index: Optional[Tensor] = None
    native_pocket_mask: Optional[Tensor] = None
    function_label: Optional[Tensor] = None
    function_mask: Optional[Tensor] = None
    pose_quality: Optional[Tensor] = None
    pose_mask: Optional[Tensor] = None
    allosteric_pocket: Optional[Tensor] = None
    allosteric_pocket_mask: Optional[Tensor] = None


@dataclass
class RIPPLELossWeights:
    interaction: float = 1.0
    affinity: float = 0.5
    effect: float = 0.5
    mechanism: float = 0.25
    native_pocket: float = 0.25
    function: float = 0.5
    pose_rank: float = 0.1
    allosteric_pocket: float = 0.1
    nonbinder_delta: float = 0.01


@dataclass
class RIPPLELossOutput:
    total: Tensor
    raw: Dict[str, Tensor]
    weighted: Dict[str, Tensor]
    label_count: Dict[str, int]


def _resolve_mask(target: Tensor, mask: Optional[Tensor]) -> Tensor:
    if target.is_floating_point():
        valid_target = torch.isfinite(target)
    else:
        valid_target = target >= 0
    if mask is None:
        return valid_target
    return mask.to(device=target.device, dtype=torch.bool) & valid_target


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()


class RIPPLEMultiTaskLoss(nn.Module):
    """Masked objectives for batches drawn from heterogeneous data sources."""

    def __init__(self, weights: Optional[RIPPLELossWeights] = None):
        super().__init__()
        self.weights = weights or RIPPLELossWeights()

    @staticmethod
    def _classification_loss(
        logits: Tensor,
        target: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, int]:
        if target is None:
            return logits.sum() * 0.0, 0
        target = target.to(device=logits.device, dtype=torch.long)
        valid = _resolve_mask(target, mask)
        valid = valid & (target < logits.size(-1))
        safe_target = target.clamp(min=0, max=logits.size(-1) - 1)
        loss = F.cross_entropy(logits, safe_target, reduction="none")
        return _masked_mean(loss, valid), int(valid.sum().item())

    @staticmethod
    def _index_nll(
        probability: Tensor,
        target: Optional[Tensor],
        mask: Optional[Tensor],
        candidate_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        if target is None:
            return probability.sum() * 0.0, 0
        target = target.to(device=probability.device, dtype=torch.long)
        valid = _resolve_mask(target, mask)
        valid = valid & (target >= 0) & (target < probability.size(1))
        safe_target = target.clamp(min=0, max=probability.size(1) - 1)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(
                device=probability.device,
                dtype=torch.bool,
            )
            if candidate_mask.shape != probability.shape:
                raise ValueError("candidate_mask must match probability shape")
            target_is_available = candidate_mask.gather(
                1,
                safe_target.unsqueeze(-1),
            ).squeeze(-1)
            valid = valid & target_is_available
        selected = probability.gather(1, safe_target.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(selected.clamp_min(1e-8))
        return _masked_mean(loss, valid), int(valid.sum().item())

    @staticmethod
    def _pairwise_ranking_loss(
        score: Tensor,
        order: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, int]:
        """Rank poses of the same ligand by docking RMSD within each pocket.

        order is -RMSD, so a larger value is the better pose. Pairs whose order
        differs by less than the margin carry no ranking signal and are dropped.
        """
        score_delta = score.unsqueeze(-1) - score.unsqueeze(-2)
        order_delta = order.unsqueeze(-1) - order.unsqueeze(-2)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask = pair_mask & (order_delta > 0.5)
        if not pair_mask.any():
            return score.sum() * 0.0, 0
        loss = F.softplus(-score_delta)[pair_mask]
        return loss.mean(), int(pair_mask.sum().item())

    def forward(
        self,
        output: RIPPLEOutput,
        labels: RIPPLELabels,
    ) -> RIPPLELossOutput:
        predictions = output.predictions
        raw: Dict[str, Tensor] = {}
        counts: Dict[str, int] = {}

        interaction_logit = predictions["interaction_logit"]
        if labels.interaction is None:
            raw["interaction"] = interaction_logit.sum() * 0.0
            counts["interaction"] = 0
        else:
            target = labels.interaction.to(
                device=interaction_logit.device, dtype=interaction_logit.dtype
            )
            valid = _resolve_mask(target, labels.interaction_mask)
            loss = F.binary_cross_entropy_with_logits(
                interaction_logit,
                target.nan_to_num().clamp(min=0.0, max=1.0),
                reduction="none",
            )
            raw["interaction"] = _masked_mean(loss, valid)
            counts["interaction"] = int(valid.sum().item())

        affinity_prediction = predictions["affinity"]
        if labels.affinity is None:
            raw["affinity"] = affinity_prediction.sum() * 0.0
            counts["affinity"] = 0
        else:
            target = labels.affinity.to(
                device=affinity_prediction.device, dtype=affinity_prediction.dtype
            )
            valid = _resolve_mask(target, labels.affinity_mask)
            loss = F.smooth_l1_loss(
                affinity_prediction, target.nan_to_num(), reduction="none"
            )
            raw["affinity"] = _masked_mean(loss, valid)
            counts["affinity"] = int(valid.sum().item())

        raw["effect"], counts["effect"] = self._classification_loss(
            predictions["effect_logits"], labels.effect, labels.effect_mask
        )
        raw["mechanism"], counts["mechanism"] = self._classification_loss(
            predictions["mechanism_logits"],
            labels.mechanism,
            labels.mechanism_mask,
        )
        raw["native_pocket"], counts["native_pocket"] = self._index_nll(
            output.match.pocket_weight,
            labels.native_pocket_index,
            labels.native_pocket_mask,
            output.pockets.pocket_valid_mask,
        )

        sink_logits = output.propagation.sink_logits
        if labels.function_label is None:
            raw["function"] = sink_logits.sum() * 0.0
            counts["function"] = 0
        else:
            target = labels.function_label.to(
                device=sink_logits.device, dtype=sink_logits.dtype
            )
            residue_valid = _resolve_mask(target, labels.function_mask)
            residue_valid = residue_valid & output.pockets.residue_mask
            loss = F.binary_cross_entropy_with_logits(
                sink_logits, target.nan_to_num().clamp(0.0, 1.0), reduction="none"
            )
            raw["function"] = _masked_mean(loss, residue_valid)
            counts["function"] = int(residue_valid.sum().item())

        pose_logits = output.match.pose_quality_logits
        if labels.pose_quality is None:
            raw["pose_rank"] = pose_logits.sum() * 0.0
            counts["pose_rank"] = 0
        else:
            order = labels.pose_quality.to(
                device=pose_logits.device, dtype=pose_logits.dtype
            )
            pose_valid = _resolve_mask(order, labels.pose_mask)
            pose_valid = pose_valid & output.conformers.conformer_mask.unsqueeze(1)
            # Pose labels are per pocket; rank the poses of each pocket separately
            # and average over the pockets that actually carry at least one pair.
            per_pocket = []
            ranked_pairs = 0
            for pocket_index in range(pose_logits.size(1)):
                pocket_loss, pocket_pairs = self._pairwise_ranking_loss(
                    pose_logits[:, pocket_index],
                    order[:, pocket_index],
                    pose_valid[:, pocket_index],
                )
                if pocket_pairs:
                    per_pocket.append(pocket_loss)
                    ranked_pairs += pocket_pairs
            if per_pocket:
                raw["pose_rank"] = torch.stack(per_pocket).mean()
            else:
                raw["pose_rank"] = pose_logits.sum() * 0.0
            counts["pose_rank"] = ranked_pairs

        allosteric_logits = predictions["allosteric_pocket_logits"]
        if labels.allosteric_pocket is None:
            raw["allosteric_pocket"] = allosteric_logits.sum() * 0.0
            counts["allosteric_pocket"] = 0
        else:
            target = labels.allosteric_pocket.to(
                device=allosteric_logits.device, dtype=allosteric_logits.dtype
            )
            valid = _resolve_mask(target, labels.allosteric_pocket_mask)
            valid = valid & output.pockets.pocket_valid_mask
            loss = F.binary_cross_entropy_with_logits(
                allosteric_logits,
                target.nan_to_num().clamp(min=0.0, max=1.0),
                reduction="none",
            )
            raw["allosteric_pocket"] = _masked_mean(loss, valid)
            counts["allosteric_pocket"] = int(valid.sum().item())

        if labels.interaction is None:
            raw["nonbinder_delta"] = output.propagation.delta_tokens.sum() * 0.0
            counts["nonbinder_delta"] = 0
        else:
            target = labels.interaction.to(device=interaction_logit.device)
            reliable = _resolve_mask(target, labels.interaction_mask) & (target == 0)
            delta_norm = output.propagation.function_state.square().mean(dim=-1)
            raw["nonbinder_delta"] = _masked_mean(delta_norm, reliable)
            counts["nonbinder_delta"] = int(reliable.sum().item())

        weighted = {
            name: value * float(getattr(self.weights, name))
            for name, value in raw.items()
        }
        total = sum(weighted.values(), interaction_logit.sum() * 0.0)
        return RIPPLELossOutput(
            total=total,
            raw=raw,
            weighted=weighted,
            label_count=counts,
        )


_STAGE_MODULES = {
    TrainingStage.POCKET: {
        "conformer_encoder",
        "residue_encoder",
        "protein_fusion",
        "matcher",
        "drug_fusion.chemical_fusion",
        "task_heads.allosteric_pocket_head",
    },
    TrainingStage.DTI: {
        "matcher",
        "drug_fusion",
        "protein_resampler",
        "interaction",
        "task_heads.global_projection",
        "task_heads.local_projection",
        "task_heads.local_gate",
        "task_heads.binding_norm",
        "task_heads.interaction_head",
        "task_heads.affinity_head",
        "task_heads.endpoint_embedding",
    },
    TrainingStage.PROPAGATION: {
        "source_fusion",
        "propagation",
        "task_heads.effect_head",
    },
    TrainingStage.PHARMACOLOGY: {
        "source_fusion",
        "propagation",
        "task_heads.effect_head",
        "task_heads.mechanism_head",
    },
    TrainingStage.JOINT: {""},
}


def _matches_prefix(parameter_name: str, prefixes: Set[str]) -> bool:
    return "" in prefixes or any(
        parameter_name == prefix or parameter_name.startswith(prefix + ".")
        for prefix in prefixes
    )


def configure_training_stage(
    model: RIPPLEDTI, stage: Union[TrainingStage, str]
) -> None:
    """Freeze modules according to the declared curriculum stage."""
    stage = TrainingStage(stage)
    prefixes = _STAGE_MODULES[stage]
    for name, parameter in model.named_parameters():
        parameter.requires_grad = _matches_prefix(name, prefixes)


def stage_loss_weights(stage: Union[TrainingStage, str]) -> RIPPLELossWeights:
    """Return objectives that have supervision in each curriculum stage."""
    stage = TrainingStage(stage)
    weights = RIPPLELossWeights()
    enabled = {
        TrainingStage.POCKET: {"native_pocket", "allosteric_pocket", "pose_rank"},
        TrainingStage.DTI: {"interaction", "affinity", "native_pocket"},
        TrainingStage.PROPAGATION: {"function", "effect", "nonbinder_delta"},
        TrainingStage.PHARMACOLOGY: {"effect", "mechanism", "nonbinder_delta"},
        TrainingStage.JOINT: {field.name for field in fields(RIPPLELossWeights)},
    }[stage]
    for field in fields(weights):
        if field.name not in enabled:
            setattr(weights, field.name, 0.0)
    return weights


def build_optimizer_param_groups(
    model: RIPPLEDTI,
    base_lr: float,
    weight_decay: float,
    backbone_lr_multiplier: float = 0.2,
) -> List[dict]:
    """Use a conservative LR for reusable geometry encoders."""
    backbone_prefixes = ("conformer_encoder.", "residue_encoder.", "protein_fusion.")
    backbone = []
    task_specific = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(backbone_prefixes):
            backbone.append(parameter)
        else:
            task_specific.append(parameter)

    groups = []
    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": base_lr * backbone_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "shared_backbone",
            }
        )
    if task_specific:
        groups.append(
            {
                "params": task_specific,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "ripple_modules",
            }
        )
    return groups
