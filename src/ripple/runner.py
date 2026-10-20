"""Training utilities for frozen manifests and source-specific task batches."""

import math
from collections import Counter
from dataclasses import fields, is_dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from src.ripple.data.dataset import collate_ripple
from src.ripple.data.schema import SOURCES
from src.ripple.training import RIPPLEMultiTaskLoss, TrainingStage, stage_loss_weights


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: to_device(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    return value


class TaskSampler(Sampler):
    """Deterministic global sampling, sharded equally across DDP ranks."""

    def __init__(self, dataset, seed, rank=0, world_size=1, balance=False):
        self.dataset, self.seed = dataset, seed
        self.rank, self.world_size, self.epoch = rank, world_size, 0
        self.weights = None
        if balance:
            keys = [
                (row.labels.get("effect"), row.evidence.get("family", row.split_group))
                for row in dataset.records
            ]
            if any(label not in (0, 1) for label, _ in keys):
                raise ValueError(
                    "Effect training records require a known binary direction"
                )
            strata = Counter(keys)
            families = Counter(label for label, _ in strata)
            self.weights = torch.tensor(
                [1.0 / (strata[key] * families[key[0]]) for key in keys]
            )

    def __len__(self):
        return math.ceil(len(self.dataset) / self.world_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        count = len(self) * self.world_size
        if self.weights is None:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            indices = (indices * math.ceil(count / len(indices)))[:count]
        else:
            indices = torch.multinomial(
                self.weights, count, replacement=True, generator=generator
            ).tolist()
        return iter(indices[self.rank :: self.world_size])


def make_loader(dataset, batch_size, seed, rank=0, world_size=1, train=True, workers=0):
    sampler = (
        TaskSampler(
            dataset,
            seed,
            rank,
            world_size,
            balance=bool(dataset.records)
            and all(row.task == "effect" for row in dataset.records),
        )
        if train
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_ripple,
        generator=torch.Generator().manual_seed(seed + rank),
    )


def task_schedule(ratios, primary, primary_batches, seed):
    """Use a finite primary traversal and explicit auxiliary batches; no masked fillers."""
    if primary not in ratios or ratios[primary] <= 0 or primary_batches < 1:
        raise ValueError("A positive primary task budget is required")
    tasks = []
    for task, ratio in ratios.items():
        if ratio < 0:
            raise ValueError("Task ratios must be nonnegative")
        count = (
            primary_batches
            if task == primary
            else round(primary_batches * ratio / ratios[primary])
        )
        tasks.extend([task] * count)
    order = torch.randperm(
        len(tasks), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return [tasks[index] for index in order]


def normalized_loss(result, weights, task, replay_weight=0.2):
    """DDP averages gradients; compensate to obtain a global valid-label mean."""
    names = [
        field.name for field in fields(weights) if getattr(weights, field.name) > 0
    ]
    reference = result.total
    counts = torch.tensor(
        [result.label_count[name] for name in names],
        device=reference.device,
        dtype=torch.float64,
    )
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(counts)
    total = reference.new_zeros(())
    active = 0
    for name, count in zip(names, counts):
        if count.item() > 0:
            total = total + result.raw[name] * (
                result.label_count[name] * world / count
            ) * getattr(weights, name)
            active += 1
    if active == 0:
        raise ValueError(f"Task batch {task} has no enabled, valid labels")
    if task == "replay":
        total = total * replay_weight / active
    return total


class TrainingStep(torch.nn.Module):
    """Expose only loss-connected tensors to DDP's unused-parameter discovery."""

    def __init__(self, model, criteria, replay_weight):
        super().__init__()
        self.model = model
        self.criteria = torch.nn.ModuleDict(criteria)
        self.replay_weight = replay_weight

    def forward(self, batch, labels, task):
        criterion = self.criteria[task]
        result = criterion(self.model(batch), labels)
        loss = normalized_loss(result, criterion.weights, task, self.replay_weight)
        return (
            loss,
            {name: value.detach() for name, value in result.raw.items()},
            result.label_count,
        )


def affinity_statistics(dataset):
    values = {}
    for row in dataset.records:
        value = row.labels.get("affinity")
        if (
            row.endpoint_type in range(4)
            and row.loss_enabled.get("affinity")
            and SOURCES[row.source].affinity
            and row.labels.get("affinity_mask", True)
            and value is not None
            and math.isfinite(value)
        ):
            values.setdefault(row.endpoint_type, []).append(value)
    stats = {}
    for endpoint, items in values.items():
        tensor = torch.tensor(items, dtype=torch.float64)
        std = float(tensor.std(unbiased=False))
        if len(items) > 1 and std > 0:
            stats[endpoint] = {"mean": float(tensor.mean()), "std": std}
    return stats


@torch.no_grad()
def evaluate_loaders(model, loaders, device, affinity_stats):
    model.eval()
    confusion = torch.zeros(2, 2, dtype=torch.long)
    squared, counts = {}, Counter()
    pocket_hits = pocket_hits3 = pocket_count = pocket_annotated = 0
    geometry_error, geometry_count = 0.0, 0
    group_support = Counter()
    for loader in loaders.values():
        for row in loader.dataset.records:
            if (
                row.task == "effect"
                and row.labels.get("effect") in (0, 1)
                and row.loss_enabled.get("effect")
            ):
                group_support[row.evidence.get("family", row.split_group)] += 1
            if (
                SOURCES[row.source].pocket
                and row.loss_enabled.get("pocket")
                and row.labels.get("native_pocket_index") is not None
            ):
                pocket_annotated += 1
        for batch, labels in loader:
            batch, labels = to_device(batch, device), to_device(labels, device)
            output = model(batch)
            predictions = output.predictions
            if labels.effect is not None:
                valid = labels.effect_mask & (labels.effect >= 0) & (labels.effect < 2)
                actual = labels.effect[valid].long().cpu()
                predicted = predictions["effect_logits"][valid].argmax(-1).cpu()
                confusion += torch.bincount(
                    actual * 2 + predicted, minlength=4
                ).reshape(2, 2)
            if labels.affinity is not None:
                valid = (
                    labels.affinity_mask
                    & predictions["affinity_valid_mask"]
                    & torch.isfinite(labels.affinity)
                )
                for endpoint in range(4):
                    selected = valid & (batch.endpoint_type == endpoint)
                    squared[endpoint] = squared.get(endpoint, 0.0) + float(
                        (predictions["affinity"][selected] - labels.affinity[selected])
                        .square()
                        .sum()
                    )
                    counts[endpoint] += int(selected.sum())
            if labels.native_pocket_index is not None:
                probability = output.match.pocket_weight
                target = labels.native_pocket_index
                safe = target.clamp(0, probability.size(1) - 1)
                valid = (
                    labels.native_pocket_mask
                    & (target >= 0)
                    & (target < probability.size(1))
                )
                valid &= output.pockets.pocket_valid_mask.gather(
                    1, safe[:, None]
                ).squeeze(1)
                ranked = probability.masked_fill(
                    ~output.pockets.pocket_valid_mask, -1
                ).argsort(-1, descending=True)
                pocket_hits += int(((ranked[:, 0] == target) & valid).sum())
                pocket_hits3 += int(
                    ((ranked[:, :3] == target[:, None]).any(-1) & valid).sum()
                )
                pocket_count += int(valid.sum())
            if labels.geometry_snapshot is not None:
                valid = (
                    labels.geometry_snapshot_mask & predictions["geometry_valid_mask"]
                )
                valid &= torch.isfinite(labels.geometry_snapshot) & (
                    labels.geometry_snapshot >= 0
                )
                geometry_error += float(
                    (
                        predictions["geometry_snapshot"][valid]
                        - labels.geometry_snapshot[valid]
                    )
                    .abs()
                    .sum()
                )
                geometry_count += int(valid.sum())
    support = confusion.sum(1)
    precision = confusion.diag().float() / confusion.sum(0).clamp_min(1)
    recall = confusion.diag().float() / support.clamp_min(1)
    f1 = 2 * confusion.diag().float() / (confusion.sum(0) + support).clamp_min(1)
    rmse = {
        str(endpoint): math.sqrt(squared[endpoint] / count)
        for endpoint, count in counts.items()
        if count
    }
    normalized = [
        value / affinity_stats[int(endpoint)]["std"]
        for endpoint, value in rmse.items()
        if int(endpoint) in affinity_stats
    ]
    return {
        "effect_macro_f1": float(f1.mean()) if (support > 0).all() else None,
        "balanced_accuracy": float(recall.mean()) if (support > 0).all() else None,
        "effect_precision": precision.tolist(),
        "effect_recall": recall.tolist(),
        "effect_support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "effect_family_support": dict(group_support),
        "affinity_rmse": rmse,
        "affinity_count": dict(counts),
        "affinity_standardized_rmse": sum(normalized) / len(normalized)
        if normalized
        else None,
        "pocket_top1": pocket_hits / pocket_count if pocket_count else None,
        "pocket_top3": pocket_hits3 / pocket_count if pocket_count else None,
        "pocket_count": pocket_count,
        "pocket_annotated": pocket_annotated,
        "pocket_coverage": pocket_count / pocket_annotated
        if pocket_annotated
        else None,
        "geometry_mae": geometry_error / geometry_count if geometry_count else None,
        "geometry_count": geometry_count,
    }


def stage_tasks(stage):
    stage = TrainingStage(stage)
    if stage == TrainingStage.FOUNDATION:
        return {
            "measurement": {"measurement"},
            "site": {"site"},
            "complex": {"complex"},
        }
    if stage == TrainingStage.EFFECT_WARMUP:
        return {"effect": {"effect"}}
    return {
        "effect": {"effect"},
        "geometry": {"geometry"},
        "replay": {"measurement", "site", "complex"},
    }


def task_criterion(stage, task, reliable_interaction=False, overrides=None):
    weights = stage_loss_weights(stage, reliable_interaction=reliable_interaction)
    enabled = {
        "measurement": {"affinity", "interaction"},
        "site": {"native_pocket"},
        "complex": {"affinity", "native_pocket", "interaction"},
        "replay": {"affinity", "native_pocket", "interaction"},
        "effect": {"effect"},
        "geometry": {"geometry_snapshot"},
    }[task]
    for field in fields(weights):
        if field.name not in enabled:
            setattr(weights, field.name, 0.0)
        elif (
            getattr(weights, field.name) > 0
            and overrides
            and field.name.upper() in overrides
        ):
            value = float(overrides[field.name.upper()])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Loss weights must be finite and nonnegative")
            setattr(weights, field.name, value)
    return RIPPLEMultiTaskLoss(weights)
