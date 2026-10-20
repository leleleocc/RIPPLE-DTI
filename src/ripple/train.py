"""Train foundation -> effect warmup -> channel fine-tuning from audited caches."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.ripple.data.dataset import RIPPLEDataset
from src.ripple.data.manifest import read_manifest
from src.ripple.model import RIPPLEDTI
from src.ripple.runner import (
    TrainingStep,
    affinity_statistics,
    evaluate_loaders,
    make_loader,
    stage_tasks,
    task_criterion,
    task_schedule,
    to_device,
)
from src.ripple.training import (
    TrainingStage,
    build_optimizer_param_groups,
    configure_training_stage,
)

CURRICULUM = (
    TrainingStage.FOUNDATION,
    TrainingStage.EFFECT_WARMUP,
    TrainingStage.CHANNEL_FINETUNE,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ripple_network.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--split", default="cold-target", choices=["cold-target"])
    parser.add_argument("--stage", choices=[stage.value for stage in CURRICULUM])
    parser.add_argument("--checkpoint", help="previous stage's best checkpoint")
    parser.add_argument(
        "--resume", help="current stage's last checkpoint, at an epoch boundary"
    )
    parser.add_argument("--save-dir", default="outputs/ripple")
    parser.add_argument(
        "--epochs", type=int, help="override selected stage's traversal budget"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.accumulation < 1
        or (args.epochs is not None and args.epochs < 1)
    ):
        parser.error("Batch size, accumulation and epochs must be positive")
    if args.resume and (not args.stage or args.checkpoint):
        parser.error(
            "--resume requires --stage and cannot be combined with --checkpoint"
        )
    if args.checkpoint and not args.stage:
        parser.error("--checkpoint requires --stage")
    return args


def load_config(path):
    with Path(path).open() as handle:
        if Path(path).suffix == ".json":
            return json.load(handle)
        import yaml

        return yaml.safe_load(handle)


def model_from_config(config):
    aliases = {
        "BIINTENTION_HEADS": "num_heads",
        "BIINTENTION_LAYERS": "biintention_layers",
    }
    return RIPPLEDTI(
        **{
            aliases.get(key, key.lower()): value
            for key, value in config["RIPPLE"].items()
        }
    )


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_stage(stage, args, resume_from=None):
    stage = TrainingStage(stage)
    config = load_config(args.config)
    settings = config.get("TRAINING", {})
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if config["RIPPLE"].get("NUM_EFFECT_CLASSES", 2) != 2:
        raise ValueError(
            "The first-release training protocol requires exactly two effect classes"
        )
    if config["RIPPLE"].get("NUM_AFFINITY_ENDPOINTS", 4) != 4:
        raise ValueError("Expected Kd, Ki, IC50 and EC50 endpoint slots")
    model = model_from_config(config).to(device)
    manifest_hash = hashlib.sha256(
        (Path(args.cache) / args.manifest).read_bytes()
    ).hexdigest()
    loaded = None
    checkpoint = args.resume or resume_from
    if stage != TrainingStage.FOUNDATION and checkpoint is None:
        raise ValueError("Effect training requires a foundation checkpoint")
    if checkpoint:
        loaded = torch.load(checkpoint, map_location=device, weights_only=True)
        if loaded["manifest_hash"] != manifest_hash:
            raise ValueError(
                "Checkpoint and manifest differ; cross-stage splits must remain frozen"
            )
        if loaded["config"]["RIPPLE"] != config["RIPPLE"]:
            raise ValueError("Checkpoint model configuration differs")
        predecessor = {
            TrainingStage.EFFECT_WARMUP: "foundation",
            TrainingStage.CHANNEL_FINETUNE: "effect_warmup",
        }
        expected = stage.value if args.resume else predecessor.get(stage)
        if expected is not None and loaded["stage"] != expected:
            raise ValueError(f"{stage.value} requires a {expected} checkpoint")
        model.load_state_dict(loaded["model"])

    task_sets = stage_tasks(stage)
    records = read_manifest(Path(args.cache) / args.manifest)
    train_sets = {
        task: RIPPLEDataset(args.cache, "train", tasks, records=records)
        for task, tasks in task_sets.items()
    }
    valid_sets = {
        task: RIPPLEDataset(args.cache, "val", tasks, records=records)
        for task, tasks in task_sets.items()
    }
    primary = "measurement" if stage == TrainingStage.FOUNDATION else "effect"
    if not train_sets[primary] or not valid_sets[primary]:
        raise ValueError(f"{primary} requires nonempty train and validation records")
    defaults = (
        {"measurement": 6, "site": 2.5, "complex": 1.5}
        if stage == TrainingStage.FOUNDATION
        else {"effect": 6, "geometry": 2, "replay": 2}
    )
    ratio_key = (
        "FOUNDATION_TASK_RATIO"
        if stage == TrainingStage.FOUNDATION
        else "FINETUNE_TASK_RATIO"
    )
    ratios = (
        {"effect": 1}
        if stage == TrainingStage.EFFECT_WARMUP
        else settings.get(ratio_key, defaults)
    )
    for task, ratio in ratios.items():
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("Task ratios must be finite and nonnegative")
        if task not in train_sets:
            raise ValueError(f"Unsupported task {task} for {stage.value}")
        if ratio > 0 and not train_sets[task]:
            raise ValueError(
                f"Task {task} has a positive ratio but no training records; set its ratio to 0 explicitly"
            )
    ratios = {task: ratio for task, ratio in ratios.items() if ratio > 0}
    if primary not in ratios:
        raise ValueError("The primary task must have a positive training ratio")
    train_loaders = {
        task: make_loader(
            train_sets[task],
            args.batch_size,
            args.seed,
            rank,
            world,
            workers=args.workers,
        )
        for task in ratios
    }
    valid_loaders = {
        task: make_loader(
            dataset, args.batch_size, args.seed, train=False, workers=args.workers
        )
        for task, dataset in valid_sets.items()
        if dataset
    }
    foundation_set = RIPPLEDataset(
        args.cache, "train", {"measurement", "site", "complex"}, records=records
    )
    stats = (
        loaded["affinity_stats"]
        if loaded is not None
        else affinity_statistics(foundation_set)
    )
    configure_training_stage(model, stage)
    # Initialize lazy drug projections before wrapping in DDP, using a real batch.
    first_batch, _ = next(iter(train_loaders[primary]))
    with torch.no_grad():
        model.eval()(to_device(first_batch, device))
    groups = build_optimizer_param_groups(
        model,
        args.lr or settings.get("LEARNING_RATE", 1e-4),
        settings.get("WEIGHT_DECAY", 1e-4),
        settings.get("BACKBONE_LR_MULTIPLIER", 0.2),
        stage=stage,
    )
    optimizer = torch.optim.AdamW(groups)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best, stale, steps = 0, None, 0, 0
    if args.resume:
        if not loaded.get("epoch_complete", False):
            raise ValueError("Resume requires an epoch-boundary last.pt checkpoint")
        if (
            loaded["seed"] != args.seed
            or loaded["run_settings"]
            != {
                "world_size": world,
                "batch_size": args.batch_size,
                "accumulation": args.accumulation,
                "amp": use_amp,
                "lr": args.lr,
            }
            or loaded["config"] != config
        ):
            raise ValueError(
                "Resume requires unchanged seed, training configuration and batch settings"
            )
        optimizer.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        start, best, stale, steps = (
            loaded["epoch"] + 1,
            loaded["best_score"],
            loaded["stale"],
            loaded["steps"],
        )
    criteria = {
        task: task_criterion(
            stage,
            task,
            settings.get("RELIABLE_INTERACTION", False),
            settings.get("LOSS_WEIGHTS"),
        )
        for task in ratios
    }
    training_step = TrainingStep(model, criteria, settings.get("REPLAY_WEIGHT", 0.2))
    wrapped = (
        DistributedDataParallel(
            training_step,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if world > 1
        else training_step
    )
    budget_key, default_budget = {
        TrainingStage.FOUNDATION: ("FOUNDATION_EPOCHS", 2),
        TrainingStage.EFFECT_WARMUP: ("WARMUP_EPOCHS", 5),
        TrainingStage.CHANNEL_FINETUNE: ("FINETUNE_EPOCHS", 30),
    }[stage]
    epochs = args.epochs or settings.get(budget_key, default_budget)
    patience = 3 if stage == TrainingStage.FOUNDATION else settings.get("PATIENCE", 5)
    output_dir = Path(args.save_dir) / args.split / stage.value
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if not args.resume and (best_path.exists() or (output_dir / "last.pt").exists()):
        raise FileExistsError(
            "Training output already exists; use --resume or a new --save-dir"
        )
    interval = settings.get("FOUNDATION_VALIDATE_EVERY_STEPS", 1000)
    if interval < 1 or patience < 1:
        raise ValueError("Validation interval and patience must be positive")

    def validate_checkpoint(
        epoch, epoch_complete, batches, label_counts, loss_sums, raw_sums
    ):
        nonlocal best, stale
        report = (
            evaluate_loaders(model, valid_loaders, device, stats) if rank == 0 else None
        )
        shared = [report]
        if world > 1:
            dist.broadcast_object_list(shared, src=0)
        report = shared[0]
        score = (
            report["affinity_standardized_rmse"]
            if stage == TrainingStage.FOUNDATION
            else report["effect_macro_f1"]
        )
        if score is None or not math.isfinite(score):
            raise ValueError(
                "Validation selection metric is not estimable; inspect endpoint support or missing effect classes"
            )
        improved = best is None or (
            score < best if stage == TrainingStage.FOUNDATION else score > best
        )
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        counters = [(dict(label_counts), dict(loss_sums), dict(raw_sums))]
        if world > 1:
            counters = [None] * world
            dist.all_gather_object(
                counters, (dict(label_counts), dict(loss_sums), dict(raw_sums))
            )
        if rank == 0:
            global_counts, weighted, raw = Counter(), Counter(), Counter()
            for counts, weighted_sum, raw_sum in counters:
                global_counts.update(counts)
                weighted.update(weighted_sum)
                raw.update(raw_sum)
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "stage": stage.value,
                "epoch": epoch,
                "epoch_complete": epoch_complete,
                "steps": steps,
                "best_score": best,
                "stale": stale,
                "manifest_hash": manifest_hash,
                "config": config,
                "affinity_stats": stats,
                "seed": args.seed,
                "metrics": report,
                "run_settings": {
                    "world_size": world,
                    "batch_size": args.batch_size,
                    "accumulation": args.accumulation,
                    "amp": use_amp,
                    "lr": args.lr,
                },
            }
            if epoch_complete:
                _save_checkpoint(output_dir / "last.pt", payload)
            if improved:
                _save_checkpoint(best_path, payload)
            log = {
                "stage": stage.value,
                "epoch": epoch + 1,
                "steps": steps,
                "epoch_complete": epoch_complete,
                "task_batches": dict(batches),
                "global_label_counts": dict(global_counts),
                "weighted_loss_sums": {
                    name: value / world for name, value in weighted.items()
                },
                "global_raw_loss_sums": dict(raw),
                "validation": report,
            }
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
        if world > 1:
            dist.barrier()
        wrapped.train()
        return stage != TrainingStage.EFFECT_WARMUP and stale >= patience

    stop = stage != TrainingStage.EFFECT_WARMUP and stale >= patience
    for epoch in range(start, epochs):
        if stop:
            break
        # Epoch-boundary resume reproduces dropout and task sampling on every rank.
        torch.manual_seed(args.seed + epoch * world + rank)
        for loader in train_loaders.values():
            loader.sampler.epoch = epoch
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        schedule = task_schedule(
            ratios, primary, len(train_loaders[primary]), args.seed + epoch
        )
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        batches, label_counts, loss_sums, raw_sums = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        for index, task in enumerate(schedule):
            try:
                batch, labels = next(iterators[task])
            except StopIteration:
                iterators[task] = iter(train_loaders[task])
                batch, labels = next(iterators[task])
            batch, labels = to_device(batch, device), to_device(labels, device)
            window_start = index // args.accumulation * args.accumulation
            window_size = min(args.accumulation, len(schedule) - window_start)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, raw_losses, counts = wrapped(batch, labels, task)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {task}")
            scaler.scale(loss / window_size).backward()
            batches[task] += 1
            loss_sums[task] += float(loss.detach())
            label_counts.update(counts)
            raw_sums.update(
                {
                    name: float(value) * counts[name]
                    for name, value in raw_losses.items()
                    if counts[name]
                }
            )
            if (index + 1) % args.accumulation == 0 or index + 1 == len(schedule):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                epoch_complete = index + 1 == len(schedule)
                if epoch_complete or (
                    stage == TrainingStage.FOUNDATION and steps % interval == 0
                ):
                    stop = validate_checkpoint(
                        epoch,
                        epoch_complete,
                        batches,
                        label_counts,
                        loss_sums,
                        raw_sums,
                    )
                    if stop:
                        break
    if not best_path.exists():
        raise ValueError(
            "No best checkpoint in save-dir; resume with the original output directory"
        )
    return best_path


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if args.device.startswith("cuda"):
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        dist.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")
    try:
        stages = [TrainingStage(args.stage)] if args.stage else list(CURRICULUM)
        checkpoint = args.checkpoint
        for stage in stages:
            checkpoint = run_stage(stage, args, checkpoint)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
