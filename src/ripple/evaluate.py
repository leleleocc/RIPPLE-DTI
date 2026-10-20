"""Evaluate a frozen checkpoint on a manifest partition without fitting parameters."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.ripple.data.dataset import RIPPLEDataset
from src.ripple.data.manifest import SPLITS
from src.ripple.runner import evaluate_loaders, make_loader, stage_tasks
from src.ripple.train import model_from_config


def evaluate(
    checkpoint,
    cache,
    split="test",
    *,
    manifest="manifest.jsonl",
    device="cpu",
    batch_size=2,
):
    if split not in SPLITS:
        raise ValueError(f"Unknown manifest split: {split}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    manifest_hash = hashlib.sha256((Path(cache) / manifest).read_bytes()).hexdigest()
    if manifest_hash != state["manifest_hash"]:
        raise ValueError("Evaluation must use the frozen training manifest")
    model = model_from_config(state["config"]).to(device)
    model.load_state_dict(state["model"])
    datasets = {
        task: RIPPLEDataset(cache, split, tasks, manifest)
        for task, tasks in stage_tasks(state["stage"]).items()
    }
    loaders = {
        task: make_loader(dataset, batch_size, state["seed"], train=False)
        for task, dataset in datasets.items()
        if dataset
    }
    if not loaders:
        raise ValueError(f"No applicable records in split {split}")
    report = evaluate_loaders(model, loaders, device, state["affinity_stats"])
    return {
        "split": split,
        "checkpoint": str(checkpoint),
        "manifest_hash": manifest_hash,
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint,
        args.cache,
        args.split,
        manifest=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
