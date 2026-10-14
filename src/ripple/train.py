"""RIPPLE training entry point.

Runs the curriculum pocket -> dti -> propagation -> pharmacology -> joint.
Each stage loads the previous stage's best checkpoint and enables only the
losses that have supervision there; see :func:`stage_loss_weights`.
"""

import argparse
from pathlib import Path

from src.ripple.training import TrainingStage, stage_loss_weights

CURRICULUM = (
    TrainingStage.POCKET,
    TrainingStage.DTI,
    TrainingStage.PROPAGATION,
    TrainingStage.PHARMACOLOGY,
    TrainingStage.JOINT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ripple_network.yaml")
    parser.add_argument("--cache", required=True, help="residue-level feature cache")
    parser.add_argument("--split", required=True, help="cold-target, cold-pocket, or cold-both")
    parser.add_argument("--stage", choices=[stage.value for stage in CURRICULUM])
    parser.add_argument("--save-dir", default="outputs/ripple")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_stage(stage: TrainingStage, args: argparse.Namespace, resume_from: Path | None) -> Path:
    """Train one curriculum stage and return its best checkpoint path."""
    weights = stage_loss_weights(stage)
    active = [name for name, value in vars(weights).items() if value > 0]
    save_dir = Path(args.save_dir) / args.split / stage.value
    save_dir.mkdir(parents=True, exist_ok=True)
    # The loop itself waits on RIPPLEDataset, which needs the residue-level
    # cache. Everything up to the batch is wired; this is the open seam.
    raise NotImplementedError(
        f"stage={stage.value} split={args.split} losses={active} "
        f"resume={resume_from} save={save_dir}"
    )


def main() -> None:
    args = parse_args()
    stages = [TrainingStage(args.stage)] if args.stage else list(CURRICULUM)
    checkpoint: Path | None = None
    for stage in stages:
        checkpoint = run_stage(stage, args, checkpoint)


if __name__ == "__main__":
    main()
