"""Align the five downloaded sources and report label/structure coverage."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ripple.source_alignment import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--mapping-limit",
        type=int,
        default=300,
        help="number of distinct PDBbind complexes to verify residue by residue",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--offline", action="store_true", help="only use cached SIFTS XML"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse completed metadata stages"
    )
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    args = parser.parse_args()
    if args.mapping_limit < 0 or args.workers < 1 or args.contact_cutoff <= 0:
        parser.error(
            "mapping-limit must be nonnegative; workers and contact-cutoff must be positive"
        )
    return args


if __name__ == "__main__":
    run(parse_args())
