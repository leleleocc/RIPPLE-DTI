"""Cold splits for the leakage-controlled evaluation.

Three splits, kept separate and never pooled into one reported number:

- cold-target: test proteins share no sequence cluster with training proteins;
- cold-pocket: test pockets share no structural cluster with training pockets;
- cold-both: both constraints hold at once.

The same compound never crosses from train to test, and every accession on the
PIEZO exclusion list is removed before splitting.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--split", required=True, choices=["cold-target", "cold-pocket", "cold-both"])
    parser.add_argument("--identity-threshold", type=float, default=0.3)
    parser.add_argument("--exclude", default="data/external/piezo_exclusion.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"{args.split} split is specified but the clustering step is not "
        f"implemented; it must honour {args.exclude} and the identity threshold"
    )


if __name__ == "__main__":
    main()
