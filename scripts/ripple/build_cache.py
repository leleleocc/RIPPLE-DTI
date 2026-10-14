"""Build the residue-level feature cache RIPPLE trains on.

RIPPLE needs, per protein:

- unpooled language-model residue features and a validity mask;
- residue coordinates from an experimental or AlphaFold structure;
- a residue graph connecting C-alpha atoms within a distance cutoff;
- candidate pockets predicted from the apo or AlphaFold structure, never from
  the test ligand's coordinates;
- pose coordinates and RMSD-to-crystal, only for samples that have them.

Run scripts/ripple/prepare_sources.py first; this script only featurizes.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="processed sample table")
    parser.add_argument("--structures", required=True, help="PDB or AlphaFold structures")
    parser.add_argument("--out", required=True, help="cache directory to write")
    parser.add_argument("--residue-cutoff", type=float, default=8.0)
    parser.add_argument("--num-pockets", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "featurization depends on the processed sample table; schema is fixed in "
        f"src/ripple/data/schema.py, output goes to {args.out}"
    )


if __name__ == "__main__":
    main()
