"""Normalize each upstream database into the shared label contract.

Each source keeps only the labels it genuinely has; see SourceTable in
src/ripple/data/schema.py. Records are not joined into one table here, because
a missing mechanism annotation is not a negative mechanism label.

PIEZO1 (Q92508), PIEZO2 (Q9H5I1), and their close homologs are written to the
exclusion list and removed from every training split. They are held out for the
external case study, not used as supervision.
"""

import argparse

SOURCES = (
    "bindingdb",
    "chembl",
    "biosnap",
    "pdbbind",
    "biolip2",
    "uniprot",
    "crossdocked",
    "casf",
    "gtopharm",
    "asd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--raw", required=True, help="directory of downloaded files")
    parser.add_argument("--out", required=True, help="processed sample table")
    parser.add_argument("--exclude", default="data/external/piezo_exclusion.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"parser for {args.source} is not written yet; it must emit only the "
        f"labels declared for that source and drop accessions listed in {args.exclude}"
    )


if __name__ == "__main__":
    main()
