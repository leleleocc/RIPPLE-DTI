"""Collect source-linked UniProt records and census-selected PDB coordinates.

Raw coordinates are not audited reference assemblies or trainable samples.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

if __package__:
    from .download_sources import ROOT, collect, digest, dump, now
else:
    from download_sources import ROOT, collect, digest, dump, now


ACCESSION = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?"
)
FIELDS = "accession,id,reviewed,protein_name,gene_names,organism_name,organism_id,length,sequence,ft_binding,ft_act_site,ft_transmem,cc_function"


def gtop_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        next(handle)
        yield from csv.DictReader(handle)


def channel_entry_names(raw):
    names = set()
    for row in gtop_rows(raw / "gtopdb/targets_and_families.csv"):
        if row["Type"] not in {"lgic", "vgic", "other_ic"}:
            continue
        for species in ("Human", "Rat", "Mouse"):
            for token in re.split(r"[;,|\s]+", row[f"{species} SwissProt"]):
                if re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", token):
                    names.add(token)
    return sorted(names)


def accession_inventory(raw, release):
    sources = defaultdict(set)
    counts = Counter()
    invalid = Counter()
    inputs = []

    def add(text, source):
        for token in re.split(r"[;,|\s]+", text.strip()):
            if not token or token in {"-", "NA", "N/A"}:
                continue
            if ACCESSION.fullmatch(token):
                sources[token].add(source)
            else:
                invalid[source] += 1

    path = raw / "bindingdb" / f"BindingDB_All_{release}_tsv.zip"
    inputs.append(path)
    with (
        zipfile.ZipFile(path) as archive,
        archive.open("BindingDB_All.tsv") as binary,
        io.TextIOWrapper(binary) as handle,
    ):
        reader = csv.reader(handle, delimiter="\t")
        fields = next(reader)
        columns = [
            i for i, key in enumerate(fields) if "Primary ID of Target Chain" in key
        ]
        if not columns:
            raise ValueError("BindingDB accession columns missing")
        for row in reader:
            counts["bindingdb_rows"] += 1
            if counts["bindingdb_rows"] % 500000 == 0:
                print(
                    f"{now()} Scanned {counts['bindingdb_rows']:,} BindingDB rows",
                    flush=True,
                )
            if len(row) != len(fields):
                raise ValueError(
                    f"Malformed BindingDB row {counts['bindingdb_rows'] + 1}"
                )
            for i in columns:
                if row[i]:
                    add(row[i], "bindingdb")
    path = raw / "biolip/BioLiP.txt.gz"
    inputs.append(path)
    biolip_chains = set()
    with gzip.open(path, "rt") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 21:
                raise ValueError("Unexpected BioLiP column count")
            counts["biolip_sites"] += 1
            add(row[17], "biolip")
            biolip_chains.add((row[0].lower(), row[1]))
    path = raw / "sifts/pdb_chain_uniprot.csv.gz"
    inputs.append(path)
    pdbbind_entries = set()
    index_path = raw / "pdbbind/index.tar.gz"
    if index_path.exists():
        inputs.append(index_path)
        with tarfile.open(index_path) as archive:
            text = (
                archive.extractfile("index/INDEX_general_PL.2020R1.lst").read().decode()
            )
        pdbbind_entries = {
            line.split()[0].lower()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        }
        counts["pdbbind_entries"] = len(pdbbind_entries)
    with gzip.open(path, "rt") as handle:
        next(handle)
        for row in csv.DictReader(handle):
            counts["sifts_chain_rows"] += 1
            if (row["PDB"].lower(), row["CHAIN"]) in biolip_chains:
                add(row["SP_PRIMARY"], "sifts_biolip")
            if row["PDB"].lower() in pdbbind_entries:
                add(row["SP_PRIMARY"], "sifts_pdbbind")
    channel_target_ids = set()
    for name in ("targets_and_families.csv", "interactions.csv"):
        path = raw / "gtopdb" / name
        inputs.append(path)
        for row in gtop_rows(path):
            counts["gtopdb_" + name] += 1
            if name == "targets_and_families.csv":
                if row["Type"] not in {"lgic", "vgic", "other_ic"}:
                    continue
                channel_target_ids.add(row["Target id"])
            elif row["Target ID"] not in channel_target_ids:
                continue
            for key, value in row.items():
                is_target_accession = key and (
                    "SwissProt" in key or key == "Target UniProt ID"
                )
                if is_target_accession and value:
                    add(value, "gtopdb")
    result = {
        "generated_at_utc": now(),
        "counts": dict(counts),
        "accessions": {k: sorted(v) for k, v in sorted(sources.items())},
        "invalid_accession_tokens": dict(invalid),
        "inputs": {str(p.relative_to(raw)): digest(p) for p in inputs},
        "note": "Source identities only; no label cleaning, split assignment or sequence clustering performed.",
    }
    dump(raw / "uniprot/accession_inventory.json", result)
    return result


def uniprot(raw, release, workers, min_free_gb):
    inventory_path = raw / "uniprot/accession_inventory.json"
    inventory = (
        json.loads(inventory_path.read_text())
        if inventory_path.exists()
        else accession_inventory(raw, release)
    )
    for name, expected in inventory["inputs"].items():
        if digest(raw / name) != expected:
            raise ValueError(
                "Source snapshot changed; create a new accession inventory"
            )
    accessions = sorted(inventory["accessions"])
    # Isoforms are preserved in the inventory. Fetch their canonical records too;
    # an isoform sequence must never silently replace a canonical sequence.
    canonical = sorted({acc.split("-")[0] for acc in accessions})
    jobs, batches = [], []
    for start in range(0, len(canonical), 100):
        batch = canonical[start : start + 100]
        name = f"uniprot/features_{start // 100:05d}.tsv"
        url = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
            {
                "query": " OR ".join("accession:" + acc for acc in batch),
                "format": "tsv",
                "fields": FIELDS,
            }
        )
        jobs.append((name, url, "tsv"))
        batches.append({"path": name, "requested": batch})
    entry_names = channel_entry_names(raw)
    if entry_names:
        name = "uniprot/features_entry_names.tsv"
        url = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
            {
                "query": " OR ".join("id:" + name for name in entry_names),
                "format": "tsv",
                "fields": FIELDS,
            }
        )
        jobs.append((name, url, "tsv"))
        batches.append(
            {"path": name, "requested": [], "requested_entry_names": entry_names}
        )
    dump(
        raw / "uniprot/requests.json",
        {"canonical_accessions": len(canonical), "batches": batches},
    )
    failures = collect(raw, jobs, workers, min_free_gb)
    returned = set()
    bad_rows = []
    resolved_entry_names = {}
    for batch in batches:
        path = raw / batch["path"]
        if not path.exists():
            continue
        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "Entry",
                "Sequence",
                "Length",
                "Organism",
                "Organism (ID)",
                "Reviewed",
                "Binding site",
                "Active site",
                "Transmembrane",
                "Function [CC]",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected UniProt fields: {path}")
            for row in reader:
                if row.get("Entry Name") in entry_names:
                    resolved_entry_names[row["Entry Name"]] = row["Entry"]
                if (
                    not row["Sequence"]
                    or not row["Length"].isdigit()
                    or int(row["Length"]) != len(row["Sequence"])
                ):
                    bad_rows.append(row["Entry"])
                returned.add(row["Entry"])
    canonical = sorted(set(canonical) | set(resolved_entry_names.values()))
    summary = {
        "generated_at_utc": now(),
        "requested_canonical_accessions": len(canonical),
        "returned_accessions": len(returned),
        "not_returned_under_requested_id": sorted(set(canonical) - returned),
        "unexpected_or_replacement_accessions": sorted(returned - set(canonical)),
        "isoforms_requiring_separate_resolution": [a for a in accessions if "-" in a],
        "invalid_sequence_rows": bad_rows,
        "resolved_entry_names": resolved_entry_names,
        "unresolved_entry_names": sorted(set(entry_names) - set(resolved_entry_names)),
        "failures": failures,
    }
    dump(raw / "uniprot/summary.json", summary)
    print(
        json.dumps(
            {k: len(v) if isinstance(v, list) else v for k, v in summary.items()},
            indent=2,
        ),
        flush=True,
    )
    return failures


def structures(raw, workers, min_free_gb):
    census = json.loads((raw / "channel_census/accession_structures.json").read_text())
    entries = sorted(
        {entry for record in census.values() for entry in record["entry_ids"]}
    )
    jobs = [
        (
            f"rcsb/cif/{pdb.lower()}.cif.gz",
            f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
            "cif_gzip",
        )
        for pdb in entries
    ]
    failures = collect(raw, jobs, workers, min_free_gb)
    dump(
        raw / "rcsb/summary.json",
        {
            "generated_at_utc": now(),
            "requested_entries": len(entries),
            "failures": failures,
            "census_sha256": digest(raw / "channel_census/accession_structures.json"),
            "note": "Asymmetric-unit mmCIF contains assembly definitions. Assembly selection, materialization and residue mapping are pending audits.",
        },
    )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("uniprot", "structures"))
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--bindingdb-release", default="202610")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=15)
    args = parser.parse_args()
    if args.kind == "uniprot":
        failures = uniprot(
            args.raw, args.bindingdb_release, args.workers, args.min_free_gb
        )
    else:
        failures = structures(args.raw, args.workers, args.min_free_gb)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
