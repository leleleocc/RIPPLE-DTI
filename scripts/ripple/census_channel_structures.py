"""Count public experimental structures for the GtoPdb ion-channel catalogue.

This is a metadata census, not a dataset of audited drug-response pairs.
Run after fetching the official GtoPdb targets_and_families.csv release.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
TARGET_URL = "https://www.guidetopharmacology.org/DATA/targets_and_families.csv"
CHANNEL_TYPES = {"lgic", "vgic", "other_ic"}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal(attribute: str, operator: str, value: object) -> dict:
    return {"type": "terminal", "service": "text", "parameters": {
        "attribute": attribute, "operator": operator, "value": value}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/planning/gtopdb/sources/targets_and_families.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/planning_20261017/structures")
    parser.add_argument("--cutoff", default="2026-10-17")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--uniprot-aliases", type=Path, help="UniProt TSV with Entry Name and Entry columns")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aliases = {}
    if args.uniprot_aliases:
        with args.uniprot_aliases.open() as handle:
            aliases = {row["Entry Name"]: row["Entry"] for row in csv.DictReader(handle, delimiter="\t")}
    text = io.StringIO(args.targets.read_text(encoding="utf-8-sig"))
    release = next(text).strip()
    rows = [row for row in csv.DictReader(text) if row["Type"] in CHANNEL_TYPES]
    excluded = [row for row in rows if row["HGNC symbol"].upper().startswith("PIEZO")]
    rows = [row for row in rows if row not in excluded]
    accession_info = defaultdict(list)
    for row in rows:
        for species in ("Human", "Rat", "Mouse"):
            for accession in row[f"{species} SwissProt"].split("|"):
                accession = accession.strip()
                if accession:
                    accession = aliases.get(accession, accession)
                    if "_" in accession:
                        raise ValueError(f"Catalogue uses entry name {accession}; resolve it with --uniprot-aliases.")
                    accession_info[accession].append({
                        "species": species.lower(), "type": row["Type"],
                        "target_id": row["Target id"], "target_name": row["Target name"],
                        "subunit_id": row["Subunit id"], "subunit_name": row["Subunit name"],
                        "family_id": row["Family id"], "family_name": row["Family name"],
                        "gene_symbol": row["HGNC symbol"],
                    })
    accessions = sorted(accession_info)
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            terminal("rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", "in", accessions),
            terminal("rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name", "exact_match", "UniProt"),
            terminal("rcsb_accession_info.initial_release_date", "less_or_equal", args.cutoff),
            terminal("rcsb_entry_info.structure_determination_methodology", "exact_match", "experimental"),
        ]},
        "return_type": "polymer_entity",
        "request_options": {
            "return_all_hits": True, "results_content_type": ["experimental"],
            "group_by": {"aggregation_method": "matching_uniprot_accession"},
            "group_by_return_type": "groups",
        },
    }
    request_path = args.output / "rcsb_request.json"
    response_path = args.output / "rcsb_response.json"
    retrieval_path = args.output / "rcsb_retrieval.json"
    if args.refresh or not response_path.exists():
        dump(request_path, query)
        subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--retry", "2", "--max-time", "120",
            "--header", "Content-Type: application/json", "--data-binary", f"@{request_path}",
            ENDPOINT, "--output", str(response_path),
        ], check=True)
        dump(retrieval_path, {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "url": ENDPOINT})
    elif json.loads(request_path.read_text(encoding="utf-8")) != query:
        raise ValueError("Cached query differs from current inputs; use --refresh or another output directory.")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    groups = response.get("group_set", [])
    if len(groups) != response.get("group_by_count", 0):
        raise ValueError("The grouped search result is incomplete.")
    per_accession = {}
    all_entities = set()
    all_entries = set()
    unexpected_accessions = []
    for group in groups:
        accession = group["identifier"]
        entities = sorted({item["identifier"] for item in group["result_set"]})
        if len(entities) != group["count"]:
            raise ValueError(f"Incomplete group for {accession}")
        if accession not in accession_info:
            unexpected_accessions.append(accession)
            continue
        entries = sorted({entity.split("_")[0] for entity in entities})
        per_accession[accession] = {
            "catalogue_records": accession_info[accession], "entity_count": len(entities),
            "entry_count": len(entries), "entry_ids": entries, "entity_ids": entities,
        }
        all_entities.update(entities)
        all_entries.update(entries)
    for accession in accessions:
        per_accession.setdefault(accession, {
            "catalogue_records": accession_info[accession], "entity_count": 0,
            "entry_count": 0, "entry_ids": [], "entity_ids": [],
        })
    with_structures = {key: value for key, value in per_accession.items() if value["entry_count"]}
    target_ids = {record["target_id"] for value in with_structures.values() for record in value["catalogue_records"]}
    family_ids = {record["family_id"] for value in with_structures.values() for record in value["catalogue_records"]}
    per_species = {}
    for species in ("human", "rat", "mouse"):
        selected = {key: value for key, value in per_accession.items()
                    if any(record["species"] == species for record in value["catalogue_records"])}
        per_species[species] = {
            "queried_accessions": len(selected),
            "accessions_with_entries": sum(value["entry_count"] > 0 for value in selected.values()),
            "accessions_with_at_least_two_entries": sum(value["entry_count"] >= 2 for value in selected.values()),
            "distinct_entries": len({entry for value in selected.values() for entry in value["entry_ids"]}),
        }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "release_cutoff": args.cutoff,
        "source_release": release, "source_url": TARGET_URL, "source_sha256": sha256(args.targets),
        "uniprot_aliases": {"path": str(args.uniprot_aliases), "sha256": sha256(args.uniprot_aliases), "mapping": aliases} if args.uniprot_aliases else None,
        "query_url": ENDPOINT, "query_sha256": sha256(request_path), "response_sha256": sha256(response_path),
        "retrieval": json.loads(retrieval_path.read_text(encoding="utf-8")),
        "channel_catalogue_rows_excluding_piezo": len(rows),
        "catalogue_target_ids_excluding_piezo": len({row["Target id"] for row in rows}),
        "excluded_piezo_target_ids": sorted({row["Target id"] for row in excluded}),
        "queried_accessions": len(accessions), "distinct_polymer_entities": len(all_entities),
        "distinct_pdb_entries": len(all_entries), "accessions_with_entries": len(with_structures),
        "accessions_with_at_least_two_entries": sum(value["entry_count"] >= 2 for value in per_accession.values()),
        "catalogue_target_ids_with_entries": len(target_ids), "catalogue_family_ids_with_entries": len(family_ids),
        "per_species": per_species, "unexpected_group_accessions": unexpected_accessions,
        "audited_drug_response_pair_count": None,
        "limitations": [
            "Catalogue includes channel subunits, associated proteins and fragments; hits are not all complete channels.",
            "An accession with multiple PDB entries is not evidence of multiple functional states or a matched drug-response pair.",
            "Same PDB entry can contain multiple catalogue proteins and is counted once in the overall entry count.",
            "Species are counted separately by accession; catalogue target IDs group orthologues and are not accession counts.",
            "Only direct PIEZO1/PIEZO2 catalogue rows are excluded; sequence-homology exclusion has not been performed.",
            "No coordinates, construct compatibility, ligand relevance, assay conditions or train/test split were audited.",
        ],
    }
    dump(args.output / "summary.json", summary)
    dump(args.output / "accession_structures.json", per_accession)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
