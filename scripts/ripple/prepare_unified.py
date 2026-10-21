"""Build versioned source staging tables without writing to raw inputs."""

import argparse
import fcntl
import json
import multiprocessing
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.ripple import staging_sources as sources
from scripts.ripple.source_alignment import log, write_json
from scripts.ripple.staging_chemistry import (
    STANDARDIZATION_VERSION,
    ChemicalRegistry,
    standardize,
)
from scripts.ripple.staging_common import (
    TableWriter,
    file_hash,
    packed,
    raw_snapshot,
    schema,
    table,
    validate_paths,
)

SCHEMA_VERSION = 1
SOURCE_TABLES = {
    "bindingdb": "sources/bindingdb/measurements.parquet",
    "biolip2": "sources/biolip2/sites.parquet",
    "pdbbind": "sources/pdbbind/complexes.parquet",
    "gtopharm": "sources/gtopharm/interactions.parquet",
}


def export_chemistry(registry, out, pool):
    db = registry.db
    done = db.execute(
        "SELECT count(*) FROM inputs WHERE result IS NOT NULL"
    ).fetchone()[0]
    total = db.execute("SELECT count(*) FROM inputs").fetchone()[0]
    while True:
        pending = db.execute(
            "SELECT input_id, raw_smiles FROM inputs WHERE result IS NULL LIMIT 8000"
        ).fetchall()
        if not pending:
            break
        results = pool.map(standardize, pending, chunksize=32)
        db.executemany(
            "UPDATE inputs SET result=? WHERE input_id=?",
            [(packed(result), key) for key, result in results],
        )
        db.commit()
        done += len(results)
        log(f"Chemistry: {done:,}/{total:,} unique inputs standardized")
    lookup, molecules, statuses = {}, {}, Counter()
    input_fields = {
        "chemical_input_id": "s",
        "raw_smiles": "s",
        "status": "s",
        "molecule_id": "s",
        "inchikey": "s",
        "canonical_smiles": "s",
        "fragments_removed": "i",
        "standardization_version": "s",
        "detail_json": "s",
    }
    with TableWriter(
        out / "identities/chemical_inputs.parquet", schema(**input_fields)
    ) as writer:
        for key, smiles, encoded in db.execute(
            "SELECT input_id, raw_smiles, result FROM inputs ORDER BY input_id"
        ):
            result = json.loads(encoded)
            molecule_id = result.get("inchikey") if result["status"] == "ok" else None
            lookup[key] = (molecule_id, result["status"])
            statuses[result["status"]] += 1
            if molecule_id:
                record = molecules.setdefault(molecule_id, set())
                record.add(result["canonical_smiles"])
            writer.add(
                {
                    "chemical_input_id": key,
                    "raw_smiles": smiles,
                    "status": result["status"],
                    "molecule_id": molecule_id,
                    "inchikey": molecule_id,
                    "canonical_smiles": result.get("canonical_smiles"),
                    "fragments_removed": result.get("fragments_removed"),
                    "standardization_version": STANDARDIZATION_VERSION,
                    "detail_json": encoded,
                }
            )
    table(
        out / "identities/molecules.parquet",
        {
            "molecule_id": "s",
            "inchikey": "s",
            "canonical_smiles": "s",
            "smiles_variants": "ss",
            "representation_count": "i",
            "standardization_version": "s",
        },
        (
            {
                "molecule_id": key,
                "inchikey": key,
                "canonical_smiles": min(variants),
                "smiles_variants": sorted(variants),
                "representation_count": len(variants),
                "standardization_version": STANDARDIZATION_VERSION,
            }
            for key, variants in sorted(molecules.items())
        ),
    )
    counts = defaultdict(set)
    for source, source_id, key in db.execute(
        "SELECT source, source_id, input_id FROM xrefs"
    ):
        if lookup[key][0]:
            counts[source, source_id].add(lookup[key][0])
    xrefs = table(
        out / "identities/molecule_xrefs.parquet",
        {
            "source": "s",
            "source_id": "s",
            "chemical_input_id": "s",
            "molecule_id": "s",
            "chemistry_status": "s",
            "source_id_conflict": "b",
        },
        (
            {
                "source": source,
                "source_id": source_id,
                "chemical_input_id": key,
                "molecule_id": lookup[key][0],
                "chemistry_status": lookup[key][1],
                "source_id_conflict": len(counts[source, source_id]) > 1,
            }
            for source, source_id, key in db.execute(
                "SELECT source, source_id, input_id FROM xrefs ORDER BY source, source_id, input_id"
            )
        ),
    )
    return lookup, {
        "unique_inputs": total,
        "unique_molecules": len(molecules),
        "source_crosswalk_rows": xrefs,
        "statuses": dict(statuses),
        "conflicting_source_ids": sum(len(value) > 1 for value in counts.values()),
        "multi_representation_molecules": sum(
            len(value) > 1 for value in molecules.values()
        ),
    }


def attach_identities(out, lookup, protein_map):
    reports, protein_ids = {}, {row["protein_id"] for row in protein_map.values()}
    # Source ligand IDs can alias the same parent. Audit conflicts after chemistry.
    gtop = pq.read_table(out / SOURCE_TABLES["gtopharm"]).to_pylist()
    effects = defaultdict(set)
    for row in gtop:
        molecule_id = lookup.get(row["chemical_input_id"], (None, None))[0]
        if molecule_id and row["protein_id"] and row["effect"] is not None:
            effects[molecule_id, row["protein_id"]].add(row["effect"])
    for source, relative in SOURCE_TABLES.items():
        path = out / relative
        reader = pq.ParquetFile(path)
        added = schema(
            molecule_id="s",
            chemistry_status="s",
            normalized_pair_direction_conflict="b",
            label_candidate="b",
            blocking_reasons="ss",
            trainable="b",
            split="s",
        )
        base_fields = [f for f in reader.schema_arrow if f.name not in added.names]
        output_schema = pa.schema(base_fields + list(added))
        counts, reasons, duplicates = Counter(), Counter(), Counter()
        with TableWriter(path, output_schema) as writer:
            for batch in reader.iter_batches(batch_size=10000):
                for row in batch.to_pylist():
                    key = row["chemical_input_id"]
                    if key and key not in lookup:
                        raise ValueError(
                            f"Dangling chemical input in {row['record_id']}"
                        )
                    molecule_id, status = lookup.get(
                        key, (None, "missing_chemical_record")
                    )
                    if row["protein_id"] and row["protein_id"] not in protein_ids:
                        raise ValueError(f"Dangling protein in {row['record_id']}")
                    blocks = []
                    if not molecule_id:
                        blocks.append("chemistry:" + status)
                    if not row["protein_id"]:
                        blocks.append("unresolved_protein")
                    conflict = False
                    if source == "gtopharm":
                        conflict = len(effects[molecule_id, row["protein_id"]]) > 1
                        if not row["effect_candidate"]:
                            blocks.append("pharmacology:" + row["direction_status"])
                        if conflict:
                            blocks.append("normalized_pair_direction_conflict")
                        if not row["channel_catalogue"]:
                            blocks.append("outside_channel_catalogue")
                        if row["species_status"] != "verified":
                            blocks.append("species:" + row["species_status"])
                    elif source == "biolip2":
                        if row["excluded_ligand_class"]:
                            blocks.append("excluded_ligand_class")
                        if not row["binding_auth"]:
                            blocks.append("no_binding_annotation")
                    else:
                        if (
                            row["affinity_value"] is None
                            or row["endpoint_type"] is None
                        ):
                            blocks.append("measurement:" + row["measurement_status"])
                        if source == "pdbbind" and (
                            row["covalent"] or row["incomplete_ligand"]
                        ):
                            blocks.append("covalent_or_incomplete_ligand")
                    candidate = not blocks
                    blocks.extend(
                        [
                            "structure_and_residue_audit_pending",
                            "split_pending",
                            "features_pending",
                        ]
                    )
                    row.update(
                        molecule_id=molecule_id,
                        chemistry_status=status,
                        normalized_pair_direction_conflict=conflict,
                        label_candidate=candidate,
                        blocking_reasons=blocks,
                        trainable=False,
                        split=None,
                    )
                    writer.add(row)
                    counts["rows"] += 1
                    counts["label_candidates_before_structural_audit"] += candidate
                    counts["mapped_chemical_rows"] += molecule_id is not None
                    counts["resolved_protein_rows"] += row["protein_id"] is not None
                    reasons.update(blocks)
                    duplicates[row["evidence_hash"]] += 1
        # Measurements from different endpoints of one raw record share evidence.
        reports[source] = dict(
            counts,
            reasons=dict(reasons),
            evidence_groups=len(duplicates),
            repeated_evidence_rows=sum(n - 1 for n in duplicates.values()),
        )
        log(f"Validated {source}: {counts['rows']:,} rows")
    return reports


def run(args):
    raw, out = validate_paths(args.raw, args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".processing.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = out / "audit"
    audit.mkdir(exist_ok=True)
    log("Hashing read-only raw inputs")
    before = raw_snapshot(raw)
    before_path = audit / "raw_before.json"
    if before_path.exists():
        if json.loads(before_path.read_text()) != before:
            raise ValueError("Raw snapshot changed since the previous run")
        if not args.resume:
            raise ValueError(
                "Existing output: pass --resume or choose an empty output directory"
            )
    else:
        write_json(before_path, before)
    environment = {
        name: version(name) for name in ("rdkit", "numpy", "pyarrow", "gemmi")
    }
    scripts = [
        Path(__file__),
        *Path(__file__).parent.glob("staging_*.py"),
        Path(__file__).with_name("source_alignment.py"),
    ]
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "standardization_version": STANDARDIZATION_VERSION,
        "environment": environment,
        "scripts": {p.name: file_hash(p) for p in sorted(scripts)},
    }
    state_path = audit / "progress.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"fingerprint": fingerprint, "stages": {}}
    )
    if state["fingerprint"] != fingerprint:
        raise ValueError(
            "Code or dependency versions changed; use a new output directory"
        )
    registry = ChemicalRegistry(out / ".chemistry.sqlite")

    def stage(name, work):
        if name not in state["stages"]:
            log(f"Starting {name}")
            result = work()
            registry.db.commit()
            state["stages"][name] = result
            write_json(state_path, state)
        return state["stages"][name]

    success = False
    try:
        protein_map, aliases, protein_count = sources.proteins(raw, out)
        chain_map, pdb_map, segment_count = sources.sifts(raw, out, protein_map)
        stage(
            "gtopharm", lambda: sources.gtopdb(raw, out, registry, protein_map, aliases)
        )
        stage(
            "biolip2",
            lambda: sources.biolip(raw, out, registry, protein_map, chain_map),
        )
        stage(
            "pdbbind", lambda: sources.pdbbind(raw, out, registry, protein_map, pdb_map)
        )
        stage("bindingdb", lambda: sources.bindingdb(raw, out, registry, protein_map))
        chain_map.clear()
        pdb_map.clear()
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            stage("rcsb", lambda: sources.structures(raw, out, pool))
            lookup, chemistry = export_chemistry(registry, out, pool)
        source_report = attach_identities(out, lookup, protein_map)
        registry.close()
        registry = None
        tables = {}
        for path in sorted(out.rglob("*.parquet")):
            metadata = pq.ParquetFile(path).metadata
            tables[str(path.relative_to(out))] = {
                "rows": metadata.num_rows,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        report = dict(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            raw=str(raw),
            output=str(out),
            python=platform.python_version(),
            **fingerprint,
            proteins=protein_count,
            sifts_segments=segment_count,
            stages=state["stages"],
            chemistry=chemistry,
            sources=source_report,
            tables=tables,
            trainable_samples=0,
            pending=[
                "structure/residue/construct audit",
                "cross-source evidence deduplication",
                "sequence/ortholog splits and PIEZO-family exclusion",
                "feature caches",
            ],
            raw_integrity_verified=False,
        )
        success = True
    finally:
        if registry is not None:
            registry.close()
        log("Verifying raw tree, content hashes, sizes, timestamps and modes")
        after = raw_snapshot(raw)
        write_json(audit / "raw_after.json", after)
        changes = [
            key
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        ]
        write_json(
            audit / "raw_integrity.json",
            {
                "unchanged": not changes,
                "files": sum("sha256" in item for item in before.values()),
                "changed_paths": changes,
                "checked_fields": [
                    "path",
                    "content_sha256",
                    "size",
                    "mtime_ns",
                    "mode",
                ],
            },
        )
        if changes:
            raise RuntimeError(f"Raw changed during processing: {changes[:10]}")
        if success:
            report["raw_integrity_verified"] = True
            write_json(out / "report.json", report)
            log(
                packed(
                    {
                        key: report[key]
                        for key in [
                            "proteins",
                            "sifts_segments",
                            "chemistry",
                            "raw_integrity_verified",
                        ]
                    }
                )
            )
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/unified_v1"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
