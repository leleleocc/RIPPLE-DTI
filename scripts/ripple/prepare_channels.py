"""Reproducible, source-preserving ion-channel pilot, without causal pseudo-labels."""

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import gemmi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


class Downloads:
    def __init__(self, directory, offline=False):
        self.directory = directory
        self.offline = offline
        self.provenance = []

    def get(self, name, url):
        path = self.directory / name
        if not path.exists():
            if self.offline:
                raise FileNotFoundError(path)
            print(f"Fetching {name}", file=sys.stderr, flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "RIPPLE-channel-audit/1"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
                headers = dict(response.headers)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            path.with_suffix(path.suffix + ".source.json").write_text(
                json.dumps(
                    {
                        "url": url,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "last_modified": headers.get("Last-Modified"),
                    },
                    indent=2,
                )
                + "\n"
            )
        content = path.read_bytes()
        metadata = json.loads(
            path.with_suffix(path.suffix + ".source.json").read_text()
        )
        self.provenance.append(
            {
                **metadata,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
        return content

    def json(self, name, url):
        return json.loads(self.get(name, url))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_table(path, rows, table_schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table_schema), path, compression="zstd"
    )


def sifts_residues(content, accession):
    """Exact observed-residue mapping; author numbering is never a linear offset."""
    root = ET.fromstring(gzip.decompress(content))
    mapping = {}
    for residue in root.iter():
        if residue.tag.rsplit("}", 1)[-1] != "residue":
            continue
        refs = [x.attrib for x in residue if x.tag.rsplit("}", 1)[-1] == "crossRefDb"]
        pdb = next((x for x in refs if x.get("dbSource") == "PDB"), None)
        uni = next(
            (
                x
                for x in refs
                if x.get("dbSource") == "UniProt"
                and x.get("dbAccessionId") == accession
            ),
            None,
        )
        if pdb and uni:
            mapping[(pdb["dbChainId"], pdb["dbResNum"])] = int(uni["dbResNum"])
    return mapping


def assembly_geometry(content, entry, entity, sifts, landmarks):
    block = gemmi.cif.read_string(gzip.decompress(content).decode()).sole_block()
    structure = gemmi.make_structure_from_block(block)
    # Downloaded assembly files have generated label_asym_ids. Match the channel
    # by entity_id, and recover source auth chains from _atom_site for SIFTS.
    source_chain = {}
    for row in block.find(["_atom_site.label_asym_id", "_atom_site.auth_asym_id"]):
        source_chain[str(row[0])] = str(row[1])
    for row in block.find(
        [
            "_pdbx_chain_remapping.label_asym_id",
            "_pdbx_chain_remapping.orig_auth_asym_id",
        ]
    ):
        source_chain[str(row[0])] = str(row[1])
    entity_id = entity["rcsb_polymer_entity_container_identifiers"]["entity_id"]
    residues = []
    ring_members = {landmark["name"]: [] for landmark in landmarks}
    for chain in structure[0]:
        for residue in chain:
            if residue.entity_id != entity_id:
                continue
            atom = residue.find_atom("CA", "*")
            if not atom or not np.isfinite([atom.pos.x, atom.pos.y, atom.pos.z]).all():
                continue
            source = source_chain.get(residue.subchain, chain.name)
            auth_number = str(residue.seqid)
            uniprot = sifts.get((source, auth_number))
            record = {
                "residue_index": len(residues),
                "chain_id": chain.name,
                "source_auth_chain": source,
                "auth_seq_id": auth_number,
                "residue_name": residue.name,
                "uniprot_position": uniprot,
                "coord": [atom.pos.x, atom.pos.y, atom.pos.z],
            }
            residues.append(record)
            for landmark in landmarks:
                if residue.seqid.num == landmark["auth_seq_id"]:
                    if residue.name != landmark["residue_name"]:
                        raise ValueError(
                            f"{entry['pdb_id']} landmark identity mismatch: {record}"
                        )
                    ring_members[landmark["name"]].append(record)
    metrics = []
    for name, members in ring_members.items():
        chains = {member["chain_id"] for member in members}
        valid = len(members) == len(chains) == entry["expected_subunits"]
        valid = valid and all(
            member["uniprot_position"] is not None for member in members
        )
        pairs = list(itertools.combinations(members, 2))
        distances = [
            float(np.linalg.norm(np.array(a["coord"]) - b["coord"])) for a, b in pairs
        ]
        metrics.append(
            {
                "metric_id": name + "_mean_cross_subunit_CA_distance",
                "value_angstrom": float(np.mean(distances)) if valid else None,
                "valid": valid,
                "observed_subunits": len(chains),
                "uniprot_positions": sorted(
                    {
                        m["uniprot_position"]
                        for m in members
                        if m["uniprot_position"] is not None
                    }
                ),
                "pair_indices": [
                    [a["residue_index"], b["residue_index"]] for a, b in pairs
                ],
            }
        )
    return residues, metrics


def census(downloads, config):
    accessions = sorted(
        {a for family in config["families"] for a in family["accessions"]}
    )
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "in",
                "value": accessions,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    result = downloads.json(
        "rcsb_census.json",
        "https://search.rcsb.org/rcsbsearch/v2/query?json="
        + urllib.parse.quote(json.dumps(query)),
    )
    return {
        "accessions": accessions,
        "entry_count": result["total_count"],
        "pdb_ids": sorted(x["identifier"] for x in result["result_set"]),
    }


def pharmacology(downloads, config):
    raw = downloads.get(
        "gtopdb_interactions.csv",
        "https://www.guidetopharmacology.org/DATA/interactions.csv",
    )
    lines = io.StringIO(raw.decode("utf-8-sig"))
    release = next(lines).strip().strip('"')
    families = config["families"]
    selected = []
    for row in csv.DictReader(lines):
        accessions = set(row["Target UniProt ID"].split("|"))
        symbols = set(row["Target Gene Symbol"].split("|"))
        for family in families:
            if accessions.intersection(family["accessions"]) or symbols.intersection(
                family["gene_symbols"]
            ):
                selected.append(
                    {
                        **row,
                        "family": family["family"],
                        "release": release,
                        "gating_direction_mask": False,
                        "state_preference_mask": False,
                        "effect_evidence_mask": bool(
                            row["PubMed ID"] or row["Webpage URLs"]
                        ),
                        "split_group": "channel_family:" + family["family"],
                    }
                )
    return release, selected


def prepare(config_path, raw_dir, output_dir, offline=False):
    config = json.loads(config_path.read_text())
    downloads = Downloads(raw_dir, offline)
    metadata_census = census(downloads, config)
    release, effects = pharmacology(downloads, config)
    states, geometries, details = [], [], {}
    for entry in config["structures"]:
        pdb = entry["pdb_id"]
        metadata = downloads.json(
            f"{pdb}/entry.json", f"https://data.rcsb.org/rest/v1/core/entry/{pdb}"
        )
        entities = [
            downloads.json(
                f"{pdb}/entity_{eid}.json",
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{eid}",
            )
            for eid in metadata["rcsb_entry_container_identifiers"][
                "polymer_entity_ids"
            ]
        ]
        entity = next(
            e
            for e in entities
            if entry["uniprot_id"]
            in e["rcsb_polymer_entity_container_identifiers"].get("uniprot_ids", [])
        )
        sifts_raw = downloads.get(
            f"{pdb}/sifts.xml.gz",
            f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb.lower()}.xml.gz",
        )
        assembly_raw = downloads.get(
            f"{pdb}/assembly{entry['assembly_id']}.cif.gz",
            f"https://files.rcsb.org/download/{pdb}-assembly{entry['assembly_id']}.cif.gz",
        )
        mapping = sifts_residues(sifts_raw, entry["uniprot_id"])
        residues, metrics = assembly_geometry(
            assembly_raw, entry, entity, mapping, config["landmarks"][entry["family"]]
        )
        poly = entity["entity_poly"]
        state = {
            **entry,
            "split_group": "channel_family:" + entry["family"],
            "title": metadata["struct"]["title"],
            "mutation_count": poly.get("rcsb_mutation_count", 0),
            "construct_sequence": poly["pdbx_seq_one_letter_code_can"],
            "construct_details": entity.get("rcsb_polymer_entity", {}),
            "source_organisms": entity.get("rcsb_entity_source_organism", []),
            "experimental_method": metadata["exptl"],
            "resolution_angstrom": metadata["rcsb_entry_info"].get(
                "resolution_combined"
            ),
            "assembly_subunits": len({r["chain_id"] for r in residues}),
            "ca_residues": len(residues),
            "mapped_ca_residues": sum(
                r["uniprot_position"] is not None for r in residues
            ),
            "modeled_nonpolymer_entities": metadata["rcsb_entry_info"].get(
                "nonpolymer_entity_count", 0
            ),
            "evidence_url": f"https://doi.org/{entry['doi']}",
            "state_evidence_level": "publication_assigned; not a functional assay label",
        }
        if state["assembly_subunits"] != entry["expected_subunits"]:
            raise ValueError(
                f"{pdb}: incomplete channel assembly: {state['assembly_subunits']}"
            )
        states.append(state)
        for metric in metrics:
            geometries.append(
                {"pdb_id": pdb, "uniprot_id": entry["uniprot_id"], **metric}
            )
        details[pdb] = {"state": state, "residues": residues, "metrics": metrics}
        write_json(output_dir / "structures" / f"{pdb}.json", details[pdb])

    pairs = []
    for pair in config["pairs"]:
        reference, target = details[pair["reference"]], details[pair["target"]]
        same_construct = (
            reference["state"]["construct_sequence"]
            == target["state"]["construct_sequence"]
        )
        for ref_metric, target_metric in zip(
            reference["metrics"], target["metrics"], strict=True
        ):
            if ref_metric["metric_id"] != target_metric["metric_id"]:
                raise ValueError("Metric correspondence mismatch")
            valid = ref_metric["valid"] and target_metric["valid"]
            valid = (
                valid
                and ref_metric["uniprot_positions"]
                == target_metric["uniprot_positions"]
            )
            pairs.append(
                {
                    **pair,
                    "pair_id": pair["reference"] + "_to_" + pair["target"],
                    "uniprot_id": reference["state"]["uniprot_id"],
                    "split_group": reference["state"]["split_group"],
                    "metric_id": ref_metric["metric_id"],
                    "reference_angstrom": ref_metric["value_angstrom"],
                    "target_angstrom": target_metric["value_angstrom"],
                    "delta_angstrom": target_metric["value_angstrom"]
                    - ref_metric["value_angstrom"]
                    if valid
                    else None,
                    "geometry_comparison_mask": valid,
                    "same_construct": same_construct,
                    "reference_state": reference["state"]["state"],
                    "target_state": target["state"]["state"],
                    "reference_ligands": reference["state"]["ligands"],
                    "target_ligands": target["state"]["ligands"],
                    "evidence_urls": [
                        reference["state"]["evidence_url"],
                        target["state"]["evidence_url"],
                    ],
                    "single_ligand_response_mask": False,
                    "state_preference_mask": False,
                    "gating_direction_mask": False,
                    "response_exclusion_reason": pair["reason"]
                    + "; no matched ligand intervention evidence curated",
                    "reference_pair_indices": ref_metric["pair_indices"],
                }
            )

    # Keep chemistry in a separate source table; association by ligand ID is exact.
    ligand_raw = downloads.get(
        "gtopdb_ligands.csv", "https://www.guidetopharmacology.org/DATA/ligands.csv"
    )
    ligand_lines = io.StringIO(ligand_raw.decode("utf-8-sig"))
    next(ligand_lines)
    ids = {row["Ligand ID"] for row in effects}
    ligands = [
        row for row in csv.DictReader(ligand_lines) if row.get("Ligand ID") in ids
    ]
    chemistry = {}
    for ligand in ligands:
        mol = Chem.MolFromSmiles(ligand.get("SMILES") or "")
        if mol is not None and mol.GetNumAtoms():
            chemistry[ligand["Ligand ID"]] = Chem.MolToSmiles(mol)
    labels = []
    effect_map = config["effect_type_map"]
    mechanism_map = config["mechanism_type_map"]
    for row in effects:
        effect = effect_map.get(row["Type"])
        mechanism = (
            4
            if row["Action"] in config["agonist_actions"]
            else mechanism_map.get(row["Type"])
        )
        smiles = chemistry.get(row["Ligand ID"])
        usable = (
            row["effect_evidence_mask"]
            and bool(smiles)
            and bool(row["Target UniProt ID"])
        )
        usable = usable and row["Ligand Type"] in config["small_molecule_types"]
        labels.append(
            {
                "ligand_id": row["Ligand ID"],
                "ligand_name": row["Ligand"],
                "uniprot_id": row["Target UniProt ID"],
                "species": row["Target Species"],
                "family": row["family"],
                "split_group": row["split_group"],
                "ligand_type": row["Ligand Type"],
                "smiles": smiles,
                "original_type": row["Type"],
                "original_action": row["Action"],
                "effect": effect,
                "effect_mask": usable and effect is not None,
                "mechanism": mechanism,
                "mechanism_mask": usable and mechanism is not None,
                "pubmed_ids": row["PubMed ID"],
                "webpage_urls": row["Webpage URLs"],
                "gating_direction_mask": False,
                "state_preference_mask": False,
            }
        )
    write_table(output_dir / "pharmacology.parquet", effects)
    write_table(output_dir / "ligands.parquet", ligands)
    write_table(output_dir / "pharmacology_labels.parquet", labels)
    write_table(output_dir / "geometry.parquet", geometries)
    write_table(output_dir / "structure_pairs.parquet", pairs)
    write_json(output_dir / "states.json", states)
    write_json(output_dir / "census.json", metadata_census)
    write_json(output_dir / "provenance.json", downloads.provenance)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "config_version": config["version"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rcsb_metadata_census": metadata_census["entry_count"],
        "curated_structures": len(states),
        "families": len(config["families"]),
        "structure_pairs": len(config["pairs"]),
        "valid_geometry_values": sum(r["valid"] for r in geometries),
        "valid_geometry_deltas": sum(r["geometry_comparison_mask"] for r in pairs),
        "single_ligand_response_labels": sum(
            r["single_ligand_response_mask"] for r in pairs
        ),
        "state_preference_labels": 0,
        "gating_direction_labels": 0,
        "pharmacology_records": len(effects),
        "pharmacology_by_family": dict(Counter(r["family"] for r in effects)),
        "pharmacology_by_species": dict(Counter(r["Target Species"] for r in effects)),
        "pharmacology_with_reference": sum(r["effect_evidence_mask"] for r in effects),
        "ligand_records": len(ligands),
        "ligands_with_valid_smiles": len(chemistry),
        "usable_effect_labels": sum(r["effect_mask"] for r in labels),
        "usable_mechanism_labels": sum(r["mechanism_mask"] for r in labels),
        "gtopdb_release": release,
        "external_accessions": config["external_accessions"],
        "limitations": [
            "Curated pilot, not an exhaustive channel or homolog audit",
            "C-alpha separation is not solvent-accessible pore radius or conductance",
            "Publication state assignments are metadata, not ligand-induced gating labels",
            "No matched single-ligand intervention or state-specific binding measurements curated",
            "No final PLM/pocket feature cache or trained RIPPLE checkpoint",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/channel_pilot.json"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/channels")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/channels"
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.raw_dir, args.output_dir, args.offline)
