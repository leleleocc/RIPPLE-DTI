"""Source-preserving alignment; residue positions on disk are one-based.

The full-source outputs are staging tables, not feature caches or audited splits.
"""

import csv
import gzip
import io
import json
import math
import re
import sys
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gemmi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

VERSION = 1
DIRECT_EXTERNAL = {"Q92508", "Q9H5I1"}
ENDPOINTS = {"Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3}
COFACTORS = {
    "ATP",
    "ADP",
    "AMP",
    "GTP",
    "GDP",
    "GMP",
    "CTP",
    "CDP",
    "CMP",
    "UTP",
    "UDP",
    "UMP",
    "NAD",
    "NAP",
    "NAI",
    "NDP",
    "FAD",
    "FMN",
    "HEM",
    "HEC",
    "PLP",
    "COA",
    "SAM",
    "SAH",
}
RDLogger.DisableLog("rdApp.*")


def log(message):
    print(message, file=sys.stderr, flush=True)


def schema(**fields):
    types = {
        "s": pa.string(),
        "i": pa.int64(),
        "f": pa.float64(),
        "b": pa.bool_(),
        "ss": pa.list_(pa.string()),
        "ii": pa.list_(pa.int64()),
    }
    return pa.schema([(name, types[kind]) for name, kind in fields.items()])


class TableWriter:
    """Bound memory while preserving nullable columns with an explicit schema."""

    def __init__(self, path, table_schema, batch_size=10000):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temp = path.with_suffix(".parquet.tmp")
        self.schema = table_schema
        self.writer = pq.ParquetWriter(self.temp, table_schema, compression="zstd")
        self.rows = []
        self.batch_size = batch_size

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.flush()
        self.writer.close()
        if exc_type is None:
            self.temp.replace(self.path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def accession_list(value):
    return sorted({x for x in re.split(r"[,;\s]+", value.strip()) if x})


def parse_affinity(value, unit="nM"):
    """Keep censored observations; only exact positive concentrations are targets."""
    match = re.fullmatch(
        r"\s*(<=|>=|<|>|=|~)?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*", value
    )
    if not match:
        return None
    number = float(match[2])
    factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
    if not math.isfinite(number) or number <= 0 or unit not in factors:
        return None
    relation = match[1] or "="
    return {
        "relation": relation,
        "value_nm": number * factors[unit] * 1e9,
        "affinity_value": -math.log10(number * factors[unit])
        if relation == "="
        else None,
    }


def parse_feature_positions(text, feature, length):
    for match in re.finditer(
        r"(?:^|;\s*)" + feature + r"\s+(\d+)(?:\.\.(\d+))?(?=;|$)", text
    ):
        first, last = int(match[1]), int(match[2] or match[1])
        if 1 <= first <= last <= length:
            yield from range(first, last + 1)


def normalize_ligand(value):
    """Select the most stereochemically specified consistent representation."""
    candidates = []
    for item in value.split(";"):
        molecule = Chem.MolFromSmiles(item.strip()) if item.strip() else None
        if molecule is None:
            continue
        connectivity = Chem.MolToSmiles(molecule, isomericSmiles=False)
        stereo = sum(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            for atom in molecule.GetAtoms()
        )
        stereo += sum(
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            for bond in molecule.GetBonds()
        )
        candidates.append(
            (
                stereo,
                Chem.MolToSmiles(molecule, isomericSmiles=True),
                connectivity,
                molecule,
            )
        )
    if not candidates:
        return {"status": "no_valid_smiles", "canon_smiles": None, "inchikey": None}
    if len({x[2] for x in candidates}) != 1:
        return {
            "status": "ambiguous_connectivity",
            "canon_smiles": None,
            "inchikey": None,
        }
    _, canonical, _, molecule = max(candidates, key=lambda item: item[:2])
    max_stereo = max(x[0] for x in candidates)
    if len({x[1] for x in candidates if x[0] == max_stereo}) > 1:
        return {
            "status": "ambiguous_stereochemistry",
            "canon_smiles": None,
            "inchikey": None,
        }
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    return {
        "status": "ok" if has_carbon else "inorganic",
        "canon_smiles": canonical,
        "inchikey": Chem.MolToInchiKey(molecule),
    }


def read_uniprot(raw, out):
    proteins, counts, missing = {}, Counter(), []
    for path in sorted((raw / "uniprot").glob("features*.tsv")):
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                counts["input_rows"] += 1
                acc, sequence = row["Entry"], row["Sequence"].strip()
                if (
                    not sequence
                    or not row["Length"].isdigit()
                    or int(row["Length"]) != len(sequence)
                ):
                    missing.append(
                        {
                            "uniprot_acc": acc,
                            "source_file": path.name,
                            "reason": "missing_or_invalid_sequence",
                        }
                    )
                    continue
                if acc in proteins and proteins[acc]["sequence"] != sequence:
                    raise ValueError(f"Conflicting UniProt sequences for {acc}")
                proteins[acc] = {
                    "uniprot_acc": acc,
                    "sequence": sequence,
                    "length": len(sequence),
                    "organism": row["Organism"],
                    "taxonomy_id": row["Organism (ID)"],
                    "reviewed": row["Reviewed"],
                    "source_file": path.name,
                    "binding": row["Binding site"],
                    "active": row["Active site"],
                    "transmembrane": row["Transmembrane"],
                    "function_text": row["Function [CC]"],
                    "external_direct": acc in DIRECT_EXTERNAL,
                }
    with TableWriter(
        out / "uniprot/missing.parquet",
        schema(uniprot_acc="s", source_file="s", reason="s"),
    ) as writer:
        for row in missing:
            writer.add(row)
    (out / "uniprot/missing_accessions.txt").write_text(
        "".join(acc + "\n" for acc in sorted({x["uniprot_acc"] for x in missing}))
    )
    protein_schema = schema(
        uniprot_acc="s",
        sequence="s",
        length="i",
        organism="s",
        taxonomy_id="s",
        reviewed="s",
        source_file="s",
        external_direct="b",
        binding="s",
        active="s",
        transmembrane="s",
        function_text="s",
    )
    site_schema = schema(uniprot_acc="s", residue_index="i", site_type="s", source="s")
    with (
        TableWriter(out / "proteins.parquet", protein_schema) as writer,
        TableWriter(out / "uniprot/function_sites.parquet", site_schema) as sites,
    ):
        for acc, row in sorted(proteins.items()):
            writer.add(row)
            for key, feature, kind in (
                ("binding", "BINDING", "binding"),
                ("active", "ACT_SITE", "active"),
                ("transmembrane", "TRANSMEM", "transmembrane"),
            ):
                positions = set(
                    parse_feature_positions(row[key], feature, row["length"])
                )
                if positions:
                    counts[kind + "_proteins"] += 1
                for position in sorted(positions):
                    sites.add(
                        {
                            "uniprot_acc": acc,
                            "residue_index": position,
                            "site_type": kind,
                            "source": "uniprot",
                        }
                    )
                    counts[kind + "_residues"] += 1
    counts["valid_proteins"] = len(proteins)
    counts["missing_sequence_rows"] = len(missing)
    return proteins, dict(counts)


def read_sifts(raw, out):
    chain_map, counts = defaultdict(set), Counter()
    table_schema = schema(
        pdb_id="s",
        chain="s",
        uniprot_acc="s",
        pdb_begin="s",
        pdb_end="s",
        uniprot_begin="i",
        uniprot_end="i",
    )
    with (
        gzip.open(raw / "sifts/pdb_chain_uniprot.csv.gz", "rt") as handle,
        TableWriter(out / "sifts/segments.parquet", table_schema) as writer,
    ):
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in rows:
            pdb, chain, acc = row["PDB"].lower(), row["CHAIN"], row["SP_PRIMARY"]
            chain_map[pdb, chain].add(acc)
            writer.add(
                {
                    "pdb_id": pdb,
                    "chain": chain,
                    "uniprot_acc": acc,
                    "pdb_begin": row["PDB_BEG"],
                    "pdb_end": row["PDB_END"],
                    "uniprot_begin": int(row["SP_BEG"]),
                    "uniprot_end": int(row["SP_END"]),
                }
            )
            counts["segments"] += 1
    counts["pdb_chain_pairs"] = len(chain_map)
    counts["pdb_ids"] = len({key[0] for key in chain_map})
    return chain_map, dict(counts)


def read_pdbbind(raw, out, proteins, chain_map):
    pdb_accessions = defaultdict(set)
    for (pdb, _), accessions in chain_map.items():
        pdb_accessions[pdb].update(accessions)
    counts, records = Counter(), {}
    with tarfile.open(raw / "pdbbind/index.tar.gz") as archive:
        lines = (
            archive.extractfile("index/INDEX_general_PL.2020R1.lst")
            .read()
            .decode()
            .splitlines()
        )
    table_schema = schema(
        pdb_id="s",
        resolution="s",
        year="i",
        raw_affinity="s",
        endpoint="s",
        endpoint_type="i",
        relation="s",
        value_nm="f",
        affinity_value="f",
        accessions="ss",
        uniprot_acc="s",
        sequence_available="b",
        covalent="b",
        incomplete_ligand="b",
        ambiguous_affinity="b",
        external_direct="b",
        affinity_candidate="b",
        reference="s",
    )
    with TableWriter(out / "pdbbind/complexes.parquet", table_schema) as writer:
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            pdb, resolution, year, affinity, *tail = line.split()
            pdb = pdb.lower()
            match = re.fullmatch(
                r"(Kd|Ki|IC50|EC50)([<>=~]*)([\d.eE+-]+)([munpf]?M)", affinity
            )
            parsed = parse_affinity(match[2] + match[3], match[4]) if match else None
            endpoint = match[1] if match else None
            accs = sorted(pdb_accessions[pdb])
            acc = accs[0] if len(accs) == 1 else None
            covalent, incomplete = (
                "covalent" in line.lower(),
                "incomplete ligand" in line.lower(),
            )
            ambiguous = "~" in line or "range" in line.lower()
            row = {
                "pdb_id": pdb,
                "resolution": resolution,
                "year": int(year),
                "raw_affinity": affinity,
                "endpoint": endpoint,
                "endpoint_type": ENDPOINTS.get(endpoint, 4),
                "relation": parsed["relation"] if parsed else None,
                "value_nm": parsed["value_nm"] if parsed else None,
                "affinity_value": parsed["affinity_value"]
                if parsed and not ambiguous
                else None,
                "accessions": accs,
                "uniprot_acc": acc,
                "sequence_available": acc in proteins,
                "covalent": covalent,
                "incomplete_ligand": incomplete,
                "ambiguous_affinity": ambiguous,
                "external_direct": bool(set(accs) & DIRECT_EXTERNAL),
                "reference": " ".join(tail),
            }
            row["affinity_candidate"] = bool(
                acc in proteins
                and not row["external_direct"]
                and not covalent
                and not incomplete
                and row["affinity_value"] is not None
            )
            records[pdb] = row
            writer.add(row)
            counts["complexes"] += 1
            counts["single_uniprot"] += acc is not None
            counts["single_uniprot_with_sequence"] += acc in proteins
            counts["affinity_candidates"] += row["affinity_candidate"]
            counts["covalent"] += covalent
            counts["incomplete_ligand"] += incomplete
            counts["censored"] += bool(parsed and parsed["relation"] != "=")
            counts["endpoint_" + str(endpoint)] += 1
    return records, dict(counts)


def choose_accession(declared, mapped):
    declared, mapped = set(declared), set(mapped)
    if declared and mapped:
        overlap = declared & mapped
        if len(overlap) == 1:
            return next(iter(overlap)), "agreement"
        return None, "conflict" if not overlap else "ambiguous"
    available = declared or mapped
    if len(available) == 1:
        return next(iter(available)), "declared_only" if declared else "sifts_only"
    return None, "ambiguous" if available else "missing"


def read_biolip(raw, out, proteins, chain_map, pdbbind):
    counts, ligands = Counter(), {}
    ligand_schema = schema(
        ccd="s",
        raw_smiles="s",
        canon_smiles="s",
        inchikey="s",
        status="s",
        excluded_class="b",
    )
    with (
        gzip.open(raw / "biolip/ligand.tsv.gz", "rt") as handle,
        TableWriter(out / "biolip2/ligands.parquet", ligand_schema) as writer,
    ):
        for row in csv.DictReader(handle, delimiter="\t"):
            ccd = row["#CCD"]
            result = normalize_ligand(row["SMILES"])
            result.update(
                ccd=ccd, raw_smiles=row["SMILES"], excluded_class=ccd in COFACTORS
            )
            ligands[ccd] = result
            writer.add(result)
            counts["ligand_" + result["status"]] += 1
    table_schema = schema(
        source_id="s",
        source_row="i",
        pdb_id="s",
        chain="s",
        site_id="s",
        ccd="s",
        ligand_chain="s",
        ligand_serial="s",
        ligand_auth_seq="s",
        declared_accessions="ss",
        mapped_accessions="ss",
        uniprot_acc="s",
        identity_status="s",
        sequence_available="b",
        canon_smiles="s",
        inchikey="s",
        ligand_status="s",
        small_molecule="b",
        binding_auth="ss",
        catalytic_auth="ss",
        has_binding_annotation="b",
        has_catalytic_annotation="b",
        external_direct="b",
        metadata_candidate="b",
        in_pdbbind="b",
    )
    with (
        gzip.open(raw / "biolip/BioLiP.txt.gz", "rt") as handle,
        TableWriter(out / "biolip2/sites.parquet", table_schema) as writer,
    ):
        for number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            counts["input_sites"] += 1
            if len(row) != 21:
                raise ValueError(
                    f"BioLiP row {number}: expected 21 columns, found {len(row)}"
                )
            pdb, chain, ccd = row[0].lower(), row[1], row[4]
            declared = accession_list(row[17])
            mapped = sorted(chain_map.get((pdb, chain), set()))
            acc, status = choose_accession(declared, mapped)
            ligand = ligands.get(ccd, {})
            small = (
                ligand.get("status") == "ok"
                and ccd not in COFACTORS
                and ccd not in {"rna", "dna", "peptide"}
            )
            external = bool((set(declared) | set(mapped)) & DIRECT_EXTERNAL)
            binding = row[7].split()
            catalytic = sorted(set(row[9].replace(";", " ").split()))
            candidate = bool(small and acc in proteins and not external and binding)
            source_id = "|".join((pdb, chain, row[3], ccd, row[5], row[6], row[19]))
            record = {
                "source_id": source_id,
                "source_row": number,
                "pdb_id": pdb,
                "chain": chain,
                "site_id": row[3],
                "ccd": ccd,
                "ligand_chain": row[5],
                "ligand_serial": row[6],
                "ligand_auth_seq": row[19],
                "declared_accessions": declared,
                "mapped_accessions": mapped,
                "uniprot_acc": acc,
                "identity_status": status,
                "sequence_available": acc in proteins,
                "canon_smiles": ligand.get("canon_smiles"),
                "inchikey": ligand.get("inchikey"),
                "ligand_status": ligand.get("status", "no_ccd_record"),
                "small_molecule": small,
                "binding_auth": binding,
                "catalytic_auth": catalytic,
                "has_binding_annotation": bool(binding),
                "has_catalytic_annotation": bool(catalytic),
                "external_direct": external,
                "metadata_candidate": candidate,
                "in_pdbbind": pdb in pdbbind,
            }
            writer.add(record)
            counts["identity_" + status] += 1
            counts["small_molecule_sites"] += small
            counts["metadata_candidates"] += candidate
            counts["catalytic_candidates"] += candidate and bool(catalytic)
            counts["pdbbind_overlap_sites"] += pdb in pdbbind
            if number % 250000 == 0:
                log(f"BioLiP: {number:,} sites")
    return dict(counts)


@contextmanager
def bindingdb_rows(path):
    with (
        zipfile.ZipFile(path) as archive,
        archive.open("BindingDB_All.tsv") as binary,
        io.TextIOWrapper(binary) as text,
    ):
        reader = csv.reader(text, delimiter="\t")
        yield next(reader), reader


def bindingdb_target(row, header):
    chain_count = int(row[header["chain_count"]])
    accessions = []
    for sp, trembl in header["chain_columns"][:chain_count]:
        accessions.extend(accession_list(row[sp] or row[trembl]))
    accessions = sorted(set(accessions))
    if chain_count == 1:
        kind = "single_chain" if len(accessions) == 1 else "unresolved_single_chain"
    else:
        kind = (
            "homomer" if len(accessions) == 1 else "multichain_unresolved_or_heteromer"
        )
    return accessions, accessions[0] if kind == "single_chain" else None, kind


def read_bindingdb(raw, out, proteins):
    counts, endpoint_rows = Counter(), defaultdict(Counter)
    table_schema = schema(
        source_id="s",
        measurement_id="s",
        source_row="i",
        smiles="s",
        source_inchikey="s",
        accessions="ss",
        uniprot_acc="s",
        target_kind="s",
        sequence_available="b",
        endpoint="s",
        endpoint_type="i",
        raw_value="s",
        relation="s",
        value_nm="f",
        affinity_value="f",
        affinity_candidate="b",
        interaction="i",
        external_direct="b",
    )
    with (
        bindingdb_rows(raw / "bindingdb/BindingDB_All_202610_tsv.zip") as (
            fields,
            rows,
        ),
        TableWriter(out / "bindingdb/measurements.parquet", table_schema) as writer,
    ):
        header = {
            "chain_count": fields.index(
                "Number of Protein Chains in Target (>1 implies a multichain complex)"
            ),
            "chain_columns": [
                (
                    fields.index(f"UniProt (SwissProt) Primary ID of Target Chain {n}"),
                    fields.index(f"UniProt (TrEMBL) Primary ID of Target Chain {n}"),
                )
                for n in range(1, 51)
            ],
        }
        columns = {
            name: fields.index(name)
            for name in (
                "BindingDB Reactant_set_id",
                "Ligand SMILES",
                "Ligand InChI Key",
            )
        }
        endpoint_columns = {key: fields.index(key + " (nM)") for key in ENDPOINTS}
        for number, row in enumerate(rows, 2):
            counts["input_records"] += 1
            if len(row) != len(fields):
                raise ValueError(
                    f"BindingDB row {number}: expected {len(fields)} columns, found {len(row)}"
                )
            accessions, acc, kind = bindingdb_target(row, header)
            counts[kind] += 1
            counts["single_chain_with_sequence"] += acc in proteins
            external = bool(set(accessions) & DIRECT_EXTERNAL)
            source_id = row[columns["BindingDB Reactant_set_id"]]
            smiles = row[columns["Ligand SMILES"]].strip()
            base = {
                "source_id": source_id,
                "source_row": number,
                "smiles": smiles,
                "source_inchikey": row[columns["Ligand InChI Key"]].strip() or None,
                "accessions": accessions,
                "uniprot_acc": acc,
                "target_kind": kind,
                "sequence_available": acc in proteins,
                "external_direct": external,
                "interaction": None,
            }
            emitted = False
            for endpoint, column in endpoint_columns.items():
                value = row[column].strip()
                if not value:
                    continue
                emitted = True
                parsed = parse_affinity(value)
                candidate = bool(
                    parsed
                    and parsed["affinity_value"] is not None
                    and acc in proteins
                    and smiles
                    and not external
                )
                writer.add(
                    {
                        **base,
                        "measurement_id": source_id + ":" + endpoint,
                        "endpoint": endpoint,
                        "endpoint_type": ENDPOINTS[endpoint],
                        "raw_value": value,
                        "relation": parsed["relation"] if parsed else None,
                        "value_nm": parsed["value_nm"] if parsed else None,
                        "affinity_value": parsed["affinity_value"] if parsed else None,
                        "affinity_candidate": candidate,
                    }
                )
                endpoint_rows[endpoint]["measurements"] += 1
                endpoint_rows[endpoint]["invalid"] += parsed is None
                endpoint_rows[endpoint]["exact"] += bool(
                    parsed and parsed["relation"] == "="
                )
                endpoint_rows[endpoint]["censored"] += bool(
                    parsed and parsed["relation"] != "="
                )
                endpoint_rows[endpoint]["affinity_candidates"] += candidate
                counts["measurements"] += 1
                counts["affinity_candidates"] += candidate
            if not emitted:
                writer.add(
                    {
                        **base,
                        "measurement_id": source_id + ":unknown",
                        "endpoint": None,
                        "endpoint_type": 4,
                        "affinity_candidate": False,
                    }
                )
                counts["records_without_concentration_endpoint"] += 1
            if counts["input_records"] % 250000 == 0:
                log(f"BindingDB: {counts['input_records']:,} records")
    result = dict(counts)
    result["endpoints"] = {key: dict(value) for key, value in endpoint_rows.items()}
    return result


def parse_sifts_xml(content):
    """Read explicit cross-references, preserving insertion codes and ambiguity."""
    root = ET.fromstring(
        gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
    )
    mapping = defaultdict(set)
    for residue in root.iter():
        if residue.tag.rsplit("}", 1)[-1] != "residue":
            continue
        references = [x.attrib for x in residue if x.attrib.get("dbSource")]
        pdb_refs = [
            x
            for x in references
            if x["dbSource"] == "PDB" and x.get("dbResNum") not in {None, "null"}
        ]
        uniprot_refs = [
            x
            for x in references
            if x["dbSource"] == "UniProt" and x.get("dbResNum", "").isdigit()
        ]
        for pdb in pdb_refs:
            for uniprot in uniprot_refs:
                mapping[
                    pdb["dbChainId"], pdb["dbResNum"], uniprot["dbAccessionId"]
                ].add(int(uniprot["dbResNum"]))
    return {key: next(iter(value)) for key, value in mapping.items() if len(value) == 1}


def fetch_sifts(pdb, directory, offline=False):
    path = directory / (pdb + ".xml.gz")
    if path.exists():
        try:
            return pdb, parse_sifts_xml(path.read_bytes()), "cached", None
        except (ET.ParseError, OSError) as exc:
            return pdb, {}, "invalid_cache", str(exc)
    if offline:
        return pdb, {}, "not_cached", None
    url = f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/{pdb[1:3]}/{pdb}.xml.gz"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            content = response.read()
        mapping = parse_sifts_xml(content)
        directory.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(content)
        temp.replace(path)
        return pdb, mapping, "downloaded", None
    except (urllib.error.URLError, OSError, ET.ParseError) as exc:
        return pdb, {}, "download_failed", str(exc)


def parse_rcsb_sifts(block, entities):
    """Compose SIFTS gapless alignment blocks with the explicit PDB numbering table."""
    mapping = defaultdict(set)
    rows = block.find(
        "_pdbx_poly_seq_scheme.",
        ["entity_id", "seq_id", "pdb_strand_id", "pdb_seq_num", "pdb_ins_code"],
    )
    for row in rows:
        entity, seq_id, chains, residue, insertion = row
        if residue in {".", "?"}:
            continue
        for alignment in entities.get(entity, {}).get("rcsb_polymer_entity_align", []):
            if (
                alignment.get("provenance_source") != "SIFTS"
                or alignment.get("reference_database_name") != "UniProt"
            ):
                continue
            for region in alignment.get("aligned_regions", []):
                start, length = region["entity_beg_seq_id"], region["length"]
                if start <= int(seq_id) < start + length:
                    index = region["ref_beg_seq_id"] + int(seq_id) - start
                    author_number = residue + (
                        insertion if insertion not in {".", "?"} else ""
                    )
                    for chain in chains.split(","):
                        mapping[
                            chain,
                            author_number,
                            alignment["reference_database_accession"],
                        ].add(index)
    return {key: next(iter(value)) for key, value in mapping.items() if len(value) == 1}


def fetch_rcsb_sifts(pdb, directory, offline=False):
    """RCSB exposes SIFTS provenance even when the PDBe FTP host is unreachable."""
    directory.mkdir(parents=True, exist_ok=True)

    def cached_download(path, url):
        if path.exists():
            return path.read_bytes()
        if offline:
            raise FileNotFoundError(f"Not cached: {path.name}")
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(content)
        temp.replace(path)
        return content

    try:
        content = cached_download(
            directory / (pdb + ".cif.gz"),
            f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
        )
        block = gemmi.cif.read_string(gzip.decompress(content).decode()).sole_block()
        entity_ids = set(block.find_values("_entity_poly.entity_id"))
        entities = {}
        for entity in sorted(entity_ids):
            content = cached_download(
                directory / f"{pdb}_entity_{entity}.json",
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}",
            )
            entities[entity] = json.loads(content)
        mapping = parse_rcsb_sifts(block, entities)
        return pdb, mapping, "rcsb_sifts" if mapping else "rcsb_no_sifts_mapping", None
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
        return pdb, {}, "rcsb_failed", str(exc)


def map_residue_tokens(tokens, chain, acc, mapping, sequence):
    positions, missing = [], []
    for token in tokens:
        match = re.fullmatch(r"([A-Za-z])(-?\d+[A-Za-z]?)", token)
        position = mapping.get((chain, match[2], acc)) if match else None
        if (
            position is None
            or not 1 <= position <= len(sequence)
            or sequence[position - 1] != match[1].upper()
        ):
            missing.append(token)
        else:
            positions.append(position)
    return sorted(set(positions)), missing


def select_pilot(out, limit):
    complexes = {
        row["pdb_id"]: row
        for row in pq.read_table(out / "pdbbind/complexes.parquet").to_pylist()
    }
    pool = {}
    columns = [
        "pdb_id",
        "uniprot_acc",
        "metadata_candidate",
        "has_catalytic_annotation",
    ]
    for batch in pq.ParquetFile(out / "biolip2/sites.parquet").iter_batches(
        columns=columns
    ):
        for row in batch.to_pylist():
            pdb = row["pdb_id"]
            complex_row = complexes.get(pdb)
            if (
                row["metadata_candidate"]
                and complex_row
                and complex_row["affinity_candidate"]
                and complex_row["uniprot_acc"] == row["uniprot_acc"]
            ):
                item = pool.setdefault(
                    pdb,
                    {
                        "pdb_id": pdb,
                        "uniprot_acc": row["uniprot_acc"],
                        "catalytic": False,
                    },
                )
                item["catalytic"] |= row["has_catalytic_annotation"]
    ranked = sorted(
        pool.values(), key=lambda row: (not row["catalytic"], row["pdb_id"])
    )
    selected, selected_ids, seen = [], set(), set()
    for distinct_pass in (True, False):
        for row in ranked:
            if len(selected) >= limit:
                break
            if row["pdb_id"] in selected_ids or (
                distinct_pass and row["uniprot_acc"] in seen
            ):
                continue
            selected.append(row)
            selected_ids.add(row["pdb_id"])
            seen.add(row["uniprot_acc"])
    return selected, len(pool)


def extract_structures(raw, out, pdb_ids):
    directory = out / "pilot/structures"
    needed = {
        pdb
        for pdb in pdb_ids
        if not all(
            (directory / pdb / (pdb + suffix)).exists()
            for suffix in ("_protein.pdb", "_ligand.sdf")
        )
    }
    if not needed:
        return
    log(f"Extracting {len(needed)} selected structures from PDBbind (one archive pass)")
    with tarfile.open(raw / "pdbbind/P-L.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            pdb = name[:4].lower()
            if pdb not in needed or name not in {
                pdb + "_protein.pdb",
                pdb + "_ligand.sdf",
            }:
                continue
            target = directory / pdb / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def native_contacts(protein_path, ligand_path, cutoff):
    supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=True)
    ligand = next(
        (mol for mol in supplier if mol is not None and mol.GetNumConformers()), None
    )
    if ligand is None:
        raise ValueError("No valid ligand with coordinates")
    ligand_coords = ligand.GetConformer().GetPositions()
    structure = gemmi.read_structure(str(protein_path))
    contacts = []
    for chain in structure[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if not info.is_amino_acid():
                continue
            atoms = np.array(
                [
                    [atom.pos.x, atom.pos.y, atom.pos.z]
                    for atom in residue
                    if not atom.element.is_hydrogen
                ],
                dtype=float,
            )
            if not len(atoms):
                continue
            if (
                np.min(
                    np.sum(
                        (atoms[:, None, :] - ligand_coords[None, :, :]) ** 2, axis=-1
                    )
                )
                < cutoff**2
            ):
                contacts.append(
                    (
                        chain.name,
                        str(residue.seqid).strip(),
                        info.one_letter_code.upper(),
                    )
                )
    return (
        contacts,
        Chem.MolToSmiles(ligand, isomericSmiles=True),
        Chem.MolToInchiKey(ligand),
    )


def verify_residues(raw, out, proteins, limit, workers, offline, cutoff):
    selected, pool_size = select_pilot(out, limit)
    write_json(out / "pilot/selection.json", selected)
    mappings, download_rows = {}, []
    cache = raw / "sifts/residue_xml"
    network_offline = offline
    if selected and not offline:
        pdb, mapping, status, error = fetch_sifts(selected[0]["pdb_id"], cache)
        mappings[pdb] = mapping
        download_rows.append({"pdb_id": pdb, "status": status, "error": error})
        network_offline = status == "download_failed"
        if network_offline:
            log(
                "SIFTS endpoint unavailable; using cached XML and writing retry manifest"
            )

    def load_mapping(pdb):
        result = fetch_sifts(pdb, cache, network_offline)
        if result[1]:
            return result
        return fetch_rcsb_sifts(pdb, raw / "sifts/rcsb", offline)

    # A failed XML probe still gets the independently cached RCSB/SIFTS fallback.
    pending = [item["pdb_id"] for item in selected if not mappings.get(item["pdb_id"])]
    download_rows = [row for row in download_rows if mappings.get(row["pdb_id"])]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_mapping, pdb) for pdb in pending]
        for future in as_completed(futures):
            pdb, mapping, status, error = future.result()
            mappings[pdb] = mapping
            download_rows.append({"pdb_id": pdb, "status": status, "error": error})
    download_rows.sort(key=lambda row: row["pdb_id"])
    write_json(out / "sifts/download_status.json", download_rows)
    (out / "sifts/pending_pdb_ids.txt").write_text(
        "".join(
            row["pdb_id"] + "\n"
            for row in download_rows
            if not mappings.get(row["pdb_id"])
        )
    )
    map_schema = schema(
        pdb_id="s", chain="s", pdb_residue="s", uniprot_acc="s", residue_index="i"
    )
    with TableWriter(out / "sifts/residue_mapping.parquet", map_schema) as writer:
        for pdb, mapping in sorted(mappings.items()):
            for (chain, residue, acc), index in sorted(mapping.items()):
                writer.add(
                    {
                        "pdb_id": pdb,
                        "chain": chain,
                        "pdb_residue": residue,
                        "uniprot_acc": acc,
                        "residue_index": index,
                    }
                )
    extract_structures(raw, out, {row["pdb_id"] for row in selected})
    native_schema = schema(
        pdb_id="s",
        uniprot_acc="s",
        canon_smiles="s",
        inchikey="s",
        status="s",
        contact_count="i",
        mapped_count="i",
        residue_indices="ii",
        missing_contacts="ss",
    )
    counts = Counter(
        pool_complexes=pool_size,
        selected_complexes=len(selected),
        xml_with_mapping=sum(bool(x) for x in mappings.values()),
    )
    with TableWriter(out / "pilot/native_pockets.parquet", native_schema) as writer:
        for row in selected:
            pdb, acc = row["pdb_id"], row["uniprot_acc"]
            directory = out / "pilot/structures" / pdb
            result = {
                "pdb_id": pdb,
                "uniprot_acc": acc,
                "residue_indices": [],
                "missing_contacts": [],
            }
            try:
                contacts, smiles, inchikey = native_contacts(
                    directory / (pdb + "_protein.pdb"),
                    directory / (pdb + "_ligand.sdf"),
                    cutoff,
                )
                mapped, missing = [], []
                for chain, residue, aa in contacts:
                    positions, absent = map_residue_tokens(
                        [aa + residue],
                        chain,
                        acc,
                        mappings.get(pdb, {}),
                        proteins[acc]["sequence"],
                    )
                    mapped.extend(positions)
                    missing.extend(chain + ":" + token for token in absent)
                result.update(
                    canon_smiles=smiles,
                    inchikey=inchikey,
                    contact_count=len(contacts),
                    mapped_count=len(mapped),
                    residue_indices=sorted(set(mapped)),
                    missing_contacts=missing,
                    status="mapped"
                    if contacts and not missing
                    else "no_contacts"
                    if not contacts
                    else "mapping_incomplete",
                )
            except (OSError, ValueError, RuntimeError) as exc:
                result.update(
                    status="structure_error",
                    missing_contacts=[str(exc)],
                    contact_count=0,
                    mapped_count=0,
                )
            writer.add(result)
            counts["native_" + result["status"]] += 1
    site_schema = schema(
        source_id="s",
        pdb_id="s",
        chain="s",
        uniprot_acc="s",
        site_type="s",
        residue_indices="ii",
        unmapped_tokens="ss",
        label_available="b",
    )
    with TableWriter(out / "pilot/biolip_sites.parquet", site_schema) as writer:
        for batch in pq.ParquetFile(out / "biolip2/sites.parquet").iter_batches():
            for row in batch.to_pylist():
                pdb, acc = row["pdb_id"], row["uniprot_acc"]
                if pdb not in mappings or not row["metadata_candidate"]:
                    continue
                for key, kind in (
                    ("binding_auth", "binding"),
                    ("catalytic_auth", "catalytic"),
                ):
                    positions, missing = map_residue_tokens(
                        row[key],
                        row["chain"],
                        acc,
                        mappings[pdb],
                        proteins[acc]["sequence"],
                    )
                    available = bool(row[key]) and not missing
                    writer.add(
                        {
                            "source_id": row["source_id"],
                            "pdb_id": pdb,
                            "chain": row["chain"],
                            "uniprot_acc": acc,
                            "site_type": kind,
                            "residue_indices": positions,
                            "unmapped_tokens": missing,
                            "label_available": available,
                        }
                    )
                    counts[kind + "_sites_checked"] += 1
                    counts[kind + "_sites_mapped"] += available
    return dict(counts)


def protein_coverage(out, proteins, chain_map):
    coverage = {acc: Counter() for acc in proteins}
    for accessions in chain_map.values():
        for acc in accessions:
            if acc in coverage:
                coverage[acc]["sifts_chains"] += 1
    for name, source, flag in (
        ("bindingdb/measurements.parquet", "bindingdb", "affinity_candidate"),
        ("biolip2/sites.parquet", "biolip", "metadata_candidate"),
        ("pdbbind/complexes.parquet", "pdbbind", "affinity_candidate"),
    ):
        for batch in pq.ParquetFile(out / name).iter_batches(
            columns=["uniprot_acc", flag]
        ):
            for row in batch.to_pylist():
                acc = row["uniprot_acc"]
                if acc in coverage:
                    coverage[acc][source + "_records"] += 1
                    coverage[acc][source + "_candidates"] += row[flag]
    table_schema = schema(
        uniprot_acc="s",
        length="i",
        external_direct="b",
        bindingdb_records="i",
        bindingdb_candidates="i",
        biolip_records="i",
        biolip_candidates="i",
        pdbbind_records="i",
        pdbbind_candidates="i",
        sifts_chains="i",
        has_uniprot_binding="b",
        has_uniprot_active="b",
        has_uniprot_transmembrane="b",
    )
    counts = Counter()
    with TableWriter(out / "protein_coverage.parquet", table_schema) as writer:
        for acc, values in sorted(coverage.items()):
            row = {
                "uniprot_acc": acc,
                "length": proteins[acc]["length"],
                "external_direct": acc in DIRECT_EXTERNAL,
                **{
                    key: values[key]
                    for key in (
                        "bindingdb_records",
                        "bindingdb_candidates",
                        "biolip_records",
                        "biolip_candidates",
                        "pdbbind_records",
                        "pdbbind_candidates",
                        "sifts_chains",
                    )
                },
                "has_uniprot_binding": bool(proteins[acc]["binding"]),
                "has_uniprot_active": bool(proteins[acc]["active"]),
                "has_uniprot_transmembrane": bool(proteins[acc]["transmembrane"]),
            }
            writer.add(row)
            for source in ("bindingdb", "biolip", "pdbbind"):
                counts[source + "_proteins"] += values[source + "_candidates"] > 0
            counts["bindingdb_biolip_overlap_proteins"] += (
                values["bindingdb_candidates"] > 0 and values["biolip_candidates"] > 0
            )
            counts["three_source_overlap_proteins"] += all(
                values[source + "_candidates"] > 0
                for source in ("bindingdb", "biolip", "pdbbind")
            )
    return dict(counts)


def write_manifests(out):
    """Separate cached structures from PDB IDs and accessions still needing work."""
    directory = out / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    candidate_pdbs = set()
    for batch in pq.ParquetFile(out / "biolip2/sites.parquet").iter_batches(
        columns=["pdb_id", "metadata_candidate"]
    ):
        candidate_pdbs.update(
            row["pdb_id"] for row in batch.to_pylist() if row["metadata_candidate"]
        )
    pdbbind = {
        row["pdb_id"]
        for row in pq.read_table(
            out / "pdbbind/complexes.parquet", columns=["pdb_id"]
        ).to_pylist()
    }
    cached_mapping = {
        row["pdb_id"]
        for row in pq.read_table(
            out / "sifts/residue_mapping.parquet", columns=["pdb_id"]
        ).to_pylist()
    }
    needed_pdbs = candidate_pdbs | pdbbind
    needs_structure = candidate_pdbs - pdbbind
    coverage = pq.read_table(out / "protein_coverage.parquet").to_pylist()
    no_known_experimental = {
        row["uniprot_acc"]
        for row in coverage
        if row["bindingdb_candidates"]
        and not row["sifts_chains"]
        and not row["external_direct"]
    }
    for name, values in (
        ("all_required_sifts_pdb_ids.txt", needed_pdbs),
        ("pending_sifts_pdb_ids.txt", needed_pdbs - cached_mapping),
        ("biolip_structures_not_in_pdbbind.txt", needs_structure),
        ("alphafold_accessions_without_sifts_structure.txt", no_known_experimental),
    ):
        (directory / name).write_text("".join(value + "\n" for value in sorted(values)))
    return {
        "required_sifts_pdbs": len(needed_pdbs),
        "pending_sifts_pdbs": len(needed_pdbs - cached_mapping),
        "biolip_pdbs_not_in_local_pdbbind": len(needs_structure),
        "bindingdb_proteins_without_sifts_structure": len(no_known_experimental),
    }


def report(out, stats):
    uni, bdb, bio, pdb = (
        stats[key] for key in ("uniprot", "bindingdb", "biolip", "pdbbind")
    )
    pilot = stats["residue_verification"]
    lines = [
        "# RIPPLE-DTI Data Alignment Report",
        "",
        f"Generated: {stats['generated_at']}",
        "",
        "## Scope",
        "",
        "Full-source identity/label audit; residue verification covers only the selected pilot complexes.",
        "These are staging tables, not a training cache. Candidate counts do not imply chemical validation, complete structures, or leakage clearance.",
        "",
        "## Full-source Coverage",
        "",
        "| Source | Downloaded records | Metadata/affinity candidates |",
        "|---|---:|---:|",
        f"| UniProt | {uni['input_rows']:,} | {uni['valid_proteins']:,} valid sequences |",
        f"| BindingDB | {bdb['input_records']:,} | {bdb.get('affinity_candidates', 0):,} endpoint measurements |",
        f"| BioLiP | {bio['input_sites']:,} | {bio.get('metadata_candidates', 0):,} small-molecule sites |",
        f"| PDBbind | {pdb['complexes']:,} | {pdb.get('affinity_candidates', 0):,} complexes |",
        "",
        "BindingDB candidates require one chain, one accession, a valid sequence, nonempty SMILES, and exact concentration.",
        "BindingDB SMILES/InChIKeys remain source values; full ligand validation and canonicalization are pending.",
        "BioLiP candidates require unambiguous identity, sequence, consistent valid ligand SMILES, and binding annotations; explicit cofactors/polymers/inorganic ligands are excluded.",
        "PDBbind candidates require one accession across mapped chains, sequence, exact affinity, and no covalent/incomplete-ligand/range flag.",
        "",
        "## BindingDB Endpoints",
        "",
        "| Endpoint | Measurements | Exact | Censored | Invalid | Affinity candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        row = bdb.get("endpoints", {}).get(endpoint, {})
        lines.append(
            "| "
            + endpoint
            + " | "
            + " | ".join(
                f"{row.get(key, 0):,}"
                for key in (
                    "measurements",
                    "exact",
                    "censored",
                    "invalid",
                    "affinity_candidates",
                )
            )
            + " |"
        )
    lines += [
        "",
        "No DTI labels are invented: interaction is null until activity thresholds/censoring rules are configured.",
        "Repeated measurements are retained. Each endpoint gets its own ID; endpoint types are never pooled.",
        "",
        "## Identity and Missing Data",
        "",
        f"- UniProt missing/invalid sequence rows: {uni['missing_sequence_rows']:,}; see `uniprot/missing.parquet` and `uniprot/missing_accessions.txt`.",
        f"- BindingDB single-chain records with sequence: {bdb.get('single_chain_with_sequence', 0):,}.",
        f"- BindingDB homomer records set aside: {bdb.get('homomer', 0):,}.",
        f"- BindingDB other/unresolved multichain records set aside: {bdb.get('multichain_unresolved_or_heteromer', 0):,}.",
        f"- BioLiP declared/SIFTS identity conflicts: {bio.get('identity_conflict', 0):,}.",
        f"- BioLiP catalytic-annotation candidates: {bio.get('catalytic_candidates', 0):,}.",
        "",
    ]
    for key, count in stats["protein_overlap"].items():
        lines.append(f"- {key}: {count:,}.")
    lines += [
        "",
        "## Residue Verification",
        "",
        f"- Eligible PDBbind/BioLiP pool: {pilot.get('pool_complexes', 0):,} complexes.",
        f"- Selected: {pilot.get('selected_complexes', 0):,}; distinct proteins are prioritized deterministically.",
        f"- PDBs with explicit SIFTS residue maps (XML or RCSB SIFTS): {pilot.get('xml_with_mapping', 0):,}.",
        f"- Fully mapped contact-derived native pockets: {pilot.get('native_mapped', 0):,}.",
        f"- Fully mapped BioLiP binding sites: {pilot.get('binding_sites_mapped', 0):,} / {pilot.get('binding_sites_checked', 0):,}.",
        f"- Fully mapped BioLiP catalytic sites: {pilot.get('catalytic_sites_mapped', 0):,} / {pilot.get('catalytic_sites_checked', 0):,}.",
        "- Author residue numbers/insertion codes are preserved. Mapped labels must match the UniProt amino acid.",
        "- Unmapped/mismatched residues do not become labels. Segment endpoints are never interpolated.",
        f"- Native contacts are recalculated from heavy atoms at < {stats['contact_cutoff_angstrom']} A; they are labels only.",
        "- Download status: `sifts/download_status.json`; retry list: `sifts/pending_pdb_ids.txt`.",
        "",
        "## Retrieval Manifests",
        "",
        f"- Full-source PDB IDs requiring residue maps: {stats['retrieval']['required_sifts_pdbs']:,}; still pending: {stats['retrieval']['pending_sifts_pdbs']:,}.",
        f"- BioLiP candidate PDBs outside the downloaded PDBbind archive: {stats['retrieval']['biolip_pdbs_not_in_local_pdbbind']:,}.",
        f"- BindingDB candidate proteins with no SIFTS experimental structure: {stats['retrieval']['bindingdb_proteins_without_sifts_structure']:,}.",
        "- Sorted retrieval lists are in `manifests/`. SIFTS structure availability does not imply an apo or ligand-independent candidate-pocket input.",
        "",
        "## Still Required Before Training",
        "",
        "- Resolve or exclude missing UniProt sequences.",
        "- Complete mapping beyond the pilot and obtain remaining structures as needed.",
        "- Validate/canonicalize BindingDB ligands; check measured constructs/mutations against the chosen protein representation.",
        "- Generate candidate pockets on apo/AlphaFold structures and measure native-pocket coverage.",
        "- Compute residue embeddings/graphs and drug features; this audit does not generate language-model embeddings.",
        "- Define DTI thresholds and residue-level positive/background/unknown handling.",
        "- Exclude PIEZO homologs and audit source-wide cold splits. Only direct PIEZO1/2 accessions are excluded from candidate counts here.",
        "- Function [CC] text is preserved, never converted into residue positions. Functional annotations are supervision only.",
        "",
    ]
    (out / "DATA_REPORT.md").write_text("\n".join(lines))
    write_json(out / "alignment_summary.json", stats)


def run(args):
    raw, out = args.raw.resolve(), args.out.resolve()
    if out == raw or raw in out.parents:
        raise ValueError("--out must be outside the immutable raw directory")
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        str(path.relative_to(raw)): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(raw.rglob("*"))
        if path.is_file()
        and "residue_xml" not in path.parts
        and "rcsb" not in path.parts
    }
    state_path = out / "metadata_state.json"
    state = (
        json.loads(state_path.read_text())
        if args.resume and state_path.exists()
        else {}
    )
    if state and (state.get("version") != VERSION or state.get("inputs") != signature):
        raise ValueError("Input files/schema changed; rerun without --resume")
    stats = state.get("stats", {})
    log("Reading UniProt sequences and positional annotations")
    proteins, stats["uniprot"] = read_uniprot(raw, out)
    log("Reading SIFTS chain identities (no residue interpolation)")
    chain_map, stats["sifts"] = read_sifts(raw, out)
    log("Reading PDBbind affinity index")
    pdbbind, stats["pdbbind"] = read_pdbbind(raw, out, proteins, chain_map)
    for stage, function, arguments, artifact in (
        (
            "biolip",
            read_biolip,
            (raw, out, proteins, chain_map, pdbbind),
            "biolip2/sites.parquet",
        ),
        (
            "bindingdb",
            read_bindingdb,
            (raw, out, proteins),
            "bindingdb/measurements.parquet",
        ),
    ):
        if args.resume and stage in stats and (out / artifact).exists():
            log(f"Reusing completed {stage} metadata")
        else:
            log(f"Processing {stage}")
            stats[stage] = function(*arguments)
        write_json(
            state_path, {"version": VERSION, "inputs": signature, "stats": stats}
        )
    log("Counting cross-source protein coverage")
    stats["protein_overlap"] = protein_coverage(out, proteins, chain_map)
    log(f"Verifying up to {args.mapping_limit} PDBbind/BioLiP complexes")
    stats["residue_verification"] = verify_residues(
        raw,
        out,
        proteins,
        args.mapping_limit,
        args.workers,
        args.offline,
        args.contact_cutoff,
    )
    stats["retrieval"] = write_manifests(out)
    stats["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    stats["schema_version"] = VERSION
    stats["contact_cutoff_angstrom"] = args.contact_cutoff
    stats["config"] = {
        "cofactors_excluded": sorted(COFACTORS),
        "bindingdb_target_policy": "single_chain_only",
        "bindingdb_chemical_validation": "pending",
        "dti_thresholds": None,
        "homolog_exclusion": "pending",
        "raw_directory": str(raw),
    }
    report(out, stats)
    log(f"Report written: {out / 'DATA_REPORT.md'}")
