"""Source-preserving CPU staging; no downloads or trainable sample claims."""

import csv
import gzip
import hashlib
import re
import tarfile
from collections import defaultdict

import gemmi
from rdkit import Chem

from scripts.ripple.source_alignment import COFACTORS, choose_accession, log
from scripts.ripple.staging_chemistry import digest
from scripts.ripple.staging_common import (
    ENDPOINTS,
    TableWriter,
    concentration,
    effect_direction,
    gtop_rows,
    ids,
    packed,
    schema,
    table,
    zipped_rows,
)


def proteins(raw, out):
    records, aliases = {}, {}
    for path in sorted((raw / "uniprot").glob("features*.tsv")):
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                acc, sequence = row["Entry"], row["Sequence"].strip().upper()
                if (
                    not sequence
                    or not row["Length"].isdigit()
                    or len(sequence) != int(row["Length"])
                ):
                    raise ValueError(f"Invalid sequence for {acc}")
                taxon = int(row["Organism (ID)"])
                record = {
                    "protein_id": f"{acc}:{taxon}",
                    "accession": acc,
                    "taxon_id": taxon,
                    "isoform": acc.split("-", 1)[1] if "-" in acc else None,
                    "sequence": sequence,
                    "sequence_hash": digest(sequence),
                    "length": len(sequence),
                    "organism": row["Organism"],
                    "entry_name": row.get("Entry Name"),
                    "gene_names": row.get("Gene Names"),
                    "reviewed": row["Reviewed"],
                    "source_file": str(path.relative_to(raw)),
                    "annotations_json": packed(row),
                }
                if (
                    acc in records
                    and records[acc]["sequence_hash"] != record["sequence_hash"]
                ):
                    raise ValueError(f"Conflicting sequence for {acc}")
                records[acc] = record
                if row.get("Entry Name"):
                    aliases[row["Entry Name"]] = acc
    count = table(
        out / "identities/proteins.parquet",
        {
            "protein_id": "s",
            "accession": "s",
            "taxon_id": "i",
            "isoform": "s",
            "sequence": "s",
            "sequence_hash": "s",
            "length": "i",
            "organism": "s",
            "entry_name": "s",
            "gene_names": "s",
            "reviewed": "s",
            "source_file": "s",
            "annotations_json": "s",
        },
        (records[k] for k in sorted(records)),
    )
    return records, aliases, count


def sifts(raw, out, protein_map):
    chain_map, pdb_map = defaultdict(set), defaultdict(set)

    def rows():
        with gzip.open(raw / "sifts/pdb_chain_uniprot.csv.gz", "rt") as handle:
            for row in csv.DictReader(
                line for line in handle if not line.startswith("#")
            ):
                pdb, chain, acc = row["PDB"].lower(), row["CHAIN"], row["SP_PRIMARY"]
                chain_map[pdb, chain].add(acc)
                pdb_map[pdb].add(acc)
                yield {
                    "pdb_id": pdb,
                    "auth_chain": chain,
                    "accession": acc,
                    "protein_id": protein_map.get(acc, {}).get("protein_id"),
                    "pdb_begin": row["PDB_BEG"],
                    "pdb_end": row["PDB_END"],
                    "uniprot_begin": int(row["SP_BEG"]),
                    "uniprot_end": int(row["SP_END"]),
                    "mapping_level": "interval_only",
                    "residue_mapping_status": "pending",
                    "raw_record_json": packed(row),
                }

    count = table(
        out / "references/sifts_segments.parquet",
        {
            "pdb_id": "s",
            "auth_chain": "s",
            "accession": "s",
            "protein_id": "s",
            "pdb_begin": "s",
            "pdb_end": "s",
            "uniprot_begin": "i",
            "uniprot_end": "i",
            "mapping_level": "s",
            "residue_mapping_status": "s",
            "raw_record_json": "s",
        },
        rows(),
    )
    return chain_map, pdb_map, count


MEASUREMENT_FIELDS = {
    "record_id": "s",
    "source_id": "s",
    "source_row": "i",
    "chemical_input_id": "s",
    "accession": "s",
    "protein_id": "s",
    "accessions": "ss",
    "identity_status": "s",
    "endpoint": "s",
    "endpoint_type": "i",
    "raw_affinity": "s",
    "relation": "s",
    "value_original": "f",
    "unit_original": "s",
    "affinity_value": "f",
    "measurement_status": "s",
    "evidence_hash": "s",
}


def bindingdb(raw, out, registry, protein_map):
    root = raw / "bindingdb"
    for filename, member, dest in [
        ("BindingDB_Assays_202610_tsv.zip", "BindingDB_Assays.tsv", "assays"),
        (
            "BindingDB_rsid_eaids_202610_tsv.zip",
            "BindingDB_rsid_eaids.tsv",
            "assay_links",
        ),
    ]:
        with zipped_rows(root / filename, member) as rows:
            fields = {x: "s" for x in rows.fieldnames}
            table(out / f"sources/bindingdb/{dest}.parquet", fields, rows)
    fields = dict(
        MEASUREMENT_FIELDS,
        monomer_id="s",
        target_name="s",
        chain_count="i",
        construct_hashes="ss",
        sequence_status="s",
        reported_organism="s",
        reported_inchikey="s",
        pmid="s",
        article_doi="s",
        patent="s",
        ph="s",
        temperature="s",
        curation_source="s",
        raw_record_json="s",
    )
    count = 0
    with (
        zipped_rows(root / "BindingDB_All_202610_tsv.zip", "BindingDB_All.tsv") as rows,
        TableWriter(
            out / "sources/bindingdb/measurements.parquet", schema(**fields)
        ) as writer,
    ):
        for number, row in enumerate(rows, 1):
            if None in row:
                raise ValueError(f"BindingDB row {number}: extra fields")
            nchain = int(
                row[
                    "Number of Protein Chains in Target (>1 implies a multichain complex)"
                ]
            )
            declared, sequences = [], []
            for chain in range(1, nchain + 1):
                declared.extend(
                    ids(
                        row.get(
                            f"UniProt (SwissProt) Primary ID of Target Chain {chain}",
                            "",
                        )
                        or row.get(
                            f"UniProt (TrEMBL) Primary ID of Target Chain {chain}", ""
                        )
                    )
                )
                sequences.append(
                    row.get(f"BindingDB Target Chain Sequence {chain}", "").strip()
                )
            declared = sorted(set(declared))
            acc = declared[0] if nchain == 1 and len(declared) == 1 else None
            protein = protein_map.get(acc, {})
            source_id = row["BindingDB Reactant_set_id"]
            chemical = registry.register(
                "bindingdb", row["BindingDB MonomerID"], row["Ligand SMILES"]
            )
            evidence = packed({k: v for k, v in row.items() if v})
            base = {
                "source_id": source_id,
                "source_row": number,
                "chemical_input_id": chemical,
                "monomer_id": row["BindingDB MonomerID"],
                "accession": acc,
                "accessions": declared,
                "protein_id": protein.get("protein_id"),
                "identity_status": "resolved"
                if protein
                else "multichain_or_ambiguous"
                if not acc
                else "sequence_unavailable",
                "target_name": row["Target Name"],
                "chain_count": nchain,
                "construct_hashes": [digest(s) if s else "" for s in sequences],
                "sequence_status": "exact_canonical"
                if protein and sequences == [protein["sequence"]]
                else "construct_audit_pending",
                "reported_organism": row[
                    "Target Source Organism According to Curator or DataSource"
                ],
                "reported_inchikey": row["Ligand InChI Key"],
                "pmid": row["PMID"],
                "article_doi": row["Article DOI"],
                "patent": row["Patent Number"],
                "ph": row["pH"],
                "temperature": row["Temp (C)"],
                "curation_source": row["Curation/DataSource"],
                "raw_record_json": evidence,
                "evidence_hash": digest(evidence),
            }
            endpoints = [
                (key, row[f"{key} (nM)"])
                for key in ENDPOINTS
                if row[f"{key} (nM)"].strip()
            ] or [(None, "")]
            for endpoint, value in endpoints:
                writer.add(
                    dict(
                        base,
                        record_id=f"bindingdb:{number}:{endpoint or 'unknown'}",
                        endpoint=endpoint,
                        endpoint_type=ENDPOINTS.get(endpoint),
                        raw_affinity=value,
                        **(
                            concentration(value)
                            if endpoint
                            else {"measurement_status": "unknown_endpoint"}
                        ),
                    )
                )
                count += 1
            if number % 100000 == 0:
                registry.db.commit()
                log(f"BindingDB: {number:,} rows, {count:,} measurements")
    registry.db.commit()
    return {"input_rows": number, "measurements": count}


def biolip(raw, out, registry, protein_map, chain_map):
    chemicals = {}
    with gzip.open(raw / "biolip/ligand.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chemicals[row["#CCD"]] = registry.register(
                "biolip2", row["#CCD"], row["SMILES"]
            )

    def rows():
        with gzip.open(raw / "biolip/BioLiP.txt.gz", "rt") as handle:
            for number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(row) != 21:
                    raise ValueError(f"BioLiP row {number}: invalid column count")
                pdb, chain, ccd = row[0].lower(), row[1], row[4]
                declared, mapped = ids(row[17]), sorted(chain_map.get((pdb, chain), []))
                acc, status = choose_accession(declared, mapped)
                evidence = packed(row)
                yield {
                    "record_id": f"biolip2:{number}",
                    "source_row": number,
                    "source_id": "|".join(
                        (pdb, chain, row[3], ccd, row[5], row[6], row[19])
                    ),
                    "pdb_id": pdb,
                    "auth_chain": chain,
                    "site_id": row[3],
                    "ccd": ccd,
                    "ligand_auth_chain": row[5],
                    "ligand_auth_seq": row[19],
                    "chemical_input_id": chemicals.get(ccd),
                    "accession": acc,
                    "protein_id": protein_map.get(acc, {}).get("protein_id"),
                    "identity_status": status,
                    "declared_accessions": declared,
                    "mapped_accessions": mapped,
                    "binding_auth": row[7].split(),
                    "catalytic_auth": row[9].replace(";", " ").split(),
                    "excluded_ligand_class": ccd in COFACTORS
                    or ccd in {"rna", "dna", "peptide"},
                    "construct_hash": digest(row[20]),
                    "residue_mapping_status": "pending",
                    "evidence_hash": digest(evidence),
                    "raw_record_json": evidence,
                }

    return table(
        out / "sources/biolip2/sites.parquet",
        {
            "record_id": "s",
            "source_row": "i",
            "source_id": "s",
            "pdb_id": "s",
            "auth_chain": "s",
            "site_id": "s",
            "ccd": "s",
            "ligand_auth_chain": "s",
            "ligand_auth_seq": "s",
            "chemical_input_id": "s",
            "accession": "s",
            "protein_id": "s",
            "identity_status": "s",
            "declared_accessions": "ss",
            "mapped_accessions": "ss",
            "binding_auth": "ss",
            "catalytic_auth": "ss",
            "excluded_ligand_class": "b",
            "construct_hash": "s",
            "residue_mapping_status": "s",
            "evidence_hash": "s",
            "raw_record_json": "s",
        },
        rows(),
    )


def gtopdb(raw, out, registry, protein_map, aliases):
    root = raw / "gtopdb"
    ligand_ids = {}
    for _, row in gtop_rows(root / "ligands.csv"):
        ligand_ids[row["Ligand ID"]] = registry.register(
            "gtopharm", row["Ligand ID"], row["SMILES"]
        )
    targets = defaultdict(set)
    for _, row in gtop_rows(root / "targets.csv"):
        targets[row["Target id"]].add(row["Type"])
    pairs, records = defaultdict(set), []
    for number, row in gtop_rows(root / "interactions.csv"):
        declared = [aliases.get(acc, acc) for acc in ids(row["Target UniProt ID"])]
        acc = declared[0] if len(declared) == 1 else None
        protein = protein_map.get(acc, {})
        species_taxon = {"Human": 9606, "Rat": 10116, "Mouse": 10090}.get(
            row["Target Species"]
        )
        species_status = (
            "verified"
            if protein and species_taxon == protein["taxon_id"]
            else "mismatch"
            if species_taxon and protein
            else "pending"
        )
        effect, direction = effect_direction(row["Type"], row["Action"])
        evidence = packed(row)
        pair = f"{row['Ligand ID']}:{acc or '|'.join(declared)}"
        if effect is not None:
            pairs[pair].add(effect)
        endpoint = row["Original Affinity Units"]
        value = row["Original Affinity Median nm"]
        relation = row["Original Affinity Relation"]
        parsed = (
            concentration(relation + value)
            if endpoint in ENDPOINTS and value and relation
            else {"measurement_status": "range_or_missing_relation"}
        )
        records.append(
            dict(
                record_id=f"gtopharm:{number}",
                source_id=str(number),
                source_row=number,
                chemical_input_id=ligand_ids.get(row["Ligand ID"]),
                ligand_id=row["Ligand ID"],
                target_id=row["Target ID"],
                accession=acc,
                accessions=declared,
                protein_id=protein.get("protein_id"),
                identity_status="resolved" if protein else "ambiguous_or_unavailable",
                species_status=species_status,
                target_species=row["Target Species"],
                channel_catalogue=bool(
                    targets[row["Target ID"]] & {"vgic", "lgic", "other_ic"}
                ),
                interaction_type=row["Type"],
                action=row["Action"],
                effect=effect,
                direction_status=direction,
                pair_key=pair,
                endpoint=endpoint if endpoint in ENDPOINTS else None,
                endpoint_type=ENDPOINTS.get(endpoint),
                raw_affinity=value,
                **parsed,
                pmids=ids(row["PubMed ID"]),
                assay_description=row["Assay Description"],
                evidence_hash=digest(evidence),
                raw_record_json=evidence,
            )
        )
    for record in records:
        record["pair_direction_conflict"] = len(pairs[record["pair_key"]]) > 1
        record["effect_candidate"] = bool(
            record["effect"] is not None
            and not record["pair_direction_conflict"]
            and record["protein_id"]
            and record["species_status"] == "verified"
            and record["channel_catalogue"]
        )
    return table(
        out / "sources/gtopharm/interactions.parquet",
        dict(
            MEASUREMENT_FIELDS,
            ligand_id="s",
            target_id="s",
            species_status="s",
            target_species="s",
            channel_catalogue="b",
            interaction_type="s",
            action="s",
            effect="i",
            direction_status="s",
            pair_key="s",
            pair_direction_conflict="b",
            effect_candidate="b",
            pmids="ss",
            assay_description="s",
            raw_record_json="s",
        ),
        records,
    )


def pdbbind(raw, out, registry, protein_map, pdb_map):
    ligands, members = {}, defaultdict(list)
    with tarfile.open(raw / "pdbbind/P-L.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            pdb = member.name.split("/")[-2].lower()
            members[pdb].append(member.name)
            if not member.name.endswith("_ligand.sdf"):
                continue
            block = archive.extractfile(member).read().decode("utf-8", errors="replace")
            mol = Chem.MolFromMolBlock(
                block.split("$$$$")[0],
                sanitize=True,
                removeHs=False,
                strictParsing=False,
            )
            smiles = (
                Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
                if mol is not None
                else ""
            )
            ligands[pdb] = {
                "chemical_input_id": registry.register("pdbbind", pdb, smiles),
                "ligand_parse_status": "ok" if mol is not None else "invalid_sdf",
                "ligand_member": member.name,
                "ligand_content_hash": digest(block),
            }
    with tarfile.open(raw / "pdbbind/index.tar.gz") as archive:
        lines = (
            archive.extractfile("index/INDEX_general_PL.2020R1.lst")
            .read()
            .decode()
            .splitlines()
        )

    def rows():
        for number, line in enumerate(lines, 1):
            if not line.strip() or line.startswith("#"):
                continue
            pdb, resolution, year, affinity, *tail = line.split()
            pdb = pdb.lower()
            match = re.fullmatch(
                r"(Kd|Ki|IC50|EC50)([<>=~]*)([\d.eE+-]+)([munpf]?M)", affinity
            )
            endpoint = match[1] if match else None
            parsed = (
                concentration(match[2] + match[3], match[4])
                if match
                else {"measurement_status": "unparsed"}
            )
            ambiguous = "~" in line or "range" in line.lower()
            if ambiguous:
                parsed.update(affinity_value=None, measurement_status="ambiguous")
            accessions = sorted(pdb_map.get(pdb, []))
            acc = accessions[0] if len(accessions) == 1 else None
            yield dict(
                record_id=f"pdbbind:{pdb}",
                source_id=pdb,
                source_row=number,
                pdb_id=pdb,
                resolution=resolution,
                year=int(year),
                accession=acc,
                accessions=accessions,
                protein_id=protein_map.get(acc, {}).get("protein_id"),
                identity_status="unique_pdb_candidate"
                if acc
                else "ambiguous_or_unmapped",
                endpoint=endpoint,
                endpoint_type=ENDPOINTS.get(endpoint),
                raw_affinity=affinity,
                **parsed,
                covalent="covalent" in line.lower(),
                incomplete_ligand="incomplete ligand" in line.lower(),
                reference=" ".join(tail),
                raw_record_json=packed(line),
                evidence_hash=digest(line),
                **ligands.get(pdb, {"ligand_parse_status": "sdf_absent"}),
                archive_members=sorted(members.get(pdb, [])),
                structure_audit_status="pending",
            )

    return table(
        out / "sources/pdbbind/complexes.parquet",
        dict(
            MEASUREMENT_FIELDS,
            pdb_id="s",
            resolution="s",
            year="i",
            covalent="b",
            incomplete_ligand="b",
            reference="s",
            raw_record_json="s",
            ligand_parse_status="s",
            ligand_member="s",
            ligand_content_hash="s",
            archive_members="ss",
            structure_audit_status="s",
        ),
        rows(),
    )


def cif_category(block, name):
    category = block.get_mmcif_category(name)
    return (
        [dict(zip(category, values)) for values in zip(*category.values())]
        if category
        else []
    )


def structure_one(path):
    pdb = path.name.split(".")[0].lower()
    with gzip.open(path, "rb") as handle:
        content = handle.read()
    block = gemmi.cif.read_string(content.decode()).sole_block()
    assemblies = cif_category(block, "_pdbx_struct_assembly.")
    generators = cif_category(block, "_pdbx_struct_assembly_gen.")
    operators = cif_category(block, "_pdbx_struct_oper_list.")
    entities = {x["id"]: x for x in cif_category(block, "_entity.")}
    chains = []
    for row in cif_category(block, "_entity_poly."):
        sequence = re.sub(r"\s+", "", row.get("pdbx_seq_one_letter_code_can") or "")
        chains.append(
            {
                "pdb_id": pdb,
                "entity_id": row["entity_id"],
                "polymer_type": row.get("type"),
                "auth_chains": (row.get("pdbx_strand_id") or "").split(","),
                "construct_sequence": sequence,
                "construct_hash": digest(sequence),
                "raw_entity_json": packed(entities.get(row["entity_id"], {})),
                "mapping_status": "pending",
                "assembly_status": "definition_only",
            }
        )
    record = {
        "pdb_id": pdb,
        "source": "rcsb",
        "raw_file": f"rcsb/cif/{path.name}",
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "coordinate_scope": "asymmetric_unit",
        "assembly_ids": [x["id"] for x in assemblies],
        "assembly_definitions_json": packed(assemblies),
        "assembly_generators_json": packed(generators),
        "assembly_operators_json": packed(operators),
        "entity_sources_json": packed(
            {
                name: cif_category(block, name)
                for name in [
                    "_entity_src_gen.",
                    "_entity_src_nat.",
                    "_pdbx_entity_src_syn.",
                ]
            }
        ),
        "audit_status": "pending",
        "trainable": False,
    }
    return record, chains


def structures(raw, out, pool):
    count = 0
    with (
        TableWriter(
            out / "references/structures.parquet",
            schema(
                pdb_id="s",
                source="s",
                raw_file="s",
                content_sha256="s",
                coordinate_scope="s",
                assembly_ids="ss",
                assembly_definitions_json="s",
                assembly_generators_json="s",
                assembly_operators_json="s",
                entity_sources_json="s",
                audit_status="s",
                trainable="b",
            ),
        ) as writer,
        TableWriter(
            out / "references/structure_entities.parquet",
            schema(
                pdb_id="s",
                entity_id="s",
                polymer_type="s",
                auth_chains="ss",
                construct_sequence="s",
                construct_hash="s",
                raw_entity_json="s",
                mapping_status="s",
                assembly_status="s",
            ),
        ) as entities,
    ):
        for record, chains in pool.map(
            structure_one, sorted((raw / "rcsb/cif").glob("*.cif.gz")), chunksize=8
        ):
            writer.add(record)
            for chain in chains:
                entities.add(chain)
            count += 1
            if count % 500 == 0:
                log(f"RCSB: {count:,} structures indexed")
    return count
