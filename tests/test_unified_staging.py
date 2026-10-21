import csv
import gzip
import io
import json
import tarfile
import zipfile
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from scripts.ripple.prepare_unified import run
from scripts.ripple.staging_chemistry import standardize
from scripts.ripple.staging_common import (
    concentration,
    effect_direction,
    raw_snapshot,
    validate_paths,
)


def test_chemistry_salts_stereo_and_mixtures():
    def normalized(value):
        return standardize(("id", value))[1]

    salt = normalized("C[C@H](O)C(=O)[O-].[Na+]")
    acid = normalized("C[C@H](O)C(=O)O")
    assert salt["inchikey"] == acid["inchikey"]
    assert salt["fragments_removed"] == 1
    assert "@" in salt["canonical_smiles"]
    assert acid["inchikey"] != normalized("C[C@@H](O)C(=O)O")["inchikey"]
    assert normalized("CCO.CCCC")["status"] == "multiple_organic_fragments"
    assert normalized("CCO.[Cu+2]")["status"] == "unsupported_inorganic_fragment"
    assert (
        normalized("C[C@H](O)C(O)F;C[C@@H](O)[C@H](O)F")["status"]
        == "conflicting_stereochemistry"
    )
    assert normalized("CC(O)F;C[C@H](O)F")["status"] == "ok"
    assert (
        normalized("C[C@@H](O)F;C[C@H](O)F")["status"] == "conflicting_stereochemistry"
    )


def test_measurements_and_direction_require_evidence():
    assert concentration("100")["affinity_value"] == pytest.approx(7)
    assert concentration("<100")["affinity_value"] is None
    assert concentration("1e-300", "fM")["affinity_value"] == pytest.approx(315)
    assert "affinity_value" not in concentration("0")
    assert effect_direction("Agonist", "Inverse agonist") == (
        None,
        "type_action_conflict",
    )
    assert effect_direction("Inhibitor", "Binding")[0] is None
    assert effect_direction("Activator", "None")[0] is None
    assert effect_direction("Allosteric modulator", "Positive")[0] == 0


def test_output_cannot_alias_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError):
        validate_paths(raw, raw / "processed")
    alias = tmp_path / "alias"
    alias.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_paths(raw, alias / "processed")
    output = tmp_path / "output"
    output.mkdir()
    (output / "identities").symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_paths(raw, output)


def csv_text(rows, delimiter="\t"):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def test_pipeline_preserves_raw_and_can_resume(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    for name in [
        "uniprot",
        "sifts",
        "gtopdb",
        "biolip",
        "pdbbind",
        "bindingdb",
        "rcsb/cif",
    ]:
        (raw / name).mkdir(parents=True)
    (raw / "uniprot/features_00000.tsv").write_text(
        csv_text(
            [
                {
                    "Entry": "P1",
                    "Sequence": "ACDE",
                    "Length": "4",
                    "Organism": "Human",
                    "Organism (ID)": "9606",
                    "Reviewed": "reviewed",
                    "Entry Name": "P1_HUMAN",
                }
            ]
        )
    )
    with gzip.open(raw / "sifts/pdb_chain_uniprot.csv.gz", "wt") as handle:
        handle.write(
            csv_text(
                [
                    {
                        "PDB": "1abc",
                        "CHAIN": "A",
                        "SP_PRIMARY": "P1",
                        "PDB_BEG": "10A",
                        "PDB_END": "13",
                        "SP_BEG": "1",
                        "SP_END": "4",
                    }
                ],
                delimiter=",",
            )
        )

    def gtop(name, rows):
        (raw / "gtopdb" / name).write_text(
            '"# release test"\n' + csv_text(rows, delimiter=",")
        )

    gtop("ligands.csv", [{"Ligand ID": "1", "SMILES": "CCO"}])
    gtop("targets.csv", [{"Target id": "1", "Type": "vgic"}])
    gtop(
        "interactions.csv",
        [
            {
                "Target UniProt ID": "P1",
                "Target Species": "Human",
                "Type": "Activator",
                "Action": "Activation",
                "Ligand ID": "1",
                "Target ID": "1",
                "Original Affinity Units": "EC50",
                "Original Affinity Median nm": "100",
                "Original Affinity Relation": "=",
                "PubMed ID": "123",
                "Assay Description": "test",
            }
        ],
    )
    with gzip.open(raw / "biolip/ligand.tsv.gz", "wt") as handle:
        handle.write(csv_text([{"#CCD": "ETL", "SMILES": "CCO"}]))
    site = [""] * 21
    for index, value in {
        0: "1abc",
        1: "A",
        3: "BS1",
        4: "ETL",
        5: "B",
        6: "1",
        7: "A10A",
        17: "P1",
        19: "1",
        20: "ACDE",
    }.items():
        site[index] = value
    with gzip.open(raw / "biolip/BioLiP.txt.gz", "wt") as handle:
        handle.write("\t".join(site) + "\n")

    def archive(name, members):
        with tarfile.open(raw / "pdbbind" / name, "w:gz") as handle:
            for member, text in members.items():
                data = text.encode()
                info = tarfile.TarInfo(member)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))

    archive(
        "index.tar.gz",
        {"index/INDEX_general_PL.2020R1.lst": "1abc 2.0 2020 Kd=100nM // reference\n"},
    )
    archive(
        "P-L.tar.gz",
        {
            "P-L/2020/1abc/1abc_ligand.sdf": Chem.MolToMolBlock(
                Chem.MolFromSmiles("CCO")
            )
            + "$$$$\n"
        },
    )
    row = {
        key: ""
        for key in [
            "BindingDB Reactant_set_id",
            "BindingDB MonomerID",
            "Ligand SMILES",
            "Ligand InChI Key",
            "Target Name",
            "Target Source Organism According to Curator or DataSource",
            "Number of Protein Chains in Target (>1 implies a multichain complex)",
            "BindingDB Target Chain Sequence 1",
            "UniProt (SwissProt) Primary ID of Target Chain 1",
            "PMID",
            "Article DOI",
            "Patent Number",
            "pH",
            "Temp (C)",
            "Curation/DataSource",
            "Kd (nM)",
            "Ki (nM)",
            "IC50 (nM)",
            "EC50 (nM)",
        ]
    }
    row.update(
        {
            "BindingDB Reactant_set_id": "1",
            "BindingDB MonomerID": "1",
            "Ligand SMILES": "CCO",
            "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
            "UniProt (SwissProt) Primary ID of Target Chain 1": "P1",
            "BindingDB Target Chain Sequence 1": "ACDE",
            "Kd (nM)": "<100",
        }
    )
    for filename, member, rows in [
        ("All", "BindingDB_All.tsv", [row]),
        (
            "Assays",
            "BindingDB_Assays.tsv",
            [{"ENTRYID": "1", "ASSAYID": "1", "DESCRIPTION": "test"}],
        ),
        (
            "rsid_eaids",
            "BindingDB_rsid_eaids.tsv",
            [{"REACTANT_SET_ID": "1", "ENTRYID_ASSAYID": "1_1"}],
        ),
    ]:
        with zipfile.ZipFile(
            raw / f"bindingdb/BindingDB_{filename}_202610_tsv.zip", "w"
        ) as handle:
            handle.writestr(member, csv_text(rows))
    before = raw_snapshot(raw)
    args = SimpleNamespace(raw=raw, out=out, workers=1, resume=False)
    run(args)
    assert raw_snapshot(raw) == before
    report = json.loads((out / "report.json").read_text())
    assert report["raw_integrity_verified"]
    assert report["chemistry"]["unique_molecules"] == 1
    result = pq.read_table(out / "sources/bindingdb/measurements.parquet").to_pylist()[
        0
    ]
    assert result["affinity_value"] is None
    assert result["molecule_id"]
    assert result["trainable"] is False
    assert result["split"] is None
    assert json.loads(result["raw_record_json"])["Kd (nM)"] == "<100"
    args.resume = True
    run(args)
    after = json.loads((out / "report.json").read_text())
    assert report["tables"] == after["tables"]
    assert raw_snapshot(raw) == before
