import csv
import gzip
import io
import zipfile

import gemmi
import pyarrow.parquet as pq
import pytest

from scripts.ripple.source_alignment import (
    bindingdb_target,
    choose_accession,
    map_residue_tokens,
    normalize_ligand,
    parse_affinity,
    parse_feature_positions,
    parse_rcsb_sifts,
    parse_sifts_xml,
    read_bindingdb,
    read_uniprot,
)


@pytest.mark.parametrize(
    ("text", "unit", "expected"), [("100", "nM", 7), ("0.1", "uM", 7), ("1e-7", "M", 7)]
)
def test_affinity_units(text, unit, expected):
    assert parse_affinity(text, unit)["affinity_value"] == pytest.approx(expected)


@pytest.mark.parametrize("text", ["<100", ">100", "<=100", ">=100", "~100"])
def test_censored_values_are_not_exact_regression_targets(text):
    parsed = parse_affinity(text)
    assert parsed["affinity_value"] is None
    assert parsed["value_nm"] == pytest.approx(100)
    assert parsed["relation"] != "="


@pytest.mark.parametrize("text", ["", "0", "-1", "nan", "10~20", "inactive"])
def test_invalid_affinity_has_no_label(text):
    assert parse_affinity(text) is None


def test_multichain_labels_are_not_assigned_to_the_first_chain():
    header = {"chain_count": 0, "chain_columns": [(1, 2), (3, 4)]}
    assert bindingdb_target(["2", "A", "", "B", ""], header) == (
        ["A", "B"],
        None,
        "multichain_unresolved_or_heteromer",
    )
    assert bindingdb_target(["2", "A", "", "A", ""], header) == (["A"], None, "homomer")
    assert bindingdb_target(["1", "", "A", "", ""], header) == (
        ["A"],
        "A",
        "single_chain",
    )


def test_conflicting_accessions_are_not_silently_reassigned():
    assert choose_accession(["A"], ["B"]) == (None, "conflict")
    assert choose_accession(["A", "B"], ["A", "B"]) == (None, "ambiguous")
    assert choose_accession([], ["A"]) == ("A", "sifts_only")


def test_smiles_variants_preserve_stereochemistry_and_reject_conflicts():
    result = normalize_ligand("CC(O)F; C[C@H](O)F")
    assert result["status"] == "ok"
    assert "@" in result["canon_smiles"]
    assert normalize_ligand("CCO; CCCC")["status"] == "ambiguous_connectivity"
    assert (
        normalize_ligand("C[C@H](O)F; C[C@@H](O)F")["status"]
        == "ambiguous_stereochemistry"
    )


def test_features_require_explicit_in_range_positions():
    text = 'BINDING 2; /ligand="ATP"; BINDING 4..6; BINDING <8; BINDING 12;'
    assert list(parse_feature_positions(text, "BINDING", 10)) == [2, 4, 5, 6]
    assert (
        list(
            parse_feature_positions(
                "Participates in catalytic activity", "ACT_SITE", 10
            )
        )
        == []
    )


def test_sifts_preserves_insertion_codes_and_drops_ambiguous_mappings():
    xml = b"""<entry xmlns="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd"><entity><segment><listResidue>
    <residue><crossRefDb dbSource="PDB" dbChainId="A" dbResNum="10A"/>
    <crossRefDb dbSource="UniProt" dbAccessionId="P1" dbResNum="2"/></residue>
    <residue><crossRefDb dbSource="PDB" dbChainId="A" dbResNum="11"/>
    <crossRefDb dbSource="UniProt" dbAccessionId="P1" dbResNum="3"/>
    <crossRefDb dbSource="UniProt" dbAccessionId="P1" dbResNum="4"/></residue>
    </listResidue></segment></entity></entry>"""
    mapping = parse_sifts_xml(gzip.compress(xml))
    assert mapping == {("A", "10A", "P1"): 2}
    assert map_residue_tokens(["C10A", "D11"], "A", "P1", mapping, "ACDE") == (
        [2],
        ["D11"],
    )
    assert map_residue_tokens(["A10A"], "A", "P1", mapping, "ACDE") == ([], ["A10A"])


def test_rcsb_uses_sifts_blocks_and_explicit_author_residues():
    block = gemmi.cif.read_string("""data_test
loop_
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
1 1 A 10 .
1 2 A 10 A
1 3 A ? .
1 4 A 15 .
""").sole_block()
    entities = {
        "1": {
            "rcsb_polymer_entity_align": [
                {
                    "provenance_source": "SIFTS",
                    "reference_database_name": "UniProt",
                    "reference_database_accession": "P1",
                    "aligned_regions": [
                        {"entity_beg_seq_id": 1, "ref_beg_seq_id": 5, "length": 2},
                        {"entity_beg_seq_id": 4, "ref_beg_seq_id": 9, "length": 1},
                    ],
                },
                {
                    "provenance_source": "other",
                    "reference_database_name": "UniProt",
                    "reference_database_accession": "P2",
                    "aligned_regions": [
                        {"entity_beg_seq_id": 1, "ref_beg_seq_id": 1, "length": 4}
                    ],
                },
            ]
        }
    }
    assert parse_rcsb_sifts(block, entities) == {
        ("A", "10", "P1"): 5,
        ("A", "10A", "P1"): 6,
        ("A", "15", "P1"): 9,
    }


def test_empty_uniprot_placeholders_become_missing_not_valid_proteins(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    (raw / "uniprot").mkdir(parents=True)
    columns = [
        "Entry",
        "Reviewed",
        "Organism",
        "Organism (ID)",
        "Length",
        "Sequence",
        "Binding site",
        "Active site",
        "Transmembrane",
        "Function [CC]",
    ]
    with (raw / "uniprot/features.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerow(
            [
                "P1",
                "reviewed",
                "Human",
                "9606",
                "4",
                "ACDE",
                "",
                "ACT_SITE 2;",
                "",
                "Catalytic protein",
            ]
        )
        writer.writerow(["P2"] + [""] * 9)
    proteins, counts = read_uniprot(raw, out)
    assert set(proteins) == {"P1"}
    assert counts["missing_sequence_rows"] == 1
    assert (out / "uniprot/missing_accessions.txt").read_text() == "P2\n"
    assert pq.read_table(out / "uniprot/function_sites.parquet").to_pylist() == [
        {
            "uniprot_acc": "P1",
            "residue_index": 2,
            "site_type": "active",
            "source": "uniprot",
        }
    ]


def test_bindingdb_keeps_endpoints_separate_missing_labels_null_and_multichain_masked(
    tmp_path,
):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    (raw / "bindingdb").mkdir(parents=True)
    fields = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI Key",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
    ]
    for endpoint in ("Kd", "Ki", "IC50", "EC50"):
        fields.append(endpoint + " (nM)")
    for number in range(1, 51):
        fields.extend(
            [
                f"UniProt (SwissProt) Primary ID of Target Chain {number}",
                f"UniProt (TrEMBL) Primary ID of Target Chain {number}",
            ]
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fields, delimiter="\t", restval="")
    writer.writeheader()
    single = {
        "BindingDB Reactant_set_id": "1",
        "Ligand SMILES": "CCO",
        fields[3]: "1",
        "UniProt (SwissProt) Primary ID of Target Chain 1": "P1",
        "Kd (nM)": "100",
        "Ki (nM)": ">1000",
    }
    writer.writerow(single)
    writer.writerow(
        {
            **single,
            "BindingDB Reactant_set_id": "2",
            fields[3]: "2",
            "UniProt (SwissProt) Primary ID of Target Chain 2": "P2",
        }
    )
    with zipfile.ZipFile(
        raw / "bindingdb/BindingDB_All_202610_tsv.zip", "w"
    ) as archive:
        archive.writestr("BindingDB_All.tsv", buffer.getvalue())
    counts = read_bindingdb(raw, out, {"P1": {}, "P2": {}})
    rows = pq.read_table(out / "bindingdb/measurements.parquet").to_pylist()
    assert counts["input_records"] == 2
    assert counts["affinity_candidates"] == 1
    assert {row["measurement_id"] for row in rows} == {"1:Kd", "1:Ki", "2:Kd", "2:Ki"}
    assert all(row["interaction"] is None for row in rows)
    assert all(not row["affinity_candidate"] for row in rows if row["source_id"] == "2")
