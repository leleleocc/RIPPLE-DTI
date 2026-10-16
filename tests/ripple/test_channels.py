import copy
import gzip
import json

import numpy as np
import pytest
import torch

from scripts.ripple.prepare_channels import assembly_geometry, sifts_residues
from src.ripple import (
    GeometryQueries,
    RIPPLELabels,
    RIPPLEMultiState,
    RIPPLEMultiTaskLoss,
    TrainingStage,
    build_optimizer_param_groups,
    configure_training_stage,
    stage_loss_weights,
)
from src.ripple.data.channels import (
    ChannelGeometryDataset,
    ChannelStateIdentity,
    validate_state_group,
)
from src.ripple.data.schema import CHANNEL_STRUCTURES, GTOPHARM, LabelMasks
from src.ripple.structure_response import StructureResponseHead
from tests.ripple.test_modules import make_batch
from tests.ripple.test_training import make_model


def queries():
    return GeometryQueries(
        torch.tensor([[[[0, 1], [2, 3]], [[-1, 0], [8, 9]]]])
        .expand(2, -1, -1, -1)
        .clone(),
        torch.ones(2, 2, 2, dtype=torch.bool),
    )


def identities():
    return [
        [
            ChannelStateIdentity("P12345", pdb, "family:test", "ACDE")
            for pdb in ["a", "b"]
        ]
        for _ in range(2)
    ]


def test_geometry_invariance_padding_and_pair_symmetry():
    torch.manual_seed(71)
    head = StructureResponseHead(8, 0.0).eval()
    apo, delta, context = (
        torch.randn(2, 10, 8),
        torch.randn(2, 10, 8),
        torch.randn(2, 8),
    )
    coords, mask, query = (
        torch.randn(2, 10, 3),
        torch.ones(2, 10, dtype=torch.bool),
        queries(),
    )
    mask[:, 9] = False
    coords[:, 9] = float("nan")
    plain = head(apo, delta, context, coords, mask, query)
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    moved = head(
        apo,
        delta,
        context,
        coords @ rotation + 30,
        mask,
        GeometryQueries(query.indices.flip(-1), query.pair_mask),
    )
    for name in ["geometry_snapshot", "geometry_delta", "geometry_reference"]:
        assert torch.isfinite(plain[name]).all()
        torch.testing.assert_close(plain[name], moved[name], atol=1e-5, rtol=1e-5)
    assert plain["geometry_valid_mask"].tolist() == [[True, False], [True, False]]
    assert plain["geometry_delta"][:, 1].count_nonzero() == 0


def test_response_losses_masks_and_gradients_reach_propagation():
    torch.manual_seed(73)
    model, batch = make_model(), make_batch()
    batch.geometry_queries = queries()
    output = model(batch)
    labels = RIPPLELabels(
        geometry_response=torch.tensor([[1.5, float("nan")], [-0.5, 1.0]]),
        geometry_response_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    result = RIPPLEMultiTaskLoss()(output, labels)
    assert result.label_count["geometry_response"] == 2
    result.total.backward()
    assert model.propagation.delta_projection.weight.grad.abs().sum() > 0
    assert model.structure_response.readout.network[-1].weight.grad.abs().sum() > 0
    assert model.gating_head[-1].weight.grad.abs().sum() == 0
    missing = RIPPLEMultiTaskLoss()(model(batch), RIPPLELabels())
    assert missing.raw["geometry_response"].item() == 0
    assert missing.raw["gating"].item() == 0
    with pytest.raises(ValueError, match="evidence mask"):
        RIPPLEMultiTaskLoss()(output, RIPPLELabels(geometry_response=torch.ones(2, 2)))


def test_response_changes_with_ligand_without_reading_target_coordinates():
    torch.manual_seed(79)
    model, batch = make_model().eval(), make_batch()
    batch.geometry_queries = queries()
    changed = copy.deepcopy(batch)
    changed.drug_1d_tokens += torch.randn_like(changed.drug_1d_tokens) * 4
    output, other = model(batch), model(changed)
    assert not torch.allclose(
        output.predictions["geometry_delta"], other.predictions["geometry_delta"]
    )
    torch.testing.assert_close(
        output.predictions["geometry_snapshot"], other.predictions["geometry_snapshot"]
    )
    assert "target_coords" not in batch.__dataclass_fields__


def test_multistate_permutation_mask_and_identity_validation():
    torch.manual_seed(83)
    model, first = RIPPLEMultiState(make_model()).eval(), make_batch()
    second = copy.deepcopy(first)
    second.residue_coords = first.residue_coords * 2
    mask = torch.tensor([[True, True], [False, False]])
    groups = identities()
    output = model([first, second], mask, groups)
    swapped = model(
        [second, first], mask.flip(1), [list(reversed(group)) for group in groups]
    )
    torch.testing.assert_close(output.scores, swapped.scores.flip(1))
    torch.testing.assert_close(
        output.preference_probability[0].sum(), torch.tensor(1.0)
    )
    assert output.preference_probability[1].count_nonzero() == 0
    labels = RIPPLELabels(
        state_preference=torch.tensor([0, 1]),
        state_preference_mask=torch.ones(2, dtype=torch.bool),
    )
    loss = RIPPLEMultiTaskLoss()(output.states[0], labels, output)
    assert loss.label_count["state_preference"] == 1
    loss.total.backward()
    assert model.state_score.network[-1].weight.grad.abs().sum() > 0
    configure_training_stage(model, TrainingStage.CHANNEL)
    assert model.state_score.network[-1].weight.requires_grad
    assert not model.core.conformer_encoder.atom_projection.weight.requires_grad
    optimizer_groups = build_optimizer_param_groups(model, 1e-3, 0.0)
    assert {id(p) for group in optimizer_groups for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    one_state = model(
        [first, second], torch.tensor([[True, False], [False, False]]), groups
    )
    empty_loss = RIPPLEMultiTaskLoss()(one_state.states[0], labels, one_state)
    assert empty_loss.label_count["state_preference"] == 0
    assert empty_loss.raw["state_preference"].item() == 0
    changed = copy.deepcopy(second)
    changed.drug_1d_tokens += 1
    with pytest.raises(ValueError, match="share ligand"):
        model([first, changed], mask, groups)
    with pytest.raises(ValueError, match="same UniProt"):
        validate_state_group(
            [groups[0][0], ChannelStateIdentity("Q00000", "c", "family:test", "ACDE")]
        )
    with pytest.raises(ValueError, match="matched protein constructs"):
        validate_state_group(
            [groups[0][0], ChannelStateIdentity("P12345", "c", "family:test", "ACDF")]
        )


def test_new_source_permissions_and_stages():
    mask = LabelMasks(geometry_snapshot=True, geometry_response=True, gating=True)
    assert mask.allows(CHANNEL_STRUCTURES, "geometry_snapshot")
    assert not mask.allows(CHANNEL_STRUCTURES, "geometry_response")
    assert not mask.allows(GTOPHARM, "gating")
    weights = stage_loss_weights(TrainingStage.STRUCTURE)
    assert weights.geometry_snapshot > 0 and weights.geometry_response == 0


def test_gating_masks_do_not_turn_missing_evidence_into_negatives():
    torch.manual_seed(89)
    model = make_model()
    output = model(make_batch())
    labels = RIPPLELabels(
        gating=torch.tensor([0, 1]), gating_mask=torch.tensor([True, False])
    )
    loss = RIPPLEMultiTaskLoss()(output, labels)
    assert loss.label_count["gating"] == 1
    assert loss.label_count["geometry_response"] == 0
    loss.total.backward()
    assert model.gating_head[-1].weight.grad.abs().sum() > 0
    assert model.propagation.delta_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="intervention evidence"):
        RIPPLEMultiTaskLoss()(output, RIPPLELabels(gating=torch.tensor([0, 1])))


def test_no_pose_fallback_retains_chemical_and_pocket_conditioning():
    torch.manual_seed(97)
    model, batch = make_model().eval(), make_batch()
    batch.conformer_mask.zero_()
    batch.atom_mask.zero_()
    changed_pocket = copy.deepcopy(batch)
    changed_pocket.pocket_indices[0, 0] = torch.tensor([4, 5, 6, 7])
    changed_ligand = copy.deepcopy(batch)
    changed_ligand.drug_2d_tokens += 2
    context = model(batch).match.pair_context
    assert not torch.allclose(context, model(changed_pocket).match.pair_context)
    assert not torch.allclose(context, model(changed_ligand).match.pair_context)


def test_sifts_exact_author_mapping_with_insertion_code():
    xml = b"""<entry><entity><segment><listResidue><residue>
    <crossRefDb dbSource="PDB" dbChainId="A" dbResNum="10A"/>
    <crossRefDb dbSource="UniProt" dbAccessionId="P12345" dbResNum="42"/>
    </residue></listResidue></segment></entity></entry>"""
    assert sifts_residues(gzip.compress(xml), "P12345") == {("A", "10A"): 42}
    assert sifts_residues(gzip.compress(xml), "Q00000") == {}


def test_assembly_chain_remapping_keeps_homomer_copies_distinct():
    cif = """data_test
loop_
_pdbx_chain_remapping.label_asym_id
_pdbx_chain_remapping.orig_auth_asym_id
A A
A-2 A
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ILE A 1 1 ? 0 0 0 1 0 323 ILE A CA 1
ATOM 2 C CA . ILE A-2 1 1 ? 3 4 0 1 0 323 ILE A-2 CA 1
"""
    residues, metrics = assembly_geometry(
        gzip.compress(cif.encode()),
        {"pdb_id": "test", "expected_subunits": 2},
        {"rcsb_polymer_entity_container_identifiers": {"entity_id": "1"}},
        {("A", "323"): 323},
        [{"auth_seq_id": 323, "residue_name": "ILE", "name": "gate"}],
    )
    assert [r["chain_id"] for r in residues] == ["A", "A-2"]
    assert [r["uniprot_position"] for r in residues] == [323, 323]
    assert metrics[0]["valid"] and metrics[0]["value_angstrom"] == 5.0


def test_geometry_loader_uses_observed_labels_without_response_targets(tmp_path):
    (tmp_path / "structures").mkdir()
    state = {
        "pdb_id": "test",
        "uniprot_id": "P12345",
        "split_group": "family:test",
        "construct_sequence": "II",
    }
    residues = [
        {
            "coord": [i * 3, i * 4, 0],
            "residue_name": "ILE",
            "chain_id": str(i),
            "uniprot_position": 323,
            "auth_seq_id": "323",
        }
        for i in range(2)
    ]
    data = {
        "residues": residues,
        "metrics": [
            {
                "metric_id": "gate",
                "valid": True,
                "pair_indices": [[0, 1]],
                "value_angstrom": 5.0,
            }
        ],
    }
    (tmp_path / "states.json").write_text(json.dumps([state]))
    (tmp_path / "structures/test.json").write_text(json.dumps(data))
    sample = ChannelGeometryDataset(tmp_path)[0]
    distance = np.linalg.norm(sample.residue_coords[0] - sample.residue_coords[1])
    assert sample.geometry_snapshot.item() == distance == 5.0
    assert sample.queries.indices.shape == (1, 1, 1, 2)
    assert not hasattr(sample, "geometry_response")
