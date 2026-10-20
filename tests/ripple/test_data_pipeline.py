import copy
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.ripple.data.dataset import (
    DRUG_FIELDS,
    POCKET_FIELDS,
    PROTEIN_FIELDS,
    RIPPLEDataset,
    RIPPLESample,
    collate_ripple,
)
from src.ripple.data.geometry import sample_geometry_queries
from src.ripple.data.manifest import (
    ChainReference,
    SampleRecord,
    StructureReference,
    validate_partition,
)
from src.ripple.data.schema import BINDINGDB, CHANNEL_STRUCTURES, GTOPHARM, LabelMasks
from src.ripple.evaluate import evaluate
from src.ripple.model import GeometryBatch
from src.ripple.runner import normalized_loss, task_schedule
from src.ripple.structure_response import GeometryQueries
from src.ripple.train import CURRICULUM, parse_args, run_stage
from src.ripple.training import (
    RIPPLELabels,
    RIPPLEMultiTaskLoss,
    TrainingStage,
    build_optimizer_param_groups,
    configure_training_stage,
    stage_loss_weights,
)
from tests.ripple.test_modules import make_batch
from tests.ripple.test_training import make_model


def single_batch(index=0, length=10):
    original = make_batch()
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            values[name] = value[index : index + 1].clone()
    for name in (
        "residue_features",
        "residue_coords",
        "residue_mask",
        "protein_sequence_tokens",
        "function_mask",
    ):
        values[name] = values[name][:, :length]
    edges = (values["edge_index"] < length).all(1)[0]
    values["edge_index"] = values["edge_index"][:, :, edges]
    values["edge_mask"] = values["edge_mask"][:, edges]
    values["edge_features"] = values["edge_features"][:, edges]
    values["pocket_mask"] &= values["pocket_indices"] < length
    return type(original)(**values)


def geometry_batch(length=10):
    batch = single_batch(length=length)
    values = {name: getattr(batch, name) for name in PROTEIN_FIELDS}
    values["geometry_queries"] = GeometryQueries(
        torch.tensor([[[[0, 1]], [[2, 3]]]]),
        torch.ones(1, 2, 1, dtype=torch.bool),
    )
    return GeometryBatch(**values)


def test_geometry_only_forward_backward_never_runs_drug_or_propagation():
    model, batch = make_model(), geometry_batch()
    batch.residue_mask[:, -1] = False
    batch.residue_coords[:, -1] = float("nan")
    labels = RIPPLELabels(
        geometry_snapshot=torch.tensor([[4.0, 5.0]]),
        geometry_snapshot_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with (
        patch.object(
            model.conformer_encoder,
            "forward",
            side_effect=AssertionError("drug called"),
        ),
        patch.object(
            model.propagation,
            "forward",
            side_effect=AssertionError("propagation called"),
        ),
        patch.object(
            model.structure_response.readout,
            "forward",
            side_effect=AssertionError("response called"),
        ),
    ):
        output = model(batch)
        result = RIPPLEMultiTaskLoss(
            stage_loss_weights(TrainingStage.CHANNEL_FINETUNE)
        )(output, labels)
        assert torch.isfinite(result.total)
        result.total.backward()
    assert model.residue_encoder.residue_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.drug_fusion.parameters())
    assert all(parameter.grad is None for parameter in model.propagation.parameters())


def test_unknown_endpoint_has_separate_embedding_and_no_regression_gradient():
    model, batch = make_model(), make_batch()
    batch.endpoint_type = torch.tensor([3, 4])
    output = model(batch)
    assert output.predictions["affinity_valid_mask"].tolist() == [True, False]
    labels = RIPPLELabels(
        affinity=torch.tensor([7.0, 99.0]),
        affinity_mask=torch.ones(2, dtype=torch.bool),
    )
    result = RIPPLEMultiTaskLoss(stage_loss_weights(TrainingStage.FOUNDATION))(
        output, labels
    )
    result.total.backward()
    assert result.label_count["affinity"] == 1
    gradient = model.task_heads.endpoint_embedding.weight.grad
    assert gradient[0].count_nonzero() == 0
    assert gradient[4].abs().sum() > 0
    batch.endpoint_type = torch.tensor([-1, 100])
    other = RIPPLEMultiTaskLoss()(model(batch), labels)
    assert other.label_count["affinity"] == 0


def test_variable_length_collation_matches_independent_forward_and_source_masks():
    torch.manual_seed(101)
    first, second = single_batch(length=6), single_batch(index=1)
    samples = [
        RIPPLESample(
            first,
            RIPPLELabels(affinity=torch.tensor([7.0]), effect=torch.tensor([0])),
            BINDINGDB,
            LabelMasks(affinity=True, effect=True),
        ),
        RIPPLESample(
            second,
            RIPPLELabels(effect=torch.tensor([1])),
            GTOPHARM,
            LabelMasks(effect=True),
        ),
    ]
    batch, labels = collate_ripple(samples)
    assert labels.effect_mask.tolist() == [False, True]
    assert labels.affinity_mask.tolist() == [True, False]
    assert torch.isnan(labels.affinity[1])
    assert labels.mechanism is None
    assert batch.function_mask is None
    torch.testing.assert_close(batch.edge_index[1], second.edge_index[0])
    model = make_model().eval()
    actual = model(batch)
    for index, sample in enumerate(samples):
        expected = model(sample.batch)
        torch.testing.assert_close(
            actual.predictions["effect_logits"][index],
            expected.predictions["effect_logits"][0],
            atol=2e-6,
            rtol=2e-5,
        )
    result = RIPPLEMultiTaskLoss()(actual, labels)
    assert torch.isfinite(result.total)


def test_geometry_collation_keeps_query_indices_local_and_missing_labels_masked():
    first, second = geometry_batch(6), geometry_batch(10)
    first.geometry_queries.indices = first.geometry_queries.indices[:, :1]
    first.geometry_queries.pair_mask = first.geometry_queries.pair_mask[:, :1]
    samples = [
        RIPPLESample(
            first,
            RIPPLELabels(geometry_snapshot=torch.tensor([[4.0]])),
            CHANNEL_STRUCTURES,
            LabelMasks(geometry_snapshot=True),
        ),
        RIPPLESample(second, RIPPLELabels(), CHANNEL_STRUCTURES, LabelMasks()),
    ]
    batch, labels = collate_ripple(samples)
    assert batch.geometry_queries.indices[1, 0, 0].tolist() == [0, 1]
    assert labels.geometry_snapshot.shape == (2, 2)
    assert labels.geometry_snapshot_mask.tolist() == [[True, False], [False, False]]
    loss = RIPPLEMultiTaskLoss()(make_model()(batch), labels)
    assert loss.label_count["geometry_snapshot"] == 1
    with pytest.raises(ValueError, match="separate task"):
        collate_ripple(
            [
                samples[0],
                RIPPLESample(single_batch(), RIPPLELabels(), GTOPHARM, LabelMasks()),
            ]
        )


def test_stage_warmup_gradients_and_first_release_disabled_tasks():
    model = make_model()
    configure_training_stage(model, TrainingStage.EFFECT_WARMUP)
    result = RIPPLEMultiTaskLoss(stage_loss_weights(TrainingStage.EFFECT_WARMUP))(
        model(make_batch()),
        RIPPLELabels(effect=torch.tensor([0, 1])),
    )
    result.total.backward()
    for module in (
        model.source_fusion,
        model.propagation,
        model.task_heads.effect_head,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(p.grad is None for p in model.residue_encoder.parameters())
    for stage in CURRICULUM:
        weights = stage_loss_weights(stage)
        for name in (
            "mechanism",
            "function",
            "pose_rank",
            "allosteric_pocket",
            "geometry_response",
            "gating",
            "state_preference",
            "nonbinder_delta",
            "interaction",
        ):
            assert getattr(weights, name) == 0
    configure_training_stage(model, TrainingStage.CHANNEL_FINETUNE)
    groups = build_optimizer_param_groups(
        model, 1e-4, 1e-4, stage=TrainingStage.CHANNEL_FINETUNE
    )
    rates = {id(p): group["lr"] for group in groups for p in group["params"]}
    assert rates[id(model.source_fusion.global_projection[1].weight)] == pytest.approx(
        2e-5
    )
    assert rates[id(model.task_heads.effect_head.network[-1].weight)] == pytest.approx(
        1e-4
    )


def test_task_schedule_and_replay_normalization():
    schedule = task_schedule({"effect": 6, "geometry": 2, "replay": 2}, "effect", 6, 42)
    assert [schedule.count(task) for task in ("effect", "geometry", "replay")] == [
        6,
        2,
        2,
    ]
    assert schedule == task_schedule(
        {"effect": 6, "geometry": 2, "replay": 2}, "effect", 6, 42
    )
    weights = stage_loss_weights(TrainingStage.FOUNDATION)
    result = RIPPLEMultiTaskLoss(weights)(
        make_model()(make_batch()),
        RIPPLELabels(
            affinity=torch.tensor([7.0, 6.0]),
            native_pocket_index=torch.tensor([0, 0]),
        ),
    )
    expected = 0.2 * (result.raw["affinity"] + result.raw["native_pocket"]) / 2
    torch.testing.assert_close(
        normalized_loss(result, weights, "replay").float(), expected
    )


def test_geometry_queries_are_reproducible_and_preserve_chain_copies():
    coords = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    valid = torch.tensor([True, True, False, True, True, True])
    chains = ["A", "A", "A", "B", "B", "B"]
    edges = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 5]])
    query, target = sample_geometry_queries(
        coords, valid, chains, edges, split="train", seed=7
    )
    other, other_target = sample_geometry_queries(
        coords, valid, chains, edges, split="train", seed=7
    )
    torch.testing.assert_close(query.indices, other.indices)
    torch.testing.assert_close(target, other_target)
    pairs = query.indices[0, :, 0]
    assert len(pairs) <= 64
    assert not (pairs == 2).any()
    assert any(chains[a] != chains[b] for a, b in pairs.tolist())
    torch.testing.assert_close(
        target[0], (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
    )


@pytest.fixture
def cache_fixture(tmp_path):
    torch.manual_seed(7)
    records = []
    drug_batch = single_batch()
    torch.save(
        {
            "cache_version": 1,
            "inchikey": "test-drug",
            "feature_version": "test-v1",
            "standardization_version": "test-v1",
            "tensors": {name: getattr(drug_batch, name)[0] for name in DRUG_FIELDS},
        },
        tmp_path / "drug.pt",
    )
    for split in ("train", "val", "test", "external"):
        batch = single_batch()
        reference = StructureReference(
            "test",
            "1",
            f"structure-{split}",
            "mapping-v1",
            (
                ChainReference(
                    "A", f"P-{split}", 9606, f"seq-{split}", 10, f"construct-{split}"
                ),
            ),
        )
        torch.save(
            {
                "cache_version": 1,
                "structure_hash": reference.structure_hash,
                "mapping_version": "mapping-v1",
                "chains": [asdict(chain) for chain in reference.chains],
                "feature_version": "test-v1",
                "residue_mapping": [
                    {
                        "auth_asym_id": "A",
                        "auth_seq_id": str(index + 101),
                        "insertion_code": "",
                        "label_asym_id": "A",
                        "label_seq_id": index + 1,
                    }
                    for index in range(10)
                ],
                "residue_keys": [["A", index] for index in range(10)],
                "tensors": {
                    name: getattr(batch, name)[0]
                    for name in PROTEIN_FIELDS | POCKET_FIELDS
                    if getattr(batch, name) is not None
                },
            },
            tmp_path / f"protein-{split}.pt",
        )
        for task, source, labels, masks in (
            ("measurement", "bindingdb", {"affinity": 7.0}, {"affinity": True}),
            ("site", "biolip2", {"native_pocket_index": 0}, {"pocket": True}),
            (
                "complex",
                "pdbbind",
                {"affinity": 6.0, "native_pocket_index": 0},
                {"affinity": True, "pocket": True},
            ),
            ("effect", "gtopharm", {"effect": 0}, {"effect": True}),
            ("effect", "gtopharm", {"effect": 1}, {"effect": True}),
            (
                "geometry",
                "channel_structures",
                {"geometry_snapshot": [3.0, 4.0]},
                {"geometry_snapshot": True},
            ),
        ):
            record = SampleRecord(
                sample_id=f"{split}-{len(records)}",
                source=source,
                task=task,
                split=split,
                split_group=f"group-{split}",
                protein_cache=f"protein-{split}.pt",
                structure=reference,
                evidence={
                    "source_version": "synthetic-test",
                    "record_ids": [str(len(records))],
                },
                drug_cache=None if task == "geometry" else "drug.pt",
                inchikey=None if task == "geometry" else "test-drug",
                endpoint_type=0 if task in {"measurement", "complex"} else None,
                labels=labels,
                loss_enabled=masks,
                geometry_queries={
                    "indices": [[[0, 1]], [[2, 3]]],
                    "pair_mask": [[True], [True]],
                }
                if task == "geometry"
                else None,
            )
            records.append(record)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in records)
    )
    config = {
        "RIPPLE": {
            "ATOM_INPUT_DIM": 8,
            "RESIDUE_INPUT_DIM": 12,
            "DIM": 16,
            "EDGE_DIM": 3,
            "CONFORMER_LAYERS": 1,
            "RESIDUE_LAYERS": 1,
            "PROPAGATION_STEPS": 2,
            "BIINTENTION_HEADS": 4,
            "DRUG_LATENTS": 8,
            "PROTEIN_LATENTS": 8,
            "DROPOUT": 0.0,
        },
        "TRAINING": {
            "FOUNDATION_VALIDATE_EVERY_STEPS": 1,
            "FOUNDATION_TASK_RATIO": {"measurement": 1, "site": 1, "complex": 1},
            "FINETUNE_TASK_RATIO": {"effect": 1, "geometry": 1, "replay": 1},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, records


def test_manifest_leakage_and_cache_identity_checks(cache_fixture):
    directory, records = cache_fixture
    dataset = RIPPLEDataset(directory, "train", {"effect"})
    assert len(dataset) == 2
    assert dataset[0].record.structure.chains[0].canonical_length == 10
    leaked = replace(records[0], sample_id="leaked", split="test")
    with pytest.raises(ValueError, match="Split leakage"):
        validate_partition([records[0], leaked])
    leaked = replace(leaked, split_group="another-group")
    with pytest.raises(ValueError, match="Protein identity"):
        validate_partition([records[0], leaked])
    bad = copy.deepcopy(torch.load(directory / "protein-train.pt", weights_only=True))
    bad["residue_keys"][0] = ["B", 0]
    torch.save(bad, directory / "protein-train.pt")
    with pytest.raises(ValueError, match="chain in canonical"):
        RIPPLEDataset(directory, "train")[0]


def test_two_stage_training_checkpoint_and_resume(cache_fixture):
    directory, _ = cache_fixture
    args = parse_args(
        [
            "--cache",
            str(directory),
            "--config",
            str(directory / "config.json"),
            "--save-dir",
            str(directory / "out"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--accumulation",
            "2",
            "--device",
            "cpu",
        ]
    )
    checkpoint = None
    for stage in CURRICULUM:
        checkpoint = run_stage(stage, args, checkpoint)
        state = torch.load(checkpoint, weights_only=True)
        assert state["stage"] == stage.value
        assert state["steps"] > 0
    assert sum(state["metrics"]["effect_support"]) == 2
    assert state["metrics"]["geometry_count"] == 2
    held_out = evaluate(checkpoint, directory)
    assert held_out["split"] == "test"
    assert sum(held_out["effect_support"]) == 2
    args.resume = str(checkpoint.parent / "last.pt")
    args.stage, args.epochs = "channel_finetune", 2
    run_stage(TrainingStage.CHANNEL_FINETUNE, args)
    resumed = torch.load(checkpoint.parent / "last.pt", weights_only=True)
    assert resumed["epoch"] == 1
    assert resumed["steps"] > state["steps"]


def test_two_process_cpu_training_uses_all_task_branches(cache_fixture):
    directory, _ = cache_fixture
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "1",
            "--nproc-per-node",
            "2",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(port),
            "--module",
            "src.ripple.train",
            "--cache",
            str(directory),
            "--config",
            str(directory / "config.json"),
            "--save-dir",
            str(directory / "ddp-out"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--accumulation",
            "2",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = torch.load(
        directory / "ddp-out/cold-target/channel_finetune/last.pt", weights_only=True
    )
    assert sum(state["metrics"]["effect_support"]) == 2
    assert state["metrics"]["geometry_count"] == 2
