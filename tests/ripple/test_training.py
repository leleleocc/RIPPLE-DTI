import torch

from src.ripple import RIPPLEDTI
from src.ripple import (
    RIPPLELabels,
    RIPPLEMultiTaskLoss,
    TrainingStage,
    build_optimizer_param_groups,
    configure_training_stage,
    stage_loss_weights,
)
from tests.ripple.test_modules import make_batch


def make_model():
    return RIPPLEDTI(
        atom_input_dim=8,
        residue_input_dim=12,
        dim=16,
        edge_dim=3,
        conformer_layers=1,
        residue_layers=1,
        propagation_steps=2,
        num_heads=4,
        drug_latents=8,
        protein_latents=8,
        dropout=0.0,
    )


def test_multisource_loss_ignores_missing_labels():
    torch.manual_seed(29)
    model = make_model()
    output = model(make_batch())
    labels = RIPPLELabels(
        interaction=torch.tensor([1.0, 0.0]),
        affinity=torch.tensor([7.2, float("nan")]),
        effect=torch.tensor([2, -1]),
        native_pocket_index=torch.tensor([0, 1]),
        native_pocket_mask=torch.tensor([True, False]),
    )
    result = RIPPLEMultiTaskLoss()(output, labels)

    assert torch.isfinite(result.total)
    assert result.label_count["interaction"] == 2
    assert result.label_count["affinity"] == 1
    assert result.label_count["effect"] == 1
    assert result.label_count["mechanism"] == 0
    assert result.label_count["native_pocket"] == 1
    result.total.backward()
    assert model.matcher.pair_scorer[-1].weight.grad is not None


def test_native_index_losses_ignore_unavailable_candidates():
    torch.manual_seed(31)
    model = make_model()
    batch = make_batch()
    output = model(batch)
    labels = RIPPLELabels(native_pocket_index=torch.tensor([0, 1]))

    result = RIPPLEMultiTaskLoss()(output, labels)

    assert result.label_count["native_pocket"] == 1
    assert result.label_count["pose_rank"] == 0
    assert torch.isfinite(result.total)


def test_function_and_pose_losses_respect_their_masks():
    torch.manual_seed(37)
    model = make_model()
    output = model(make_batch())
    residue_count = output.propagation.sink_logits.size(1)
    function_label = torch.zeros(2, residue_count)
    function_label[0, 8] = 1.0
    function_label[1, 3] = 1.0
    # Only the first sample has a functional annotation. The second must not
    # contribute, and must not be treated as an all-negative residue label.
    function_mask = torch.zeros(2, residue_count, dtype=torch.bool)
    function_mask[0] = True

    pose_quality = torch.tensor(
        [[[0.0, -1.5, -4.0], [0.0, -3.0, -6.0]], [[0.0, -2.0, -5.0], [0.0, 0.0, 0.0]]]
    )
    pose_mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    pose_mask[0, 0, :2] = True
    labels = RIPPLELabels(
        function_label=function_label,
        function_mask=function_mask,
        pose_quality=pose_quality,
        pose_mask=pose_mask,
    )
    result = RIPPLEMultiTaskLoss()(output, labels)

    assert result.label_count["function"] == residue_count
    assert result.label_count["pose_rank"] > 0
    assert torch.isfinite(result.total)
    result.total.backward()
    assert model.propagation.sink_score[-1].weight.grad is not None
    assert model.matcher.pose_quality_head[-1].weight.grad is not None


def test_training_stages_freeze_expected_modules():
    model = make_model()
    configure_training_stage(model, TrainingStage.PROPAGATION)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert any(name.startswith("propagation.") for name in trainable)
    assert any(name.startswith("source_fusion.") for name in trainable)
    assert not any(name.startswith("matcher.") for name in trainable)
    assert not any(name.startswith("conformer_encoder.") for name in trainable)

    weights = stage_loss_weights(TrainingStage.PROPAGATION)
    assert weights.function > 0
    assert weights.effect > 0
    assert weights.nonbinder_delta > 0
    assert weights.interaction == 0
    assert weights.pose_rank == 0

    pocket_weights = stage_loss_weights(TrainingStage.POCKET)
    assert pocket_weights.native_pocket > 0
    assert pocket_weights.pose_rank > 0
    assert pocket_weights.interaction == 0


def test_joint_stage_and_optimizer_groups_cover_trainable_parameters():
    model = make_model()
    configure_training_stage(model, TrainingStage.JOINT)
    groups = build_optimizer_param_groups(
        model,
        base_lr=1e-4,
        weight_decay=1e-4,
        backbone_lr_multiplier=0.2,
    )
    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    assert grouped_ids == trainable_ids
    assert {group["name"] for group in groups} == {
        "shared_backbone",
        "ripple_modules",
    }
    assert groups[0]["lr"] < groups[1]["lr"]
