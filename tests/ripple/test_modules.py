import torch

from src.ripple import RIPPLEDTI, RIPPLEBatch
from src.ripple.propagation import DifferentialResiduePropagation


def make_batch(batch_size=2, dim=16, edge_dim=3):
    conformer_count = 3
    atom_count = 16
    residue_count = 10
    pocket_count = 2
    pocket_size = 4

    chain_source = []
    chain_target = []
    for index in range(residue_count - 1):
        chain_source.extend([index, index + 1])
        chain_target.extend([index + 1, index])
    edge_index = torch.tensor([chain_source, chain_target], dtype=torch.long)
    edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1).clone()
    edge_count = edge_index.size(-1)

    atom_mask = torch.ones(batch_size, conformer_count, atom_count, dtype=torch.bool)
    atom_mask[:, :, -2:] = False
    conformer_mask = torch.ones(batch_size, conformer_count, dtype=torch.bool)
    conformer_mask[-1, -1] = False
    atom_mask = atom_mask & conformer_mask.unsqueeze(-1)

    residue_mask = torch.ones(batch_size, residue_count, dtype=torch.bool)
    residue_mask[-1, -1] = False
    pocket_indices = (
        torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8]]], dtype=torch.long)
        .expand(batch_size, -1, -1)
        .clone()
    )
    pocket_mask = torch.ones(batch_size, pocket_count, pocket_size, dtype=torch.bool)
    pocket_valid_mask = torch.ones(batch_size, pocket_count, dtype=torch.bool)
    pocket_valid_mask[-1, -1] = False
    pocket_mask = pocket_mask & pocket_valid_mask.unsqueeze(-1)

    function_mask = torch.zeros(batch_size, residue_count, dtype=torch.bool)
    function_mask[:, 8] = True

    return RIPPLEBatch(
        atom_features=torch.randn(batch_size, conformer_count, atom_count, 8),
        atom_coords=torch.randn(batch_size, conformer_count, atom_count, 3),
        atom_mask=atom_mask,
        conformer_mask=conformer_mask,
        conformer_energy=torch.randn(batch_size, conformer_count),
        drug_1d_tokens=torch.randn(batch_size, 7, dim),
        drug_1d_mask=torch.ones(batch_size, 7, dtype=torch.bool),
        drug_2d_tokens=torch.randn(batch_size, 9, dim),
        drug_2d_mask=torch.ones(batch_size, 9, dtype=torch.bool),
        residue_features=torch.randn(batch_size, residue_count, 12),
        residue_coords=torch.randn(batch_size, residue_count, 3),
        residue_mask=residue_mask,
        edge_index=edge_index,
        edge_mask=torch.ones(batch_size, edge_count, dtype=torch.bool),
        edge_features=torch.randn(batch_size, edge_count, edge_dim),
        pocket_indices=pocket_indices,
        pocket_mask=pocket_mask,
        pocket_valid_mask=pocket_valid_mask,
        pocket_score=torch.rand(batch_size, pocket_count),
        function_mask=function_mask,
        protein_sequence_tokens=torch.randn(batch_size, residue_count, dim),
        endpoint_type=torch.tensor([0, 2], dtype=torch.long),
    )


def test_full_model_forward_masks_and_backward():
    torch.manual_seed(7)
    model = RIPPLEDTI(
        atom_input_dim=8,
        residue_input_dim=12,
        dim=16,
        edge_dim=3,
        conformer_layers=1,
        residue_layers=2,
        propagation_steps=3,
        num_heads=4,
        dropout=0.0,
    )
    batch = make_batch()
    output = model(batch)

    assert output.predictions["interaction_logit"].shape == (2,)
    assert output.predictions["affinity"].shape == (2,)
    assert output.predictions["effect_logits"].shape == (2, 3)
    assert output.predictions["mechanism_logits"].shape == (2, 6)
    assert output.match.joint_weight.shape == (2, 2, 3)
    assert output.propagation.delta_tokens.shape == (2, 10, 16)
    assert output.propagation.edge_flux.shape == (2, 3, 18)
    assert output.apo_tokens.shape == (2, 10, 16)
    assert output.interaction.global_context.shape == (2, 32)

    torch.testing.assert_close(output.match.joint_weight.sum(dim=(1, 2)), torch.ones(2))
    assert output.match.joint_weight[-1, -1].count_nonzero() == 0
    assert output.match.joint_weight[-1, :, -1].count_nonzero() == 0
    torch.testing.assert_close(
        output.match.pocket_weight,
        output.match.joint_weight.sum(dim=-1),
    )
    torch.testing.assert_close(
        output.match.conformer_weight,
        output.match.joint_weight.sum(dim=-2),
    )
    torch.testing.assert_close(
        output.propagation.sink_weight.sum(dim=-1),
        torch.ones(2),
    )

    loss = (
        output.predictions["interaction_logit"].mean()
        + output.predictions["affinity"].mean()
        + output.predictions["effect_logits"].mean()
        + output.predictions["mechanism_logits"].mean()
    )
    loss.backward()
    assert model.matcher.pair_scorer[-1].weight.grad is not None
    assert model.matcher.pair_scorer[-1].weight.grad.abs().sum() > 0
    assert model.propagation.delta_projection.weight.grad is not None


def test_zero_source_produces_exact_zero_delta():
    torch.manual_seed(11)
    batch_size, residue_count, dim = 2, 6, 8
    source = torch.arange(residue_count - 1)
    target = source + 1
    edge_index = torch.stack([source, target]).unsqueeze(0).expand(batch_size, -1, -1)
    module = DifferentialResiduePropagation(
        dim=dim,
        edge_dim=0,
        num_steps=3,
        dropout=0.0,
    )
    residue_tokens = torch.randn(batch_size, residue_count, dim)
    output = module(
        residue_tokens=residue_tokens,
        residue_coords=torch.randn(batch_size, residue_count, 3),
        residue_mask=torch.ones(batch_size, residue_count, dtype=torch.bool),
        edge_index=edge_index,
        edge_mask=torch.ones(batch_size, residue_count - 1, dtype=torch.bool),
        residue_source=torch.zeros_like(residue_tokens),
        pair_context=torch.randn(batch_size, dim),
    )

    torch.testing.assert_close(
        output.delta_tokens, torch.zeros_like(output.delta_tokens)
    )
    torch.testing.assert_close(output.ligand_on_tokens, residue_tokens)
    torch.testing.assert_close(output.edge_flux, torch.zeros_like(output.edge_flux))


def test_function_annotation_does_not_leak_into_the_sink():
    """Functional annotations are a training target, not an input.

    Passing the annotation must not change the predicted sink, otherwise the
    pocket-to-function head would be reading its own label.
    """
    torch.manual_seed(13)
    batch_size, residue_count, dim = 1, 5, 8
    edge_index = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4]]], dtype=torch.long)
    module = DifferentialResiduePropagation(dim=dim, num_steps=1, dropout=0.0)
    residue_tokens = torch.randn(batch_size, residue_count, dim)
    residue_coords = torch.randn(batch_size, residue_count, 3)
    residue_mask = torch.ones(batch_size, residue_count, dtype=torch.bool)
    edge_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    residue_source = torch.randn(batch_size, residue_count, dim)
    pair_context = torch.randn(batch_size, dim)
    function_mask = torch.tensor([[False, True, False, True, False]])

    plain = module(
        residue_tokens=residue_tokens,
        residue_coords=residue_coords,
        residue_mask=residue_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        residue_source=residue_source,
        pair_context=pair_context,
    )
    annotated = module(
        residue_tokens=residue_tokens,
        residue_coords=residue_coords,
        residue_mask=residue_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        residue_source=residue_source,
        pair_context=pair_context,
        function_mask=function_mask,
    )

    torch.testing.assert_close(annotated.sink_weight, plain.sink_weight)
    torch.testing.assert_close(annotated.sink_logits, plain.sink_logits)
    assert plain.sink_weight.shape == (batch_size, residue_count)


def test_missing_poses_keep_the_dti_path_finite():
    """Pose confidence is auxiliary: DTI must survive with no usable pose.

    With every conformer masked out but pockets still present, the interaction
    prediction stays finite and its gradient still reaches the drug and protein
    encoders. The old contract zeroed the whole path; that dropped pocket-aware
    DTI exactly on the samples that have no docking pose.
    """
    torch.manual_seed(17)
    model = RIPPLEDTI(
        atom_input_dim=8,
        residue_input_dim=12,
        dim=16,
        edge_dim=3,
        conformer_layers=1,
        residue_layers=1,
        propagation_steps=2,
        num_heads=4,
        dropout=0.0,
    )
    batch = make_batch()
    batch.conformer_mask.zero_()
    batch.atom_mask.zero_()
    batch.protein_sequence_tokens = None
    output = model(batch)

    assert torch.isfinite(output.predictions["interaction_logit"]).all()
    assert torch.isfinite(output.interaction.global_context).all()
    torch.testing.assert_close(
        output.match.pose_confidence, torch.zeros_like(output.match.pose_confidence)
    )
    assert output.match.pocket_weight.sum(dim=-1).gt(0).all()

    output.predictions["interaction_logit"].sum().backward()
    assert model.drug_fusion.chemical_fusion[-1].weight.grad.abs().sum() > 0
    assert model.protein_fusion.sequence_projection[-1].weight.grad.abs().sum() > 0


def test_conformer_encoder_handles_fewer_atoms_than_knn_neighbors():
    torch.manual_seed(19)
    model = RIPPLEDTI(
        atom_input_dim=8,
        residue_input_dim=12,
        dim=16,
        edge_dim=3,
        conformer_layers=1,
        conformer_neighbors=16,
        residue_layers=1,
        propagation_steps=1,
        num_heads=4,
        dropout=0.0,
    )
    batch = make_batch(batch_size=2)
    batch.atom_features = batch.atom_features[:, :, :7]
    batch.atom_coords = batch.atom_coords[:, :, :7]
    batch.atom_mask = batch.atom_mask[:, :, :7]
    output = model(batch)
    assert output.conformers.tokens.shape == (2, 3, 7, 16)
    assert torch.isfinite(output.conformers.tokens).all()
