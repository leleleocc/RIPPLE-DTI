"""Train the structure subgraph on real pilot geometry; not a performance benchmark."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ripple import (
    RIPPLEDTI,
    RIPPLEBatch,
    RIPPLELabels,
    RIPPLEMultiTaskLoss,
    TrainingStage,
    configure_training_stage,
)
from src.ripple.data.channels import ChannelGeometryDataset, spatial_edges


def make_structure_inputs(sample):
    coords = sample.residue_coords[None]
    edges = spatial_edges(sample.residue_coords)
    return {
        "residue_features": sample.residue_features[None],
        "residue_coords": coords,
        "residue_mask": torch.ones(coords.shape[:2], dtype=torch.bool),
        "edge_index": edges,
        "edge_mask": torch.ones(edges.shape[0], edges.shape[-1], dtype=torch.bool),
    }


def snapshot_forward(model, sample, inputs):
    structural = model.residue_encoder.encode_residues(**inputs)
    apo, _ = model.protein_fusion(
        inputs["residue_features"], structural, inputs["residue_mask"]
    )
    return model.structure_response(
        apo,
        torch.zeros_like(apo),
        torch.zeros(1, model.dim),
        inputs["residue_coords"],
        inputs["residue_mask"],
        sample.queries,
    )


def full_forward(model, sample, inputs, smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features = torch.tensor(
        [
            [
                a.GetAtomicNum() / 100,
                a.GetDegree() / 4,
                a.GetFormalCharge(),
                float(a.GetIsAromatic()),
                a.GetTotalNumHs() / 4,
                a.GetMass() / 200,
                int(a.GetHybridization()) / 10,
                float(a.IsInRing()),
            ]
            for a in mol.GetAtoms()
        ]
    )
    chemical_tokens = torch.nn.functional.pad(atom_features, (0, model.dim - 8))[None]
    # The smoke run has no pretrained chemical embeddings, docked conformers,
    # or predicted pockets. Missing modalities stay explicitly unavailable.
    batch = RIPPLEBatch(
        atom_features=torch.zeros(1, 1, 1, 8),
        atom_coords=torch.zeros(1, 1, 1, 3),
        atom_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        conformer_mask=torch.zeros(1, 1, dtype=torch.bool),
        conformer_energy=torch.zeros(1, 1),
        drug_1d_tokens=torch.zeros(1, 1, model.dim),
        drug_1d_mask=torch.zeros(1, 1, dtype=torch.bool),
        drug_2d_tokens=chemical_tokens,
        drug_2d_mask=torch.ones(chemical_tokens.shape[:2], dtype=torch.bool),
        pocket_indices=torch.zeros(1, 1, 1, dtype=torch.long),
        pocket_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        pocket_valid_mask=torch.zeros(1, 1, dtype=torch.bool),
        pocket_score=torch.zeros(1, 1),
        geometry_queries=sample.queries,
        **inputs,
    )
    return model(batch)


def run(data_dir, output_dir, steps, seed):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dataset = ChannelGeometryDataset(data_dir)
    samples = [dataset[i] for i in range(len(dataset))]
    inputs = [make_structure_inputs(sample) for sample in samples]
    model_kwargs = {
        "atom_input_dim": 8,
        "residue_input_dim": 21,
        "dim": 32,
        "conformer_layers": 1,
        "residue_layers": 2,
        "propagation_steps": 2,
        "num_heads": 4,
        "drug_latents": 8,
        "protein_latents": 8,
        "dropout": 0.0,
    }
    model = RIPPLEDTI(**model_kwargs)
    configure_training_stage(model, TrainingStage.STRUCTURE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )

    def evaluate():
        rows = []
        model.eval()
        with torch.no_grad():
            for sample, item in zip(samples, inputs, strict=True):
                prediction = snapshot_forward(model, sample, item)["geometry_snapshot"]
                for i, metric in enumerate(sample.metric_ids):
                    if sample.geometry_snapshot_mask[0, i]:
                        rows.append(
                            {
                                "pdb_id": sample.identity.structure_id,
                                "metric_id": metric,
                                "observed": sample.geometry_snapshot[0, i].item(),
                                "predicted": prediction[0, i].item(),
                            }
                        )
        return sum(abs(row["predicted"] - row["observed"]) for row in rows) / len(
            rows
        ), rows

    initial_mae, _ = evaluate()
    history = []
    for step in range(steps):
        model.train()
        index = step % len(samples)
        sample = samples[index]
        prediction = snapshot_forward(model, sample, inputs[index])["geometry_snapshot"]
        mask = sample.geometry_snapshot_mask
        loss = torch.nn.functional.smooth_l1_loss(
            prediction[mask], sample.geometry_snapshot[mask]
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(loss.item())
    final_mae, predictions = evaluate()
    if not final_mae < initial_mae:
        raise RuntimeError("Geometry smoke optimization did not improve training error")

    # Use a documented same-accession ligand for a complete model smoke pass.
    # Protein-level pharmacology is not transferred to a mutated construct.
    pharmacology = pq.read_table(data_dir / "pharmacology_labels.parquet").to_pylist()
    atp = next(
        row
        for row in pharmacology
        if row["uniprot_id"] == "P56373"
        and row["ligand_name"] == "ATP"
        and row["effect_mask"]
    )
    index = next(
        i for i, sample in enumerate(samples) if sample.identity.structure_id == "5SVJ"
    )
    configure_training_stage(model, TrainingStage.JOINT)
    output = full_forward(model, samples[index], inputs[index], atp["smiles"])
    labels = RIPPLELabels(
        geometry_snapshot=samples[index].geometry_snapshot,
        geometry_snapshot_mask=samples[index].geometry_snapshot_mask,
    )
    losses = RIPPLEMultiTaskLoss()(output, labels)
    model.zero_grad()
    losses.total.backward()
    finite = all(torch.isfinite(value).all() for value in output.predictions.values())
    if not finite or not torch.isfinite(losses.total):
        raise RuntimeError("Nonfinite full-model smoke output")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kwargs": model_kwargs,
            "structure_state_dict": {
                name: value
                for name, value in model.state_dict().items()
                if name.startswith(
                    (
                        "residue_encoder.",
                        "protein_fusion.",
                        "structure_response.snapshot_readout.",
                    )
                )
            },
            "purpose": "geometry smoke only; 21-way amino acid features; no held-out evaluation",
            "steps": steps,
            "seed": seed,
        },
        output_dir / "geometry_smoke.pt",
    )
    report = {
        "purpose": "pipeline smoke, not generalization evaluation",
        "steps": steps,
        "seed": seed,
        "structures": len(samples),
        "observed_metrics": len(predictions),
        "initial_training_mae_angstrom": initial_mae,
        "final_training_mae_angstrom": final_mae,
        "full_forward_backward": "5SVJ + ATP; observed geometry labels only; no construct-matched effect label",
        "full_model_outputs_finite": finite,
        "label_counts": losses.label_count,
        "loss_history": history,
        "predictions": predictions,
        "limitations": [
            "No held-out evaluation; all 8 structures used for smoke optimization",
            "No full five-source pretraining or channel-response fine-tuning",
            "Raw atom/residue features; no pretrained embeddings, predicted pockets or poses",
            "No gating/state-preference/causal-response supervision used",
        ],
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"predictions", "loss_history"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/processed/channels"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/channel_geometry_smoke"
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.steps, args.seed)
