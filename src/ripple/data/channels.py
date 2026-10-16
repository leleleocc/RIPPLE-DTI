"""Channel geometry staging data, distinct from the final multimodal feature cache."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.ripple.structure_response import GeometryQueries


@dataclass(frozen=True)
class ChannelStateIdentity:
    uniprot_id: str
    structure_id: str
    split_group: str
    construct_sequence: str


def validate_state_group(states):
    if not states:
        raise ValueError("A state comparison requires at least one structure")
    if len({state.uniprot_id for state in states}) != 1:
        raise ValueError("Candidate states must have the same UniProt identity")
    if len({state.split_group for state in states}) != 1:
        raise ValueError("All conformations must stay in one split group")
    if len({state.construct_sequence for state in states}) != 1:
        raise ValueError("State preference requires matched protein constructs")


@dataclass
class ChannelGeometrySample:
    identity: ChannelStateIdentity
    residue_coords: Tensor
    residue_features: Tensor
    residue_keys: list
    queries: GeometryQueries
    metric_ids: list
    geometry_snapshot: Tensor
    geometry_snapshot_mask: Tensor


AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


class ChannelGeometryDataset(Dataset):
    """Real observed geometry for representation pretraining, not drug-response labels.

    Features are explicit 21-way residue identities for a smoke run. They are not
    protein-language-model embeddings. The reference coordinates never include
    another state's coordinates. A drug-response target requires a separate,
    intervention-supported label and cannot be obtained from this loader.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.states = json.loads((self.directory / "states.json").read_text())

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        data = json.loads(
            (self.directory / "structures" / (state["pdb_id"] + ".json")).read_text()
        )
        residues, metrics = data["residues"], data["metrics"]
        pair_count = max(1, max(len(metric["pair_indices"]) for metric in metrics))
        indices = torch.full((1, len(metrics), pair_count, 2), -1, dtype=torch.long)
        pair_mask = torch.zeros(indices.shape[:-1], dtype=torch.bool)
        for i, metric in enumerate(metrics):
            if metric["valid"]:
                n = len(metric["pair_indices"])
                indices[0, i, :n] = torch.tensor(
                    metric["pair_indices"], dtype=torch.long
                )
                pair_mask[0, i, :n] = True
        amino = {name: i for i, name in enumerate(AMINO_ACIDS)}
        codes = torch.tensor(
            [amino.get(residue["residue_name"], 20) for residue in residues]
        )
        features = torch.nn.functional.one_hot(codes, 21).float()
        identity = ChannelStateIdentity(
            state["uniprot_id"],
            state["pdb_id"],
            state["split_group"],
            state["construct_sequence"],
        )
        return ChannelGeometrySample(
            identity,
            torch.tensor([r["coord"] for r in residues]),
            features,
            [
                (r["chain_id"], r["uniprot_position"], r["auth_seq_id"])
                for r in residues
            ],
            GeometryQueries(indices, pair_mask),
            [m["metric_id"] for m in metrics],
            torch.tensor(
                [[m["value_angstrom"] if m["valid"] else float("nan") for m in metrics]]
            ),
            torch.tensor([[m["valid"] for m in metrics]], dtype=torch.bool),
        )


def spatial_edges(coords: Tensor, neighbors: int = 8):
    """Deterministic C-alpha neighbor graph; no target-state coordinates used."""
    distances = torch.cdist(coords, coords)
    distances.fill_diagonal_(float("inf"))
    count = min(neighbors, len(coords) - 1)
    target = distances.topk(count, largest=False).indices
    source = torch.arange(len(coords))[:, None].expand_as(target)
    return torch.stack([source.flatten(), target.flatten()])[None]
