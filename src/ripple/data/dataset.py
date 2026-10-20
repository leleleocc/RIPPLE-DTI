"""Load audited manifests and tensor caches; preserve sparse supervision."""

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.ripple.data.manifest import SPLITS, SampleRecord, read_manifest
from src.ripple.data.schema import ENDPOINT_UNKNOWN, SOURCES, LabelMasks, SourceTable
from src.ripple.model import GeometryBatch, RIPPLEBatch
from src.ripple.structure_response import GeometryQueries
from src.ripple.training import RIPPLELabels

LABEL_FIELDS = {
    "interaction": ("interaction", "interaction_mask"),
    "affinity": ("affinity", "affinity_mask"),
    "effect": ("effect", "effect_mask"),
    "mechanism": ("mechanism", "mechanism_mask"),
    "pocket": ("native_pocket_index", "native_pocket_mask"),
    "function": ("function_label", "function_mask"),
    "pose": ("pose_quality", "pose_mask"),
    "allosteric": ("allosteric_pocket", "allosteric_pocket_mask"),
    "geometry_snapshot": ("geometry_snapshot", "geometry_snapshot_mask"),
    "geometry_response": ("geometry_response", "geometry_response_mask"),
    "gating": ("gating", "gating_mask"),
    "state_preference": ("state_preference", "state_preference_mask"),
}
INTEGER_LABELS = {
    "effect",
    "mechanism",
    "native_pocket_index",
    "gating",
    "state_preference",
}
PROTEIN_FIELDS = {
    "residue_features",
    "residue_coords",
    "residue_mask",
    "edge_index",
    "edge_mask",
    "edge_features",
    "protein_sequence_tokens",
}
POCKET_FIELDS = {"pocket_indices", "pocket_mask", "pocket_valid_mask", "pocket_score"}
DRUG_FIELDS = {
    "atom_features",
    "atom_coords",
    "atom_mask",
    "conformer_mask",
    "conformer_energy",
    "drug_1d_tokens",
    "drug_1d_mask",
    "drug_2d_tokens",
    "drug_2d_mask",
}


@dataclass
class RIPPLESample:
    """A single sample keeps a leading tensor dimension of one."""

    batch: RIPPLEBatch | GeometryBatch
    labels: RIPPLELabels
    source: SourceTable
    masks: LabelMasks
    record: SampleRecord | None = None


class RIPPLEDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str,
        tasks=None,
        manifest="manifest.jsonl",
        *,
        records=None,
    ):
        self.cache_dir = Path(cache_dir)
        if split not in SPLITS:
            raise ValueError(f"Unknown manifest split: {split}")
        self.split = split
        all_records = (
            read_manifest(self.cache_dir / manifest) if records is None else records
        )
        self.records = [
            row
            for row in all_records
            if row.split == split and (tasks is None or row.task in tasks)
        ]
        self._read_cache = lru_cache(maxsize=16)(self._load_cache)

    def _load_cache(self, relative_path):
        value = torch.load(
            self.cache_dir / relative_path, map_location="cpu", weights_only=True
        )
        if not isinstance(value, dict) or value.get("cache_version") != 1:
            raise ValueError(f"{relative_path}: expected tensor cache version 1")
        if not isinstance(value.get("tensors"), dict):
            raise TypeError(f"{relative_path}: missing tensors")
        return value

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_read_cache")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._read_cache = lru_cache(maxsize=16)(self._load_cache)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        protein = self._read_cache(record.protein_cache)
        tensors = protein["tensors"]
        if protein.get("structure_hash") != record.structure.structure_hash:
            raise ValueError("Protein cache structure hash does not match manifest")
        if protein.get("mapping_version") != record.structure.mapping_version:
            raise ValueError("Protein cache mapping version does not match manifest")
        if protein.get("chains") != [
            asdict(chain) for chain in record.structure.chains
        ]:
            raise ValueError(
                "Protein cache sequence/construct identities do not match manifest"
            )
        if not protein.get("feature_version"):
            raise ValueError("Protein cache requires a feature extraction version")
        expected_keys = [
            [chain.chain_id, position]
            for chain in record.structure.chains
            for position in range(chain.canonical_length)
        ]
        if protein.get("residue_keys") != expected_keys:
            raise ValueError(
                "Residue keys must enumerate each chain in canonical sequence order"
            )
        if tensors["residue_features"].size(0) != len(expected_keys):
            raise ValueError(
                "Residue tensors and chain/canonical mapping have different lengths"
            )
        mapping = protein.get("residue_mapping")
        if not isinstance(mapping, list) or len(mapping) != len(expected_keys):
            raise ValueError(
                "Protein cache requires a residue numbering map aligned to all nodes"
            )
        required_numbering = {
            "auth_asym_id",
            "auth_seq_id",
            "insertion_code",
            "label_asym_id",
            "label_seq_id",
        }
        for node_index in tensors["residue_mask"].nonzero().flatten().tolist():
            if (
                not isinstance(mapping[node_index], dict)
                or not required_numbering <= mapping[node_index].keys()
            ):
                raise ValueError(
                    "Every valid coordinate requires author and label residue numbering"
                )
        tensor_fields = (
            PROTEIN_FIELDS
            if record.task == "geometry"
            else PROTEIN_FIELDS | POCKET_FIELDS
        )
        inputs = {
            name: value.unsqueeze(0)
            for name, value in tensors.items()
            if name in tensor_fields
        }
        if record.geometry_queries is not None:
            query = record.geometry_queries
            inputs["geometry_queries"] = GeometryQueries(
                torch.tensor(query["indices"], dtype=torch.long).unsqueeze(0),
                torch.tensor(query["pair_mask"], dtype=torch.bool).unsqueeze(0),
            )
        if record.task == "geometry":
            batch = GeometryBatch(**inputs)
        else:
            drug = self._read_cache(record.drug_cache)
            if drug.get("inchikey") != record.inchikey:
                raise ValueError("Drug cache InChIKey does not match manifest")
            if not drug.get("standardization_version") or not drug.get(
                "feature_version"
            ):
                raise ValueError(
                    "Drug cache requires chemical standardization and feature versions"
                )
            inputs.update(
                {
                    name: value.unsqueeze(0)
                    for name, value in drug["tensors"].items()
                    if name in DRUG_FIELDS
                }
            )
            inputs["endpoint_type"] = torch.tensor(
                [
                    ENDPOINT_UNKNOWN
                    if record.endpoint_type is None
                    else record.endpoint_type,
                ]
            )
            batch = RIPPLEBatch(**inputs)
        labels = {}
        known_labels = {field.name for field in fields(RIPPLELabels)}
        for name, value in record.labels.items():
            if name not in known_labels:
                raise ValueError(f"Unknown label field {name}")
            if value is not None:
                dtype = (
                    torch.bool
                    if name.endswith("_mask")
                    else (torch.long if name in INTEGER_LABELS else torch.float32)
                )
                labels[name] = torch.tensor(value, dtype=dtype).unsqueeze(0)
        sample = RIPPLESample(
            batch,
            RIPPLELabels(**labels),
            SOURCES[record.source],
            LabelMasks(**record.loss_enabled),
            record,
        )
        validate_sample(sample)
        return sample


def validate_sample(sample):
    batch = sample.batch
    if batch.residue_mask.ndim != 2 or batch.residue_mask.size(0) != 1:
        raise ValueError("Each sample must have a leading batch dimension of one")
    mask = batch.residue_mask.bool()
    if not mask.any() or batch.residue_coords.shape != (*mask.shape, 3):
        raise ValueError("A sample requires mapped residue coordinates")
    if batch.residue_features.shape[:2] != mask.shape:
        raise ValueError("Residue features and mask must align")
    if not torch.isfinite(batch.residue_features[mask]).all():
        raise ValueError("Valid residue features must be finite")
    if not torch.isfinite(batch.residue_coords[mask]).all():
        raise ValueError("Valid residue coordinates must be finite")
    if batch.edge_index.shape != (1, 2, batch.edge_mask.size(-1)):
        raise ValueError("edge_index must have shape [1, 2, E]")
    active = batch.edge_index[0, :, batch.edge_mask[0]]
    if active.numel() and ((active < 0).any() or (active >= mask.size(1)).any()):
        raise ValueError("Edge indices must be sample-local residue indices")
    if isinstance(batch, GeometryBatch):
        query = batch.geometry_queries
        if (
            query.indices.ndim != 4
            or query.indices.shape[:1] != (1,)
            or query.indices.size(-1) != 2
        ):
            raise ValueError("Geometry indices must have shape [1, M, P, 2]")
        if query.pair_mask.shape != query.indices.shape[:-1]:
            raise ValueError("Geometry pair mask must match query indices")
        active = query.indices[query.pair_mask]
        if not active.numel() or (active < 0).any() or (active >= mask.size(1)).any():
            raise ValueError(
                "Geometry queries require valid sample-local residue indices"
            )
        if (active[:, 0] == active[:, 1]).any() or not mask[0, active].all():
            raise ValueError("Geometry queries require two distinct mapped residues")
        if batch.geometry_queries.pair_mask.sum() > 64:
            raise ValueError(
                "At most 64 geometry pairs are allowed per assembly sample"
            )
        if any(
            getattr(sample.labels, label) is not None
            for label, _ in LABEL_FIELDS.values()
            if label != "geometry_snapshot"
        ):
            raise ValueError("Geometry samples may only carry current-geometry labels")
    else:
        if not batch.drug_1d_mask.any() or not batch.drug_2d_mask.any():
            raise ValueError("Drug tasks require real sequence and graph drug features")
        active = batch.pocket_mask & batch.pocket_valid_mask[..., None]
        indices = batch.pocket_indices[active]
        if (
            not indices.numel()
            or (indices < 0).any()
            or (indices >= mask.size(1)).any()
        ):
            raise ValueError(
                "Candidate pockets require valid sample-local residue indices"
            )


def _pad(values, fill=0, variable_axes=None, target_shape=None):
    """Pad unbatched tensors only on declared axes; feature dimensions stay fixed."""
    template = next(value for value in values if value is not None)
    variable_axes = (
        set(range(template.ndim)) if variable_axes is None else set(variable_axes)
    )
    shape = list(template.shape)
    for value in values:
        if value is None:
            continue
        if value.ndim != template.ndim or value.dtype != template.dtype:
            raise ValueError("Incompatible tensor ranks or dtypes in batch")
        for axis, size in enumerate(value.shape):
            if axis not in variable_axes and size != shape[axis]:
                raise ValueError("Incompatible feature dimensions in batch")
            shape[axis] = max(shape[axis], size)
    if target_shape is not None:
        if len(shape) != len(target_shape) or any(
            a > b for a, b in zip(shape, target_shape)
        ):
            raise ValueError("Label tensor exceeds its corresponding input dimensions")
        shape = list(target_shape)
    result = template.new_full((len(values), *shape), fill)
    for index, value in enumerate(values):
        if value is not None:
            result[(index, *(slice(0, size) for size in value.shape))] = value
    return result


def collate_ripple(
    samples: list[RIPPLESample],
) -> tuple[RIPPLEBatch | GeometryBatch, RIPPLELabels]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_class = type(samples[0].batch)
    if any(type(sample.batch) is not batch_class for sample in samples):
        raise ValueError("Geometry and drug samples must use separate task batches")
    for sample in samples:
        validate_sample(sample)
    inputs = {}
    feature_fields = {
        "atom_features",
        "atom_coords",
        "drug_1d_tokens",
        "drug_2d_tokens",
        "residue_features",
        "residue_coords",
        "edge_features",
        "protein_sequence_tokens",
    }
    for field in fields(batch_class):
        name = field.name
        values = [getattr(sample.batch, name) for sample in samples]
        if all(value is None for value in values):
            inputs[name] = None
            continue
        if name == "geometry_queries":
            inputs[name] = GeometryQueries(
                _pad(
                    [None if value is None else value.indices[0] for value in values],
                    -1,
                    (0, 1),
                ),
                _pad(
                    [None if value is None else value.pair_mask[0] for value in values],
                    False,
                ),
            )
            continue
        if name in {"edge_features", "protein_sequence_tokens"} and any(
            value is None for value in values
        ):
            raise ValueError(
                f"{name} must be present for all samples or absent for all"
            )
        if name == "function_mask":
            inputs[name] = None
            continue
        values = [None if value is None else value[0] for value in values]
        if name == "endpoint_type":
            inputs[name] = _pad(values, ENDPOINT_UNKNOWN, ())
            continue
        exemplar = next(value for value in values if value is not None)
        axes = (
            (1,)
            if name == "edge_index"
            else (tuple(range(exemplar.ndim - 1)) if name in feature_fields else None)
        )
        fill = -1 if name in {"edge_index", "pocket_indices"} else 0
        inputs[name] = _pad(values, fill, axes)

    labels = {}
    for permission, (target_name, mask_name) in LABEL_FIELDS.items():
        targets, masks = [], []
        for sample in samples:
            value = getattr(sample.labels, target_name)
            allowed = sample.masks.allows(sample.source, permission)
            if value is None or not allowed:
                targets.append(None)
                masks.append(None)
                continue
            valid = getattr(sample.labels, mask_name)
            value = value[0]
            valid = (
                torch.ones_like(value, dtype=torch.bool)
                if valid is None
                else valid[0].bool()
            )
            valid = valid & (
                torch.isfinite(value) if value.is_floating_point() else value >= 0
            )
            if target_name == "effect":
                valid = valid & (value < 2)
            if target_name == "affinity":
                endpoint = sample.batch.endpoint_type
                valid = valid & (
                    False
                    if endpoint is None
                    else ((endpoint[0] >= 0) & (endpoint[0] < 4))
                )
            targets.append(value)
            masks.append(valid)
        if any(value is not None for value in targets):
            target_shape = None
            if target_name == "function_label":
                target_shape = inputs["residue_mask"].shape[1:]
            elif target_name == "allosteric_pocket":
                target_shape = inputs["pocket_valid_mask"].shape[1:]
            elif target_name == "pose_quality":
                target_shape = (
                    inputs["pocket_indices"].size(1),
                    inputs["conformer_mask"].size(1),
                )
            elif target_name in {"geometry_snapshot", "geometry_response"}:
                target_shape = inputs["geometry_queries"].pair_mask.shape[1:2]
            labels[target_name] = _pad(
                targets,
                -1 if target_name in INTEGER_LABELS else float("nan"),
                target_shape=target_shape,
            )
            labels[mask_name] = _pad(masks, False, target_shape=target_shape)
    return batch_class(**inputs), RIPPLELabels(**labels)


def pocket_residue_indices(contacts: Tensor, cutoff: float = 4.0) -> Tensor:
    """Indices of residues contacting a ligand, for labels only ([R, A] distances)."""
    if contacts.ndim != 2 or cutoff <= 0:
        raise ValueError(
            "Expected [residue, ligand atom] distances and a positive cutoff"
        )
    return (
        (torch.isfinite(contacts) & (contacts >= 0) & (contacts <= cutoff))
        .any(-1)
        .nonzero()
        .flatten()
    )
