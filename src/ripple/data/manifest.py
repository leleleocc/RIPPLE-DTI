"""Versioned sample identities and evidence, separate from cached tensors."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.ripple.data.schema import SOURCES, LabelMasks

SPLITS = {"train", "val", "test", "external"}
TASKS = {"measurement", "site", "complex", "effect", "geometry"}


@dataclass(frozen=True)
class ChainReference:
    chain_id: str
    accession: str
    taxon_id: int
    sequence_hash: str
    canonical_length: int
    construct_hash: str

    def __post_init__(self):
        if not all(
            (self.chain_id, self.accession, self.sequence_hash, self.construct_hash)
        ):
            raise ValueError(
                "Chain identity, sequence and construct hashes are required"
            )
        if self.taxon_id <= 0 or self.canonical_length <= 0:
            raise ValueError("Taxon and canonical length must be positive")


@dataclass(frozen=True)
class StructureReference:
    pdb_id: str
    assembly_id: str
    structure_hash: str
    mapping_version: str
    chains: tuple[ChainReference, ...]

    def __post_init__(self):
        if not all(
            (
                self.pdb_id,
                self.assembly_id,
                self.structure_hash,
                self.mapping_version,
                self.chains,
            )
        ):
            raise ValueError("An audited structure, assembly and mapping are required")
        if len({chain.chain_id for chain in self.chains}) != len(self.chains):
            raise ValueError("Assembly chain copies must have distinct chain IDs")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source: str
    task: str
    split: str
    split_group: str
    protein_cache: str
    structure: StructureReference
    evidence: dict
    drug_cache: str | None = None
    inchikey: str | None = None
    endpoint_type: int | None = None
    labels: dict = field(default_factory=dict)
    loss_enabled: dict = field(default_factory=dict)
    geometry_queries: dict | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported sample schema version")
        aliases = {
            "native_pocket": "pocket",
            "pose_rank": "pose",
            "allosteric_pocket": "allosteric",
        }
        permissions = {}
        for name, value in self.loss_enabled.items():
            name = aliases.get(name, name)
            if name in permissions and permissions[name] != value:
                raise ValueError(f"Conflicting loss permissions for {name}")
            permissions[name] = value
        LabelMasks(**permissions)
        object.__setattr__(self, "loss_enabled", permissions)
        if not self.sample_id or not self.split_group or not self.protein_cache:
            raise ValueError("Sample ID, split group and protein cache are required")
        if (
            self.source not in SOURCES
            or self.task not in TASKS
            or self.split not in SPLITS
        ):
            raise ValueError("Unknown source, task or split")
        if not self.evidence.get("source_version") or not self.evidence.get(
            "record_ids"
        ):
            raise ValueError("Evidence requires source_version and record_ids")
        if self.task == "geometry":
            if self.drug_cache is not None or self.inchikey is not None:
                raise ValueError("Geometry records must not contain a drug")
            if self.geometry_queries is None:
                raise ValueError("Geometry queries must be frozen in the sample record")
        elif not self.drug_cache or not self.inchikey:
            raise ValueError(
                "Drug tasks require a real drug cache and standardized InChIKey"
            )

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        structure = dict(value["structure"])
        structure["chains"] = tuple(
            ChainReference(**chain) for chain in structure["chains"]
        )
        value["structure"] = StructureReference(**structure)
        return cls(**value)


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(SampleRecord.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    validate_partition(records)
    return records


def validate_partition(records: list[SampleRecord]):
    """Validate the complete manifest before selecting any stage or split."""
    sample_ids, groups, proteins, sequences, structures = set(), {}, {}, {}, {}
    for record in records:
        if record.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample ID: {record.sample_id}")
        sample_ids.add(record.sample_id)
        if groups.setdefault(record.split_group, record.split) != record.split:
            raise ValueError(f"Split leakage in group {record.split_group}")
        for chain in record.structure.chains:
            identity = (chain.accession, chain.taxon_id)
            assignment = (record.split_group, record.split)
            if proteins.setdefault(identity, assignment) != assignment:
                raise ValueError(
                    f"Protein identity crosses groups or splits: {identity}"
                )
            if sequences.setdefault(chain.sequence_hash, assignment) != assignment:
                raise ValueError("Identical protein sequences cross groups or splits")
        structure = (record.structure.structure_hash, record.structure.assembly_id)
        if structures.setdefault(structure, record.split_group) != record.split_group:
            raise ValueError("The same assembly crosses protein groups")
