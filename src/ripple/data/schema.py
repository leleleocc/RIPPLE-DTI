"""Label contract shared by every RIPPLE data source.

Sources are never merged into one wide table. Each source emits the columns it
actually has, and a missing label stays missing: it is never coerced to zero.
The masks below decide which loss terms a sample contributes to.
"""

from dataclasses import dataclass
from typing import Optional

# Affinity endpoints are never pooled into one regression target.
ENDPOINT_KD = 0
ENDPOINT_KI = 1
ENDPOINT_IC50 = 2
ENDPOINT_EC50 = 3
ENDPOINT_UNKNOWN = 4

# effect: agonist/activator, antagonist/inhibitor, other.
# mechanism: inhibitor, blocker, opener, modulator, agonist, other.
# Mappings from Guide to Pharmacology and ChEMBL live in configs, not here.


@dataclass
class SourceTable:
    """Columns a single source contributes. None means the source has no such label."""

    name: str
    interaction: bool = False
    affinity: bool = False
    pocket: bool = False
    pose: bool = False
    function: bool = False
    effect: bool = False
    mechanism: bool = False
    allosteric: bool = False
    geometry_snapshot: bool = False
    geometry_response: bool = False
    gating: bool = False
    state_preference: bool = False


# What each upstream database is allowed to supervise. A source not listed here
# cannot silently gain a label by being joined in.
BINDINGDB = SourceTable("bindingdb", interaction=True, affinity=True)
CHEMBL = SourceTable("chembl", interaction=True, affinity=True, mechanism=True)
BIOSNAP = SourceTable("biosnap", interaction=True)
PDBBIND = SourceTable("pdbbind", affinity=True, pocket=True, pose=True)
BIOLIP2 = SourceTable("biolip2", pocket=True, function=True)
UNIPROT = SourceTable("uniprot", function=True)
CROSSDOCKED = SourceTable("crossdocked", pose=True)
CASF = SourceTable("casf", pose=True)
GTOPHARM = SourceTable("gtopharm", effect=True, mechanism=True)
ASD = SourceTable("asd", allosteric=True)
CHANNEL_STRUCTURES = SourceTable("channel_structures", geometry_snapshot=True)
# These sources require separately curated intervention/state-specific evidence.
CHANNEL_INTERVENTIONS = SourceTable("channel_interventions", geometry_response=True, gating=True)
STATE_SPECIFIC_BINDING = SourceTable("state_specific_binding", state_preference=True)


@dataclass
class LabelMasks:
    """Per-sample availability. True means the loss term may use this sample."""

    interaction: Optional[bool] = None
    affinity: Optional[bool] = None
    pocket: Optional[bool] = None
    pose: Optional[bool] = None
    function: Optional[bool] = None
    effect: Optional[bool] = None
    mechanism: Optional[bool] = None
    allosteric: Optional[bool] = None
    geometry_snapshot: Optional[bool] = None
    geometry_response: Optional[bool] = None
    gating: Optional[bool] = None
    state_preference: Optional[bool] = None

    def allows(self, source: SourceTable, label: str) -> bool:
        """A label counts only when both the source and the sample carry it."""
        return bool(getattr(source, label)) and bool(getattr(self, label))
