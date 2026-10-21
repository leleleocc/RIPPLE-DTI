"""Versioned, conservative chemical identities for source staging tables."""

import hashlib
import json
import sqlite3
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

STANDARDIZATION_VERSION = "ripple-parent-v1"
RDLogger.DisableLog("rdApp.*")
# Common carbon-free counterions and hydrate water. Other disconnected
# components require a reviewed mixture/coordination decision.
COUNTERIONS = {
    "[Na+]",
    "[K+]",
    "[Li+]",
    "[Ca+2]",
    "[Mg+2]",
    "[NH4+]",
    "[F-]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "F",
    "Cl",
    "Br",
    "I",
    "O",
    "O=S(=O)([O-])[O-]",
    "O=S(=O)([O-])O",
    "O=S(=O)(O)O",
    "O=P([O-])([O-])[O-]",
    "O=P([O-])([O-])O",
    "O=P([O-])(O)O",
    "O=P(O)(O)O",
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def standardize_one(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        return {"status": "invalid_smiles"}
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        return {"status": "unspecified_atom"}
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    if not organic:
        return {"status": "inorganic"}
    # Multiple organic components need a reviewed salt/mixture decision.
    # Selecting the largest would silently merge combination drugs with a parent.
    if len(organic) != 1:
        return {"status": "multiple_organic_fragments"}
    inorganic = [
        m for m in fragments if not any(a.GetAtomicNum() == 6 for a in m.GetAtoms())
    ]
    if any(
        Chem.MolToSmiles(m, isomericSmiles=True) not in COUNTERIONS for m in inorganic
    ):
        return {"status": "unsupported_inorganic_fragment"}
    parent = rdMolStandardize.Cleanup(organic[0])
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.SanitizeMol(parent)
    Chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(parent, isomericSmiles=True)
    key = Chem.MolToInchiKey(parent)
    if not key:
        return {"status": "inchi_failed"}
    return {
        "status": "ok",
        "canonical_smiles": canonical,
        "inchikey": key,
        "fragments_removed": len(fragments) - 1,
        "stereo_centers": sum(
            a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            for a in parent.GetAtoms()
        )
        + sum(b.GetStereo() != Chem.BondStereo.STEREONONE for b in parent.GetBonds()),
        "connectivity_smiles": Chem.MolToSmiles(parent, isomericSmiles=False),
    }


def standardize(item):
    input_id, raw_smiles = item
    try:
        if not raw_smiles.strip():
            result = {"status": "missing_smiles"}
        else:
            variants = [standardize_one(x.strip()) for x in raw_smiles.split(";")]
            failures = {x["status"] for x in variants if x["status"] != "ok"}
            if failures:
                result = {"status": min(failures)}
            elif len({x["connectivity_smiles"] for x in variants}) != 1:
                result = {"status": "conflicting_connectivity"}
            else:
                most = max(x["stereo_centers"] for x in variants)
                best = [x for x in variants if x["stereo_centers"] == most]
                if len({x["canonical_smiles"] for x in best}) != 1:
                    result = {"status": "conflicting_stereochemistry"}
                else:
                    result = best[0]
                    if len(variants) > 1:
                        chosen = Chem.MolFromSmiles(result["canonical_smiles"])
                        if any(
                            not chosen.HasSubstructMatch(
                                Chem.MolFromSmiles(v["canonical_smiles"]),
                                useChirality=True,
                            )
                            for v in variants
                        ):
                            result = {"status": "conflicting_stereochemistry"}
    except (ValueError, RuntimeError) as exc:
        result = {"status": "standardization_error", "error": str(exc)[:250]}
    return input_id, result


class ChemicalRegistry:
    """Durable input cache and many-to-many source crosswalk, all outside raw."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS inputs (
                input_id TEXT PRIMARY KEY, raw_smiles TEXT NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS xrefs (
                source TEXT, source_id TEXT, input_id TEXT,
                PRIMARY KEY (source, source_id, input_id)
            );
        """)

    @lru_cache(maxsize=100000)  # noqa: B019 - explicitly cleared by close().
    def register(self, source, source_id, smiles):
        input_id = digest(STANDARDIZATION_VERSION + "\n" + smiles.strip())
        self.db.execute(
            "INSERT OR IGNORE INTO inputs VALUES (?, ?, NULL)",
            (input_id, smiles.strip()),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO xrefs VALUES (?, ?, ?)",
            (source, source_id, input_id),
        )
        return input_id

    def result(self, input_id):
        value = self.db.execute(
            "SELECT result FROM inputs WHERE input_id=?", (input_id,)
        ).fetchone()
        return json.loads(value[0]) if value and value[0] else {}

    def close(self):
        self.db.commit()
        self.db.close()
        self.register.cache_clear()
