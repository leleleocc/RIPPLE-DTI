# RIPPLE-DTI

**R**esidue **I**nteraction **P**ropagation from **P**ocket to **L**ong-range
**E**ffects for drug-target interaction prediction.

Pocket-aware DTI whose interaction state is conditioned on candidate pockets,
with ligand-conditioned propagation from the binding pocket to annotated
functional regions. Pose confidence is a structural auxiliary: it re-ranks
poses and modulates the interaction state, but never gates the prediction.

## Contributions

```text
SMILES + protein sequence/structure + candidate pockets
                         |
            pocket-conditioned interaction      (primary)
                         |
        pose confidence, soft modulation only   (auxiliary)
                         |
   pocket-to-function propagation on the        (primary)
   residue graph
                         |
       DTI / affinity / effect / mechanism
```

- Candidate pockets are generated without test ligand coordinates, and pocket
  selection is evaluated alongside DTI under cold-target, cold-pocket, and
  cold-both splits.
- Propagation predicts functional, allosteric, and disease-associated residues
  from the pocket-conditioned state. Annotations supervise that prediction;
  they are never fed in as inputs.
- Pose quality is trained only where an RMSD-to-crystal label exists
  (PDBbind, CrossDocked2020, CASF) and contributes nothing otherwise.

## Layout

```text
src/ripple/model.py            assembles the three modules
src/ripple/interaction.py      pocket-conditioned drug-protein fusion
src/ripple/propagation.py      pocket-to-function propagation
src/ripple/pose_confidence.py  pose re-ranking and soft modulation
src/ripple/training.py         masked multi-task loss and curriculum
src/ripple/train.py            training entry point
src/ripple/evaluate.py         metrics reported per contribution
src/ripple/data/schema.py      which source may supervise which label
scripts/ripple/                source parsing, cache, cold splits
configs/ripple_network.yaml    model and loss weights
```

## Training

The curriculum is `pocket -> dti -> propagation -> pharmacology -> joint`.
Each stage enables only the losses that have supervision, and a sample
contributes to a loss only when both its source and its record carry that label.

```bash
python -m src.ripple.train \
  --config configs/ripple_network.yaml \
  --cache cache/ripple \
  --split cold-target \
  --save-dir outputs/ripple
```

## Tests

```bash
python -m pytest -q
```

The tests check that the DTI path stays finite when no pose is available, that
functional annotations do not leak into the propagation sink, and that the
function and pose losses contribute nothing on samples without those labels.
