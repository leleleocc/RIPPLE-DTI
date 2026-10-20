# RIPPLE-DTI

**R**esidue **I**nteraction **P**ropagation from **P**ocket to **L**ong-range
**E**ffects for drug-target interaction prediction.

The research objective is small-molecule modulation of ion-channel proteins:
where a ligand binds, how it relates to conformational changes, and how those
changes affect function. Explicit membrane modeling is deferred.

The original pocket-aware DTI and residue-propagation model remains the base.
The channel extension adds geometry readouts and comparison of multiple states
of the same protein. These branches are implemented; there is no trained or
independently evaluated channel-gating predictor yet.

## Start Here

Read the [model guide](docs/MODEL.md) for inputs, outputs and the two-stage plan.
The [documentation map](docs/README.md) links the three consolidated guides:
data, model, and training strategy with scripts.

| Area | Actual status as of 2026-10-20 |
|---|---|
| Base data | Five-source metadata audit; 300 complexes checked at residue level |
| Channel pilot | 8 structures checked, 16 geometry values, 10 structural differences, 103 pharmacology records |
| Model | Base interaction/propagation and channel readouts implemented |
| Local execution | Mac CPU, small 32-dimensional model, 64 updates of the structure subgraph only |
| Training infrastructure | Two-stage training, sparse-label batching, checkpoint resume and held-out evaluation implemented; CPU and two-process checks available |
| Formal experiments | Audited splits and real feature caches still pending; four-GPU CUDA/AMP not yet measured |

The local check learned to reconstruct current geometry. It did not train
ligand-induced changes or channel opening/closing. A separate full-model
forward/backward check performed no optimizer update. See the
[data guide](docs/DATA.md) for sources and historical run details.

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

- Candidate pockets must be generated without test ligand coordinates. Pocket
  selection and DTI evaluation under cold-start protocols are planned.
- Propagation provides residue-level functional-site readouts. Annotations
  supervise the prediction; they are not read as inputs. Site-type selection and
  formal training remain to be completed.
- Pose ranking contributes only where an appropriate RMSD-to-crystal ordering
  exists; it is a structural auxiliary and does not gate DTI prediction.

## Layout

```text
src/ripple/model.py            assembles the base and channel modules
src/ripple/interaction.py      pocket-conditioned drug-protein fusion
src/ripple/propagation.py      pocket-to-function propagation
src/ripple/pose_confidence.py  pose re-ranking and soft modulation
src/ripple/training.py         masked multi-task loss and curriculum
src/ripple/structure_response.py geometry readouts and shared-weight state comparison
src/ripple/data/channels.py    observed-geometry pilot loader
src/ripple/train.py            two-stage training, checkpoint transfer and resume
src/ripple/evaluate.py         frozen-partition evaluation CLI
src/ripple/data/manifest.py    versioned identities, evidence and partition checks
src/ripple/data/dataset.py     tensor cache reader and sparse-label collation
src/ripple/data/schema.py      which source may supervise which label
scripts/ripple/                source parsing, cache, cold splits
configs/ripple_network.yaml    model and loss weights
```

## Data Preparation

Align the downloaded BindingDB, BioLiP, PDBbind, UniProt, and SIFTS files:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-data.txt
.venv/bin/python scripts/ripple/prepare_sources.py --mapping-limit 300
```

The full-source audit and selected-complex residue verification write
`data/processed/DATA_REPORT.md`. See the [data guide](docs/DATA.md)
for the output tables, filters, and remaining training prerequisites.

## Training

The plan is base pretraining followed by channel enhancement/inhibition tuning,
with current-geometry reconstruction and base-task replay as auxiliary tasks.
The training loop implements this plan with a warmup for source fusion,
propagation and the effect head. Geometry batches run without drug features.
Read the [cache contract and commands](docs/CACHE.md) and
[training strategy](docs/TRAINING.md). Raw feature extraction and sequence
clustering remain upstream prerequisites; synthetic regression tests do not
establish predictive performance on real data.

## Ion-Channel Pilot

The extension keeps multiple conformations under the same protein identity and
adds invariant geometry readouts, ligand-conditioned structural response and a
shared-weight state comparison. Membrane environment is deferred.

```bash
uv pip install --python .venv/bin/python -r requirements-data.txt torch einops
.venv/bin/python scripts/ripple/prepare_channels.py
.venv/bin/python scripts/ripple/smoke_channel_geometry.py --steps 64
```

The first audit contains eight structures, 16 observed geometry values and 103
pharmacology records. Structural differences are not automatically used as
drug-induced response or state-preference labels. See
[data guide](docs/DATA.md) for verified counts and historical smoke results;
the [training guide](docs/TRAINING.md) lists the remaining prerequisites.

## Tests

```bash
python -m pytest -q
```

The tests check that the DTI path stays finite when no pose is available, that
functional annotations do not leak into the propagation sink, and that the
function and pose losses contribute nothing on samples without those labels.
