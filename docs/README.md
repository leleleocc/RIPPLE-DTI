# Documentation Map

Updated: 2026-10-16. Start with [design.md](design.md) for the Chinese project
context: the research question, accepted scope, data roles and actual progress.

The current objective is small-molecule modulation of ion channels, extending
the original DTI/pocket/residue framework to conformational and functional
response. Explicit membrane modeling is deferred. There is no trained channel
gating predictor or completed production training run yet.

## Reading Order

| Read | Document | Purpose |
|---|---|---|
| 1 | [Research context and design](design.md) | Why the objective changed, shared terminology, supervision boundaries and current status |
| 2 | [Five-source audit](DATA_ALIGNMENT_RESULTS.md) | What is downloaded/aligned, what counts mean, and remaining coverage gaps |
| 3 | [Channel extension](CHANNEL_EXTENSION.md) | Additional official sources, the 8 structures, geometry labels and Mac CPU smoke run |
| 4 | [Network contract](NETWORK_MODULES.md) | Actual interfaces in `src/ripple/`, masks, outputs and implemented stage policies |
| 5 | [Implementation plan](IMPLEMENTATION_PLAN.md) | Completed work, open decisions and concrete next deliverables |
| As needed | [Data preparation](DATA_PREPARATION.md) | Commands, staging tables and residue-mapping rules |

## Status Vocabulary

| Wording | Required evidence |
|---|---|
| Downloaded | Local raw files with identifiable sources |
| Aligned | Checked identities/residue mappings within the stated subset |
| Implemented | Model/data/loss code exists; this alone says nothing about prediction quality |
| Smoke checked | Small execution or gradient check, with its actual configuration stated |
| Trained and evaluated | A defined dataset/split, reproducible run, checkpoint and held-out metrics |

The current geometry smoke run used a local Mac CPU, 8 structures and 64 updates
of a small structure subgraph. It did not train ligand-induced response, gating
or state preference. The full model also had one forward/backward check without
an optimizer update. See the channel document before citing its training error.

## Evidence And Local Files

Generated data and run artifacts are git-ignored and may be absent in a fresh
checkout. Reproduction commands are in the corresponding data documents.

- Five-source counts: `data/processed/alignment_summary.json` and `DATA_REPORT.md`.
- Channel counts and provenance: `data/processed/channels/summary.json` and `provenance.json`.
- Local smoke record: `outputs/channel_geometry_smoke/smoke_report.json`.
- Source eligibility: `src/ripple/data/schema.py`.
- Current source and batch interfaces: `src/ripple/model.py` and `training.py`.

`overview.html` is a historical illustration of the base DTI design. Its scope
does not include the ion-channel extension. The current design and network
contract replace earlier hard-gating, conformer-label and path descriptions.
