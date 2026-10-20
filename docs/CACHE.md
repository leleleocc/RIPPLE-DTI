# Manifest And Tensor Cache V1

The training entry point reads `manifest.jsonl` under `--cache`. Each line is a
`SampleRecord` from `src/ripple/data/manifest.py`. The manifest contains all
partitions for both stages and is frozen before training. A checkpoint records
its SHA-256 and rejects a different manifest during stage transfer, resume or
evaluation. This validates declared assignments; sequence clustering, ortholog
grouping and PIEZO exclusion still require an audited upstream split procedure.

## Sample Records

| Field | Contract |
|---|---|
| `schema_version` | `1` |
| `sample_id` | Unique across all sources and partitions |
| `source` | Registered source, e.g. `bindingdb`, `pdbbind`, `biolip2`, `gtopharm`, `channel_structures` |
| `task` | `measurement`, `site`, `complex`, `effect` or `geometry` |
| `split` | `train`, `val`, `test` or `external` |
| `split_group` | Frozen sequence/ortholog group, shared by all related structures |
| `protein_cache` | Path to a protein `.pt` file, relative to the cache directory |
| `structure` | PDB ID, assembly ID, structure hash, mapping version and ordered chain list |
| `drug_cache`, `inchikey` | Drug `.pt` path and standardized InChIKey; both null for geometry |
| `endpoint_type` | Kd=0, Ki=1, IC50=2, EC50=3; absent/unknown does not supervise regression |
| `labels` | Only measured task targets and their optional explicit masks |
| `loss_enabled` | Evidence permissions intersected with the registered source's permissions |
| `evidence` | Requires `source_version` and `record_ids`; retain references, units, relation symbols and assay conditions here |
| `geometry_queries` | Frozen `indices: [M,P,2]` and `pair_mask: [M,P]`, at most 64 active pairs |

`structure.chains` contains one entry per assembly chain copy:
`chain_id`, `accession`, `taxon_id`, `sequence_hash`, `canonical_length`,
`construct_hash`. Two copies of the same protein have distinct chain IDs.
Sequence and construct hashes describe the actual inputs. Isoform accession,
mutations and construct differences remain traceable in the source evidence.

The main label names are `affinity`, `native_pocket_index`, `effect` and
`geometry_snapshot`. Effect encoding is activation=0, inhibition=1. Unknown
directions carry no effect target. All four numeric endpoints use the frozen
train-time units and transformation, normally `-log10(value in mol/L)`; the loader
does not infer units or convert source measurements. Explicit masks use the names
in `RIPPLELabels`, for example `affinity_mask` and `native_pocket_mask`.

Permission keys follow `LabelMasks`: `affinity`, `pocket`, `effect`,
`geometry_snapshot`, etc. The demo's `native_pocket` key is accepted as an alias
for `pocket`. Source and per-record permissions must both allow supervision.
Native pocket index `-1` records an annotated site outside the candidates; it is
excluded from ranking loss and counted when reporting candidate coverage.

Replay retains the original source, sample ID and labels. It selects only
foundation tasks from `train`; it is not a new evidence source. Action, Type,
references and assay measurements never become protein/drug cache features.

## Tensor Files

Files are plain dictionaries saved with `torch.save` and loaded with
`weights_only=True`. Only tensors and primitive metadata belong in them.
Every file requires `cache_version: 1`, `feature_version`, and `tensors`.
`feature_version` identifies the feature model revision, tokenizer, windowing,
conformer generation and other preprocessing settings. Cache filenames should
include these versions or content hashes so a preprocessing change cannot
silently reuse an older file. The training manifest hash does not hash tensor
file contents; cache artifacts must remain immutable during a run.

Protein files additionally contain `structure_hash`, `mapping_version`, `chains`
(exactly the manifest's chain dictionaries), `residue_keys`, and `residue_mapping`.
`residue_keys` lists `[chain_id, canonical_index]` in chain order, with a **zero-based**
canonical index for every position in each chain's full sequence. For a homotrimer
with sequence length L, tensors have N=3L nodes. Missing coordinates keep their
node positions and have `residue_mask=False`; no sequence truncation occurs.

`residue_mapping` has one entry per node, null for unmapped positions. Each valid
coordinate requires `auth_asym_id`, `auth_seq_id`, `insertion_code`, `label_asym_id`
and `label_seq_id`. Additional construct and mutation annotations may be retained.
This allows predictions to be exported back to source chain/residue numbering.

| Protein tensor | Unbatched shape |
|---|---|
| `residue_features` | `[N, residue_input_dim]` |
| `residue_coords` | `[N, 3]`, Angstrom |
| `residue_mask` | `[N]`, bool: mapped and observed C-alpha |
| `edge_index`, `edge_mask` | `[2,E]` int64, `[E]` bool; node indices within this assembly |
| `edge_features` | Optional `[E, edge_dim]` |
| `protein_sequence_tokens` | Optional `[N,D]`; omit to train the model's projection from frozen residue features |
| `pocket_indices`, `pocket_mask` | `[K,R]` int64/bool; independent detector candidates |
| `pocket_valid_mask`, `pocket_score` | `[K]` bool/float |

Geometry-only samples use the protein fields without requiring candidate pockets.
Missing coordinates may be NaN in cache; the model sanitizes masked positions
before graph arithmetic. Candidate pockets must be generated without the test
ligand or its contact labels. Source numbering is metadata, never a graph index.

Drug files additionally contain `inchikey` and `standardization_version`.
Chemical standardization must preserve stereochemistry, retain raw identity and
quarantine ambiguous mixtures or unsupported compounds before creating a record.

| Drug tensor | Unbatched shape |
|---|---|
| `atom_features` | `[C,A,atom_input_dim]` |
| `atom_coords`, `atom_mask` | `[C,A,3]`, `[C,A]` bool |
| `conformer_mask`, `conformer_energy` | `[C]` bool/float |
| `drug_1d_tokens`, `drug_1d_mask` | `[T1,F1]`, `[T1]` bool |
| `drug_2d_tokens`, `drug_2d_mask` | `[T2,F2]`, `[T2]` bool |

Feature dimensions must be consistent across the dataset. Learned projections
accept F1/F2 and map them to model dimension D. The dataset adds a leading dimension
of one; `collate_ripple` pads variable lengths and retains sample-local edge,
pocket and geometry indices. Geometry and drug samples use different batches.
Whole missing labels remain `None`; within mixed batches missing targets are
masked with `False`, with NaN/-1 used only as storage sentinels.

## Geometry Queries

`sample_geometry_queries` in `src/ripple/data/geometry.py` generates reproducible
local and cross-chain pairs from mapped nodes after a partition is assigned.
Persist its queries and distances in the sample record, along with seed and
sampling settings in evidence. The training loader never resamples validation or
test queries. Each output query has P=1; the readout also supports averaging an
explicit group of symmetry-related pairs. These targets describe the input
reference structure and do not supervise drug-induced changes.

## Execution

With an audited manifest and actual feature caches prepared:

```console
python -m src.ripple.train --cache data/cache/ripple --device cpu --batch-size 2 --accumulation 8
torchrun --nproc-per-node 4 --master-addr 127.0.0.1 --master-port 29500 --module src.ripple.train --cache data/cache/ripple --device cuda --amp
python -m src.ripple.evaluate --cache data/cache/ripple --checkpoint outputs/ripple/cold-target/channel_finetune/best.pt --split test --output outputs/ripple/test_metrics.json
```

Use a free rendezvous port. Each rank executes the same task schedule. The default
schedule is foundation, five effect warmup traversals, then up to 30 effect
traversals with geometry and replay. Set a missing auxiliary task's ratio to zero
explicitly; a positive ratio with no records is an error. All-mask task batches
are errors as well. Loss weights and task ratios are applied once each.

Warmup trains source fusion, propagation and the effect head. Fine-tuning uses
1e-4 for the effect/snapshot heads and 2e-5 for other enabled modules. Reliable
interaction training requires both `RELIABLE_INTERACTION: true` and a positive
`LOSS_WEIGHTS.INTERACTION`, plus audited positive/negative labels in records.

Every stage writes `best.pt`, an epoch-boundary `last.pt`, and `metrics.jsonl`.
Foundation validation also runs every 1,000 optimizer steps; intermediate best
checkpoints can initialize the next stage but are not resume checkpoints.

```console
python -m src.ripple.train --cache data/cache/ripple --stage channel_finetune --checkpoint outputs/ripple/cold-target/effect_warmup/best.pt
python -m src.ripple.train --cache data/cache/ripple --stage channel_finetune --resume outputs/ripple/cold-target/channel_finetune/last.pt
```

Resume requires the original output directory, manifest, model/training config,
seed, device precision, world size and batch settings. A fresh run refuses to
overwrite existing checkpoints. Evaluation uses the complete selected partition
without class resampling, and reports unavailable metrics as null. Checkpoint
selection rejects an unestimable primary validation metric rather than changing
the partition. Actual four-GPU CUDA/AMP throughput and memory still need testing
on the training server; local regression tests exercise CPU and two-process Gloo.
