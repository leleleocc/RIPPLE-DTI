# RIPPLE-DTI Network and Training Contract

The legacy `LDMDTI` and `BINDTI` classes remain unchanged for baseline
reproduction. `RIPPLEDTI` reuses their representation types as one shared
pipeline rather than retaining a second, complete TAMR prediction branch.

## Data Flow

```text
ChemBERTa 1D tokens + MolecularGCN 2D tokens
                    |
             chemical context
                    |
multi-conformer EGNN + fused protein residues/pockets
                    |
       joint pocket-conformer matching
                    |
      selected 3D + 1D + 2D drug tokens
                    |
         pocket-conditioned BiIntention
                    |
        interaction-modulated pocket source
                    |
     ligand-off/on differential propagation
                    |
   DTI / affinity / effect / mechanism heads
```

The order is intentional:

1. Sequence and structural residue features are fused into one ligand-off
   state `H_apo`. Pocket extraction, protein resampling, and propagation all
   consume this same state.
2. The joint matcher uses 1D/2D chemical context but preserves every 3D
   conformer until it produces the `[Kp,Kc]` posterior.
3. Only the posterior-selected 3D representation is fused with 1D and 2D drug
   tokens. No positional alignment between those modalities is assumed.
4. The selected local interaction conditions both BiIntention inputs. Its
   global output then modulates the local propagation source.
5. If no valid pocket-conformer pair exists, the learned interaction context
   and ligand-induced residue difference are zero. There is no complete global
   prediction path that can bypass matching.

## Input Contract

`RIPPLEBatch` uses padded tensors with explicit masks:

| Field | Shape | Meaning |
|---|---|---|
| `drug_1d_tokens` | `[B,N1,D]` | ChemBERTa token representation |
| `drug_1d_mask` | `[B,N1]` | Valid ChemBERTa tokens |
| `drug_2d_tokens` | `[B,N2,D]` | MolecularGCN node representation |
| `drug_2d_mask` | `[B,N2]` | Real molecular graph nodes |
| `atom_features` | `[B,Kc,Na,Fa]` | Real heavy-atom features per conformer |
| `atom_coords` | `[B,Kc,Na,3]` | Conformer coordinates |
| `atom_mask` | `[B,Kc,Na]` | Valid conformer atoms |
| `conformer_mask` | `[B,Kc]` | Valid conformers |
| `conformer_energy` | `[B,Kc]` | Normalized conformer energy |
| `residue_features` | `[B,L,Fr]` | Unpooled PLM residue features |
| `protein_sequence_tokens` | `[B,L,D]` | Optional ProteinACmix output |
| `residue_coords` | `[B,L,3]` | Residue coordinates |
| `residue_mask` | `[B,L]` | Structurally mapped residues |
| `edge_index` | `[B,2,E]` | Sparse residue graph edges |
| `edge_mask` | `[B,E]` | Valid edges |
| `pocket_indices` | `[B,Kp,Np]` | Pocket-to-protein residue indices |
| `pocket_mask` | `[B,Kp,Np]` | Valid pocket residues |
| `pocket_valid_mask` | `[B,Kp]` | Valid candidate pockets |
| `pocket_score` | `[B,Kp]` | External pocket confidence |
| `function_mask` | `[B,L]` | Optional functional residues |

The ChemBERTa, MolecularGCN, and optional ProteinACmix outputs are produced by
the retained TAMR encoders. They are shared inputs, not outputs of a separate
TAMR classifier.

## Outputs

`RIPPLEOutput` exposes task predictions and the intermediate quantities needed
for ablations:

- joint pocket-conformer probabilities and their marginals;
- the shared ligand-off residue state and sequence/structure gate;
- pocket-conditioned BiIntention state;
- local and interaction-modulated residue sources;
- ligand-on/off difference, functional sink weights, and edge flux.

The edge flux is learned predictive information flow, not a physical
energy-transfer trajectory.

## Training Curriculum

`ripple_training.py` provides masked multi-task losses and stage freezing:

| Stage | Trainable purpose | Active supervision |
|---|---|---|
| `structure` | Geometry encoders and joint matcher | Native pocket, bound conformer, allosteric pocket |
| `dti` | Modal fusion and conditioned interaction | DTI, affinity, native pocket |
| `propagation` | Source fusion and sparse diffusion | Effect direction, nonbinder delta |
| `pharmacology` | Propagation and pharmacology heads | Effect, mechanism, nonbinder delta |
| `joint` | Low-rate end-to-end refinement | Every available label |

Every label has an independent mask. A batch from BindingDB, PDBbind,
AlloBench, or GtoPdb contributes only the objectives it actually contains.
Shared geometry encoders use a lower optimizer learning rate during joint
training.
