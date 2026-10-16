# Data Alignment

Scope: the original BindingDB, BioLiP, PDBbind, UniProt and SIFTS downloads.
Start with [design.md](design.md) for the research context and
[DATA_ALIGNMENT_RESULTS.md](DATA_ALIGNMENT_RESULTS.md) for audited counts.
Additional channel data and its separate local smoke run are documented in
[CHANNEL_EXTENSION.md](CHANNEL_EXTENSION.md).

The five downloaded sources are aligned with:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-data.txt
.venv/bin/python scripts/ripple/prepare_sources.py --mapping-limit 300
```

Use `--resume` to reuse completed BioLiP/BindingDB metadata stages when inputs
have not changed. Use `--offline` to verify only cached SIFTS/RCSB mappings.
`--mapping-limit` limits distinct complexes for residue verification; it never
limits the full-source metadata scan. Increasing it extends the verified subset.

All generated tables and reports are in `data/processed/` and are git-ignored.
Downloaded mapping files are cached under `data/raw/sifts/residue_xml/` or
`data/raw/sifts/rcsb/`; existing source files are never modified.

| Artifact | Meaning |
|---|---|
| `DATA_REPORT.md` | Counts, filters, mapping results, and remaining work |
| `alignment_summary.json` | Machine-readable report and processing rules |
| `proteins.parquet` | Valid UniProt full-length sequences and annotations |
| `protein_coverage.parquet` | Source coverage per protein |
| `uniprot/missing_accessions.txt` | Missing/invalid sequences, including accession-only placeholders |
| `uniprot/function_sites.parquet` | Explicit binding/active/transmembrane positions, kept as separate types |
| `bindingdb/measurements.parquet` | Source measurements, separate endpoints and retained censoring relations |
| `biolip2/sites.parquet` | Protein identities, normalized ligands, author-numbered site labels |
| `biolip2/ligands.parquet` | SMILES selection outcomes, including ambiguities |
| `pdbbind/complexes.parquet` | Affinity index, target identity, and quality flags |
| `sifts/segments.parquet` | Full-source chain identity/segment information |
| `sifts/residue_mapping.parquet` | Explicit residue maps for the verification subset |
| `pilot/native_pockets.parquet` | Recalculated ligand contacts and mapping completeness |
| `pilot/biolip_sites.parquet` | Verified BioLiP labels; `label_available` requires complete mapping |
| `manifests/` | Full-source mapping/structure retrieval lists |

Residue indices are one-based UniProt positions on disk. Author numbering and
insertion codes are preserved during mapping. The XML mapping is preferred; the
fallback uses RCSB alignments explicitly marked as SIFTS, composed with the
structure's residue-number table. Every mapped label is checked against the
UniProt amino acid. Failed mappings stay unavailable.

These are source-preserving staging tables, not the final `samples.parquet`
training contract. BindingDB ligands still need chemical validation, activity
thresholds are not chosen, and measured constructs/mutations need checking.
Candidate counts exclude direct PIEZO accessions but are not homolog-audited.
No embeddings, predicted pockets, or training splits are generated here.
These sources support the base binding/site tasks; their alignment does not
create ligand-induced gating or state-preference labels. The separate channel
geometry loader is not a completed generic `RIPPLEDataset` feature cache.

The original UniProt download notes claim zero missing records. The alignment
audit checks actual sequence content and writes its own missing list; it does
not overwrite the raw query lists or download notes.

Tests for parsing and label availability:

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -q tests/test_source_alignment.py
```
