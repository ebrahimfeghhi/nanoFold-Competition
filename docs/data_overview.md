# Data Overview

A plain-language walkthrough of where the training data comes from, what proteins are included, and how raw files become model inputs.

## Where the Data Comes From

Three sources feed the pipeline:

**OpenProteinSet / OpenFold (S3)**
The primary source. OpenProteinSet is a public, precomputed dataset built by the OpenFold team to replicate AlphaFold2's training data at scale. It lives in an anonymous S3 bucket (`s3://openfold`) and contains:
- `chain_data_cache.json` — metadata for every chain (sequence, length, resolution, oligomeric state)
- `duplicate_pdb_chains.txt` — maps chains to their representative equivalents when duplicates exist
- Per-chain MSA files (`uniref90_hits.a3m`) — multiple sequence alignments against UniRef90, precomputed so participants don't need to run their own MSA search

Using a fixed, shared MSA source is intentional: MSA generation is expensive, non-deterministic across database snapshots, and would be a leakage surface if participants could retrieve extra evolutionary information.

**RCSB PDB (mmCIF files)**
Atomic coordinates come from RCSB via `https://files.rcsb.org/download/<PDB>.cif`. mmCIF is the modern PDB archival format and is used to extract atom14 coordinate labels — the 14-atom-per-residue representation used for both supervision and scoring.

**Structural classification databases (for split balancing only)**
CATH, SCOPe, and ECOD provide broad domain/fold annotations that are used when constructing the train/val split to avoid overrepresenting any one structural family. These are not model inputs.

---

## What Proteins Are Included

Not all chains in the PDB make it into the dataset. A chain must pass every filter below:

| Filter | Threshold | Why |
|---|---|---|
| Sequence length | 40–256 residues | Matches the official crop size; keeps examples in the single-domain/small-chain regime |
| Resolution | ≤ 3.0 Å (when known) | Reduces coordinate noise in atom14 labels |
| Oligomeric state | Monomer only | Avoids chains whose conformation depends on missing binding partners |
| Amino acid vocabulary | 20 standard AAs only | Keeps the input/output vocabulary clean |
| OpenFold MSA availability | Must have `uniref90_hits.a3m` | Ensures every participant gets the same input distribution |
| Label processability | Atom14 projection must pass quality thresholds | Ensures the coordinate labels are reliable |

After filtering, chains are clustered at 30% sequence identity / 80% coverage using MMseqs2, and entire PDB entries are kept together. This prevents close homologs or same-experiment chains from appearing in both train and validation.

**Final split sizes:**
- Train: 10,000 chains
- Public validation: 1,000 chains
- Hidden validation: 1,000 chains (held out by maintainers)

The distribution is balanced across secondary structure class (alpha, beta, alpha/beta, coil), domain architecture, length bin, and resolution bin using Jensen-Shannon divergence as the quality metric.

---

## Preprocessing Steps

Preprocessing converts per-chain raw files into the NPZ tensors that the data loader consumes. It is handled by `scripts/preprocess.py` and runs once per chain.

**For each chain:**

1. **Locate the OpenProteinSet directory** for the encoded chain ID
2. **Read the A3M file** (`uniref90_hits.a3m`)
3. **Clean the MSA** — remove query-gap columns, merge and deduplicate rows, cap depth with `--max-msa-seqs`
4. **Load the mmCIF file** from RCSB
5. **Extract atom14 coordinates** — map PDB atom records into the canonical 14-atom-per-residue layout
6. **Align sequences** — align the mmCIF structure sequence to the MSA query sequence
7. **Project coordinates** — map atom14 positions onto query sequence positions
8. **Quality check** — reject chains that fail any of these thresholds:
   - Sequence identity ≥ 90%
   - Coverage ≥ 70%
   - Aligned fraction ≥ 70%
   - At least 32 valid Cα atoms
9. **Write two NPZ files** per chain:
   - `data/processed_features/<chain_id>.npz` — MSA, residue indices, template slots (empty for this track)
   - `data/processed_labels/<chain_id>.npz` — Cα coordinates, atom14 coordinates and masks, resolution

**Feature NPZ contents:**

| Key | Shape | Description |
|---|---|---|
| `aatype` | `(L,)` | Amino acid IDs (vocabulary: `ARNDCQEGHILKMFPSTWYV`) |
| `msa` | `(N, L)` | MSA rows after gap removal |
| `deletions` | `(N, L)` | A3M deletion counts |
| `residue_index` | `(L,)` | Positions 0..L-1 |
| `template_*` | `(0, L, ...)` | Empty — templates are disabled in this track |

**Label NPZ contents:**

| Key | Shape | Description |
|---|---|---|
| `ca_coords` | `(L, 3)` | True Cα coordinates |
| `ca_mask` | `(L,)` | Mask for present Cα atoms |
| `atom14_positions` | `(L, 14, 3)` | Full atom14 coordinates |
| `atom14_mask` | `(L, 14)` | Mask for present atoms |
| `resolution` | scalar | Crystal structure resolution in Å |

---

## Atom14 Layout

The 14 slots per residue follow AlphaFold2 convention:

```
slot 0:  N
slot 1:  CA
slot 2:  C
slot 3:  O
slot 4:  CB (when present)
slots 5–13: residue-specific side-chain atoms
```

The canonical slot order for each amino acid is defined in `nanofold/residue_constants.py`. Glycine has no CB; unused slots are zero-padded and masked out.

---

## What Is Not in This Track

The official track intentionally excludes:
- Template lookup (T=0 throughout)
- External MSA/database retrieval beyond the fixed OpenProteinSet files
- Protein language model embeddings
- Multimers, ligands, non-standard amino acids, or chains longer than 256 residues
