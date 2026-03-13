# funcscreen 🧬🛡️

**Benchmarking Function-Based Biosecurity Screening: Can Protein Language Model Embeddings Detect AI-Redesigned Toxins that Evade BLAST?**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/preprint-PREPRINT.md-green)](PREPRINT.md)

---

## Overview

`funcscreen` is a proof-of-concept implementation of the **TEVV (Toxin Embedding Verification & Validation)** framework. It demonstrates that:

1. **ProteinMPNN** can redesign toxin sequences to <30% sequence identity while preserving structural fold (TM-score > 0.5) — making them invisible to BLAST-based biosecurity screening tools.
2. **ESM-2 latent embeddings** detect these structurally-retained "stealth" variants with significantly higher recall than BLAST at low sequence identity.

This addresses a vulnerability identified by Wittmann et al. (2024) and called for explicitly in the March 2026 preprint on alternate biosecurity screening approaches (see [Citations](#citations)).

### Key Result

| Metric | BLAST (blastp, ≥30% identity, E≤1e-3) | ESM-2 LR (5-fold CV) |
|--------|----------------------------------------|----------------------|
| Precision | 1.00 | 1.00 ± 0.00 |
| Recall | 0.50 | 1.00 ± 0.00 |
| F1 | 0.67 | 1.00 ± 0.00 |
| AUROC | — | 1.00 ± 0.00 |

*Variants with <29% sequence recovery from Ricin A-chain, Botulinum neurotoxin type A, and Staphylococcal enterotoxin B were correctly flagged by the ESM-2 classifier with >90% confidence, while BLAST missed 50% of stealth variants.*

---

## Proxy Toxins Used

All sequences are publicly available in UniProt Swiss-Prot. No controlled or restricted sequences are used. Computational work with publicly available sequences does not require BSL handling.

| Toxin | UniProt | Category | HHS/CDC Status |
|-------|---------|----------|---------------|
| Ricin A-chain | P02879 | Ribosome-inactivating protein | Schedule 1 |
| Botulinum neurotoxin A (light chain) | P10844 | Zn-metalloprotease | BSAT Schedule 1 |
| Staphylococcal enterotoxin B | P01552 | Superantigen | Select Agent |

---

## Repository Structure

```text
funcscreen/
├── scripts/
│   └── experimental/          # Archive: Generation & benchmarking
│       ├── build_negative_class.py
│       ├── fetch_multitoxin.py
│       ├── step1.py
│       ├── step2_fold.py
│       ├── blast_baseline.py
│       ├── train_detector.py
│       └── [Other experimental scripts...]
├── ProteinMPNN/               # Submodule for variant generation
├── rebuild_embeddings.py      # Step 1: Prepare latent space
├── embedding_geometry.py      # Step 2: Math/Drift analysis
├── train_detector_cv.py       # Step 3: 5-fold CV classifier
├── evasion_analysis_stratified.py # Step 4: Final Evasion Table
├── final_master_plot.py       # Step 5: Publication UMAPs
├── calculate_tm.py            # Step 6: Structural verification
├── negatives/                 # Documented benign proteins
├── toxin_seeds/               # Wildtype seed sequences
├── variants_output/           # Generated variant repository
├── results/                   # CV metrics and plots
├── PREPRINT.md                # Full manuscript
├── requirements.txt           # Environment config
└── README.md                  # This file

---

### Installation

```bash
git clone https://github.com/FridaArrey/funcscreen.git --recurse-submodules
cd funcscreen
pip install -r requirements.txt
```

### Hardware Requirements

| Step | CPU | GPU | Notes |
|------|-----|-----|-------|
| Sequence fetch (UniProt) | ✓ | — | No special hardware needed |
| ProteinMPNN variant generation | ✓ | Optional | ~30 sec/sequence on CPU |
| ESMFold structure prediction | ✗ | ≥16 GB VRAM | Use `--use_api` flag for hosted API |
| ESM-2 embedding (650M) | ✓ (slow) | ✓ Recommended | ~2 min/seq CPU; ~5 sec/seq GPU |
| BLAST screening | ✓ | — | Requires BLAST+ installed separately |

**BLAST+ installation:**
```bash
# macOS
brew install blast

# Ubuntu/Debian
sudo apt install ncbi-blast+

# Or via conda
conda install -c bioconda blast
```

**Commec (optional, preferred for Step 2):**
```bash
pip install commec
commec download-databases   # ~30 min, downloads HMM + BLAST databases
```

---

## Step-by-Step Execution

### Step 1a — Fetch Toxin Seed Sequences

```bash
python fetch_multitoxin.py --outdir toxin_seeds/
```

Fetches Ricin A-chain (P02879), Botulinum NTx-A (P10844), and Staph Enterotoxin B (P01552) from UniProt Swiss-Prot. Outputs individual FASTA files and `all_toxins.fasta`.

### Step 1b — Build Negative Class

```bash
python build_negative_class.py --outdir negatives/ --n_positives_expected 30
```

Fetches 16 curated benign proteins from Swiss-Prot (cytoskeletal, metabolic, chaperone, transport, and plant structural categories). Outputs `negatives.fasta`, `negatives_metadata.json`, and a plain-English `negative_class_report.txt` suitable for the methods section.

### Step 1c — Generate ProteinMPNN Variants

```bash
# Parse wildtype PDB structures (required for ProteinMPNN)
python ProteinMPNN/helper_scripts/parse_multiple_chains.py \
    --input_path pdb_inputs/ \
    --output_path parsed_pdbs.jsonl

# Generate stealth variants (low sequence recovery, preserving structure)
python step1.py \
    --jsonl_path parsed_pdbs.jsonl \
    --out_folder variants_stealth/ \
    --num_seq_per_target 10 \
    --sampling_temp 0.3

# Generate dud variants (high sequence recovery = diverged structure)
python step1.py \
    --jsonl_path parsed_pdbs.jsonl \
    --out_folder variants_dud/ \
    --num_seq_per_target 5 \
    --sampling_temp 0.1
```

### Step 1d — Fold Variants and Compute TM-Scores

```bash
# Fold stealth variants
python fold_variant_stealth.py --input variants_stealth/seqs/ --output variants_stealth/pdbs/

# Fold dud variants
python fold_dud.py --input variants_dud/seqs/ --output variants_dud/pdbs/

# Compute TM-scores against wildtype
python calculate_tm.py \
    --wildtype_pdb pdb_inputs/ \
    --variant_pdbs variants_stealth/pdbs/ variants_dud/pdbs/ \
    --output tm_scores.json
```

*Stealth variants: expect TM-score > 0.5. Dud variants: expect TM-score < 0.5.*

### Step 2 — BLAST Baseline

```bash
# Build BLAST database from HHS BSAT sequences (download from selectagents.gov)
makeblastdb -in bsat_sequences.fasta -dbtype prot -out bsat_db/bsat

# Screen stealth variants (should be detected — ground truth positive)
python blast_baseline.py --mode blast \
    --query_fasta variants_stealth/seqs/ \
    --db bsat_db/bsat \
    --category stealth \
    --outdir blast_results/

# Screen dud variants (should NOT be detected — ground truth negative)
python blast_baseline.py --mode blast \
    --query_fasta variants_dud/seqs/ \
    --db bsat_db/bsat \
    --category dud \
    --outdir blast_results/ \
    --append

# Alternative: use Commec (preferred for policy-relevant comparison)
python blast_baseline.py --mode commec \
    --query_fasta variants_stealth/seqs/ \
    --category stealth \
    --outdir blast_results/
```

BLAST parameters: `blastp`, E-value ≤ 1e-3, BLOSUM62, identity threshold ≥30%, query coverage ≥50%. All parameters documented in `blast_results/blast_summary.json`.

### Step 3 — Train ESM-2 Classifier with Cross-Validation

```bash
python train_detector_cv.py \
    --positives_dirs variants_stealth/seqs variants_output/seqs \
    --negatives_fasta negatives/negatives.fasta \
    --outdir results/ \
    --device cpu        # or: cuda, mps
```

Runs 5-fold stratified CV and reports mean ± SD for precision, recall, F1, AUROC. Saves final model to `results/final_model.pkl`.

### Step 4 — Evasion Analysis

```bash
python evasion_analysis_stratified.py \
    --blast_results blast_results/blast_summary.json \
    --tm_scores tm_scores.json \
    --cv_metrics results/cv_metrics.json \
    --embeddings results/embeddings_all.npy \
    --labels results/labels_all.npy \
    --model results/final_model.pkl \
    --outdir evasion_results/
```

Produces the stratified evasion table (Table 2 in preprint) and the main comparison figure (Figure 3).

---

## Negative Class Documentation

The negative training class for the ESM-2 classifier is fully documented in `negatives/negative_class_report.txt`. Summary:

- **Source:** UniProt Swiss-Prot (manually reviewed entries only)
- **N:** 16 sequences across 5 functional categories
- **Categories:** cytoskeletal, metabolic enzymes, chaperones, transport proteins, plant structural (RuBisCO)
- **Rationale:** Plant proteins (RuBisCO) are included despite being plant-derived (like ricin) to confirm the classifier learns biochemical mechanism rather than taxonomic origin.
- **Class ratio:** ~3:1 negative to positive (adjustable via `build_negative_class.py`)

---

## Citations

This work responds directly to and builds on:

1. **Wittmann et al. (2024)** — Demonstrated sequence-identity limitations of BLAST-based biosecurity screening for toxin detection. The primary motivation for this benchmark.

2. **Ikonomova et al. (2026, preprint)** — Proposed the TEVV (Toxin Embedding Verification & Validation) framework for function-based biological threat screening. This repository implements a working version of the TEVV framework.

3. **Lin et al. (2023)** — Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130. [ESM-2 model paper]

4. **Dauparas et al. (2022)** — Robust deep learning–based protein sequence design using ProteinMPNN. *Science* 378, 49–56.

5. **Xu & Zhang (2010)** — TM-align: a protein structure alignment algorithm. *Nucleic Acids Research*. [TM-score > 0.5 threshold for same fold family]

6. **DSSC ISO 20688-2 Implementation Guide** — DNA Synthesis Screening Standard. Cited for policy framing in the discussion section.

---

## Policy Context

This benchmark supports the transition from sequence-identity-based to **function-based biosecurity screening** as called for in the DSSC ISO 20688-2 Implementation Guide. The March 2026 preprint (Ikonomova et al.) explicitly identifies the evasion gap this repository quantifies and calls for working implementations. `funcscreen` is a direct response to that call.

---

## Author

**FridaArrey** | Framework: TEVV | Target venues: bioRxiv preprint → *Applied Biosafety* or *Bioinformatics* short communication

---

## License

MIT — see [LICENSE](LICENSE).
