# funcscreen 🧬🛡️

**Benchmarking Function-Based Biosecurity Screening: Can Protein Language Model Embeddings Detect AI-Redesigned Toxins that Evade Sequence-Based Screening?**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/preprint-PREPRINT.md-green)](PREPRINT.md)
[![Framework: TEVV](https://img.shields.io/badge/framework-TEVV-orange)](PREPRINT.md)

---

## Overview

`funcscreen` is a proof-of-concept implementation of the **TEVV (Toxin Embedding Verification & Validation)** framework. It demonstrates that:

1. **ProteinMPNN** can redesign toxin sequences to <30% sequence identity while preserving structural fold (TM-score > 0.5) — making them invisible to both BLAST and HMMER-based biosecurity screening.
2. **ESM-2 latent embeddings** detect these structurally-retained "stealth" variants with significantly higher recall than sequence-based tools at low sequence identity.
3. A **defense-in-depth architecture** is required: generation-time activation monitoring (Rao et al., 2026) AND synthesis-gate embedding screening (this work). Neither layer alone is sufficient.

This addresses a vulnerability identified by Wittmann et al. (2025) and called for explicitly by Ikonomova et al. (2025) in the TEVV framework preprint.

---

## Key Results (v2.0 — Proteome-Scale Validation)

| Metric | BLAST | ESM-2 v1.0 (16 negatives) | ESM-2 v2.0 (1,173 proteome negatives) |
|--------|-------|---------------------------|----------------------------------------|
| Precision | 1.00 | 1.000 ± 0.000 | **0.994 ± 0.012** |
| Recall | 0.50 | 1.000 ± 0.000 | **0.994 ± 0.013** |
| F1 | 0.67 | 1.000 ± 0.000 | **0.994 ± 0.008** |
| AUROC | — | 1.000 ± 0.000 | **1.000 ± 0.000** |

*v2.0 negative set: 1,173 sequences across E. coli K-12, H. sapiens, S. cerevisiae, and A. thaliana (7.7:1 ratio). AUROC holds at 1.000 across all five folds — the classifier's ranking is perfect across kingdoms. The 0.6pp drop from v1.0 reflects edge cases at the decision boundary (human MMPs, plant enzymes), exactly as predicted by Kratz (pers. comm., 2026).*

---

## Defense-in-Depth Architecture

funcscreen addresses **Layer 3** (Synthesis Gate) of a four-layer defense stack:

| Layer | Stage | Technology | Source |
|:---|:---|:---|:---|
| 0. Source control | Generation | Evo2 activation probing | Rao et al., 2026 — BioGuardrails |
| 1. Organism verify | Pre-screening | Sequence-organism consistency | Parsons et al., 2026 — ProteinRisk |
| 2. Motif / profile | Screening | HMMER / Commec | Wittmann et al., 2025 |
| **3. Evasion detect** | **Synthesis gate** | **ESM-2 embeddings (this repo)** | **This work** |

*Concurrent independent projects from Oxbridge Varsity Hackathon 2026: [Parallax](https://github.com/swarnim-j/parallax) (Cambridge), [ProteinRisk](https://github.com/PixelSergey/ProteinRisk) (Oxford), [BioGuardrails](https://github.com/marapowney/Varsity26BioGaurdrails) (Cambridge).*

---

## Proxy Toxins Used

All sequences are publicly available in UniProt Swiss-Prot. No controlled or restricted sequences are used. No wet-laboratory work was conducted.

| Toxin | UniProt | Mechanism | HHS/CDC Status |
|-------|---------|-----------|---------------|
| Ricin A-chain | P02879 | N-glycosidase; depurinates 28S rRNA | Schedule 1 |
| Botulinum neurotoxin A (light chain) | P10844 | Zn-endopeptidase; cleaves SNAP-25 | BSAT Schedule 1 |
| Staphylococcal enterotoxin B | P01552 | Superantigen; crosslinks MHC-II/TCR | Select Agent |

---

## Repository Structure

```text
funcscreen/
├── Core pipeline
│   ├── rebuild_embeddings.py           # Step 1: Embed all sequences
│   ├── embedding_geometry.py           # Step 2: PCA, cosine sim, drift analysis
│   ├── train_detector_cv.py            # Step 3: 5-fold CV classifier
│   ├── run_hmmer_baseline.py           # Step 4: HMMER/Commec + 4-layer comparison
│   ├── evasion_analysis_stratified.py  # Step 5: Stratified evasion table
│   ├── final_master_plot.py            # Step 6: Publication figures
│   ├── attribution.py                  # Step 7: Nearest-toxin + defense-in-depth verdict
│   └── scaling_analysis.py             # Step 8: Compute cost model (4-tier funnel)
│
├── Data generation
│   ├── fetch_multitoxin.py             # Fetch 3 toxin seeds from UniProt
│   ├── build_proteome_negatives.py     # 1,173 proteome-scale negatives
│   └── calculate_tm.py                # TM-score vs wildtype
│
├── Data
│   ├── variants_output/                # 153 ProteinMPNN stealth variants (3 toxins)
│   ├── negatives/
│   │   ├── negatives.fasta             # v1.0: 16 curated (kept for reproducibility)
│   │   └── proteome/                   # v2.0: 1,173 proteome-scale
│   │       ├── all_proteome_negatives.fasta
│   │       ├── k12_negatives.fasta
│   │       ├── human_negatives.fasta
│   │       ├── yeast_negatives.fasta
│   │       ├── arabidopsis_negatives.fasta
│   │       └── sampling_report.txt
│   └── toxin_seeds/                    # Wildtype seed sequences
│
├── Results
│   ├── results/                        # CV metrics, embeddings, figures
│   │   ├── cv_metrics.json
│   │   ├── embeddings_all.npy
│   │   ├── final_model.pkl
│   │   ├── roc_v1_vs_v2.png            # Figure 4: v1 vs v2 ROC comparison
│   │   ├── figure3_evasion_comparison.png
│   │   ├── umap_by_class.png           # Figure S2a
│   │   ├── umap_by_toxin.png           # Figure S2b
│   │   ├── umap_by_category.png        # Figure S2c
│   │   └── embedding_geometry/         # PCA, cosine sim, variant drift
│   └── results/v2_proteome_negatives/  # v2.0 CV results
│
├── scripts/experimental/               # Archived v1.0 scripts
├── ProteinMPNN/                        # Submodule
├── PREPRINT.md                         # Full manuscript (v2.0)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/FridaArrey/funcscreen.git --recurse-submodules
cd funcscreen
pip install -r requirements.txt
```

**Hardware requirements:**

| Step | CPU | GPU | Estimate |
|------|-----|-----|----------|
| UniProt fetch | ✓ | — | ~10 min |
| ProteinMPNN generation | ✓ | Optional | ~30 sec/sequence |
| ESMFold structure prediction | ✗ | >=16 GB VRAM | Use `--use_api` |
| ESM-2 embedding (1,326 sequences) | ✓ slow | ✓ recommended | ~60 min CPU / ~8 min MPS |
| HMMER screening | ✓ | — | Requires HMMER3 |

```bash
# HMMER3
brew install hmmer                    # macOS
conda install -c bioconda hmmer       # conda

# Commec (preferred for Step 4)
pip install commec && commec download-databases
```

---

## Execution Guide

### Step 0 — Generate proteome-scale negatives
```bash
python build_proteome_negatives.py \
    --n_k12 500 --n_human 500 --n_yeast 200 --n_arabidopsis 100 \
    --outdir negatives/proteome/
```
Fetches 1,173 reviewed Swiss-Prot sequences across four kingdoms. Arabidopsis included because Ricin is plant-derived — classifier must learn toxic mechanism, not plant taxonomy.

### Step 1 — Embed all sequences
```bash
python rebuild_embeddings.py --device mps   # or cpu / cuda
```
Embeds 153 stealth variants + 1,173 negatives using ESM-2 650M with residue-only mean pooling (skips [CLS] and [EOS] tokens).

### Step 2 — Embedding geometry
```bash
python embedding_geometry.py
```
PCA: PC1 explains 54.9% variance, class separation 3.66 units. Within-toxin cosine sim: 0.982 ± 0.011. Cross-class: 0.873 ± 0.062.

### Step 3 — Train with 5-fold CV
```bash
python train_detector_cv.py \
    --positives_dirs variants_output/ricin_A_chain/stealth/seqs \
                     variants_output/botulinum_ntx_A/stealth/seqs \
                     variants_output/staph_enterotoxin_B/stealth/seqs \
    --negatives_fasta negatives/proteome/all_proteome_negatives.fasta \
    --outdir results/v2_proteome_negatives/ \
    --device mps
```
v2.0 results: Precision 0.994 ± 0.012, Recall 0.994 ± 0.013, F1 0.994 ± 0.008, AUROC 1.000 ± 0.000.

### Step 4 — HMMER baseline + four-layer comparison
```bash
python run_hmmer_baseline.py --mode commec \
    --query_fasta variants_output/ricin_A_chain/stealth/seqs/ \
    --category stealth --outdir hmmer_results/ \
    --evo2_results path/to/bioguardrails_output.json   # optional Layer 0
```

### Step 5 — Evasion analysis
```bash
python evasion_analysis_stratified.py \
    --blast_results blast_results/blast_summary.json \
    --cv_metrics results/v2_proteome_negatives/cv_metrics.json \
    --outdir evasion_results/
```

### Step 6 — Publication figures
```bash
python final_master_plot.py
```

### Step 7 — Attribution + defense-in-depth verdict
```bash
python attribution.py build_db
python attribution.py screen --fasta query.fasta \
    --hmmer hmmer_results/hmmer_summary.json \
    --outdir attribution_results/
```
Outputs per-sequence verdict with nearest known toxin by cosine similarity and named evasion case (EVASION_CONFIRMED_DUAL, EVASION_ESM2_ONLY, etc.). Evo2 Layer 0 stub: replace `get_evo2_activation_score()` with BioGuardrails API call.

### Step 8 — Compute scaling analysis
```bash
python scaling_analysis.py --device cpu
```
Four-tier funnel (Evo2 → FoldSeek → HMMER → ESM-2): only ~50 sequences/day reach ESM-2 at 10K orders/day. Layer 0 (Evo2) runs at generation time — zero cost to synthesis providers.

---

## Negative Class: v1.0 vs v2.0

| Version | N | Source | Neg:Pos ratio | Script |
|---------|---|--------|---------------|--------|
| v1.0 | 16 | 5 curated Swiss-Prot categories | 1:10 | `build_negative_class.py` |
| v2.0 | 1,173 | K-12 + Human + Yeast + Arabidopsis proteomes | 7.7:1 | `build_proteome_negatives.py` |

v2.0 includes human matrix metalloproteases (HEXXH motif, same structural grammar as Botulinum but non-dangerous) and Arabidopsis plant proteins (structurally adjacent to Ricin) — the hardest challenge cases. AUROC held at 1.000 across both versions. See `negatives/proteome/sampling_report.txt`.

---

## Related Work

Four concurrent independent projects from Oxbridge Varsity Hackathon 2026:

| Project | Team | Approach |
|---------|------|----------|
| [Parallax](https://github.com/swarnim-j/parallax) | Hao, Chen & Jain (Cambridge) | ESM-2 + MLP + web UI + DNA translation |
| [ProteinRisk](https://github.com/PixelSergey/ProteinRisk) | Parsons, Torubarov, Ichtchenko & Bonato (Oxford) | Organism verification + motif detection |
| [BioGuardrails](https://github.com/marapowney/Varsity26BioGaurdrails) | Rao et al. (Cambridge) | Evo2 internal activation probing at generation time |

BioGuardrails is Layer 0 of this stack: MLP probes on Evo2 blocks 8 and 14 detect pathogenicity during sequence generation. Dashboard: [ragharao314159.github.io/evo2_probing_dashboard](https://ragharao314159.github.io/evo2_probing_dashboard/)

---

## Citations

1. Wittmann, B.J. et al. (2025). Strengthening nucleic acid biosecurity screening against generative protein design tools. *Science* 387(6730). https://doi.org/10.1126/science.adu8578
2. Ikonomova, S.P. et al. (2025). Experimental Evaluation of AI-Driven Protein Design Risks. *bioRxiv*. https://doi.org/10.1101/2025.05.15.654077
3. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123-1130. https://doi.org/10.1126/science.ade2574
4. Dauparas, J. et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science* 378, 49-56. https://doi.org/10.1126/science.add2187
5. Rao, R. et al. (2026). BioGuardrails. Varsity Hackathon 2026. https://github.com/marapowney/Varsity26BioGaurdrails
6. Xu, J. & Zhang, Y. (2010). TM-score = 0.5 threshold. *Bioinformatics* 26(7). https://doi.org/10.1093/bioinformatics/btq066
7. Kratz, M. (2026). Expert peer review of funcscreen v1.0. Personal communication, March 2026.

---

## Policy Context

funcscreen supports the transition from sequence-identity to **function-based biosecurity screening** as required by the DSSC ISO 20688-2 Implementation Guide. Synthesis screening is the only node in the AI biosecurity kill chain where a regulatory instrument breaks physical realisation of a threat regardless of what happens at the model layer.

---

## Author

**Dr. Frida Arrey Takubetang** | Biosecurity & Biopolicy Expert | Berlin, Germany
Framework: TEVV | Target: bioRxiv → *Applied Biosafety* / *Bioinformatics*

---

## License

MIT — see [LICENSE](LICENSE).
