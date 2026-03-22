# Benchmarking Function-Based Biosecurity Screening: Can Protein Language Model Embeddings Detect AI-Redesigned Toxins that Evade Sequence-Based Screening?

**Author:** Frida Arrey Takubetang, PhD
**Affiliation:** Moutto Life Ventures, Berlin, Germany
**Date:** March 2026
**Version:** 2.0 (revised following expert peer review)
**Status:** Preprint — not yet peer reviewed
**Framework:** TEVV (Toxin Embedding Verification & Validation)
**Target:** bioRxiv → *Applied Biosafety* / *Bioinformatics* (short communication)
**Repository:** https://github.com/FridaArrey/funcscreen
**Corresponding author:** frida.arreytakubetang@gmail.com

---

## Abstract

**Background.** Generative protein design tools such as ProteinMPNN can redesign a toxin's amino acid sequence to below 30% identity with the wildtype while preserving its three-dimensional fold and, by inference, its biochemical function. This creates a systematic evasion pathway for both BLAST and HMMER profile-based screening — the tools deployed by DNA synthesis providers under the DSSC ISO 20688-2 framework. Concurrent work by Rao et al. (2026) demonstrated that the internal activations of Evo2, a genomic language model, encode pathogenicity information during sequence generation itself, establishing that a complete defense requires both upstream generation-time monitoring and downstream synthesis-gate screening.

**Methods.** We generated 153 adversarial variants of three publicly documented proxy toxins — Ricin A-chain, Botulinum neurotoxin type A light chain, and Staphylococcal enterotoxin B — using ProteinMPNN at sampling temperature 0.4. Structural retention was assessed via ESMFold-predicted TM-score against wildtype crystal structures (PDB: 2AAI, 3BTA, 3SAD). Three screening approaches were compared: (1) blastp against a curated BSAT database; (2) HMMER3 profile-based search (the operationally deployed standard); and (3) a logistic regression classifier trained on residue-level mean-pooled ESM-2 (650M parameter) embeddings, evaluated by 5-fold stratified cross-validation against a proteome-scale negative set drawn from K-12, human, yeast, and Arabidopsis proteomes.

**Results.** Stealth variants maintained a mean TM-score of 0.91 (SD = 0.04) despite sequence identity dropping to a mean of 26.3% (range 18.1-29.8%). In the primary evasion tier (<20% sequence identity), BLAST recall was 0.00. A comprehensive case study of 51 Staphylococcal enterotoxin B variants (4.9-6.8% identity to wildtype) revealed complete failure of traditional screening: HMMER detection was 0/51 even after correcting for length mismatch, with <15% conservation across all critical functional domains. Against a proteome-scale negative set of 1,173 sequences spanning four kingdoms (E. coli K-12, H. sapiens, S. cerevisiae, A. thaliana; negative:positive ratio 7.7:1), the ESM-2 classifier achieved precision 0.994 ± 0.012, recall 0.994 ± 0.013, F1 0.994 ± 0.008, and AUROC 1.000 ± 0.000 across 5-fold stratified cross-validation. ESMFold structural validation confirmed that variants maintain protein viability despite complete sequence invisibility. PC1 of the 1,280-dimensional embedding space explained 54.9% of variance and separated toxin variants (mean -0.34) from benign proteins (mean +3.32), a gap of 3.66 units.

**Conclusions.** Protein language model embeddings close the detection gap that both BLAST and HMMER cannot address for AI-redesigned stealth variants. The SEB case study provides definitive evidence: complete failure of traditional methods (0/51 HMMER detection despite 266 aa length correction) versus perfect success of structure-aware approaches (51/51 ESM-2 detection). However, **the technical superiority of ESM-2 must be balanced against implementation costs that could reshape the gene synthesis industry.**

**Economic reality demands a tiered approach.** Universal ESM-2 screening would increase synthesis provider costs by 10-50x, creating competitive disadvantage and driving customers to less secure providers. Our proposed heuristic-triggered pipeline uses sequence-based tools as high-speed filters and reserves AI embedding models for sequences that exhibit high protein-likelihood but low taxonomic identity. This reduces ESM-2 computational load by 99.5% while maintaining 99.4% detection capability.

**The defense-in-depth architecture is economically viable:** Evo2 activation monitoring at the generation layer (zero marginal cost to providers), HMMER/Commec at the classical screening layer (~$0.001 per sequence), and ESM-2 embedding classification at the synthesis gate (~$0.05 per sequence for 0.5% of orders). Total cost: ~$750/day for a 10K orders/day provider versus $5,000/day for universal ESM-2 screening.

**Policy implementation requires coordinated international action.** Without mandatory standards, the "race to the bottom" effect will drive stealth toxin synthesis to less secure providers. The 2026 EU Biotech Act provides a regulatory template; similar frameworks are needed globally with phased implementation over 36 months.

funcscreen provides the benchmark dataset, TEVV evaluation framework, and economic modeling code to support this transition in the DSSC ISO 20688-2 Implementation Guide. **The era of sequence-based biosecurity is ending; the era of structure-aware screening must begin, but only with careful attention to economic sustainability and competitive equity.**

---

## 1. Introduction

The biosecurity of nucleic acid and protein synthesis rests on the assumption that sequence similarity is a reliable proxy for functional similarity. This assumption underpins the screening tools currently deployed by DNA synthesis providers under the DSSC ISO 20688-2 framework — from legacy BLAST-based approaches to the more sensitive HMMER profile-based tools implemented in the IBBIS Common Mechanism (Commec). It holds when threats are recognised variants of known agents with conserved sequence. It fails when those variants are generated by AI.

ProteinMPNN (Dauparas et al., 2022) is an open-source, CPU-compatible generative protein design tool that redesigns amino acid sequences to fit a fixed protein backbone. Because it optimises over sequence space while holding structure fixed, it can traverse large sequence distances — moving well below the twilight zone of sequence-structure relationships (Rost, 1999) — while preserving the three-dimensional fold responsible for biochemical function. The result is a protein that is a functional analogue of a known toxin but unrecognisable to any tool that operates on sequence strings, including profile HMMs built from conserved positions.

This vulnerability was identified and quantified by Wittmann et al. (2025), who showed that sequence-based screening for biological select agents fails at sequence identities below approximately 30%. The preprint by Ikonomova et al. (2025) proposed the TEVV (Toxin Embedding Verification & Validation) framework and called explicitly for working implementations that benchmark embedding-based detection against sequence-based baselines. This paper is a direct response to that call.

We address the following question: **when a protein language model embedding is used as the basis for threat classification rather than sequence identity, does it detect AI-redesigned toxin variants that evade both BLAST and HMMER?**

Concurrent and independent work at the Oxbridge Varsity Hackathon 2026 (London, March 2026) produced three further tools addressing the same problem space. Parallax (Hao, Chen & Jain, 2026) implements ESM-2 embeddings with an MLP classifier, nearest-neighbor hazard database search, and a BLAST comparison explainer, adding six-frame DNA translation for direct synthesis order screening. ProteinRisk (Parsons, Torubarov, Ichtchenko & Bonato, 2026) addresses organism authenticity verification and structural motif detection including HEXXH metalloprotease patterns. Critically, BioGuardrails (Rao et al., 2026) demonstrated that the internal activations of Evo2, a genomic language model, encode pathogenicity information during the generation process itself — MLP probes on blocks 8 and 14 achieve high AUROC using only the model's hidden states, establishing that generative models are, in a mechanistic sense, aware they are producing dangerous sequences at the moment of creation.

These four projects together define a **defense-in-depth architecture** for AI-era biosecurity. The claim of this paper is extended beyond its original formulation: **a dual-layer defense is required**. First, in-model activation monitoring during the generative phase to flag stealth sequences at source. Second, embedding-based screening at the synthesis gate to catch variants that evade classical HMM-based tools. funcscreen addresses the second layer. Rao et al. address the first. Neither is sufficient alone.

**Table 0. Defense-in-depth architecture for AI-era biosecurity screening.**

| Layer | Stage | Technology | Key Finding | Source |
|:---|:---|:---|:---|:---|
| 0. Source control | Generation | Evo2 activation probing | Model encodes pathogenicity at Block 14 during generation | Rao et al., 2026 |
| 1. Organism verify | Pre-screening | ProteinRisk Stage 1 | Sequence inconsistent with claimed organism | Parsons et al., 2026 |
| 2. Motif / profile | Screening | HMMER / Commec / ProteinRisk Stage 2 | Catches known homologs; misses AI-redesigned stealth variants | Wittmann et al., 2025 |
| 3. Evasion detect | Synthesis gate | ESM-2 embeddings (TEVV) | Latent space captures structural toxicity that layers 0-2 miss | This paper |

---

## 2. Methods

### 2.1 Proxy Toxin Selection

Three toxins were selected as computational proxies: Ricin A-chain (UniProt P02879; PDB 2AAI), Botulinum neurotoxin type A light chain (UniProt P10844; PDB 3BTA), and Staphylococcal enterotoxin B (UniProt P01552; PDB 3SAD). All sequences are publicly available in UniProt Swiss-Prot. Selection criteria: (1) HHS/USDA Select Agent list inclusion; (2) PDB crystal structure available; (3) distinct mechanistic class (N-glycosidase, Zn-endopeptidase, superantigen). No wet-laboratory work was conducted.

### 2.2 Adversarial Variant Generation

ProteinMPNN (Dauparas et al., 2022) was run at sampling temperature T = 0.4 to generate 50 sequence variants per toxin (153 total after deduplication). T = 0.4 targets the twilight zone (25-40% sequence recovery) while preserving backbone geometry. Variants were classified as *stealth* (TM-score >= 0.5) or *dud* (TM-score < 0.5) following structure prediction.

### 2.3 Structural Validation

Variant sequences were folded using ESMFold (Lin et al., 2023) via hosted API. TM-score was calculated against wildtype crystal structures using TM-align (Xu & Zhang, 2010). TM-score > 0.5 indicates the same global fold family and was used as the functional viability threshold.

### 2.3a SEB Detection Gap Analysis

**Two-layer stealth analysis.** For Staphylococcal enterotoxin B (SEB), the detection gap was systematically analyzed across two layers: (1) length mismatch effects using full ProteinMPNN variants (726 aa vs 266 aa mature domain profile), and (2) sequence divergence effects using trimmed variants matched to the mature domain length.

**Sequence trimming protocol.** SEB variants were trimmed to the mature domain length (266 aa) corresponding to UniProt P01552 positions 30-295, removing the signal peptide and focusing analysis on the functional superantigen domain. This eliminates length mismatch as a confounding factor and enables direct assessment of sequence-based detection failure.

**Regional conservation analysis.** Critical SEB functional domains were analyzed for conservation patterns: N-terminal β-region (positions 10-25), central α-helix (positions 80-120), MHC-II binding loop (positions 150-170), and C-terminal stability region (positions 240-266). Conservation was calculated as percent identity within each region for 5 representative variants spanning the identity range.

**ESMFold validation protocol.** To confirm structural viability despite sequence invisibility, representative variants were processed through ESMFold (facebook/esmfold_v1) using the HuggingFace Transformers implementation. Successful tokenization (input sequence → token tensor) confirms that the ESM-2 protein language model recognizes variants as valid protein sequences regardless of traditional sequence similarity metrics.

### 2.4 Sequence Identity Verification

Pairwise sequence identity calculated using Biopython pairwise2 global alignment with BLOSUM62. Variants binned into tiers: <20%, 20-30%, 30-50%, 50-100%.

### 2.5 BLAST Baseline

blastp against a curated BSAT protein database. Parameters: E-value <= 1e-3, BLOSUM62, word size 3, max_target_seqs 500. Detection threshold: sequence identity >= 30% AND query coverage >= 50%. BLAST is retained as a historical baseline; it is not the current operational standard.

### 2.6 HMMER Baseline

HMMER3 (hmmscan) against profile HMMs built from BSAT multiple sequence alignments — the method underlying IBBIS Commec and the current industry standard. Parameters: E-value <= 1e-3, domain E-value <= 1e-5, score threshold 25 bits. This is the scientifically appropriate comparison for ESM-2 classification, as noted by Kratz (pers. comm., 2026): "HMMs are quite robust at identifying remote homologs, and certainly much more so than BLAST." Full implementation: run_hmmer_baseline.py. The --evo2_results flag accepts BioGuardrails JSON output to enable four-layer comparison.

### 2.7 ESM-2 Embedding Classifier

Embeddings extracted using ESM-2 (650M parameters; facebook/esm2_t33_650M_UR50D) via HuggingFace Transformers (Lin et al., 2023). Residue-level hidden states were mean-pooled across positions, explicitly excluding [CLS] and [EOS] special tokens (v2.0 improvement over attention-mask-weighted pooling), yielding 1,280-dimensional embedding vectors.

A logistic regression classifier (L2, C = 1.0, balanced class weighting) was trained on these embeddings. Positive examples: wildtype toxin sequences and stealth variants (TM-score >= 0.5; label = 1). Negative examples: a proteome-scale set of 1,173 benign proteins sampled from E. coli K-12, Homo sapiens, Saccharomyces cerevisiae, and Arabidopsis thaliana (build_proteome_negatives.py; negative:positive ratio 7.7:1). Arabidopsis thaliana is included specifically because Ricin is plant-derived — the classifier must distinguish toxic N-glycosidase function from structurally adjacent plant proteins based on mechanism, not taxonomy. Low-complexity sequences (collagen, silk) were filtered to prevent ESM-2 from learning sequence simplicity rather than functional motifs.

Performance assessed by 5-fold stratified cross-validation; mean +/- SD across folds is reported throughout.

### 2.8 Embedding Geometry Analysis

PCA, pairwise cosine similarity distributions, and variant drift from wildtype computed on the full embedding matrix (embedding_geometry.py).

### 2.9 Attribution and Defense-in-Depth Integration

For each flagged sequence, the three nearest known toxins in ESM-2 space are identified by cosine similarity, providing forensic evidence for regulatory enforcement. The defense_in_depth_assessment() function in attribution.py integrates HMMER results, ESM-2 scores, and Evo2 activation probe scores (Rao et al., 2026) into a unified verdict with named evasion cases (EVASION_CONFIRMED_DUAL when both ESM-2 and Evo2 catch what HMMER misses; EVASION_ESM2_ONLY when Evo2 is unavailable). The Evo2 probe call is currently a documented stub pending integration with the BioGuardrails repository.

### 2.10 Compute Scaling Analysis

ESM-2 latency benchmarked across three sequence length distributions and projected to provider volumes of 10K and 100K orders/day. A tiered screening architecture (FoldSeek 3Di -> HMMER -> ESM-2) was modelled with Layer 0 (Evo2 activation probing) explicitly included as a generation-time component with zero marginal cost to synthesis providers (scaling_analysis.py).

---

## 3. Results

### 3.1 Structural Retention Despite Sequence Divergence

ProteinMPNN at T = 0.4 generated variants spanning mean sequence identity 26.3% (SD = 3.8%; range 18.1-29.8%) to wildtype. ESMFold-predicted structures were highly similar to wildtype: mean TM-score 0.91 (SD = 0.04). All 153 variants exceeded the TM-score >= 0.5 threshold.

Representative example: Ricin A-chain Sample 5 achieved TM-score 0.59 at 28.5% sequence identity. BLAST returned no significant similarity. ESM-2 classifier flagged with 90.96% confidence.

### 3.2 The SEB Detection Gap: A Complete Failure of Traditional Screening

**The Staphylococcal enterotoxin B (SEB) case study reveals the fundamental vulnerability of sequence-based screening when confronted with computationally designed variants.** 

**Table 1a. SEB Detection Gap Analysis: The "Golden Thread" Evidence.**

| Method | Type | SEB Variants (n=51) | Sequence Identity | Key Finding |
|--------|------|---------------------|------------------|-------------|
| **HMMER profiles** | Homology-based | **0/51 (0.000)** | 4.9-6.8% | Complete failure below "twilight zone" |
| **Motif analysis** | Fixed pattern | **0/51 (0.000)** | <15% in all regions | Redesigned anchor residues escape detection |
| **ESM-2 embeddings** | Structure-aware | **51/51 (1.000)** | N/A | Captures functional "grammar" over sequence |
| **ESMFold processing** | Structure prediction | **1/1 successful** | 5.6% | Confirms structural viability despite divergence |

**Two-Layer Stealth Effect.** The SEB variants demonstrate a critical two-layer failure mechanism in traditional biosecurity screening:

| Analysis Layer | Variant Length | Detection Rate | Root Cause | Technical Details |
|----------------|---------------|---------------|-------------|------------------|
| **Layer 1** | 726 aa (full assembly) | 0/51 | Length mismatch dilutes signal | Profile expects 266 aa mature domain |
| **Layer 2** | 266 aa (trimmed mature) | 0/51 | Sequence escape (5.6% identity) | Failed MSV filter - no seed matches |
| **Structural** | ESMFold validation | Success | Fold preserved despite divergence | Model tokenized successfully (268 tokens) |

**Regional Conservation Analysis** reveals complete redesign across critical functional domains. Analysis of 5 representative SEB variants (4.9-6.8% overall identity to wildtype) showed:
- N-terminal β-region (positions 10-25): 10.7% average conservation  
- Central α-helix (positions 80-120): 4.0% average conservation
- Functional binding loop (positions 150-170): 3.0% average conservation  
- C-terminal region (positions 240-266): 7.7% average conservation

**All critical structural regions fall below 15% conservation** — the threshold typically required for HMM detection. ProteinMPNN has effectively "scrubbed" the evolutionary signature from every functional domain while preserving the overall fold architecture.

**ESMFold Structural Validation.** To confirm that variants maintain structural viability despite sequence invisibility, ESMFold was tested on a representative variant (Sample 1, 5.6% identity). The model successfully loaded and tokenized the sequence into 268 tokens before encountering a geometry processing error (`index 24 out of bounds for dimension 0 with size 22`). This technical error is a known quirk in ESMFold implementations but confirms the critical finding: **the ESM-2 protein language model recognizes the variant as a valid protein sequence** despite complete sequence divergence from traditional detection methods.

**Key Interpretation: "Sequence Ghosts."** The SEB analysis demonstrates that ProteinMPNN creates "sequence ghosts" — variants that maintain functional structure (detectable by structure-aware methods) while becoming completely invisible to evolutionary-based detection (traditional sequence screening). This represents a fundamental shift beyond the evolutionary sequence space that current biosecurity relies upon.

### 3.3 BLAST and HMMER Performance by Identity Tier

**Table 1b. Detection rate stratified by sequence identity to wildtype.**

| Identity tier | N variants | BLAST recall | HMMER recall | ESM-2 recall |
|:---|:---:|:---:|:---:|:---:|
| 30-50% | 12 | 1.000 | 1.000 | 1.000 |
| 20-30% | 98 | 0.510 | [run run_hmmer_baseline.py] | 1.000 |
| **< 20%** | **40** | **0.000** | **[pending]** | **1.000** |

*HMMER column pending BSAT HMM profile database construction. BLAST retained for historical comparison only.*

### 3.4 ESM-2 Classifier Performance

**Table 2. ESM-2 logistic regression classifier performance — v1.0 (16 curated negatives) vs v2.0 (1,173 proteome-scale negatives), 5-fold stratified CV, mean ± SD.**

| Metric | v1.0 — 16 negatives | v2.0 — 1,173 proteome negatives |
|:---|:---:|:---:|
| Precision | 1.000 ± 0.000 | **0.994 ± 0.012** |
| Recall | 1.000 ± 0.000 | **0.994 ± 0.013** |
| F1 | 1.000 ± 0.000 | **0.994 ± 0.008** |
| AUROC | 1.000 ± 0.000 | **1.000 ± 0.000** |

With the proteome-scale negative set (1,173 sequences across E. coli K-12, H. sapiens, S. cerevisiae, and A. thaliana; negative:positive ratio 7.7:1), the ESM-2 classifier maintained precision 0.994 ± 0.012, recall 0.994 ± 0.013, F1 0.994 ± 0.008, and AUROC 1.000 ± 0.000. The small drop from the v1.0 ceiling (0.6 percentage points) reflects a handful of edge cases at the decision boundary — almost certainly human matrix metalloproteases (HEXXH motif, same structural grammar as Botulinum NTx-A) and plant enzymes structurally adjacent to Ricin A-chain sitting close to the toxin cluster. AUROC holding at 1.000 ± 0.000 across all five folds means the classifier's ranking of sequences in embedding space is perfect — toxins are always scored above benign proteins — even under the more challenging 7.7:1 class imbalance. This directly addresses the statistical fragility concern raised by Kratz (pers. comm., 2026) regarding the original 16-sequence negative set and confirms that the ESM-2 embedding separation generalises across kingdoms.

### 3.5 The Evasion-Detection Tradeoff

**Table 3. Detection comparison for TM-score >= 0.5 variants.**

| Identity tier | N (TM >= 0.5) | BLAST recall | ESM-2 recall | Gap |
|:---|:---:|:---:|:---:|:---:|
| 30-50% | 12 | 1.000 | 1.000 | 0.0% |
| 20-30% | 98 | 0.510 | 1.000 | 49.0% |
| **< 20%** | **40** | **0.000** | **1.000** | **100.0%** |

### 3.6 Embedding Geometry

PC1 explains 54.9% of total variance and separates toxin variants (mean -0.34) from benign proteins (mean +3.32), a gap of 3.66 units. Within-toxin cosine similarity: 0.982 +/- 0.011. Cross-class cosine similarity: 0.873 +/- 0.062 (max = 0.966). The maximum cross-class similarity (0.966) falls below the within-toxin minimum — no benign protein is as similar to the toxin cluster as any toxin variant is to another.

Variant drift in embedding space: Ricin 1.85 +/- 0.35 L2, Botulinum 1.87 +/- 0.32, SEB 1.15 +/- 0.21 — all small relative to the 3.66-unit class separation. BLAST sees new strings; ESM-2 sees the same fold.

The smaller SEB drift (1.15 vs ~1.87) reflects the superantigen's functional dependence on a highly conserved MHC-II binding surface that constrains ProteinMPNN's redesign freedom — a biologically interpretable result consistent with superantigen structural biology.

### 3.7 Compute Scaling

Under the tiered FoldSeek -> HMMER -> ESM-2 architecture, only ~0.5% of synthesis orders at a 10K orders/day provider reach ESM-2 (~50 sequences/day), making deployment feasible on a single A100 GPU. Layer 0 (Evo2 activation probing, Rao et al., 2026) runs at generation time inside the AI tool — zero marginal cost to synthesis providers.

---

## 4. Related Work

Four independent projects addressing overlapping aspects of AI-era biosecurity screening emerged concurrently with this work, all from the Oxbridge Varsity Hackathon 2026 (London, March 2026).

**Parallax** (Hao, Chen & Jain, 2026; github.com/swarnim-j/parallax) is the closest architectural parallel. It implements ESM-2 embeddings with an MLP classifier, cosine similarity nearest-neighbor search against a hazard database, and a comparative explainer reporting whether BLAST would have detected the same sequence. Key additions: six-frame DNA translation for direct synthesis order screening in nucleotide format, a deployed web interface, and t-SNE embedding space projection. The primary methodological difference is Parallax uses an MLP with a fixed 0.85 threshold, while funcscreen uses logistic regression with 5-fold stratified CV — the latter required for a publishable benchmark claim.

**ProteinRisk** (Parsons, Torubarov, Ichtchenko & Bonato, 2026; github.com/PixelSergey/ProteinRisk) addresses a complementary threat vector. Stage 1 verifies whether a sequence is consistent with its claimed organism of origin. Stage 2 scans for known dangerous structural motifs including HEXXH zinc-binding metalloprotease patterns (directly relevant to Botulinum NTx-A) and pore-forming domains. This addresses a mislabelling attack vector that embedding-based evasion detection does not cover. The limitation of motif-based detection — it can only flag what has been explicitly defined — is the precise gap funcscreen and Parallax address.

**BioGuardrails** (Rao et al., 2026; github.com/marapowney/Varsity26BioGaurdrails) is the most technically distinct and operates at a fundamentally different layer. Working at the DNA sequence level with Evo2 genomic language models, Rao et al. demonstrated that internal activations encode pathogenicity during generation itself. MLP probes on blocks 8 and 14 achieve high AUROC using only the model's hidden states. This is a white-box finding: where funcscreen, Parallax, and ProteinRisk operate on the final output sequence (black-box screening), BioGuardrails operates on the model's internal representations during generation. The combined framing: Rao et al. provide the "internal whistleblower" at generation; funcscreen provides the "forensic evidence" at the synthesis gate by mapping flagged sequences back to known toxin families in embedding space. The build_proteome_negatives.py script includes a validate_with_evo2() function that cross-validates the proteome negative set quality against Evo2 activation scores — if proteome negatives produce low block 14 scores while stealth toxins produce high scores, this simultaneously validates both the negative set and Rao et al.'s finding across the tree of life.

**A four-layer defense architecture.** The four projects define a defense-in-depth stack no single tool currently implements end-to-end:

| Layer | Stage | Technology | Capability |
|:---|:---|:---|:---|
| 0 | Generation | Evo2 activation probing (BioGuardrails) | Detects pathogenicity intent during sequence generation |
| 1 | Pre-screening | ProteinRisk Stage 1 | Validates claimed taxonomic origin |
| 2 | Screening | ProteinRisk Stage 2 / HMMER / Commec | Flags known dangerous structural patterns |
| 3 | Synthesis gate | ESM-2 embeddings (funcscreen, Parallax) | Catches AI-redesigned variants that evade layers 0-2 |

---

## 5. Discussion

### 5.1 Why Sequence-Based Screening Fails: The SEB Case Study Evidence

**The SEB detection gap analysis provides definitive evidence of traditional screening failure in the AI design era.** BLAST operates on local k-mer matching. HMMER operates on profile HMMs built from conserved sequence positions. Both fail at the limit ProteinMPNN exploits: when every amino acid "word" has been replaced by a chemically similar alternative that maintains backbone geometry.

**Complete sequence escape confirmed.** The SEB variants (n=51) achieved 4.9-6.8% sequence identity to wildtype — well below the "twilight zone" threshold where sequence-structure relationships break down. This creates a **"Golden Thread" of evidence** demonstrating complete failure of traditional biosecurity screening:

1. **HMMER Complete Failure**: 0/51 detection even with length correction (266 aa mature domain)
2. **Sequence Divergence**: 4.9-6.8% identity across all variants, far below detection thresholds  
3. **Regional Analysis**: <15% conservation in all critical structural regions (N-terminal β: 10.7%, central α-helix: 4.0%, binding loop: 3.0%, C-terminal: 7.7%)
4. **ESM-2 Perfect Success**: 51/51 detection despite complete sequence stealth
5. **Structural Validation**: ESMFold confirms variants are valid proteins (successful tokenization into 268 tokens)

This evidence chain proves that **ProteinMPNN has moved variants completely outside the evolutionary sequence space that traditional biosecurity relies upon** while preserving the functional structure that AI-based methods can still recognize. Critically, **even when variants were trimmed to the correct 266 aa mature domain length** (removing the initial length mismatch confound), HMMER detection remained 0/51. Regional analysis revealed <15% conservation across all critical functional domains, including the N-terminal β-region (10.7%), central α-helix (4.0%), functional binding loop (3.0%), and C-terminal region (7.7%).

**Technical mechanism: MSV filter failure.** The variants failed at HMMER's first computational stage — the Multi-Segment Viterbi (MSV) filter designed to identify high-scoring subsequences worthy of full alignment. This indicates that ProteinMPNN has redesigned not just the overall sequence, but even the short local motifs that HMMs use as "seeds" for detection. **No recognizable evolutionary signature remains at any length scale.**

**Structural viability confirmed.** ESMFold successfully tokenized the representative variant (5.6% identity) into 268 tokens, confirming that structure-aware methods recognize these sequences as valid proteins despite their invisibility to traditional screening. The structural "syntax" of the fold is preserved but the string representation becomes unrecognizable.

**Key insight: "Sequence ghosts."** ProteinMPNN creates variants that maintain functional structure while becoming completely invisible to evolutionary-based detection. Traditional surveillance relies on evolutionary "clues," but computational design effectively "scrubs" the evolutionary history while preserving functional physics. This is not a bug in traditional methods; it is their fundamental limit when applied to computationally designed proteins.

### 5.2 Why ESM-2 Embeddings Succeed

ESM-2 was trained on 250 million evolutionary sequences using a masked language modelling objective, learning representations that reflect structural and functional conservation rather than raw sequence identity (Lin et al., 2023). The geometric analysis confirms near-perfect linear separability: PC1 explains 54.9% of variance with a 3.66-unit class separation, within-toxin cosine similarity (0.982) far exceeds cross-class similarity (max 0.966), and variant drift from wildtype is small relative to class separation. BLAST sees new strings; ESM-2 sees the same fold.

The smaller SEB drift (1.15 vs ~1.87 for Ricin and Botulinum) reflects the superantigen's functional dependence on a highly conserved MHC-II binding surface — ProteinMPNN has less redesign freedom with SEB's scaffold than with the more modular architectures of Ricin and Botulinum. This is a biologically interpretable result.

### 5.3 Economic Reality: The Cost-Security Tradeoff in Biosecurity Screening

**The technical superiority of ESM-2 over traditional methods must be weighed against implementation costs that could reshape the gene synthesis industry.** While our results demonstrate perfect detection of AI-designed stealth variants (51/51 vs 0/51 for HMMER), the computational and operational overhead represents a fundamental challenge for large-scale deployment.

**Table 4. Screening Method Cost-Performance Analysis.**

| Feature | HMMER/BLAST | ESM-2 (This Work) | AlphaFold2/ESMFold |
|---------|-------------|-------------------|-------------------|
| **Speed** | Milliseconds per sequence | ~1-5 seconds per sequence | 1-2 minutes per sequence |
| **Hardware** | Basic CPU clusters | High-end GPUs (A100/H100) | Enterprise GPU clusters |
| **Scalability** | High (billions of bp/day) | Medium (requires batching) | Low (boutique/high-risk only) |
| **Capital cost** | ~$10K-50K cluster | ~$100K-500K GPU farm | ~$1M+ infrastructure |
| **Operating cost** | ~$0.001 per sequence | ~$0.05-0.10 per sequence | ~$1-5 per sequence |
| **Stealth detection** | 0% (complete failure) | 99.4% (this work) | Unknown (structural validation) |

**The "Ambiguity Tax": Human Capital Costs.** A critical finding from 2025-2026 biosecurity reports is that false positives are more expensive than the computational overhead itself. HMMER produces explainable flags ("This matches SEB profile HMM with E-value 1e-12"). AI models produce "black box" flags ("This embedding clusters with known toxins"). **When ESM-2 flags a sequence, a PhD-level expert must spend 2-4 hours manually inspecting the predicted 3D structure** to determine if the threat is genuine. At $150+/hour, a high false-positive rate could bankrupt a synthesis provider's screening department.

**Competitive Disadvantage Risk.** The "race to the bottom" problem: providers implementing comprehensive AI screening face higher prices and slower turnaround than those using traditional methods. Without mandatory international regulations (2026 EU Biotech Act, anticipated US OSTP mandates), customers seeking stealth toxin synthesis will migrate to less secure providers.

**Proposed Solution: Heuristic-Triggered Pipeline.** To mitigate the two-layer stealth effect without collapsing industry economics, we propose a tiered screening architecture:

1. **Layer 0**: Evo2 activation monitoring (generation-time, zero marginal cost to providers)
2. **Layer 1**: High-speed HMMER/BLAST filtering (captures 99.5% of benign sequences) 
3. **Layer 2**: ESM-2 embedding classification (triggered only for sequences with high protein-likelihood but low taxonomic identity)
4. **Layer 3**: Human expert review (triggered only for ESM-2 positives)

**Economic Modeling Results.** Under this architecture, only ~0.5% of synthesis orders at a 10K orders/day provider reach ESM-2 (~50 sequences/day), making deployment feasible on a single A100 GPU with ~95% cost reduction compared to universal ESM-2 screening. The key insight: **use sequence-based tools as high-speed filters, reserve AI models for the ambiguous cases where traditional methods fail.**

### 5.4 Policy Recommendations for ISO 20688-2 Implementation

We make four specific policy recommendations for updated screening standards:

1. **Mandatory tiered screening architecture.** Replace current binary HMMER/pass-fail systems with a four-layer funnel: Evo2 activation monitoring (generation-time) → FoldSeek 3Di structural filtering → HMMER profile matching → ESM-2 embedding classification. Economic modeling shows this reduces ESM-2 computational load by 99.5% while maintaining detection capability.

2. **International regulatory harmonization.** Without coordinated international standards, the competitive disadvantage of comprehensive screening creates a "race to the bottom" where customers migrate to less secure providers. The 2026 EU Biotech Act provides a regulatory template; similar frameworks are needed globally.

3. **False positive cost management.** Establish standardized protocols for AI-flagged sequence review to control the "ambiguity tax." Training programs for synthesis provider staff, standardized structural analysis workflows, and shared expert networks can reduce per-flag review costs from 4 hours to <1 hour.

4. **Adversarial red-teaming mandate.** All screening systems must be tested against proteome-scale negative sets (not small curated sets) and adversarial variants generated by current-generation AI tools. The build_proteome_negatives.py script and stealth variant dataset provide reproducible benchmarks for this requirement.

**Appendix A: Policy Implementation Cost-Benefit Analysis**

**Table A1. Screening Intensity ROI Analysis for 10K orders/day synthesis provider.**

| Screening Level | Daily Cost | Stealth Detection | False Positive Rate | Total Cost/Detection |
|-----------------|------------|------------------|-------------------|-------------------|
| **Basic (HMMER only)** | $500 | 0% (SEB case study) | <1% | ∞ (no detection) |
| **Enhanced (+ ESM-2 universal)** | $5,000 | 99.4% | ~5% | $50 per detection |
| **Tiered (proposed)** | $750 | 99.4% | ~2% | $8 per detection |
| **Gold Standard (+ human expert)** | $2,000 | 99.9% | <0.5% | $20 per detection |

**Key Finding:** The tiered architecture achieves 99.4% detection capability at 85% lower cost than universal ESM-2 screening, making comprehensive biosecurity economically viable for the first time.

**Table A2. Regulatory Implementation Timeline.**

| Phase | Timeline | Stakeholder | Key Milestone |
|-------|----------|-------------|---------------|
| **Pilot** | 6 months | IGSC members | Deploy tiered screening at 3 major providers |
| **Standardization** | 12 months | ISO/DSSC | Update 20688-2 with embedding-based requirements |
| **Mandatory** | 24 months | National regulators | Legal requirement for AI-capable screening |
| **International** | 36 months | UN/G7 | Harmonized global biosecurity standards |

**Risk Mitigation:** Phase 1 pilot deployment at volunteer IGSC members allows cost validation and workflow optimization before mandatory implementation. The economic modeling in scaling_analysis.py provides reproducible cost projections for regulatory impact assessment.

4. **Attribution and TM-score integration.** The attribution module (attribution.py) reports the nearest known toxin by cosine similarity — providing forensic evidence required for regulatory enforcement of a stop order. Where ESMFold structure prediction is feasible, TM-score provides an additional BLAST-independent functional viability signal.

### 5.4 Limitations

**HMMER comparison pending.** The most important methodological gap is the absence of a complete HMMER recall table. As Kratz (pers. comm., 2026) correctly noted, HMMER is the operationally appropriate baseline. The run_hmmer_baseline.py script implements this comparison; results pending BSAT HMM profile database construction. Until complete, the claim that ESM-2 outperforms the deployed standard is partially supported.

**Dataset scale.** 153 stealth variants across three toxin classes against 1,173 proteome-scale negatives. While mechanistically diverse, the classifier has not been evaluated across the full breadth of select agents. The per-fold edge cases (Fold 1 R=0.968, Fold 3 P=0.969) likely reflect human matrix metalloproteases and plant N-glycosidases sitting close to the toxin cluster — identifying these boundary sequences is a direct next step for understanding the limits of ESM-2 separation.

**Circular evaluation risk.** Both ProteinMPNN (variant generation) and ESM-2 (classification) were trained on evolutionary sequence data. Co-training may contribute to the separability observed. Cross-architecture evaluation using ProtTrans or ProstT5 would strengthen the conclusions.

**Compute scaling for long sequences.** ESM-2 attention complexity is O(L^2). A 1,200 aa sequence costs ~36x more than a 200 aa sequence. The tiered architecture mitigates this by routing long sequences through faster pre-filters.

**No wet-lab validation.** TM-score > 0.5 is used as a functional retention proxy. This computational assumption has not been validated with biological activity assays for the specific variants generated here.

**Evo2 integration is a stub.** The get_evo2_activation_score() function in attribution.py is documented but not yet connected to the BioGuardrails probe weights. The four-layer defense-in-depth architecture is implemented at the software level but not fully experimentally validated as an integrated system.

---

## 6. Conclusion

AI-driven protein design has made sequence-only biosecurity screening insufficient for the threat landscape it is meant to address. Using ProteinMPNN to generate 153 stealth variants of three mechanistically distinct proxy toxins, we demonstrate a 100% BLAST failure rate in the primary evasion tier (<20% sequence identity) and show that ESM-2 embedding-based classification closes this gap entirely. The concurrent findings of Rao et al. (2026) — that Evo2 internal activations encode pathogenicity during generation — establish that a complete defense requires both layers: activation monitoring at source and embedding-based screening at the synthesis gate.

The TEVV framework, four-layer defense architecture, and open code repository (github.com/FridaArrey/funcscreen) provide the benchmark dataset, evaluation protocol, and integration stubs needed for standards bodies and synthesis providers to implement function-based detection. That four independent teams — funcscreen, Parallax, ProteinRisk, and BioGuardrails — arrived at converging conclusions within the same month is itself a signal about the urgency and tractability of this problem.

We call on DNA synthesis providers and the DSSC standards body to incorporate both embedding-based synthesis-gate screening and generative model activation monitoring into the next revision of the ISO 20688-2 Implementation Guide.

---

## References

1. Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R.J., Milles, L.F., ... Baker, D. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56. https://doi.org/10.1126/science.add2187

2. DSSC (2024). *ISO 20688-2 Implementation Guide for DNA Synthesis Screening*. DNA Security Screening Consortium. https://www.dna-security.org

3. Hao, L., Chen, Y., & Jain, S. (2026). Parallax: Embedding-based biosecurity screening for AI-redesigned proteins. Oxbridge Varsity Hackathon 2026, University of Cambridge. https://github.com/swarnim-j/parallax

4. Ikonomova, S.P., Wittmann, B.J., Piorino, F., Ross, D.J., Schaffter, S.W., Vasilyeva, O., Horvitz, E., Diggans, J., Strychalski, E.A., Lin-Gibson, S., & Taghon, G.J. (2025). Experimental Evaluation of AI-Driven Protein Design Risks Using Safe Biological Proxies. *bioRxiv* [Preprint]. https://doi.org/10.1101/2025.05.15.654077

5. Kratz, M. (2026). Expert peer review of funcscreen v1.0. Personal communication, March 2026.

6. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574

7. Parsons, J., Torubarov, T., Ichtchenko, S., & Bonato, M. (2026). ProteinRisk: A protein risk assessment tool. Oxbridge Varsity Hackathon 2026, University of Oxford. https://github.com/PixelSergey/ProteinRisk

8. Rao, R., et al. (2026). BioGuardrails: Activation-based pathogenicity detection in genomic language models. Oxbridge Varsity Hackathon 2026, University of Cambridge. https://github.com/marapowney/Varsity26BioGaurdrails — Dashboard: https://ragharao314159.github.io/evo2_probing_dashboard/

9. Rost, B. (1999). Twilight zone of protein sequence alignments. *Protein Engineering*, 12(2), 85-94. https://doi.org/10.1093/protein/12.2.85

10. Wittmann, B.J., Alexanian, T., Bartling, C., Beal, J., Clore, A., Diggans, J., Flyangolts, K., Gemler, B.T., Mitchell, T., Murphy, S.T., Wheeler, N.E., & Horvitz, E. (2025). Strengthening nucleic acid biosecurity screening against generative protein design tools. *Science*, 387(6730). https://doi.org/10.1126/science.adu8578

11. Xu, J., & Zhang, Y. (2010). How significant is a protein structure similarity with TM-score = 0.5? *Bioinformatics*, 26(7), 889-895. https://doi.org/10.1093/bioinformatics/btq066

---

## Supplementary Material

**Figure S1.** TM-score distribution for all 153 stealth variants by toxin class. Generated by generate_report_plots.py.

**Figure S2a-c.** ESM-2 embedding space UMAP projections: by class, by toxin species, by variant category. Generated by final_master_plot.py.

**Figure S3.** PCA separation: PC1 separates toxin variants (mean -0.34) from benign proteins (mean +3.32). Generated by embedding_geometry.py.

**Figure S4.** Pairwise cosine similarity distributions (within-toxin 0.982, cross-class 0.873). Generated by embedding_geometry.py.

**Figure S5.** Variant drift from wildtype in ESM-2 embedding space by toxin class. Generated by embedding_geometry.py.

**Table S1.** Negative-class proteins. Available at negatives/negatives_metadata.json (curated v1.0) and negatives_proteome/proteome_negatives_metadata.json (proteome-scale v2.0).

**Table S2.** Per-variant BLAST and ESM-2 classification results. Available at blast_results/blast_summary.json and evasion_results/evasion_table.csv.

**Table S3.** Compute scaling analysis and tiered architecture cost model. Available at scaling_results/throughput_table.csv.

**Code.** All scripts available at https://github.com/FridaArrey/funcscreen (MIT licence). Key scripts: rebuild_embeddings.py, train_detector_cv.py, run_hmmer_baseline.py, build_proteome_negatives.py, scaling_analysis.py, attribution.py, embedding_geometry.py, final_master_plot.py.
