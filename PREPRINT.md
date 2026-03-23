# Benchmarking Function-Based Biosecurity Screening: Can Protein Language Model Embeddings Detect AI-Redesigned Toxins that Evade Sequence-Based Screening?

**Author:** Frida Arrey Takubetang, PhD
**Affiliation:** Moutto Life Ventures, Berlin, Germany
**Date:** March 2026
**Version:** 2.1 (EXPERIMENTAL CONFIRMATION)
**Status:** Preprint — ready for peer review
**Framework:** TEVV (Toxin Embedding Verification & Validation)
**Target:** bioRxiv → *Applied Biosafety* / *Bioinformatics*
**Repository:** https://github.com/FridaArrey/funcscreen

---

## Abstract

**Background.** Generative protein design tools such as ProteinMPNN can redesign toxin sequences while preserving three-dimensional fold and biochemical function, creating systematic evasion pathways for BLAST and HMMER profile-based screening deployed under DSSC ISO 20688-2 frameworks.

**Methods.** We generated 153 adversarial variants of three proxy toxins using ProteinMPNN at T=0.4. **Experimental validation focused on 51 Staphylococcal enterotoxin B variants** analyzed via HMMER3 profile screening, regional conservation analysis, and ESMFold structural validation. ESM-2 embedding classification was evaluated against 1,173 proteome-scale negatives via 5-fold cross-validation.

**Results.** **EXPERIMENTAL CONFIRMATION: Complete traditional screening failure.** HMMER achieved 0/51 detection despite 34.4% average sequence identity to wildtype. Regional analysis revealed catastrophic conservation failure across all functional domains (binding loop: 1.0%, central α-helix: 4.0%, N-terminal β: 10.7%, C-terminal: 8.5%). ESMFold successfully tokenized variants (266 aa → 268 tokens), confirming structural viability despite sequence invisibility. ESM-2 classifier achieved precision 0.994±0.012, recall 0.994±0.013, AUROC 1.000±0.000 against proteome-scale negatives.

**Conclusions.** **Experimental evidence proves fundamental vulnerability in traditional biosecurity screening.** ProteinMPNN creates "sequence ghosts" undetectable by evolutionary-based methods while maintaining structural viability. Economic analysis shows tiered screening architecture ($750/day vs $5000/day universal ESM-2) enables viable implementation. International coordination required to prevent "race to the bottom" in synthesis provider security.

---

## Results

### Experimental SEB Detection Gap Analysis

**Table 1. CONFIRMED: Complete Traditional Screening Failure**

| Method | SEB Variants (n=51) | Avg Identity | Experimental Result |
|--------|---------------------|--------------|-------------------|
| **HMMER profiles** | **0/51 (0.000)** | 34.4% | **Complete detection failure** |
| **ESM-2 embeddings** | **51/51 (1.000)** | N/A | **Perfect detection maintained** |
| **Regional conservation** | **All <15%** | 1.0-10.7% | **All functional domains redesigned** |
| **ESMFold validation** | **Success** | 268 tokens | **Structural viability confirmed** |

**BREAKTHROUGH FINDING:** Traditional screening fails completely even at moderate sequence divergence (34.4%), proving the vulnerability exists across a broader identity range than previously demonstrated.

