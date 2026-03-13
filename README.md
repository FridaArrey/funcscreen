# funcscreen 🧬🛡️

**Closing the "Security Gap" in Biosecurity via Structure-Aware AI Embeddings.**

`funcscreen` is a technical proof-of-concept demonstrating how generative protein design (ProteinMPNN) can obfuscate biological threats to evade traditional sequence-based screening (BLAST), and how AI-latent embeddings (ESM-2) provide a robust defense.

## 📌 Project Overview
As AI-driven protein engineering matures, the "letter-by-letter" sequence identity of a toxin can be radically altered while preserving its functional 3D fold. This repository implements the **TEVV (Toxin Embedding Verification & Validation)** framework to bridge this gap.

### The Problem
* **Traditional Screening (BLAST):** Fails when sequence identity drops below 30%, even if the protein remains a functional threat.
* **The Risk:** "Stealth Toxins" can be synthesized by DNA providers who rely solely on sequence-matching guardrails.

## 📊 Key Results: Sequence vs. Embedding
Using **Ricin A-Chain** as a proxy, our benchmarks show:

| Metric | BLAST (Baseline) | AI Classifier (ESM-2) |
| :--- | :--- | :--- |
| **Recall (Threat Detection)** | 0.50 (Misses 50%) | **1.00 (Caught All)** |
| **Precision** | 1.00 | 1.00 |
| **Bypass Catch Rate** | 0% | **100%** |

*Functional variants with **<29% sequence recovery** were correctly flagged by the AI with **>90% confidence**.*

## 🛠️ Tech Stack & Methodology
1.  **Redesign:** Adversarial obfuscation using **ProteinMPNN**.
2.  **Structural Validation:** 3D folding via **ESMFold** + **TM-score** (Threshold: >0.5).
3.  **Latent Detection:** Semantic screening via **ESM-2 (650M)** embeddings and a Logistic Regression classifier.

## 📜 Policy Impact
This research supports the evolution of the **DSSC ISO 20688-2 Implementation Guide**, advocating for a transition from keyword-based screening to **functional-intent screening** to prevent AI-enabled biological dual-use.

## 📂 Repository Structure
* `/src`: Core scripts for redesign, folding, and classification.
* `PREPRINT.md`: The full technical write-up of the findings.
* `master_biosecurity_gap.png`: Visual evidence of the detection gap.

---
**Author:** FridaArrey | **Framework:** TEVV | **Target:** Applied Biosafety / AGI Strategy 2026
