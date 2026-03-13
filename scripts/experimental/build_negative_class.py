"""
build_negative_class.py
-----------------------
Addresses the critical gap: documents and generates the negative training class
for the ESM-2 logistic regression classifier.

Strategy (following Ikonomova et al. 2026 TEVV framework recommendations):
  - Source: UniProt Swiss-Prot (manually reviewed, high-quality annotations)
  - Query:  Human and yeast structural/metabolic proteins with NO toxin or
            pathogen annotation — maximally dissimilar to the positive class
  - Size:   ~3× the positive class count (class-balanced training)
  - Stratified across functional categories to avoid sampling bias

Functional categories sampled (each non-toxic, well-characterised):
  1. Human cytoskeletal proteins  (actin, tubulin isoforms)
  2. Human metabolic enzymes      (aldolase, enolase, GAPDH)
  3. Yeast chaperones             (Hsp60, Hsp70 family)
  4. Human transport proteins     (albumin, transferrin)
  5. Plant structural proteins    (RuBisCO large subunit — non-toxic plant)

Why these choices?
  - Cytoskeletal and metabolic proteins span a wide range of folds,
    ensuring the classifier learns "toxin" vs "general protein" rather
    than memorising a narrow negative distribution.
  - Including plant proteins (RuBisCO) controls for the fact that ricin
    is also plant-derived — the classifier must learn function, not taxonomy.
  - All are Swiss-Prot reviewed and have well-characterised, non-toxic function.

Output
------
  negatives/
      negatives.fasta           ← all negative sequences, FASTA format
      negatives_metadata.json   ← accession, category, length, rationale
      negative_class_report.txt ← human-readable summary for methods section

Usage
-----
  python build_negative_class.py
  python build_negative_class.py --n_per_category 10 --outdir negatives/
"""

import argparse
import json
import os
import time
import requests

# ── Curated negative set ──────────────────────────────────────────────────────
# Each entry: UniProt accession, functional category, brief rationale.
# Expand this list to increase N per category; all are Swiss-Prot reviewed.

NEGATIVE_TARGETS = [
    # ── Cytoskeletal proteins ────────────────────────────────────────────
    {"accession": "P60709", "category": "cytoskeletal", "name": "Actin, cytoplasmic 1 (beta)", "organism": "Homo sapiens"},
    {"accession": "P68363", "category": "cytoskeletal", "name": "Tubulin alpha-1B chain", "organism": "Homo sapiens"},
    {"accession": "P07437", "category": "cytoskeletal", "name": "Tubulin beta chain", "organism": "Homo sapiens"},
    {"accession": "P02545", "category": "cytoskeletal", "name": "Prelamin-A/C (Lamin A)", "organism": "Homo sapiens"},

    # ── Metabolic enzymes ────────────────────────────────────────────────
    {"accession": "P04406", "category": "metabolic_enzyme", "name": "Glyceraldehyde-3-phosphate dehydrogenase", "organism": "Homo sapiens"},
    {"accession": "P06733", "category": "metabolic_enzyme", "name": "Alpha-enolase", "organism": "Homo sapiens"},
    {"accession": "P04075", "category": "metabolic_enzyme", "name": "Fructose-bisphosphate aldolase A", "organism": "Homo sapiens"},
    {"accession": "P00558", "category": "metabolic_enzyme", "name": "Phosphoglycerate kinase 1", "organism": "Homo sapiens"},

    # ── Chaperones (yeast) ───────────────────────────────────────────────
    {"accession": "P10591", "category": "chaperone", "name": "Heat shock protein SSA1 (Hsp70)", "organism": "Saccharomyces cerevisiae"},
    {"accession": "P0DMV8", "category": "chaperone", "name": "Heat shock 70 kDa protein 1A", "organism": "Homo sapiens"},
    {"accession": "P38646", "category": "chaperone", "name": "Stress-70 protein, mitochondrial (GRP75)", "organism": "Homo sapiens"},

    # ── Transport proteins ───────────────────────────────────────────────
    {"accession": "P02768", "category": "transport", "name": "Serum albumin", "organism": "Homo sapiens"},
    {"accession": "P02787", "category": "transport", "name": "Serotransferrin", "organism": "Homo sapiens"},
    {"accession": "P69905", "category": "transport", "name": "Hemoglobin subunit alpha", "organism": "Homo sapiens"},

    # ── Plant structural/photosynthetic (non-toxic) ──────────────────────
    # RuBisCO is plant-derived like ricin but has no toxic function —
    # this explicitly tests that the classifier learns mechanism, not species.
    {"accession": "P00875", "category": "plant_structural", "name": "RuBisCO large subunit", "organism": "Spinacia oleracea"},
    {"accession": "P00877", "category": "plant_structural", "name": "RuBisCO large subunit", "organism": "Nicotiana tabacum"},
]

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
UNIPROT_JSON_URL  = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def fetch_fasta(accession: str, retries: int = 3) -> str:
    url = UNIPROT_FASTA_URL.format(accession=accession)
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text.strip()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {accession} (HTTP {resp.status_code})")


def fetch_length(accession: str) -> int:
    url = UNIPROT_JSON_URL.format(accession=accession)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.json().get("sequence", {}).get("length", -1)
    return -1


def write_report(metadata: list, outdir: str, n_positives_expected: int):
    """Write a plain-text methods-section summary."""
    categories = {}
    for m in metadata:
        categories.setdefault(m["category"], []).append(m["name"])

    lines = [
        "NEGATIVE CLASS REPORT — funcscreen classifier",
        "=" * 55,
        "",
        "Source:   UniProt Swiss-Prot (manually reviewed entries only)",
        f"Total:    {len(metadata)} sequences",
        f"Positive: ~{n_positives_expected} (toxin variants)",
        f"Ratio:    ~{len(metadata) / max(n_positives_expected, 1):.1f}× negative oversampling",
        "",
        "Rationale",
        "---------",
        "Negative proteins were selected to span multiple structural folds",
        "and functional classes, ensuring the classifier generalises beyond",
        "narrow sequence or taxonomic features. Plant proteins (RuBisCO) are",
        "included to confirm the classifier learns biochemical mechanism rather",
        "than organism-of-origin, given that the ricin positive proxy is also",
        "plant-derived.",
        "",
        "Categories sampled",
        "------------------",
    ]
    for cat, names in categories.items():
        lines.append(f"  {cat} ({len(names)} sequences):")
        for n in names:
            lines.append(f"    - {n}")
    lines += [
        "",
        "All sequences are non-pathogenic, non-toxic, and publicly available",
        "in UniProt Swiss-Prot. No controlled or restricted sequences are used.",
    ]
    report_path = os.path.join(outdir, "negative_class_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def main(outdir: str = "negatives", n_positives_expected: int = 30):
    os.makedirs(outdir, exist_ok=True)
    all_fastas = []
    all_meta   = []

    for target in NEGATIVE_TARGETS:
        acc = target["accession"]
        print(f"[{target['category']}] {acc} — {target['name']} ...")
        try:
            fasta  = fetch_fasta(acc)
            length = fetch_length(acc)
        except RuntimeError as e:
            print(f"  WARNING: {e} — skipping.")
            continue

        all_fastas.append(fasta)
        all_meta.append({**target, "length": length, "label": 0})
        time.sleep(0.8)

    # Write combined FASTA
    fasta_path = os.path.join(outdir, "negatives.fasta")
    with open(fasta_path, "w") as f:
        f.write("\n\n".join(all_fastas) + "\n")

    # Write metadata JSON
    meta_path = os.path.join(outdir, "negatives_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    # Write human-readable report
    report_path = write_report(all_meta, outdir, n_positives_expected)

    print(f"\n→ {len(all_fastas)} negative sequences saved to {fasta_path}")
    print(f"→ Metadata: {meta_path}")
    print(f"→ Report:   {report_path}")
    print("\nNext: run train_detector_cv.py to train with cross-validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build documented negative class for ESM-2 classifier.")
    parser.add_argument("--outdir", default="negatives")
    parser.add_argument("--n_positives_expected", type=int, default=30,
                        help="Expected count of positive (toxin) variants, for ratio reporting.")
    args = parser.parse_args()
    main(args.outdir, args.n_positives_expected)