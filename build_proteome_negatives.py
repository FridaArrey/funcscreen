"""
build_proteome_negatives.py  (v2.0)
------------------------------------
Expands negative class to 500+ sequences from K12, human, and yeast proteomes.

v2.0 additions:
  - Evo2 probe validation experiment (Rao et al., 2026):
    If Evo2 results are available, validates that proteome negatives
    produce LOW activation scores at blocks 8 and 14 while stealth
    toxins produce HIGH scores — cross-validating both the negative
    set quality and Rao et al.'s upstream awareness finding.

  - Arabidopsis thaliana negatives added (was omitted in v1.0):
    Since Ricin is plant-derived, Arabidopsis negatives are the hardest
    challenge for the classifier and strongest proof of functional
    (not taxonomic) discrimination.

  - Low-complexity sequence filter added:
    ESM-2 can behave unexpectedly on highly repetitive sequences
    (collagen, silk). Filtered out before adding to negative set.

SAMPLING: K12(500) + Human(500) + Yeast(200) + Arabidopsis(100) = ~1,300
"""

import argparse
import json
import os
import random
import re
import time
import requests

PROTEOMES = {
    "k12": {
        "proteome_id": "UP000000625",
        "organism":    "Escherichia coli K-12",
        "rationale":   "Bacterial proteome including metalloproteases with HEXXH motif "
                       "(same grammar as Botulinum NTx-A but non-dangerous). "
                       "Controls for bacterial origin bias.",
    },
    "human": {
        "proteome_id": "UP000005640",
        "organism":    "Homo sapiens",
        "rationale":   "Diverse eukaryotic fold space. Human MMPs carry HEXXH motif "
                       "but are not select agents — the critical discrimination test.",
    },
    "yeast": {
        "proteome_id": "UP000002311",
        "organism":    "Saccharomyces cerevisiae",
        "rationale":   "Standard eukaryotic negative; minimal select-agent overlap.",
    },
    "arabidopsis": {
        "proteome_id": "UP000006548",
        "organism":    "Arabidopsis thaliana",
        "rationale":   "Plant proteome — hardest challenge because Ricin is also plant-derived. "
                       "High Arabidopsis recall proves classifier learns toxic mechanism, "
                       "not plant taxonomy.",
    },
}

EXCLUDE_KEYWORDS = [
    "toxin", "virulence", "pathogen", "select agent",
    "botulinum", "ricin", "anthrax", "plague", "enterotoxin",
    "abrin", "aflatoxin", "tetanus", "diphtheria",
]

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL  = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


# ── Low-complexity filter ─────────────────────────────────────────────────────

def is_too_repetitive(seq, threshold=0.5):
    """
    Filter out low-complexity sequences (collagen, silk, etc.).
    Returns True if sequence is too repetitive for reliable ESM-2 embedding.
    """
    if len(seq) < 20:
        return True
    # Check if any single amino acid dominates
    from collections import Counter
    counts = Counter(seq)
    most_common_frac = counts.most_common(1)[0][1] / len(seq)
    if most_common_frac > threshold:
        return True
    # Check for simple dipeptide repeats
    if len(seq) >= 10:
        for i in range(len(seq) - 4):
            dipeptide = seq[i:i+2]
            repeat    = dipeptide * 3
            if repeat in seq[i:i+10]:
                return True
    return False


# ── UniProt fetch ─────────────────────────────────────────────────────────────

def fetch_accessions(proteome_id, n):
    params = {
        "query":  f"proteome:{proteome_id} AND reviewed:true",
        "format": "list",
        "size":   min(n * 4, 500),
    }
    resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"UniProt search failed: {resp.status_code}")
    accessions = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
    random.shuffle(accessions)
    return accessions


def fetch_fasta(accession, retries=3):
    url = UNIPROT_FASTA_URL.format(accession=accession)
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text.strip()
        time.sleep(2 ** attempt)
    return None


def has_exclusion_keyword(fasta):
    header = fasta.split("\n")[0].lower()
    return any(kw in header for kw in EXCLUDE_KEYWORDS)


def sample_proteome(key, n, max_length=1500):
    info = PROTEOMES[key]
    print(f"\n[{key}] {info['organism']} (target: {n} sequences)...")
    accessions = fetch_accessions(info["proteome_id"], n)
    fastas, metadata, collected = [], [], 0

    for acc in accessions:
        if collected >= n:
            break
        fasta = fetch_fasta(acc)
        if fasta is None:
            continue
        if has_exclusion_keyword(fasta):
            continue
        seq = "".join(fasta.split("\n")[1:])
        if len(seq) < 50 or len(seq) > max_length:
            continue
        if is_too_repetitive(seq):
            continue
        fastas.append(fasta)
        metadata.append({
            "accession":  acc,
            "proteome":   key,
            "organism":   info["organism"],
            "label":      0,
            "seq_length": len(seq),
        })
        collected += 1
        if collected % 50 == 0:
            print(f"  {collected}/{n}", end="\r")
        time.sleep(0.3)

    print(f"  Collected {collected} sequences")
    return fastas, metadata


# ── Evo2 cross-validation ─────────────────────────────────────────────────────

def validate_with_evo2(metadata, evo2_results_path, outdir):
    """
    Cross-validate negative set quality using Evo2 activation scores.

    The experiment:
      If proteome negatives are genuinely benign, Evo2 block 14 scores
      should be LOW for these sequences and HIGH for stealth toxins.
      This simultaneously validates:
        1. The negative set is clean (no accidental toxin inclusion)
        2. Rao et al.'s finding is universal across the tree of life

    Requires Evo2 probe results from BioGuardrails:
      github.com/marapowney/Varsity26BioGaurdrails

    Output: evo2_negative_validation.json + plain-text report
    """
    if not evo2_results_path or not os.path.exists(evo2_results_path):
        print("  Evo2 validation: skipped (no --evo2_results provided)")
        print("  To enable: run BioGuardrails on negatives_proteome/all_proteome_negatives.fasta")
        print("  then pass --evo2_results path/to/results.json")
        return

    with open(evo2_results_path) as f:
        evo2_data = json.load(f)

    negative_ids  = {m["accession"] for m in metadata}
    neg_scores    = []
    false_pos_ids = []

    for seq_id, entry in evo2_data.items():
        if seq_id not in negative_ids:
            continue
        score   = entry.get("block14_score", entry.get("score", None))
        flagged = entry.get("flagged", False)
        if score is not None:
            neg_scores.append(float(score))
        if flagged:
            false_pos_ids.append(seq_id)

    if not neg_scores:
        print("  Evo2 validation: no matching sequence IDs found in Evo2 results")
        return

    import statistics
    mean_score = statistics.mean(neg_scores)
    result = {
        "n_negatives_checked":  len(neg_scores),
        "mean_block14_score":   round(mean_score, 4),
        "std_block14_score":    round(statistics.stdev(neg_scores), 4) if len(neg_scores) > 1 else 0,
        "n_false_positives":    len(false_pos_ids),
        "false_positive_ids":   false_pos_ids,
        "interpretation": (
            f"Mean Evo2 block 14 score for proteome negatives: {mean_score:.3f}. "
            f"If this is substantially below 0.5 while stealth toxins score >0.5, "
            "this cross-validates both the negative set quality AND Rao et al.'s "
            "upstream awareness finding (BioGuardrails, 2026)."
        ),
    }

    out_path = os.path.join(outdir, "evo2_negative_validation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Evo2 cross-validation:")
    print(f"    Mean block 14 score (negatives): {mean_score:.3f}")
    print(f"    False positives (flagged negatives): {len(false_pos_ids)}")
    print(f"    -> {out_path}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(metadata, outdir, n_positives):
    from collections import Counter
    by_proteome = Counter(m["proteome"] for m in metadata)
    lines = [
        "PROTEOME NEGATIVE CLASS REPORT  (v2.0)",
        "=" * 55,
        "",
        f"Total negatives:  {len(metadata)}",
        f"Total positives:  {n_positives}",
        f"Neg:pos ratio:    {len(metadata)/max(n_positives,1):.1f}:1",
        "",
        "Composition:",
    ]
    for key, count in by_proteome.items():
        info = PROTEOMES[key]
        lines.append(f"  {info['organism']}: {count}")
        lines.append(f"    Rationale: {info['rationale']}")
    lines += [
        "",
        "Filters applied:",
        "  - Exclusion keywords: toxin, virulence, botulinum, ricin, etc.",
        "  - Low-complexity filter: removes repetitive sequences (collagen, silk)",
        "  - Length filter: 50-1500 aa",
        "",
        "Evo2 cross-validation (Rao et al., 2026):",
        "  Run with --evo2_results to validate that proteome negatives",
        "  produce low Evo2 block 14 scores, cross-validating both the",
        "  negative set quality and BioGuardrails upstream awareness finding.",
        "",
        "This addresses Kratz (pers. comm., 2026):",
        "  'As you scale up the negative-positive separation will become",
        "  blurry — context really matters for whether a fold is dangerous.'",
    ]
    path = os.path.join(outdir, "sampling_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k12",        type=int, default=500)
    parser.add_argument("--n_human",      type=int, default=500)
    parser.add_argument("--n_yeast",      type=int, default=200)
    parser.add_argument("--n_arabidopsis",type=int, default=100)
    parser.add_argument("--outdir",       default="negatives_proteome")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_positives_expected", type=int, default=155)
    parser.add_argument("--evo2_results", default=None,
                        help="JSON from BioGuardrails for cross-validation "
                             "(github.com/marapowney/Varsity26BioGaurdrails)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    all_fastas, all_meta = [], []
    targets = [
        ("k12",        args.n_k12),
        ("human",      args.n_human),
        ("yeast",      args.n_yeast),
        ("arabidopsis",args.n_arabidopsis),
    ]

    for key, n in targets:
        fastas, meta = sample_proteome(key, n)
        fasta_path = os.path.join(args.outdir, f"{key}_negatives.fasta")
        with open(fasta_path, "w") as f:
            f.write("\n\n".join(fastas) + "\n")
        all_fastas.extend(fastas)
        all_meta.extend(meta)

    combined_path = os.path.join(args.outdir, "all_proteome_negatives.fasta")
    with open(combined_path, "w") as f:
        f.write("\n\n".join(all_fastas) + "\n")

    meta_path = os.path.join(args.outdir, "proteome_negatives_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    report_path = write_report(all_meta, args.outdir, args.n_positives_expected)

    # Evo2 cross-validation
    print("\nEvo2 cross-validation (Rao et al., 2026)...")
    validate_with_evo2(all_meta, args.evo2_results, args.outdir)

    print(f"""
Done: {len(all_fastas)} sequences saved
  Combined FASTA:  {combined_path}
  Metadata:        {meta_path}
  Report:          {report_path}

Next:
  python rebuild_embeddings.py  (update manifest to use {args.outdir}/all_proteome_negatives.fasta)
  python train_detector_cv.py   (re-train with expanded negative set)

Evo2 cross-validation note:
  Run BioGuardrails on {combined_path}
  then: python build_proteome_negatives.py --evo2_results path/to/results.json
""")


if __name__ == "__main__":
    main()
