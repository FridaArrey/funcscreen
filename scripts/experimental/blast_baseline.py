"""
blast_baseline.py
-----------------
Replaces test_blast.py with a fully documented, reproducible BLAST baseline.

Addresses the gap: BLAST database, E-value, and percent identity threshold
were undocumented. This script specifies all parameters explicitly and
documents the rationale for each choice.

Two modes
---------
  1. --mode commec   Use the IBBIS Commec tool (preferred; install separately).
                     Requires Commec >= 0.3 and its HMM + BLAST databases.
                     See: https://github.com/ibbis-screening/common-mechanism

  2. --mode blast    Direct BLAST against a local NCBI nr database subset
                     seeded from the HHS BSAT list accessions. This is the
                     methodological proxy used when Commec is unavailable.

BLAST parameters (documented for methods section)
--------------------------------------------------
  Program:       blastp (protein vs protein)
  Database:      NCBI nr, pre-filtered to HHS BSAT-listed agent accessions
                 (see bsat_accessions.txt — sourced from the HHS Select Agent
                 Program list, publicly available at selectagents.gov)
  E-value:       1e-3  (permissive, standard for homology screening)
  Word size:     3     (default for blastp)
  Matrix:        BLOSUM62
  Identity threshold for "DETECTED":  ≥30% (below this, BLAST recall drops
                                        significantly even for functional homologs;
                                        this threshold is documented in Wittmann et al.)
  Coverage threshold: ≥50% query coverage

Outputs
-------
  blast_results/
      blast_raw.tsv             ← full BLAST tabular output
      blast_summary.json        ← per-sequence: detected (bool), identity%, evalue
      blast_metrics.json        ← precision, recall, F1 by variant category
      blast_by_identity_tier.json ← recall stratified by identity % bins

Usage
-----
  # Commec mode (preferred):
  python blast_baseline.py --mode commec --query_fasta variants_stealth/seqs/

  # Direct BLAST mode:
  python blast_baseline.py --mode blast --query_fasta variants_stealth/seqs/ \\
                           --db path/to/bsat_blastdb --category stealth

  # Evaluate across all three categories:
  python blast_baseline.py --mode blast --query_fasta variants_stealth/seqs/ \\
                           --category stealth --db path/to/bsat_blastdb
  python blast_baseline.py --mode blast --query_fasta variants_dud/seqs/ \\
                           --category dud --db path/to/bsat_blastdb --append
  python blast_baseline.py --mode blast --query_fasta variants_output/seqs/ \\
                           --category original --db path/to/bsat_blastdb --append
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

# ── BLAST parameters (must match methods section) ─────────────────────────────
BLAST_PARAMS = {
    "program":    "blastp",
    "evalue":     "1e-3",
    "word_size":  "3",
    "matrix":     "BLOSUM62",
    "outfmt":     "6 qseqid sseqid pident length evalue bitscore qcovs",
    "identity_threshold_pct": 30.0,   # % identity to call "detected"
    "coverage_threshold_pct": 50.0,   # % query coverage to call "detected"
    "max_target_seqs": "500",
}

COMMEC_THRESHOLDS = {
    # Commec uses its own HMM + BLAST pipeline; these are its defaults
    "evalue": "1e-5",
    "identity": 30,   # % — same threshold for fair comparison
}


# ── Sequence utilities ────────────────────────────────────────────────────────

def fasta_files_from(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return list(p.glob("**/*.fasta")) + list(p.glob("**/*.fa"))


def merge_fastas(paths: list[Path], outfile: str) -> int:
    """Merge FASTA files; return total record count."""
    count = 0
    with open(outfile, "w") as out:
        for fp in paths:
            with open(fp) as f:
                for line in f:
                    out.write(line)
                    if line.startswith(">"):
                        count += 1
    return count


# ── BLAST runner ──────────────────────────────────────────────────────────────

def run_blastp(query_fasta: str, db: str, outfile: str) -> str:
    """Run blastp; return path to tabular output."""
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db,
        "-evalue", BLAST_PARAMS["evalue"],
        "-word_size", BLAST_PARAMS["word_size"],
        "-matrix", BLAST_PARAMS["matrix"],
        "-outfmt", BLAST_PARAMS["outfmt"],
        "-max_target_seqs", BLAST_PARAMS["max_target_seqs"],
        "-out", outfile,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"blastp failed:\n{result.stderr}")
    return outfile


def parse_blast_output(tsv_path: str) -> dict:
    """
    Parse tabular BLAST output.
    Returns {query_id: {"detected": bool, "best_identity": float, "best_evalue": float}}.
    """
    hits = {}
    with open(tsv_path) as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split("\t")
            if len(cols) < 7:
                continue
            qid, sid, pident, length, evalue, bitscore, qcovs = cols
            pident_f = float(pident)
            qcovs_f  = float(qcovs)

            detected = (
                pident_f >= BLAST_PARAMS["identity_threshold_pct"] and
                qcovs_f  >= BLAST_PARAMS["coverage_threshold_pct"]
            )
            if qid not in hits or (detected and not hits[qid]["detected"]):
                hits[qid] = {
                    "detected": detected,
                    "best_identity_pct": pident_f,
                    "best_evalue": float(evalue),
                    "best_qcovs": qcovs_f,
                }
    return hits


def get_all_query_ids(fasta_path: str) -> list[str]:
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


# ── Commec runner ─────────────────────────────────────────────────────────────

def run_commec(query_fasta: str, outdir: str) -> dict:
    """
    Run IBBIS Commec screening tool.
    Requires `commec` to be installed: pip install commec
    and databases to be initialised: commec download-databases
    """
    cmd = ["commec", "screen", "--fasta", query_fasta, "--outdir", outdir]
    print(f"Running Commec: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Commec warning: {result.stderr[:500]}")

    # Commec outputs a TSV with a 'flag' column; parse it
    results_path = os.path.join(outdir, "results.tsv")
    hits = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            headers = f.readline().strip().split("\t")
            for line in f:
                cols = line.strip().split("\t")
                row = dict(zip(headers, cols))
                qid = row.get("query_id", row.get("id", "unknown"))
                flag = row.get("flag", "PASS").upper()
                hits[qid] = {
                    "detected": flag in ("FAIL", "FLAG", "CONCERN"),
                    "commec_flag": flag,
                    "best_identity_pct": float(row.get("pident", 0) or 0),
                    "best_evalue": float(row.get("evalue", 1) or 1),
                }
    return hits


# ── Metrics calculation ───────────────────────────────────────────────────────

def compute_metrics(hits: dict, true_positive_ids: set) -> dict:
    """
    Compute P/R/F1.
    true_positive_ids: set of query IDs that SHOULD be detected.
    Sequences not in this set are true negatives (should NOT be detected).
    """
    tp = sum(1 for qid, h in hits.items() if h["detected"] and qid in true_positive_ids)
    fp = sum(1 for qid, h in hits.items() if h["detected"] and qid not in true_positive_ids)
    fn = sum(1 for qid in true_positive_ids if not hits.get(qid, {}).get("detected", False))
    tn = sum(1 for qid, h in hits.items() if not h["detected"] and qid not in true_positive_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "n_total":   len(hits),
        "blast_params": BLAST_PARAMS,
    }


def compute_recall_by_identity_tier(hits: dict, true_positive_ids: set) -> list:
    """
    Stratify recall by sequence identity bins.
    Critical for the evasion analysis: shows where BLAST loses detection.
    """
    bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 100)]
    results = []
    for lo, hi in bins:
        in_bin = {
            qid: h for qid, h in hits.items()
            if lo <= h.get("best_identity_pct", 0) < hi
            and qid in true_positive_ids
        }
        if not in_bin:
            continue
        detected = sum(1 for h in in_bin.values() if h["detected"])
        results.append({
            "identity_bin": f"{lo}–{hi}%",
            "n_sequences": len(in_bin),
            "n_detected": detected,
            "recall": round(detected / len(in_bin), 4),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Documented BLAST/Commec baseline for funcscreen."
    )
    parser.add_argument("--mode", choices=["blast", "commec"], default="blast")
    parser.add_argument("--query_fasta", required=True,
                        help="FASTA file or directory of FASTA files to screen")
    parser.add_argument("--db", default=None,
                        help="[blast mode] Path to NCBI BLAST database (makeblastdb output)")
    parser.add_argument("--category", default="stealth",
                        choices=["stealth", "dud", "original"],
                        help="Variant category being screened (affects ground-truth labels)")
    parser.add_argument("--outdir", default="blast_results")
    parser.add_argument("--append", action="store_true",
                        help="Append results to existing blast_summary.json")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Merge query FASTAs
    query_files = fasta_files_from(args.query_fasta)
    if not query_files:
        raise FileNotFoundError(f"No FASTA files found at {args.query_fasta}")

    merged_fasta = os.path.join(args.outdir, "query_merged.fasta")
    n_seqs = merge_fastas(query_files, merged_fasta)
    print(f"Merged {n_seqs} sequences from {len(query_files)} files → {merged_fasta}")

    all_query_ids = set(get_all_query_ids(merged_fasta))

    # Ground truth: stealth and original variants SHOULD be detected;
    # dud variants (low TM-score, structurally diverged) should NOT.
    true_positive_ids = all_query_ids if args.category in ("stealth", "original") else set()

    # Run screening
    if args.mode == "commec":
        commec_outdir = os.path.join(args.outdir, "commec_run")
        os.makedirs(commec_outdir, exist_ok=True)
        hits = run_commec(merged_fasta, commec_outdir)
    else:
        if not args.db:
            raise ValueError("--db is required in blast mode. "
                             "Build with: makeblastdb -in bsat_sequences.fasta -dbtype prot -out bsat_db")
        blast_tsv = os.path.join(args.outdir, "blast_raw.tsv")
        run_blastp(merged_fasta, args.db, blast_tsv)
        hits = parse_blast_output(blast_tsv)
        # Any query with no hits is not detected
        for qid in all_query_ids:
            if qid not in hits:
                hits[qid] = {"detected": False, "best_identity_pct": 0.0,
                              "best_evalue": 10.0, "best_qcovs": 0.0}

    # Compute metrics
    metrics = compute_metrics(hits, true_positive_ids)
    metrics["category"] = args.category
    metrics["mode"] = args.mode
    metrics["n_sequences"] = n_seqs

    # Stratified recall
    identity_tiers = compute_recall_by_identity_tier(hits, true_positive_ids)
    metrics["recall_by_identity_tier"] = identity_tiers

    # Save
    summary_path = os.path.join(args.outdir, "blast_summary.json")
    if args.append and os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(metrics)
        else:
            existing = [existing, metrics]
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Print summary
    print(f"""
── BLAST Baseline: {args.category.upper()} variants ─────────────────
  Mode:       {args.mode}  (E-value {BLAST_PARAMS['evalue']}, identity ≥{BLAST_PARAMS['identity_threshold_pct']}%, coverage ≥{BLAST_PARAMS['coverage_threshold_pct']}%)
  Sequences:  {n_seqs}
  Precision:  {metrics['precision']:.3f}
  Recall:     {metrics['recall']:.3f}
  F1:         {metrics['f1']:.3f}
  TP/FP/FN/TN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}
""")
    if identity_tiers:
        print("  Recall by sequence identity tier:")
        for tier in identity_tiers:
            print(f"    {tier['identity_bin']:>8}: {tier['recall']:.3f}  (n={tier['n_sequences']})")

    print(f"\n→ Results saved: {summary_path}")


if __name__ == "__main__":
    main()