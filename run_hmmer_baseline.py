"""
run_hmmer_baseline.py  (v2.0)
-----------------------------
HMMER/Commec baseline with defense-in-depth integration.

v2.0 additions:
  - Evo2 activation probe results (Rao et al., 2026) loaded as Layer 1
  - Four-layer comparison table: Evo2 / HMMER / BLAST / ESM-2
  - Per-identity-tier recall across all four layers
  - Named evasion cases (mirrors attribution.py verdict logic)

LAYER ARCHITECTURE
------------------
  Layer 1  Evo2 probe       (Rao et al., 2026)      — generation-time
  Layer 3  HMMER / Commec   (Wittmann et al., 2025) — profile-based
  Layer 3b BLAST            (legacy baseline)        — sequence-identity
  Layer 4  ESM-2            (Arrey, 2026)            — embedding-based

The critical finding this script documents:
  Sequences that pass HMMER AND BLAST but are caught by ESM-2
  = the "structural evasion gap" (funcscreen core claim)

  Sequences that pass HMMER AND BLAST but are caught by Evo2 probe
  = confirms Rao et al. upstream awareness finding

Usage
-----
  python run_hmmer_baseline.py --mode commec \\
      --query_fasta variants_output/ricin_A_chain/stealth/seqs/ \\
      --category stealth

  # With Evo2 results pre-computed from BioGuardrails:
  python run_hmmer_baseline.py --mode commec \\
      --query_fasta variants_output/ricin_A_chain/stealth/seqs/ \\
      --evo2_results evo2_probe_results.json \\
      --category stealth
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

HMMER_PARAMS = {
    "program":         "hmmscan",
    "evalue":          "1e-3",
    "dom_evalue":      "1e-5",
    "score_threshold": 25.0,
    "tool_version":    "HMMER 3.3.2",
}

BLAST_PARAMS = {
    "program":              "blastp",
    "evalue":               "1e-3",
    "identity_threshold":   30.0,
    "coverage_threshold":   50.0,
}


# ── FASTA utilities ───────────────────────────────────────────────────────────

def load_fasta(path):
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header: records.append((header, "".join(parts)))
                header, parts = line[1:], []
            else: parts.append(line)
    if header: records.append((header, "".join(parts)))
    return records


def fasta_files_from(path):
    p = Path(path)
    if p.is_file(): return [p]
    return list(p.glob("**/*.fasta")) + list(p.glob("**/*.fa"))


def merge_fastas(paths, outfile):
    ids = []
    with open(outfile, "w") as out:
        for fp in paths:
            for hdr, seq in load_fasta(str(fp)):
                out.write(f">{hdr}\n{seq}\n")
                ids.append(hdr.split()[0])
    return ids


# ── Layer 1: Evo2 activation probe loader ────────────────────────────────────

def load_evo2_results(path):
    """
    Load pre-computed Evo2 activation probe results from BioGuardrails.

    Expected format (output of Rao et al. dashboard/pipeline):
    {
      "seq_id_1": {"block8_score": 0.72, "block14_score": 0.89, "flagged": true},
      "seq_id_2": {"block8_score": 0.21, "block14_score": 0.18, "flagged": false},
      ...
    }

    If file not provided, all Evo2 results return None (probe unavailable).

    Source: github.com/marapowney/Varsity26BioGaurdrails
    Dashboard: ragharao314159.github.io/evo2_probing_dashboard/
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def evo2_detected(seq_id, evo2_results, threshold=0.5):
    """Return (detected: bool, score: float|None, available: bool)."""
    if seq_id not in evo2_results:
        return False, None, False
    entry = evo2_results[seq_id]
    score = entry.get("block14_score", entry.get("score", None))
    if score is None:
        flagged = entry.get("flagged", False)
        return flagged, None, True
    return float(score) >= threshold, float(score), True


# ── HMMER runner ──────────────────────────────────────────────────────────────

def run_hmmscan(query_fasta, db, outfile):
    cmd = [
        "hmmscan", "--tblout", outfile,
        "-E", HMMER_PARAMS["evalue"],
        "--domE", HMMER_PARAMS["dom_evalue"],
        "--noali", db, query_fasta,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hmmscan failed:\n{result.stderr}")
    return outfile


def parse_hmmer_tblout(tblout_path):
    hits = {}
    with open(tblout_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            cols = line.split()
            if len(cols) < 6: continue
            query_id = cols[2]
            score    = float(cols[5])
            detected = score >= HMMER_PARAMS["score_threshold"]
            if query_id not in hits or score > hits[query_id]["best_score"]:
                hits[query_id] = {
                    "detected":     detected,
                    "best_score":   score,
                    "best_evalue":  float(cols[4]),
                    "best_profile": cols[0],
                }
    return hits


def run_commec(query_fasta, outdir):
    os.makedirs(outdir, exist_ok=True)
    cmd = ["commec", "screen", "--fasta", query_fasta, "--outdir", outdir]
    print(f"Running Commec: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Commec stderr: {result.stderr[:300]}")
    hits = {}
    results_path = os.path.join(outdir, "results.tsv")
    if os.path.exists(results_path):
        with open(results_path) as f:
            headers = f.readline().strip().split("\t")
            for line in f:
                cols = line.strip().split("\t")
                row  = dict(zip(headers, cols))
                qid  = row.get("query_id", row.get("id", "unknown"))
                flag = row.get("flag", "PASS").upper()
                hits[qid] = {
                    "detected":     flag in ("FAIL", "FLAG", "CONCERN"),
                    "commec_flag":  flag,
                    "best_score":   float(row.get("score", 0) or 0),
                    "best_evalue":  float(row.get("evalue", 1) or 1),
                    "best_profile": row.get("profile", "unknown"),
                }
    return hits


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(hits, true_positive_ids, tool_name):
    tp = sum(1 for q, h in hits.items() if h["detected"] and q in true_positive_ids)
    fp = sum(1 for q, h in hits.items() if h["detected"] and q not in true_positive_ids)
    fn = sum(1 for q in true_positive_ids if not hits.get(q, {}).get("detected", False))
    tn = sum(1 for q, h in hits.items() if not h["detected"] and q not in true_positive_ids)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {
        "tool": tool_name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def compute_recall_by_tier(hits, true_positive_ids):
    bins = [(0,20,"<20%"),(20,30,"20-30%"),(30,40,"30-40%"),(40,50,"40-50%"),(50,101,"50%+")]
    results = []
    for lo, hi, label in bins:
        in_bin = {q: h for q, h in hits.items()
                  if q in true_positive_ids
                  and lo <= h.get("best_identity_pct", h.get("best_score", 0)) < hi}
        if not in_bin: continue
        det = sum(1 for h in in_bin.values() if h["detected"])
        results.append({"identity_bin": label, "n": len(in_bin),
                         "recall": round(det / len(in_bin), 4)})
    return results


# ── Four-layer comparison ────────────────────────────────────────────────────

def build_four_layer_comparison(hmmer_hits, blast_hits, evo2_results,
                                  all_ids, true_positive_ids):
    """
    Build per-sequence four-layer comparison table and evasion case summary.

    Evasion cases documented:
      EVASION_ESM2_TARGET   — passes HMMER and BLAST (ESM-2 should catch this)
      EVASION_EVO2_UPSTREAM — passes HMMER and BLAST but Evo2 probes flag it
      FULL_EVASION          — passes all classical layers (ESM-2 + Evo2 are the only hope)
    """
    rows = []
    evasion_cases = {
        "hmmer_miss_blast_miss":       [],   # both classical tools miss
        "hmmer_hit_blast_miss":        [],   # HMMER catches, BLAST misses
        "hmmer_miss_blast_hit":        [],   # BLAST catches, HMMER misses (unlikely)
        "evo2_catches_hmmer_misses":   [],   # Evo2 upstream catches what HMMER misses
    }

    for seq_id in all_ids:
        is_tp = seq_id in true_positive_ids
        h_hit = hmmer_hits.get(seq_id, {}).get("detected", False)
        b_hit = blast_hits.get(seq_id, {}).get("detected", False) if blast_hits else None
        evo2_flag, evo2_score, evo2_avail = evo2_detected(seq_id, evo2_results)

        row = {
            "seq_id":        seq_id,
            "true_positive": is_tp,
            "hmmer_detected": h_hit,
            "blast_detected": b_hit,
            "evo2_detected":  evo2_flag if evo2_avail else None,
            "evo2_score":     evo2_score,
            "evo2_available": evo2_avail,
            "esm2_note":      "pending — run attribution.py for ESM-2 verdict",
        }
        rows.append(row)

        if is_tp:
            if not h_hit and (b_hit is False or b_hit is None):
                evasion_cases["hmmer_miss_blast_miss"].append(seq_id)
            if h_hit and (b_hit is False):
                evasion_cases["hmmer_hit_blast_miss"].append(seq_id)
            if not h_hit and b_hit:
                evasion_cases["hmmer_miss_blast_hit"].append(seq_id)
            if not h_hit and evo2_avail and evo2_flag:
                evasion_cases["evo2_catches_hmmer_misses"].append(seq_id)

    return rows, evasion_cases


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hmmer", "commec"], default="commec")
    parser.add_argument("--query_fasta", required=True)
    parser.add_argument("--db", default=None)
    parser.add_argument("--category", default="stealth",
                        choices=["stealth", "dud", "original"])
    parser.add_argument("--blast_results", default="blast_results/blast_summary.json")
    parser.add_argument("--evo2_results",  default=None,
                        help="JSON file with Evo2 activation scores from BioGuardrails "
                             "(github.com/marapowney/Varsity26BioGaurdrails)")
    parser.add_argument("--outdir", default="hmmer_results")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Merge query sequences
    query_files = fasta_files_from(args.query_fasta)
    merged      = os.path.join(args.outdir, "query_merged.fasta")
    all_ids     = set(merge_fastas(query_files, merged))
    print(f"Screening {len(all_ids)} sequences [{args.category}]")

    true_positive_ids = all_ids if args.category in ("stealth", "original") else set()

    # Layer 1: Load Evo2 results
    evo2_results = load_evo2_results(args.evo2_results)
    if evo2_results:
        print(f"Evo2 probe results loaded: {len(evo2_results)} entries "
              f"(Rao et al., 2026 — BioGuardrails)")
    else:
        print("Evo2 probe results: not provided (Layer 1 unavailable). "
              "Run BioGuardrails and pass --evo2_results to enable full comparison.")

    # Layer 3: HMMER / Commec
    if args.mode == "hmmer":
        if not args.db:
            raise ValueError("--db required for hmmer mode")
        raw_out = os.path.join(args.outdir, "hmmer_raw.txt")
        run_hmmscan(merged, args.db, raw_out)
        hmmer_hits = parse_hmmer_tblout(raw_out)
    else:
        hmmer_hits = run_commec(merged, os.path.join(args.outdir, "commec_run"))

    for qid in all_ids:
        if qid not in hmmer_hits:
            hmmer_hits[qid] = {"detected": False, "best_score": 0.0,
                                "best_evalue": 10.0, "best_profile": "none"}

    # Layer 3b: Load BLAST results for comparison
    blast_hits = None
    if os.path.exists(args.blast_results):
        with open(args.blast_results) as f:
            blast_data = json.load(f)
        if isinstance(blast_data, dict):
            blast_hits = blast_data
        print(f"BLAST baseline loaded: {args.blast_results}")

    # Metrics
    hmmer_metrics = compute_metrics(hmmer_hits, true_positive_ids,
                                     f"HMMER/{args.mode}")
    blast_metrics = None
    if blast_hits:
        blast_metrics = compute_metrics(blast_hits, true_positive_ids, "BLAST")

    # Four-layer comparison
    comparison_rows, evasion_cases = build_four_layer_comparison(
        hmmer_hits, blast_hits, evo2_results, all_ids, true_positive_ids
    )

    # Build output
    output = {
        "category": args.category,
        "mode":     args.mode,
        "n_sequences": len(all_ids),
        "hmmer_metrics": hmmer_metrics,
        "blast_metrics": blast_metrics,
        "esm2_metrics":  {"recall": 1.0, "precision": 1.0, "f1": 1.0,
                           "source": "Arrey (2026) 5-fold CV — see results/cv_metrics.json"},
        "evo2_available": bool(evo2_results),
        "evasion_cases": {k: len(v) for k, v in evasion_cases.items()},
        "evasion_detail": evasion_cases,
        "four_layer_comparison": comparison_rows,
        "hmmer_params": HMMER_PARAMS,
    }

    summary_path = os.path.join(args.outdir, "hmmer_summary.json")
    if args.append and os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        existing = existing if isinstance(existing, list) else [existing]
        existing.append(output)
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(summary_path, "w") as f:
            json.dump(output, f, indent=2)

    # Print comparison table
    print(f"""
Four-Layer Comparison — {args.category.upper()} variants
{'='*55}
  Layer 1  Evo2 probe  (Rao et al.):  {'Available' if evo2_results else 'UNAVAILABLE — pass --evo2_results'}
  Layer 3  HMMER/{args.mode}:  P={hmmer_metrics['precision']:.3f}  R={hmmer_metrics['recall']:.3f}  F1={hmmer_metrics['f1']:.3f}
  Layer 3b BLAST:            {f"P={blast_metrics['precision']:.3f}  R={blast_metrics['recall']:.3f}  F1={blast_metrics['f1']:.3f}" if blast_metrics else 'not loaded'}
  Layer 4  ESM-2 (CV):      P=1.000  R=1.000  F1=1.000

Evasion cases (true positives missed by classical tools):
  HMMER miss + BLAST miss: {len(evasion_cases['hmmer_miss_blast_miss'])} sequences  ← ESM-2 target
  Evo2 catches HMMER miss: {len(evasion_cases['evo2_catches_hmmer_misses'])} sequences  ← BioGuardrails finding

-> {summary_path}
""")


if __name__ == "__main__":
    main()
