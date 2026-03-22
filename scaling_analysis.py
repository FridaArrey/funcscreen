"""
scaling_analysis.py  (v2.0)
----------------------------
Compute cost analysis for the full four-layer defense-in-depth pipeline.

v2.0 additions:
  - Layer 0: Evo2 activation probe (Rao et al., 2026) added to tiered model
    Runs at generation time — zero marginal cost at the synthesis gate
  - Full four-tier funnel: Evo2 → FoldSeek → HMMER → ESM-2
  - Cost model updated: Evo2 monitoring is a model-side overhead, not
    a screening overhead — synthesis providers don't pay for it
  - Throughput table updated with realistic four-layer estimates

TIERED STRATEGY (v2.0)
-----------------------
  Layer 0 — Evo2 activation probe     (generation time, model-side)
    Cost to synthesis provider: ZERO (runs inside generative model)
    Operated by: AI tool providers (Evo2, EvoDiff, successors)
    Catches: sequences the model itself flags as pathogenic at Block 14

  Layer 1 — FoldSeek 3Di              (~1ms/sequence, CPU)
    Eliminates structurally dissimilar sequences

  Layer 2 — HMMER3 / Commec           (~100ms/sequence, CPU)
    Catches classical homologs (current deployed standard)

  Layer 3 — ESM-2 embedding           (~3s/sequence, CPU)
    Catches AI-redesigned variants that evade Layers 1-2
    Applied to <1% of total orders under realistic threat prevalence
"""

import argparse
import csv
import json
import os
import time
import numpy as np


LENGTH_DISTRIBUTIONS = {
    "short":  (50,   200,  "Short peptides"),
    "medium": (200,  600,  "Typical gene-length (most common)"),
    "long":   (600, 1200,  "Large domain / full-length"),
}

BATCH_SIZES = [1, 4, 8, 16]


# ── ESM-2 benchmarking ────────────────────────────────────────────────────────

def benchmark_esm2(sequences, device, batch_size=8):
    """
    Benchmark ESM-2 latency using v2.0 residue-only pooling.
    (Matches embed_sequences_batch in attribution.py v2.0)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model     = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()

    timings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch  = sequences[i : i + batch_size]
            t0     = time.perf_counter()
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=1022, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out    = model(**tokens)

            # v2.0 residue-only pooling
            residue_mask = tokens["attention_mask"].clone().float()
            residue_mask[:, 0] = 0
            for j in range(len(batch)):
                seq_len = int(tokens["attention_mask"][j].sum().item())
                if seq_len > 1:
                    residue_mask[j, seq_len - 1] = 0
            mask_exp = residue_mask.unsqueeze(-1)
            _ = (out.last_hidden_state * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

            elapsed = time.perf_counter() - t0
            timings.extend([elapsed / len(batch)] * len(batch))

    return {
        "n":         len(sequences),
        "batch_size": batch_size,
        "mean_s":    round(float(np.mean(timings)), 4),
        "std_s":     round(float(np.std(timings)), 4),
        "total_s":   round(float(sum(timings)), 4),
    }


def generate_dummy_sequences(n, length_range):
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    lo, hi = length_range
    return [
        "".join(np.random.choice(aa, np.random.randint(lo, hi)))
        for _ in range(n)
    ]


# ── Throughput projection ─────────────────────────────────────────────────────

def project_throughput(per_seq_s):
    return {
        "per_seq_seconds":   round(per_seq_s, 3),
        "seqs_per_hour":     round(3600 / per_seq_s),
        "seqs_per_day_1cpu": round(3600 * 24 / per_seq_s),
        "cpus_for_10k_8h":   round(10000 / (3600 / per_seq_s * 8)),
    }


# ── Four-tier funnel projection ───────────────────────────────────────────────

def project_four_tier_funnel(daily_orders):
    """
    Projects volume and cost at each of four defensive layers.

    Layer 0 (Evo2) runs at generation time inside the AI tool —
    zero marginal cost to synthesis providers.
    """
    # Assumptions (conservative)
    evo2_catch_rate   = 0.30   # 30% of threats caught at generation by Evo2 probe
    foldseek_pass     = 0.05   # 5% of orders pass FoldSeek to HMMER
    hmmer_pass        = 0.10   # 10% of HMMER inputs unresolved → ESM-2

    tier0_vol = int(daily_orders * evo2_catch_rate)   # caught at generation
    tier1_vol = daily_orders                           # all orders through FoldSeek
    tier2_vol = int(daily_orders * foldseek_pass)
    tier3_vol = int(tier2_vol * hmmer_pass)

    return {
        "daily_orders": daily_orders,
        "layer0_evo2": {
            "volume":       tier0_vol,
            "latency":      "0ms (generation-time, model-side)",
            "cost_to_provider": "ZERO — runs inside generative AI tool",
            "tool":         "Evo2 activation probing (Rao et al., 2026)",
            "note":         "Catches threats during generation; synthesis order may never be placed",
        },
        "layer1_foldseek": {
            "volume":    tier1_vol,
            "latency_ms": 1,
            "cpu_hours": round((tier1_vol * 0.001) / 3600, 2),
            "tool":      "FoldSeek 3Di structural alphabet",
        },
        "layer2_hmmer": {
            "volume":    tier2_vol,
            "latency_ms": 100,
            "cpu_hours": round((tier2_vol * 0.1) / 3600, 2),
            "tool":      "HMMER3 pHMM / IBBIS Commec",
        },
        "layer3_esm2": {
            "volume":   tier3_vol,
            "latency_s": 3.0,
            "cpu_hours": round((tier3_vol * 3.0) / 3600, 2),
            "tool":     "ESM-2 650M embedding classifier (funcscreen)",
            "note":     f"Only {tier3_vol}/{daily_orders} orders reach ESM-2 "
                        f"({100*tier3_vol/daily_orders:.2f}%)",
        },
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(benchmark, throughput, funnel_10k, funnel_100k, outpath):
    lines = [
        "COMPUTE SCALING ANALYSIS  (v2.0 — Four-Layer Defense-in-Depth)",
        "=" * 60,
        "",
        "v2.0 change: Added Layer 0 (Evo2 activation probe, Rao et al., 2026)",
        "Evo2 runs at generation time inside the AI tool — zero cost to providers.",
        "",
        f"ESM-2 650M measured latency (medium sequences, CPU): "
        f"{benchmark.get('mean_s', 'N/A')}s/sequence",
        f"Throughput: {throughput.get('seqs_per_hour', 'N/A')} sequences/hour/CPU",
        "",
        "FOUR-TIER FUNNEL (10K orders/day)",
        "-" * 40,
        "Layer 0 — Evo2 probe (Rao et al., 2026):",
        f"  ~{funnel_10k['layer0_evo2']['volume']} threats caught at generation",
        f"  Cost to synthesis provider: {funnel_10k['layer0_evo2']['cost_to_provider']}",
        "",
        "Layer 1 — FoldSeek 3Di:",
        f"  {funnel_10k['layer1_foldseek']['volume']} sequences, "
        f"{funnel_10k['layer1_foldseek']['cpu_hours']} CPU-hours/day",
        "",
        "Layer 2 — HMMER3 / Commec:",
        f"  {funnel_10k['layer2_hmmer']['volume']} sequences, "
        f"{funnel_10k['layer2_hmmer']['cpu_hours']} CPU-hours/day",
        "",
        "Layer 3 — ESM-2 (funcscreen):",
        f"  {funnel_10k['layer3_esm2']['volume']} sequences, "
        f"{funnel_10k['layer3_esm2']['cpu_hours']} CPU-hours/day",
        f"  {funnel_10k['layer3_esm2']['note']}",
        "",
        "KEY INSIGHT (v2.0):",
        "  With Evo2 monitoring at Layer 0, many threats never reach the",
        "  synthesis gate at all. ESM-2 at Layer 3 catches the remainder",
        "  that evade classical screening. The two layers are complementary:",
        "  Evo2 prevents generation; funcscreen catches what was generated.",
        "  Neither is sufficient alone. (Arrey, 2026 + Rao et al., 2026)",
        "",
        "Reference: Kratz, M. (pers. comm., 2026)",
        "           Rao, R. et al. (2026) BioGuardrails, Varsity Hackathon 2026",
    ]
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--n_benchmark", type=int, default=20)
    parser.add_argument("--outdir",      default="scaling_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(42)

    print("Benchmarking ESM-2 (v2.0 residue-only pooling)...")
    all_results = {}

    for key, (lo, hi, desc) in LENGTH_DISTRIBUTIONS.items():
        print(f"[{key}] {desc}...")
        seqs    = generate_dummy_sequences(args.n_benchmark, (lo, hi))
        results = benchmark_esm2(seqs, args.device, batch_size=8)
        results.update({"length_range": f"{lo}-{hi}aa", "description": desc})
        all_results[key] = results
        print(f"  Mean: {results['mean_s']:.3f}s ± {results['std_s']:.3f}s")

    medium_per_seq = all_results["medium"]["mean_s"]
    throughput     = project_throughput(medium_per_seq)
    funnel_10k     = project_four_tier_funnel(10_000)
    funnel_100k    = project_four_tier_funnel(100_000)

    # Save
    with open(os.path.join(args.outdir, "latency_benchmark.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    full_output = {
        "esm2_benchmark":  all_results,
        "throughput":      throughput,
        "funnel_10k":      funnel_10k,
        "funnel_100k":     funnel_100k,
        "evo2_note":       "Layer 0 (Evo2) runs at generation time inside AI tools. "
                           "Zero cost to synthesis providers. "
                           "Source: Rao et al. (2026) BioGuardrails. "
                           "Dashboard: ragharao314159.github.io/evo2_probing_dashboard/",
    }
    with open(os.path.join(args.outdir, "throughput_table.json"), "w") as f:
        json.dump(full_output, f, indent=2)

    # CSV table
    csv_path = os.path.join(args.outdir, "throughput_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Tool", "Latency", "Volume (10K/day)",
                          "CPU-hours/day", "Cost to provider"])
        writer.writerow(["0", "Evo2 probe (Rao et al.)", "0ms (gen-time)",
                          funnel_10k["layer0_evo2"]["volume"], 0, "ZERO"])
        writer.writerow(["1", "FoldSeek 3Di", "~1ms",
                          funnel_10k["layer1_foldseek"]["volume"],
                          funnel_10k["layer1_foldseek"]["cpu_hours"], "Low"])
        writer.writerow(["2", "HMMER3 / Commec", "~100ms",
                          funnel_10k["layer2_hmmer"]["volume"],
                          funnel_10k["layer2_hmmer"]["cpu_hours"], "Medium"])
        writer.writerow(["3", "ESM-2 650M (funcscreen)", f"~{medium_per_seq:.1f}s",
                          funnel_10k["layer3_esm2"]["volume"],
                          funnel_10k["layer3_esm2"]["cpu_hours"], "High (A100 recommended)"])

    report_path = os.path.join(args.outdir, "scaling_report.txt")
    write_report(all_results.get("medium", {}), throughput,
                 funnel_10k, funnel_100k, report_path)

    print(f"""
Four-tier funnel (10K orders/day):
  Layer 0 Evo2:      {funnel_10k['layer0_evo2']['volume']} caught at generation (cost: ZERO to providers)
  Layer 1 FoldSeek:  {funnel_10k['layer1_foldseek']['volume']} sequences, {funnel_10k['layer1_foldseek']['cpu_hours']} CPU-hrs
  Layer 2 HMMER:     {funnel_10k['layer2_hmmer']['volume']} sequences, {funnel_10k['layer2_hmmer']['cpu_hours']} CPU-hrs
  Layer 3 ESM-2:     {funnel_10k['layer3_esm2']['volume']} sequences, {funnel_10k['layer3_esm2']['cpu_hours']} CPU-hrs

-> {report_path}
-> {csv_path}
""")


if __name__ == "__main__":
    main()
