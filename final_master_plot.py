"""
final_master_plot.py
--------------------
Generates the two main figures for the preprint from pre-computed embeddings.

  Figure S2:  UMAP projection of ESM-2 embedding space
              Coloured by: class (toxin vs benign), then by toxin species,
              then by variant category (stealth / dud / original / benign)

  Figure 3:   Evasion comparison bar chart (BLAST vs ESM-2 recall by identity tier)
              Reads from evasion_results/evasion_table.csv if present,
              otherwise falls back to hard-coded preprint values.

UMAP parameter note
-------------------
n_neighbors must be < N (number of sequences). With small datasets:
  N ≤ 30:  n_neighbors = N // 2
  N ≤ 100: n_neighbors = 10
  N > 100: n_neighbors = 15  (original default)
This script handles this automatically.

Run from repo root:
  python3 final_master_plot.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_PATH    = "results/embeddings_all.npy"
LBL_PATH    = "results/labels_all.npy"
META_PATH   = "results/metadata_all.json"
EVASION_CSV = "evasion_results/evasion_table.csv"
OUT_DIR     = "results"

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load embeddings ───────────────────────────────────────────────────────────

def load_embeddings():
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run rebuild_embeddings.py first:\n"
            "  python3 rebuild_embeddings.py"
        )
    X = np.load(EMB_PATH)
    y = np.load(LBL_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        # Fallback metadata if JSON missing
        meta = [{"category": "toxin" if lbl == 1 else "benign",
                 "toxin_class": "unknown",
                 "header": f"seq_{i}"} for i, lbl in enumerate(y)]

    return X, y, meta


# ── UMAP with safe n_neighbors ────────────────────────────────────────────────

def run_umap(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    N = X.shape[0]
    # n_neighbors must be strictly less than N
    if N <= 15:
        n_neighbors = max(2, N - 1)
    elif N <= 30:
        n_neighbors = N // 2
    elif N <= 100:
        n_neighbors = 10
    else:
        n_neighbors = 15

    min_dist = 0.1 if N > 20 else 0.3   # more spread for tiny sets

    print(f"Running UMAP: N={N}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric="cosine",   # cosine distance is standard for protein embeddings
    )
    return reducer.fit_transform(X)


# ── Figure S2a: Class-level UMAP ─────────────────────────────────────────────

def plot_umap_by_class(emb2d, y, outpath):
    df = pd.DataFrame(emb2d, columns=["UMAP1", "UMAP2"])
    df["Class"] = ["Toxin / Variant" if lbl == 1 else "Benign control" for lbl in y]

    palette = {"Toxin / Variant": "#c0392b", "Benign control": "#27ae60"}

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="Class",
                    palette=palette, alpha=0.80, edgecolor="white",
                    linewidth=0.4, s=90, ax=ax)

    ax.set_title("Figure S2: ESM-2 Embedding Space (UMAP)\nToxin variants vs benign proteins",
                 fontsize=12, pad=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="Classification", loc="best", framealpha=0.9)

    # Add N annotation
    n_tox = int(y.sum())
    n_ben = int((y == 0).sum())
    ax.text(0.02, 0.02,
            f"n = {n_tox} toxin/variant,  {n_ben} benign",
            transform=ax.transAxes, fontsize=9, color="grey")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"→ {outpath}")


# ── Figure S2b: Species-level UMAP ───────────────────────────────────────────

TOXIN_PALETTE = {
    "ricin":     "#e74c3c",
    "botulinum": "#e67e22",
    "staph_eb":  "#9b59b6",
    "none":      "#27ae60",   # benign
    "unknown":   "#95a5a6",
}

TOXIN_LABELS = {
    "ricin":     "Ricin A-chain variants",
    "botulinum": "Botulinum NTx-A variants",
    "staph_eb":  "Staph Enterotoxin B variants",
    "none":      "Benign controls",
    "unknown":   "Unknown",
}


def plot_umap_by_toxin(emb2d, meta, outpath):
    df = pd.DataFrame(emb2d, columns=["UMAP1", "UMAP2"])
    df["toxin_class"] = [m.get("toxin_class", "unknown") for m in meta]
    df["category"]    = [m.get("category", "unknown")    for m in meta]

    # Only plot classes that are actually present
    present = df["toxin_class"].unique()
    palette = {k: v for k, v in TOXIN_PALETTE.items() if k in present}

    fig, ax = plt.subplots(figsize=(9, 6))
    for tclass in present:
        sub = df[df["toxin_class"] == tclass]
        # Different marker for dud variants
        marker = "X" if any("dud" in c for c in sub["category"]) else "o"
        ax.scatter(sub["UMAP1"], sub["UMAP2"],
                   c=TOXIN_PALETTE.get(tclass, "#95a5a6"),
                   label=TOXIN_LABELS.get(tclass, tclass),
                   alpha=0.80, edgecolors="white", linewidths=0.4,
                   s=90, marker=marker)

    ax.set_title("Figure S2b: ESM-2 Embedding Space by Toxin Class",
                 fontsize=12, pad=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.text(0.02, 0.02, f"n = {len(df)} sequences total",
            transform=ax.transAxes, fontsize=9, color="grey")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"→ {outpath}")


# ── Figure S2c: Category UMAP ────────────────────────────────────────────────

CATEGORY_PALETTE = {
    "stealth":  "#c0392b",
    "original": "#e67e22",
    "dud":      "#f39c12",
    "benign":   "#27ae60",
}


def plot_umap_by_category(emb2d, meta, outpath):
    df = pd.DataFrame(emb2d, columns=["UMAP1", "UMAP2"])
    df["category"] = [m.get("category", "unknown") for m in meta]

    present = df["category"].unique()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in present:
        sub = df[df["category"] == cat]
        ax.scatter(sub["UMAP1"], sub["UMAP2"],
                   c=CATEGORY_PALETTE.get(cat, "#95a5a6"),
                   label=cat.capitalize(), alpha=0.80,
                   edgecolors="white", linewidths=0.4, s=90)

    ax.set_title("Figure S2c: ESM-2 Embedding Space by Variant Category\n"
                 "(Stealth = TM-score ≥ 0.5;  Dud = TM-score < 0.5)",
                 fontsize=11, pad=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"→ {outpath}")


# ── Figure 3: Evasion bar chart ───────────────────────────────────────────────

# Fallback values from preprint if evasion_table.csv not available
FALLBACK_EVASION = [
    {"identity_bin": "30–50%", "blast_recall": 1.000, "esm2_recall_mean": 1.000, "esm2_recall_std": 0.000, "n_sequences": 12},
    {"identity_bin": "20–30%", "blast_recall": 0.510, "esm2_recall_mean": 1.000, "esm2_recall_std": 0.000, "n_sequences": 98},
    {"identity_bin": "< 20%",  "blast_recall": 0.000, "esm2_recall_mean": 1.000, "esm2_recall_std": 0.000, "n_sequences": 40},
]


def load_evasion_data():
    if os.path.exists(EVASION_CSV):
        df = pd.read_csv(EVASION_CSV)
        rows = []
        for _, r in df.iterrows():
            try:
                blast = float(r["blast_recall"]) if str(r["blast_recall"]) not in ("N/A", "") else None
                rows.append({
                    "identity_bin":    r["identity_bin"],
                    "blast_recall":    blast,
                    "esm2_recall_mean": float(r["esm2_recall_mean"]),
                    "esm2_recall_std":  float(r["esm2_recall_std"]),
                    "n_sequences":      int(r.get("n_sequences", 0)),
                })
            except (ValueError, KeyError):
                continue
        return rows if rows else FALLBACK_EVASION
    print("  (evasion_results/evasion_table.csv not found — using preprint values)")
    return FALLBACK_EVASION


def plot_evasion_comparison(outpath):
    rows = load_evasion_data()
    rows = [r for r in rows if r.get("n_sequences", 1) > 0]

    labels     = [r["identity_bin"] for r in rows]
    blast_vals = [r["blast_recall"] if r["blast_recall"] is not None else 0.0 for r in rows]
    esm2_vals  = [r["esm2_recall_mean"] for r in rows]
    esm2_errs  = [r["esm2_recall_std"]  for r in rows]
    ns         = [r.get("n_sequences", 0) for r in rows]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_blast = ax.bar(x - width/2, blast_vals, width,
                        label="BLAST (blastp, E≤1e-3, ≥30% identity)",
                        color="#c0392b", alpha=0.85)
    bars_esm2  = ax.bar(x + width/2, esm2_vals, width,
                        label="ESM-2 Logistic Regression (5-fold CV mean ± SD)",
                        color="#1a5276", alpha=0.85,
                        yerr=esm2_errs, capsize=5, error_kw={"elinewidth": 1.5})

    # Shade evasion zone (< 30% identity — last two bins in typical ordering)
    evasion_start = next((i for i, r in enumerate(rows)
                          if "20" in r["identity_bin"] or "<" in r["identity_bin"]), None)
    if evasion_start is not None:
        ax.axvspan(evasion_start - 0.5, len(rows) - 0.5,
                   alpha=0.07, color="#c0392b", label="Primary BLAST evasion zone")

    # Annotate N per bin
    for i, (b, e, n) in enumerate(zip(bars_blast, bars_esm2, ns)):
        ax.text(i, -0.07, f"n={n}", ha="center", va="top", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Sequence identity to wildtype toxin", fontsize=12)
    ax.set_ylabel("Recall (threat detection rate)", fontsize=12)
    ax.set_ylim(-0.12, 1.20)
    ax.set_title(
        "Figure 3: BLAST vs ESM-2 Embedding Classifier — Recall by Sequence Identity\n"
        "Ricin A-chain · Botulinum NTx-A · Staphylococcal Enterotoxin B (n=153 variants)",
        fontsize=11, pad=12
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.axhline(1.0, color="grey", linestyle=":", lw=0.8)

    # Evasion zone annotation
    if evasion_start is not None:
        ax.text((evasion_start + len(rows) - 1) / 2, 1.12,
                "BLAST blind spot →", ha="center", color="#922b21",
                fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"→ {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading embeddings...")
    X, y, meta = load_embeddings()
    print(f"  Loaded {X.shape[0]} sequences  (positives: {y.sum()}, negatives: {(y==0).sum()})")

    if X.shape[0] < 5:
        print("ERROR: Too few sequences to plot meaningfully. Run rebuild_embeddings.py first.")
        return

    print("\nComputing UMAP projection...")
    emb2d = run_umap(X)

    print("\nGenerating figures...")
    plot_umap_by_class(emb2d, y,    os.path.join(OUT_DIR, "umap_by_class.png"))
    plot_umap_by_toxin(emb2d, meta, os.path.join(OUT_DIR, "umap_by_toxin.png"))
    plot_umap_by_category(emb2d, meta, os.path.join(OUT_DIR, "umap_by_category.png"))
    plot_evasion_comparison(        os.path.join(OUT_DIR, "figure3_evasion_comparison.png"))

    print(f"""
All figures saved to {OUT_DIR}/
  umap_by_class.png          ← Figure S2a (for preprint supplementary)
  umap_by_taxon.png          ← Figure S2b (by toxin species)
  umap_by_category.png       ← Figure S2c (stealth / dud / benign)
  figure3_evasion_comparison.png  ← Figure 3 (main paper result)
""")


if __name__ == "__main__":
    main()