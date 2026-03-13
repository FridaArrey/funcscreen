"""
train_detector_cv.py
--------------------
Replaces / extends train_detector.py with rigorous 5-fold cross-validation.

Addresses two blocking gaps identified in the assessment:
  1. Single train/test split produces untrustworthy metrics (especially
     Precision = Recall = 1.00 on a small dataset).
  2. Negative class was undocumented — this script requires negatives/
     produced by build_negative_class.py.

Pipeline
--------
  1. Load ESM-2 (650M) from HuggingFace and extract mean-pooled embeddings.
  2. Load positive sequences (toxin variants — all three categories).
  3. Load negative sequences from negatives/negatives.fasta.
  4. Run stratified 5-fold CV with LogisticRegression.
  5. Report: per-fold + mean ± SD for precision, recall, F1, AUROC.
  6. Plot: PR curve (averaged across folds) + confusion matrix (summed).
  7. Save: final model trained on full data, embeddings, and metrics JSON.

Outputs
-------
  results/
      cv_metrics.json           ← per-fold and mean±SD metrics
      confusion_matrix.png
      pr_curve_cv.png
      embeddings_all.npy        ← shape (N, 1280) for downstream analysis
      labels_all.npy
      final_model.pkl           ← LogisticRegression trained on full dataset

Usage
-----
  # CPU-only (slow but functional):
  python train_detector_cv.py --positives_dir variants_stealth/seqs \\
                               --negatives_fasta negatives/negatives.fasta

  # With GPU:
  python train_detector_cv.py --device cuda ...
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler

import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 1022   # ESM-2 context limit (minus special tokens)


# ── Sequence loading ──────────────────────────────────────────────────────────

def load_fasta_sequences(path: str) -> list[tuple[str, str]]:
    """Return list of (header, sequence) from a FASTA file."""
    records = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def collect_sequences(positives_dirs: list[str], negatives_fasta: str):
    """Load all positive and negative sequences; return (seqs, labels, headers)."""
    seqs, labels, headers = [], [], []

    # Positives — walk all provided directories for .fasta / .fa files
    pos_count = 0
    for d in positives_dirs:
        for fpath in Path(d).glob("**/*.fasta"):
            for hdr, seq in load_fasta_sequences(str(fpath)):
                if len(seq) > MAX_SEQ_LEN:
                    seq = seq[:MAX_SEQ_LEN]
                seqs.append(seq)
                labels.append(1)
                headers.append(hdr)
                pos_count += 1
        for fpath in Path(d).glob("**/*.fa"):
            for hdr, seq in load_fasta_sequences(str(fpath)):
                if len(seq) > MAX_SEQ_LEN:
                    seq = seq[:MAX_SEQ_LEN]
                seqs.append(seq)
                labels.append(1)
                headers.append(hdr)
                pos_count += 1

    print(f"Loaded {pos_count} positive sequences from {positives_dirs}")

    # Negatives
    neg_count = 0
    for hdr, seq in load_fasta_sequences(negatives_fasta):
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        seqs.append(seq)
        labels.append(0)
        headers.append(hdr)
        neg_count += 1

    print(f"Loaded {neg_count} negative sequences from {negatives_fasta}")
    print(f"Class balance — positives: {pos_count}, negatives: {neg_count}")

    return seqs, np.array(labels), headers


# ── ESM-2 embedding ───────────────────────────────────────────────────────────

def embed_sequences(seqs: list[str], device: str = "cpu",
                    batch_size: int = 4) -> np.ndarray:
    """Extract mean-pooled ESM-2 embeddings. Returns shape (N, 1280)."""
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_embeddings = []
    n_batches = (len(seqs) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_SEQ_LEN + 2)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Mean-pool over sequence positions (excluding padding)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).float()
            mean_emb = (
                (token_embeddings * input_mask_expanded).sum(dim=1)
                / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            )
            all_embeddings.append(mean_emb.cpu().numpy())

            batch_num = i // batch_size + 1
            print(f"  Embedded batch {batch_num}/{n_batches}", end="\r")

    print()
    return np.vstack(all_embeddings)


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Stratified k-fold CV. Returns per-fold metrics and mean±SD."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_y_true, all_y_prob = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=42
        )
        clf.fit(X_tr_s, y_tr)

        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "fold": fold + 1,
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "auroc":     roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan"),
            "n_test":    len(y_te),
            "n_pos_test": int(y_te.sum()),
        }
        fold_metrics.append(metrics)
        all_y_true.extend(y_te.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(
            f"  Fold {fold+1}: P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
            f"AUROC={metrics['auroc']:.3f}"
        )

    # Summary stats
    for metric in ["precision", "recall", "f1", "auroc"]:
        vals = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
        print(f"  Mean {metric}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return {
        "folds": fold_metrics,
        "mean": {
            m: float(np.mean([f[m] for f in fold_metrics if not np.isnan(f[m])]))
            for m in ["precision", "recall", "f1", "auroc"]
        },
        "std": {
            m: float(np.std([f[m] for f in fold_metrics if not np.isnan(f[m])]))
            for m in ["precision", "recall", "f1", "auroc"]
        },
        "all_y_true": all_y_true,
        "all_y_prob": all_y_prob,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_prob, outpath, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign (pred)", "Threat (pred)"])
    ax.set_yticklabels(["Benign (true)", "Threat (true)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Confusion Matrix (all folds pooled, threshold=0.5)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  → {outpath}")


def plot_pr_curve(y_true, y_prob, outpath):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1a5276", lw=2, label=f"ESM-2 LR (AP={ap:.3f})")
    ax.axhline(y=sum(y_true) / len(y_true), color="gray", linestyle="--", label="Baseline (random)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (5-fold CV pooled)")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  → {outpath}")


# ── Final model ───────────────────────────────────────────────────────────────

def train_final_model(X, y, outdir):
    """Train on full dataset; save scaler + classifier."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=42
    )
    clf.fit(X_s, y)
    bundle = {"scaler": scaler, "classifier": clf,
              "model_name": MODEL_NAME, "embedding_dim": X.shape[1]}
    pkl_path = os.path.join(outdir, "final_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  → Final model saved: {pkl_path}")
    return clf, scaler


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives_dirs", nargs="+",
                        default=["variants_stealth/seqs", "variants_output/seqs"],
                        help="Directories containing positive (toxin variant) FASTA files")
    parser.add_argument("--negatives_fasta", default="negatives/negatives.fasta")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load sequences
    seqs, labels, headers = collect_sequences(args.positives_dirs, args.negatives_fasta)

    # 2. Embed
    embeddings = embed_sequences(seqs, device=args.device, batch_size=args.batch_size)

    # 3. Save embeddings
    np.save(os.path.join(args.outdir, "embeddings_all.npy"), embeddings)
    np.save(os.path.join(args.outdir, "labels_all.npy"), labels)

    # 4. Cross-validation
    print(f"\n── {args.n_folds}-fold stratified CV ──")
    cv_results = run_cv(embeddings, labels, n_splits=args.n_folds)

    # 5. Save metrics
    metrics_path = os.path.join(args.outdir, "cv_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n→ CV metrics saved: {metrics_path}")

    # 6. Plots
    print("\nGenerating plots...")
    plot_confusion_matrix(
        cv_results["all_y_true"], cv_results["all_y_prob"],
        os.path.join(args.outdir, "confusion_matrix.png")
    )
    plot_pr_curve(
        cv_results["all_y_true"], cv_results["all_y_prob"],
        os.path.join(args.outdir, "pr_curve_cv.png")
    )

    # 7. Final model
    print("\nTraining final model on full dataset...")
    train_final_model(embeddings, labels, args.outdir)

    # 8. Summary
    m = cv_results["mean"]
    s = cv_results["std"]
    print(f"""
╔══════════════════════════════════════════════╗
║  5-Fold CV Results (mean ± SD)               ║
╠══════════════════════════════════════════════╣
║  Precision : {m['precision']:.3f} ± {s['precision']:.3f}                   ║
║  Recall    : {m['recall']:.3f} ± {s['recall']:.3f}                   ║
║  F1        : {m['f1']:.3f} ± {s['f1']:.3f}                   ║
║  AUROC     : {m['auroc']:.3f} ± {s['auroc']:.3f}                   ║
╚══════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()