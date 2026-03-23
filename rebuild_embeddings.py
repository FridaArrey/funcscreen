"""
rebuild_embeddings.py
---------------------
Re-embeds ALL sequences from the funcscreen repo, handling the mixed
directory structures that exist:

  variants_stealth/seqs/*.fa           ← flat structure (ricin only)
  variants_output/seqs/*.fa            ← flat structure
  variants_dud/seqs/*.fa               ← flat structure
  variants_output/<toxin>/stealth/seqs/*.fa   ← nested structure (multi-toxin)

Also loads negatives from negatives/negatives.fasta (build_negative_class.py)
or falls back to scanning for any other benign FASTA provided.

Outputs (overwrite-safe):
  results/embeddings_all.npy    shape (N, 1280)
  results/labels_all.npy        shape (N,)  — 1=toxin/variant, 0=benign
  results/metadata_all.json     per-sequence: path, label, toxin_class, category

Run from the repo root:
  python3 rebuild_embeddings.py
  python3 rebuild_embeddings.py --device mps   # Apple Silicon
  python3 rebuild_embeddings.py --device cuda  # NVIDIA
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LEN    = 1022  # ESM-2 context limit minus special tokens

# ── Directory layout ──────────────────────────────────────────────────────────
# Each entry: (glob_pattern, label, category_tag, toxin_class_tag)
# label 1 = positive (toxin/variant), 0 = benign negative

POSITIVE_GLOBS = [
    # Nested multi-toxin structures (primary — 51 seqs each)
    ("variants_output/ricin_A_chain/stealth/seqs/*.fa",       1, "stealth", "ricin"),
    ("variants_output/ricin_A_chain/stealth/seqs/*.fasta",    1, "stealth", "ricin"),
    ("variants_output/botulinum_ntx_A/stealth/seqs/*.fa",     1, "stealth", "botulinum"),
    ("variants_output/botulinum_ntx_A/stealth/seqs/*.fasta",  1, "stealth", "botulinum"),
    ("variants_output/staph_enterotoxin_B/stealth/seqs/*.fa", 1, "stealth", "staph_eb"),
    ("variants_output/staph_enterotoxin_B/stealth/seqs/*.fasta", 1, "stealth", "staph_eb"),
    # Flat stealth (ricin extras in variants_stealth/seqs/ — deduplicated automatically)
    ("variants_stealth/seqs/*.fa",    1, "stealth", "ricin"),
    ("variants_stealth/seqs/*.fasta", 1, "stealth", "ricin"),
]

DUD_GLOBS = [
    ("variants_dud/seqs/*.fa",    1, "dud", "ricin"),
    ("variants_dud/seqs/*.fasta", 1, "dud", "ricin"),
]

NEGATIVE_PATHS = [
    "negatives/negatives.fasta",
    "negatives/negatives.fa",
]

# Directories to explicitly exclude (ProteinMPNN test outputs, toxin seeds, blast merged)
EXCLUDE_DIRS = ["ProteinMPNN/", "toxin_seeds/", "blast_results/", "tmp_mpnn/"]


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Return list of (header, sequence)."""
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def infer_toxin_class(filepath: str) -> str:
    """Infer toxin class from file path."""
    fp = filepath.lower()
    if "ricin" in fp:      return "ricin"
    if "botulinum" in fp:  return "botulinum"
    if "staph" in fp:      return "staph_eb"
    return "unknown"


def collect_all_sequences(repo_root: str = ".") -> list[dict]:
    """
    Walk all known directory patterns and collect sequences with metadata.
    Deduplicates by (header, sequence) to avoid double-counting.
    """
    root    = Path(repo_root)
    seen    = set()   # (header, seq) deduplication
    records = []

    def add_from_glob(pattern, label, category, toxin_override):
        for fpath in sorted(root.glob(pattern)):
            # Skip excluded directories
            if any(excl in str(fpath) for excl in EXCLUDE_DIRS):
                continue
            for header, seq in parse_fasta(str(fpath)):
                key = (header, seq[:50])  # deduplicate on header+seq prefix
                if key in seen:
                    continue
                seen.add(key)
                tclass = toxin_override if toxin_override != "auto" else infer_toxin_class(str(fpath))
                seq_trunc = seq[:MAX_LEN]
                records.append({
                    "header":      header,
                    "sequence":    seq_trunc,
                    "label":       label,
                    "category":    category,
                    "toxin_class": tclass,
                    "source_file": str(fpath.relative_to(root)),
                })

    # Positives
    for pattern, label, cat, tclass in POSITIVE_GLOBS:
        add_from_glob(pattern, label, cat, tclass)

    # Duds (positive class — same toxin, but low TM-score; keep labelled separately)
    for pattern, label, cat, tclass in DUD_GLOBS:
        add_from_glob(pattern, label, cat, tclass)

    # Negatives
    neg_loaded = False
    for neg_path in NEGATIVE_PATHS:
        full = root / neg_path
        if full.exists():
            for header, seq in parse_fasta(str(full)):
                key = (header, seq[:50])
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "header":      header,
                    "sequence":    seq[:MAX_LEN],
                    "label":       0,
                    "category":    "benign",
                    "toxin_class": "none",
                    "source_file": neg_path,
                })
            neg_loaded = True
            break

    if not neg_loaded:
        print("WARNING: No negatives/negatives.fasta found. Run build_negative_class.py first.")

    return records


# ── ESM-2 embedding ───────────────────────────────────────────────────────────

def embed(sequences: list[str], device: str, batch_size: int) -> np.ndarray:
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_emb  = []
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch   = sequences[i : i + batch_size]
            inputs  = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LEN + 2)
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            out     = model(**inputs)
            mask    = inputs["attention_mask"].unsqueeze(-1).float()
            mean_e  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(mean_e.cpu().numpy())
            print(f"  Embedded {min(i+batch_size, len(sequences))}/{len(sequences)} sequences", end="\r")

    print()
    return np.vstack(all_emb)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root",  default=".")
    parser.add_argument("--outdir",     default="results")
    parser.add_argument("--device",     default="cpu",
                        help="cpu | cuda | mps  (mps = Apple Silicon)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Collect
    print("Scanning for sequences...")
    records = collect_all_sequences(args.repo_root)

    if not records:
        print("ERROR: No sequences found. Check your directory structure.")
        return

    # Summary
    from collections import Counter
    cats   = Counter(r["category"]    for r in records)
    toxins = Counter(r["toxin_class"] for r in records)
    labels = Counter(r["label"]       for r in records)
    print(f"\nFound {len(records)} sequences total:")
    print(f"  Labels:      {dict(labels)}  (1=toxin/variant, 0=benign)")
    print(f"  Categories:  {dict(cats)}")
    print(f"  Toxin class: {dict(toxins)}")

    if len(records) < 10:
        print("\nWARNING: Very few sequences found. Check that .fa files are in the expected paths.")

    seqs = [r["sequence"] for r in records]

    # 2. Embed
    embeddings = embed(seqs, args.device, args.batch_size)
    labels_arr = np.array([r["label"] for r in records])

    # 3. Save
    emb_path  = os.path.join(args.outdir, "embeddings_all.npy")
    lbl_path  = os.path.join(args.outdir, "labels_all.npy")
    meta_path = os.path.join(args.outdir, "metadata_all.json")

    np.save(emb_path, embeddings)
    np.save(lbl_path, labels_arr)
    with open(meta_path, "w") as f:
        # Don't write full sequences to JSON — just metadata
        json.dump([{k: v for k, v in r.items() if k != "sequence"} for r in records], f, indent=2)

    print(f"\n✓ embeddings_all.npy  shape: {embeddings.shape}")
    print(f"✓ labels_all.npy      shape: {labels_arr.shape}")
    print(f"✓ metadata_all.json   ({len(records)} entries)")
    print(f"\nNext: python3 final_master_plot.py")


if __name__ == "__main__":
    main()