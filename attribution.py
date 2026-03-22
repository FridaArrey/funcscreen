"""
attribution.py  (v2.0)
----------------------
Explainable biosecurity screening with nearest-toxin attribution
and defense-in-depth integration of Ragha's upstream Evo2 findings.

ARCHITECTURE
------------
Layer 1 — Source Control    : Evo2 activation probing (Rao et al., 2026)
                               get_evo2_activation_score() — stub below
Layer 3 — Motif detection   : HMMER / Commec (Wittmann et al., 2025)
Layer 4 — Evasion detection : ESM-2 embeddings (this script / funcscreen)

defense_in_depth_assessment() integrates all layers into a unified verdict.

EMBEDDING NOTE (v2.0)
---------------------
Switched to explicit [:, 1:-1, :] residue-only mean pooling — skipping
[CLS] and [EOS] special tokens — matching Lin et al. (2023).
"""

import argparse
import json
import os
import pickle
import numpy as np

KNOWN_TOXINS = {
    "P02879": {
        "name":      "Ricin A-chain",
        "organism":  "Ricinus communis",
        "mechanism": "N-glycosidase; depurinates 28S rRNA -> ribosome inactivation",
        "list":      "HHS BSAT Schedule 1",
    },
    "P10844": {
        "name":      "Botulinum neurotoxin type A light chain",
        "organism":  "Clostridium botulinum",
        "mechanism": "Zn-endopeptidase; cleaves SNAP-25 -> inhibits neurotransmitter release",
        "list":      "HHS BSAT Schedule 1",
    },
    "P01552": {
        "name":      "Staphylococcal enterotoxin B",
        "organism":  "Staphylococcus aureus",
        "mechanism": "T-cell superantigen; crosslinks MHC-II and TCR -> cytokine storm",
        "list":      "CDC/USDA Select Agent",
    },
}


# ── Layer 1 stub: Evo2 activation probing (Rao et al., 2026) ─────────────────

def get_evo2_activation_score(seq_id: str, sequence: str = None) -> float | None:
    """
    STUB — Layer 1 (Source Control): Evo2 internal activation probe.

    Returns pathogenicity score from MLP probes on blocks 8 and 14 of Evo2,
    as described in Rao et al. (2026):
      github.com/marapowney/Varsity26BioGaurdrails
      Dashboard: ragharao314159.github.io/evo2_probing_dashboard/

    INTEGRATION INSTRUCTIONS:
      Option A (local): from bioguardrails import Evo2Prober
                        prober = Evo2Prober.load("path/to/weights/")
                        return prober.score(sequence)

      Option B (API):   import requests
                        resp = requests.post("https://[ragha-api]/probe",
                                             json={"sequence": sequence})
                        return resp.json()["block14_probe_score"]

    Returns None when probe unavailable (stub mode).
    Note: Original probe operates on DNA sequences. Protein sequences
    require back-translation or a protein-adapted probe variant.
    """
    return None   # stub


# ── Defense-in-depth integration ─────────────────────────────────────────────

def defense_in_depth_assessment(
    seq_id: str,
    sequence: str,
    esm2_score: float,
    hmmer_hit: bool,
    esm2_threshold: float = 0.5,
    evo2_threshold:  float = 0.5,
) -> dict:
    """
    Integrates Layer 1 (Evo2), Layer 3 (HMMER), and Layer 4 (ESM-2)
    into a unified verdict with case attribution.

    Key evasion cases surfaced:

      EVASION_ESM2_ONLY:
        HMMER MISS + ESM-2 FLAG — confirms the structural evasion gap
        identified by funcscreen (Arrey, 2026).

      EVASION_CONFIRMED_DUAL:
        HMMER MISS + ESM-2 FLAG + Evo2 FLAG — dual-layer independent
        confirmation of both the synthesis-gate evasion (funcscreen)
        and the generation-time awareness finding (Rao et al., 2026).

      ALL_CLEAR:
        No flags from any layer. Honest pass — not a guaranteed safe,
        but outside the detection envelope of all current tools.
    """
    esm2_flagged   = esm2_score >= esm2_threshold
    evo2_score     = get_evo2_activation_score(seq_id, sequence)
    evo2_available = evo2_score is not None
    evo2_flagged   = evo2_available and (evo2_score >= evo2_threshold)

    if hmmer_hit and esm2_flagged:
        case    = "CONFIRMED_THREAT"
        verdict = "BLOCK"
        note    = "Detected by HMMER (sequence-level) and ESM-2 (embedding-level). High confidence."

    elif not hmmer_hit and esm2_flagged and evo2_flagged:
        case    = "EVASION_CONFIRMED_DUAL"
        verdict = "BLOCK"
        note    = (
            f"ALARM: Sequence {seq_id} bypassed HMMER but was flagged by "
            f"ESM-2 (score={esm2_score:.3f}) AND Evo2 activation probes "
            f"(block 14 score={evo2_score:.3f}). "
            "This confirms the evasion gap (Arrey, 2026) AND the "
            "upstream generation-time awareness finding (Rao et al., 2026)."
        )

    elif not hmmer_hit and esm2_flagged and not evo2_available:
        case    = "EVASION_ESM2_ONLY"
        verdict = "BLOCK"
        note    = (
            f"ALARM: Sequence {seq_id} bypassed HMMER but was flagged by "
            f"ESM-2 embedding classifier (score={esm2_score:.3f}). "
            "This confirms the structural evasion gap identified by funcscreen. "
            "Evo2 probe unavailable for upstream confirmation."
        )

    elif not hmmer_hit and esm2_flagged and evo2_available and not evo2_flagged:
        case    = "EVASION_ESM2_ONLY_EVO2_CLEAR"
        verdict = "REVIEW"
        note    = (
            f"Sequence {seq_id} bypassed HMMER and Evo2 probes but was "
            f"flagged by ESM-2 (score={esm2_score:.3f}). "
            "Layers disagree — possible false positive or sequence outside "
            "Evo2 probe training distribution. Human review recommended."
        )

    elif hmmer_hit and not esm2_flagged:
        case    = "HMMER_ONLY"
        verdict = "BLOCK"
        note    = "Detected by HMMER (known homolog). ESM-2 did not flag."

    else:
        case    = "ALL_CLEAR"
        verdict = "PASS"
        note    = "No flags from HMMER, ESM-2, or Evo2 probes."

    return {
        "seq_id":  seq_id,
        "verdict": verdict,
        "case":    case,
        "note":    note,
        "layer_results": {
            "evo2_probe": {
                "available": evo2_available,
                "score":     evo2_score,
                "flagged":   evo2_flagged,
                "source":    "Rao et al. (2026) — BioGuardrails",
            },
            "hmmer": {
                "hit":    hmmer_hit,
                "source": "Wittmann et al. (2025) / IBBIS Commec",
            },
            "esm2": {
                "score":   round(esm2_score, 4),
                "flagged": esm2_flagged,
                "source":  "Arrey (2026) — funcscreen",
            },
        },
    }


# ── ESM-2 embedding (v2.0: residue-only pooling, skip [CLS] and [EOS]) ───────

def embed_sequences_batch(sequences: list[str], device: str = "cpu",
                           batch_size: int = 8) -> np.ndarray:
    """
    Extract ESM-2 mean-pooled embeddings.

    v2.0: Uses residue-only pooling — explicitly zeroes [CLS] (position 0)
    and [EOS] (last non-padding position) before mean-pooling, matching
    Lin et al. (2023). Optimal batch_size=8 for GPU throughput.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model     = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]

            tokens = tokenizer(
                batch_seqs, padding=True, truncation=True,
                max_length=1022, return_tensors="pt"
            )
            tokens  = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)

            # Residue-only mask: zero [CLS] and [EOS], keep middle tokens
            residue_mask = tokens["attention_mask"].clone().float()
            residue_mask[:, 0] = 0   # [CLS]
            for j in range(len(batch_seqs)):
                seq_len = int(tokens["attention_mask"][j].sum().item())
                if seq_len > 1:
                    residue_mask[j, seq_len - 1] = 0   # [EOS]

            mask_expanded = residue_mask.unsqueeze(-1)
            embeddings = (
                (outputs.last_hidden_state * mask_expanded).sum(dim=1)
                / mask_expanded.sum(dim=1).clamp(min=1e-9)
            )
            all_embeddings.append(embeddings.cpu().numpy())
            print(f"  Embedded {min(i + batch_size, len(sequences))}/{len(sequences)}", end="\r")

    print()
    return np.vstack(all_embeddings)


def embed_sequence(seq: str, device: str = "cpu") -> np.ndarray:
    return embed_sequences_batch([seq], device=device, batch_size=1).flatten()


# ── Nearest-toxin attribution ─────────────────────────────────────────────────

def find_nearest_toxins(query_emb: np.ndarray, db: dict, top_k: int = 3) -> list[dict]:
    q_norm  = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    sims    = db["embeddings_norm"] @ q_norm
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        meta       = db["metadata"][idx]
        toxin_id   = meta.get("toxin_class", "unknown")
        toxin_info = next(
            (info for acc, info in KNOWN_TOXINS.items()
             if toxin_id in info["name"].lower() or toxin_id in acc.lower()),
            {}
        )
        results.append({
            "rank":              rank + 1,
            "cosine_similarity": round(float(sims[idx]), 4),
            "toxin_class":       toxin_id,
            "toxin_name":        toxin_info.get("name", toxin_id),
            "mechanism":         toxin_info.get("mechanism", "unknown"),
            "regulatory_list":   toxin_info.get("list", "unknown"),
        })
    return results


# ── Reference database build/load ────────────────────────────────────────────

def build_reference_db(emb_path, lbl_path, meta_path, outpath):
    X    = np.load(emb_path)
    y    = np.load(lbl_path)
    with open(meta_path) as f:
        meta = json.load(f)
    pos_idx  = np.where(y == 1)[0]
    pos_embs = X[pos_idx]
    norms    = np.linalg.norm(pos_embs, axis=1, keepdims=True)
    db = {
        "embeddings_norm": (pos_embs / (norms + 1e-9)).tolist(),
        "metadata":        [meta[i] for i in pos_idx],
        "known_toxins":    KNOWN_TOXINS,
        "n_references":    len(pos_idx),
        "embedding_note":  "v2.0: residue-only pooling (skip [CLS] and [EOS])",
    }
    with open(outpath, "w") as f:
        json.dump(db, f, indent=2)
    print(f"Reference DB: {len(pos_idx)} positives -> {outpath}")


def load_reference_db(path):
    with open(path) as f:
        db = json.load(f)
    db["embeddings_norm"] = np.array(db["embeddings_norm"])
    return db


# ── Full screening pipeline ───────────────────────────────────────────────────

def screen_with_attribution(seq, seq_id, model_bundle, db,
                             hmmer_results=None, device="cpu", threshold=0.5):
    emb    = embed_sequence(seq, device)
    scaler = model_bundle["scaler"]
    clf    = model_bundle["classifier"]
    prob   = float(clf.predict_proba(scaler.transform(emb.reshape(1, -1)))[0, 1])
    hmmer_hit = (hmmer_results or {}).get(seq_id, {}).get("detected", False)
    nearest   = find_nearest_toxins(emb, db, top_k=3)
    verdict   = defense_in_depth_assessment(seq_id, seq, prob, hmmer_hit,
                                             esm2_threshold=threshold)
    return {**verdict, "esm2_confidence": round(prob, 4),
            "nearest_known_toxins": nearest}


def format_report(result):
    icon  = {"BLOCK": "BLOCK", "REVIEW": "REVIEW", "PASS": "PASS"}.get(result["verdict"], "?")
    lr    = result["layer_results"]
    evo2  = lr["evo2_probe"]
    evo2_str = (f"score={evo2['score']:.3f}, flagged={evo2['flagged']}"
                if evo2["available"] else "UNAVAILABLE (stub)")
    lines = [
        "=" * 60,
        f"Sequence: {result['seq_id']}",
        f"Verdict:  [{icon}]  Case: {result['case']}",
        f"Note: {result['note']}",
        "",
        "Layer results:",
        f"  Evo2 probe  (Layer 1): {evo2_str}",
        f"  HMMER       (Layer 3): hit={lr['hmmer']['hit']}",
        f"  ESM-2       (Layer 4): score={lr['esm2']['score']:.4f}, flagged={lr['esm2']['flagged']}",
        "",
        "Nearest known threats (ESM-2 embedding space):",
    ]
    for hit in result.get("nearest_known_toxins", []):
        lines.append(f"  #{hit['rank']} {hit['toxin_name']} — cosine={hit['cosine_similarity']:.4f}")
        lines.append(f"      Mechanism: {hit['mechanism']}")
    lines.append("=" * 60)
    return "\n".join(lines)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="command")

    b = sub.add_parser("build_db")
    b.add_argument("--embeddings", default="results/embeddings_all.npy")
    b.add_argument("--labels",     default="results/labels_all.npy")
    b.add_argument("--metadata",   default="results/metadata_all.json")
    b.add_argument("--outpath",    default="results/reference_db.json")

    s = sub.add_parser("screen")
    s.add_argument("--fasta",     default=None)
    s.add_argument("--sequence",  default=None)
    s.add_argument("--model",     default="results/final_model.pkl")
    s.add_argument("--db",        default="results/reference_db.json")
    s.add_argument("--hmmer",     default=None)
    s.add_argument("--device",    default="cpu")
    s.add_argument("--outdir",    default="attribution_results")
    s.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "build_db":
        build_reference_db(args.embeddings, args.labels,
                           args.metadata, args.outpath)

    elif args.command == "screen":
        os.makedirs(args.outdir, exist_ok=True)
        with open(args.model, "rb") as f:
            model_bundle = pickle.load(f)
        db = load_reference_db(args.db)
        hmmer_results = None
        if args.hmmer and os.path.exists(args.hmmer):
            with open(args.hmmer) as f:
                hmmer_results = json.load(f)
        sequences = load_fasta(args.fasta) if args.fasta else [("query", args.sequence)]
        all_results, report_lines = {}, []
        for seq_id, seq in sequences:
            result = screen_with_attribution(seq, seq_id, model_bundle, db,
                                             hmmer_results, args.device, args.threshold)
            all_results[seq_id] = result
            report_lines.append(format_report(result))
        with open(os.path.join(args.outdir, "attribution_report.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(args.outdir, "attribution_report.txt"), "w") as f:
            f.write("\n\n".join(report_lines) + "\n")
        n_block = sum(1 for r in all_results.values() if r["verdict"] == "BLOCK")
        print(f"\nDone: {len(all_results)} screened | {n_block} blocked")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
