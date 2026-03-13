import argparse, json, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022

FILE_MANIFEST = [
    ("variants_output/ricin_A_chain/stealth/seqs/ricin_A_chain.fa", 1, "stealth", "ricin"),
    ("variants_output/botulinum_ntx_A/stealth/seqs/botulinum_ntx_A.fa", 1, "stealth", "botulinum"),
    ("variants_output/staph_enterotoxin_B/stealth/seqs/staph_enterotoxin_B.fa", 1, "stealth", "staph_eb"),
    ("variants_dud/seqs/ricin_wt_predicted.fa", 1, "dud", "ricin"),
    ("negatives/negatives.fasta", 0, "benign", "none"),
]

def parse_fasta(path):
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None: records.append((header, "".join(parts)))
                header, parts = line[1:], []
            else: parts.append(line)
    if header is not None: records.append((header, "".join(parts)))
    return records

def collect_sequences():
    all_records, seen = [], set()
    for path, label, category, toxin_class in FILE_MANIFEST:
        if not os.path.exists(path):
            print(f"  WARNING: not found — {path}"); continue
        added = 0
        for header, seq in parse_fasta(path):
            key = (header, seq[:30])
            if key in seen: continue
            seen.add(key)
            all_records.append({"header": header, "sequence": seq[:MAX_LEN],
                                 "label": label, "category": category,
                                 "toxin_class": toxin_class, "source_file": path})
            added += 1
        print(f"  {added:3d} seqs  <- {path}")
    return all_records

def embed(sequences, device, batch_size):
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LEN+2)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mean_e = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(mean_e.cpu().numpy())
            print(f"  Embedded {min(i+batch_size, len(sequences))}/{len(sequences)}", end="\r")
    print()
    return np.vstack(all_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Collecting sequences:")
    records = collect_sequences()
    if not records:
        print("ERROR: No sequences loaded."); return
    print(f"\nTotal: {len(records)} | Labels: {dict(Counter(r['label'] for r in records))} | Toxin: {dict(Counter(r['toxin_class'] for r in records))}")
    embeddings = embed([r["sequence"] for r in records], args.device, args.batch_size)
    labels_arr = np.array([r["label"] for r in records])
    np.save(os.path.join(args.outdir, "embeddings_all.npy"), embeddings)
    np.save(os.path.join(args.outdir, "labels_all.npy"), labels_arr)
    meta_out = [{k: v for k, v in r.items() if k != "sequence"} for r in records]
    with open(os.path.join(args.outdir, "metadata_all.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nembeddings_all.npy {embeddings.shape} | labels_all.npy {labels_arr.shape} | metadata_all.json {len(records)} entries")
    print("Next: python3 embedding_geometry.py")

if __name__ == "__main__":
    main()
