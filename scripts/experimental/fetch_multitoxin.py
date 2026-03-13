"""
fetch_multitoxin.py
-------------------
Extends fetch_uniprot.py to pull sequences for THREE proxy toxins:
  1. Ricin A-chain         (UniProt P02879)  — already in the repo
  2. Botulinum neurotoxin  (UniProt P10844)  — light chain, type A
  3. Staphylococcal enterotoxin B (UniProt P01552)

All sequences are publicly available in UniProt Swiss-Prot and are used here
purely as computational proxies for biosecurity screening benchmarking.
No wet-lab handling is involved.

Output
------
  toxin_seeds/
      ricin_A_chain.fasta
      botulinum_ntx_A.fasta
      staph_enterotoxin_B.fasta
      all_toxins.fasta          ← combined file for downstream steps
      metadata.json             ← accession, length, organism, function per entry

Usage
-----
  python fetch_multitoxin.py
  python fetch_multitoxin.py --outdir custom_dir/
"""

import argparse
import json
import os
import time
import requests

# ── Target sequences ─────────────────────────────────────────────────────────
TOXIN_TARGETS = [
    {
        "id": "ricin_A_chain",
        "accession": "P02879",
        "name": "Ricin A-chain",
        "organism": "Ricinus communis",
        "category": "ribosome-inactivating protein",
        "notes": "N-glycosidase; depurinates 28S rRNA. Reference proxy for BLAST evasion benchmark.",
    },
    {
        "id": "botulinum_ntx_A",
        "accession": "P10844",
        "name": "Botulinum neurotoxin type A light chain",
        "organism": "Clostridium botulinum",
        "category": "metalloprotease",
        "notes": "Zn-endopeptidase; cleaves SNAP-25. HHS BSAT Schedule 1.",
    },
    {
        "id": "staph_enterotoxin_B",
        "accession": "P01552",
        "name": "Staphylococcal enterotoxin B",
        "organism": "Staphylococcus aureus",
        "category": "superantigen",
        "notes": "T-cell superantigen; potent emetic. CDC/USDA select agent.",
    },
]

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
UNIPROT_JSON_URL  = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def fetch_fasta(accession: str, retries: int = 3) -> str:
    """Fetch canonical FASTA from UniProt REST API."""
    url = UNIPROT_FASTA_URL.format(accession=accession)
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text.strip()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {accession} after {retries} attempts (HTTP {resp.status_code})")


def fetch_metadata(accession: str) -> dict:
    """Fetch sequence length and annotated function from UniProt JSON."""
    url = UNIPROT_JSON_URL.format(accession=accession)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    seq_len = data.get("sequence", {}).get("length", "unknown")
    # Pull first annotated function comment if available
    functions = []
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for text_block in comment.get("texts", []):
                functions.append(text_block.get("value", ""))
    return {
        "length": seq_len,
        "uniprot_function": functions[0] if functions else "See UniProt entry",
    }


def main(outdir: str = "toxin_seeds"):
    os.makedirs(outdir, exist_ok=True)
    combined_records = []
    all_metadata = []

    for toxin in TOXIN_TARGETS:
        acc = toxin["accession"]
        label = toxin["id"]
        print(f"[{label}] Fetching {acc} from UniProt...")

        fasta = fetch_fasta(acc)
        meta  = fetch_metadata(acc)

        # Write individual FASTA
        out_path = os.path.join(outdir, f"{label}.fasta")
        with open(out_path, "w") as f:
            f.write(fasta + "\n")
        print(f"  → Saved {out_path}  ({meta.get('length', '?')} aa)")

        combined_records.append(fasta)
        all_metadata.append({**toxin, **meta})

        time.sleep(1)   # polite rate-limiting for UniProt

    # Combined FASTA
    combined_path = os.path.join(outdir, "all_toxins.fasta")
    with open(combined_path, "w") as f:
        f.write("\n\n".join(combined_records) + "\n")
    print(f"\n→ Combined FASTA: {combined_path}")

    # Metadata JSON
    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"→ Metadata:       {meta_path}")

    print("\nDone. Run step1_mpnn_variants.py next to generate redesigned variants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch multi-toxin seed sequences from UniProt.")
    parser.add_argument("--outdir", default="toxin_seeds", help="Output directory (default: toxin_seeds/)")
    args = parser.parse_args()
    main(args.outdir)