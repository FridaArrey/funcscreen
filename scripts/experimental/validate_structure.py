import os
import json
import glob
import requests
import subprocess
from pathlib import Path

# --- Configuration ---
VARIANTS_DIR = "variants_output"
SEEDS_DIR = "toxin_seeds"
OUTPUT_PDB_DIR = "variants_predicted_pdbs"
TM_SCORE_FILE = "tm_scores.json"
# Ensure you have TMalign installed (brew install tm-align)
TMALIGN_BIN = "TMalign" 

def get_esmfold_structure(sequence, name):
    """Predicts 3D structure using the ESMFold API."""
    print(f"  → Folding {name}...")
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    try:
        response = requests.post(url, data=sequence, timeout=60)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"    Connection error folding {name}: {e}")
    return None

def run_tm_align(predict_pdb, native_pdb):
    """Runs TMalign and parses the TM-score (normalized by native chain)."""
    try:
        result = subprocess.run(
            [TMALIGN_BIN, predict_pdb, native_pdb],
            capture_output=True, text=True, check=True
        )
        # Parse TM-score normalized by the second chain (the native toxin)
        for line in result.stdout.split('\n'):
            if "TM-score=" in line and "Chain_2" in line:
                return float(line.split()[1])
    except Exception as e:
        print(f"    Alignment error: {e}")
        return 0.0
    return 0.0

def main():
    os.makedirs(OUTPUT_PDB_DIR, exist_ok=True)
    tm_results = {}

    # Find all .fa or .fasta files in the stealth directories
    variant_files = glob.glob(os.path.join(VARIANTS_DIR, "**", "*.fa"), recursive=True)
    
    # Filter to ensure we only get the redesigned sequences, 
    # and ignore any 'wt' (wild-type) files if they exist.
    variant_files = [f for f in variant_files if "stealth" in f and "wt" not in f]

    if not variant_files:
        print(f"❌ No variant files found in {VARIANTS_DIR}.")
        print("Checking folder structure...")
        subprocess.run(["ls", "-R", VARIANTS_DIR]) # This will help us debug if it fails again
        return

    for v_file in variant_files:
        # Extract toxin ID from the folder structure
        toxin_id = Path(v_file).parts[1]
        native_pdb = os.path.join(SEEDS_DIR, f"{toxin_id}.pdb")
        
        if not os.path.exists(native_pdb):
            print(f"⚠️ Native PDB {native_pdb} not found, skipping.")
            continue

        with open(v_file, 'r') as f:
            lines = f.readlines()
            
        current_id = ""
        for line in lines:
            if line.startswith(">"):
                current_id = line.strip().lstrip(">").replace("/", "_")
            else:
                seq = line.strip()
                pdb_path = os.path.join(OUTPUT_PDB_DIR, f"{current_id}.pdb")
                
                # 1. Fold the sequence
                if not os.path.exists(pdb_path):
                    pdb_content = get_esmfold_structure(seq, current_id)
                    if pdb_content:
                        with open(pdb_path, "w") as out_f:
                            out_f.write(pdb_content)
                
                # 2. Align and Score
                if os.path.exists(pdb_path):
                    score = run_tm_align(pdb_path, native_pdb)
                    print(f"    TM-score for {current_id} vs {toxin_id}: {score:.4f}")
                    
                    tm_results[current_id] = {
                        "tm_score": score,
                        "toxin_type": toxin_id
                    }

    # Save to JSON
    with open(TM_SCORE_FILE, "w") as f:
        json.dump(tm_results, f, indent=4)
    print(f"\n✅ Results saved to {TM_SCORE_FILE}")

if __name__ == "__main__":
    main()