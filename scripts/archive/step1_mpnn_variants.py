import os
import subprocess
import glob
import argparse

# --- CONFIGURATION ---
SEED_DIR = "toxin_seeds"
OUTPUT_BASE = "variants_output"
# !!! DOUBLE CHECK THIS PATH !!!
PROTEIN_MPNN_PATH = "./ProteinMPNN/protein_mpnn_run.py" 

def run_mpnn_on_seeds():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=str, default="0.4")
    parser.add_argument("--num", type=int, default=50)
    args = parser.parse_args()

    # We need the PDBs we downloaded for the redesign
    seeds = glob.glob(os.path.join(SEED_DIR, "*.pdb"))

    for seed_path in seeds:
        toxin_id = os.path.basename(seed_path).replace(".pdb", "")
        out_dir = os.path.join(OUTPUT_BASE, toxin_id, "stealth")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n🚀 Running ProteinMPNN for: {toxin_id}")
        
        cmd = [
            "python", PROTEIN_MPNN_PATH,
            "--pdb_path", seed_path,
            "--out_folder", out_dir,  
            "--num_seq_per_target", str(args.num),
            "--sampling_temp", args.temp,
            "--seed", "42"
        ]
        
        # This will show the exact command in the terminal
        print(f"Executing: {' '.join(cmd)}")
        
      
        subprocess.run(cmd, check=True) 

    print("\n✅ Multi-toxin redesign complete.")

if __name__ == "__main__":
    run_mpnn_on_seeds()