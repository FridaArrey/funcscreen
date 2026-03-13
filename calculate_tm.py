from tmtools import tm_align
import numpy as np

def parse_pdb(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords)

coords_wt = parse_pdb("ricin_wt_predicted.pdb")
coords_var = parse_pdb("dud_variant_predicted.pdb")

# Create dummy sequences of the correct length for the aligner
seq_wt = "A" * len(coords_wt)
seq_var = "A" * len(coords_var)

res = tm_align(coords_wt, coords_var, seq_wt, seq_var)

# Using the attribute discovered in your 'Available attributes' list
score = res.tm_norm_chain1

print("-" * 30)
print(f"🏆 TM-score: {score:.4f}")
print(f"🧬 RMSD:     {res.rmsd:.4f}")
print("-" * 30)

if score > 0.5:
    print("✅ RESULT: Same fold detected. Functional retention is LIKELY.")
else:
    print("❌ RESULT: Different fold. Functional retention is UNLIKELY.")