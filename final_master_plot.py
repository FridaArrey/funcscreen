import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Wild Type', 'Stealth (T=0.8)', 'Dud (T=2.0)']
tm_scores = [1.0, 0.5882, 0.2988]
ai_sim = [1.0, 0.9801, 0.9682]
blast_coverage = [1.0, 0.285, 0.102] # Based on seq_recovery results

x = np.arange(len(labels))
width = 0.25

# Colorblind friendly palette (Okabe-Ito colors)
# Vermillion (#D55E00), Blue (#0072B2), Bluish Green (#009E73)
colors = ['#D55E00', '#0072B2', '#009E73']

fig, ax = plt.subplots(figsize=(10, 6))

# Adding hatch patterns for redundant encoding (texture + color)
ax.bar(x - width, blast_coverage, width, label='BLAST Coverage (Sequence)', 
       color=colors[0], hatch='//', edgecolor='white')

ax.bar(x, tm_scores, width, label='TM-Score (Structure)', 
       color=colors[1], hatch='..', edgecolor='white')

ax.bar(x + width, ai_sim, width, label='ESM-2 Similarity (AI)', 
       color=colors[2], hatch='xx', edgecolor='white')

# Labeling and styling
ax.set_ylabel('Score / Percentage', fontweight='bold')
ax.set_title('Biosecurity Guardrail Analysis: Sequence vs Structure vs AI', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower left', frameon=True, shadow=True)

# Add a prominent "Danger Zone" line for TM-Score
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.8)
ax.text(1.5, 0.53, 'Functional Threshold (TM > 0.5)', 
        fontsize=10, color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('master_biosecurity_gap.png')
print("✅ Master visualization saved to master_biosecurity_gap.png")