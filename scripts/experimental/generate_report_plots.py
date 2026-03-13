import matplotlib.pyplot as plt
import numpy as np

# Data from your experiments
labels = ['Baseline (WT)', 'Conservative (T=0.1)', 'Stealth (T=0.8)']

# Metrics (normalized to 1.0 for comparison)
# BLAST Identity based on your seq_recovery
seq_identity = [1.0, 0.40, 0.28] 
# Structural similarity from your calculate_tm.py
tm_scores = [1.0, 0.95, 0.58]
# AI Recognition from your check_stealth_similarity.py
embedding_sim = [1.0, 0.99, 0.98]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, seq_identity, width, label='Sequence Identity (BLAST)', color='#e74c3c')
rects2 = ax.bar(x, tm_scores, width, label='Structural Similarity (TM-Score)', color='#3498db')
rects3 = ax.bar(x + width, embedding_sim, width, label='AI Embedding Similarity (ESM-2)', color='#2ecc71')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Similarity Score (0.0 - 1.0)')
ax.set_title('The Biosecurity Gap: Sequence vs. Structure vs. AI Embeddings')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add a horizontal line at 0.5 (Common significance threshold)
ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--')
ax.text(2.1, 0.52, 'Significance Threshold', fontsize=9)

fig.tight_layout()
plt.savefig('biosecurity_gap_analysis.png')
print("✅ Plot saved as biosecurity_gap_analysis.png")
plt.show()