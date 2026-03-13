import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score

# 1. THE DATA (Based on your experimental results)
# Categories: [Wild Type, Stealth (Sample 5), Dud (T=2.0), Benign 1, Benign 2]
y_true = np.array([1, 1, 0, 0, 0])  # 1 = Functional Toxin (TM > 0.5), 0 = Not a threat

# 2. BLAST Detection (Categorical: 1 if Identity > 30% and Coverage > 80%, else 0)
# Sample 5 'Evades' BLAST because identity is 28.5% and coverage is partial.
blast_preds = np.array([1, 0, 0, 0, 0]) 

# 3. Embedding Classifier Detection (Probabilities from your earlier run)
# We use your 90.96% (0.9096) for Sample 5
ai_probs = np.array([0.99, 0.9096, 0.12, 0.05, 0.08])
ai_preds = (ai_probs > 0.5).astype(int)

# --- METRIC COMPUTATION ---
def get_metrics(true, pred, name):
    precision = np.sum((pred == 1) & (true == 1)) / np.sum(pred == 1) if np.sum(pred == 1) > 0 else 0
    recall = np.sum((pred == 1) & (true == 1)) / np.sum(true == 1)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

p_blast, r_blast, f1_blast = get_metrics(y_true, blast_preds, "BLAST")
p_ai, r_ai, f1_ai = get_metrics(y_true, ai_preds, "AI")

# --- STEP 4 REPORTING ---
# Xu & Zhang 2010 criteria check
stealth_evaded_blast = (blast_preds[1] == 0) and (y_true[1] == 1)
stealth_caught_by_ai = (ai_preds[1] == 1)

print("📊 STEP 4: EVASION-DETECTION TRADEOFF")
print(f"{'-'*40}")
print(f"BLAST: Precision={p_blast:.2f}, Recall={r_blast:.2f}, F1={f1_blast:.2f}")
print(f"AI:    Precision={p_ai:.2f},    Recall={r_ai:.2f},    F1={f1_ai:.2f}")
print(f"{'-'*40}")
print(f"🚀 BLAST EVASION ANALYSIS (TM > 0.5):")
if stealth_evaded_blast:
    catch_rate = 1.0 if stealth_caught_by_ai else 0.0
    print(f"Fraction of BLAST-evasive variants caught by AI: {catch_rate * 100:.1f}%")

# --- PRECISION-RECALL CURVE ---
precision, recall, _ = precision_recall_curve(y_true, ai_probs)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='#0072B2', lw=3, label=f'AI Classifier (AUC = {auc(recall, precision):.2f})')
plt.axvline(x=r_blast, color='#D55E00', linestyle='--', label='BLAST Recall Baseline')
plt.xlabel('Recall (Sensitivity to Threats)')
plt.ylabel('Precision (Accuracy of Flags)')
plt.title('Precision-Recall Curve: Closing the Biosecurity Gap')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('precision_recall_final.png')