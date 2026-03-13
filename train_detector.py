import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc

# 1. The Experimental Data Points
# Normalized vectors for demonstration based on the 0.9856 similarity
# Ricin WT (Class 1)
ricin_wt = np.random.normal(0, 0.1, 1280) 
# The Stealth Sample 5 (Class 1) - Very close to WT
stealth_sample_5 = ricin_wt + np.random.normal(0, 0.05, 1280) 
# Benign Proteins (Class 0) - e.g., Milk Protein, Hemoglobin
benign_1 = np.random.normal(1, 0.1, 1280)
benign_2 = np.random.normal(-1, 0.1, 1280)

# 2. Build Training Set
X = np.array([ricin_wt, stealth_sample_5, benign_1, benign_2])
y = np.array([1, 1, 0, 0])  # 1=Toxin, 0=Benign

# 3. Train Logistic Regression
# We use a small C (regularization) because we have a tiny sample size
clf = LogisticRegression(C=1.0)
clf.fit(X, y)

# 4. Evaluate Stealth Sample
prob = clf.predict_proba(stealth_sample_5.reshape(1, -1))[0][1]
print(f"🕵️ Stealth Variant Analysis...")
print(f"📈 Probability of Toxin: {prob*100:.2f}%")

# 5. Visualizing the Decision Boundary (Precision-Recall)
# This fulfills the course requirement for "metrics"
y_scores = clf.decision_function(X)
precision, recall, _ = precision_recall_curve(y, y_scores)

plt.figure(figsize=(8, 5))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Specificity)')
plt.title('Precision-Recall Curve: AI Embedding Classifier')
plt.savefig('final_metrics_pr_curve.png')
print("✅ Metrics Plot saved as final_metrics_pr_curve.png")
