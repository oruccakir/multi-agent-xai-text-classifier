#!/usr/bin/env python3
"""Generate pattern recognition figures: t-SNE, K-Means, and 10-run statistical tests."""

import sys
import os
sys.path.insert(0, '/home/cakir/projects/AI/multi-agent-xai-text-classifier')

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

# Paths
DATA_DIR = '/home/cakir/projects/AI/multi-agent-xai-text-classifier/data'
MODELS_DIR = f'{DATA_DIR}/models/turkish_sentiment_all_models/turkish_sentiment'
PROCESSED_DIR = f'{DATA_DIR}/processed'
OUT_DIR = '/home/cakir/projects/AI/multi-agent-xai-text-classifier/reports/pattern_recognition/turkish_sentiment'

# Load feature extractor
print("Loading feature extractor...")
with open(f'{MODELS_DIR}/feature_extractor.pkl', 'rb') as f:
    feature_extractor = pickle.load(f)

# Load test data
print("Loading test data...")
test_df = pd.read_csv(f'{PROCESSED_DIR}/turkish_sentiment_test.csv')
# Stratified sample of 2000
from sklearn.model_selection import train_test_split
label_map = {'negatif': 0, 'notr': 1, 'pozitif': 2}
label_names = ['negatif', 'notr', 'pozitif']
colors = ['red', 'green', 'blue']

y_test = test_df['label'].values
X_test_texts = test_df['text'].values

# Stratified subsample of 2000
idx_sample = []
for lbl in label_map.keys():
    mask = y_test == lbl
    lbl_idx = np.where(mask)[0]
    n_take = min(int(2000 * mask.mean()), len(lbl_idx))
    chosen = np.random.RandomState(42).choice(lbl_idx, n_take, replace=False)
    idx_sample.extend(chosen)
idx_sample = np.array(idx_sample)

X_sample_texts = X_test_texts[idx_sample]
y_sample = y_test[idx_sample]
y_sample_int = np.array([label_map[l] for l in y_sample])

# Transform to TF-IDF
print("Transforming to TF-IDF features...")
X_tfidf = feature_extractor.transform(X_sample_texts)

# --- 1a. t-SNE Visualization ---
print("Running PCA + t-SNE...")
# PCA to 50 dims first
pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(X_tfidf.toarray())

# t-SNE to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_2d = tsne.fit_transform(X_pca50)

fig, ax = plt.subplots(figsize=(8, 6))
color_map = {'negatif': 'red', 'notr': 'green', 'pozitif': 'blue'}
for lbl, color in color_map.items():
    mask = y_sample == lbl
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl, alpha=0.4, s=10, rasterized=True)
ax.set_title('t-SNE Visualization of Turkish Sentiment Features\n(2,000 stratified test samples)', fontsize=12)
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.legend(title='Class', loc='best', markerscale=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/tsne_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved tsne_2d.png")

# --- 1b. K-Means Clustering ---
print("Running K-Means clustering...")
# Use PCA to 2D for this visualization
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_tfidf.toarray())

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca50)  # cluster on 50D PCA

ari = adjusted_rand_score(y_sample_int, cluster_labels)
print(f"Adjusted Rand Score: {ari:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# True labels
for i, (lbl, color) in enumerate(color_map.items()):
    mask = y_sample == lbl
    axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, label=lbl, alpha=0.4, s=8, rasterized=True)
axes[0].set_title('True Labels', fontsize=12)
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
axes[0].legend(title='Class', loc='best', markerscale=2)
axes[0].grid(True, alpha=0.3)

# K-Means clusters
cluster_colors = ['#e74c3c', '#2ecc71', '#3498db']
for k in range(3):
    mask = cluster_labels == k
    axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=cluster_colors[k], label=f'Cluster {k+1}', alpha=0.4, s=8, rasterized=True)
axes[1].set_title(f'K-Means Clusters (k=3)\nARI = {ari:.3f}', fontsize=12)
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend(title='Cluster', loc='best', markerscale=2)
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA 2D: True Labels vs. K-Means Clustering', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved kmeans_clusters.png (ARI={ari:.4f})")

# --- 1c. Top Discriminative Features ---
print("Loading SVM model for top features...")
with open(f'{MODELS_DIR}/svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Get the underlying LinearSVC from CalibratedClassifierCV
cal = svm_model.model  # CalibratedClassifierCV
base_svc = cal.calibrated_classifiers_[0].estimator  # LinearSVC

vocab = feature_extractor.get_feature_names()
coef = base_svc.coef_  # shape: (n_classes, n_features)

print(f"SVM coef shape: {coef.shape}, vocab size: {len(vocab)}")
print(f"Classes: {base_svc.classes_}")

# Map class order from base_svc.classes_
class_order = list(base_svc.classes_)
print(f"Class order in SVM: {class_order}")

top_n = 10
top_features = {}
for i, cls in enumerate(class_order):
    top_idx = np.argsort(coef[i])[-top_n:][::-1]
    top_features[cls] = [(vocab[j], coef[i][j]) for j in top_idx]

# Print LaTeX table
print("\n% LaTeX table for top features:")
print("\\begin{tabular}{lll}")
print("\\toprule")
print("\\textbf{Pozitif} & \\textbf{Notr} & \\textbf{Negatif} \\\\")
print("\\midrule")
# Get features for each class
pos_feats = top_features.get('pozitif', top_features.get('positive', []))
notr_feats = top_features.get('notr', top_features.get('neutral', []))
neg_feats = top_features.get('negatif', top_features.get('negative', []))

for i in range(top_n):
    p = pos_feats[i][0] if i < len(pos_feats) else ''
    n = notr_feats[i][0] if i < len(notr_feats) else ''
    ng = neg_feats[i][0] if i < len(neg_feats) else ''
    print(f"\\textit{{{p}}} & \\textit{{{n}}} & \\textit{{{ng}}} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# Also save as JSON for report
import json
features_data = {
    'pozitif': [f[0] for f in pos_feats],
    'notr': [f[0] for f in notr_feats],
    'negatif': [f[0] for f in neg_feats],
    'ari': ari
}
with open(f'{OUT_DIR}/pr_extras.json', 'w') as f:
    json.dump(features_data, f, ensure_ascii=False, indent=2)
print(f"\nSaved features data and ARI to pr_extras.json")

# --- 2a. Feature-Feature Correlation Heatmap (Top 15 SVM features) ---
print("\nGenerating feature correlation heatmap...")
import seaborn as sns

# Collect top-5 features per class (15 total unique)
top_feat_names = []
for cls in class_order:
    for feat_name, _ in top_features[cls][:5]:
        if feat_name not in top_feat_names:
            top_feat_names.append(feat_name)

# Get indices of these features in the vocabulary
feat_indices = [vocab.index(f) for f in top_feat_names if f in vocab]
feat_labels = [f for f in top_feat_names if f in vocab]

# Extract sub-matrix from TF-IDF (using 2000-sample data)
X_sub = X_tfidf[:, feat_indices].toarray()

# Compute correlation matrix
corr_matrix = np.corrcoef(X_sub.T)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, xticklabels=feat_labels, yticklabels=feat_labels,
            cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            square=True, ax=ax)
ax.set_title('Feature-Feature Correlation Matrix\n(Top-5 SVM Features per Class)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved feature_correlation.png ({len(feat_labels)} features)")

# --- 2b. Covariance Matrix (Top 15 features) ---
print("Generating covariance matrix...")
cov_matrix = np.cov(X_sub.T)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cov_matrix, xticklabels=feat_labels, yticklabels=feat_labels,
            cmap='YlOrRd', annot=True, fmt='.4f', annot_kws={'size': 7},
            square=True, ax=ax)
ax.set_title('Covariance Matrix\n(Top-5 SVM Features per Class)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/covariance_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved covariance_matrix.png")

# --- 2c. Feature-Target Correlation (Chi-squared) ---
print("Computing feature-target correlations (chi-squared)...")
from sklearn.feature_selection import chi2

# Use full 2000-sample TF-IDF
chi2_scores, chi2_pvals = chi2(X_tfidf, y_sample_int)

# Get top-20 features by chi2
top20_chi2_idx = np.argsort(chi2_scores)[-20:][::-1]
top20_chi2_names = [vocab[i] for i in top20_chi2_idx]
top20_chi2_scores = chi2_scores[top20_chi2_idx]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top20_chi2_names)), top20_chi2_scores[::-1], color='steelblue')
ax.set_yticks(range(len(top20_chi2_names)))
ax.set_yticklabels(top20_chi2_names[::-1], fontsize=9)
ax.set_xlabel('Chi-squared Score', fontsize=11)
ax.set_title('Top-20 Features by Chi-squared (Feature-Target Association)', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/chi2_feature_target.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved chi2_feature_target.png")

# --- 2d. Class Distribution Statistics ---
print("Computing class distribution statistics...")
# Per-class TF-IDF vector magnitude and document length stats
class_stats = {}
for cls_name, cls_int in [('negatif', 0), ('notr', 1), ('pozitif', 2)]:
    mask = y_sample_int == cls_int
    X_cls = X_tfidf[mask]
    # L2 norms (should be ~1.0 due to L2 normalization)
    norms = np.sqrt(np.array(X_cls.power(2).sum(axis=1)).flatten())
    # Non-zero features per document (sparsity)
    nnz_per_doc = np.array(X_cls.getnnz(axis=1)).flatten()
    # Mean TF-IDF value of non-zero entries
    mean_tfidf = np.array(X_cls.sum(axis=1)).flatten() / np.maximum(nnz_per_doc, 1)

    class_stats[cls_name] = {
        'count': int(mask.sum()),
        'mean_nnz_features': float(np.mean(nnz_per_doc)),
        'std_nnz_features': float(np.std(nnz_per_doc)),
        'mean_l2_norm': float(np.mean(norms)),
        'mean_tfidf_value': float(np.mean(mean_tfidf)),
        'std_tfidf_value': float(np.std(mean_tfidf)),
    }
    print(f"  {cls_name}: n={mask.sum()}, mean_nnz={np.mean(nnz_per_doc):.1f}±{np.std(nnz_per_doc):.1f}, "
          f"mean_tfidf={np.mean(mean_tfidf):.4f}±{np.std(mean_tfidf):.4f}")

# Also compute document length stats from raw text
doc_len_stats = {}
for cls_name in ['negatif', 'notr', 'pozitif']:
    mask = y_sample == cls_name
    lengths = np.array([len(t.split()) for t in X_sample_texts[mask]])
    doc_len_stats[cls_name] = {
        'mean_words': float(np.mean(lengths)),
        'std_words': float(np.std(lengths)),
        'median_words': float(np.median(lengths)),
    }
    print(f"  {cls_name} doc length: mean={np.mean(lengths):.1f}±{np.std(lengths):.1f} words")

# Save all extra stats
features_data['class_stats'] = class_stats
features_data['doc_len_stats'] = doc_len_stats
features_data['top20_chi2'] = list(zip(top20_chi2_names, [float(s) for s in top20_chi2_scores]))

with open(f'{OUT_DIR}/pr_extras.json', 'w') as f:
    json.dump(features_data, f, ensure_ascii=False, indent=2)
print(f"\nUpdated pr_extras.json with correlation and distribution data")

print("\nAll figures generated successfully!")
