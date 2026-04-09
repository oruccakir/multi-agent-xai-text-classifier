#!/usr/bin/env python3
"""Run 10-run statistical experiment for Turkish Sentiment dataset."""

import sys
sys.path.insert(0, '/home/cakir/projects/AI/multi-agent-xai-text-classifier')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import json

DATA_DIR = '/home/cakir/projects/AI/multi-agent-xai-text-classifier/data'
PROCESSED_DIR = f'{DATA_DIR}/processed'
OUT_DIR = '/home/cakir/projects/AI/multi-agent-xai-text-classifier/reports/pattern_recognition/turkish_sentiment'

print("Loading training data...")
train_df = pd.read_csv(f'{PROCESSED_DIR}/turkish_sentiment_train.csv')
print(f"Train data shape: {train_df.shape}")

# 10,000 stratified subsample
_, sub_df = train_test_split(
    train_df, test_size=10000/len(train_df),
    stratify=train_df['label'], random_state=42
)
print(f"Subsample shape: {sub_df.shape}")
print(f"Label distribution:\n{sub_df['label'].value_counts()}")

X_all = sub_df['text'].values
y_all_str = sub_df['label'].values
le = LabelEncoder()
le.fit(y_all_str)
y_all = le.transform(y_all_str)

# TF-IDF config matching original
tfidf_params = {
    'max_features': 30000,
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_df': 0.95,
    'sublinear_tf': True,
}

# Model factories
def make_models():
    return {
        'NaiveBayes': ComplementNB(alpha=0.5),
        'SVM': LinearSVC(C=0.5, max_iter=2000, random_state=0),
        'LogisticReg': LogisticRegression(C=0.5, solver='saga', max_iter=1000, random_state=0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(max_depth=20, criterion='gini', random_state=0),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=0, eval_metric='mlogloss',
                                  verbosity=0),
    }

n_runs = 10
results = {name: [] for name in ['NaiveBayes', 'SVM', 'LogisticReg', 'RandomForest', 'KNN', 'DecisionTree', 'XGBoost']}

for seed in range(n_runs):
    print(f"Run {seed+1}/{n_runs}...")
    # Random 80/20 split with this seed
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=seed
    )

    # Fit TF-IDF on training portion
    vec = TfidfVectorizer(**tfidf_params)
    X_tr_vec = vec.fit_transform(X_tr)
    X_val_vec = vec.transform(X_val)

    models = make_models()
    for name, clf in models.items():
        # Update seed-dependent random state
        if hasattr(clf, 'random_state'):
            clf.random_state = seed
        clf.fit(X_tr_vec, y_tr)
        y_pred = clf.predict(X_val_vec)
        f1 = f1_score(y_val, y_pred, average='macro')
        results[name].append(f1)

# Compute statistics
print("\n=== 10-Run Statistical Results ===")
print(f"{'Model':<15} {'Mean F1':>8} {'Std F1':>8} {'p-val vs SVM':>14}")
print("-" * 50)

svm_scores = results['SVM']
stats_output = {}
for name, scores in results.items():
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores, ddof=1)
    if name == 'SVM':
        pval_str = "—"
        pval = None
    else:
        t_stat, pval = stats.ttest_ind(svm_scores, scores)
        pval_str = f"{pval:.4f}"

    stats_output[name] = {'mean': mean_f1, 'std': std_f1, 'pval_vs_svm': pval}
    print(f"{name:<15} {mean_f1:>8.4f} {std_f1:>8.4f} {pval_str:>14}")

# Save results
with open(f'{OUT_DIR}/ten_run_stats.json', 'w') as f:
    json.dump({
        'raw_scores': results,
        'stats': {k: {'mean': v['mean'], 'std': v['std'], 'pval_vs_svm': v['pval_vs_svm']}
                  for k, v in stats_output.items()}
    }, f, indent=2)

print(f"\nSaved 10-run stats to ten_run_stats.json")
