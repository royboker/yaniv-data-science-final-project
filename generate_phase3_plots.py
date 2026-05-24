"""
Generates the 4 missing Phase 3 plots using pre-saved models.
Saves: roc_curves.png, pr_curves.png, cv_results.png, learning_curve.png
Run: python generate_phase3_plots.py
"""
import re, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|n't|dont|cant|wont|isn't|wasn't|didn't|doesn't|"
    r"won't|can't|couldn't|shouldn't|wouldn't|hardly|barely|scarcely)\s+(\w+)",
    re.IGNORECASE
)

def preprocess_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'^rt\s+', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = text.replace('#', '')
    text = NEGATION_PATTERN.sub(lambda m: f"not_{m.group(2)}", text)
    text = re.sub(r'[^a-z_\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if (w.startswith('not_') or w not in STOP_WORDS) and len(w) > 1]
    tokens = [LEMMATIZER.lemmatize(w) if not w.startswith('not_') else w for w in tokens]
    return ' '.join(tokens)

DATASET_PATH = 'data-sets/training.1600000.processed.noemoticon.csv'
MODELS_DIR   = 'models'
OUT_DIR      = 'phase3_plots'
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE = 50_000   # per class — fast but representative

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, encoding='latin-1', header=None,
                 names=['sentiment','id','date','query','user','text'])
df['label'] = df['sentiment'].map({0: 0, 4: 1})
neg = df[df['label']==0].sample(SAMPLE, random_state=42)
pos = df[df['label']==1].sample(SAMPLE, random_state=42)
df  = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Preprocessing...")
df['clean'] = df['text'].apply(preprocess_tweet)

print("Loading vectorizer + models...")
tfidf = joblib.load(f'{MODELS_DIR}/tfidf_vectorizer.joblib')
models = {
    'Naive Bayes':         joblib.load(f'{MODELS_DIR}/naive_bayes.joblib'),
    'Logistic Regression': joblib.load(f'{MODELS_DIR}/logistic_regression.joblib'),
    'Linear SVC':          joblib.load(f'{MODELS_DIR}/linear_svc.joblib'),
    'Random Forest':       joblib.load(f'{MODELS_DIR}/random_forest.joblib'),
}

X = tfidf.transform(df['clean'])
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

COLORS = ['#4a90d9','#4dc97a','#a06adf','#e8973a']

# ── 1. ROC Curves ─────────────────────────────────────────────────────────────
print("Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.plot([0,1],[0,1], color='#555', lw=1, linestyle='--', label='Random (AUC=0.50)')

for (name, mdl), color in zip(models.items(), COLORS):
    if hasattr(mdl, 'predict_proba'):
        scores = mdl.predict_proba(X_test)[:, 1]
    else:
        scores = mdl.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

ax.set_xlabel('False Positive Rate', color='#aab8c2')
ax.set_ylabel('True Positive Rate', color='#aab8c2')
ax.set_title('ROC Curves — All Models', color='#ecf0f1', fontsize=13, fontweight='bold')
ax.tick_params(colors='#aab8c2')
for spine in ax.spines.values():
    spine.set_edgecolor('#4a5568')
ax.legend(facecolor='#1a1a2e', edgecolor='#4a5568', labelcolor='#ecf0f1', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/roc_curves.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"  Saved {OUT_DIR}/roc_curves.png")

# ── 2. Precision-Recall Curves ─────────────────────────────────────────────────
print("Plotting PR curves...")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

for (name, mdl), color in zip(models.items(), COLORS):
    if hasattr(mdl, 'predict_proba'):
        scores = mdl.predict_proba(X_test)[:, 1]
    else:
        scores = mdl.decision_function(X_test)
    prec, rec, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    ax.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')

ax.set_xlabel('Recall', color='#aab8c2')
ax.set_ylabel('Precision', color='#aab8c2')
ax.set_title('Precision-Recall Curves — All Models', color='#ecf0f1', fontsize=13, fontweight='bold')
ax.tick_params(colors='#aab8c2')
for spine in ax.spines.values():
    spine.set_edgecolor('#4a5568')
ax.legend(facecolor='#1a1a2e', edgecolor='#4a5568', labelcolor='#ecf0f1', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/pr_curves.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"  Saved {OUT_DIR}/pr_curves.png")

# ── 3. 5-Fold Cross-Validation ────────────────────────────────────────────────
print("Running 5-Fold CV (3 models, no Random Forest)...")
cv_models = {k: v for k, v in models.items() if k != 'Random Forest'}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, mdl in cv_models.items():
    print(f"  CV: {name}...")
    scores = cross_val_score(mdl, X, y, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"    F1 = {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
names  = list(cv_results.keys())
means  = [cv_results[n].mean() for n in names]
stds   = [cv_results[n].std()  for n in names]
bars   = ax.bar(names, means, color=COLORS[:len(names)], alpha=0.85,
                yerr=stds, capsize=6, error_kw={'color':'#ecf0f1','lw':1.5})
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.004,
            f'{mean:.4f}', ha='center', va='bottom', color='#ecf0f1', fontsize=10, fontweight='bold')
ax.set_ylabel('F1 Score (5-fold CV)', color='#aab8c2')
ax.set_title('5-Fold Cross-Validation — F1 Scores', color='#ecf0f1', fontsize=13, fontweight='bold')
ax.tick_params(colors='#aab8c2')
ax.set_ylim(0.70, 0.82)
for spine in ax.spines.values():
    spine.set_edgecolor('#4a5568')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cv_results.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"  Saved {OUT_DIR}/cv_results.png")

# ── 4. Learning Curve ─────────────────────────────────────────────────────────
print("Plotting learning curve (LinearSVC)...")
from sklearn.svm import LinearSVC
lsvc = models['Linear SVC']
train_sizes = np.linspace(0.1, 1.0, 8)
ts, train_scores, val_scores = learning_curve(
    lsvc, X, y, train_sizes=train_sizes, cv=3, scoring='f1', n_jobs=-1)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.plot(ts, train_scores.mean(axis=1), color='#4dc97a', lw=2, marker='o', label='Training F1')
ax.fill_between(ts,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color='#4dc97a')
ax.plot(ts, val_scores.mean(axis=1), color='#4a90d9', lw=2, marker='o', label='Validation F1')
ax.fill_between(ts,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color='#4a90d9')
ax.set_xlabel('Training set size', color='#aab8c2')
ax.set_ylabel('F1 Score', color='#aab8c2')
ax.set_title('Learning Curve — Linear SVC', color='#ecf0f1', fontsize=13, fontweight='bold')
ax.tick_params(colors='#aab8c2')
for spine in ax.spines.values():
    spine.set_edgecolor('#4a5568')
ax.legend(facecolor='#1a1a2e', edgecolor='#4a5568', labelcolor='#ecf0f1')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/learning_curve.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"  Saved {OUT_DIR}/learning_curve.png")

print(f"\nDone! All 4 plots saved to ./{OUT_DIR}/")