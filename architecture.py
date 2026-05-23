"""
Architecture diagram — Tweet Sentiment Analyzer
Run: python architecture.py  →  architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('#16213e')
ax.set_facecolor('#16213e')

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    'data':  ('#1a3461', '#4a90d9'),
    'proc':  ('#0d3d56', '#3aafe0'),
    'model': ('#0d3d1a', '#4dc97a'),
    'bert':  ('#3a1a56', '#a06adf'),
    'app':   ('#5a2d00', '#e8973a'),
    'head':  '#0f1f3d',
    'txt':   '#ecf0f1',
    'sub':   '#aab8c2',
    'line':  '#4a5568',
    'arrow': '#718096',
}

# ── helpers ───────────────────────────────────────────────────────────────────
def block(x, y, w, h, kind, title, lines=None, title_size=10.5):
    bg, border = C[kind]
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.08',
                          facecolor=bg, edgecolor=border,
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    title_y = y + h - 0.38 if lines else y + h / 2
    ax.text(x + w/2, title_y, title,
            ha='center', va='center', color=C['txt'],
            fontsize=title_size, fontweight='bold', zorder=4)
    if lines:
        for i, ln in enumerate(lines):
            ax.text(x + w/2, y + h - 0.72 - i*0.33, ln,
                    ha='center', va='center', color=C['sub'],
                    fontsize=8, zorder=4)

def header(x, y, w, label):
    rect = FancyBboxPatch((x, y), w, 0.42,
                          boxstyle='round,pad=0.05',
                          facecolor=C['head'], edgecolor=C['line'],
                          linewidth=1, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + 0.21, label,
            ha='center', va='center', color='#8899bb',
            fontsize=8.5, fontweight='bold', style='italic', zorder=4)

def arr(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arrow'],
                                lw=1.8, mutation_scale=14,
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)

def divider(x, y, w):
    ax.plot([x + 0.2, x + w - 0.2], [y, y],
            color=C['line'], lw=0.8, zorder=4)

# ── column geometry ───────────────────────────────────────────────────────────
# x-starts and widths
X1, W1 = 0.2,  3.1   # Data
X2, W2 = 3.7,  4.3   # Processing
X3, W3 = 8.4,  4.5   # Models
X4, W4 = 13.3, 6.5   # Application

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(10, 11.55, 'Tweet Sentiment Analyzer — System Architecture',
        ha='center', va='center', color='#e8eaf6',
        fontsize=16, fontweight='bold')
ax.text(10, 11.1, 'Sentiment140  ·  Classical ML + DistilBERT  ·  Desktop GUI',
        ha='center', va='center', color='#6b7a99', fontsize=9.5)

# ── layer headers ─────────────────────────────────────────────────────────────
header(X1, 10.55, W1, '① DATA')
header(X2, 10.55, W2, '② PROCESSING')
header(X3, 10.55, W3, '③ MODELS')
header(X4, 10.55, W4, '④ APPLICATION')

# ── DATA column ───────────────────────────────────────────────────────────────
block(X1, 8.2, W1, 2.1, 'data', 'Sentiment140',
      ['1.6M tweets (CSV · 250 MB)',
       'Auto-labeled via emoji 😊☹️',
       '800K positive · 800K negative'])

block(X1, 5.5, W1, 2.4, 'data', 'Balanced Sampling',
      ['Phase 3: 500K / class',
       'Phase 4 / 5: 100K / class',
       'stratify=y · random_state=42'])

block(X1, 3.1, W1, 2.1, 'data', 'BERT Data',
      ['50K tweets (25K / class)',
       'Light preprocessing only',
       'Uploaded via Google Drive'])

# ── PROCESSING column ─────────────────────────────────────────────────────────
block(X2, 6.9, W2, 3.5, 'proc', 'Preprocessing Pipeline',
      ['lowercase  →  remove URLs & @mentions',
       'Negation:  "not bad" → not_bad',
       'alpha filter  →  NLTK tokenize',
       'stopwords removal  →  lemmatize',
       '(WordNetLemmatizer)'])

block(X2, 4.4, W2, 2.2, 'proc', 'TF-IDF Vectorizer',
      ['max_features = 5,000',
       'ngram_range = (1, 2)',
       'sparsity 99.87%'])

block(X2, 2.4, W2, 1.7, 'proc', 'DistilBERT Tokenizer',
      ['DistilBertTokenizerFast',
       'max_length = 64  ·  padding = True'])

# ── MODEL column ──────────────────────────────────────────────────────────────
my = [9.35, 8.3, 7.25, 6.2]
for y, name, sub in zip(my,
    ['Naive Bayes', 'Logistic Regression', 'Linear SVC', 'Random Forest'],
    ['MultinomialNB(α=1.0)  ·  F1 = 0.757',
     'LogisticRegression(C=1.0)  ·  F1 = 0.778',
     'LinearSVC(C=1.0)  ·  F1 = 0.778',
     'RandomForest(n=100)  ·  F1 = 0.756']):
    block(X3, y, W3, 0.85, 'model', name, [sub], title_size=9.5)

block(X3, 2.4, W3, 3.5, 'bert', 'DistilBERT',
      ['Fine-tuned  ·  3 epochs  ·  AdamW lr=2e-5',
       'Batch = 64  ·  Warmup scheduler',
       '─────────────────────────',
       'Accuracy = 82.7%   F1 = 0.824',
       'Google Colab + Tesla T4 GPU'])

# ── APPLICATION column ────────────────────────────────────────────────────────
block(X4, 6.6, W4, 3.7, 'app', 'Desktop GUI  (gui.py)',
      ['CustomTkinter  ·  dark mode  ·  threading',
       '─────────────────────────────────────────',
       'Input tweet  ·  Example chips  ·  Ctrl+Enter',
       'Ensemble Vote  ·  4 Model cards  ·  Confidence bars',
       'Word highlighting  (green = positive / red = negative)',
       'Explanation window  (Word Analysis + Pipeline tabs)',
       'Analysis History  (last 5 tweets)'])

block(X4, 4.2, W4, 2.1, 'app', 'save_model.py',
      ['Train all 4 models on 200K tweets',
       'Serialize with joblib → models/ folder',
       'Run once before launching the GUI'])

block(X4, 1.8, W4, 2.1, 'app', 'Jupyter Notebooks  (Phase 1 – 6)',
      ['EDA · Preprocessing · 4 model notebooks',
       'Evaluation: ROC · PR curves · 5-fold CV · Learning curve',
       'Error analysis · Iterative improvements · BERT'])

# ── ARROWS ────────────────────────────────────────────────────────────────────
# Data → Processing
arr(X1+W1, 9.25, X2, 8.65)   # Sentiment140 → Preprocessing
arr(X1+W1, 6.7,  X2, 5.5)    # Sampling → TF-IDF
arr(X1+W1, 4.15, X2, 3.25)   # BERT Data → DistilBERT Tokenizer

# Processing → Models
mid_tfidf = 5.5
for y in my:
    arr(X2+W2, mid_tfidf, X3, y + 0.43)
arr(X2+W2, 3.25, X3, 4.15)   # DistilBERT Tokenizer → DistilBERT model

# Models → Application
for y in my:
    arr(X3+W3, y + 0.43, X4, 8.2)
arr(X3+W3, 4.15, X4, 5.25)   # DistilBERT → Notebooks

# ── LEGEND ───────────────────────────────────────────────────────────────────
legend = [('data','Data'), ('proc','Processing'),
          ('model','Classical ML'), ('bert','DistilBERT'), ('app','Application')]
for i, (kind, lbl) in enumerate(legend):
    bg, border = C[kind]
    rx = 0.5 + i * 3.8
    r = FancyBboxPatch((rx, 0.15), 0.32, 0.28,
                       boxstyle='round,pad=0.03',
                       facecolor=bg, edgecolor=border, lw=1.2, zorder=3)
    ax.add_patch(r)
    ax.text(rx + 0.45, 0.29, lbl,
            ha='left', va='center', color='#aab8c2', fontsize=8.5, zorder=4)

plt.tight_layout(pad=0.2)
plt.savefig('architecture.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved: architecture.png')
plt.show()