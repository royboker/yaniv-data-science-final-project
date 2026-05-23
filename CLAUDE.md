# CLAUDE.md — Project Context for Claude Code

## What this project is
Binary sentiment analysis on Twitter data (Sentiment140 dataset, 1.6M tweets).
Participants: Noam Kadosh · Roy Boker · Gal Tayeb · Noam Dahan

## Dataset
- File: `data-sets/training.1600000.processed.noemoticon.csv`
- Columns: `sentiment` (0=Negative, 4=Positive), `id`, `date`, `query`, `user`, `text`
- Labels must be remapped: `{0: 0, 4: 1}` before modeling

## Project structure
```
notebooks/
  phase1_sentiment140.ipynb     — EDA & preprocessing (1M sample)
  phase2_naive_bayes.ipynb      — MultinomialNB(alpha=1.0)
  phase2_logistic_regression.ipynb — LogisticRegression(C=1.0)
  phase2_linear_svc.ipynb       — LinearSVC(C=1.0)
  phase2_random_forest.ipynb    — RandomForestClassifier(n_estimators=100)
  phase3_evaluation.ipynb       — comparison table + experiments + ROC + PR + 5-fold CV + learning curve
  phase4_error_analysis.ipynb   — confusion matrix, high-confidence errors, linguistic patterns
  phase5_improvements.ipynb     — negation + vocab 15k + char n-grams, F1 progression chart
  phase6_bert_colab.ipynb       — DistilBERT fine-tuning (run on Google Colab with GPU, real results)
save_model.py                   — trains all 4 models on 200K tweets, saves to models/
gui.py                          — CustomTkinter desktop GUI, all 4 models side by side
requirements.txt
```

## Standard preprocessing pipeline
```python
NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|n't|dont|cant|wont|isn't|wasn't|didn't|doesn't|"
    r"won't|can't|couldn't|shouldn't|wouldn't|hardly|barely|scarcely)\s+(\w+)",
    re.IGNORECASE
)

def preprocess_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)      # remove URLs
    text = re.sub(r'@\w+', '', text)                 # remove @mentions
    text = re.sub(r'^rt\s+', '', text)               # remove RT prefix
    text = re.sub(r'&\w+;', '', text)                # remove HTML entities
    text = text.replace('#', '')                      # keep hashtag word
    text = NEGATION_PATTERN.sub(lambda m: f"not_{m.group(2)}", text)  # negation
    text = re.sub(r'[^a-z_\s]', '', text)            # alpha + underscore only
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if (w.startswith('not_') or w not in STOP_WORDS) and len(w) > 1]
    tokens = [LEMMATIZER.lemmatize(w) if not w.startswith('not_') else w for w in tokens]
    return ' '.join(tokens)
```
**Important:** negation prefix is lowercase `not_` (not `NOT_`) — uppercase would be stripped by the alpha filter.

## Vectorizer
`TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` — used in Phase 1/2/3/4.
Phase 5 best pipeline uses 15k word features + 5k char n-grams (hstacked).

## Baseline results (Phase 3 — 1M sample, 200K test)
| Model | Accuracy | F1 |
|-------|----------|----|
| Linear SVC | 0.7744 | 0.7801 |
| Logistic Regression | 0.7748 | 0.7794 |
| Random Forest | 0.7636 | 0.7636 |
| Naive Bayes | 0.7569 | 0.7548 |

Best: **Linear SVC** (F1 = 0.7801). Sentiment140 benchmark: ~82–84% accuracy.

## Phase 5 results (LinearSVC, 200K sample)
| Experiment | F1 | ΔF1 |
|------------|-----|------|
| Baseline TF-IDF 5k | 0.7728 | — |
| + Negation handling | 0.7780 | +0.0052 |
| + Vocab 15k | 0.7757 | −0.0023 (slight drop) |
| + Char N-grams | 0.7808 | +0.0028 |

## GUI
Run: `python save_model.py` (once) → `python gui.py`
- Models trained on 200K tweets (100K per class) for speed — slightly lower than Phase 3 (1M) but close
- GUI model results (200K, fixed negation): NB=0.7574, LR=0.7777, SVC=0.7781, RF=0.7563
- Threading: analysis runs on daemon thread, results posted back via `self.after(0, ...)`
- EnsembleCard: majority vote + consensus % — shows "SPLIT" in orange when 2-2
- Word highlighting: green/red words in input using Linear SVC contributions (`self.text_input._textbox` tags)
- History panel: last 5 analyses at the bottom
- Example chips: 4 preset tweets (Positive, Negative, Negation, Sarcasm) for demo
- Ctrl+Enter: keyboard shortcut to run analysis
- ExplanationWindow: Word Analysis tab + Pipeline tab

## Phase 6 — BERT (results confirmed)
Run on Google Colab with GPU (Tesla T4). DistilBERT fine-tuned on 50K tweets (25K per class).
**Real results: Accuracy=0.8266, F1=0.8244** — within the 82-84% benchmark range.
Trained on only 40K samples (train split) vs 800K for classical ML. Model saved to Google Drive.

## TF-IDF ceiling
Classical ML on Sentiment140 hits a hard ceiling of ~0.79 F1. The reason: ~20% of the dataset is inherently ambiguous (auto-labeled by emoji, so sarcasm/negation/context is lost). No amount of feature engineering can fix label noise. This is why BERT is needed — it reduces but doesn't eliminate the ambiguity.

## What NOT to change without good reason
- `random_state=42` everywhere — reproducibility
- 80/20 train/test split with `stratify=y`
- The negation prefix `not_` (lowercase) — changing to `NOT_` breaks the alpha filter
- save_model.py sample size (100K per class) — larger = better accuracy but slower GUI startup