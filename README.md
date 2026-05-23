# Data Science Final Project

**Participants:**
| Name | ID |
|------|-----|
| Noam Kadosh | 207328428 |
| Roy Boker | 208577098 |
| Gal Tayeb | 207338104 |
| Noam Dahan | 318821774 |

Text classification on Twitter data using NLP techniques.

## Project Overview

Binary sentiment analysis on 1.6M tweets from the Sentiment140 dataset. Given the raw text of a tweet, the system predicts whether its sentiment is **Positive** or **Negative** using classical ML models trained on TF-IDF features.

## Notebooks

- **Phase 1 — EDA & Preprocessing** (`notebooks/phase1_sentiment140.ipynb`): EDA and preprocessing of the Sentiment140 dataset (1.6M tweets, binary sentiment).
- **Phase 2 — Models** (one notebook per classifier, each self-contained):
  - `notebooks/phase2_naive_bayes.ipynb` — `MultinomialNB(alpha=1.0)`
  - `notebooks/phase2_logistic_regression.ipynb` — `LogisticRegression(C=1.0)`
  - `notebooks/phase2_linear_svc.ipynb` — `LinearSVC(C=1.0)`
  - `notebooks/phase2_random_forest.ipynb` — `RandomForestClassifier(n_estimators=100)`
- **Phase 3 — Cross-model evaluation** (`notebooks/phase3_evaluation.ipynb`): comparison table sorted by F1, three additional experiments (TF-IDF vocabulary variants, hybrid features, GridSearchCV on SVM `C`), ROC curves, Precision-Recall curves, 5-fold cross-validation, learning curve, and the Sentiment140 benchmark verdict.
- **Phase 4 — Error Analysis** (`notebooks/phase4_error_analysis.ipynb`): deep-dive into model failures — confusion matrix analysis, high-confidence errors, word-level confusion, tweet length vs error rate, linguistic patterns (negation, sarcasm, slang), and model limitations.
- **Phase 5 — Iterative Improvements** (`notebooks/phase5_improvements.ipynb`): step-by-step pipeline improvements — negation handling, larger vocabulary (15k), character n-grams — with F1 comparison chart showing each gain.
- **Phase 6 — BERT (Google Colab + GPU)** (`notebooks/phase6_bert_colab.ipynb`): fine-tuning DistilBERT on 50K tweets to reach ~84% accuracy. Run on Colab with GPU enabled.

## Demo — GUI Application

A desktop GUI application that lets you type any tweet and instantly see predictions from all 4 models side by side.

**Features:**
- **Ensemble Vote** — majority verdict across all 4 models with consensus %
- **Word highlighting** — positive/negative words colored directly in the input
- **Explanation window** — word contributions + step-by-step pipeline per model
- **Analysis history** — last 5 tweets analyzed
- **Example tweets** — preset buttons for Positive, Negative, Negation, Sarcasm cases
- **Ctrl+Enter** — keyboard shortcut to analyze

### Run the demo

**Step 1 — Train and save the models (run once):**
```bash
python save_model.py
```

**Step 2 — Launch the GUI:**
```bash
python gui.py
```

## Datasets

The dataset is too large for GitHub and needs to be downloaded manually.

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1vunlVSQtdbLfMzc2UK5e9GzQ9wsmRSCa?usp=sharing)
2. Place the file inside the `data-sets/` folder:
   - `training.1600000.processed.noemoticon.csv` — Sentiment140 corpus (1.6M tweets, positive/negative).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Results Summary

| Model | Accuracy | F1 | Training data |
|-------|----------|----|---------------|
| Naive Bayes | 0.7569 | 0.7548 | 800K tweets |
| Random Forest | 0.7636 | 0.7636 | 800K tweets |
| Logistic Regression | 0.7748 | 0.7794 | 800K tweets |
| Linear SVC | 0.7744 | 0.7801 | 800K tweets |
| Linear SVC + Improvements | ~0.780 | 0.7808 | 80K tweets |
| **DistilBERT (Phase 6)** | **0.8266** | **0.8244** | **40K tweets** |

Best classical model: **Linear SVC** (F1 = 0.7801)
Best overall: **DistilBERT** (F1 = 0.8244, +4.4% over Linear SVC, trained on 5% of the data)

Published Sentiment140 benchmark: ~82–84% accuracy. Our DistilBERT reaches **82.7%** — within the benchmark range.

## Project Structure

```
├── notebooks/
│   ├── phase1_sentiment140.ipynb
│   ├── phase2_naive_bayes.ipynb
│   ├── phase2_logistic_regression.ipynb
│   ├── phase2_linear_svc.ipynb
│   ├── phase2_random_forest.ipynb
│   ├── phase3_evaluation.ipynb
│   ├── phase4_error_analysis.ipynb
│   ├── phase5_improvements.ipynb
│   └── phase6_bert_colab.ipynb    # run on Google Colab with GPU
├── data-sets/
│   └── training.1600000.processed.noemoticon.csv
├── models/                  # created after running save_model.py
│   ├── tfidf_vectorizer.joblib
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── linear_svc.joblib
│   └── random_forest.joblib
├── save_model.py            # train & save all 4 models
├── gui.py                   # desktop GUI demo
└── requirements.txt
```

## Run Notebooks

```bash
jupyter notebook notebooks/
```