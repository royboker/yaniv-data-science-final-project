# Data Science Final Project

**Participants:**
| Name | ID |
|------|-----|
| Noam Kadosh | 207328428 |
| Roy Boker | 208577098 |
| Gal Tayeb | 207338104 |
| Noam Dahan | 318821774 |

Text classification on Twitter data using NLP techniques.

## Notebooks

- **Phase 1 — EDA & Preprocessing** (`notebooks/phase1_sentiment140.ipynb`): EDA and preprocessing of the Sentiment140 dataset (1.6M tweets, binary sentiment).
- **Phase 2 — Models** (one notebook per classifier, each self-contained):
  - `notebooks/phase2_naive_bayes.ipynb` — `MultinomialNB(alpha=1.0)`
  - `notebooks/phase2_logistic_regression.ipynb` — `LogisticRegression(C=1.0)`
  - `notebooks/phase2_linear_svc.ipynb` — `LinearSVC(C=1.0)`
  - `notebooks/phase2_random_forest.ipynb` — `RandomForestClassifier(n_estimators=100)`
- **Phase 3 — Cross-model evaluation** (`notebooks/phase3_evaluation.ipynb`): comparison table sorted by F1, three additional experiments (TF-IDF vocabulary variants, hybrid features, GridSearchCV on SVM `C`), and the Sentiment140 benchmark verdict.

## Datasets

The dataset is too large for GitHub and needs to be downloaded manually.

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1vunlVSQtdbLfMzc2UK5e9GzQ9wsmRSCa?usp=sharing)
2. Place the file inside the `data-sets/` folder:
   - `training.1600000.processed.noemoticon.csv` — Sentiment140 corpus (1.6M tweets, positive/negative).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook notebooks/
```
