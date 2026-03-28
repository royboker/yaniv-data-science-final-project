# Data Science Final Project

Text classification on Twitter data using NLP techniques.

## Notebooks

- **Phase 1 — Hate Speech Detection** (`notebooks/phase1_hate_speech.ipynb`): EDA and preprocessing of the hate speech & offensive language dataset (~25K tweets, 3 classes).
- **Phase 1 — Sentiment Analysis** (`notebooks/phase1_sentiment140.ipynb`): EDA and preprocessing of the Sentiment140 dataset (1.6M tweets, binary sentiment).

## Datasets

Located in `data-sets/`:

- `labeled_data.csv` — Tweets labeled as Hate Speech, Offensive Language, or Neither.
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
