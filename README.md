# Data Science Final Project

Text classification on Twitter data using NLP techniques.

## Notebooks

- **Phase 1 — Sentiment Analysis** (`notebooks/phase1_sentiment140.ipynb`): EDA and preprocessing of the Sentiment140 dataset (1.6M tweets, binary sentiment).

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
