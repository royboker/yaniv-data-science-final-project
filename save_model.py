"""
Train all 4 models on 200K tweets and save them to models/.
Run this once before launching the GUI:
    python save_model.py
"""
import os
import re
import joblib
import pandas as pd
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
DATA_PATH = os.path.join('data-sets', 'training.1600000.processed.noemoticon.csv')
MODELS_DIR = 'models'
SAMPLE_PER_CLASS = 100_000


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
    # Negation handling — join negation word with the following word
    text = NEGATION_PATTERN.sub(lambda m: f"not_{m.group(2)}", text)
    text = re.sub(r'[^a-z_\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if (w.startswith('not_') or w not in STOP_WORDS) and len(w) > 1]
    tokens = [LEMMATIZER.lemmatize(w) if not w.startswith('not_') else w for w in tokens]
    return ' '.join(tokens)


MODELS = [
    ('naive_bayes',          MultinomialNB(alpha=1.0)),
    ('logistic_regression',  LogisticRegression(C=1.0, max_iter=1000)),
    ('linear_svc',           LinearSVC(C=1.0, max_iter=2000)),
    ('random_forest',        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f'Loading dataset ...')
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df_full = pd.read_csv(DATA_PATH, encoding='latin-1', header=None, names=columns)

    df = pd.concat([
        df_full[df_full['sentiment'] == 0].sample(SAMPLE_PER_CLASS, random_state=42),
        df_full[df_full['sentiment'] == 4].sample(SAMPLE_PER_CLASS, random_state=42),
    ]).reset_index(drop=True)
    print(f'Sample: {len(df):,} tweets ({SAMPLE_PER_CLASS:,} per class)')

    print('Preprocessing ...')
    df['text_clean'] = df['text'].apply(preprocess_tweet)

    print('Fitting TF-IDF ...')
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['text_clean'])
    y = df['sentiment'].map({0: 0, 4: 1}).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    print('Vectorizer saved.')

    for name, model in MODELS:
        print(f'Training {name} ...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f'  Accuracy: {acc:.4f}  F1: {f1:.4f}')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.joblib'))
        print(f'  Saved to models/{name}.joblib')

    print('\nDone! You can now run: python gui.py')


if __name__ == '__main__':
    main()