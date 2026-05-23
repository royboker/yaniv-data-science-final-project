"""
Tweet Sentiment Analyzer — CustomTkinter GUI
4 models + Ensemble vote + Word highlighting + Analysis history
Run: python gui.py  (requires python save_model.py first)
"""
import os, re, math, threading
import joblib
import numpy as np
import customtkinter as ctk
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

# ── constants ────────────────────────────────────────────────────────────────
STOP_WORDS  = set(stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()
MODELS_DIR  = 'models'
COLOR_POS   = '#2ecc71'
COLOR_NEG   = '#e74c3c'
COLOR_NEUT  = '#3d3d3d'
COLOR_WARN  = '#f39c12'

NEGATION_WORDS  = ["not","no","never","n't","dont","doesnt","didnt","cant","wont","isnt","wasnt",
                   "couldn't","shouldn't","wouldn't","hardly","barely","scarcely"]
NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|n't|dont|cant|wont|isn't|wasn't|didn't|doesn't|"
    r"won't|can't|couldn't|shouldn't|wouldn't|hardly|barely|scarcely)\s+(\w+)",
    re.IGNORECASE
)
SARCASM_PHRASES = ["oh great","just great","wonderful","fantastic","love it","thanks a lot","brilliant"]
SLANG_POSITIVE  = ["sick","crazy","insane","wicked","savage","beast","fire","lit","goat","dope","killer"]

MODEL_DISPLAY = [
    ('naive_bayes',        'Naive Bayes'),
    ('logistic_regression','Logistic Regression'),
    ('linear_svc',         'Linear SVC'),
    ('random_forest',      'Random Forest'),
]

# ── preprocessing ─────────────────────────────────────────────────────────────
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

def pipeline_steps(text):
    steps = []
    steps.append(('Original tweet', text))
    t = str(text).lower()
    steps.append(('Lowercase', t))
    t = re.sub(r'http\S+|www\S+', '', t)
    steps.append(('Remove URLs', t))
    t = re.sub(r'@\w+', '', t)
    steps.append(('Remove @mentions', t))
    t = re.sub(r'^rt\s+', '', t)
    steps.append(('Remove RT prefix', t))
    t = re.sub(r'&\w+;', '', t)
    steps.append(('Remove HTML entities', t))
    t = t.replace('#', '')
    steps.append(('Remove # symbol', t))
    t = NEGATION_PATTERN.sub(lambda m: f"not_{m.group(2)}", t)
    steps.append(('Negation handling  (not X → not_X)', t))
    t = re.sub(r'[^a-z_\s]', '', t)
    steps.append(('Remove punctuation', t))
    t = re.sub(r'\s+', ' ', t).strip()
    tokens = word_tokenize(t)
    steps.append(('Tokenize', str(tokens)))
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    steps.append(('Remove stopwords', str(tokens)))
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    steps.append(('Lemmatize', str(tokens)))
    return steps

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

# ── model loading & prediction ────────────────────────────────────────────────
def load_models():
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    if not os.path.exists(vec_path):
        return None, {}
    vectorizer = joblib.load(vec_path)
    models = {}
    for key, _ in MODEL_DISPLAY:
        path = os.path.join(MODELS_DIR, f'{key}.joblib')
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return vectorizer, models

def predict_all(text, vectorizer, models):
    cleaned = preprocess_tweet(text)
    if not cleaned.strip():
        return {}
    X = vectorizer.transform([cleaned])
    results = {}
    for key, model in models.items():
        label = int(model.predict(X)[0])
        if hasattr(model, 'predict_proba'):
            confidence = float(max(model.predict_proba(X)[0]))
        else:
            score = float(model.decision_function(X)[0])
            confidence = sigmoid(abs(score))
        results[key] = (label, confidence)
    return results

def get_word_contributions(text, vectorizer, model, model_key, top_n=8):
    cleaned  = preprocess_tweet(text)
    words    = cleaned.split()
    if not words:
        return [], []

    X             = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_row     = np.asarray(X.todense()).flatten()

    if hasattr(model, 'coef_'):
        coefs = model.coef_.flatten()
    elif hasattr(model, 'feature_log_prob_'):
        coefs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        coefs = model.feature_importances_

    label = int(model.predict(X)[0])

    contributions = {}
    for word in set(words):
        indices = np.where(feature_names == word)[0]
        if len(indices) > 0:
            idx   = indices[0]
            score = float(tfidf_row[idx] * coefs[idx])
            contributions[word] = score

    sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    if label == 1:
        supporting = [(w, s)      for w, s in sorted_contribs if s > 0][:top_n]
        opposing   = [(w, abs(s)) for w, s in sorted_contribs if s < 0][:top_n]
    else:
        supporting = [(w, abs(s)) for w, s in sorted_contribs if s < 0][:top_n]
        opposing   = [(w, s)      for w, s in sorted_contribs if s > 0][:top_n]

    return supporting, opposing

def detect_warnings(text):
    warnings = []
    t = text.lower()
    if any(w in t for w in NEGATION_WORDS):
        warnings.append(('Negation detected', 'Words like "not", "don\'t" can confuse the model — it may miss the reversed meaning.'))
    if any(p in t for p in SARCASM_PHRASES):
        warnings.append(('Possible sarcasm', 'Phrases like "oh great" or "just wonderful" are often sarcastic — the model may predict the opposite.'))
    if any(w in t for w in SLANG_POSITIVE):
        warnings.append(('Slang word detected', 'Words like "sick", "fire", "crazy" can be positive slang — the model may misread them as negative.'))
    words = preprocess_tweet(text).split()
    if len(words) <= 3:
        warnings.append(('Very short tweet', f'Only {len(words)} word(s) after preprocessing — short tweets are harder to classify accurately.'))
    return warnings


# ── Ensemble Card ─────────────────────────────────────────────────────────────
class EnsembleCard(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, corner_radius=10, **kwargs)
        self.configure(fg_color='#1a2818')

        left = ctk.CTkFrame(self, fg_color='transparent')
        left.pack(side='left', padx=(16, 0), pady=12)

        ctk.CTkLabel(left, text='Ensemble Vote',
                     font=ctk.CTkFont(size=11, weight='bold'),
                     text_color='#777').pack(anchor='w')
        self.result_label = ctk.CTkLabel(left, text='—',
                                          font=ctk.CTkFont(size=22, weight='bold'),
                                          text_color='gray')
        self.result_label.pack(anchor='w')

        right = ctk.CTkFrame(self, fg_color='transparent')
        right.pack(side='right', fill='both', expand=True, padx=16, pady=12)

        self.agreement_label = ctk.CTkLabel(right,
                                             text='Run analysis to see consensus',
                                             font=ctk.CTkFont(size=11),
                                             text_color='#666', anchor='e')
        self.agreement_label.pack(anchor='e', pady=(4, 6))
        self.bar = ctk.CTkProgressBar(right, height=16, corner_radius=8)
        self.bar.pack(fill='x')
        self.bar.set(0)
        self.bar.configure(progress_color='#3d3d3d')

    def update(self, results):
        labels    = [v[0] for v in results.values()]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        is_split  = pos_count == neg_count
        majority  = 1 if pos_count > neg_count else 0
        agreement = max(pos_count, neg_count) / len(labels)
        agree_n   = max(pos_count, neg_count)

        if is_split:
            color = COLOR_WARN
            self.result_label.configure(text='SPLIT', text_color=color)
            self.agreement_label.configure(
                text=f'2/4 — Models disagree  ·  no clear winner')
        else:
            color = COLOR_POS if majority == 1 else COLOR_NEG
            self.result_label.configure(
                text='POSITIVE' if majority == 1 else 'NEGATIVE', text_color=color)
            self.agreement_label.configure(
                text=f'{agree_n}/{len(labels)} models agree  ·  {agreement*100:.0f}% consensus')
        self.bar.set(agreement)
        self.bar.configure(progress_color=color)

    def reset(self):
        self.result_label.configure(text='—', text_color='gray')
        self.agreement_label.configure(text='Run analysis to see consensus')
        self.bar.set(0)
        self.bar.configure(progress_color='#3d3d3d')


# ── Explanation Window ────────────────────────────────────────────────────────
class ExplanationWindow(ctk.CTkToplevel):
    def __init__(self, parent, raw_text, model_key, model_name,
                 label, confidence, vectorizer, model):
        super().__init__(parent)
        self.title(f'{model_name} — Explanation')
        self.geometry('680x640')
        self.resizable(False, False)
        self.grab_set()

        self.raw_text   = raw_text
        self.model_key  = model_key
        self.model_name = model_name
        self.label      = label
        self.confidence = confidence
        self.vectorizer = vectorizer
        self.model      = model

        self._build_ui()

    def _build_ui(self):
        color  = COLOR_POS if self.label == 1 else COLOR_NEG
        result = 'POSITIVE' if self.label == 1 else 'NEGATIVE'

        header = ctk.CTkFrame(self, fg_color='#1e1e1e', corner_radius=0)
        header.pack(fill='x')
        ctk.CTkLabel(header, text=self.model_name,
                     font=ctk.CTkFont(size=18, weight='bold')).pack(side='left', padx=20, pady=14)
        ctk.CTkLabel(header, text=f'{result}  {self.confidence*100:.1f}%',
                     font=ctk.CTkFont(size=15, weight='bold'),
                     text_color=color).pack(side='right', padx=20)

        tab_frame = ctk.CTkFrame(self, fg_color='#2b2b2b', corner_radius=0)
        tab_frame.pack(fill='x')

        self.tab_word = ctk.CTkButton(
            tab_frame, text='Word Analysis', width=180, height=36,
            font=ctk.CTkFont(size=13, weight='bold'),
            corner_radius=0, fg_color='#3b3b3b',
            command=self._show_word_tab)
        self.tab_word.pack(side='left', padx=(10, 2), pady=8)

        self.tab_pipe = ctk.CTkButton(
            tab_frame, text='Pipeline', width=180, height=36,
            font=ctk.CTkFont(size=13),
            corner_radius=0, fg_color='transparent',
            command=self._show_pipeline_tab)
        self.tab_pipe.pack(side='left', padx=2, pady=8)

        self.content = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.content.pack(fill='both', expand=True, padx=0, pady=0)

        self._show_word_tab()

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _show_word_tab(self):
        self.tab_word.configure(fg_color='#3b3b3b')
        self.tab_pipe.configure(fg_color='transparent')
        self._clear_content()

        color  = COLOR_POS if self.label == 1 else COLOR_NEG
        result = 'POSITIVE' if self.label == 1 else 'NEGATIVE'

        ctk.CTkLabel(self.content, text='Original tweet:',
                     font=ctk.CTkFont(size=12), text_color='gray',
                     anchor='w').pack(fill='x', padx=20, pady=(16, 2))
        tweet_box = ctk.CTkTextbox(self.content, height=60,
                                   font=ctk.CTkFont(size=12), state='normal')
        tweet_box.pack(fill='x', padx=20)
        tweet_box.insert('1.0', self.raw_text)
        tweet_box.configure(state='disabled')

        cleaned = preprocess_tweet(self.raw_text)
        ctk.CTkLabel(self.content, text=f'After preprocessing:  {cleaned}',
                     font=ctk.CTkFont(size=11), text_color='gray',
                     wraplength=580, justify='left', anchor='w').pack(
                         fill='x', padx=20, pady=(4, 14))

        supporting, opposing = get_word_contributions(
            self.raw_text, self.vectorizer, self.model, self.model_key)

        self._word_section(f'Words pushing toward {result}', supporting, color)
        opp_color = COLOR_NEG if self.label == 1 else COLOR_POS
        opp_label = 'NEGATIVE' if self.label == 1 else 'POSITIVE'
        self._word_section(f'Words pushing toward {opp_label}', opposing, opp_color)

        warnings = detect_warnings(self.raw_text)
        if warnings:
            ctk.CTkLabel(self.content, text='Warnings from Error Analysis:',
                         font=ctk.CTkFont(size=13, weight='bold'),
                         text_color=COLOR_WARN, anchor='w').pack(
                             fill='x', padx=20, pady=(16, 6))
            for title, desc in warnings:
                card = ctk.CTkFrame(self.content, fg_color='#2a1f00', corner_radius=8)
                card.pack(fill='x', padx=20, pady=4)
                ctk.CTkLabel(card, text=f'  {title}',
                             font=ctk.CTkFont(size=12, weight='bold'),
                             text_color=COLOR_WARN, anchor='w').pack(
                                 fill='x', padx=10, pady=(8, 2))
                ctk.CTkLabel(card, text=desc,
                             font=ctk.CTkFont(size=11), text_color='#cccccc',
                             wraplength=560, justify='left', anchor='w').pack(
                                 fill='x', padx=10, pady=(0, 8))
        else:
            ctk.CTkLabel(self.content,
                         text='No warnings detected — tweet looks straightforward.',
                         font=ctk.CTkFont(size=12), text_color='gray',
                         anchor='w').pack(fill='x', padx=20, pady=(16, 0))

    def _word_section(self, title, word_scores, color):
        ctk.CTkLabel(self.content, text=title,
                     font=ctk.CTkFont(size=13, weight='bold'),
                     anchor='w').pack(fill='x', padx=20, pady=(14, 4))
        if not word_scores:
            ctk.CTkLabel(self.content, text='  No significant words found.',
                         font=ctk.CTkFont(size=11), text_color='gray',
                         anchor='w').pack(fill='x', padx=20)
            return
        max_score = max(s for _, s in word_scores) or 1
        for word, score in word_scores:
            row = ctk.CTkFrame(self.content, fg_color='transparent')
            row.pack(fill='x', padx=20, pady=2)
            ctk.CTkLabel(row, text=word, width=120,
                         font=ctk.CTkFont(size=12), anchor='w').pack(side='left')
            bar = ctk.CTkProgressBar(row, height=14, corner_radius=4,
                                     progress_color=color)
            bar.pack(side='left', fill='x', expand=True, padx=(6, 8))
            bar.set(score / max_score)
            ctk.CTkLabel(row, text=f'{score:.3f}', width=54,
                         font=ctk.CTkFont(size=11), text_color='gray').pack(side='left')

    def _show_pipeline_tab(self):
        self.tab_pipe.configure(fg_color='#3b3b3b')
        self.tab_word.configure(fg_color='transparent')
        self._clear_content()

        ctk.CTkLabel(self.content,
                     text='How the tweet was transformed step by step:',
                     font=ctk.CTkFont(size=13, weight='bold'),
                     anchor='w').pack(fill='x', padx=20, pady=(16, 10))

        steps = pipeline_steps(self.raw_text)
        for i, (name, result) in enumerate(steps):
            card = ctk.CTkFrame(self.content, fg_color='#2b2b2b', corner_radius=8)
            card.pack(fill='x', padx=20, pady=4)

            ctk.CTkLabel(card, text=f'{i+1}',
                         font=ctk.CTkFont(size=13, weight='bold'),
                         text_color='#888', width=28).pack(side='left', padx=(10, 0), pady=10)

            inner = ctk.CTkFrame(card, fg_color='transparent')
            inner.pack(side='left', fill='x', expand=True, padx=10, pady=8)
            ctk.CTkLabel(inner, text=name,
                         font=ctk.CTkFont(size=11, weight='bold'),
                         text_color='#aaa', anchor='w').pack(fill='x')
            ctk.CTkLabel(inner, text=result if result.strip() else '(empty)',
                         font=ctk.CTkFont(size=12),
                         text_color='white' if result.strip() else '#666',
                         wraplength=520, justify='left', anchor='w').pack(fill='x')

            if i < len(steps) - 1:
                ctk.CTkLabel(self.content, text='↓',
                             font=ctk.CTkFont(size=14), text_color='#555').pack()

        ctk.CTkLabel(self.content, text='TF-IDF — Top weighted words in this tweet:',
                     font=ctk.CTkFont(size=13, weight='bold'),
                     anchor='w').pack(fill='x', padx=20, pady=(16, 4))

        cleaned = preprocess_tweet(self.raw_text)
        if cleaned.strip():
            X             = self.vectorizer.transform([cleaned])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_row     = np.asarray(X.todense()).flatten()
            top_idx       = tfidf_row.argsort()[-8:][::-1]
            top_items     = [(feature_names[i], tfidf_row[i])
                             for i in top_idx if tfidf_row[i] > 0]

            if top_items:
                max_val = max(v for _, v in top_items)
                for word, val in top_items:
                    row = ctk.CTkFrame(self.content, fg_color='transparent')
                    row.pack(fill='x', padx=20, pady=2)
                    ctk.CTkLabel(row, text=word, width=130,
                                 font=ctk.CTkFont(size=12), anchor='w').pack(side='left')
                    bar = ctk.CTkProgressBar(row, height=12, corner_radius=4,
                                             progress_color='#3498db')
                    bar.pack(side='left', fill='x', expand=True, padx=(6, 8))
                    bar.set(val / max_val)
                    ctk.CTkLabel(row, text=f'{val:.4f}', width=60,
                                 font=ctk.CTkFont(size=11),
                                 text_color='gray').pack(side='left')
            else:
                ctk.CTkLabel(self.content,
                             text='No words matched the vocabulary.',
                             font=ctk.CTkFont(size=11), text_color='gray',
                             anchor='w').pack(fill='x', padx=20)


# ── Model Card ────────────────────────────────────────────────────────────────
class ModelCard(ctk.CTkFrame):
    def __init__(self, parent, key, title, on_click, **kwargs):
        super().__init__(parent, corner_radius=10, **kwargs)
        self.configure(fg_color='#2b2b2b', cursor='hand2')
        self.key      = key
        self.on_click = on_click
        self._label   = None
        self._conf    = 0.0

        ctk.CTkLabel(self, text=title,
                     font=ctk.CTkFont(size=13, weight='bold'),
                     text_color='white').pack(pady=(12, 4))

        self.result_label = ctk.CTkLabel(self, text='—',
                                         font=ctk.CTkFont(size=18, weight='bold'),
                                         text_color='gray')
        self.result_label.pack()

        self.conf_label = ctk.CTkLabel(self, text='',
                                       font=ctk.CTkFont(size=11),
                                       text_color='gray')
        self.conf_label.pack(pady=(2, 4))

        self.bar = ctk.CTkProgressBar(self, height=10, corner_radius=5)
        self.bar.pack(padx=14, fill='x', pady=(0, 6))
        self.bar.set(0)
        self.bar.configure(progress_color=COLOR_NEUT)

        self.detail_btn = ctk.CTkButton(
            self, text='View explanation', height=26,
            font=ctk.CTkFont(size=11), corner_radius=6,
            fg_color='#3d3d3d', hover_color='#505050',
            state='disabled', command=self._on_click)
        self.detail_btn.pack(padx=14, fill='x', pady=(0, 12))

    def update(self, label, confidence):
        self._label = label
        self._conf  = confidence
        color = COLOR_POS if label == 1 else COLOR_NEG
        text  = 'POSITIVE' if label == 1 else 'NEGATIVE'
        self.result_label.configure(text=text, text_color=color)
        self.conf_label.configure(text=f'{confidence*100:.1f}% confidence')
        self.bar.set(confidence)
        self.bar.configure(progress_color=color)
        self.detail_btn.configure(state='normal')

    def reset(self):
        self._label = None
        self.result_label.configure(text='—', text_color='gray')
        self.conf_label.configure(text='')
        self.bar.set(0)
        self.bar.configure(progress_color=COLOR_NEUT)
        self.detail_btn.configure(state='disabled')

    def _on_click(self):
        if self._label is not None:
            self.on_click(self.key, self._label, self._conf)


# ── Main App ──────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self, vectorizer, models):
        super().__init__()
        self.vectorizer = vectorizer
        self.models     = models
        self._raw_text  = ''
        self.history    = []  # list of (snippet, majority_label, pos_count, total)

        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')

        self.title('Tweet Sentiment Analyzer')
        self.geometry('580x780')
        self.resizable(True, True)
        self.minsize(520, 680)

        self._build_ui()

    def _build_ui(self):
        # Scrollable container — lets user scroll through all content
        scroll = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color='transparent')
        scroll.pack(fill='both', expand=True)
        p = scroll  # shorthand — all widgets go into p

        ctk.CTkLabel(p, text='Tweet Sentiment Analyzer',
                     font=ctk.CTkFont(size=22, weight='bold')).pack(pady=(20, 2))
        ctk.CTkLabel(p, text='Sentiment140  ·  4 Models  ·  TF-IDF',
                     font=ctk.CTkFont(size=12), text_color='gray').pack(pady=(0, 12))

        ctk.CTkLabel(p, text='Enter a tweet:', anchor='w',
                     font=ctk.CTkFont(size=13)).pack(padx=36, fill='x')
        self.text_input = ctk.CTkTextbox(p, height=80,
                                         font=ctk.CTkFont(size=13),
                                         corner_radius=10)
        self.text_input.pack(padx=36, fill='x', pady=(4, 10))
        self.text_input.bind('<Control-v>', self._paste)
        self.text_input.bind('<Control-V>', self._paste)
        self.text_input.bind('<Control-Return>', lambda e: self._on_analyze())

        # Example tweet chips
        ex_frame = ctk.CTkFrame(p, fg_color='transparent')
        ex_frame.pack(padx=36, fill='x', pady=(0, 8))
        ctk.CTkLabel(ex_frame, text='Try:',
                     font=ctk.CTkFont(size=11), text_color='#555').pack(side='left', padx=(0, 6))
        examples = [
            ('😊 Positive',    'I absolutely love this! Best day ever, feeling amazing.'),
            ('😞 Negative',    'Worst experience ever. Completely ruined my day.'),
            ('🔀 Negation',    "This is not bad at all, I'm not disappointed."),
            ('😏 Sarcasm',     'Oh great, another Monday. Just wonderful.'),
        ]
        for chip_label, tweet in examples:
            ctk.CTkButton(ex_frame, text=chip_label, height=26,
                          font=ctk.CTkFont(size=11), corner_radius=14,
                          fg_color='#2b2b2b', hover_color='#3d3d3d',
                          command=lambda t=tweet: self._load_example(t)
                          ).pack(side='left', padx=3)

        btn_frame = ctk.CTkFrame(p, fg_color='transparent')
        btn_frame.pack(padx=36, fill='x', pady=(0, 10))
        ctk.CTkButton(btn_frame, text='Analyze',
                      font=ctk.CTkFont(size=14, weight='bold'),
                      height=40, corner_radius=10,
                      command=self._on_analyze).pack(side='left', expand=True,
                                                      fill='x', padx=(0, 6))
        ctk.CTkButton(btn_frame, text='Clear',
                      font=ctk.CTkFont(size=14), height=40, corner_radius=10,
                      fg_color='transparent', border_width=1,
                      command=self._on_clear).pack(side='left', expand=True,
                                                    fill='x', padx=(6, 0))

        # Ensemble card — prominent verdict
        self.ensemble_card = EnsembleCard(p)
        self.ensemble_card.pack(padx=36, fill='x', pady=(0, 10))

        # 2×2 model grid
        grid = ctk.CTkFrame(p, fg_color='transparent')
        grid.pack(padx=36, fill='x', pady=(0, 10))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.cards = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (row, col), (key, title) in zip(positions, MODEL_DISPLAY):
            card = ModelCard(grid, key, title, on_click=self._open_explanation)
            card.grid(row=row, column=col, padx=6, pady=6, sticky='nsew')
            self.cards[key] = card

        # History section
        hist_header = ctk.CTkFrame(p, fg_color='transparent')
        hist_header.pack(padx=36, fill='x', pady=(4, 0))
        ctk.CTkLabel(hist_header, text='Recent analyses',
                     font=ctk.CTkFont(size=12, weight='bold'),
                     text_color='#666').pack(side='left')

        self.hist_frame = ctk.CTkFrame(p, fg_color='#232323', corner_radius=8)
        self.hist_frame.pack(padx=36, fill='x', pady=(4, 20))
        ctk.CTkLabel(self.hist_frame, text='No analyses yet',
                     font=ctk.CTkFont(size=11), text_color='#555').pack(pady=10)

    def _on_analyze(self):
        self._raw_text = self.text_input.get('1.0', 'end').strip()
        if not self._raw_text:
            self._on_clear()
            return

        for card in self.cards.values():
            card.reset()
        self.ensemble_card.reset()
        self._clear_highlights()

        def run():
            results = predict_all(self._raw_text, self.vectorizer, self.models)
            self.after(0, lambda: self._on_results(results))

        threading.Thread(target=run, daemon=True).start()

    def _on_results(self, results):
        for key, card in self.cards.items():
            if key in results:
                card.update(*results[key])
        if results:
            self.ensemble_card.update(results)
            self._highlight_words_in_input(self._raw_text)
            self._add_to_history(self._raw_text, results)

    def _highlight_words_in_input(self, raw_text):
        """Color words directly in the input box: green = positive signal, red = negative."""
        if 'linear_svc' not in self.models:
            return
        try:
            supporting, opposing = get_word_contributions(
                raw_text, self.vectorizer, self.models['linear_svc'], 'linear_svc')
            pos_set = {w for w, _ in supporting}
            neg_set = {w for w, _ in opposing}

            tb = self.text_input._textbox
            tb.tag_configure('pos_word', foreground=COLOR_POS)
            tb.tag_configure('neg_word', foreground=COLOR_NEG)

            for m in re.finditer(r'\w+', raw_text):
                word    = m.group()
                cleaned = preprocess_tweet(word)
                if cleaned in pos_set:
                    tb.tag_add('pos_word', f'1.0+{m.start()}c', f'1.0+{m.end()}c')
                elif cleaned in neg_set:
                    tb.tag_add('neg_word', f'1.0+{m.start()}c', f'1.0+{m.end()}c')
        except Exception:
            pass

    def _clear_highlights(self):
        try:
            tb = self.text_input._textbox
            tb.tag_remove('pos_word', '1.0', 'end')
            tb.tag_remove('neg_word', '1.0', 'end')
        except Exception:
            pass

    def _add_to_history(self, text, results):
        labels    = [v[0] for v in results.values()]
        pos_count = sum(labels)
        majority  = 1 if pos_count >= len(labels) / 2 else 0
        snippet   = text[:55] + '…' if len(text) > 55 else text
        self.history.insert(0, (snippet, majority, pos_count, len(labels)))
        if len(self.history) > 5:
            self.history.pop()
        self._refresh_history()

    def _refresh_history(self):
        for w in self.hist_frame.winfo_children():
            w.destroy()

        if not self.history:
            ctk.CTkLabel(self.hist_frame, text='No analyses yet',
                         font=ctk.CTkFont(size=11), text_color='#555').pack(pady=10)
            return

        for i, (snippet, majority, pos_count, total) in enumerate(self.history):
            color = COLOR_POS if majority == 1 else COLOR_NEG
            label = 'POSITIVE' if majority == 1 else 'NEGATIVE'
            agree = max(pos_count, total - pos_count)

            row = ctk.CTkFrame(self.hist_frame, fg_color='transparent')
            row.pack(fill='x', padx=10,
                     pady=(8 if i == 0 else 2, 8 if i == len(self.history) - 1 else 2))

            ctk.CTkLabel(row, text='●', font=ctk.CTkFont(size=10),
                         text_color=color, width=16).pack(side='left')
            ctk.CTkLabel(row, text=snippet,
                         font=ctk.CTkFont(size=11), text_color='#ccc',
                         anchor='w').pack(side='left', fill='x', expand=True, padx=(4, 8))
            ctk.CTkLabel(row, text=f'{label}  {agree}/{total}',
                         font=ctk.CTkFont(size=11, weight='bold'),
                         text_color=color).pack(side='right')

            if i < len(self.history) - 1:
                ctk.CTkFrame(self.hist_frame, height=1,
                             fg_color='#333').pack(fill='x', padx=10)

    def _load_example(self, tweet):
        self.text_input.delete('1.0', 'end')
        self.text_input.insert('1.0', tweet)
        self._on_analyze()

    def _paste(self, event=None):
        try:
            text = self.clipboard_get()
            self.text_input.insert('insert', text)
        except Exception:
            pass
        return 'break'

    def _on_clear(self):
        self.text_input.delete('1.0', 'end')
        self._raw_text = ''
        self._clear_highlights()
        for card in self.cards.values():
            card.reset()
        self.ensemble_card.reset()

    def _open_explanation(self, key, label, confidence):
        _, title = next(d for d in MODEL_DISPLAY if d[0] == key)
        ExplanationWindow(self, self._raw_text, key, title,
                          label, confidence,
                          self.vectorizer, self.models[key])


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    vectorizer, models = load_models()
    if vectorizer is None or not models:
        import tkinter.messagebox as mb, tkinter as tk
        root = tk.Tk(); root.withdraw()
        mb.showerror('Models not found',
                     'Run  python save_model.py  first.')
        root.destroy()
        return
    App(vectorizer, models).mainloop()


if __name__ == '__main__':
    main()