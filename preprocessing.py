"""
preprocessing.py
================
Text normalisation, linguistic feature engineering, and vectorisation
for the Fake News Classification pipeline.

Supports four dataset formats:
    - WELFake  : 'title', 'text', 'label' (0=Real, 1=Fake)
    - LIAR     : TSV with statement + speaker metadata, 6-class label
    - ISOT     : 'title', 'text', 'subject' separated in two CSVs
    - GossipCop: 'title', 'content', 'label'
"""

from __future__ import annotations

import re
import string
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

DATASET_SCHEMAS = {
    "welfake":   {"title": "title",   "body": "text",    "label": "label"},
    "isot":      {"title": "title",   "body": "text",    "label": "label"},
    "gossipcop": {"title": "title",   "body": "content", "label": "label"},
    "liar":      {"title": "speaker", "body": "statement", "label": "label"},
}


def load_dataset(
    path: str | Path,
    dataset: str = "welfake",
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Read a fake-news CSV / TSV, standardise column names to
    ['title', 'body', 'label'], optionally subsample.

    Parameters
    ----------
    path        : Path to the dataset file.
    dataset     : One of 'welfake', 'liar', 'isot', 'gossipcop'.
    sample_n    : If set, return a stratified random sample of this size.
    random_state: Reproducibility seed.

    Returns
    -------
    pd.DataFrame with columns ['title', 'body', 'label'].
    """
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, on_bad_lines="skip")

    schema = DATASET_SCHEMAS.get(dataset.lower())
    if schema is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_SCHEMAS)}")

    df = df.rename(columns={v: k for k, v in schema.items()})
    for col in ("title", "body", "label"):
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["body"]  = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    if sample_n is not None:
        df = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), sample_n // df["label"].nunique()),
                                        random_state=random_state))
              .reset_index(drop=True)
        )

    print(f"[load_dataset] Loaded {len(df):,} rows — label distribution:\n"
          f"{df['label'].value_counts().to_string()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE   = re.compile(r"<[^>]+>")
_MULTI_SP  = re.compile(r"\s+")
_ELLIPSIS  = re.compile(r"\.{2,}")
_PUNCT_RE  = re.compile(r"[^\w\s!?.,;:'\"()-]")   # keep basic punctuation


def clean_text(text: str, keep_punctuation: bool = True) -> str:
    """
    Lightweight cleaning: lowercase, URL/HTML strip, unicode normalise,
    collapse whitespace.

    Parameters
    ----------
    text             : Raw input string.
    keep_punctuation : If False, strip all punctuation (for TF-IDF).

    Returns
    -------
    Cleaned string.
    """
    text = unicodedata.normalize("NFKD", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.lower()
    if not keep_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    else:
        text = _PUNCT_RE.sub(" ", text)
        text = _ELLIPSIS.sub(".", text)
    text = _MULTI_SP.sub(" ", text).strip()
    return text


class TextNormaliser(BaseEstimator, TransformerMixin):
    """
    spaCy-based tokeniser with stopword removal and lemmatisation.

    Falls back to simple whitespace tokenisation if spaCy is unavailable.

    Parameters
    ----------
    model           : spaCy model name (default: 'en_core_web_sm').
    remove_stopwords: Drop spaCy stopwords.
    lemmatise       : Replace tokens with their base lemma.
    min_token_len   : Drop tokens shorter than this.
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        remove_stopwords: bool = True,
        lemmatise: bool = True,
        min_token_len: int = 2,
    ):
        self.model            = model
        self.remove_stopwords = remove_stopwords
        self.lemmatise        = lemmatise
        self.min_token_len    = min_token_len
        self._nlp             = None

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load(self.model, disable=["parser", "ner"])
        except (ImportError, OSError):
            warnings.warn(
                "spaCy model not found — falling back to basic tokenisation. "
                "Run: python -m spacy download en_core_web_sm",
                stacklevel=2,
            )
            self._nlp = None

    def _normalise_one(self, text: str) -> str:
        text = clean_text(text, keep_punctuation=False)
        if self._nlp is None:
            return text
        doc = self._nlp(text)
        tokens = []
        for tok in doc:
            if tok.is_space:
                continue
            if self.remove_stopwords and tok.is_stop:
                continue
            word = tok.lemma_ if self.lemmatise else tok.text
            if len(word) >= self.min_token_len:
                tokens.append(word)
        return " ".join(tokens)

    def fit(self, X, y=None):
        self._load_spacy()
        return self

    def transform(self, X):
        return [self._normalise_one(str(text)) for text in X]


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinguisticFeatures:
    """Flat container for per-document linguistic signals."""
    title_len:        float = 0.0
    body_len:         float = 0.0
    avg_word_len:     float = 0.0
    type_token_ratio: float = 0.0
    exclamation_count: float = 0.0
    question_count:   float = 0.0
    caps_ratio:       float = 0.0
    quote_count:      float = 0.0
    number_count:     float = 0.0
    has_date:         float = 0.0
    past_tense_ratio: float = 0.0
    flesch_kincaid:   float = 0.0
    vader_compound:   float = 0.0
    vader_pos:        float = 0.0
    vader_neg:        float = 0.0
    vader_neu:        float = 0.0
    source_count:     float = 0.0
    ellipsis_count:   float = 0.0

    def to_list(self) -> List[float]:
        return list(self.__dict__.values())

    @staticmethod
    def feature_names() -> List[str]:
        return list(LinguisticFeatures.__dataclass_fields__.keys())


_DATE_RE   = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4})\b",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"\b(according to|reported by|said|stated|told reporters|"
    r"confirmed|spokesperson|officials said)\b",
    re.IGNORECASE,
)
_PAST_RE   = re.compile(r"\b\w+ed\b")


def _flesch_kincaid(text: str) -> float:
    """Approximate Flesch-Kincaid Grade Level."""
    words     = text.split()
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    syllables = sum(_count_syllables(w) for w in words)
    n_words   = max(len(words), 1)
    return 0.39 * (n_words / sentences) + 11.8 * (syllables / n_words) - 15.59


def _count_syllables(word: str) -> int:
    word = word.lower().rstrip("e")
    return max(len(re.findall(r"[aeiou]+", word)), 1)


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Computes 18 handcrafted linguistic features per (title, body) pair.

    Expects input as a pd.DataFrame with columns ['title', 'body'],
    or a list of strings (body-only mode).

    Returns
    -------
    np.ndarray of shape (n_samples, 18).
    """

    def __init__(self, use_vader: bool = True):
        self.use_vader = use_vader
        self._sid = None

    def _load_vader(self):
        if self.use_vader and self._sid is None:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                self._sid = SentimentIntensityAnalyzer()
            except ImportError:
                self.use_vader = False

    def fit(self, X, y=None):
        self._load_vader()
        return self

    def _extract_one(self, title: str, body: str) -> LinguisticFeatures:
        full = f"{title} {body}"
        words = full.split()
        n_words = max(len(words), 1)

        vader_scores = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        if self._sid is not None:
            vader_scores = self._sid.polarity_scores(full[:512])

        return LinguisticFeatures(
            title_len          = len(title.split()),
            body_len           = len(body.split()),
            avg_word_len       = sum(len(w) for w in words) / n_words,
            type_token_ratio   = len(set(w.lower() for w in words)) / n_words,
            exclamation_count  = full.count("!"),
            question_count     = full.count("?"),
            caps_ratio         = sum(1 for c in full if c.isupper()) / max(len(full), 1),
            quote_count        = full.count('"') // 2 + full.count("'") // 2,
            number_count       = len(re.findall(r"\b\d+\b", full)),
            has_date           = float(bool(_DATE_RE.search(full))),
            past_tense_ratio   = len(_PAST_RE.findall(full)) / n_words,
            flesch_kincaid     = _flesch_kincaid(body),
            vader_compound     = vader_scores["compound"],
            vader_pos          = vader_scores["pos"],
            vader_neg          = vader_scores["neg"],
            vader_neu          = vader_scores["neu"],
            source_count       = len(_SOURCE_RE.findall(full)),
            ellipsis_count     = full.count("..."),
        )

    def transform(self, X) -> np.ndarray:
        rows = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                rows.append(self._extract_one(str(row.get("title", "")),
                                               str(row.get("body", ""))))
        else:
            for text in X:
                rows.append(self._extract_one("", str(text)))
        return np.array([r.to_list() for r in rows], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a single string column from a DataFrame."""
    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        if isinstance(X, pd.DataFrame):
            return X[self.column].fillna("").astype(str).tolist()
        return [str(x) for x in X]


class TfidfPipeline(BaseEstimator, TransformerMixin):
    """
    Dual TF-IDF vectoriser (title + body) producing a sparse feature matrix.

    Parameters
    ----------
    max_title_features : Max vocabulary size for the title vectoriser.
    max_body_features  : Max vocabulary size for the body vectoriser.
    ngram_range        : Token n-gram range (default: unigrams + bigrams).
    min_df             : Minimum document frequency.
    """

    def __init__(
        self,
        max_title_features: int = 20_000,
        max_body_features:  int = 50_000,
        ngram_range:        Tuple[int, int] = (1, 2),
        min_df:             int = 3,
    ):
        self.max_title_features = max_title_features
        self.max_body_features  = max_body_features
        self.ngram_range        = ngram_range
        self.min_df             = min_df

        self._title_vec = TfidfVectorizer(
            max_features=max_title_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._body_vec = TfidfVectorizer(
            max_features=max_body_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._normaliser = TextNormaliser()

    def _get_columns(self, X) -> Tuple[List[str], List[str]]:
        if isinstance(X, pd.DataFrame):
            titles = X["title"].fillna("").astype(str).tolist()
            bodies = X["body"].fillna("").astype(str).tolist()
        else:
            titles = [""] * len(X)
            bodies = [str(x) for x in X]
        return titles, bodies

    def fit(self, X, y=None):
        titles, bodies = self._get_columns(X)
        self._normaliser.fit(titles + bodies)
        clean_titles = self._normaliser.transform(titles)
        clean_bodies = self._normaliser.transform(bodies)
        self._title_vec.fit(clean_titles)
        self._body_vec.fit(clean_bodies)
        return self

    def transform(self, X) -> csr_matrix:
        titles, bodies = self._get_columns(X)
        clean_titles = self._normaliser.transform(titles)
        clean_bodies = self._normaliser.transform(bodies)
        T = self._title_vec.transform(clean_titles)
        B = self._body_vec.transform(clean_bodies)
        return hstack([T, B], format="csr")

    @property
    def feature_names(self) -> List[str]:
        return (
            [f"title_{t}" for t in self._title_vec.get_feature_names_out()]
            + [f"body_{t}"  for t in self._body_vec.get_feature_names_out()]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full feature pipeline (TF-IDF + linguistic features)
# ─────────────────────────────────────────────────────────────────────────────

class CombinedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Horizontal stack of:
        1. Dual TF-IDF (sparse)
        2. 18 linguistic features (dense, standardised)

    Returns a sparse CSR matrix compatible with all sklearn classifiers.
    """

    def __init__(self, tfidf_kw: Optional[dict] = None):
        self.tfidf_kw = tfidf_kw or {}
        self._tfidf   = TfidfPipeline(**self.tfidf_kw)
        self._ling    = LinguisticFeatureExtractor()
        self._scaler  = StandardScaler(with_mean=False)

    def fit(self, X: pd.DataFrame, y=None):
        self._tfidf.fit(X)
        ling_features = self._ling.fit_transform(X)
        self._scaler.fit(ling_features)
        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        from scipy.sparse import csr_matrix as _csr
        tfidf_mat = self._tfidf.transform(X)
        ling_mat  = self._scaler.transform(self._ling.transform(X))
        return hstack([tfidf_mat, _csr(ling_mat)], format="csr")


def build_feature_pipeline(tfidf_kw: Optional[dict] = None) -> CombinedFeatureTransformer:
    """
    Convenience factory — returns a fitted-ready CombinedFeatureTransformer.

    Usage
    -----
    >>> pipeline = build_feature_pipeline()
    >>> X_train_feats = pipeline.fit_transform(df_train)
    >>> X_test_feats  = pipeline.transform(df_test)
    """
    return CombinedFeatureTransformer(tfidf_kw=tfidf_kw)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace dataset builder (for BERT fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def build_bert_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 512,
):
    """
    Tokenise a DataFrame for DistilBERT fine-tuning.

    Input format: [CLS] {title} [SEP] {body[:max_length]} [SEP]

    Parameters
    ----------
    df         : DataFrame with 'title', 'body', 'label' columns.
    tokenizer  : HuggingFace tokenizer instance.
    max_length : Maximum sequence length.

    Returns
    -------
    datasets.Dataset ready for HuggingFace Trainer.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install `datasets`: pip install datasets") from exc

    records = {
        "text":  (df["title"].fillna("") + " [SEP] " + df["body"].fillna("")).tolist(),
        "label": df["label"].tolist(),
    }
    hf_dataset = Dataset.from_dict(records)

    def tokenise_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return hf_dataset.map(tokenise_batch, batched=True)
