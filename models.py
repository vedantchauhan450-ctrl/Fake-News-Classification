"""
models.py
=========
Classifier wrappers, soft-voting ensemble, and DistilBERT fine-tuning
for the Fake News Classification pipeline.

All classical classifiers expose a unified interface:
    .fit(X, y)  →  self
    .evaluate(X, y)  →  ClassificationMetrics
    .predict(X)  →  np.ndarray
    .predict_proba(X)  →  np.ndarray
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    """Structured container for all evaluation metrics."""
    model_name:   str
    accuracy:     float
    f1_macro:     float
    f1_weighted:  float
    precision:    float
    recall:       float
    auc_roc:      Optional[float]
    report:       str

    def to_series(self) -> pd.Series:
        return pd.Series({
            "Model":       self.model_name,
            "Accuracy":    round(self.accuracy, 4),
            "F1 (Macro)":  round(self.f1_macro, 4),
            "F1 (Wt.)":    round(self.f1_weighted, 4),
            "Precision":   round(self.precision, 4),
            "Recall":      round(self.recall, 4),
            "AUC-ROC":     round(self.auc_roc, 4) if self.auc_roc else "N/A",
        })

    def __str__(self) -> str:
        lines = [
            f"\n{'─'*55}",
            f"  Model     : {self.model_name}",
            f"  Accuracy  : {self.accuracy:.4f}",
            f"  F1 Macro  : {self.f1_macro:.4f}",
            f"  F1 Wtd.   : {self.f1_weighted:.4f}",
            f"  Precision : {self.precision:.4f}",
            f"  Recall    : {self.recall:.4f}",
        ]
        if self.auc_roc is not None:
            lines.append(f"  AUC-ROC   : {self.auc_roc:.4f}")
        lines.append(f"{'─'*55}")
        lines.append(self.report)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core classifier wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FakeNewsClassifier:
    """
    Unified wrapper around sklearn classifiers for fake news detection.

    Use the factory class methods to instantiate specific models:
        clf = FakeNewsClassifier.logistic()
        clf = FakeNewsClassifier.naive_bayes()
        clf = FakeNewsClassifier.passive_aggressive()
        clf = FakeNewsClassifier.random_forest()
        clf = FakeNewsClassifier.xgboost()
    """

    def __init__(self, estimator, name: str):
        self.estimator  = estimator
        self.name       = name
        self._le        = LabelEncoder()
        self._is_fitted = False

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def logistic(cls, C: float = 1.0, max_iter: int = 1000, **kw) -> "FakeNewsClassifier":
        """L2-regularised Logistic Regression."""
        est = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter,
            multi_class="auto", n_jobs=-1, **kw
        )
        return cls(est, name="Logistic Regression")

    @classmethod
    def naive_bayes(cls, alpha: float = 0.1, **kw) -> "FakeNewsClassifier":
        """Multinomial Naive Bayes with Laplace smoothing."""
        return cls(MultinomialNB(alpha=alpha, **kw), name="Naive Bayes")

    @classmethod
    def passive_aggressive(cls, C: float = 1.0, max_iter: int = 1000, **kw) -> "FakeNewsClassifier":
        """Passive Aggressive Classifier — fast online-learnable baseline."""
        est = CalibratedClassifierCV(
            PassiveAggressiveClassifier(C=C, max_iter=max_iter, random_state=42, **kw),
            cv=3,
        )
        return cls(est, name="Passive Aggressive")

    @classmethod
    def random_forest(
        cls,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        **kw,
    ) -> "FakeNewsClassifier":
        """Random Forest with calibrated probability estimates."""
        est = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            **kw,
        )
        return cls(est, name="Random Forest")

    @classmethod
    def xgboost(
        cls,
        eta: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 300,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kw,
    ) -> "FakeNewsClassifier":
        """XGBoost gradient boosting classifier."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost: pip install xgboost") from exc

        est = XGBClassifier(
            learning_rate=eta,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **kw,
        )
        return cls(est, name="XGBoost")

    # ── Training & inference ───────────────────────────────────────────────────

    def fit(self, X, y) -> "FakeNewsClassifier":
        """
        Train the classifier.

        Parameters
        ----------
        X : Sparse or dense feature matrix (n_samples, n_features).
        y : Integer label array.
        """
        self._le.fit(y)
        y_enc = self._le.transform(y)
        self.estimator.fit(X, y_enc)
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self._le.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        """Return probability estimates (n_samples, n_classes)."""
        self._check_fitted()
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        raise AttributeError(f"{self.name} does not support predict_proba.")

    def evaluate(self, X, y_true) -> ClassificationMetrics:
        """
        Compute full evaluation metrics on held-out data.

        Returns
        -------
        ClassificationMetrics dataclass.
        """
        self._check_fitted()
        y_pred   = self.predict(X)
        n_classes = len(self._le.classes_)

        try:
            proba   = self.predict_proba(X)
            auc_roc = (
                roc_auc_score(y_true, proba[:, 1])
                if n_classes == 2
                else roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
        except Exception:
            auc_roc = None

        return ClassificationMetrics(
            model_name  = self.name,
            accuracy    = accuracy_score(y_true, y_pred),
            f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0),
            precision   = precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall      = recall_score(y_true, y_pred, average="macro", zero_division=0),
            auc_roc     = auc_roc,
            report      = classification_report(y_true, y_pred, zero_division=0),
        )

    def save(self, path: str | Path) -> None:
        """Persist the fitted classifier to disk."""
        self._check_fitted()
        joblib.dump(self, path)
        print(f"[{self.name}] Saved → {path}")

    @staticmethod
    def load(path: str | Path) -> "FakeNewsClassifier":
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"Call .fit() before using {self.name}.")


# ─────────────────────────────────────────────────────────────────────────────
# Soft-voting ensemble
# ─────────────────────────────────────────────────────────────────────────────

class VotingEnsemble:
    """
    Weighted soft-voting blend of multiple FakeNewsClassifier instances.

    Weights are optimised on a held-out validation set to maximise F1 macro
    by grid search over [0.0, 0.5, 1.0, 1.5, 2.0] per model.

    Parameters
    ----------
    classifiers : List of fitted FakeNewsClassifier instances.
    weights     : Optional manual weight list (same length as classifiers).
    """

    def __init__(
        self,
        classifiers: List[FakeNewsClassifier],
        weights: Optional[List[float]] = None,
    ):
        self.classifiers = classifiers
        self.weights     = weights or [1.0] * len(classifiers)
        self.name        = "Voting Ensemble"
        self._is_fitted  = False

    def fit(self, X, y) -> "VotingEnsemble":
        """Fit all constituent classifiers (if not already fitted)."""
        for clf in self.classifiers:
            if not clf._is_fitted:
                clf.fit(X, y)
        self._le = LabelEncoder().fit(y)
        self._is_fitted = True
        return self

    def optimise_weights(self, X_val, y_val) -> List[float]:
        """
        Grid-search optimal per-model weights on a validation split.

        Returns
        -------
        List of optimised weights (also stored in self.weights).
        """
        from itertools import product

        candidates = [0.0, 0.5, 1.0, 1.5, 2.0]
        best_f1, best_w = -np.inf, self.weights

        for combo in product(candidates, repeat=len(self.classifiers)):
            if sum(combo) == 0:
                continue
            self.weights = list(combo)
            preds = self.predict(X_val)
            score = f1_score(y_val, preds, average="macro", zero_division=0)
            if score > best_f1:
                best_f1, best_w = score, list(combo)

        self.weights = best_w
        print(f"[VotingEnsemble] Optimised weights: {best_w}  (val F1={best_f1:.4f})")
        return best_w

    def predict_proba(self, X) -> np.ndarray:
        """Weighted average of class probabilities across all classifiers."""
        total = sum(self.weights)
        blended = sum(
            w * clf.predict_proba(X)
            for clf, w in zip(self.classifiers, self.weights)
        ) / total
        return blended

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self._le.inverse_transform(np.argmax(proba, axis=1))

    def evaluate(self, X, y_true) -> ClassificationMetrics:
        y_pred   = self.predict(X)
        n_classes = len(self._le.classes_)
        try:
            proba   = self.predict_proba(X)
            auc_roc = (
                roc_auc_score(y_true, proba[:, 1])
                if n_classes == 2
                else roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
        except Exception:
            auc_roc = None

        return ClassificationMetrics(
            model_name  = self.name,
            accuracy    = accuracy_score(y_true, y_pred),
            f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0),
            precision   = precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall      = recall_score(y_true, y_pred, average="macro", zero_division=0),
            auc_roc     = auc_roc,
            report      = classification_report(y_true, y_pred, zero_division=0),
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        print(f"[VotingEnsemble] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT fine-tuning wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BERTClassifier:
    """
    DistilBERT fine-tuning wrapper using the HuggingFace Trainer API.

    Architecture:
        [CLS] title [SEP] body[:480] [SEP]
            → DistilBERT (6 layers, 768-d)
            → [CLS] pooled → Dropout(0.3) → Linear(768, n_classes) → Softmax

    Parameters
    ----------
    model_name  : HuggingFace model identifier (default: 'distilbert-base-uncased').
    n_classes   : Number of output classes (2 for binary, 6 for LIAR).
    max_length  : Token sequence length.
    epochs      : Fine-tuning epochs.
    batch_size  : Per-device training batch size.
    lr          : Peak learning rate (linear warmup then decay).
    output_dir  : Directory to save checkpoints.
    """

    def __init__(
        self,
        model_name:  str = "distilbert-base-uncased",
        n_classes:   int = 2,
        max_length:  int = 512,
        epochs:      int = 3,
        batch_size:  int = 32,
        lr:          float = 2e-5,
        output_dir:  str = "models/distilbert",
    ):
        self.model_name  = model_name
        self.n_classes   = n_classes
        self.max_length  = max_length
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.output_dir  = output_dir
        self._model      = None
        self._tokenizer  = None
        self._trainer    = None

    def _load(self):
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError as exc:
            raise ImportError("Install transformers: pip install transformers") from exc

        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self._model     = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_classes
        )

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> "BERTClassifier":
        """
        Fine-tune DistilBERT on the training set.

        Parameters
        ----------
        df_train : DataFrame with 'title', 'body', 'label'.
        df_val   : Validation split for early-stopping metric.
        """
        from transformers import TrainingArguments, Trainer
        from preprocessing import build_bert_dataset
        import evaluate as hf_evaluate

        self._load()
        train_ds = build_bert_dataset(df_train, self._tokenizer, self.max_length)
        val_ds   = build_bert_dataset(df_val,   self._tokenizer, self.max_length)

        metric = hf_evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels, average="macro")

        args = TrainingArguments(
            output_dir                  = self.output_dir,
            num_train_epochs            = self.epochs,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size * 2,
            learning_rate               = self.lr,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            evaluation_strategy         = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            logging_steps               = 50,
            fp16                        = True,   # requires GPU
            report_to                   = "none",
        )

        self._trainer = Trainer(
            model           = self._model,
            args            = args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            compute_metrics = compute_metrics,
        )
        self._trainer.train()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return softmax probabilities (n_samples, n_classes)."""
        import torch
        from preprocessing import build_bert_dataset

        ds = build_bert_dataset(df, self._tokenizer, self.max_length)
        predictions = self._trainer.predict(ds)
        logits = predictions.predictions
        proba  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def evaluate(self, df: pd.DataFrame) -> ClassificationMetrics:
        y_true = df["label"].tolist()
        y_pred = self.predict(df)
        proba  = self.predict_proba(df)
        n_classes = self.n_classes

        try:
            auc_roc = (
                roc_auc_score(y_true, proba[:, 1])
                if n_classes == 2
                else roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
        except Exception:
            auc_roc = None

        return ClassificationMetrics(
            model_name  = f"DistilBERT ({self.model_name})",
            accuracy    = accuracy_score(y_true, y_pred),
            f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0),
            precision   = precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall      = recall_score(y_true, y_pred, average="macro", zero_division=0),
            auc_roc     = auc_roc,
            report      = classification_report(y_true, y_pred, zero_division=0),
        )

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        print(f"[BERTClassifier] Saved → {directory}")
