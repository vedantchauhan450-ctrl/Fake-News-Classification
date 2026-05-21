"""
tuning.py
=========
Bayesian hyperparameter optimisation (Optuna TPE) for all classical
fake-news classifiers, with RandomizedSearchCV fallback.

Usage
-----
    from tuning import tune_model

    best_params = tune_model("xgboost", X_train, y_train, n_trials=60)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Search space definitions
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_logistic(trial) -> Dict[str, Any]:
    return {
        "C":        trial.suggest_float("C", 0.001, 100.0, log=True),
        "solver":   trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": 1000,
    }


def _suggest_naive_bayes(trial) -> Dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 0.001, 2.0, log=True)}


def _suggest_passive_aggressive(trial) -> Dict[str, Any]:
    return {
        "C":        trial.suggest_float("C", 0.001, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 2000),
    }


def _suggest_random_forest(trial) -> Dict[str, Any]:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth":         trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _suggest_xgboost(trial) -> Dict[str, Any]:
    return {
        "eta":              trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


_SEARCH_SPACES = {
    "lr":               _suggest_logistic,
    "logistic":         _suggest_logistic,
    "nb":               _suggest_naive_bayes,
    "naive_bayes":      _suggest_naive_bayes,
    "pa":               _suggest_passive_aggressive,
    "passive_aggressive": _suggest_passive_aggressive,
    "rf":               _suggest_random_forest,
    "random_forest":    _suggest_random_forest,
    "xgb":              _suggest_xgboost,
    "xgboost":          _suggest_xgboost,
}

# ─────────────────────────────────────────────────────────────────────────────
# Classifier factory (parameterised)
# ─────────────────────────────────────────────────────────────────────────────

def _build_estimator(model_key: str, params: Dict[str, Any]):
    """Return an instantiated sklearn estimator from the model key + params."""
    from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV

    k = model_key.lower()
    if k in ("lr", "logistic"):
        return LogisticRegression(n_jobs=-1, **params)
    elif k in ("nb", "naive_bayes"):
        return MultinomialNB(**params)
    elif k in ("pa", "passive_aggressive"):
        return CalibratedClassifierCV(
            PassiveAggressiveClassifier(random_state=42, **params), cv=3
        )
    elif k in ("rf", "random_forest"):
        return RandomForestClassifier(n_jobs=-1, random_state=42,
                                       class_weight="balanced", **params)
    elif k in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("pip install xgboost") from exc
        return XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, **params
        )
    else:
        raise ValueError(f"Unknown model key '{model_key}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    model_key:    str,
    X_train,
    y_train,
    n_trials:     int = 60,
    cv:           int = 5,
    scoring:      str = "f1_macro",
    n_jobs:       int = 1,
    random_state: int = 42,
    verbose:      bool = True,
) -> Dict[str, Any]:
    """
    Optimise hyperparameters for a fake-news classifier using Optuna.

    Falls back to RandomizedSearchCV if Optuna is not installed.

    Parameters
    ----------
    model_key    : One of 'lr', 'nb', 'pa', 'rf', 'xgb'.
    X_train      : Feature matrix (sparse or dense).
    y_train      : Integer label array.
    n_trials     : Number of Optuna TPE trials.
    cv           : Cross-validation folds.
    scoring      : sklearn scoring metric (default: 'f1_macro').
    n_jobs       : Parallelism (1 = sequential, safe for sparse inputs).
    random_state : Reproducibility seed.
    verbose      : Print best result summary.

    Returns
    -------
    Dict of best hyperparameters.
    """
    suggest_fn = _SEARCH_SPACES.get(model_key.lower())
    if suggest_fn is None:
        raise ValueError(f"No search space defined for '{model_key}'.")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )

        def objective(trial):
            params = suggest_fn(trial)
            est    = _build_estimator(model_key, params)
            scores = cross_val_score(
                est, X_train, y_train,
                cv=skf, scoring=scoring, n_jobs=n_jobs, error_score=0.0,
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        best_params = study.best_params

        if verbose:
            print(f"\n[Optuna | {model_key}]  Best {scoring} = {study.best_value:.4f}")
            print(f"  Best params: {best_params}\n")

        return best_params

    except ImportError:
        warnings.warn(
            "Optuna not found — falling back to RandomizedSearchCV. "
            "Install with: pip install optuna",
            stacklevel=2,
        )
        return _randomised_search_fallback(
            model_key, X_train, y_train,
            n_iter=n_trials, cv=skf, scoring=scoring,
            random_state=random_state, verbose=verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RandomizedSearchCV fallback
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_DISTRIBUTIONS = {
    "lr": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "nb": {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "pa": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [500, 1000, 1500],
    },
    "rf": {
        "n_estimators":      [100, 200, 300, 400],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "xgb": {
        "eta":              [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth":        [3, 4, 6, 8],
        "n_estimators":     [100, 200, 300],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
}


def _randomised_search_fallback(
    model_key:    str,
    X_train,
    y_train,
    n_iter:       int,
    cv,
    scoring:      str,
    random_state: int,
    verbose:      bool,
) -> Dict[str, Any]:
    from sklearn.model_selection import RandomizedSearchCV

    est        = _build_estimator(model_key, {})
    param_dist = _PARAM_DISTRIBUTIONS.get(model_key.lower(), {})

    search = RandomizedSearchCV(
        est, param_dist,
        n_iter=min(n_iter, 20),
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        random_state=random_state,
        verbose=1 if verbose else 0,
    )
    search.fit(X_train, y_train)

    if verbose:
        print(f"\n[RandomizedSearch | {model_key}]  "
              f"Best {scoring} = {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}\n")

    return search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: tune all models
# ─────────────────────────────────────────────────────────────────────────────

def tune_all_models(
    X_train,
    y_train,
    n_trials: int = 60,
    models:   Optional[list] = None,
    **kw,
) -> Dict[str, Dict[str, Any]]:
    """
    Run hyperparameter tuning for every classical model.

    Parameters
    ----------
    X_train  : Feature matrix.
    y_train  : Labels.
    n_trials : Trials per model.
    models   : Subset to tune (default: all).

    Returns
    -------
    Dict mapping model_key → best_params.
    """
    targets = models or ["lr", "nb", "pa", "rf", "xgb"]
    results = {}
    for key in targets:
        print(f"\n{'═'*55}\n  Tuning: {key.upper()}\n{'═'*55}")
        results[key] = tune_model(key, X_train, y_train, n_trials=n_trials, **kw)
    return results
