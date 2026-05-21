"""
train.py
========
End-to-end command-line training script for the Fake News Classification
pipeline. Trains all classical models and (optionally) fine-tunes DistilBERT.

Usage
-----
    # Train all classical models with Optuna tuning
    python src/train.py --data_path data/WELFake_Dataset.csv \
                        --dataset welfake \
                        --tune --n_trials 60 \
                        --save_models

    # Fine-tune DistilBERT only
    python src/train.py --data_path data/WELFake_Dataset.csv \
                        --model bert --epochs 3 --save_models

    # Quick smoke-test (no tuning, 5000 sample)
    python src/train.py --data_path data/WELFake_Dataset.csv \
                        --sample_n 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Ensure src/ is importable when called from project root
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import load_dataset, build_feature_pipeline
from models import (
    FakeNewsClassifier,
    VotingEnsemble,
    BERTClassifier,
    ClassificationMetrics,
)
from tuning import tune_all_models, tune_model
from visualise import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train fake-news classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path",   required=True,  help="Path to dataset CSV/TSV")
    p.add_argument("--dataset",     default="welfake",
                   choices=["welfake", "liar", "isot", "gossipcop"],
                   help="Dataset format schema")
    p.add_argument("--model",       default="all",
                   help="Model(s) to train: lr|nb|pa|rf|xgb|ensemble|bert|all")
    p.add_argument("--tune",        action="store_true", help="Enable Optuna tuning")
    p.add_argument("--n_trials",    type=int, default=60, help="Optuna trials per model")
    p.add_argument("--epochs",      type=int, default=3,  help="BERT training epochs")
    p.add_argument("--batch_size",  type=int, default=32, help="BERT batch size")
    p.add_argument("--val_size",    type=float, default=0.15, help="Validation fraction")
    p.add_argument("--test_size",   type=float, default=0.15, help="Test fraction")
    p.add_argument("--sample_n",    type=int, default=None, help="Subsample N rows")
    p.add_argument("--save_models", action="store_true", help="Persist models to disk")
    p.add_argument("--models_dir",  default="models/", help="Model output directory")
    p.add_argument("--outputs_dir", default="outputs/", help="Predictions output directory")
    p.add_argument("--figures_dir", default="figures/", help="Figures output directory")
    p.add_argument("--no_plots",    action="store_true", help="Skip plot generation")
    p.add_argument("--explain",     action="store_true", help="Generate SHAP explanations")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print(f"\n{'═'*60}\n  {text}\n{'═'*60}")


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs/60:.1f} min" if secs > 90 else f"{secs:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Main training workflow
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    # Create output dirs
    for d in (args.models_dir, args.outputs_dir, args.figures_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── 1. Load & split data ─────────────────────────────────────────────────
    _banner("1 / 6  ·  Loading dataset")
    df = load_dataset(
        args.data_path,
        dataset=args.dataset,
        sample_n=args.sample_n,
        random_state=args.random_state,
    )

    df_train_full, df_test = train_test_split(
        df, test_size=args.test_size,
        stratify=df["label"],
        random_state=args.random_state,
    )
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=args.val_size / (1 - args.test_size),
        stratify=df_train_full["label"],
        random_state=args.random_state,
    )

    print(f"  Train : {len(df_train):,}  |  Val : {len(df_val):,}  |  Test : {len(df_test):,}")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    selected = args.model.lower()
    run_classical = selected in ("all", "lr", "nb", "pa", "rf", "xgb", "ensemble")
    run_bert      = selected in ("all", "bert")

    all_metrics: list[ClassificationMetrics] = []
    trained_classifiers: dict = {}

    if run_classical:
        _banner("2 / 6  ·  Building feature matrix")
        pipeline = build_feature_pipeline()
        print("  Fitting TF-IDF + linguistic features on training set...")
        t0 = time.time()
        X_train = pipeline.fit_transform(df_train)
        X_val   = pipeline.transform(df_val)
        X_test  = pipeline.transform(df_test)
        y_train = df_train["label"].values
        y_val   = df_val["label"].values
        y_test  = df_test["label"].values
        print(f"  Feature matrix: {X_train.shape}  [{_elapsed(t0)}]")

        # ── 3. Hyperparameter tuning ─────────────────────────────────────────
        best_params: dict = {}
        if args.tune:
            _banner("3 / 6  ·  Hyperparameter tuning (Optuna TPE)")
            model_keys = (
                ["lr", "nb", "pa", "rf", "xgb"]
                if selected in ("all", "ensemble")
                else [selected]
            )
            best_params = tune_all_models(
                X_train, y_train,
                n_trials=args.n_trials,
                models=model_keys,
            )
            params_path = Path(args.outputs_dir) / "best_hyperparams.json"
            params_path.write_text(json.dumps(best_params, indent=2))
            print(f"\n  Best params saved → {params_path}")
        else:
            _banner("3 / 6  ·  Skipping tuning (use --tune to enable)")

        # ── 4. Train classical models ─────────────────────────────────────────
        _banner("4 / 6  ·  Training classical classifiers")

        model_factories = {
            "lr":  lambda: FakeNewsClassifier.logistic(
                **{k: v for k, v in best_params.get("lr", {}).items()}),
            "nb":  lambda: FakeNewsClassifier.naive_bayes(
                **{k: v for k, v in best_params.get("nb", {}).items()}),
            "pa":  lambda: FakeNewsClassifier.passive_aggressive(
                **{k: v for k, v in best_params.get("pa", {}).items()}),
            "rf":  lambda: FakeNewsClassifier.random_forest(
                **{k: v for k, v in best_params.get("rf", {}).items()}),
            "xgb": lambda: FakeNewsClassifier.xgboost(
                **{k: v for k, v in best_params.get("xgb", {}).items()}),
        }

        keys_to_train = (
            list(model_factories)
            if selected in ("all", "ensemble")
            else [selected] if selected in model_factories else []
        )

        for key in keys_to_train:
            t0  = time.time()
            clf = model_factories[key]()
            print(f"  Training {clf.name}...")
            clf.fit(X_train, y_train)
            metrics = clf.evaluate(X_test, y_test)
            all_metrics.append(metrics)
            trained_classifiers[key] = clf
            print(metrics)

            if args.save_models:
                clf.save(Path(args.models_dir) / f"{key}.joblib")

        # ── 4b. Voting ensemble ───────────────────────────────────────────────
        if selected in ("all", "ensemble") and len(trained_classifiers) >= 2:
            print("\n  Building voting ensemble...")
            ensemble = VotingEnsemble(list(trained_classifiers.values()))
            ensemble.fit(X_train, y_train)
            ensemble.optimise_weights(X_val, y_val)
            metrics  = ensemble.evaluate(X_test, y_test)
            all_metrics.append(metrics)
            trained_classifiers["ensemble"] = ensemble
            print(metrics)
            if args.save_models:
                ensemble.save(Path(args.models_dir) / "voting_ensemble.joblib")

    # ── 5. BERT fine-tuning ──────────────────────────────────────────────────
    if run_bert:
        _banner("5 / 6  ·  Fine-tuning DistilBERT")
        n_classes = df["label"].nunique()
        bert_clf  = BERTClassifier(
            n_classes  = n_classes,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            output_dir = str(Path(args.models_dir) / "distilbert"),
        )
        bert_clf.fit(df_train, df_val)
        metrics = bert_clf.evaluate(df_test)
        all_metrics.append(metrics)
        print(metrics)
        if args.save_models:
            bert_clf.save(Path(args.models_dir) / "distilbert")
    elif not run_classical:
        _banner("5 / 6  ·  (BERT skipped)")

    # ── 6. Results & visualisation ───────────────────────────────────────────
    _banner("6 / 6  ·  Results summary")

    if all_metrics:
        results_df = pd.DataFrame([m.to_series() for m in all_metrics])
        results_df = results_df.sort_values("F1 (Macro)", ascending=False).reset_index(drop=True)
        print(f"\n{results_df.to_string(index=False)}\n")
        results_path = Path(args.outputs_dir) / "model_comparison.csv"
        results_df.to_csv(results_path, index=False)
        print(f"  Results saved → {results_path}")

    if not args.no_plots and all_metrics and run_classical:
        figs = Path(args.figures_dir)
        print("\n  Generating figures...")
        try:
            # Use best classical classifier for detailed plots
            best_key = max(
                [k for k in trained_classifiers if k != "ensemble"],
                key=lambda k: trained_classifiers[k].evaluate(X_test, y_test).f1_macro,
                default=None,
            )
            if best_key:
                best_clf = trained_classifiers[best_key]
                plot_confusion_matrix(
                    y_test, best_clf.predict(X_test),
                    title=f"Confusion Matrix — {best_clf.name}",
                    save_path=figs / "confusion_matrix.png",
                )
                plot_roc_curves(
                    {name: clf for name, clf in trained_classifiers.items()},
                    X_test, y_test,
                    save_path=figs / "roc_curves.png",
                )
                plot_model_comparison(
                    all_metrics,
                    save_path=figs / "model_comparison.png",
                )
                if hasattr(best_clf, "estimator"):
                    plot_feature_importance(
                        best_clf, pipeline,
                        save_path=figs / "feature_importance.png",
                    )
            print(f"  Figures saved → {figs}/")
        except Exception as e:
            print(f"  [Warning] Figure generation failed: {e}")

    if args.explain and run_classical and "rf" in trained_classifiers:
        _banner("  SHAP Explanations")
        try:
            from explainability import SHAPExplainer
            explainer = SHAPExplainer(trained_classifiers["rf"])
            explainer.fit(X_train)
            explainer.plot_summary(
                X_test[:500],
                save_path=Path(args.figures_dir) / "shap_summary.png",
            )
            print(f"  SHAP summary saved → {args.figures_dir}/shap_summary.png")
        except Exception as e:
            print(f"  [Warning] SHAP failed: {e}")

    _banner(f"✓  Done  ·  Total time: {_elapsed(t_start)}")


if __name__ == "__main__":
    main()
