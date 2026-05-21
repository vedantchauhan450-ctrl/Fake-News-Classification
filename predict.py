"""
predict.py
==========
Inference script for the Fake News Classification pipeline.

Supports:
  - Single-article prediction (--title / --body flags)
  - Batch prediction from a CSV (--input_csv flag)
  - Optional LIME explanation (--explain flag)

Usage
-----
    # Single article
    python src/predict.py \
        --model_path models/voting_ensemble.joblib \
        --title "BREAKING: Scientists Discover Miracle Cure for Cancer" \
        --body  "Researchers at an unnamed lab claim they have found..." \
        --explain

    # Batch from CSV
    python src/predict.py \
        --model_path models/xgb.joblib \
        --input_csv  data/new_articles.csv \
        --output_csv outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from models import FakeNewsClassifier, VotingEnsemble, BERTClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────────────

BINARY_LABELS = {0: "REAL", 1: "FAKE"}
LIAR_LABELS   = {
    0: "pants-fire",
    1: "false",
    2: "mostly-false",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run fake-news predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True,
                   help="Path to saved .joblib model or DistilBERT directory")
    p.add_argument("--pipeline_path", default=None,
                   help="Path to saved feature pipeline .joblib (classical models)")
    p.add_argument("--title",      default="", help="Article headline (single-article mode)")
    p.add_argument("--body",       default="", help="Article body text (single-article mode)")
    p.add_argument("--input_csv",  default=None, help="CSV with 'title' and 'body' columns")
    p.add_argument("--output_csv", default="outputs/predictions.csv")
    p.add_argument("--explain",    action="store_true", help="Generate LIME explanation")
    p.add_argument("--liar",       action="store_true",
                   help="Use 6-class LIAR label map instead of binary")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Confidence threshold for FAKE label (binary only)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str):
    """Load a joblib classifier or DistilBERT directory."""
    import joblib
    p = Path(path)
    if p.is_dir():
        # DistilBERT directory
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        print(f"[predict] Loading DistilBERT from {p}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(p))
        model     = DistilBertForSequenceClassification.from_pretrained(str(p))
        return ("bert", model, tokenizer)
    else:
        print(f"[predict] Loading classical model from {p}")
        return ("classical", joblib.load(p))


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(title: str, body: str) -> pd.DataFrame:
    return pd.DataFrame([{"title": title, "body": body}])


def predict_classical(
    model,
    pipeline,
    df:        pd.DataFrame,
    threshold: float = 0.5,
    label_map: dict  = BINARY_LABELS,
) -> pd.DataFrame:
    """Run inference with a classical sklearn-based model."""
    X = pipeline.transform(df)
    proba = model.predict_proba(X)
    n_classes = proba.shape[1]

    if n_classes == 2:
        fake_prob = proba[:, 1]
        preds     = (fake_prob >= threshold).astype(int)
    else:
        preds     = np.argmax(proba, axis=1)
        fake_prob = np.max(proba, axis=1)

    labels = [label_map.get(p, str(p)) for p in preds]

    result = df.copy()
    result["prediction"]  = labels
    result["confidence"]  = np.round(fake_prob, 4)
    for i in range(n_classes):
        result[f"prob_class_{i}"] = np.round(proba[:, i], 4)
    return result


def predict_bert(
    model,
    tokenizer,
    df:        pd.DataFrame,
    max_length: int = 512,
    label_map: dict = BINARY_LABELS,
) -> pd.DataFrame:
    """Run inference with a fine-tuned DistilBERT model."""
    import torch

    texts = (df["title"].fillna("") + " [SEP] " + df["body"].fillna("")).tolist()
    enc   = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
    proba  = torch.softmax(logits, dim=-1).numpy()
    preds  = np.argmax(proba, axis=1)
    labels = [label_map.get(p, str(p)) for p in preds]

    result = df.copy()
    result["prediction"] = labels
    result["confidence"] = np.round(np.max(proba, axis=1), 4)
    for i in range(proba.shape[1]):
        result[f"prob_class_{i}"] = np.round(proba[:, i], 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LIME explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(model, pipeline, title: str, body: str, n_features: int = 15) -> None:
    """Print a LIME explanation for a single article."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("[explain] Install lime: pip install lime")
        return

    explainer = LimeTextExplainer(class_names=list(BINARY_LABELS.values()))

    def predict_fn(texts):
        rows = [{"title": t.split("[SEP]")[0].strip(),
                 "body":  t.split("[SEP]")[1].strip() if "[SEP]" in t else ""}
                for t in texts]
        X = pipeline.transform(pd.DataFrame(rows))
        return model.predict_proba(X)

    combined_text = f"{title} [SEP] {body}"
    exp = explainer.explain_instance(
        combined_text, predict_fn, num_features=n_features, num_samples=1000
    )

    print(f"\n{'─'*55}")
    print(f"  LIME Explanation")
    print(f"{'─'*55}")
    for feat, weight in sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
        direction = "→ FAKE" if weight > 0 else "→ REAL"
        print(f"  {feat:<30}  {weight:+.4f}  {direction}")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    label_map = LIAR_LABELS if args.liar else BINARY_LABELS
    loaded    = load_model(args.model_path)
    model_type = loaded[0]

    # Load feature pipeline for classical models
    pipeline = None
    if model_type == "classical":
        if args.pipeline_path:
            import joblib
            pipeline = joblib.load(args.pipeline_path)
        else:
            # Try default location
            default_path = Path(args.model_path).parent / "feature_pipeline.joblib"
            if default_path.exists():
                import joblib
                pipeline = joblib.load(default_path)
            else:
                print("[Warning] No feature pipeline found. "
                      "Pass --pipeline_path or ensure feature_pipeline.joblib is in models/")

    # ── Single-article mode ──────────────────────────────────────────────────
    if args.title or args.body:
        df = _make_df(args.title, args.body)

        if model_type == "classical" and pipeline:
            _, model = loaded
            result   = predict_classical(model, pipeline, df, args.threshold, label_map)
        elif model_type == "bert":
            _, model, tokenizer = loaded
            result = predict_bert(model, tokenizer, df, label_map=label_map)
        else:
            print("[Error] Could not run prediction — check model and pipeline paths.")
            return

        pred       = result["prediction"].iloc[0]
        confidence = result["confidence"].iloc[0]

        print(f"\n{'═'*55}")
        print(f"  Prediction  : {pred}")
        print(f"  Confidence  : {confidence:.1%}")
        print(f"  Title       : {args.title[:80]}...")
        print(f"{'═'*55}\n")

        if args.explain and model_type == "classical" and pipeline:
            _, model = loaded
            explain_with_lime(model, pipeline, args.title, args.body)

    # ── Batch mode ───────────────────────────────────────────────────────────
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        for col in ("title", "body"):
            if col not in df.columns:
                df[col] = ""

        print(f"[predict] Running batch inference on {len(df):,} articles...")

        if model_type == "classical" and pipeline:
            _, model = loaded
            result   = predict_classical(model, pipeline, df, args.threshold, label_map)
        elif model_type == "bert":
            _, model, tokenizer = loaded
            result = predict_bert(model, tokenizer, df, label_map=label_map)
        else:
            print("[Error] Could not run batch prediction.")
            return

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        print(f"[predict] Predictions saved → {args.output_csv}")

        fake_count = (result["prediction"] == "FAKE").sum()
        real_count = (result["prediction"] == "REAL").sum()
        print(f"\n  FAKE articles : {fake_count:,} ({fake_count/len(result):.1%})")
        print(f"  REAL articles : {real_count:,} ({real_count/len(result):.1%})")
        print(f"  Mean confidence: {result['confidence'].mean():.1%}\n")

    else:
        print("[predict] Provide --title/--body for single-article or --input_csv for batch.")


if __name__ == "__main__":
    main()
