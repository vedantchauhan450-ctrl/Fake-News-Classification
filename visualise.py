"""
visualise.py
============
Publication-quality figures for the Fake News Classification pipeline.

All functions save to disk if `save_path` is provided, and also call
plt.show() for interactive use in Jupyter notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
})

PALETTE = {
    "REAL":  "#2ecc71",
    "FAKE":  "#e74c3c",
    "model": sns.color_palette("tab10"),
}


def _save(fig: plt.Figure, path: Optional[Path | str]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [visualise] Saved → {path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels:     Optional[List[str]] = None,
    title:      str = "Confusion Matrix",
    normalise:  bool = True,
    save_path:  Optional[str | Path] = None,
) -> None:
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    if labels is None:
        n = cm.shape[0]
        labels = ["REAL", "FAKE"] if n == 2 else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f" if normalise else "d",
        cmap="Blues", xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")

    # Overlay raw counts if normalised
    if normalise:
        for i, j in np.ndindex(cm.shape):
            ax.text(j + 0.5, i + 0.75, f"n={cm[i,j]:,}",
                    ha="center", va="center", fontsize=8, color="grey")

    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    classifiers: Dict[str, object],
    X_test,
    y_test,
    save_path:   Optional[str | Path] = None,
) -> None:
    """Multi-model ROC comparison with AUC legend."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = PALETTE["model"]

    for i, (name, clf) in enumerate(classifiers.items()):
        try:
            proba = clf.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            auc_val      = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colours[i % len(colours)],
                    label=f"{name.replace('_', ' ').title()}  (AUC = {auc_val:.3f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision–Recall curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(
    classifiers: Dict[str, object],
    X_test,
    y_test,
    save_path:   Optional[str | Path] = None,
) -> None:
    """Precision-Recall curves (preferred for imbalanced classes)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(7, 6))
    colours = PALETTE["model"]

    for i, (name, clf) in enumerate(classifiers.items()):
        try:
            proba  = clf.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ap   = average_precision_score(y_test, proba)
            ax.plot(rec, prec, lw=2, color=colours[i % len(colours)],
                    label=f"{name.replace('_', ' ').title()}  (AP = {ap:.3f})")
        except Exception:
            continue

    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision–Recall Curves", fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    metrics_list,
    metrics:   List[str] = ("Accuracy", "F1 (Macro)", "AUC-ROC"),
    save_path: Optional[str | Path] = None,
) -> None:
    """Grouped bar chart comparing all models across key metrics."""
    rows = [m.to_series() for m in metrics_list]
    df   = pd.DataFrame(rows).set_index("Model")

    # Keep only numeric columns that exist
    cols = [c for c in metrics if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    df   = df[cols].astype(float)

    n_models  = len(df)
    n_metrics = len(cols)
    x         = np.arange(n_models)
    width     = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 5))

    colours = PALETTE["model"]
    for i, metric in enumerate(cols):
        bars = ax.bar(
            x + i * width - (n_metrics - 1) * width / 2,
            df[metric],
            width=width * 0.9,
            color=colours[i % len(colours)],
            label=metric,
            zorder=3,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Model Performance Comparison", fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    classifier,
    pipeline,
    top_n:     int = 40,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Horizontal bar chart of top-N feature importances.
    Works with Random Forest (feature_importances_) and Logistic Regression
    (coef_).
    """
    from preprocessing import LinguisticFeatures

    try:
        est = classifier.estimator
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_[0])
        else:
            print("[visualise] Model does not expose feature importances.")
            return
    except AttributeError:
        return

    # Build feature name list
    tfidf_names = pipeline._tfidf.feature_names if hasattr(pipeline, "_tfidf") else []
    ling_names  = [f"ling:{n}" for n in LinguisticFeatures.feature_names()]
    all_names   = tfidf_names + ling_names

    n = min(len(importances), len(all_names))
    importances = importances[:n]
    all_names   = all_names[:n]

    top_idx   = np.argsort(importances)[-top_n:]
    top_names = [all_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    colours = ["#e74c3c" if "ling:" in n else "#3498db" for n in top_names]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(range(len(top_names)), top_vals, color=colours, edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances — {classifier.name}",
                 fontweight="bold", pad=15)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#3498db", label="TF-IDF token"),
                 Patch(color="#e74c3c", label="Linguistic feature")],
        loc="lower right", framealpha=0.9,
    )
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Learning curves (BERT training)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    train_losses:  List[float],
    val_losses:    List[float],
    val_f1_scores: List[float],
    save_path:     Optional[str | Path] = None,
) -> None:
    """Train vs validation loss + F1 per BERT epoch."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, "o-", color="#3498db", lw=2, label="Train loss")
    ax1.plot(epochs, val_losses,   "s-", color="#e74c3c",  lw=2, label="Val loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # F1
    ax2.plot(epochs, val_f1_scores, "D-", color="#2ecc71", lw=2, label="Val F1 (macro)")
    ax2.axhline(max(val_f1_scores), color="grey", ls="--", lw=1,
                label=f"Best: {max(val_f1_scores):.4f}")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("F1 Score (Macro)", fontweight="bold")
    ax2.set_title("Validation F1 by Epoch", fontweight="bold")
    ax2.legend()
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("DistilBERT Fine-Tuning Progress", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Label distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(
    df:        pd.DataFrame,
    label_map: Optional[Dict[int, str]] = None,
    title:     str = "Dataset Label Distribution",
    save_path: Optional[str | Path] = None,
) -> None:
    """Annotated bar chart of class frequencies."""
    counts    = df["label"].value_counts().sort_index()
    if label_map:
        index = [label_map.get(i, str(i)) for i in counts.index]
    else:
        index = [str(i) for i in counts.index]

    colours = [PALETTE["FAKE"] if "FAKE" in lbl or "false" in lbl.lower()
               else PALETTE["REAL"] for lbl in index]

    fig, ax = plt.subplots(figsize=(max(5, len(index) * 1.4), 4))
    bars = ax.bar(index, counts.values, color=colours, edgecolor="white", zorder=3)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}\n({count/counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, save_path)
