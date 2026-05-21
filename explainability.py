"""
explainability.py
=================
SHAP and LIME explainability wrappers for the Fake News Classification
pipeline.

Supported models:
  - SHAPExplainer: TreeExplainer (RF, XGB), LinearExplainer (LR, PA)
  - LIMEExplainer: Any model with predict_proba (model-agnostic)

Usage
-----
    from explainability import SHAPExplainer, LIMEExplainer

    # Global SHAP summary
    shap_exp = SHAPExplainer(clf)
    shap_exp.fit(X_train_sample)
    shap_exp.plot_summary(X_test[:500], save_path="figures/shap_summary.png")

    # Local LIME explanation
    lime_exp = LIMEExplainer(clf, pipeline)
    lime_exp.explain("Miracle cure discovered", "Unnamed scientists claim...", top_n=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SHAP wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP-based global and local explainability.

    Selects the appropriate SHAP explainer based on model type:
      - TreeExplainer  → Random Forest, XGBoost
      - LinearExplainer → Logistic Regression, Passive Aggressive

    Parameters
    ----------
    classifier  : Fitted FakeNewsClassifier instance.
    feature_names : Optional list of feature names for plot labels.
    """

    def __init__(self, classifier, feature_names: Optional[List[str]] = None):
        self.classifier    = classifier
        self.feature_names = feature_names
        self._explainer    = None

    def fit(self, X_background) -> "SHAPExplainer":
        """
        Initialise the SHAP explainer with a background dataset.

        Parameters
        ----------
        X_background : Background data (e.g. X_train or a sample thereof).
                       For TreeExplainer this is not required but improves
                       expected-value calculation.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install shap: pip install shap") from exc

        est = self.classifier.estimator

        # Unwrap CalibratedClassifierCV
        if hasattr(est, "estimator"):
            est = est.estimator

        model_type = type(est).__name__
        if "Forest" in model_type or "XGB" in model_type or "Boost" in model_type:
            self._explainer = shap.TreeExplainer(est)
        else:
            self._explainer = shap.LinearExplainer(est, X_background, feature_perturbation="interventional")

        return self

    def shap_values(self, X) -> np.ndarray:
        """Compute SHAP values for samples X."""
        if self._explainer is None:
            raise RuntimeError("Call .fit(X_background) first.")
        return self._explainer.shap_values(X)

    def plot_summary(
        self,
        X,
        max_display: int = 20,
        save_path:   Optional[str | Path] = None,
    ) -> None:
        """
        Beeswarm summary plot showing global feature impact.

        Parameters
        ----------
        X           : Feature matrix to explain (recommend ≤ 1000 samples).
        max_display : Number of top features to display.
        save_path   : If provided, save figure to this path.
        """
        import shap
        import matplotlib.pyplot as plt

        shap_vals = self.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class 1 = FAKE for binary

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title(f"SHAP Feature Importance — {self.classifier.name}",
                  fontsize=13, fontweight="bold", pad=12)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [SHAP] Saved → {save_path}")
        plt.show()
        plt.close()

    def plot_waterfall(
        self,
        X,
        idx:       int = 0,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Waterfall plot for a single prediction — shows feature contributions
        from the expected value to the final prediction.
        """
        import shap
        import matplotlib.pyplot as plt

        shap_vals = self.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values         = shap_vals[idx],
                base_values    = self._explainer.expected_value
                                 if not isinstance(self._explainer.expected_value, list)
                                 else self._explainer.expected_value[1],
                data           = X[idx] if not hasattr(X, "toarray") else X[idx].toarray().flatten(),
                feature_names  = self.feature_names,
            ),
            show=False,
        )
        plt.title("SHAP Waterfall — Single Prediction", fontweight="bold", pad=12)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def force_plot_html(self, X, idx: int = 0) -> str:
        """Generate an interactive SHAP force plot as an HTML string."""
        import shap
        shap_vals = self.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        ev = (self._explainer.expected_value
              if not isinstance(self._explainer.expected_value, list)
              else self._explainer.expected_value[1])
        html = shap.force_plot(ev, shap_vals[idx], X[idx],
                               feature_names=self.feature_names, matplotlib=False)
        return html


# ─────────────────────────────────────────────────────────────────────────────
# LIME wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    LIME-based local explainability — works with any classifier that has
    predict_proba, including fine-tuned BERT.

    Parameters
    ----------
    classifier : Fitted FakeNewsClassifier (or BERTClassifier).
    pipeline   : Fitted CombinedFeatureTransformer (not needed for BERT).
    class_names: Display labels for classes.
    """

    def __init__(
        self,
        classifier,
        pipeline     = None,
        class_names: List[str] = ("REAL", "FAKE"),
    ):
        self.classifier  = classifier
        self.pipeline    = pipeline
        self.class_names = list(class_names)
        self._explainer  = None

    def _build_explainer(self):
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as exc:
            raise ImportError("Install lime: pip install lime") from exc
        self._explainer = LimeTextExplainer(class_names=self.class_names)

    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """Prediction function wrapper for LIME."""
        rows = []
        for t in texts:
            parts = t.split("[SEP]", 1)
            rows.append({
                "title": parts[0].strip(),
                "body":  parts[1].strip() if len(parts) > 1 else "",
            })
        df = pd.DataFrame(rows)
        if self.pipeline is not None:
            X = self.pipeline.transform(df)
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict_proba(df)

    def explain(
        self,
        title:      str,
        body:       str,
        top_n:      int = 15,
        num_samples: int = 1000,
        save_html:  Optional[str | Path] = None,
        print_explanation: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Generate a LIME explanation for a single article.

        Parameters
        ----------
        title        : Article headline.
        body         : Article body text.
        top_n        : Number of top features to return.
        num_samples  : LIME perturbation samples.
        save_html    : If provided, save interactive HTML to this path.
        print_explanation : Print ranked token contributions.

        Returns
        -------
        List of (token, weight) sorted by |weight| descending.
        """
        if self._explainer is None:
            self._build_explainer()

        combined = f"{title} [SEP] {body}"
        exp = self._explainer.explain_instance(
            combined,
            self._predict_fn,
            num_features=top_n,
            num_samples=num_samples,
            labels=(1,),   # explain class 1 = FAKE
        )

        features = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=True)

        if print_explanation:
            # Get prediction confidence
            proba = self._predict_fn([combined])[0]
            pred_class = np.argmax(proba)
            confidence = proba[pred_class]
            label      = self.class_names[pred_class]

            print(f"\n{'─'*60}")
            print(f"  LIME Local Explanation")
            print(f"{'─'*60}")
            print(f"  Prediction : {label}  (confidence: {confidence:.1%})")
            print(f"  Title      : {title[:70]}...")
            print(f"{'─'*60}")
            print(f"  {'Token':<30}  {'Weight':>8}  Direction")
            print(f"  {'─'*50}")
            for token, weight in features:
                direction = "→ FAKE" if weight > 0 else "→ REAL"
                bar = "█" * int(abs(weight) * 60)
                print(f"  {token:<30}  {weight:>+8.4f}  {direction}  {bar}")
            print(f"{'─'*60}\n")

        if save_html:
            html_path = Path(save_html)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(exp.as_html())
            print(f"  [LIME] Interactive HTML saved → {html_path}")

        return features

    def batch_explain(
        self,
        df:         pd.DataFrame,
        top_n:      int = 10,
        num_samples: int = 500,
    ) -> pd.DataFrame:
        """
        Generate LIME explanations for a batch of articles.

        Returns
        -------
        DataFrame with columns: title, prediction, confidence,
        and top_feature_1..top_feature_N with their weights.
        """
        records = []
        for _, row in df.iterrows():
            features = self.explain(
                str(row.get("title", "")),
                str(row.get("body",  "")),
                top_n=top_n,
                num_samples=num_samples,
                print_explanation=False,
            )
            rec = {"title": row.get("title", "")}
            for i, (token, weight) in enumerate(features[:top_n]):
                rec[f"feature_{i+1}"]       = token
                rec[f"feature_{i+1}_weight"] = round(weight, 5)
            records.append(rec)

        return pd.DataFrame(records)
