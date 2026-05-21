"""
fake-news-classification
========================
A production-grade NLP pipeline for detecting misinformation.

Modules
-------
preprocessing   Text cleaning, linguistic feature engineering, TF-IDF vectorisation
models          Classical classifiers, voting ensemble, DistilBERT fine-tuning
tuning          Optuna TPE and RandomizedSearchCV hyperparameter search
explainability  SHAP global explanations and LIME local explanations
visualise       Publication-quality plots (confusion matrix, ROC, SHAP, etc.)
train           End-to-end CLI training script
predict         Inference on new articles (single or batch)

Typical workflow
----------------
>>> from preprocessing import load_dataset, build_feature_pipeline
>>> from models import FakeNewsClassifier, VotingEnsemble
>>> from tuning import tune_all_models
>>> from visualise import plot_model_comparison

>>> df      = load_dataset("data/WELFake_Dataset.csv", dataset="welfake")
>>> pipe    = build_feature_pipeline()
>>> X_train = pipe.fit_transform(df_train)
>>> clf     = FakeNewsClassifier.xgboost()
>>> clf.fit(X_train, y_train)
>>> metrics = clf.evaluate(X_test, y_test)
>>> print(metrics)
"""

__version__  = "1.0.0"
__author__   = "Vedant Chauhan"
__license__  = "MIT"
