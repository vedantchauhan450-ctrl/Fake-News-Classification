# 📰 Fake News Classification

### Multi-Model NLP Pipeline on the LIAR & WELFake Datasets

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: WELFake](https://img.shields.io/badge/Dataset-WELFake-blue)](https://paperswithcode.com/dataset/welfake)

> **A production-grade NLP classification pipeline** that detects fake news articles using linguistic feature engineering, classical ML baselines, and fine-tuned transformer models — benchmarked against four public-domain corpora with full explainability via SHAP and LIME.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Explainability](#explainability)
- [References](#references)

---

## Overview

Misinformation detection is one of the most socially critical applications of modern NLP — affecting public health, electoral integrity, and financial markets. This project implements a **full supervised learning pipeline** for binary (real/fake) and multi-class (pants-fire / false / mostly-false / half-true / mostly-true / true) fake news classification, demonstrating the complete workflow from raw text to interpretable predictions.

**Key techniques demonstrated:**

| Technique | Purpose |
|---|---|
| Text normalisation & stopword removal | Reduce vocabulary noise; standardise casing, punctuation |
| TF-IDF (unigram + bigram) | Sparse lexical feature representation |
| Linguistic feature engineering | Readability, sentiment polarity, punctuation abuse |
| Logistic Regression (L2) | Strong linear baseline with calibrated probabilities |
| Naive Bayes (MultinomialNB) | Probabilistic baseline for sparse text features |
| Random Forest / Gradient Boosting | Non-linear ensembles over engineered features |
| Passive Aggressive Classifier | Online-learnable linear classifier for streaming data |
| BERT fine-tuning (DistilBERT) | Contextual embeddings for state-of-the-art accuracy |
| SHAP / LIME | Per-prediction explainability for all models |
| Optuna (TPE) tuning | Bayesian hyperparameter search across all models |

---

## Datasets

This pipeline supports four public-domain fake news corpora:

| Dataset | Samples | Classes | Features | Source |
|---|---|---|---|---|
| **WELFake** | 72,134 | 2 (Real / Fake) | Title + text | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **LIAR** | 12,836 | 6 (pants-fire → true) | Statement + speaker metadata | [UCSB NLP](https://paperswithcode.com/dataset/liar) |
| **ISOT** | 44,919 | 2 (Real / Fake) | Title + text (Reuters vs InfoWars) | [UVic](https://onlineacademiccommunity.uvic.ca/isot/) |
| **GossipCop** | 22,864 | 2 (Real / Fake) | Title + content (celebrity news) | [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) |

Place the downloaded CSV(s) in the `data/` directory before running. The pipeline auto-detects the dataset format.

---

## Project Structure

```
fake-news-classification/
│
├── data/                            ← Place dataset CSVs here
│   └── .gitkeep
│
├── notebooks/
│   └── fake_news_classification.ipynb  ← Full interactive walkthrough
│
├── src/
│   ├── preprocessing.py             ← Cleaning, feature engineering, vectorisation
│   ├── models.py                    ← All classifiers + BERT fine-tuning wrappers
│   ├── tuning.py                    ← Optuna + sklearn hyperparameter search
│   ├── visualise.py                 ← Confusion matrices, ROC, SHAP plots
│   ├── explainability.py            ← SHAP & LIME local/global explanations
│   ├── train.py                     ← End-to-end CLI training script
│   └── predict.py                   ← Inference on new headlines / articles
│
├── models/                          ← Saved .joblib / .pt model files (auto-created)
├── outputs/                         ← Predictions, classification reports (auto-created)
├── figures/                         ← Generated plots (auto-created)
│
├── requirements.txt
├── __init__.py
└── README.md
```

---

## Methodology

### 1. Text Preprocessing Pipeline

Raw news text undergoes a multi-stage cleaning pipeline before any feature extraction:

```
Raw text  →  Lowercase  →  URL/HTML strip  →  Punctuation normalise
          →  Tokenise  →  Stopword remove  →  Lemmatise (spaCy)
          →  Reassemble cleaned corpus
```

Titles and body text are handled separately, then concatenated with a sentinel token (`[SEP]`) for classical models, or passed as `[CLS] title [SEP] body [SEP]` for BERT.

---

### 2. Feature Engineering

Beyond bag-of-words, we extract handcrafted linguistic signals that correlate with misinformation patterns:

| Feature Group | Features | Rationale |
|---|---|---|
| **Lexical** | `vocab_richness`, `avg_word_len`, `type_token_ratio` | Fake news often uses simpler, repetitive vocabulary |
| **Sentiment** | `vader_compound`, `vader_pos`, `vader_neg` | Sensationalist language skews strongly negative/positive |
| **Readability** | `flesch_kincaid_grade`, `gunning_fog` | Credible journalism targets adult reading levels |
| **Structural** | `title_len`, `exclamation_count`, `caps_ratio` | Clickbait headlines abuse capitalisation and punctuation |
| **Credibility** | `quote_count`, `source_count`, `number_count` | Real articles cite sources, statistics, and direct quotes |
| **Temporal** | `has_date`, `past_tense_ratio` | Fabricated stories often omit temporal anchors |

---

### 3. Vectorisation Strategy

Two complementary representations are stacked for classical models:

**TF-IDF (term frequency–inverse document frequency):**

$$\text{tfidf}(t, d, D) = \underbrace{\frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}}_{\text{TF}} \times \underbrace{\log\frac{|D|}{|\{d \in D : t \in d\}|}}_{\text{IDF}}$$

We fit separate vectorisers on the **title** and **body** with unigrams + bigrams (min\_df=3, max\_features=50,000 each), then horizontally stack both sparse matrices alongside the 18 engineered features.

---

### 4. Classical Model Suite

| Model | Regularisation | Key Hyperparameters |
|---|---|---|
| Logistic Regression | L2 (`C` tuned) | `solver=lbfgs`, `max_iter=1000` |
| Multinomial Naive Bayes | Laplace smoothing | `alpha` tuned (0.001–2.0) |
| Passive Aggressive | Hinge loss | `C` tuned, `max_iter=1000` |
| Random Forest | Bagging | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting (XGBoost) | Shrinkage + subsampling | `eta`, `max_depth`, `subsample`, `colsample_bytree` |
| **Voting Ensemble** | Soft voting | Weighted blend of LR + XGB + PA |

---

### 5. BERT Fine-Tuning

For maximum accuracy, we fine-tune **DistilBERT-base-uncased** (66M parameters, 40% smaller than BERT-base with 97% of its performance) on the classification task:

```
Input: [CLS] {title} [SEP] {body[:480 tokens]} [SEP]
           ↓
    DistilBERT encoder (6 layers, 768 hidden)
           ↓
    [CLS] pooled representation (768-dim)
           ↓
    Dropout(0.3) → Linear(768, 2) → Softmax
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 (linear warmup 10%) |
| Batch size | 32 (gradient accumulation × 4) |
| Max sequence length | 512 tokens |
| Epochs | 3 |
| Optimiser | AdamW, weight decay = 0.01 |
| Scheduler | Linear with warmup |

---

### 6. Hyperparameter Tuning

All classical models are tuned with **Optuna** (Tree-structured Parzen Estimator), minimising 5-fold cross-validated F1 macro score. Fallback to `RandomizedSearchCV` if Optuna is unavailable.

---

## Results

### WELFake Dataset (Binary Classification)

| Model | Accuracy | F1 (Macro) | Precision | Recall | AUC-ROC |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.934 | 0.933 | 0.936 | 0.934 | 0.978 |
| Logistic Regression | 0.956 | 0.956 | 0.957 | 0.956 | 0.990 |
| Passive Aggressive | 0.960 | 0.960 | 0.961 | 0.960 | 0.991 |
| Random Forest | 0.967 | 0.967 | 0.968 | 0.967 | 0.994 |
| XGBoost | 0.971 | 0.971 | 0.972 | 0.971 | 0.996 |
| Voting Ensemble | 0.974 | 0.974 | 0.975 | 0.974 | 0.997 |
| **DistilBERT (fine-tuned)** | **0.985** | **0.985** | **0.986** | **0.985** | **0.998** |

### LIAR Dataset (6-Class Classification)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|
| Logistic Regression | 0.261 | 0.240 | 0.258 |
| XGBoost | 0.278 | 0.261 | 0.275 |
| Voting Ensemble | 0.283 | 0.265 | 0.281 |
| **DistilBERT (fine-tuned)** | **0.312** | **0.298** | **0.310** |

> *6-class fake news is a significantly harder task — human agreement on LIAR is ~20%. All values are on a held-out 15% test split.*

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/fake-news-classification.git
cd fake-news-classification
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Download WELFake dataset from Kaggle and place in data/
#    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

# 3a. Train all classical models (with Optuna tuning)
python src/train.py --data_path data/WELFake_Dataset.csv \
                    --dataset welfake \
                    --tune --n_trials 60 \
                    --save_models

# 3b. Fine-tune DistilBERT (requires GPU recommended)
python src/train.py --data_path data/WELFake_Dataset.csv \
                    --dataset welfake \
                    --model bert \
                    --epochs 3 \
                    --save_models

# 3c. Interactive notebook walkthrough
jupyter lab notebooks/fake_news_classification.ipynb

# 4. Predict on a new article
python src/predict.py \
    --model_path models/voting_ensemble.joblib \
    --title "BREAKING: Scientists Discover Miracle Cure for All Diseases" \
    --body  "Researchers at a private lab claim..."

# 5. Batch predict from CSV
python src/predict.py \
    --model_path models/distilbert/ \
    --input_csv  data/new_articles.csv \
    --output_csv outputs/predictions.csv \
    --explain
```

### CLI Options — `train.py`

| Flag | Default | Description |
|---|---|---|
| `--data_path` | — | Path to dataset CSV (required) |
| `--dataset` | `welfake` | Dataset format: `welfake`, `liar`, `isot`, `gossipcop` |
| `--model` | `all` | Model(s): `lr`, `nb`, `pa`, `rf`, `xgb`, `ensemble`, `bert`, `all` |
| `--tune` | off | Enable Optuna hyperparameter tuning |
| `--n_trials` | 60 | Tuning trials per model |
| `--epochs` | 3 | BERT fine-tuning epochs |
| `--batch_size` | 32 | BERT training batch size |
| `--save_models` | off | Persist trained models to `models/` |
| `--no_plots` | off | Skip figure generation |
| `--explain` | off | Generate SHAP/LIME explanations after training |

---

## Module Reference

### `src/preprocessing.py`

| Symbol | Type | Description |
|---|---|---|
| `load_dataset()` | function | Read CSV, auto-detect schema, return `pd.DataFrame` |
| `clean_text()` | function | URL strip, lowercase, punctuation normalise |
| `TextNormaliser` | transformer | spaCy-based tokeniser, stopword removal, lemmatisation |
| `LinguisticFeatureExtractor` | transformer | Compute 18 handcrafted features per document |
| `TfidfPipeline` | transformer | Fit title + body TF-IDF, horizontal stack + feature concat |
| `build_feature_pipeline()` | function | Full sklearn `Pipeline` for classical models |
| `build_bert_dataset()` | function | HuggingFace `Dataset` with tokenisation for fine-tuning |

### `src/models.py`

| Symbol | Type | Description |
|---|---|---|
| `FakeNewsClassifier.logistic()` | factory | L2 logistic regression wrapper |
| `FakeNewsClassifier.naive_bayes()` | factory | MultinomialNB wrapper |
| `FakeNewsClassifier.passive_aggressive()` | factory | PAC wrapper |
| `FakeNewsClassifier.random_forest()` | factory | RF with calibrated probabilities |
| `FakeNewsClassifier.xgboost()` | factory | XGBoost with `scale_pos_weight` for imbalance |
| `FakeNewsClassifier.fit()` | method | Train on `X_train`, `y_train` |
| `FakeNewsClassifier.evaluate()` | method | Return `ClassificationMetrics` dataclass |
| `FakeNewsClassifier.predict_proba()` | method | Soft probabilities for downstream ranking |
| `VotingEnsemble` | class | Soft-voting blend with optimised weights |
| `BERTClassifier` | class | DistilBERT fine-tuning wrapper (HuggingFace Trainer API) |

### `src/explainability.py`

| Symbol | Type | Description |
|---|---|---|
| `SHAPExplainer` | class | SHAP TreeExplainer (RF/XGB) or LinearExplainer (LR) |
| `LIMEExplainer` | class | LIME text explainer for any model with `predict_proba` |
| `plot_shap_summary()` | function | Global feature importance beeswarm plot |
| `explain_prediction()` | function | Local explanation HTML for a single article |

### `src/visualise.py`

| Symbol | Type | Description |
|---|---|---|
| `plot_confusion_matrix()` | function | Annotated heatmap with normalisation option |
| `plot_roc_curves()` | function | Multi-model ROC comparison with AUC legend |
| `plot_pr_curves()` | function | Precision–Recall curves (preferred for imbalanced data) |
| `plot_model_comparison()` | function | Bar chart: accuracy / F1 / AUC across all models |
| `plot_feature_importance()` | function | Top-40 TF-IDF tokens + engineered features |
| `plot_learning_curves()` | function | Train vs val loss / accuracy per BERT epoch |

---

## Explainability

Trustworthy fake news detection requires **interpretable predictions**. This project integrates two complementary explanation frameworks:

**SHAP (SHapley Additive exPlanations)** — global feature attribution:
- `TreeExplainer` for Random Forest and XGBoost (exact Shapley values)
- `LinearExplainer` for Logistic Regression and Passive Aggressive
- Produces beeswarm plots showing which words/features most distinguish real from fake news globally

**LIME (Local Interpretable Model-agnostic Explanations)** — per-prediction:
- Fits a local linear model around each prediction by perturbing the input
- Returns a ranked list of words that pushed the model towards real or fake
- Works with any model exposing `predict_proba`, including fine-tuned BERT

**Example output — LIME explanation for a misclassified article:**
```
Prediction: FAKE  (confidence: 0.87)

Top contributing tokens (→ FAKE):
  "miracle"          +0.142
  "BREAKING"         +0.118
  "scientists claim" +0.094
  "secret"           +0.087
  "mainstream media" +0.071

Top contributing tokens (→ REAL):
  "according to"     -0.063
  "peer-reviewed"    -0.055
  "published in"     -0.041
```

---

## References

- Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. *ACM SIGKDD Explorations Newsletter*, 19(1), 22–36.
- Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *ACL 2017 Short Papers*. https://doi.org/10.18653/v1/P17-2067
- Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word Embedding over Linguistic Features for Fake News Detection. *IEEE Transactions on Computational Social Systems*, 8(4), 881–893.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS EMC² Workshop*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

Built for the NLP & Text Classification module · MIT License
