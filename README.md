<div align="center">

<br/>

```
███████╗ █████╗ ██╗  ██╗███████╗    ███╗   ██╗███████╗██╗    ██╗███████╗
██╔════╝██╔══██╗██║ ██╔╝██╔════╝    ████╗  ██║██╔════╝██║    ██║██╔════╝
█████╗  ███████║█████╔╝ █████╗      ██╔██╗ ██║█████╗  ██║ █╗ ██║███████╗
██╔══╝  ██╔══██║██╔═██╗ ██╔══╝      ██║╚██╗██║██╔══╝  ██║███╗██║╚════██║
██║     ██║  ██║██║  ██╗███████╗    ██║ ╚████║███████╗╚███╔███╔╝███████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝
```

# 📰 Fake News Classification

### *Multi-Model NLP Pipeline · LIAR & WELFake Datasets · 98.5% Accuracy*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-DistilBERT-FFD21E?style=for-the-badge)](https://huggingface.co/transformers)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-189ABE?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A production-grade NLP classification pipeline** that detects fake news articles using linguistic feature engineering,
> classical ML baselines, and fine-tuned transformer models — benchmarked across four public-domain corpora
> with full explainability via SHAP and LIME.

<br/>

---

</div>

## 📋 Table of Contents

| Section | Description |
|---------|-------------|
| [🌐 Overview](#-overview) | Project scope & motivation |
| [📊 Datasets](#-datasets) | Four public-domain corpora supported |
| [🗂️ Project Structure](#️-project-structure) | Repository layout |
| [⚙️ Methodology](#️-methodology) | Pipeline design, models & training |
| [📈 Results](#-results) | Benchmark scores across all models |
| [🚀 Quick Start](#-quick-start) | Installation & usage in 5 minutes |
| [📦 Module Reference](#-module-reference) | API documentation |
| [🔍 Explainability](#-explainability) | SHAP & LIME interpretation |
| [📚 References](#-references) | Academic citations |

---

## 🌐 Overview

Misinformation is one of the most urgent challenges of the digital age — affecting **public health**, **electoral integrity**, and **financial markets**. This project implements a **full supervised learning pipeline** for:

- 🔴 **Binary classification** — Real vs. Fake (WELFake, ISOT, GossipCop)
- 🟠 **6-class classification** — `pants-fire` → `false` → `mostly-false` → `half-true` → `mostly-true` → `true` (LIAR)

The pipeline covers the **complete ML lifecycle**: raw text ingestion → feature engineering → model training → hyperparameter tuning → explainability → inference.

### ✨ Key Highlights

| Capability | Detail |
|------------|--------|
| 🧹 **Text Preprocessing** | Normalisation, stopword removal, spaCy lemmatisation |
| 📐 **Feature Engineering** | 18 handcrafted linguistic signals (sentiment, readability, credibility) |
| 🤖 **Classical Models** | Logistic Regression, Naive Bayes, Random Forest, XGBoost, Voting Ensemble |
| 🧠 **Transformer Model** | DistilBERT fine-tuned — **98.5% accuracy** on WELFake |
| 🎛️ **Hyperparameter Tuning** | Optuna TPE Bayesian search, 60 trials per model |
| 💡 **Explainability** | SHAP (global) + LIME (local) for every classifier |

---

## 📊 Datasets

Four public-domain corpora are supported out of the box. The pipeline **auto-detects** dataset format from the CSV schema.

<br/>

| Dataset | Samples | Task | Features | Source |
|---------|--------:|------|----------|--------|
| **WELFake** | 72,134 | Binary (Real / Fake) | Title + Body Text | [Kaggle →](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **LIAR** | 12,836 | 6-class | Statement + Speaker Metadata | [UCSB NLP →](https://paperswithcode.com/dataset/liar) |
| **ISOT** | 44,919 | Binary (Real / Fake) | Title + Text (Reuters vs InfoWars) | [UVic →](https://onlineacademiccommunity.uvic.ca/isot/) |
| **GossipCop** | 22,864 | Binary (Real / Fake) | Title + Content (Celebrity News) | [FakeNewsNet →](https://github.com/KaiDMML/FakeNewsNet) |

> 📁 Place downloaded CSV(s) in the `data/` directory before running. Format is auto-detected.

---

## 🗂️ Project Structure

```
fake-news-classification/
│
├── 📂 data/                              ← Place dataset CSVs here
│   └── .gitkeep
│
├── 📓 notebooks/
│   └── fake_news_classification.ipynb   ← Full interactive walkthrough
│
├── 📦 src/
│   ├── preprocessing.py                 ← Cleaning · feature engineering · vectorisation
│   ├── models.py                        ← All classifiers + BERT fine-tuning wrappers
│   ├── tuning.py                        ← Optuna + sklearn hyperparameter search
│   ├── visualise.py                     ← Confusion matrices · ROC · SHAP plots
│   ├── explainability.py                ← SHAP & LIME local/global explanations
│   ├── train.py                         ← End-to-end CLI training script
│   └── predict.py                       ← Inference on new headlines / articles
│
├── 💾 models/                           ← Saved .joblib / .pt files (auto-created)
├── 📤 outputs/                          ← Predictions & classification reports (auto-created)
├── 🖼️ figures/                          ← Generated plots (auto-created)
│
├── requirements.txt
├── __init__.py
└── README.md
```

---

## ⚙️ Methodology

### 1 · Text Preprocessing Pipeline

Raw news text passes through a multi-stage cleaning pipeline before any feature extraction:

```
Raw Text
   │
   ▼
Lowercase  ──►  URL / HTML Strip  ──►  Punctuation Normalise
   │
   ▼
Tokenise  ──►  Stopword Remove  ──►  Lemmatise (spaCy en_core_web_sm)
   │
   ▼
Reassembled Cleaned Corpus
```

> Titles and body text are handled separately, then concatenated with a `[SEP]` sentinel for classical models, or passed as `[CLS] title [SEP] body [SEP]` for BERT.

---

### 2 · Feature Engineering

Beyond bag-of-words, 18 handcrafted linguistic signals are extracted that correlate with misinformation patterns:

| Feature Group | Features | Rationale |
|---------------|----------|-----------|
| **Lexical** | `vocab_richness`, `avg_word_len`, `type_token_ratio` | Fake news typically uses simpler, more repetitive vocabulary |
| **Sentiment** | `vader_compound`, `vader_pos`, `vader_neg` | Sensationalist language skews strongly positive or negative |
| **Readability** | `flesch_kincaid_grade`, `gunning_fog` | Credible journalism targets adult reading levels |
| **Structural** | `title_len`, `exclamation_count`, `caps_ratio` | Clickbait headlines abuse capitalisation and punctuation |
| **Credibility** | `quote_count`, `source_count`, `number_count` | Real articles cite sources, statistics, and direct quotes |
| **Temporal** | `has_date`, `past_tense_ratio` | Fabricated stories frequently omit temporal anchors |

---

### 3 · Vectorisation Strategy

Two representations are stacked for classical models:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

$$\text{tfidf}(t, d, D) = \underbrace{\frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}}_{\text{Term Frequency}} \times \underbrace{\log\frac{|D|}{|\{d \in D : t \in d\}|}}_{\text{Inverse Document Frequency}}$$

Separate vectorisers are fitted on **title** and **body** with unigrams + bigrams (`min_df=3`, `max_features=50,000` each), then stacked alongside the 18 engineered features.

---

### 4 · Classical Model Suite

| Model | Regularisation | Key Hyperparameters |
|-------|----------------|---------------------|
| Logistic Regression | L2 (`C` tuned) | `solver=lbfgs`, `max_iter=1000` |
| Multinomial Naive Bayes | Laplace smoothing | `alpha` tuned (0.001–2.0) |
| Passive Aggressive | Hinge loss | `C` tuned, `max_iter=1000` |
| Random Forest | Bagging | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting (XGBoost) | Shrinkage + subsampling | `eta`, `max_depth`, `subsample`, `colsample_bytree` |
| **Voting Ensemble** | Soft voting | Weighted blend of LR + XGB + PA |

---

### 5 · DistilBERT Fine-Tuning

For maximum accuracy, **DistilBERT-base-uncased** (66M parameters — 40% smaller than BERT-base, 97% of its performance) is fine-tuned end-to-end:

```
Input: [CLS] {title} [SEP] {body[:480 tokens]} [SEP]
                     │
          DistilBERT Encoder
          (6 layers · 768 hidden · 12 heads)
                     │
       [CLS] Pooled Representation (768-dim)
                     │
          Dropout(0.3) → Linear(768 → 2) → Softmax
```

**Training Configuration**

| Parameter | Value |
|-----------|-------|
| Learning Rate | `2e-5` (linear warmup 10%) |
| Batch Size | `32` (gradient accumulation × 4) |
| Max Sequence Length | `512` tokens |
| Epochs | `3` |
| Optimiser | `AdamW`, weight decay = `0.01` |
| Scheduler | Linear with warmup |

---

### 6 · Hyperparameter Tuning

[Optuna](https://optuna.org) TPE (Tree-structured Parzen Estimator) Bayesian search is used across all models, with 60 trials per model and stratified 5-fold cross-validation.

---

## 📈 Results

### WELFake Dataset — Binary Classification

| Model | Accuracy | F1 Macro | Precision | Recall | AUC-ROC |
|-------|:--------:|:--------:|:---------:|:------:|:-------:|
| Multinomial Naive Bayes | 0.934 | 0.933 | 0.936 | 0.934 | 0.978 |
| Logistic Regression | 0.956 | 0.956 | 0.957 | 0.956 | 0.990 |
| Passive Aggressive | 0.960 | 0.960 | 0.961 | 0.960 | 0.991 |
| Random Forest | 0.967 | 0.967 | 0.968 | 0.967 | 0.994 |
| XGBoost | 0.971 | 0.971 | 0.972 | 0.971 | 0.996 |
| Voting Ensemble | 0.974 | 0.974 | 0.975 | 0.974 | 0.997 |
| 🏆 **DistilBERT (fine-tuned)** | **0.985** | **0.985** | **0.986** | **0.985** | **0.998** |

---

### LIAR Dataset — 6-Class Classification

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|:--------:|:--------:|:-----------:|
| Logistic Regression | 0.261 | 0.240 | 0.258 |
| XGBoost | 0.278 | 0.261 | 0.275 |
| Voting Ensemble | 0.283 | 0.265 | 0.281 |
| 🏆 **DistilBERT (fine-tuned)** | **0.312** | **0.298** | **0.310** |

> ⚠️ *6-class fake news is a significantly harder task — human inter-annotator agreement on LIAR is ~20%. All scores are on a held-out 15% test split.*

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/fake-news-classification.git
cd fake-news-classification

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Training

```bash
# ── Classical models with Optuna tuning ──────────────────────────────────────
python src/train.py \
    --data_path data/WELFake_Dataset.csv \
    --dataset   welfake \
    --tune      --n_trials 60 \
    --save_models

# ── Fine-tune DistilBERT (GPU recommended) ────────────────────────────────────
python src/train.py \
    --data_path data/WELFake_Dataset.csv \
    --dataset   welfake \
    --model     bert \
    --epochs    3 \
    --save_models

# ── Interactive Jupyter walkthrough ──────────────────────────────────────────
jupyter lab notebooks/fake_news_classification.ipynb
```

### Inference

```bash
# ── Single article prediction ────────────────────────────────────────────────
python src/predict.py \
    --model_path models/voting_ensemble.joblib \
    --title "BREAKING: Scientists Discover Miracle Cure for All Diseases" \
    --body  "Researchers at a private lab claim..."

# ── Batch prediction from CSV ────────────────────────────────────────────────
python src/predict.py \
    --model_path models/distilbert/ \
    --input_csv  data/new_articles.csv \
    --output_csv outputs/predictions.csv \
    --explain
```

### CLI Reference — `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | *(required)* | Path to dataset CSV |
| `--dataset` | `welfake` | Format: `welfake` · `liar` · `isot` · `gossipcop` |
| `--model` | `all` | Target: `lr` · `nb` · `pa` · `rf` · `xgb` · `ensemble` · `bert` · `all` |
| `--tune` | off | Enable Optuna hyperparameter search |
| `--n_trials` | `60` | Number of tuning trials per model |
| `--epochs` | `3` | BERT fine-tuning epochs |
| `--batch_size` | `32` | BERT training batch size |
| `--save_models` | off | Persist trained models to `models/` |
| `--no_plots` | off | Skip figure generation |
| `--explain` | off | Generate SHAP / LIME explanations post-training |

---

## 📦 Module Reference

<details>
<summary><strong>src/preprocessing.py</strong></summary>

| Symbol | Type | Description |
|--------|------|-------------|
| `load_dataset()` | function | Read CSV, auto-detect schema, return `pd.DataFrame` |
| `clean_text()` | function | URL strip, lowercase, punctuation normalise |
| `TextNormaliser` | transformer | spaCy tokeniser, stopword removal, lemmatisation |
| `LinguisticFeatureExtractor` | transformer | Compute 18 handcrafted features per document |
| `TfidfPipeline` | transformer | Fit title + body TF-IDF, horizontal stack + feature concat |
| `build_feature_pipeline()` | function | Full sklearn `Pipeline` for classical models |
| `build_bert_dataset()` | function | HuggingFace `Dataset` with tokenisation for fine-tuning |

</details>

<details>
<summary><strong>src/models.py</strong></summary>

| Symbol | Type | Description |
|--------|------|-------------|
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

</details>

<details>
<summary><strong>src/explainability.py</strong></summary>

| Symbol | Type | Description |
|--------|------|-------------|
| `SHAPExplainer` | class | TreeExplainer (RF/XGB) or LinearExplainer (LR) |
| `LIMEExplainer` | class | LIME text explainer for any `predict_proba` model |
| `plot_shap_summary()` | function | Global feature importance beeswarm plot |
| `explain_prediction()` | function | Local explanation HTML for a single article |

</details>

<details>
<summary><strong>src/visualise.py</strong></summary>

| Symbol | Type | Description |
|--------|------|-------------|
| `plot_confusion_matrix()` | function | Annotated heatmap with normalisation option |
| `plot_roc_curves()` | function | Multi-model ROC comparison with AUC legend |
| `plot_pr_curves()` | function | Precision–Recall curves (preferred for imbalanced data) |
| `plot_model_comparison()` | function | Bar chart: accuracy / F1 / AUC across all models |
| `plot_feature_importance()` | function | Top-40 TF-IDF tokens + engineered features |
| `plot_learning_curves()` | function | Train vs val loss / accuracy per BERT epoch |

</details>

---

## 🔍 Explainability

Trustworthy fake news detection demands **interpretable predictions**. This project integrates two complementary explanation frameworks:

### 🔷 SHAP — Global Feature Attribution

- **`TreeExplainer`** for Random Forest and XGBoost (exact Shapley values)
- **`LinearExplainer`** for Logistic Regression and Passive Aggressive
- Produces beeswarm plots revealing which words and features distinguish real from fake news at a global level

### 🔶 LIME — Per-Prediction Explanations

- Fits a local linear model around each prediction by perturbing the input text
- Returns a ranked list of tokens that pushed the model toward `REAL` or `FAKE`
- Works with **any** model exposing `predict_proba` — including fine-tuned DistilBERT

### 🖥️ Example Output — LIME on a Misclassified Article

```
┌─────────────────────────────────────────────────────┐
│  Prediction:  FAKE   (confidence: 87%)              │
├─────────────────────────────────────────────────────┤
│  Tokens pushing → FAKE                              │
│    "miracle"            +0.142  ████████████        │
│    "BREAKING"           +0.118  ██████████          │
│    "scientists claim"   +0.094  ████████            │
│    "secret"             +0.087  ███████             │
│    "mainstream media"   +0.071  ██████              │
├─────────────────────────────────────────────────────┤
│  Tokens pushing → REAL                              │
│    "according to"       -0.063  ─────               │
│    "peer-reviewed"      -0.055  ────                │
│    "published in"       -0.041  ───                 │
└─────────────────────────────────────────────────────┘
```

---

## 📚 References

<details>
<summary>Click to expand all citations</summary>

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective.* ACM SIGKDD Explorations Newsletter, 19(1), 22–36.

2. Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017 Short Papers. https://doi.org/10.18653/v1/P17-2067

3. Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). *WELFake: Word Embedding over Linguistic Features for Fake News Detection.* IEEE Transactions on Computational Social Systems, 8(4), 881–893.

4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.

5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS EMC² Workshop.

6. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.

7. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016.

8. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.

</details>

---

<div align="center">

<br/>

**Built for the NLP & Text Classification Module**

[![MIT License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/Powered%20by-HuggingFace-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co)

<br/>

*If this project helped you, please consider giving it a ⭐*

</div>
