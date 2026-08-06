# Predicting Decisions of the EPO's Boards of Appeal with Machine Learning

A reproducible pipeline for predicting the outcome of patent appeals at the EPO's Technical Boards of Appeal. An earlier version of this work won **Best Doctoral Consortium Paper** at JURIX 2023 — see the [paper](https://davidbareham.co.uk/assets/pdf/pred_EPO.pdf) and its [supplementary material](https://davidbareham.co.uk/assets/pdf/jurix_23_supp.pdf).

## Overview

The task is binary classification of appeal outcomes (grant / refuse) for two case types: *ex parte* (examining division, `pf`) and *inter partes* (opposition, `op`), plus a combined mode (`both`). Two evaluation designs are used:

- **Exp 1** — stratified random split.
- **Exp 2** — temporal split (train on earlier decisions, test on recent), including a sliding-window analysis that tests on each year 2021–2024 in turn.

Models span classical ML (Logistic Regression, LinearSVC, Random Forest, XGBoost over N-Grams / TF-IDF and dense embeddings) and transformers (LegalBERT, PatentBERT, Longformer).

## Setup

### 1. Environment

Requires Python ≥ 3.11. DL experiments need a CUDA GPU (tested on A100, CUDA 12.8).

```bash
git clone https://github.com/dahrb/EPO-Project.git
cd EPO-Project

pip install uv && uv venv .venv && source .venv/bin/activate && uv sync
# or: python -m venv .venv && source .venv/bin/activate && pip install -e .
```

### 2. Raw EPO data

The source code and all `Results/results_*.json` are committed. To re-derive the tables and figures from the published results, nothing further is needed. To re-run the pipeline, download the raw decisions and embeddings below.

Download the **"Decisions of the Boards of Appeal"** product (free, no login) and unzip the XML into `Data/`:

```bash
# https://publication-bdds.apps.epo.org/raw-data/products/public/product/21
unzip EPDecisions_March2026.zip -d Data/
```

`Data/data_processing.py` auto-detects `Data/EPDecisions_*.xml` (no renaming needed) and caps processing at decisions from **2000–2024**. Any snapshot from March 2025 onward therefore reproduces the **identical** dataset behind all results.

### 3. Word embeddings → `Models/`

| Embedding | How to obtain |
|---|---|
| **Patent2Vec + Doc2Vec** (custom) | Download `embeddings.zip` from [GitHub Releases](https://github.com/dahrb/EPO-Project/releases) and `unzip embeddings.zip -d Models/`. Alternatively re-train via Step 2. |
| **Word2Vec (Google News, 300d)** | `python -c "import gensim.downloader as api; api.load('word2vec-google-news-300').save('Models/Word2Vec-google-300d')"` |
| **Law2Vec (200d)** | Download `Law2Vec.200d.txt` from the [Internet Archive](https://archive.org/details/Law2Vec) into `Models/`. |

Transformer weights (LegalBERT, PatentBERT, Longformer) download automatically via `transformers` on first use.

## Reproducing the pipeline

Run from the project root. Each step is a plain `python -m` module — no environment-specific wrappers are required.

**1. Process the raw XML** → cleaned, feature-engineered CSVs in `Data/Matching/`:

```bash
python -m Data.data_processing
```

**2. Train custom embeddings** (optional — skip if you downloaded `embeddings.zip`):

> **Note:** The corpus used to train the original Patent2Vec and Doc2Vec embeddings is not distributed with this repository. `PatentEmbeddings.py` can only train *new* embeddings on data you supply. To use the exact embeddings from the published results, download `embeddings.zip` from [GitHub Releases](https://github.com/dahrb/EPO-Project/releases) instead.

```bash
python -m Experiments.PatentEmbeddings          # → Models/Patent2Vec_1.0, Models/Doc2Vec_1.0
```

**3. Generate train/test splits** → `Data/Final_Processed/*.pkl`:

```bash
python -m Data.experiment_processing
```

**4. Classical ML** — 4 models × {N-Grams, TF-IDF} × 4 embedding inputs × 2 experiments × 3 cases (`RandomizedSearchCV`, 50 iters):

```bash
python -m Experiments.run_experiment \
    xgboost 1 false TF-IDF \
    Data/Final_Processed/X_Train_1_pf.pkl Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl  Data/Final_Processed/y_test_1_pf.pkl \
    false                                       # → Results/results_main.json
```

**5. DL hyperparameter tuning** — TPE Optuna study per model × exp × case (10 trials for BERT models, 3 for Longformer). Studies persist to SQLite and resume if interrupted; the best checkpoint is saved to `Results/optuna/best_{model}_{case}_exp{N}.pt`:

```bash
python -m Experiments.run_deep_learning_experiment \
    legalbert 1 false \
    Data/Final_Processed/X_Train_1_pf.pkl Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl  Data/Final_Processed/y_test_1_pf.pkl \
    --n_trials 10 \
    --optuna_storage "sqlite:///Results/optuna/legalbert_exp1_pf.db" \
    --study_name legalbert_exp1_pf \
    --results_path Results/results_deep_learning.json
```

Set the `opposition` positional arg to `true` for `op`/`both`, `false` for `pf`. CLI ids: `legalbert`, `patentbert`, `longformer_base` (use `--n_trials 3`).

**6. DL full-test evaluation** — loads the best checkpoint from Step 5 and scores the held-out test set (no retraining):

```bash
python -m Experiments.evaluation --skip_ml --skip_sliding_window \
    --experiments 1 --only_algos legalbert --only_cases pf \
    --dl_results Results/results_deep_learning.json \
    --output Results/results_dl_full_test.json
```

**7. Sliding-window temporal evaluation** (Exp 2 only) — train on ≤ T−1, test on year T for T ∈ 2021–2024. The runner is **resumable**: it skips any `(model, case, year)` already present, so re-runs never duplicate records.

```bash
# ML
python -m Experiments.evaluation --skip_dl --skip_full_test \
    --experiments 2 --only_cases pf \
    --output Results/results_ml_sliding_window.json

# DL
python -m Experiments.evaluation --skip_ml --skip_full_test \
    --experiments 2 --only_algos legalbert --only_cases pf \
    --dl_results Results/results_deep_learning.json \
    --output Results/results_dl_sliding_window.json
```

## Results

Full tables and methodology are in `Results/results.ipynb`

**ML full-test — best model per exp × case**

| Exp | Case | Model | Input | Test F1 | MCC |
|-----|------|-------|-------|---------|-----|
| 1 | pf | XGBoost | TF-IDF | **0.903** | 0.689 |
| 1 | op | XGBoost | N-Grams | 0.673 | 0.545 |
| 1 | both | XGBoost | N-Grams | **0.807** | 0.653 |
| 2 | pf | Logistic Regression | TF-IDF | **0.890** | 0.688 |
| 2 | op | XGBoost | N-Grams | 0.745 | 0.581 |
| 2 | both | XGBoost | N-Grams | 0.798 | 0.622 |

**DL full-test — best case per model (retrained=False)**

| Model | Exp 2 pf | Exp 2 op | Exp 2 both |
|-------|----|----|------|
| LegalBERT | 0.887 | 0.711 | 0.776 |
| PatentBERT | 0.882 | 0.684 | 0.779 |
| Longformer | **0.899** | 0.684 | 0.782 |

**Sliding window (Exp 2, mean F1 over 2021–2024)**

| Model | pf | op | both |
|-------|----|----|------|
| LegalBERT | 0.899 | 0.707 | 0.815 |
| PatentBERT | 0.891 | 0.706 | 0.799 |
| Longformer | **0.927** | 0.722 | **0.826** |
| ML best (XGBoost) | 0.907 | 0.749 | 0.808 |

## Explainability

Two SHAP pipelines write figures under `Figs/`:

```bash
# XGBoost (TreeExplainer) — cross-mode importance + sliding-window trajectory
python -m Experiments.shap_analysis all --top_n 20 --top_k 8      # → Figs/shap/

# BERT (PartitionExplainer) — word-level attributions per model × exp × case
python Experiments/shap_bert_run.py --model legalbert --experiment 2 --case pf --n_samples 75   # → Figs/shap_bert/
```

## Notebooks

- `Experiments/patent2vec.ipynb` — explore trained embeddings, t-SNE projections.
- `Experiments/visualisations.ipynb` — dataset composition figures.
- `Results/results.ipynb` — load `results_*.json`, render summary tables and plots.

## Key parameters

| | Value |
|---|---|
| ML CV search | `RandomizedSearchCV`, 50 iters; macro-F1 scoring |
| ML CV strategy | `RepeatedStratifiedKFold` (Exp 1) / `TimeSeriesSplit` (Exp 2), 3 splits |
| Optuna | TPE sampler, seed 42; 10 trials (BERT) / 3 (Longformer) |
| DL training | max 30 epochs, step-level early stopping (patience 8) |
| Max sequence length | 512 (BERT) / 4096 (Longformer) |
| Random seed | 42 |
