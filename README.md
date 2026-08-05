# Predicting Decisions of the EPO's Boards of Appeal Using Machine Learning

This repo shows a pipeline implemented to help predict the outcome of patent application appeals at the EPO's Technical Boards of Appeals. An earlier version of these experiments can be found in --- which was submitted to the JURIX 2023 'Doctoral Consortium' and was awarded 'Best Doctoral Consortium Paper'.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Reproducing Experiments](#reproducing-experiments)
  - [Step 1: Data Processing](#step-1-data-processing)
  - [Step 2: Train Patent Embeddings](#step-2-train-patent-embeddings)
  - [Step 3: Generate Train/Test Splits](#step-3-generate-traintest-splits)
  - [Step 4: Classical ML Experiments](#step-4-classical-ml-experiments)
  - [Step 5: DL Hyperparameter Tuning (Optuna)](#step-5-dl-hyperparameter-tuning-optuna)
  - [Step 6: DL Full-Test Evaluation](#step-6-dl-full-test-evaluation)
  - [Step 7: Sliding-Window Temporal Evaluation](#step-7-sliding-window-temporal-evaluation)
- [Results Summary](#results-summary)
- [Configuration & Key Parameters](#configuration--key-parameters)

---

## Project Overview

This project classifies the outcome of EPO patent appeal decisions — grant/refuse for *ex parte* (examining division) cases and for *inter partes* (opposition division) cases — across two experimental splits:

| Split | Description |
|-------|-------------|
| **Exp 1** | Stratified random split |
| **Exp 2** | Temporal split — earlier decisions for training, recent for test. This includes a Sliding Window temporal evaluation to measure how the performance changes over time.|

Each split is evaluated in three **modes**: `pf` (ex parte), `op` (opposition), `both` (combined).

The full pipeline is:
1. Parse raw XML → processed CSV
2. Train domain-specific embeddings (Patent2Vec, PatentDoc2Vec)
3. Generate train/test files per experiment × mode
4. ML experiments and hyperparameter tuning
5. DL hyperparameter tuning via Optuna
6. DL full-test evaluation
7. Sliding-window temporal evaluation 

---

## Project Structure

```
EPO-Project/
├── Data/
│   ├── EPDecisions_March2025.xml            # Raw EPO XML data
│   ├── Final_Processed/                     # Train/test splits (generated, *.pkl)
│   └── Matching/                            # Intermediate processed CSVs
├── Experiments/
│   ├── data_processing.py                   # Step 1: XML → processed data
│   ├── experiment_processing.py             # Step 3: generate train/test splits
│   ├── PatentEmbeddings.py                  # Step 2: Patent2Vec / Doc2Vec training
│   ├── ml_experiments.py                    # Classical ML model logic
│   ├── run_experiment.py                    # CLI runner for ML experiments
│   ├── deep_learning_experiments.py         # Optuna HPO loop for all DL models
│   ├── run_deep_learning_experiment.py      # CLI runner for DL HP tuning
│   ├── evaluation.py                        # CLI runner for full-test & SW eval
│   └── shap_analysis.py                     # SHAP explainability analysis (XGBoost)
├── Models/
│   ├── Patent2Vec_1.0                       # Word2Vec trained on patent text
│   ├── Doc2Vec_1.0                          # Doc2Vec trained on patent documents
│   ├── Word2Vec-google-300d                 # Google News Word2Vec (pre-trained)
│   └── Law2Vec.200d.txt                     # Law2Vec embeddings (pre-trained)
├── Results/
│   ├── results_main.json                    # ML full-test results (144 records)
│   ├── results_ml_sliding_window.json       # ML sliding-window results (288 records)
│   ├── results_deep_learning.json           # DL HP tuning best-trial records
│   ├── results_dl_full_test.json            # DL full-test evaluation results
│   ├── results_dl_sliding_window.json       # DL sliding-window evaluation results
│   ├── optuna/
│   │   ├── *.db                             # Persistent Optuna SQLite studies
│   │   └── best_{model}_{case}_exp{N}.pt    # Best HP checkpoint per model/case/exp
│   ├── grid_logs/                           # SLURM stdout logs
│   ├── experiment_report_skeleton.md        # Detailed results report
│   └── experiment_report.html              # Rendered HTML version of the report
├── run_data_processing.sh
├── run_hp_{legalbert,patentbert,longformer}_exp{1,2}_{pf,op,both}.sh   # HP tuning
├── run_eval_{lb,pb,lf}_ft_exp{1,2}_{pf,op,both}.sh                     # Full-test
├── run_eval_{lb,pb,lf}_sw_exp2_{pf,op,both}.sh                         # SW eval
├── run_eval_ml_sw_{pf,op,both}_balanced.sh                             # ML SW eval
└── pyproject.toml
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dahrb/EPO-Project.git
cd EPO-Project
```

### 2. Create Environment

Requires Python ≥ 3.11. A CUDA-capable GPU is required for DL experiments (tested on A100 with CUDA 12.8).

```bash
# Recommended — uv
pip install uv
uv venv .venv
source .venv/bin/activate
uv sync                       # installs from pyproject.toml lock

# Alternative — pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Key dependencies (see `pyproject.toml`): `torch`, `transformers`, `optuna`, `scikit-learn`, `xgboost`, `gensim`, `spacy[cuda12x]`, `pandas`.

---

## Reproducing Experiments

All steps assume you are in the project root.

---

### Step 1: Data Processing

Parses `Data/EPDecisions_March2025.xml` and produces cleaned, feature-engineered CSV files.

```bash
# Locally:
python -m Experiments.data_processing
```

**8-step pipeline:** XML parse → decision-type filter → legal-provision extraction → outcome classification → technical-field sub-classification → feature engineering → `pf`/`op` dataset split → save to `Data/Matching/`.

---

### Step 2: Train Patent Embeddings

Trains domain-specific Word2Vec and Doc2Vec models on the processed patent text. Data not included to reproduce the original embeddings trained for this work. The Patent2Vec and PatentDoc2Vec embeddings themselves are included, so it is recommend to use those for reproducibility. Feel free to use this script to train your own patent embeddings using similar data.

```bash
python -m Experiments.PatentEmbeddings
```

Output saved to `Models/Patent2Vec_1.0` and `Models/Doc2Vec_1.0`. Pre-trained `Word2Vec-google-300d` and `Law2Vec.200d.txt` are already provided.

---

### Step 3: Generate Train/Test Splits

Produces one pickle file per dataset component per experiment × mode combination.

```bash
python -m Experiments.experiment_processing
```

**Output** (`Data/Final_Processed/`):
```
X_Train_{1,2}_{pf,op,both}.pkl    y_Train_{1,2}_{pf,op,both}.pkl
X_test_{1,2}_{pf,op,both}.pkl     y_test_{1,2}_{pf,op,both}.pkl
```

---

### Step 4: Classical ML Experiments

Runs Logistic Regression, LinearSVC, Random Forest, and XGBoost with sparse (N-Gram, TF-IDF) and dense embedding inputs across all experiment × mode combinations.

Hyperparameter search uses **`RandomizedSearchCV`** (50 iterations). Cross-validation strategy follows the experiment type: `RepeatedStratifiedKFold` (3 splits) for Exp 1, `TimeSeriesSplit` (3 splits) for Exp 2. All models share a text pre-processing search space (`stopwords`, `numbers`, `lemmatisation` ∈ {True, False}) and vectoriser tuning (`ngram_range`, `norm`, `min_df`, `use_idf`).

#### Model-specific hyperparameters

| Model | Parameter | Values |
|-------|-----------|--------|
| **LinearSVC** | `C` | {0.1, 1, 10, 100} |
| **Logistic Regression** | `C` | {0.1, 1, 10, 100} |
| | `solver` | {lbfgs, sag} |
| | `penalty` | {None, l2} |
| | `max_iter` | {100, 250, 500} |
| **Random Forest** | `n_estimators` | {100, 200, 300} |
| | `max_features` | {sqrt, log2} |
| | `max_depth` | {10, 50, 100, None} |
| **XGBoost** | `n_estimators` | {100, 200, 300} |
| | `learning_rate` | {0.01, 0.02, 0.05, 0.1, 0.2} |
| | `gamma` | {0.0, 0.1, 0.2} |
| | `max_depth` | {3, 6, 9} |

```bash
# Single run example (xgboost, exp 1, pf, TF-IDF, no embeddings):
python -m Experiments.run_experiment \
    xgboost 1 false TF-IDF \
    Data/Final_Processed/X_Train_1_pf.pkl \
    Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl \
    Data/Final_Processed/y_test_1_pf.pkl \
    false
```

**Grid:** 4 models × 2 sparse inputs × 4 embedding inputs × 2 experiments × 3 modes = 144 records appended to `Results/results_main.json`. Original results for this work can also be found in the results folder.

---

### Step 5: DL Hyperparameter Tuning (Optuna)

Runs TPE Optuna studies for each model × experiment × mode combination (10 trials for LegalBERT and PatentBERT; 3 trials for Longformer due to the much longer per-trial wall time). Each trial uses **step-level early stopping** (`eval_every = spe//4`, `patience = 8` steps, `min_global_steps = 3 × spe`). Studies are stored in persistent SQLite databases and resume automatically if a job is interrupted.

The globally best checkpoint across all trials is saved to `Results/optuna/best_{model}_{case}_exp{N}.pt`.

#### Supported models

| CLI identifier | HuggingFace model |
|---|---|
| `legalbert` | `nlpaueb/legal-bert-base-uncased` |
| `patentbert` | `anferico/bert-for-patents` |
| `longformer_base` | `allenai/longformer-base-4096` |

#### HP search space

Search spaces differ slightly by model architecture:

| Parameter | LegalBERT | PatentBERT | Longformer |
|---|---|---|---|
| `lr` | log-uniform [1e-5, 5e-5] | log-uniform [1e-6, 2e-5] | log-uniform [1e-5, 5e-5] |
| `batch_size` | {8, 16, 32} | {16, 32} | fixed {2} |
| `dropout` | {0.1, 0.2, 0.3} | {0.1, 0.2, 0.3} | {0.1, 0.2, 0.3} |
| `weight_decay` | {0.0, 0.01, 0.05, 0.1} | {0.0, 0.01, 0.05, 0.1} | {0.0, 0.01, 0.05, 0.1} |

PatentBERT uses a lower LR range (`[1e-6, 2e-5]`) to prevent gradient explosion (BERT-Large scale). Longformer `batch_size` is fixed at 2 with gradient accumulation of 8 steps, giving an effective batch size of 16.

#### SLURM scripts (one per model × exp × case)

```bash
python -m Experiments.run_deep_learning_experiment \
    legalbert 1 false \
    Data/Final_Processed/X_Train_1_pf.pkl \
    Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl \
    Data/Final_Processed/y_test_1_pf.pkl \
    --n_trials 10 \
    --optuna_storage "sqlite:///Results/optuna/legalbert_exp1_pf.db" \
    --study_name    "legalbert_exp1_pf" \
    --results_path  Results/results_deep_learning.json
```

For Longformer use `--n_trials 3` (each trial takes ~6–8 h on an A100 with a 24 h wall limit).

Set `opposition` (`true`/`false`) to `true` for `op` and `both` modes; `false` for `pf`.

Results are appended to `Results/results_deep_learning.json`. Similarly the the ML runs, the original results can also be found in the Results folder.

---

### Step 6: DL Full-Test Evaluation

Loads the saved `best_*.pt` checkpoint from Step 5 and evaluates on the held-out test set — **no retraining is performed**. Requires Step 5 to be complete for the target model/exp/case.

```bash
python -m Experiments.evaluation \
    --skip_ml \
    --skip_sliding_window \
    --experiments 1 \
    --only_algos legalbert \
    --only_cases pf \
    --dl_results Results/results_deep_learning.json \
    --output     Results/results_dl_full_test.json
```

Results are appended to `Results/results_dl_full_test.json`.

---

### Step 7: Sliding-Window Temporal Evaluation

Trains a fresh model on all data up to year `T−1` (with the year `T−1` held out as a validation set) and tests on year `T`, for `T` in 2021–2024. Runs for **Exp 2** only, and uses the hyperparameters from Step 5 except for epoch as it uses early-stopping.

#### ML sliding window

```bash
# SLURM (one script per case):
sbatch run_eval_ml_sw_pf_balanced.sh
sbatch run_eval_ml_sw_op_balanced.sh
sbatch run_eval_ml_sw_both_balanced.sh

# Direct Python call:
python -m Experiments.evaluation \
    --skip_dl \
    --skip_full_test \
    --experiments 2 \
    --only_cases pf \
    --output Results/results_ml_sliding_window.json
```

#### DL sliding window

```bash
# SLURM — LegalBERT (one script per case):
for case in pf op both; do sbatch run_eval_lb_sw_exp2_${case}.sh; done

# PatentBERT:
for case in pf op both; do sbatch run_eval_pb_sw_exp2_${case}.sh; done

# Longformer (after HP complete):
for case in pf op both; do sbatch run_eval_lf_sw_exp2_${case}.sh; done

# Direct Python call:
python -m Experiments.evaluation \
    --skip_ml \
    --skip_full_test \
    --experiments 2 \
    --only_algos legalbert \
    --only_cases pf \
    --dl_results Results/results_deep_learning.json \
    --output     Results/results_dl_sliding_window.json
```

ML results → `Results/results_ml_sliding_window.json`  
DL results → `Results/results_dl_sliding_window.json`

---

## Results Summary

Full results, methodology notes, and cross-model comparisons are in `Results/experiment_report_skeleton.md` (rendered as `Results/experiment_report.html`).

### ML Full-Test (best model per exp × case)

| Exp | Case | Model | Input | Test F1 | MCC |
|-----|------|-------|-------|---------|-----|
| 1 | pf | XGBoost | TF-IDF | **0.9026** | 0.6891 |
| 1 | op | XGBoost | N-Grams | 0.6734 | 0.5450 |
| 1 | both | XGBoost | N-Grams | **0.8067** | 0.6526 |
| 2 | pf | Logistic Regression | TF-IDF | **0.8895** | 0.6877 |
| 2 | op | XGBoost | N-Grams | 0.7447 | 0.5810 |
| 2 | both | XGBoost | N-Grams | 0.7980 | 0.6215 |

### ML Sliding Window (Exp 2, mean F1 across 2021–2024)

| Case | Best Model | Input | Mean F1 |
|------|-----------|-------|---------|
| pf | XGBoost | N-Grams | 0.9069 |
| op | XGBoost | N-Grams | 0.7488 |
| both | XGBoost | N-Grams | 0.8078 |

### DL Full-Test (HP checkpoint, retrained=False)

| Model | Exp | Case | Test F1 | MCC |
|-------|-----|------|---------|-----|
| LegalBERT | 1 | pf | **0.8949** | 0.6431 |
| LegalBERT | 1 | op | 0.5874 | 0.4232 |
| LegalBERT | 1 | both | 0.7663 | 0.5565 |
| LegalBERT | 2 | pf | **0.8868** | 0.6811 |
| LegalBERT | 2 | op | 0.7111 | 0.4994 |
| LegalBERT | 2 | both | 0.7762 | 0.5989 |
| PatentBERT | 1 | pf | **0.8899** | 0.6198 |
| PatentBERT | 1 | op | 0.5840 | 0.4107 |
| PatentBERT | 1 | both | 0.7743 | 0.5732 |
| PatentBERT | 2 | pf | **0.8816** | 0.6551 |
| PatentBERT | 2 | op | 0.6841 | 0.4444 |
| PatentBERT | 2 | both | 0.7789 | 0.5581 |
| Longformer | 1 | pf | **0.9054** | 0.6576 |
| Longformer | 1 | op | 0.6598 | 0.5247 |
| Longformer | 1 | both | 0.7843 | 0.6066 |
| Longformer | 2 | pf | **0.8989** | 0.7192 |
| Longformer | 2 | op | 0.6838 | 0.4433 |
| Longformer | 2 | both | 0.7818 | 0.5784 |

### DL Sliding Window (Exp 2, mean F1 across 2021–2024)

| Model | pf | op | both |
|-------|----|----|------|
| LegalBERT | 0.899 | 0.707 | 0.815 |
| PatentBERT | 0.891 | 0.706 | 0.799 |
| Longformer | 0.927 | 0.722 | *(2023–2024 pending)* |
| ML best | 0.907 | 0.749 | 0.808 |

---

## Explainability Analysis

SHAP (TreeExplainer) analysis of the best-tuned XGBoost models from Experiment 2 is in `Experiments/shap_analysis.py`. Two independent analyses are available, each runnable via CLI.

### Part A — Cross-mode feature importance comparison

Trains XGBoost on the Exp 2 train set for all three case modes (`pf`, `op`, `both`), computes global mean |SHAP| importance on the test set, and compares:

1. **Grouped bar chart** — top-N features ranked by mean importance across modes (each mode normalised to its own max for scale comparability).
2. **Spearman rank-correlation heatmap** — pairwise ρ between the full feature importance rankings of each mode pair, with p-values.

```bash
python -m Experiments.shap_analysis cross_mode --top_n 20 --out_dir Results/shap
```

Output: `Results/shap/cross_mode_bar.png`, `Results/shap/cross_mode_spearman.png`

### Part B — Sliding-window SHAP trajectory (pf)

Re-trains `pf` XGBoost on each sliding window (train ≤ T−1, test = T, T ∈ 2021–2024) and computes global mean |SHAP| importance per window:

1. **Heatmap** — top-N features × test year, row-normalised to show relative shift in each feature's importance over time.
2. **Line plot** — top-K feature trajectories across years.

```bash
python -m Experiments.shap_analysis sliding_window --top_n 20 --top_k 8 \
    --first_year 2021 --last_year 2024 --out_dir Results/shap
```

Output: `Results/shap/sw_heatmap.png`, `Results/shap/sw_lineplot.png`

### Run both

```bash
python -m Experiments.shap_analysis all --top_n 20 --top_k 8
```

> **Note:** The script uses the best hyperparameters identified by `RandomizedSearchCV` for each case mode (hardcoded from `Results/results_main.json`). SHAP values are computed on the **test set** to reflect importance under the held-out distribution.


## Configuration & Key Parameters

### ML (`ml_experiments.py`)

| Parameter | Value |
|-----------|-------|
| CV search method | `RandomizedSearchCV`, 50 iterations |
| CV strategy (Exp 1) | `RepeatedStratifiedKFold`, 3 splits |
| CV strategy (Exp 2) | `TimeSeriesSplit`, 3 splits |
| Scoring metric | macro F1 |
| Random seed | 42 |

### DL (`deep_learning_experiments.py`)

| Parameter | Value |
|-----------|-------|
| Optuna trials | 10 (LegalBERT, PatentBERT) / 3 (Longformer) |
| Optuna sampler | TPE, seed=42 |
| Max epochs | 30 |
| Early-stop eval frequency | `spe // 4` steps |
| Early-stop patience | 8 evaluation steps |
| Min training steps | `3 × spe` |
| Gradient accumulation (Longformer) | 8 steps |
| Max sequence length (BERT) | 512 tokens (sliding-window mean pool) |
| Max sequence length (Longformer) | 4096 tokens |
| CLS pooling (PatentBERT) | `last_hidden_state[:, 0, :]` (no pooler) |
| Random seed | 42 |
