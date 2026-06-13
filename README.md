# EPO Patent Appeal Outcome Prediction

A research pipeline for predicting the outcomes of European Patent Office (EPO) Board of Appeals decisions using classical ML and transformer-based deep learning (LegalBERT, PatentBERT, Legal-Longformer).

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

This project classifies the outcome of EPO patent appeal decisions — grant/refuse for *ex parte* cases and upheld/overturned for *inter partes* (opposition) cases — across two experimental splits:

| Split | Description |
|-------|-------------|
| **Exp 1** | Temporal split — earlier decisions for training, recent for test |
| **Exp 2** | Stratified random split |

Each split is evaluated in three **modes**: `pf` (ex parte), `op` (opposition), `both` (combined).

The full pipeline is:
1. Parse raw XML → processed CSV
2. Train domain-specific embeddings (Patent2Vec, PatentDoc2Vec)
3. Generate train/test pickle files per experiment × mode
4. Classical ML experiments (grid search)
5. DL hyperparameter tuning via Optuna (persistent SQLite, resumable)
6. DL full-test evaluation (loads saved HP checkpoint, no retraining)
7. Sliding-window temporal evaluation (test years 2021–2024)

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
│   └── evaluation.py                        # CLI runner for full-test & SW eval
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

All steps assume you are in the project root. On an HPC cluster with SLURM use the provided `.sh` scripts; locally call the Python modules directly.

---

### Step 1: Data Processing

Parses `Data/EPDecisions_March2025.xml` and produces cleaned, feature-engineered CSV files.

```bash
# SLURM:
sbatch run_data_processing.sh

# Locally:
python -m Experiments.data_processing
```

**8-step pipeline:** XML parse → decision-type filter → legal-provision extraction → outcome classification → technical-field sub-classification → feature engineering → `pf`/`op` dataset split → save to `Data/Matching/`.

---

### Step 2: Train Patent Embeddings

Trains domain-specific Word2Vec and Doc2Vec models on the processed patent text.

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

```bash
# SLURM — full grid (submits one job per combination):
bash run_deep_learning_grids.sh

# Single run example:
python -m Experiments.run_experiment \
    xgboost 1 false TF-IDF \
    Data/Final_Processed/X_Train_1_pf.pkl \
    Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl \
    Data/Final_Processed/y_test_1_pf.pkl \
    false
```

**Grid:** 4 models × 2 sparse inputs × 4 embedding inputs × 2 experiments × 3 modes = 144 records appended to `Results/results_main.json`.

---

### Step 5: DL Hyperparameter Tuning (Optuna)

Runs 10-trial TPE Optuna studies for each model × experiment × mode combination. Each trial uses **step-level early stopping** (`eval_every = spe//4`, `patience = 8` steps, `min_global_steps = 3 × spe`). Studies are stored in persistent SQLite databases and resume automatically if a job is interrupted.

The globally best checkpoint across all trials is saved to `Results/optuna/best_{model}_{case}_exp{N}.pt`.

#### Supported models

| CLI identifier | HuggingFace model |
|---|---|
| `legalbert` | `nlpaueb/legal-bert-base-uncased` |
| `patentbert` | `anferico/bert-for-patents` |
| `longformer_base` | `allenai/longformer-base-4096` |

#### HP search space (all models)

| Parameter | Range / choices |
|---|---|
| `lr` | log-uniform [1e-5, 5e-5] |
| `batch_size` | {8, 16, 32} |
| `dropout` | {0.1, 0.2, 0.3} |
| `weight_decay` | {0.0, 0.01, 0.05, 0.1} |

Longformer additionally uses gradient accumulation of 8 steps (effective batch = `batch_size × 8`, fixed `batch_size = 2`).

#### SLURM scripts (one per model × exp × case)

```bash
# LegalBERT — submit all 6:
for exp in 1 2; do for case in pf op both; do
    sbatch run_hp_legalbert_exp${exp}_${case}.sh
done; done

# PatentBERT:
for exp in 1 2; do for case in pf op both; do
    sbatch run_hp_patentbert_exp${exp}_${case}.sh
done; done

# Longformer:
for exp in 1 2; do for case in pf op both; do
    sbatch run_hp_longformer_exp${exp}_${case}.sh
done; done
```

#### Direct Python call (single run)

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

Set `opposition` (`true`/`false`) to `true` for `op` and `both` modes; `false` for `pf`.

Results are appended to `Results/results_deep_learning.json`.

---

### Step 6: DL Full-Test Evaluation

Loads the saved `best_*.pt` checkpoint from Step 5 and evaluates on the held-out test set — **no retraining is performed**. Requires Step 5 to be complete for the target model/exp/case.

#### SLURM scripts

```bash
# LegalBERT full-test — all 6:
for exp in 1 2; do for case in pf op both; do
    sbatch run_eval_lb_ft_exp${exp}_${case}.sh
done; done

# PatentBERT:
for exp in 1 2; do for case in pf op both; do
    sbatch run_eval_pb_ft_exp${exp}_${case}.sh
done; done

# Longformer (after HP complete):
for exp in 1 2; do for case in pf op both; do
    sbatch run_eval_lf_ft_exp${exp}_${case}.sh
done; done
```

#### Direct Python call

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

Trains a fresh model on all data up to year `T−1` (with the year `T−1` held out as a validation set) and tests on year `T`, for `T` in 2021–2024. Runs for **Exp 2** only.

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
| PatentBERT | 2 | pf | **0.8816** | 0.6551 |
| PatentBERT | 2 | op | 0.6841 | 0.4444 |
| PatentBERT | 2 | both | 0.7789 | 0.5581 |
| Longformer | all | all | *(HP in progress)* | — |

### DL Sliding Window (Exp 2, mean F1 across 2021–2024)

| Model | pf | op | both |
|-------|----|----|------|
| LegalBERT | 0.899 | 0.707 | 0.815 |
| PatentBERT | 0.891 | 0.706 | 0.799 |
| ML best | 0.910 | 0.752 | 0.810 |
| Longformer | *(pending)* | *(pending)* | *(pending)* |

---

## Configuration & Key Parameters

| Parameter | Location | Value |
|-----------|----------|-------|
| Optuna trials | `run_hp_*.sh` | 10 (BERT), 10 (Longformer) |
| Optuna sampler | `deep_learning_experiments.py` | TPE, seed=42 |
| Max epochs | `deep_learning_experiments.py` | 30 |
| Early-stop eval frequency | `deep_learning_experiments.py` | `spe // 4` steps |
| Early-stop patience | `deep_learning_experiments.py` | 8 evaluation steps |
| Min training steps | `deep_learning_experiments.py` | `3 × spe` |
| Gradient accumulation (Longformer) | `deep_learning_experiments.py` | 8 steps |
| Max sequence length (BERT) | `deep_learning_experiments.py` | 512 tokens (sliding-window mean pool) |
| Max sequence length (Longformer) | `deep_learning_experiments.py` | 4096 tokens |
| CLS pooling (PatentBERT) | `deep_learning_experiments.py` | `last_hidden_state[:, 0, :]` (no pooler) |
| Random seed | all runners | 42 |
| SLURM partition (DL) | all `run_hp_*.sh` / `run_eval_*.sh` | `gpu-a100-lowbig` |
| SLURM wall time (DL HP) | `run_hp_*.sh` | 24 h |
| SLURM wall time (DL eval) | `run_eval_*_ft_*.sh` | 12 h |
| SLURM wall time (DL SW) | `run_eval_*_sw_*.sh` | 24 h |
| SLURM excluded nodes | all DL scripts | `gpu07`, `gpu08` |
