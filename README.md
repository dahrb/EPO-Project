# EPO Patent Appeal Outcome Prediction

A research pipeline for predicting the outcomes of European Patent Office (EPO) Board of Appeals decisions using classical ML, embedding-based models, and transformer-based deep learning.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Reproducing Experiments](#reproducing-experiments)
  - [Step 1: Data Processing](#step-1-data-processing)
  - [Step 2: Train Patent Embeddings](#step-2-train-patent-embeddings)
  - [Step 3: Generate Train/Test Splits](#step-3-generate-traintest-splits)
  - [Step 4: Classical ML & Embedding Experiments](#step-4-classical-ml--embedding-experiments)
  - [Step 5: Deep Learning Experiments](#step-5-deep-learning-experiments)
    - [LegalBERT](#legalbert)
    - [Longformer](#longformer)
- [Results](#results)
- [Configuration & Key Parameters](#configuration--key-parameters)

---

## Project Overview

This project classifies the outcome of EPO patent appeal decisions (grant/refuse for *ex parte* cases; upheld/overturned for *inter partes* / opposition cases) across two experimental splits:

| Split | Description |
|-------|-------------|
| **Exp 1** | Earlier decisions used for training, recent for test |
| **Exp 2** | Stratified random split |

Each split is run in three **modes**: `pf` (ex parte), `op` (opposition), `both` (combined).

The full pipeline is:
1. Parse raw XML → processed CSV  
2. Train domain-specific embeddings (Patent2Vec, PatentDoc2Vec)  
3. Generate train/test pickle files  
4. Run classical ML / embedding experiments  
5. Run deep learning (LegalBERT / Longformer) experiments with Optuna HPO

---

## Project Structure

```
EPO-Project/
├── Data/
│   ├── EPDecisions_March2025.xml       # Raw EPO XML data
│   ├── Final_Processed/                # Train/test splits (generated)
│   └── Matching/                       # Intermediate processed files
├── Experiments/
│   ├── data_processing.py              # Step 1: XML → processed data
│   ├── experiment_processing.py        # Step 3: generate train/test splits
│   ├── PatentEmbeddings.py             # Step 2: train Patent2Vec / Doc2Vec
│   ├── ml_experiments.py               # Step 4: classical ML models
│   ├── run_experiment.py               # CLI runner for classical experiments
│   ├── deep_learning_experiments.py    # Step 5: Optuna HPO loop for DL models
│   ├── run_deep_learning_experiment.py # CLI runner for LegalBERT / Longformer
│   └── sliding_window.py               # Sliding-window tokenisation utility
├── Models/                             # Pre-trained / trained embeddings
│   ├── Patent2Vec_1.0                  # Word2Vec patent embeddings
│   ├── Doc2Vec_1.0                     # Doc2Vec patent embeddings
│   ├── Word2Vec-google-300d            # Google News Word2Vec
│   └── Law2Vec.200d.txt                # Law2Vec embeddings
├── Results/
│   ├── results_main.json               # All classical / embedding results
│   └── results.ipynb                   # Results analysis notebook
├── Utilities/
│   └── utils.py
├── run_data_processing.sh              # SLURM: data processing
├── run_experiment_grids.sh             # SLURM: classical ML grid
├── run_deep_learning_grids.sh          # SLURM: DL grid (LegalBERT / Longformer)
├── run_dl_legalbert.sh                 # SLURM: LegalBERT single run
├── run_dl_longformer_base.sh           # SLURM: Longformer pf
├── run_missing_xgb_tfidf.sh            # SLURM: fill missing XGB/TF-IDF runs
└── pyproject.toml
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-org>/EPO-Project.git
cd EPO-Project
```

### 2. Create Environment
```bash
# Using uv (recommended)
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install .

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install .
```

---

## Reproducing Experiments

All steps below assume you are in the project root. On an HPC cluster with SLURM, use the provided `.sh` scripts. Locally, call the Python scripts directly.

---

### Step 1: Data Processing

Parses `Data/EPDecisions_March2025.xml` and produces cleaned, feature-engineered CSV files in `Data/Matching/`.

```bash
# On SLURM:
sbatch run_data_processing.sh

# Locally:
python Experiments/data_processing.py
```

**What it does (8-step pipeline):**
1. Parse raw XML decisions
2. Filter to relevant decision types
3. Extract cited legal provisions
4. Classify outcomes (grant/refuse/upheld/overturned)
5. Sub-classify by technical field
6. Engineer features (text length, IPC codes, etc.)
7. Produce `pf` (ex parte) and `op` (opposition) datasets
8. Save processed CSVs to `Data/Matching/`

---

### Step 2: Train Patent Embeddings

Trains domain-specific Word2Vec and Doc2Vec models on the processed patent text.

```bash
python Experiments/PatentEmbeddings.py
```

**Output models saved to `Models/`:**
- `Patent2Vec_1.0` — Word2Vec trained on patent claims/descriptions
- `Doc2Vec_1.0` — Doc2Vec trained on patent documents

Pre-trained models (Word2Vec-google-300d, Law2Vec) are already provided in `Models/`.

---

### Step 3: Generate Train/Test Splits

Produces pickle files for each experiment × mode combination used by all downstream runners.

```bash
python Experiments/experiment_processing.py
```

**Output** (written to `Data/Final_Processed/`):
```
X_Train_1_pf.pkl  y_Train_1_pf.pkl  X_test_1_pf.pkl  y_test_1_pf.pkl
X_Train_1_op.pkl  y_Train_1_op.pkl  X_test_1_op.pkl  y_test_1_op.pkl
X_Train_1_both.pkl ...
X_Train_2_pf.pkl  ...  (Exp 2 stratified split)
```

---

### Step 4: Classical ML & Embedding Experiments

Runs Logistic Regression, Random Forest, and XGBoost with sparse (N-Gram, TF-IDF) and dense embedding inputs (Word2Vec, Law2Vec, Patent2Vec, Doc2Vec).

```bash
# On SLURM — runs the full grid:
bash run_experiment_grids.sh

# Single run example:
python Experiments/run_experiment.py \
    xgboost 1 pf TF-IDF \
    Data/Final_Processed/X_Train_1_pf.pkl \
    Data/Final_Processed/y_Train_1_pf.pkl \
    Data/Final_Processed/X_test_1_pf.pkl \
    Data/Final_Processed/y_test_1_pf.pkl
```

**Grid dimensions:**
- Models: `logistic`, `forest`, `xgboost`
- Sparse inputs: `N-Gram`, `TF-IDF`
- Embedding inputs: `Word2Vec`, `Law2Vec`, `Patent2Vec`, `Doc2Vec`
- Experiments: `1`, `2`
- Modes: `pf`, `op`, `both`

Results are appended to `Results/results_main.json`.

---

### Step 5: Deep Learning Experiments

All DL experiments use **Optuna** (TPE sampler) for hyperparameter optimisation and early stopping. Results are saved as JSON in `Results/`.

#### Common Hyperparameters (all models)
| Parameter | Values searched |
|-----------|----------------|
| `lr` | log-uniform [1e-5, 5e-5] |
| `weight_decay` | categorical [0.0, 0.01, 0.001] |
| `batch_size` | categorical [8, 16] |
| `epochs` | **fixed at 30** |

#### LegalBERT

Model: `nlpaueb/legal-bert-base-uncased`. Encodes full decision text via sliding window + mean pooling.

```bash
# SLURM grid (all experiments × modes):
bash run_deep_learning_grids.sh

# Single run:
python Experiments/run_deep_learning_experiment.py \
    --model_name nlpaueb/legal-bert-base-uncased \
    --experiment 1 \
    --mode pf \
    --n_trials 10 \
    --results_file Results/results_legalbert_pf_exp1.json
```

Additional HPO params for LegalBERT: `dropout` [0.1, 0.2, 0.3]  
Patience: **3 epochs**

#### Longformer

Model: `allenai/longformer-base-4096`. Handles long sequences natively (global attention on CLS token).

```bash
# SLURM — pf split:
sbatch run_dl_longformer_base.sh

# SLURM — op/both splits:
sbatch run_dl_longformer_op.sh

# Single run:
python Experiments/run_deep_learning_experiment.py \
    --model_name allenai/longformer-base-4096 \
    --experiment 1 \
    --mode pf \
    --n_trials 3 \
    --results_file Results/results_longformer_pf_exp1.json
```

Additional HPO params: `dropout` [0.1, 0.2, 0.3]  
Patience: **3 epochs** | Trials: **3**

> **Note:** Longformer does not have a pooler layer. CLS representation is taken from `last_hidden_state[:, 0, :]`.

---

## Results

Results are stored as JSON files in `Results/`. Use `Results/results.ipynb` to load, aggregate, and visualise them.

| Model | Exp | Mode | Best F1 |
|-------|-----|------|---------|
| LegalBERT | 2 | pf | 0.8971 |
| LegalBERT | 2 | op | 0.7699 |

---

## Configuration & Key Parameters

| Setting | Location | Default |
|---------|----------|---------|
| Epochs | `deep_learning_experiments.py` | 30 (fixed) |
| Patience (DL) | `deep_learning_experiments.py` | 3 |
| Optuna trials (Longformer) | `run_dl_longformer_base.sh` | 3 |
| Random seed | All experiment runners | 42 |
