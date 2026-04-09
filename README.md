
<div align="center">
	<h1>EPO Project: Patent Data Processing & Embedding Training</h1>
	<p>
		<strong>End-to-end pipeline for processing patent data and training PatentEmbeddings</strong>
	</p>
</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Processing Pipeline](#data-processing-pipeline)
- [PatentEmbedding Training](#patentembedding-training)
- [Reproducibility](#reproducibility)
- [Citing & License](#citing--license)

---

## Project Overview
This repository provides a complete workflow for processing European Patent Office (EPO) data and training custom PatentEmbeddings. The pipeline includes:
- Parsing and cleaning raw XML patent data
- Feature engineering and data transformation
- Training Doc2Vec/Word2Vec-based PatentEmbeddings
- Utilities for downstream ML experiments

## Project Structure
```
EPO-Project/
├── Data/                # Raw and processed data
│   ├── EPDecisions_March2025.xml
│   └── ...
├── Experiments/         # Data processing scripts, embedding training, experiments
│   ├── data_processing.py
│   ├── PatentEmbeddings.py
│   └── ...
├── Models/              # Saved embedding models
├── Utilities/           # Utility scripts (e.g., TableCreator)
├── main.py              # Main entry point (if applicable)
├── run_data_processing.sh
├── run_hyperparameter_search.py
├── pyproject.toml       # Python project metadata
├── README.md
└── ...
```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-org>/EPO-Project.git
cd EPO-Project
```

### 2. Create and Activate a Virtual Environment (Recommended: `uv`)
We recommend using [uv](https://github.com/astral-sh/uv) for fast, reproducible Python environments:
```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv pip install .
```

## Data Processing Pipeline

The main data processing scripts are in `Experiments/data_processing.py` and related files. To process the raw XML data and generate feature tables:

```bash
# Run the data processing pipeline
bash run_data_processing.sh
# or
python Experiments/data_processing.py
```

- **Input:** Raw XML files in `Data/`
- **Output:** Processed data tables (CSV/Parquet) in `Data/` or `Experiments/`

## PatentEmbedding Training

To train PatentEmbeddings (e.g., Doc2Vec, Word2Vec):

```bash
python Experiments/PatentEmbeddings.py --config configs/embedding_config.yaml
```

- **Input:** Processed data from previous step
- **Output:** Trained embedding models in `Models/`

You can adjust hyperparameters and model settings in the config file or via command-line arguments.

## Experiment Runners

The project includes lightweight CLI entrypoints for experiment execution.

### Flat CV experiments

Use `Experiments/run_experiment.py` to run the new flat experiment workflow on
pre-generated train/test pickle files.

```bash
python Experiments/run_experiment.py \
	linear 1 false N-Gram \
	Data/Final_Processed/X_Train_1_pf.pkl \
	Data/Final_Processed/y_Train_1_pf.pkl \
	Data/Final_Processed/X_test_1_pf.pkl \
	Data/Final_Processed/y_test_1_pf.pkl \
	false
```

- `model`: `linear`, `logistic`, `forest`, or `xgboost`
- `experiment`: usually `1` or `2`
- `opposition`: `true` or `false`
- `input_representation`: e.g. `N-Gram`, `TF-IDF`, `Word2Vec`
- final boolean: `true` runs the embedding path, `false` runs the sparse path

### Nested CV experiments

The older nested runner remains available via `run_nested.py` and follows the
same minimal positional-argument style.

## Reproducibility

- All experiments are designed to be reproducible with fixed random seeds.
- Use the provided `pyproject.toml` and/or `requirements.txt` for consistent environments.
- For SLURM/HPC users, adapt the provided shell scripts (e.g., `epo_2_xgb_pf_l2v.sh`) for batch processing.
- For Jupyter-based exploration, see `Experiments/patentJudgements.ipynb` and `visualisations.ipynb`.

## Citing & License

If you use this code or data, please cite appropriately (add citation instructions here).

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
