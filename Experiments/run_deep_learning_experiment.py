"""CLI runner for EPO experiments (DL)

One model per invocation — the shell script loops over models, mirroring how
run_experiment.py is called once per word embedding.

Usage
-----
python run_deep_learning_experiment.py MODEL EXPERIMENT OPPOSITION \
        X_TRAIN Y_TRAIN X_TEST Y_TEST \
        [--cv_num 3] [--repeat 1] \
        [--results_path path/to/results.json]

Arguments
---------
- `MODEL`: transformer model identifier:
    legalbert        — nlpaueb/legal-bert-base-uncased   (512 tokens)
    longformer_base  — lexlms/legal-longformer-base      (4096 tokens)
    longformer_large — lexlms/legal-longformer-large     (4096 tokens)
    roberta_base     — lexlms/legal-roberta-base         (512 tokens)
    roberta_large    — lexlms/legal-roberta-large        (512 tokens)
- `EXPERIMENT`: experiment identifier, `1` or `2`
- `OPPOSITION`: boolean string such as `true` or `false`; enables structured
  auxiliary feature fusion (encoder CLS token + one-hot encoded columns)
- `X_TRAIN`, `Y_TRAIN`, `X_TEST`, `Y_TEST`: pickle file paths

Optional
--------
- `--cv_num`: number of CV splits (default: 3)
- `--repeat`: number of CV repeats, Exp 1 only (default: 1)
- `--n_iter`: number of random hyperparameter combos to sample (default: 10)
- `--results_path`: path to output JSON; default:
  <project_root>/Results/results_deep_learning.json

Last Updated: 09.04.26

Status: Done
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.deep_learning_experiments import DeepLearningExperiments
from Utilities.utils import parse_bool, to_numpy_labels

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "Results" / "results_deep_learning.json"
ALL_MODELS = [
    "legalbert",
    "longformer_base",
    "longformer_large",
    "roberta_base",
    "roberta_large",
]


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run deep learning EPO experiments from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional args
    parser.add_argument("model", choices=ALL_MODELS, help="Transformer model identifier")
    parser.add_argument("experiment", help="Experiment identifier, e.g. 1 or 2")
    parser.add_argument("opposition", help="Opposition mode: true/false")
    parser.add_argument("x_train", help="Path to X_train pickle file")
    parser.add_argument("y_train", help="Path to y_train pickle file")
    parser.add_argument("x_test", help="Path to X_test pickle file")
    parser.add_argument("y_test", help="Path to y_test pickle file")
    # Optional args
    parser.add_argument(
        "--cv_num",
        type=int,
        default=3,
        help="Number of CV splits (default: 3)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="CV repeats for Exp 1 RepeatedStratifiedKFold (default: 1)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="Number of random hyperparameter combos to sample (default: 10)",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Output JSON path (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser


def main():
    """Main entry point."""
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"CV: cv_num={args.cv_num}, repeat={args.repeat}, n_iter={args.n_iter}")
    print(f"Results → {args.results_path}")

    exp = DeepLearningExperiments(
        model_name=args.model,
        experiment=args.experiment,
        opposition=parse_bool(args.opposition),
        device=device,
        cv_num=args.cv_num,
        repeat=args.repeat,
        n_iter=args.n_iter,
        results_json_path=str(args.results_path),
    )

    X_train = pd.read_pickle(args.x_train)
    y_train = pd.read_pickle(args.y_train)
    X_test = pd.read_pickle(args.x_test)
    y_test = pd.read_pickle(args.y_test)

    y_train = to_numpy_labels(y_train)
    y_test = to_numpy_labels(y_test)

    results = exp.training_loop(X_train, y_train, X_test=X_test, y_test=y_test)

    print(f"\n{'='*60}")
    print(f"Completed {len(results)} run(s)")
    print(f"Results saved to: {exp.results_json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

