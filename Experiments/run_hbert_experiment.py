"""CLI runner for EPO experiments (H-BERT)

Runs the H-BERT experiments for each model

Usage
-----
python run_hbert_experiment.py EXPERIMENT OPPOSITION \
        X_TRAIN Y_TRAIN X_TEST Y_TEST \
        [--n_trials 10] [--timeout SECONDS] \
        [--max_chunks 64] [--chunk_size 128] \
        [--results_path path/to/results.json]

Arguments
---------
- ``EXPERIMENT``: experiment identifier, ``1`` or ``2``
- ``OPPOSITION``: boolean string such as ``true`` or ``false``
- ``X_TRAIN``, ``Y_TRAIN``, ``X_TEST``, ``Y_TEST``: pickle file paths

Optional
--------
- ``--n_trials``: number of Optuna trials (default: 10)
- ``--timeout``: wall-clock budget in seconds (default: None)
- ``--max_chunks``: maximum chunks per document (default: 64)
- ``--chunk_size``: tokens per chunk incl. special tokens (default: 128)
- ``--results_path``: output JSON path

Last Updated: 12.04.26

Status: Done
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.deep_learning_experiments import DeepLearningExperiments
from Utilities.utils import parse_bool, to_numpy_labels

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "Results" / "results_hbert.json"

def infer_case_mode(path: str) -> str:
    """Infer dataset case mode (pf/op/both) from input filename."""
    name = Path(path).name.lower()
    match = re.search(r"_(pf|op|both)\.pkl$", name)
    return match.group(1) if match else "unknown"

def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run H-BERT EPO experiments from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional args
    parser.add_argument("experiment", help="Experiment identifier, e.g. 1 or 2")
    parser.add_argument("opposition", help="Opposition mode: true/false")
    parser.add_argument("x_train", help="Path to X_train pickle file")
    parser.add_argument("y_train", help="Path to y_train pickle file")
    parser.add_argument("x_test", help="Path to X_test pickle file")
    parser.add_argument("y_test", help="Path to y_test pickle file")
    # Optional args
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of Optuna trials (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock budget in seconds; study stops at n_trials or timeout, whichever comes first (default: None)",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=64,
        help="Maximum chunks per document (default: 64)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Tokens per chunk including special tokens (default: 128)",
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
    case_mode = infer_case_mode(args.x_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Architecture: H-BERT (hierarchical)")
    print(f"Case mode: {case_mode}")
    print(f"Optuna: n_trials={args.n_trials}, timeout={args.timeout}")
    print(f"Chunks: size={args.chunk_size}, max={args.max_chunks}")
    print(f"Results → {args.results_path}")

    exp = DeepLearningExperiments(
        model_name="hbert",
        experiment=args.experiment,
        opposition=parse_bool(args.opposition),
        case_mode=case_mode,
        device=device,
        n_trials=args.n_trials,
        timeout=args.timeout,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        results_json_path=str(args.results_path),
    )

    X_train = pd.read_pickle(args.x_train)
    y_train = pd.read_pickle(args.y_train)

    y_train = to_numpy_labels(y_train)

    results = exp.training_loop(X_train, y_train)

    print(f"\n{'='*60}")
    print(f"Completed {len(results)} run(s)")
    print(f"Results saved to: {exp.results_json_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
