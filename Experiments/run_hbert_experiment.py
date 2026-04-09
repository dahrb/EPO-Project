"""CLI runner for EPO experiments (H-BERT)

Hierarchical BERT — one invocation runs the full CV search + retrain + test
pipeline.  No model selection argument needed (always Legal-BERT encoder).

Usage
-----
python run_hbert_experiment.py EXPERIMENT OPPOSITION \
        X_TRAIN Y_TRAIN X_TEST Y_TEST \
        [--cv_num 3] [--repeat 1] [--n_iter 10] \
        [--max_chunks 64] [--chunk_size 128] [--accum_steps 4] \
        [--results_path path/to/results.json]

Arguments
---------
- ``EXPERIMENT``: experiment identifier, ``1`` or ``2``
- ``OPPOSITION``: boolean string such as ``true`` or ``false``
- ``X_TRAIN``, ``Y_TRAIN``, ``X_TEST``, ``Y_TEST``: pickle file paths

Optional
--------
- ``--cv_num``: number of CV splits (default: 3)
- ``--repeat``: number of CV repeats, Exp 1 only (default: 1)
- ``--n_iter``: number of random hyperparameter combos to sample (default: 10)
- ``--max_chunks``: maximum chunks per document (default: 64)
- ``--chunk_size``: tokens per chunk incl. special tokens (default: 128)
- ``--accum_steps``: gradient accumulation steps (default: 4)
- ``--results_path``: output JSON path

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

from Experiments.hierarchical_bert import HBertExperiment
from Utilities.utils import parse_bool, to_numpy_labels

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "Results" / "results_hbert.json"


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
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
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
    print(f"Architecture: H-BERT (128×64 hierarchical)")
    print(
        f"CV: cv_num={args.cv_num}, repeat={args.repeat}, n_iter={args.n_iter}"
    )
    print(
        f"Chunks: size={args.chunk_size}, max={args.max_chunks}, "
        f"accum_steps={args.accum_steps}"
    )
    print(f"Results → {args.results_path}")

    exp = HBertExperiment(
        experiment=args.experiment,
        opposition=parse_bool(args.opposition),
        device=device,
        cv_num=args.cv_num,
        repeat=args.repeat,
        n_iter=args.n_iter,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        accum_steps=args.accum_steps,
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
