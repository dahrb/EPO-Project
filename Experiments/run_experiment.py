"""CLI runner for EPO experiments (ML)

Usage
-----
python run_experiment.py MODEL EXPERIMENT OPPOSITION INPUT_REPRESENTATION \
        X_TRAIN Y_TRAIN X_TEST Y_TEST USE_EMBEDDINGS

Arguments
---------
- `MODEL`: `linear`, `logistic`, `forest`, or `xgboost`
- `EXPERIMENT`: experiment identifier, typically `1` or `2`
- `OPPOSITION`: boolean string such as `true` or `false`
- `INPUT_REPRESENTATION`: e.g. `N-Gram`, `TF-IDF`, `Word2Vec`
- `X_TRAIN`, `Y_TRAIN`, `X_TEST`, `Y_TEST`: pickle file paths
- `USE_EMBEDDINGS`: boolean string selecting `training_loop_we()` vs `training_loop()`

Last Updated: 03.04.26

Status: Done
"""

import argparse
import re
import sys
import types
from pathlib import Path

# numpy 2.0 removed np.float_ / np.complex_ etc.; cupy (spacy[cuda12x] dep)
# still references them, crashing before spacy's own try/except can catch it.
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "AxisError"):
    np.AxisError = np.exceptions.AxisError

# Stub out cupy entirely so spacy falls back to CPU cleanly.
_cupy_stub = types.ModuleType("cupy")
_cupy_stub.ndarray = np.ndarray
for _sub in ["cuda", "cuda.stream", "testing", "_core", "_core.core"]:
    sys.modules[f"cupy.{_sub}"] = types.ModuleType(f"cupy.{_sub}")
sys.modules["cupy"] = _cupy_stub
del _cupy_stub, _sub

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.ml_experiments import Experiments
from Utilities.utils import parse_bool, to_numpy_labels


def infer_case_mode(path: str) -> str:
    """Infer dataset case mode (pf/op/both) from input filename."""
    name = Path(path).name.lower()
    match = re.search(r"_(pf|op|both)\.pkl$", name)
    return match.group(1) if match else "unknown"


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run flat EPO experiments from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Model type: linear, logistic, forest, xgboost")
    parser.add_argument("experiment", help="Experiment identifier, e.g. 1 or 2")
    parser.add_argument("opposition", help="Opposition mode: true/false")
    parser.add_argument("input_representation", help="Input type: N-Gram, TF-IDF, Word2Vec, etc.")
    parser.add_argument("x_train", help="Path to X_train pickle file")
    parser.add_argument("y_train", help="Path to y_train pickle file")
    parser.add_argument("x_test", help="Path to X_test pickle file")
    parser.add_argument("y_test", help="Path to y_test pickle file")
    parser.add_argument("use_embeddings", help="Use embedding path instead of sparse path: true/false")
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional path to write results JSON (default: Results/results_main.json). "
             "Use a dedicated file to keep variant runs out of results_main.json.",
    )
    return parser


def main():
    """Main entry point."""
    args = build_parser().parse_args()

    results_json_path = args.results_json or str(PROJECT_ROOT / "Results" / "results_main.json")
    case_mode = infer_case_mode(args.x_train)

    exp = Experiments(
        [args.model],
        experiment=args.experiment,
        opposition=parse_bool(args.opposition),
        case_mode=case_mode,
        input_representation=args.input_representation,
        results_json_path=results_json_path,
    )

    X_train = pd.read_pickle(args.x_train)
    y_train = pd.read_pickle(args.y_train)
    X_test = pd.read_pickle(args.x_test)
    y_test = pd.read_pickle(args.y_test)

    y_train = to_numpy_labels(y_train)
    y_test = to_numpy_labels(y_test)

    if parse_bool(args.use_embeddings):
        exp.training_loop_we(X_train, y_train, X_test=X_test, y_test=y_test)
    else:
        exp.training_loop(X_train, y_train, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()