"""SHAP Explainability Analysis for EPO XGBoost Models (Experiment 2).

Two independent analyses, each runnable via CLI:

  Part A — Cross-mode SHAP comparison (pf / op / both)
  -----------------------------------------------------
  Trains the best-tuned XGBoost model for each case mode on the Exp 2
  train set, computes global mean-|SHAP| importance on the test set,
  and produces:
    1. A grouped bar chart of the top-N overlapping features across modes.
    2. A Spearman rank-correlation heatmap of the feature importance
       rankings between modes.
  Output: Results/shap/cross_mode_bar.png
           Results/shap/cross_mode_spearman.png

  Part B — Sliding-window SHAP trajectory (pf only)
  --------------------------------------------------
  Re-runs Exp 2 pf XGBoost on each sliding window (train ≤ T-1, test = T,
  T ∈ 2021..2024), computes global mean-|SHAP| importance per window, and
  produces:
    1. A heatmap of feature importance over time (top-N features by mean
       importance across all windows).
    2. A line plot of the top-K individual feature trajectories.
  Output: Results/shap/sw_heatmap.png
           Results/shap/sw_lineplot.png

Usage
-----
  # Part A only:
  python -m Experiments.shap_analysis cross_mode [--top_n 20] [--out_dir Results/shap]

  # Part B only:
  python -m Experiments.shap_analysis sliding_window [--top_n 20] [--top_k 8]
                                                      [--first_year 2021]
                                                      [--last_year 2024]
                                                      [--out_dir Results/shap]

  # Both:
  python -m Experiments.shap_analysis all [options]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# numpy 2.0 removed np.float_; cupy (installed as spacy[cuda12x] dep) still
# references it, raising AttributeError before spacy's own try/except can catch
# it. Restore the alias so the import resolves cleanly on CPU-only nodes.
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

# cupy (spacy[cuda12x] dependency) is incompatible with numpy ≥ 2.0 and raises
# AttributeError during init — spacy's own try/except only catches ImportError.
# Stub out the entire cupy package so spacy falls back to its CPU path cleanly.
import sys, types
_cupy_stub = types.ModuleType("cupy")
_cupy_stub.ndarray = np.ndarray
for _sub in ["cuda", "cuda.stream", "testing", "_core", "_core.core"]:
    sys.modules[f"cupy.{_sub}"] = types.ModuleType(f"cupy.{_sub}")
sys.modules["cupy"] = _cupy_stub

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

# ── project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from Utilities.utils import TextProcess
from Experiments.ml_experiments import PreprocessTransformer

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "Data" / "Final_Processed"
FULL_PF_DS = PROJECT_ROOT / "Data" / "Train&TestData_1.0_PatentRefusal.pkl"
FULL_OP_DS = PROJECT_ROOT / "Data" / "Train&TestData_1.0_OppositionDivision.pkl"

RESULTS_JSON = PROJECT_ROOT / "Results" / "results_main.json"


def _load_best_params() -> dict:
    """Read best XGBoost (Exp 2) hyperparameters per case mode from results_main.json.

    Selects the record with highest test F1 for algo=XGBClassifier, experiment=2
    per case mode, then maps sklearn pipeline param names to the flat dict format
    expected by _build_pipeline().
    """
    with open(RESULTS_JSON, encoding="utf-8") as f:
        records = json.load(f)

    # pf uses 'vect__<key>', op/both use 'vect__num__<key>'
    vect_prefixes = {"pf": "vect__", "op": "vect__num__", "both": "vect__num__"}
    vect_keys = ("use_idf", "norm", "ngram_range", "min_df")
    clf_keys  = ("n_estimators", "max_depth", "learning_rate", "gamma")
    prep_keys = ("stopwords", "numbers", "lemmatisation")

    best_params = {}
    for case in ("pf", "op", "both"):
        candidates = [
            r for r in records
            if r.get("algo") == "XGBClassifier"
            and str(r.get("experiment")) == "2"
            and r.get("case_mode") == case
            and r.get("test_metrics") is not None
        ]
        if not candidates:
            raise ValueError(
                f"No XGBClassifier Exp 2 records found for case='{case}' in {RESULTS_JSON}. "
                "Run Step 4 (ML experiments) first."
            )
        best = max(candidates, key=lambda r: r["test_metrics"].get("f1", 0))
        raw = best["best_params"]
        prefix = vect_prefixes[case]

        params = {}
        for k in vect_keys:
            params[k] = raw[f"{prefix}{k}"]
        for k in clf_keys:
            params[k] = raw[f"clf__{k}"]
        for k in prep_keys:
            params[k] = raw[f"prep__{k}"]
        # ngram_range comes out of JSON as a list; pipeline expects a tuple
        params["ngram_range"] = tuple(params["ngram_range"])

        best_params[case] = params
        print(
            f"[BEST_PARAMS] {case}: F1={best['test_metrics']['f1']:.4f}  "
            f"input={best.get('input_representation')}  "
            f"n_estimators={params['n_estimators']}  lr={params['learning_rate']}  "
            f"ngram={params['ngram_range']}  depth={params['max_depth']}"
        )

    return best_params

# Loaded once at import time so all functions share the same params object.
BEST_PARAMS: dict = {}  # populated in main() / each entry-point to avoid slow
                        # JSON load on --help; call _ensure_params() before use.

def _ensure_params():
    """Lazily populate BEST_PARAMS from results_main.json on first use."""
    if not BEST_PARAMS:
        BEST_PARAMS.update(_load_best_params())

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline(case: str, params: dict) -> Pipeline:
    """Build an sklearn Pipeline for the given case mode with fixed HPs."""
    p = params
    tfidf = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        use_idf=p["use_idf"],
        norm=p["norm"],
        ngram_range=tuple(p["ngram_range"]),
        min_df=p["min_df"],
    )
    clf = xgb.XGBClassifier(
        random_state=42,
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
    )

    if case in ("op", "both"):
        # Opposition / combined modes have auxiliary one-hot columns alongside
        # the text column; these pass through a ColumnTransformer.
        preprocessor = ColumnTransformer(
            transformers=[("num", tfidf, "New Summary Facts"),
                          ("cats", Binarizer(), _aux_cols_for(case))],
        )
        return Pipeline([
            ("prep", PreprocessTransformer(
                stopwords=p["stopwords"],
                numbers=p["numbers"],
                lemmatisation=p["lemmatisation"],
                opposition=True,
            )),
            ("vect", preprocessor),
            ("clf", clf),
        ])
    else:
        return Pipeline([
            ("prep", PreprocessTransformer(
                stopwords=p["stopwords"],
                numbers=p["numbers"],
                lemmatisation=p["lemmatisation"],
                opposition=False,
            )),
            ("vect", tfidf),
            ("clf", clf),
        ])


def _aux_cols_for(case: str) -> list:
    """Return the auxiliary (non-text) column names for a given case mode."""
    if case == "op":
        return ["Matches_1", "Matches_2", "Matches_3"]
    if case == "both":
        return [
            "Category_Opposition Division", "Category_Patent Refusal",
            "OD_Match_0", "OD_Match_1", "OD_Match_2",
        ]
    return []


def _load_exp2(case: str):
    """Load Exp 2 train/test pickles and return (X_train, y_train, X_test, y_test)."""
    X_train = pd.read_pickle(DATA_DIR / f"X_Train_2_{case}.pkl")
    y_train = pd.read_pickle(DATA_DIR / f"y_Train_2_{case}.pkl").iloc[:, 0].to_numpy()
    X_test  = pd.read_pickle(DATA_DIR / f"X_test_2_{case}.pkl")
    y_test  = pd.read_pickle(DATA_DIR / f"y_test_2_{case}.pkl").iloc[:, 0].to_numpy()
    return X_train, y_train, X_test, y_test


def _fit_and_transform(case: str, X_train, y_train, X_test):
    """Fit pipeline, return (clf, X_train_matrix, X_test_matrix, feature_names)."""
    # Pre-parse raw text → spaCy Docs, mirroring Experiments.text_preprocess.
    # PreprocessTransformer detects Docs (not strings) and skips nlp.pipe internally.
    tp = TextProcess()
    if case in ("op", "both"):
        X_tr = X_train.copy()
        X_tr["New Summary Facts"] = list(tp.nlp.pipe(X_train["New Summary Facts"].tolist()))
        X_te = X_test.copy()
        X_te["New Summary Facts"] = list(tp.nlp.pipe(X_test["New Summary Facts"].tolist()))
    else:
        X_tr = list(tp.nlp.pipe(X_train["New Summary Facts"].tolist()))
        X_te = list(tp.nlp.pipe(X_test["New Summary Facts"].tolist()))

    pipe = _build_pipeline(case, BEST_PARAMS[case])
    pipe.fit(X_tr, y_train)

    prep = pipe.named_steps["prep"]
    vect = pipe.named_steps["vect"]
    clf  = pipe.named_steps["clf"]

    X_tr_mat = vect.transform(prep.transform(X_tr))
    X_te_mat = vect.transform(prep.transform(X_te))

    if case in ("op", "both"):
        feat_names = (
            list(vect.named_transformers_["num"].get_feature_names_out())
            + _aux_cols_for(case)
        )
    else:
        feat_names = list(vect.get_feature_names_out())

    return clf, X_tr_mat, X_te_mat, feat_names


def _compute_shap(clf, X_matrix, feat_names: list):
    """Return (explainer, shap_values Explanation, mean_abs_Series).

    Computes TreeExplainer SHAP values and returns the full Explanation object
    alongside the mean |SHAP| importance Series used for global ranking.
    """
    explainer   = shap.TreeExplainer(clf, feature_names=feat_names)
    shap_values = explainer(X_matrix)
    # shap_values.values shape: (n_samples, n_features) for binary XGB
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return explainer, shap_values, pd.Series(mean_abs, index=feat_names)


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Cross-mode comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_mode(top_n: int, out_dir: Path):
    """Compute global SHAP importance for pf/op/both and produce comparison plots."""
    _ensure_params()
    out_dir.mkdir(parents=True, exist_ok=True)
    importance = {}

    shap_cache: dict = {}
    for case in ("pf", "op", "both"):
        print(f"\n[Cross-mode] Processing case='{case}'...")
        X_train, y_train, X_test, y_test = _load_exp2(case)
        clf, _, X_te_mat, feat_names = _fit_and_transform(case, X_train, y_train, X_test)
        explainer_c, shap_vals_c, importance[case] = _compute_shap(clf, X_te_mat, feat_names)
        shap_cache[case] = (explainer_c, shap_vals_c, X_te_mat, y_test, feat_names, clf)
        print(f"  → {len(feat_names)} features, top feature: {importance[case].idxmax()} "
              f"({importance[case].max():.4f})")

    # ── 1. Identify shared top-N features across all three modes ─────────────
    # Union of top-N from each mode, then rank by mean importance
    top_per_mode = {c: set(s.nlargest(top_n).index) for c, s in importance.items()}
    union_features = top_per_mode["pf"] | top_per_mode["op"] | top_per_mode["both"]

    mean_across = pd.concat(
        [importance[c].reindex(union_features, fill_value=0) for c in ("pf", "op", "both")],
        axis=1, keys=("pf", "op", "both")
    ).mean(axis=1).sort_values(ascending=False)

    plot_features = mean_across.head(top_n).index.tolist()

    df_plot = pd.DataFrame(
        {c: importance[c].reindex(plot_features, fill_value=0) for c in ("pf", "op", "both")},
        index=plot_features,
    )

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(plot_features))
    w = 0.26
    colors = {"pf": "#2196F3", "op": "#FF9800", "both": "#4CAF50"}
    labels = {"pf": "Patent Refusal", "op": "Opposition Division", "both": "Combined"}
    for i, case in enumerate(("pf", "op", "both")):
        vals = df_plot[case].values
        # Normalise each mode to [0,1] for comparability across different SHAP scales
        norm_vals = vals / (vals.max() + 1e-12)
        ax.bar(x + i * w, norm_vals, width=w, label=labels[case],
               color=colors[case], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + w)
    ax.set_xticklabels(plot_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised mean |SHAP| (per-mode max = 1)", fontsize=11)
    ax.set_title(f"Global SHAP Feature Importance — Top {top_n} features across modes\n"
                 f"XGBoost, Experiment 2 (Temporal Split), test set", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    bar_path = out_dir / "cross_mode_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Cross-mode] Bar chart saved → {bar_path}")

    # ── 2. Spearman rank correlation ─────────────────────────────────────────
    # Use the full importance series (all shared vocabulary) for a meaningful rank
    all_features = list(
        set(importance["pf"].index) | set(importance["op"].index) | set(importance["both"].index)
    )
    rank_df = pd.DataFrame(
        {c: importance[c].reindex(all_features, fill_value=0) for c in ("pf", "op", "both")},
        index=all_features,
    )
    # Rank each column (higher importance = lower rank number)
    ranked = rank_df.rank(ascending=False)

    corr_matrix = np.zeros((3, 3))
    p_matrix    = np.zeros((3, 3))
    cases = ["pf", "op", "both"]
    for i, c1 in enumerate(cases):
        for j, c2 in enumerate(cases):
            rho, p = spearmanr(ranked[c1], ranked[c2])
            corr_matrix[i, j] = rho
            p_matrix[i, j]    = p

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Patent Refusal\n(pf)", "Opposition Division\n(op)", "Combined\n(both)"], fontsize=10)
    ax.set_yticklabels(["Patent Refusal\n(pf)", "Opposition Division\n(op)", "Combined\n(both)"], fontsize=10)
    for i in range(3):
        for j in range(3):
            p_str = f"\np={p_matrix[i,j]:.3f}" if i != j else ""
            ax.text(j, i, f"ρ={corr_matrix[i,j]:.3f}{p_str}",
                    ha="center", va="center", fontsize=8.5,
                    color="black" if abs(corr_matrix[i,j]) < 0.85 else "white")
    ax.set_title("Spearman Rank Correlation\nof SHAP Feature Importance Rankings", fontsize=11)
    plt.tight_layout()
    spear_path = out_dir / "cross_mode_spearman.png"
    fig.savefig(spear_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Cross-mode] Spearman heatmap saved → {spear_path}")

    # Print correlation summary
    print("\n[Cross-mode] Spearman ρ summary:")
    for i, c1 in enumerate(cases):
        for j, c2 in enumerate(cases):
            if j > i:
                print(f"  {c1} vs {c2}: ρ={corr_matrix[i,j]:.4f}  p={p_matrix[i,j]:.4f}")

    # ── 3. Auxiliary-feature SHAP analysis (op / both only) ──────────────────
    # These features encode who is bringing the case (appellant type / case
    # category) and are the structured counterpart to the text TF-IDF features.
    #
    # op   → Matches_1 (patentee appellant), Matches_2 (opponent appellant),
    #         Matches_3 (both parties appealing)
    # both → Category_Opposition Division, Category_Patent Refusal,
    #         OD_Match_0 (PR case), OD_Match_1 (patentee OD), OD_Match_2 (opponent OD)
    _aux_labels = {
        "Matches_1":                    "Patentee appellant",
        "Matches_2":                    "Opponent appellant",
        "Matches_3":                    "Both parties appealing",
        "Category_Opposition Division": "Case type: Opposition Division",
        "Category_Patent Refusal":      "Case type: Patent Refusal",
        "OD_Match_0":                   "OD sub-type: PR case",
        "OD_Match_1":                   "OD sub-type: patentee OD",
        "OD_Match_2":                   "OD sub-type: opponent OD",
    }

    for case in ("op", "both"):
        explainer_c, shap_vals_c, X_te_mat_c, y_test_c, feat_names_c, clf_c = shap_cache[case]
        aux_cols = _aux_cols_for(case)
        aux_idx  = [feat_names_c.index(c) for c in aux_cols if c in feat_names_c]
        if not aux_idx:
            print(f"\n[Cross-mode] No auxiliary features found for case='{case}', skipping.")
            continue

        mode_label = {"op": "Opposition Division", "both": "Combined"}[case]
        print(f"\n[Cross-mode] Auxiliary-feature SHAP for case='{case}' ({mode_label})...")

        X_arr_full = pd.DataFrame(
            X_te_mat_c.toarray() if hasattr(X_te_mat_c, "toarray") else np.asarray(X_te_mat_c),
            columns=feat_names_c,
        )
        # Slice to auxiliary columns only
        aux_feat_names = [feat_names_c[i] for i in aux_idx]
        X_aux  = X_arr_full[aux_feat_names]
        sv_aux = shap_vals_c.values[:, aux_idx]   # (n_samples, n_aux_feats)

        # Pretty labels for axis ticks
        pretty_names = [_aux_labels.get(f, f) for f in aux_feat_names]

        # -- 1. Bar chart of mean |SHAP| for auxiliary features ---------------
        mean_abs_aux = np.abs(sv_aux).mean(axis=0)
        order = np.argsort(mean_abs_aux)
        fig, ax = plt.subplots(figsize=(7, max(3, len(aux_feat_names) * 0.55)))
        bars = ax.barh(
            [pretty_names[i] for i in order],
            mean_abs_aux[order],
            color="#FF9800" if case == "op" else "#4CAF50",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
        ax.set_title(
            f"Appellant / Case-Type Feature SHAP Importance\n{mode_label}, XGBoost Exp 2",
            fontsize=12,
        )
        plt.tight_layout()
        bar_aux_path = out_dir / f"{case}_aux_shap_bar.png"
        fig.savefig(bar_aux_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Aux bar chart saved → {bar_aux_path}")

        # -- 2. Beeswarm (dot summary) for auxiliary features only ------------
        shap_aux_obj = shap.Explanation(
            values=sv_aux,
            base_values=shap_vals_c.base_values,
            data=X_aux.values,
            feature_names=pretty_names,
        )
        shap.summary_plot(
            shap_aux_obj,
            features=X_aux,
            feature_names=pretty_names,
            plot_type="dot",
            show=False,
            color_bar=True,
            max_display=len(aux_feat_names),
        )
        plt.title(
            f"Appellant / Case-Type Feature SHAP (beeswarm)\n{mode_label}, XGBoost Exp 2",
            fontsize=11,
        )
        bee_aux_path = out_dir / f"{case}_aux_shap_beeswarm.png"
        plt.savefig(bee_aux_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  → Aux beeswarm saved → {bee_aux_path}")

        # Print summary stats
        for fi, fname in enumerate(aux_feat_names):
            sv_col = sv_aux[:, fi]
            print(f"  {_aux_labels.get(fname, fname):45s}  "
                  f"mean|SHAP|={np.abs(sv_col).mean():.4f}  "
                  f"mean SHAP={sv_col.mean():+.4f}  "
                  f"(+→Affirmed)")

    # ── 4. Per-case beeswarm + local force plots ──────────────────────────────
    _mode_labels = {
        "pf": "Patent Refusal",
        "op": "Opposition Division",
        "both": "Combined",
    }
    for case in ("pf", "op", "both"):
        explainer_c, shap_vals_c, X_te_mat_c, y_test_c, feat_names_c, clf_c = shap_cache[case]
        mode_label = _mode_labels[case]
        X_arr = pd.DataFrame(
            X_te_mat_c.toarray() if hasattr(X_te_mat_c, "toarray") else np.asarray(X_te_mat_c),
            columns=feat_names_c,
        )

        # -- Beeswarm (dot summary plot, mirrors original shap_xgboost_run.py) --
        print(f"\n[Cross-mode] Beeswarm for case='{case}' ({mode_label})...")
        shap.summary_plot(
            shap_vals_c,
            features=X_arr,
            feature_names=feat_names_c,
            plot_type="dot",
            show=False,
            color_bar=True,
            max_display=20,
        )
        bee_path = out_dir / f"{case}_beeswarm.png"
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  → Beeswarm saved → {bee_path}")

        # -- Local force plots (HTML) --
        np.random.seed(42)
        y_pred_c = clf_c.predict(X_te_mat_c)
        base_val = explainer_c.expected_value
        if hasattr(base_val, "__len__"):
            base_val = float(base_val[0])

        scenarios = {
            "correct_reversed": np.where((y_test_c == y_pred_c) & (y_test_c == 0))[0],
            "correct_affirmed": np.where((y_test_c == y_pred_c) & (y_test_c == 1))[0],
            "incorrect":        np.where(y_test_c != y_pred_c)[0],
        }
        for label, idx_arr in scenarios.items():
            if len(idx_arr) == 0:
                print(f"  → No '{label}' samples for case={case}, skipping force plot.")
                continue
            i = int(np.random.choice(idx_arr))
            fp = shap.force_plot(
                base_value=base_val,
                shap_values=shap_vals_c.values[i],
                features=X_arr.iloc[i],
                feature_names=feat_names_c,
                show=False,
            )
            html_path = out_dir / f"{case}_force_{label}.html"
            shap.save_html(str(html_path), fp)
            print(f"  → Force plot ({label}) saved → {html_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Sliding-window SHAP trajectory (pf only)
# ─────────────────────────────────────────────────────────────────────────────

def _build_window_split(df: pd.DataFrame, train_end: int, test_year: int):
    """Return (X_train, y_train, X_test, y_test) for a sliding window on the pf full dataset."""
    train_mask = df["Year"] <= train_end
    test_mask  = df["Year"] == test_year

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    # Balance train set (match minority class size)
    n_aff = (train_df["Outcome"] == "Affirmed").sum()
    n_rev = (train_df["Outcome"] == "Reversed").sum()
    if n_aff != n_rev:
        majority = "Affirmed" if n_aff > n_rev else "Reversed"
        minority_n = min(n_aff, n_rev)
        drop_idx = (train_df[train_df["Outcome"] == majority]
                    .sample(n=len(train_df[train_df["Outcome"] == majority]) - minority_n,
                            random_state=42).index)
        train_df = train_df.drop(drop_idx)

    X_train = train_df[["New Summary Facts"]].reset_index(drop=True)
    y_train = (train_df["Outcome"] == "Affirmed").astype(int).to_numpy()
    X_test  = test_df[["New Summary Facts"]].reset_index(drop=True)
    y_test  = (test_df["Outcome"] == "Affirmed").astype(int).to_numpy()

    return X_train, y_train, X_test, y_test


def run_sliding_window(top_n: int, top_k: int, first_year: int, last_year: int, out_dir: Path):
    """Compute per-window global SHAP importance for pf XGBoost and produce trajectory plots."""
    _ensure_params()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[SW SHAP] Loading full pf dataset...")
    df = pd.read_pickle(FULL_PF_DS)

    windows = list(range(first_year, last_year + 1))
    window_importance: dict[int, pd.Series] = {}

    case = "pf"

    for test_year in windows:
        train_end = test_year - 1
        print(f"\n[SW SHAP] Window: train ≤ {train_end}, test = {test_year}")
        X_train, y_train, X_test, y_test = _build_window_split(df, train_end, test_year)

        if len(X_test) == 0:
            print(f"  → No test samples for year {test_year}, skipping.")
            continue

        print(f"  train={len(X_train)}, test={len(X_test)}")
        clf, _, X_test_mat, feat_names = _fit_and_transform(case, X_train, y_train, X_test)
        _, _, importance = _compute_shap(clf, X_test_mat, feat_names)
        window_importance[test_year] = importance
        print(f"  → top feature: {importance.idxmax()} ({importance.max():.4f})")

    if not window_importance:
        print("[SW SHAP] No windows completed.")
        return

    # ── Combine into a DataFrame ──────────────────────────────────────────────
    all_features = sorted(
        set().union(*[s.index for s in window_importance.values()])
    )
    imp_df = pd.DataFrame(
        {yr: window_importance[yr].reindex(all_features, fill_value=0)
         for yr in sorted(window_importance)},
        index=all_features,
    )  # shape: (n_features, n_windows)

    # Select top-N features by mean importance across windows
    mean_imp = imp_df.mean(axis=1)
    top_features = mean_imp.nlargest(top_n).index.tolist()
    imp_top = imp_df.loc[top_features]

    # ── 1. Heatmap ────────────────────────────────────────────────────────────
    # Normalise each feature row to [0,1] so visual scale is comparable
    imp_norm = imp_top.div(imp_top.max(axis=1) + 1e-12, axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(windows) * 1.2), max(8, top_n * 0.42)))
    im = ax.imshow(imp_norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised mean |SHAP| (per-feature max = 1)")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([str(y) for y in sorted(window_importance)], fontsize=11)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_title(f"SHAP Feature Importance Trajectory — Patent Refusal (pf)\n"
                 f"XGBoost, Exp 2 Sliding Window, Top {top_n} features", fontsize=12)
    plt.tight_layout()
    hmap_path = out_dir / "sw_heatmap.png"
    fig.savefig(hmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SW SHAP] Heatmap saved → {hmap_path}")

    # ── 2. Line plot — top-K feature trajectories ─────────────────────────────
    top_k_features = mean_imp.nlargest(top_k).index.tolist()
    imp_topk = imp_df.loc[top_k_features]

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_years = sorted(window_importance.keys())
    for idx, feat in enumerate(top_k_features):
        vals = [imp_df.loc[feat, yr] for yr in sorted_years]
        ax.plot(sorted_years, vals, marker="o", linewidth=2,
                label=feat, color=cmap(idx % 10))

    ax.set_xticks(sorted_years)
    ax.set_xticklabels([str(y) for y in sorted_years], fontsize=11)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_ylabel("Mean |SHAP| importance", fontsize=11)
    ax.set_title(f"SHAP Feature Importance Over Time — Top {top_k} Features\n"
                 f"Patent Refusal (pf), XGBoost, Exp 2 Sliding Window", fontsize=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    line_path = out_dir / "sw_lineplot.png"
    fig.savefig(line_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SW SHAP] Line plot saved → {line_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP explainability analysis for EPO XGBoost (Exp 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["cross_mode", "sliding_window", "all"],
        help="Which analysis to run.",
    )
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="Number of top features to display in plots (default: 20).",
    )
    parser.add_argument(
        "--top_k", type=int, default=8,
        help="Number of features in the SW line plot (default: 8).",
    )
    parser.add_argument(
        "--first_year", type=int, default=2021,
        help="First sliding-window test year (default: 2021).",
    )
    parser.add_argument(
        "--last_year", type=int, default=2024,
        help="Last sliding-window test year (default: 2024).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(PROJECT_ROOT / "Results" / "shap"),
        help="Output directory for plots (default: Results/shap).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)

    if args.mode in ("cross_mode", "all"):
        run_cross_mode(top_n=args.top_n, out_dir=out_dir)

    if args.mode in ("sliding_window", "all"):
        run_sliding_window(
            top_n=args.top_n,
            top_k=args.top_k,
            first_year=args.first_year,
            last_year=args.last_year,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
