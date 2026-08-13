"""Build the per-issue prediction cache used by the results notebook.

WHAT IT DOES
    Regenerates per-example FULL-TEST predictions for the best ML (XGBoost) and
    best DL (Longformer) models, joins each test case to its patentability issue
    (`Matched Articles` in EPO_Data.pkl), and caches everything to disk. Each
    model is re-fit / re-loaded and its predictions are validated against the
    stored aggregate metrics (results_main.json / results_dl_full_test.json), so
    the reconstruction is known to be faithful.

WHEN TO RUN IT
    Run this once to (re)create `Results/per_issue_predictions.pkl` before the
    per-issue analysis section of `Results/results.ipynb` (per-issue confusion
    matrices, significance tests, and the pooled combined-case F1). Re-run it
    only when the underlying best models or their saved checkpoints change; the
    notebook otherwise just loads the cached pickle. The Longformer inference
    needs a GPU, so submit it via `run_regen_per_issue.sh` (SLURM). The run is
    resumable: existing combos in the pickle are skipped and it saves after
    every combo, so an interrupted job can be re-submitted to finish the rest.

SCOPE
  - Full-test only (Exp 1 & 2), all three case modes (pf / op / both).
  - Best ML model  = XGBoost (best sparse input representation per case x exp).
  - Best DL model  = Longformer (loaded from the saved HP checkpoints).
  - Issue buckets (4 disjoint): IS-only, Novelty-only, IS+Novelty, Other.

OUTPUT
    Results/per_issue_predictions.pkl  (long format, one row per test case)
    columns: model, case, exp, opposition, input_representation,
             y_true, y_pred, y_score, issue
"""
import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the evaluation module: reuses its exact ML pipeline reconstruction and
# sets up the numpy/cupy shims + project sys.path.
from Experiments import evaluation as ev
from Experiments.deep_learning_experiments import DeepLearningExperiments

BASE = PROJECT_ROOT / "Results"
DATA_DIR = PROJECT_ROOT / "Data" / "Final_Processed"
EPO_PKL = PROJECT_ROOT / "Data" / "EPO_Data.pkl"
OUT_PATH = BASE / "per_issue_predictions.pkl"


def _save(frames):
    """Persist all frames collected so far (incremental, crash-safe)."""
    if frames:
        pd.concat(frames, ignore_index=True).to_pickle(OUT_PATH)

CASES = [
    ("pf",   False),
    ("op",   True),
    ("both", True),
]
EXPS = ["1", "2"]


# ---------------------------------------------------------------------------
# Issue bucketing
# ---------------------------------------------------------------------------
def bucket_issue(articles):
    """Map a `Matched Articles` list to one of 4 disjoint buckets."""
    s = set(articles) if isinstance(articles, (list, tuple)) else {articles}
    has_is = "Inventive Step" in s
    has_nov = "Novelty" in s
    if has_is and has_nov:
        return "IS+Novelty"
    if has_is:
        return "IS-only"
    if has_nov:
        return "Novelty-only"
    return "Other"


def build_issue_map():
    """New Summary Facts -> issue bucket, from the raw EPO data."""
    d = pd.read_pickle(EPO_PKL)[["New Summary Facts", "Matched Articles"]].copy()
    d = d.drop_duplicates("New Summary Facts", keep="first")
    d["issue"] = d["Matched Articles"].apply(bucket_issue)
    return dict(zip(d["New Summary Facts"], d["issue"]))


# ---------------------------------------------------------------------------
# ML (XGBoost) — reuse evaluation._ml_fit_and_predict
# ---------------------------------------------------------------------------
def best_xgb_record(experiment, case_mode):
    """Pick the XGBoost record (from results_main) with the highest stored
    full-test F1 for this experiment x case."""
    with open(BASE / "results_main.json") as f:
        main = json.load(f)
    cands = [
        r for r in main
        if r.get("algo") == "XGBClassifier"
        and str(r.get("experiment")) == experiment
        and str(r.get("case_mode")) == case_mode
        and r.get("test_metrics")
    ]
    if not cands:
        return None
    return max(cands, key=lambda r: r["test_metrics"].get("f1", -1))


def run_xgb(experiment, case_mode, opposition, issue_map):
    rec = best_xgb_record(experiment, case_mode)
    if rec is None:
        print(f"  [XGB] no record for exp{experiment} {case_mode}")
        return None

    X_train = pd.read_pickle(DATA_DIR / f"X_Train_{experiment}_{case_mode}.pkl")
    X_test = pd.read_pickle(DATA_DIR / f"X_test_{experiment}_{case_mode}.pkl")
    y_train = np.array(pd.read_pickle(DATA_DIR / f"y_Train_{experiment}_{case_mode}.pkl")).ravel()
    y_test = np.array(pd.read_pickle(DATA_DIR / f"y_test_{experiment}_{case_mode}.pkl")).ravel()

    tp = ev.TextProcess()
    _, _, _, y_pred, y_score = ev._ml_fit_and_predict(
        rec, opposition, X_train, y_train, X_test, y_test, tp=tp,
    )
    y_pred = np.asarray(y_pred).ravel()

    # Validate against stored metrics
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    stored = rec["test_metrics"]
    print(f"  [XGB] exp{experiment} {case_mode} ({rec['input_representation']}): "
          f"F1 regen={f1:.4f} stored={stored['f1']:.4f}  "
          f"Acc regen={acc:.4f} stored={stored['accuracy']:.4f}")

    issues = X_test["New Summary Facts"].map(issue_map).values
    return pd.DataFrame({
        "model": "XGBoost",
        "case": case_mode,
        "exp": experiment,
        "opposition": opposition,
        "input_representation": rec["input_representation"],
        "y_true": y_test.astype(int),
        "y_pred": y_pred.astype(int),
        "y_score": np.asarray(y_score).ravel() if y_score is not None else np.nan,
        "issue": issues,
    })


# ---------------------------------------------------------------------------
# DL (Longformer) — load saved HP checkpoint, run inference
# ---------------------------------------------------------------------------
def longformer_record(experiment, case_mode):
    with open(BASE / "results_deep_learning.json") as f:
        hp = json.load(f)
    cands = [
        r for r in hp
        if r.get("algo") == "longformer_base"
        and str(r.get("experiment")) == experiment
        and str(r.get("case_mode")) == case_mode
    ]
    if not cands:
        return None
    # prefer one with an existing saved model
    for r in cands:
        p = r.get("best_model_path")
        if p and os.path.exists(p):
            return r
    return cands[0]


@torch.no_grad()
def _dl_predict(dl, model, X_test_proc, y_test, opposition):
    """Per-example inference, mirroring DeepLearningExperiments._compute_test_metrics."""
    loader = dl._create_dataloader(X_test_proc, y_test, is_train=False, batch_size=16)
    model.eval()
    preds, scores, labels = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(dl.device)
        attention_mask = batch["attention_mask"].to(dl.device)
        lab = batch["labels"].to(dl.device)
        if opposition:
            outputs = dl._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
            text_embedding = outputs.pooler_output if outputs.pooler_output is not None \
                else outputs.last_hidden_state[:, 0, :]
            aux = batch["auxiliary_features"].to(dl.device)
            logits = dl.custom_head(text_embedding, aux)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        p = torch.argmax(logits, dim=1)
        pr = torch.softmax(logits.float(), dim=1)[:, 1]
        preds.extend(p.cpu().numpy())
        scores.extend(pr.cpu().numpy())
        labels.extend(lab.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(scores)


def run_longformer(experiment, case_mode, opposition, issue_map, device):
    rec = longformer_record(experiment, case_mode)
    if rec is None:
        print(f"  [LF] no record for exp{experiment} {case_mode}")
        return None
    model_path = rec.get("best_model_path")
    if not (model_path and os.path.exists(model_path)):
        print(f"  [LF] missing checkpoint for exp{experiment} {case_mode}: {model_path}")
        return None

    params = rec.get("best_params", {})
    batch_size = int(params.get("batch_size", 32))
    dropout = float(params.get("dropout", 0.1))

    X_train = pd.read_pickle(DATA_DIR / f"X_Train_{experiment}_{case_mode}.pkl")
    X_test = pd.read_pickle(DATA_DIR / f"X_test_{experiment}_{case_mode}.pkl")
    y_train = np.array(pd.read_pickle(DATA_DIR / f"y_Train_{experiment}_{case_mode}.pkl")).ravel()
    y_test = np.array(pd.read_pickle(DATA_DIR / f"y_test_{experiment}_{case_mode}.pkl")).ravel()

    dl = DeepLearningExperiments(
        model_name="longformer_base",
        experiment=experiment,
        opposition=opposition,
        case_mode=case_mode,
        device=device,
        results_json_path="/dev/null",
    )
    X_train_proc = dl._preprocess_text_for_bert(X_train.reset_index(drop=True))
    X_test_proc = dl._preprocess_text_for_bert(X_test.reset_index(drop=True))
    dl.aux_encoder = None

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    model = dl._build_model(dropout)
    if opposition:
        # fit aux encoder + size opposition head using a train probe loader
        probe = dl._create_dataloader(X_train_proc, y_train, is_train=True, batch_size=batch_size)
        dl._fix_opposition_head(model, probe)
        del probe
    payload = torch.load(model_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    if opposition and payload.get("custom_head_state") is not None:
        dl.custom_head.load_state_dict(payload["custom_head_state"])
    model.to(device)

    y_true, y_pred, y_score = _dl_predict(dl, model, X_test_proc, y_test, opposition)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    # stored full-test metric for comparison
    with open(BASE / "results_dl_full_test.json") as f:
        ft = json.load(f)
    stored = next((r["test_metrics"] for r in ft
                   if r.get("algo") == "longformer_base"
                   and str(r.get("experiment")) == experiment
                   and str(r.get("case_mode")) == case_mode), {})
    print(f"  [LF] exp{experiment} {case_mode}: F1 regen={f1:.4f} "
          f"stored={stored.get('f1', float('nan')):.4f}  "
          f"Acc regen={acc:.4f} stored={stored.get('accuracy', float('nan')):.4f}")

    issues = X_test["New Summary Facts"].map(issue_map).values
    return pd.DataFrame({
        "model": "Longformer",
        "case": case_mode,
        "exp": experiment,
        "opposition": opposition,
        "input_representation": "transformer",
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        "y_score": y_score,
        "issue": issues,
    })


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    issue_map = build_issue_map()
    print(f"Issue map: {len(issue_map)} cases")

    # Resume: load any existing cache and skip combos already computed.
    frames = []
    done = set()
    if OUT_PATH.exists():
        prev = pd.read_pickle(OUT_PATH)
        frames.append(prev)
        done = set(map(tuple, prev[["model", "case", "exp"]].drop_duplicates().values))
        print(f"Resuming: {len(prev)} rows cached, {len(done)} combos already done: {sorted(done)}")

    print("\n=== XGBoost ===")
    for exp in EXPS:
        for case, opp in CASES:
            if ("XGBoost", case, exp) in done:
                print(f"  [skip] XGBoost {case} exp{exp} (cached)")
                continue
            df = run_xgb(exp, case, opp, issue_map)
            if df is not None:
                frames.append(df)
                _save(frames)  # incremental, crash-safe

    print("\n=== Longformer ===")
    for exp in EXPS:
        for case, opp in CASES:
            if ("Longformer", case, exp) in done:
                print(f"  [skip] Longformer {case} exp{exp} (cached)")
                continue
            df = run_longformer(exp, case, opp, issue_map, device)
            if df is not None:
                frames.append(df)
                _save(frames)  # incremental, crash-safe

    out = pd.concat(frames, ignore_index=True)
    n_unmapped = out["issue"].isna().sum()
    if n_unmapped:
        print(f"\n[Warning] {n_unmapped} test cases could not be mapped to an issue.")
    out.to_pickle(OUT_PATH)
    print(f"\nSaved {len(out)} rows -> {OUT_PATH}")
    print(out.groupby(["model", "issue"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
