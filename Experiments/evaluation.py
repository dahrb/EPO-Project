"""Evaluation experiment

1. Provides results on the full test set 

2. Provides results on a sliding window retrain and evaluation:
    train on 2000-2020  -> test on 2021
    train on 2000-2021  -> test on 2022
    train on 2000-2022  -> test on 2023
    train on 2000-2023  -> test on 2024   

For each window the following evaluation is stored:
  (a) test_metrics  - model trained on full window-train, evaluated on test year
  (b) cv_train_metrics - held-out temporal validation metrics for BERT models only

Usage - run all combinations automatically
------------------------------------------
python sliding_window.py
        [--pf_dataset   Data/Train&TestData_1.0_PatentRefusal.pkl]
        [--op_dataset   Data/Train&TestData_1.0_OppositionDivision.pkl]
        [--experiments  1 2]
        [--opposition   both|true|false]
        [--ml_results   Results/results_main.json]
        [--dl_results   Results/results_deep_learning.json]
        [--output       Results/results_sliding_window.json]
        [--first_test   2021]
        [--last_test    2024]
        [--data_dir     Data/Final_Processed]
        [--skip_ml] [--skip_dl]
        [--skip_full_test] [--skip_sliding_window]

Last Updated: 08.06.26

Status: Done
"""

import argparse
import copy
import fcntl
import json
import os
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC
import xgboost as xgb

from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss


from Experiments.deep_learning_experiments import DeepLearningExperiments
from Experiments.ml_experiments import PreprocessTransformer
from Utilities.utils import TextProcess, Word2VecTransform, append_json_result, jsonable


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
        "auc":       None,
    }
    if y_score is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
    return metrics


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

def _build_window_split(df, train_end, test_year, opposition, combined=False):
    """Slice df into (X_train, y_train, X_test, y_test) for one window.

    Parameters
    ----------
    combined : bool
        If True the dataframe must already contain 'Category' and 'OD_Match'
        columns (pf+OD combined schema, matching the both pre-split pickles).
        feature_cols will be ['Category', 'OD_Match'] → dummies:
        Category_Opposition Division, Category_Patent Refusal,
        OD_Match_0, OD_Match_1, OD_Match_2, OD_Match_3.
    opposition : bool
        If True and combined is False, include 'Matches' dummies
        (Matches_1/2/3/Other) — used for the op case.
    """
    if combined and opposition:
        raise ValueError("combined and opposition cannot both be True.")

    df = df[df["Outcome"].isin(["Affirmed", "Reversed"])].copy()

    train_df = df[df["Year"] <= train_end].copy()
    test_df  = df[df["Year"] == test_year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    counts = train_df["Outcome"].value_counts()
    if {"Affirmed", "Reversed"}.issubset(counts.index):
        min_count = int(counts[["Affirmed", "Reversed"]].min())
        train_df = pd.concat(
            [
                train_df[train_df["Outcome"] == label].sample(n=min_count, random_state=42)
                for label in ["Affirmed", "Reversed"]
            ],
            ignore_index=True,
        )
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if combined:
        feature_cols = ["Category", "OD_Match"]
    elif opposition:
        feature_cols = ["Matches"]
    else:
        feature_cols = []

    X_train = _build_data(train_df, feature_cols).reset_index(drop=True)
    X_test  = _build_data(test_df,  feature_cols).reset_index(drop=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train     = np.array([1 if v == "Affirmed" else 0 for v in train_df["Outcome"]])
    y_test      = np.array([1 if v == "Affirmed" else 0 for v in test_df["Outcome"]])
    years_train = train_df["Year"].values

    return X_train, y_train, X_test, y_test, years_train

def _build_data(data,feature_cols=None):
        if not feature_cols:
            return data[["New Summary Facts"]].copy()
        X = data[["New Summary Facts"] + feature_cols].copy()
        dummies = pd.get_dummies(X[feature_cols], prefix=feature_cols).astype(int)
        return pd.concat([X.drop(columns=feature_cols), dummies], axis=1)

# -----------------------------------------------------------------------------
# Best-record selection
# -----------------------------------------------------------------------------

def _load_best_records(ml_results_path, dl_results_path, experiment, opposition, case_mode=None):
    """Load results JSONs and return one best record per unique combo."""
    records = []
    for path in [ml_results_path, dl_results_path]:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                print("[Warning] Could not parse %s, skipping." % path)
                continue
        records.extend(data if isinstance(data, list) else [data])

    records = [
        r for r in records
        if str(r.get("experiment", "")) == str(experiment)
        and bool(r.get("opposition", False)) == opposition
        and (case_mode is None or str(r.get("case_mode", "")) == str(case_mode))
    ]

    if not records:
        return []

    best = {}
    for r in records:
        key = (
            r.get("mode", ""),
            r.get("algo", ""),
            r.get("input_representation", ""),
            r.get("embedding", ""),
        )
        # Always prefer the most recent HP search run (latest timestamp).
        # If timestamps are identical, fall back to best val/cv score.
        ts    = r.get("timestamp_utc", "")
        score = r.get("best_score_cv") or r.get("best_score_val") or -1
        if key not in best:
            best[key] = r
        else:
            prev_ts    = best[key].get("timestamp_utc", "")
            prev_score = best[key].get("best_score_cv") or best[key].get("best_score_val") or -1
            if ts > prev_ts or (ts == prev_ts and score > prev_score):
                best[key] = r

    return list(best.values())


def _safe_load_json_array(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        return []


def _existing_eval_key(record):
    mode = record.get("mode", "")
    case_mode = record.get("case_mode")
    if not case_mode:
        case_mode = "pf" if not bool(record.get("opposition", False)) else "both"
    base = (
        mode,
        str(record.get("experiment", "")),
        str(case_mode),
        str(record.get("algo", "")),
        str(record.get("input_representation", "")),
        str(record.get("embedding") or ""),
    )
    if mode == "sliding_window_ml":
        return base + (str(record.get("test_year", "")),)
    if mode == "sliding_window_dl":
        return base + (str(record.get("test_year", "")),)
    return base

# -----------------------------------------------------------------------------
# ML estimator reconstruction
# -----------------------------------------------------------------------------

def _rebuild_ml_estimator(record, opposition):
    """Re-create the best sklearn Pipeline from a stored result record."""
    algo      = record.get("algo", "")
    params    = record.get("best_params", {})
    mode      = record.get("mode", "")
    embedding = record.get("embedding", None)

    if algo.startswith("Log"):
        clf = LogisticRegression(random_state=42)
    elif algo.startswith("Lin"):
        clf = LinearSVC(random_state=42)
    elif algo.startswith("Ran"):
        clf = RandomForestClassifier(random_state=42)
    else:
        clf = xgb.XGBClassifier(
            random_state=42, objective="binary:logistic",
            tree_method="hist", device="cpu",
        )

    clf_kwargs = {k[len("clf__"):]: v for k, v in params.items() if k.startswith("clf__")}
    clf.set_params(**clf_kwargs)

    num_prefix = "num__" if opposition else ""

    if mode == "embedding":
        scaler = StandardScaler()
        return Pipeline([("scal", scaler), ("clf", clf)]), embedding, mode

    # JSON deserialises tuples as lists; TfidfVectorizer requires actual tuples
    # for ngram_range etc., so convert any list values back to tuples.
    _TUPLE_PARAMS = {"ngram_range", "analyzer"}
    vect_kwargs = {}
    for k, v in params.items():
        if not k.startswith("vect__" + num_prefix):
            continue
        short_key = k[len("vect__" + num_prefix):]
        if isinstance(v, list) and (short_key in _TUPLE_PARAMS or short_key.endswith("_range")):
            v = tuple(v)
        vect_kwargs[short_key] = v
    prep_kwargs = {
        k[len("prep__"):]: v
        for k, v in params.items()
        if k.startswith("prep__")
    }
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, preprocessor=lambda x: x, **vect_kwargs,
    )
    if opposition:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", vectorizer, "New Summary Facts"),
                ("Cats", Binarizer(), []),
            ]
        )
        steps = Pipeline([
            ("prep", PreprocessTransformer(opposition=True, **prep_kwargs)),
            ("vect", preprocessor),
            ("clf", clf),
        ])
    else:
        steps = Pipeline([
            ("prep", PreprocessTransformer(opposition=False, **prep_kwargs)),
            ("vect", vectorizer),
            ("clf", clf),
        ])
    return steps, None, mode

def _ml_predict_pipeline(pipeline, X, opposition):
    y_pred  = pipeline.predict(X)
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                y_score = proba[:, 1]
        except Exception:
            pass
    elif hasattr(pipeline, "decision_function"):
        try:
            y_score = pipeline.decision_function(X)
        except Exception:
            pass
    return y_pred, y_score

def _ml_fit_and_predict(record, opposition, X_train, y_train, X_test, y_test, tp=None):
    """Fit a fresh pipeline on (X_train, y_train) and predict on X_test."""
    pipeline, embedding, mode = _rebuild_ml_estimator(record, opposition)
    if tp is None:
        tp = TextProcess()

    if mode == "embedding":
        if opposition:
            aux_cols = [c for c in X_train.columns if c != "New Summary Facts"]

            def _embed_opp(X_df):
                X_copy = X_df.copy()
                docs = list(tp.nlp.pipe(X_copy["New Summary Facts"].tolist()))
                token_lists = tp.fit_transform(docs)
                X_copy["New Summary Facts"] = token_lists
                ct = ColumnTransformer(transformers=[
                    ("num", Word2VecTransform(embedding=embedding, opposition=True), "New Summary Facts"),
                    ("Cats", Binarizer(), aux_cols),
                ])
                return ct.fit_transform(X_copy)

            X_tr_feat = _embed_opp(X_train)
            X_te_feat = _embed_opp(X_test)
        else:
            def _embed(X_df):
                docs = list(tp.nlp.pipe(X_df["New Summary Facts"].tolist()))
                token_lists = tp.fit_transform(docs)
                w2v  = Word2VecTransform(embedding=embedding)
                return w2v.fit_transform(token_lists)

            X_tr_feat = _embed(X_train)
            X_te_feat = _embed(X_test)

        pipeline.fit(X_tr_feat, y_train)
        y_pred, y_score = _ml_predict_pipeline(pipeline, X_te_feat, opposition)
        return pipeline, X_tr_feat, X_te_feat, y_pred, y_score

    # Sparse
    if opposition:
        aux_cols = [c for c in X_train.columns if c != "New Summary Facts"]
        ct = pipeline.named_steps["vect"]
        ct.transformers = [
            ("num", ct.transformers[0][1], "New Summary Facts"),
            ("Cats", Binarizer(), aux_cols),
        ]
        pipeline.fit(X_train, y_train)
        y_pred, y_score = _ml_predict_pipeline(pipeline, X_test, opposition)
        return pipeline, X_train, X_test, y_pred, y_score

    pipeline.fit(X_train["New Summary Facts"], y_train)
    y_pred, y_score = _ml_predict_pipeline(
        pipeline, X_test["New Summary Facts"], opposition
    )
    return pipeline, X_train["New Summary Facts"], X_test["New Summary Facts"], y_pred, y_score

# -----------------------------------------------------------------------------
# ML window evaluation
# -----------------------------------------------------------------------------

def _run_ml_window(record, df, train_end, test_year, opposition, dataset_label, case_mode, output_path, combined=False):
    """Train on full window-train set and evaluate on the test year."""
    # When combined=True the schema already encodes the case via Category/OD_Match;
    # passing opposition=True would conflict, so override it to False.
    split = _build_window_split(df, train_end, test_year, False if combined else opposition, combined=combined)
    if split is None:
        print("  [Skip] Empty split: train_end=%d test_year=%d" % (train_end, test_year))
        return

    X_train, y_train, X_test, y_test, _ = split
    tp = TextProcess()

    _, _, _, y_pred, y_score = _ml_fit_and_predict(
        record, opposition, X_train, y_train, X_test, y_test, tp=tp,
    )
    test_metrics = _compute_metrics(y_test, y_pred, y_score)
    embedding    = record.get("embedding", None)

    result_record = {
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "mode":                 "sliding_window_ml",
        "algo":                 record["algo"],
        "experiment":           record["experiment"],
        "case_mode":            case_mode,
        "opposition":           opposition,
        "dataset_label":        dataset_label,
        "input_representation": record.get("input_representation", ""),
        "embedding":            embedding,
        "model_name":           None,
        "max_context_length":   None,
        "train_years":          "2000-%d" % train_end,
        "test_year":            test_year,
        "train_size":           int(len(y_train)),
        "test_size":            int(len(y_test)),
        "params":               record.get("best_params", {}),
        "test_metrics":         test_metrics,
    }

    append_json_result(result_record, output_path)
    print(
        "    OK ML [%-12s | %-10s%s] train<=%d -> test=%d  test_F1=%.4f" % (
            record["algo"],
            record.get("input_representation", ""),
            (" / " + embedding) if embedding else "",
            train_end, test_year,
            test_metrics["f1"],
        )
    )

# -----------------------------------------------------------------------------
# DL window evaluation
# -----------------------------------------------------------------------------

def _run_dl_window(record, df, train_end, test_year, opposition, dataset_label, case_mode, output_path, device, combined=False):
    """Train/test one DL combo on a single sliding window."""
    split = _build_window_split(df, train_end, test_year, False if combined else opposition, combined=combined)
    if split is None:
        print("  [Skip] Empty split: train_end=%d test_year=%d" % (train_end, test_year))
        return

    X_train, y_train, X_test, y_test, years_train = split

    model_key    = record.get("algo", "legalbert")
    hf_name      = record.get("model_name", "nlpaueb/legal-bert-base-uncased")
    max_len      = record.get("max_context_length", 512)
    params       = record.get("best_params", {})

    lr           = float(params.get("learning_rate", 2e-5))
    batch_size   = int(params.get("batch_size", 32))
    max_epochs   = int(record.get("best_epoch") or params.get("epochs", 10))
    dropout      = float(params.get("dropout", 0.1))
    weight_decay = float(params.get("weight_decay", 0.0))
    warmup_ratio = float(params.get("warmup_ratio", 0.10))

    dl = DeepLearningExperiments(
        model_name=model_key,
        experiment=record.get("experiment", "1"),
        opposition=opposition,
        case_mode=case_mode,
        device=device,
        results_json_path="/dev/null",
    )

    X_train_proc = dl._preprocess_text_for_bert(X_train.reset_index(drop=True))
    X_test_proc  = dl._preprocess_text_for_bert(X_test.reset_index(drop=True))

    dl.aux_encoder = None

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)
    model = dl._build_model(dropout)

    # ── Validation: full previous year to the test year ─────────────────────
    _val_mask = (years_train == train_end)
    if _val_mask.sum() == 0:
        print("    [Warning] No data for val year=%d; falling back to last 20%% of training." % train_end)
        _n        = len(X_train_proc)
        _split_i  = int(_n * 0.80)
        _X_tr  = X_train_proc.iloc[:_split_i].reset_index(drop=True)
        _y_tr  = y_train[:_split_i]
        _X_val = X_train_proc.iloc[_split_i:].reset_index(drop=True)
        _y_val = y_train[_split_i:]
        _val_strategy = "pct_fallback_0.20"
    else:
        _tr_mask = ~_val_mask
        _X_tr  = X_train_proc.loc[_tr_mask].reset_index(drop=True)
        _y_tr  = y_train[_tr_mask]
        _X_val = X_train_proc.loc[_val_mask].reset_index(drop=True)
        _y_val = y_train[_val_mask]
        _val_strategy = "prev_year_%d" % train_end
        print("    Val set: year=%d  n_val=%d  n_train=%d" % (train_end, int(_val_mask.sum()), int(_tr_mask.sum())))
    _is_lf        = "longformer" in model_key.lower()
    _accum        = 8 if _is_lf else 1
    _train_loader = dl._create_dataloader(_X_tr,  _y_tr,  is_train=True,  batch_size=batch_size)
    _val_loader   = dl._create_dataloader(_X_val, _y_val, is_train=False, batch_size=batch_size)
    dl._fix_opposition_head(model, _train_loader)
    _optimizer  = dl._build_optimizer(model, lr, weight_decay)
    _criterion  = CrossEntropyLoss()
    _spe        = -(-len(_train_loader) // _accum)
    # Use 30-epoch horizon for the scheduler to exactly replicate the LR
    # trajectory seen during HP tuning (which always used epochs=30 as ceiling).
    _sched_epochs = 30
    _scheduler  = get_linear_schedule_with_warmup(
        _optimizer,
        num_warmup_steps=int(warmup_ratio * _spe * _sched_epochs),
        num_training_steps=_spe * _sched_epochs,
    )
    # ── Step-level patience (uniform with HP tuning) ──────────────────────────
    _eval_every       = max(1, _spe // 4)   # ~4 evals / epoch
    _patience         = 8
    _min_global_steps = 3 * _spe            # hard minimum: 3 full epochs
    best_val_f1       = -1.0
    best_epoch        = 1
    _best_state       = copy.deepcopy(model.state_dict())
    _best_head_state  = copy.deepcopy(dl.custom_head.state_dict()) if opposition else None
    _patience_count   = 0
    _eval_count       = 0
    _global_step      = 0
    _epoch_history    = []
    _done             = False
    _label = "Window train<=%d test=%d" % (train_end, test_year)
    print("    %s: step-level patience=%d eval_every=%d min_steps=%d" % (
        _label, _patience, _eval_every, _min_global_steps))
    for _ep in range(max_epochs):
        if _done:
            break
        for _eff_step, _avg_tl in dl._iter_train_steps(
            model, _optimizer, _train_loader, _criterion, _scheduler
        ):
            _global_step += 1
            _should_eval = (_global_step % _eval_every == 0) or (_eff_step == _spe)
            if _should_eval:
                _vl, _vf1 = dl._validate_epoch(model, _val_loader, _criterion)
                _eval_count += 1
                _epoch_history.append({
                    "epoch":       _ep + 1,
                    "global_step": _global_step,
                    "eval_index":  _eval_count,
                    "train_loss":  round(float(_avg_tl), 6),
                    "val_loss":    round(float(_vl),     6),
                    "val_f1":      round(float(_vf1),    6),
                })
                print("    %s Ep %d step %d eval %d: train=%.4f val=%.4f f1=%.4f" % (
                    _label, _ep + 1, _global_step, _eval_count, _avg_tl, _vl, _vf1))
                if _vf1 > best_val_f1:
                    best_val_f1 = _vf1
                    best_epoch  = _ep + 1
                    _best_state = copy.deepcopy(model.state_dict())
                    if opposition:
                        _best_head_state = copy.deepcopy(dl.custom_head.state_dict())
                    _patience_count = 0
                else:
                    _patience_count += 1
                    if _global_step >= _min_global_steps and _patience_count >= _patience:
                        print("    %s Early stopping at epoch %d step %d (best=%d val_f1=%.4f)" % (
                            _label, _ep + 1, _global_step, best_epoch, best_val_f1))
                        _done = True
                        break
    model.load_state_dict(_best_state)
    if opposition and _best_head_state is not None:
        dl.custom_head.load_state_dict(_best_head_state)

    test_metrics = dl._compute_test_metrics(model, X_test_proc, y_test)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logged_params = {
        **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "epochs":       best_epoch,
        "best_epoch":   best_epoch,
        "max_epochs":   max_epochs,
        "warmup_ratio": warmup_ratio,
        "best_val_f1":  best_val_f1,
        "val_strategy": _val_strategy,
    }

    result_record = {
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "mode":                 "sliding_window_dl",
        "epoch_history":        _epoch_history,
        "algo":                 model_key,
        "experiment":           record.get("experiment", ""),
        "case_mode":            case_mode,
        "opposition":           opposition,
        "dataset_label":        dataset_label,
        "input_representation": record.get("input_representation", model_key),
        "embedding":            None,
        "model_name":           hf_name,
        "max_context_length":   max_len,
        "train_years":          "2000-%d" % train_end,
        "test_year":            test_year,
        "train_size":           int(len(y_train)),
        "test_size":            int(len(y_test)),
        "params":               logged_params,
        "test_metrics":         test_metrics,
    }

    append_json_result(result_record, output_path)
    print(
        "    OK DL [%-20s] train<=%d -> test=%d  best_epoch=%d  test_F1=%.4f" % (
            model_key, train_end, test_year, best_epoch, test_metrics["f1"],
        )
    )

# -----------------------------------------------------------------------------
# Full test-set evaluation (pre-split pickles)
# -----------------------------------------------------------------------------

def _run_ml_full_test(record, X_train, y_train, X_test, y_test,
                      opposition, dataset_label, case_mode, output_path):
    """Train on full X_train and evaluate on X_test."""
    tp = TextProcess()

    _, _, _, y_pred, y_score = _ml_fit_and_predict(
        record, opposition, X_train, y_train, X_test, y_test, tp=tp,
    )
    test_metrics = _compute_metrics(y_test, y_pred, y_score)
    embedding    = record.get("embedding", None)

    result_record = {
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "mode":                 "full_test_ml",
        "algo":                 record["algo"],
        "experiment":           record["experiment"],
        "case_mode":            case_mode,
        "opposition":           opposition,
        "dataset_label":        dataset_label,
        "input_representation": record.get("input_representation", ""),
        "embedding":            embedding,
        "model_name":           None,
        "max_context_length":   None,
        "train_years":          "full",
        "test_year":            "full_test",
        "train_size":           int(len(y_train)),
        "test_size":            int(len(y_test)),
        "params":               record.get("best_params", {}),
        "test_metrics":         test_metrics,
    }

    append_json_result(result_record, output_path)
    print(
        "    OK ML-FullTest [%-12s | %-10s%s]  test_F1=%.4f" % (
            record["algo"],
            record.get("input_representation", ""),
            (" / " + embedding) if embedding else "",
            test_metrics["f1"],
        )
    )

def _run_dl_full_test(record, X_train, y_train, X_test, y_test,
                      opposition, dataset_label, case_mode, output_path, device):
    """Train on full X_train with early stopping, then evaluate on X_test."""

    model_key    = record.get("algo", "legalbert")
    hf_name      = record.get("model_name", "nlpaueb/legal-bert-base-uncased")
    max_len      = record.get("max_context_length", 512)
    params       = record.get("best_params", {})

    lr           = float(params.get("learning_rate", 2e-5))
    batch_size   = int(params.get("batch_size", 32))
    max_epochs   = int(record.get("best_epoch") or params.get("epochs", 10))
    dropout      = float(params.get("dropout", 0.1))
    weight_decay = float(params.get("weight_decay", 0.0))
    warmup_ratio = float(params.get("warmup_ratio", 0.10))

    dl = DeepLearningExperiments(
        model_name=model_key,
        experiment=record.get("experiment", "1"),
        opposition=opposition,
        case_mode=case_mode,
        device=device,
        results_json_path="/dev/null",
    )

    X_train_proc = dl._preprocess_text_for_bert(X_train.reset_index(drop=True))
    X_test_proc  = dl._preprocess_text_for_bert(X_test.reset_index(drop=True))

    dl.aux_encoder = None

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    # ── Try to load the best model saved during HP tuning ─────────────────────
    _best_model_path = record.get("best_model_path")
    if _best_model_path and os.path.exists(_best_model_path):
        print("    [FullTest] Loading saved HP model: %s" % _best_model_path)
        model = dl._build_model(dropout)
        # Build a probe loader to fit the aux encoder and size the opposition head.
        # Must use is_train=True so _encode_opposition_features fits the encoder.
        _probe_loader = dl._create_dataloader(X_train_proc, y_train, is_train=True, batch_size=batch_size)
        dl._fix_opposition_head(model, _probe_loader)
        del _probe_loader
        _payload = torch.load(_best_model_path, map_location=device)
        model.load_state_dict(_payload["model_state"])
        if opposition and _payload.get("custom_head_state") is not None:
            dl.custom_head.load_state_dict(_payload["custom_head_state"])
        best_epoch    = int(_payload.get("best_epoch", max_epochs))
        best_val_f1   = float(_payload.get("val_f1", -1.0))
        _epoch_history = []
        _retrained     = False
        _val_strategy  = "loaded_hp_model"
        print("    [FullTest] Loaded: best_epoch=%d val_f1=%.4f" % (best_epoch, best_val_f1))
    else:
        if _best_model_path:
            print("    [Warning] best_model_path not found (%s); retraining." % _best_model_path)
        else:
            print("    [FullTest] No saved model path in record; training from scratch.")
        model = dl._build_model(dropout)
        # ── Step-level early stopping (uniform with HP tuning) ────────────────
        _n        = len(X_train_proc)
        _split_i  = int(_n * 0.80)
        _X_tr  = X_train_proc.iloc[:_split_i].reset_index(drop=True)
        _y_tr  = y_train[:_split_i]
        _X_val = X_train_proc.iloc[_split_i:].reset_index(drop=True)
        _y_val = y_train[_split_i:]
        _is_lf        = "longformer" in model_key.lower()
        _accum        = 8 if _is_lf else 1
        _train_loader = dl._create_dataloader(_X_tr,  _y_tr,  is_train=True,  batch_size=batch_size)
        _val_loader   = dl._create_dataloader(_X_val, _y_val, is_train=False, batch_size=batch_size)
        dl._fix_opposition_head(model, _train_loader)
        _optimizer  = dl._build_optimizer(model, lr, weight_decay)
        _criterion  = CrossEntropyLoss()
        _spe        = -(-len(_train_loader) // _accum)
        _sched_epochs = 30
        _scheduler  = get_linear_schedule_with_warmup(
            _optimizer,
            num_warmup_steps=int(warmup_ratio * _spe * _sched_epochs),
            num_training_steps=_spe * _sched_epochs,
        )
        _eval_every       = max(1, _spe // 4)
        _patience         = 8
        _min_global_steps = 3 * _spe
        best_val_f1       = -1.0
        best_epoch        = 1
        _best_state       = copy.deepcopy(model.state_dict())
        _best_head_state  = copy.deepcopy(dl.custom_head.state_dict()) if opposition else None
        _patience_count   = 0
        _eval_count       = 0
        _global_step      = 0
        _epoch_history    = []
        _done             = False
        print("    [FullTest] step-level patience=%d eval_every=%d min_steps=%d" % (
            _patience, _eval_every, _min_global_steps))
        for _ep in range(max_epochs):
            if _done:
                break
            for _eff_step, _avg_tl in dl._iter_train_steps(
                model, _optimizer, _train_loader, _criterion, _scheduler
            ):
                _global_step += 1
                _should_eval = (_global_step % _eval_every == 0) or (_eff_step == _spe)
                if _should_eval:
                    _vl, _vf1 = dl._validate_epoch(model, _val_loader, _criterion)
                    _eval_count += 1
                    _epoch_history.append({
                        "epoch":       _ep + 1,
                        "global_step": _global_step,
                        "eval_index":  _eval_count,
                        "train_loss":  round(float(_avg_tl), 6),
                        "val_loss":    round(float(_vl),     6),
                        "val_f1":      round(float(_vf1),    6),
                    })
                    print("    FullTest Ep %d step %d eval %d: train=%.4f val=%.4f f1=%.4f" % (
                        _ep + 1, _global_step, _eval_count, _avg_tl, _vl, _vf1))
                    if _vf1 > best_val_f1:
                        best_val_f1 = _vf1
                        best_epoch  = _ep + 1
                        _best_state = copy.deepcopy(model.state_dict())
                        if opposition:
                            _best_head_state = copy.deepcopy(dl.custom_head.state_dict())
                        _patience_count = 0
                    else:
                        _patience_count += 1
                        if _global_step >= _min_global_steps and _patience_count >= _patience:
                            print("    FullTest Early stopping at epoch %d step %d (best=%d val_f1=%.4f)" % (
                                _ep + 1, _global_step, best_epoch, best_val_f1))
                            _done = True
                            break
        model.load_state_dict(_best_state)
        if opposition and _best_head_state is not None:
            dl.custom_head.load_state_dict(_best_head_state)
        _retrained    = True
        _val_strategy = "temporal_step_patience_0.20"

    test_metrics = dl._compute_test_metrics(model, X_test_proc, y_test)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logged_params = {
        **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "epochs":       best_epoch,
        "best_epoch":   best_epoch,
        "max_epochs":   max_epochs,
        "warmup_ratio": warmup_ratio,
        "best_val_f1":  best_val_f1,
        "val_strategy": _val_strategy,
        "retrained":    _retrained,
    }

    result_record = {
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "mode":                 "full_test_dl",
        "epoch_history":        _epoch_history,
        "algo":                 model_key,
        "experiment":           record.get("experiment", ""),
        "case_mode":            case_mode,
        "opposition":           opposition,
        "dataset_label":        dataset_label,
        "input_representation": record.get("input_representation", model_key),
        "embedding":            None,
        "model_name":           hf_name,
        "max_context_length":   max_len,
        "train_years":          "full",
        "test_year":            "full_test",
        "train_size":           int(len(y_train)),
        "test_size":            int(len(y_test)),
        "params":               logged_params,
        "test_metrics":         test_metrics,
    }

    append_json_result(result_record, output_path)
    print(
        "    OK DL-FullTest [%-20s]  best_epoch=%d  test_F1=%.4f" % (
            model_key, best_epoch, test_metrics["f1"],
        )
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sliding-window temporal evaluation using best CV hyperparameters. "
            "Runs all experiment x opposition x dataset combinations by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pf_dataset",
        default=str(PROJECT_ROOT / "Data" / "Train&TestData_1.0_PatentRefusal.pkl"),
        help="Full Patent Refusal dataset pickle (must contain Year column)",
    )
    parser.add_argument(
        "--op_dataset",
        default=str(PROJECT_ROOT / "Data" / "Train&TestData_1.0_OppositionDivision.pkl"),
        help="Full Opposition Division dataset pickle (must contain Year column)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["1", "2"],
        metavar="EXP",
        help="Which experiments to run, e.g. --experiments 1 2 (default: 1 2)",
    )
    parser.add_argument(
        "--opposition",
        default="both",
        choices=["true", "false", "both"],
        help="Which opposition modes: true, false, or both (default: both)",
    )
    parser.add_argument(
        "--ml_results",
        default=str(PROJECT_ROOT / "Results" / "results_main.json"),
        help="Path to flat-CV ML results JSON",
    )
    parser.add_argument(
        "--dl_results",
        default=str(PROJECT_ROOT / "Results" / "results_deep_learning.json"),
        help="Path to flat-CV DL results JSON",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "Results" / "results_sliding_window.json"),
        help="Path to write sliding-window results JSON",
    )
    parser.add_argument(
        "--first_test", type=int, default=2021,
        help="First test year (default: 2021 -> first window trains on 2000-2020)",
    )
    parser.add_argument(
        "--last_test", type=int, default=2024,
        help="Last test year inclusive (default: 2024)",
    )
    parser.add_argument(
        "--data_dir",
        default=str(PROJECT_ROOT / "Data" / "Final_Processed"),
        help="Directory containing pre-split X_Train/X_test/y_Train/y_test pickles",
    )
    parser.add_argument("--skip_ml", action="store_true", help="Skip ML models")
    parser.add_argument("--skip_dl", action="store_true", help="Skip DL models")
    parser.add_argument("--skip_full_test", action="store_true",
                        help="Skip full pre-split test-set evaluation")
    parser.add_argument("--skip_sliding_window", action="store_true",
                        help="Skip the sliding-window evaluation")
    parser.add_argument(
        "--only_cases",
        default=None,
        help="Comma-separated list of case modes to run, e.g. 'both' or 'op,both'. "
             "If not set, all modes implied by --opposition are run.",
    )
    parser.add_argument(
        "--only_algos",
        default=None,
        help="Comma-separated list of algo names to include, e.g. 'legalbert' or "
             "'legalbert,longformer_base'. If not set, all algos are run.",
    )
    return parser

def _resolve_opposition_modes(flag):
    if flag == "true":
        return [True]
    if flag == "false":
        return [False]
    return [False, True]

def _load_df(path, label):
    if not os.path.exists(path):
        print("[Warning] %s dataset not found at '%s', skipping." % (label, path))
        return None
    df = pd.read_pickle(path)
    required = {"New Summary Facts", "Outcome", "Year"}
    missing  = required - set(df.columns)
    if missing:
        print("[Warning] %s dataset missing columns %s, skipping." % (label, missing))
        return None
    print("  %s: %d rows | years %d-%d" % (
        label, len(df), int(df["Year"].min()), int(df["Year"].max())
    ))
    return df

def main():
    args   = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    print("\nLoading datasets ...")
    datasets = {}
    pf_df = _load_df(args.pf_dataset, "PatentRefusal")
    op_df = _load_df(args.op_dataset, "OppositionDivision")
    if pf_df is not None:
        datasets["PatentRefusal"]      = pf_df
    if op_df is not None:
        datasets["OppositionDivision"] = op_df

    if not datasets:
        print("No valid datasets found. Exiting.")
        return

    requested_case_modes = {
        "false": {"pf"},
        "true": {"op", "both"},
        "both": {"pf", "op", "both"},
    }[args.opposition]
    if args.only_cases:
        requested_case_modes &= {c.strip() for c in args.only_cases.split(",")}

    only_algos_set = {a.strip() for a in args.only_algos.split(",")} if args.only_algos else None

    case_configs = []
    if pf_df is not None and "pf" in requested_case_modes:
        case_configs.append({
            "case_mode": "pf",
            "dataset_label": "PatentRefusal",
            "opposition": False,
            "df": pf_df,
        })
    if op_df is not None and "op" in requested_case_modes:
        case_configs.append({
            "case_mode": "op",
            "dataset_label": "OppositionDivision",
            "opposition": True,
            "df": op_df,
        })
    if pf_df is not None and op_df is not None and "both" in requested_case_modes:
        # Build pf+OD combined dataframe with Category and OD_Match columns,
        # matching the schema produced by experiment_processing.py for the
        # both pre-split pickles (Category_* + OD_Match_* after get_dummies).
        _pf_for_both = pf_df.copy()
        _op_for_both = op_df.copy()
        _pf_for_both["Category"] = "Patent Refusal"
        _op_for_both["Category"] = "Opposition Division"
        _pf_for_both["OD_Match"] = "0"
        _op_for_both["OD_Match"] = "0"  # default; overwrite with Matches where present
        if "Matches" in _op_for_both.columns:
            _op_for_both["OD_Match"] = _op_for_both["Matches"].astype(str).where(
                _op_for_both["Matches"].isin(["1", "2", "3"]), other="0"
            )
        combined_df = pd.concat([_pf_for_both, _op_for_both], ignore_index=True)
        case_configs.append({
            "case_mode": "both",
            "dataset_label": "PatentRefusal+OppositionDivision",
            "opposition": True,   # use custom opposition head for aux features
            "combined": True,     # use Category+OD_Match schema in SW
            "df": combined_df,
        })
    elif op_df is not None and "both" in requested_case_modes:
        # Fallback if pf_df is unavailable — run OD-only (degraded mode)
        print("[Warning] pf_df unavailable; both case will run OD-only (schema mismatch risk).")
        case_configs.append({
            "case_mode": "both",
            "dataset_label": "OppositionDivision",
            "opposition": True,
            "combined": False,
            "df": op_df,
        })

    if not case_configs:
        print("No runnable case modes found for --opposition=%s. Exiting." % args.opposition)
        return

    experiments = [str(e) for e in args.experiments]
    windows = [
        (test_year - 1, test_year)
        for test_year in range(args.first_test, args.last_test + 1)
    ]

    print("\nCase modes: %s" % [c["case_mode"] for c in case_configs])
    print("Window schedule: %s" % [(tr, te) for tr, te in windows])
    print("Output: %s\n" % args.output)

    existing_keys = {
        _existing_eval_key(r)
        for r in _safe_load_json_array(args.output)
    }

    # -------------------------------------------------------------------------
    # Phase 1: Full pre-split test-set evaluation
    # -------------------------------------------------------------------------
    if not args.skip_full_test:
        print("\n" + "#" * 70)
        print("# PHASE 1: Full test-set evaluation (pre-split pickles)")
        print("#" * 70)

        for case in case_configs:
            case_mode = case["case_mode"]
            dataset_label = case["dataset_label"]
            opposition = case["opposition"]
            df = case["df"]

            for experiment in experiments:
                pkl_suffix = case_mode

                x_train_path = os.path.join(args.data_dir, "X_Train_%s_%s.pkl" % (experiment, pkl_suffix))
                x_test_path  = os.path.join(args.data_dir, "X_test_%s_%s.pkl"  % (experiment, pkl_suffix))
                y_train_path = os.path.join(args.data_dir, "y_Train_%s_%s.pkl" % (experiment, pkl_suffix))
                y_test_path  = os.path.join(args.data_dir, "y_test_%s_%s.pkl"  % (experiment, pkl_suffix))

                missing = [p for p in [x_train_path, x_test_path, y_train_path, y_test_path]
                           if not os.path.exists(p)]
                if missing:
                    print("  [Skip] Missing pickles for case=%s exp=%s opp=%s:" % (
                        case_mode, experiment, opposition))
                    for p in missing:
                        print("         %s" % p)
                    continue

                X_train_full = pd.read_pickle(x_train_path)
                X_test_full  = pd.read_pickle(x_test_path)
                y_train_full = np.array(pd.read_pickle(y_train_path)).ravel()
                y_test_full  = np.array(pd.read_pickle(y_test_path)).ravel()

                print("\n  Loaded: case=%s (%s) | Exp %s | opp=%s  train=%d test=%d" % (
                    case_mode, dataset_label, experiment, opposition,
                    len(y_train_full), len(y_test_full),
                ))

                combo_tag = "case=%s | %s | Exp %s | opposition=%s" % (
                    case_mode, dataset_label, experiment, opposition)
                best_records = _load_best_records(
                    args.ml_results, args.dl_results, experiment, opposition, case_mode=case_mode,
                )
                if not best_records:
                    print("  No matching CV records for %s. Skipping." % combo_tag)
                    continue

                for record in best_records:
                    rec_mode = record.get("mode", "")
                    is_dl    = rec_mode == "deep_learning"
                    is_ml    = rec_mode in ("sparse", "embedding")

                    if is_dl and args.skip_dl:
                        continue
                    if is_ml and args.skip_ml:
                        continue
                    if only_algos_set and record.get("algo") not in only_algos_set:
                        continue

                    if is_dl:
                        key = (
                            "full_test_dl",
                            str(record.get("experiment", "")),
                            case_mode,
                            str(record.get("algo", "")),
                            str(record.get("input_representation", "")),
                            str(record.get("embedding") or ""),
                        )
                    else:
                        key = (
                            "full_test_ml",
                            str(record.get("experiment", "")),
                            case_mode,
                            str(record.get("algo", "")),
                            str(record.get("input_representation", "")),
                            str(record.get("embedding") or ""),
                        )

                    if key in existing_keys:
                        print("    [Skip existing] %s | %s | %s" % (
                            key[0], record.get("algo", ""), record.get("input_representation", "")))
                        continue

                    print(
                        "\n  %s\n  %s: %s | %s | case=%s | opp=%s\n  %s" % (
                            "-" * 66,
                            "DL" if is_dl else "ML",
                            record["algo"],
                            record.get("input_representation", ""),
                            case_mode,
                            opposition,
                            "-" * 66,
                        )
                    )

                    if is_dl:
                        _run_dl_full_test(
                            record=record,
                            X_train=X_train_full,
                            y_train=y_train_full,
                            X_test=X_test_full,
                            y_test=y_test_full,
                            opposition=opposition,
                            dataset_label=dataset_label,
                            case_mode=case_mode,
                            output_path=args.output,
                            device=device,
                        )
                    else:
                        _run_ml_full_test(
                            record=record,
                            X_train=X_train_full,
                            y_train=y_train_full,
                            X_test=X_test_full,
                            y_test=y_test_full,
                            opposition=opposition,
                            dataset_label=dataset_label,
                            case_mode=case_mode,
                            output_path=args.output,
                        )

                    existing_keys.add(key)

        print("\n" + "#" * 70)
        print("# PHASE 1 complete.")
        print("#" * 70)
    else:
        print("\n[Skip] Full test-set evaluation (--skip_full_test).")

    # -------------------------------------------------------------------------
    # Phase 2: Sliding-window temporal evaluation
    # -------------------------------------------------------------------------
    if args.skip_sliding_window:
        print("\n[Skip] Sliding-window evaluation (--skip_sliding_window).")
        print("\n" + "=" * 70)
        print("Evaluation complete.")
        print("Results written to: %s" % args.output)
        print("=" * 70)
        return

    print("\n" + "#" * 70)
    print("# PHASE 2: Sliding-window temporal evaluation")
    print("#" * 70)

    for case in case_configs:
        case_mode = case["case_mode"]
        dataset_label = case["dataset_label"]
        opposition = case["opposition"]
        df = case["df"]

        for experiment in experiments:
            if str(experiment) != "2":
                print("  [Skip] Sliding window only runs for Experiment 2 (exp=%s skipped)." % experiment)
                continue

            combo_tag = "case=%s | %s | Exp %s | opposition=%s" % (
                case_mode, dataset_label, experiment, opposition)
            print("\n" + "=" * 70)
            print("Combo: %s" % combo_tag)
            print("=" * 70)

            best_records = _load_best_records(
                args.ml_results, args.dl_results, experiment, opposition, case_mode=case_mode,
            )
            if not best_records:
                print("  No matching CV records for %s. Skipping." % combo_tag)
                continue

            print("  %d unique (algo x input) combos:" % len(best_records))
            for r in best_records:
                emb = r.get("embedding") or ""
                print(
                    "    [%-10s] %-20s | %-12s%s  (CV F1=%.4f)" % (
                        r["mode"], r["algo"],
                        r.get("input_representation", ""),
                        (" / " + emb) if emb else "",
                        r.get("best_score_cv", 0),
                    )
                )

            for record in best_records:
                rec_mode = record.get("mode", "")
                is_dl    = rec_mode == "deep_learning"
                is_ml    = rec_mode in ("sparse", "embedding")

                if is_dl and args.skip_dl:
                    continue
                if is_ml and args.skip_ml:
                    continue
                if only_algos_set and record.get("algo") not in only_algos_set:
                    continue

                print(
                    "\n  %s\n  %s: %s | %s | case=%s | opp=%s\n  %s" % (
                        "-" * 66,
                        "DL" if is_dl else "ML",
                        record["algo"],
                        record.get("input_representation", ""),
                        case_mode,
                        opposition,
                        "-" * 66,
                    )
                )

                for train_end, test_year in windows:
                    if (df["Year"] == test_year).sum() == 0:
                        print("  [Skip] No data for test_year=%d" % test_year)
                        continue

                    if is_dl:
                        key = (
                            "sliding_window_dl",
                            str(record.get("experiment", "")),
                            case_mode,
                            str(record.get("algo", "")),
                            str(record.get("input_representation", "")),
                            str(record.get("embedding") or ""),
                            str(test_year),
                        )
                    else:
                        key = (
                            "sliding_window_ml",
                            str(record.get("experiment", "")),
                            case_mode,
                            str(record.get("algo", "")),
                            str(record.get("input_representation", "")),
                            str(record.get("embedding") or ""),
                            str(test_year),
                        )

                    if key in existing_keys:
                        print("    [Skip existing] %s year=%s | %s | %s" % (
                            key[0], test_year, record.get("algo", ""), record.get("input_representation", "")))
                        continue

                    if is_dl:
                        _run_dl_window(
                            record=record,
                            df=df,
                            train_end=train_end,
                            test_year=test_year,
                            opposition=opposition,
                            dataset_label=dataset_label,
                            case_mode=case_mode,
                            output_path=args.output,
                            device=device,
                            combined=case.get("combined", False),
                        )
                    else:
                        _run_ml_window(
                            record=record,
                            df=df,
                            train_end=train_end,
                            test_year=test_year,
                            opposition=opposition,
                            dataset_label=dataset_label,
                            case_mode=case_mode,
                            output_path=args.output,
                            combined=case.get("combined", False),
                        )

                    existing_keys.add(key)

    print("\n" + "=" * 70)
    print("Sliding window evaluation complete.")
    print("Results written to: %s" % args.output)
    print("=" * 70)

if __name__ == "__main__":
    main()
