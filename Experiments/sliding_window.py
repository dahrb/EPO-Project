"""Sliding-Window Temporal Evaluation.

For each (algo, experiment, opposition, input_representation) combination, loads
the best hyperparameters found during flat-CV (from results_main.json and/or
results_deep_learning.json), then re-runs a sliding-window evaluation across
ALL dataset x experiment x opposition combinations by default.

Window schedule (default):
    train on 2000-2020  -> test on 2021
    train on 2000-2021  -> test on 2022
    train on 2000-2022  -> test on 2023
    train on 2000-2023  -> test on 2024   (if data available)

For each window two evaluations are stored:
  (a) test_metrics  - model trained on full window-train, evaluated on test year
  (b) cv_train_metrics - 10-fold StratifiedKFold CV on the window-train set
      (uses the same best params; gives an in-distribution quality signal without
      a held-out test set; analogous to learning-curve tracking)

Output record schema (per window):
    {
      "timestamp_utc":        str,
      "mode":                 "sliding_window_ml" | "sliding_window_dl",
      "algo":                 str,
      "experiment":           "1" | "2",
      "opposition":           bool,
      "dataset_label":        str,           # "PatentRefusal" | "OppositionDivision"
      "input_representation": str,
      "embedding":            str | null,    # ML embedding key, null for DL
      "model_name":           str | null,    # HuggingFace ID, null for ML
      "max_context_length":   int | null,
      "train_years":          "2000-YYYY",
      "test_year":            int,
      "train_size":           int,
      "test_size":            int,
      "params":               dict,
      "cv_train_metrics":     {mean_f1, std_f1, mean_accuracy, std_accuracy,
                               mean_precision, std_precision, mean_recall,
                               std_recall, mean_mcc, std_mcc},
      "test_metrics":         {accuracy, f1, precision, recall, mcc, auc}
    }

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
        [--cv_folds     10]
        [--first_test   2021]
        [--last_test    2024]
        [--skip_ml] [--skip_dl]

Last Updated: 09.04.26

Status: In Progress
"""

import argparse
import fcntl
import json
import os
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC
import xgboost as xgb

from transformers import get_linear_schedule_with_warmup

from Experiments.deep_learning_experiments import DeepLearningExperiments
from Experiments.ml_experiments import PreprocessTransformer
from Utilities.utils import TextProcess, Word2VecTransform


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _append_json_result(record, path):
    """Append one record to a JSON array file with exclusive file locking."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        content = fh.read().strip()
        try:
            data = json.loads(content) if content else []
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            data = []
        data.append(_jsonable(record))
        fh.seek(0)
        fh.truncate()
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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


def _cv_mean_metrics(fold_metric_list):
    """Aggregate per-fold metric dicts into mean +/- std summary."""
    if not fold_metric_list:
        return {}
    keys = [k for k in fold_metric_list[0] if fold_metric_list[0][k] is not None]
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metric_list if m.get(k) is not None]
        result["mean_" + k] = float(np.mean(vals))
        result["std_"  + k] = float(np.std(vals))
    return result


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

def _build_window_split(df, train_end, test_year, opposition):
    """Slice df into (X_train, y_train, X_test, y_test) for one window."""
    train_df = df[df["Year"] <= train_end].copy()
    test_df  = df[df["Year"] == test_year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    feature_cols = ["Matches"] if opposition and "Matches" in df.columns else []

    def _build_X(data):
        if not feature_cols:
            return data[["New Summary Facts"]].copy()
        X = data[["New Summary Facts"] + feature_cols].copy()
        dummies = pd.get_dummies(X[feature_cols], prefix=feature_cols).astype(int)
        return pd.concat([X.drop(columns=feature_cols), dummies], axis=1)

    X_train = _build_X(train_df).reset_index(drop=True)
    X_test  = _build_X(test_df).reset_index(drop=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = np.array([1 if v == "Affirmed" else 0 for v in train_df["Outcome"]])
    y_test  = np.array([1 if v == "Affirmed" else 0 for v in test_df["Outcome"]])

    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# Best-record selection
# -----------------------------------------------------------------------------

def _load_best_records(ml_results_path, dl_results_path, experiment, opposition):
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
        score = r.get("best_score_cv", -1) or -1
        if key not in best or score > best[key].get("best_score_cv", -1):
            best[key] = r

    return list(best.values())


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

    vect_kwargs = {
        k[len("vect__" + num_prefix):]: v
        for k, v in params.items()
        if k.startswith("vect__" + num_prefix)
    }
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
                X_copy["New Summary Facts"] = list(tp.nlp.pipe(X_copy["New Summary Facts"].tolist()))
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
                w2v  = Word2VecTransform(embedding=embedding)
                return w2v.fit_transform(docs)

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
# 10-fold CV on training window (ML)
# -----------------------------------------------------------------------------

def _ml_cv_train(record, X_train, y_train, opposition, cv_folds=10):
    """Run k-fold CV on the training window to get an in-distribution quality signal."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    tp  = TextProcess()
    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
        X_fold_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_fold_tr  = y_train[tr_idx]
        y_fold_val = y_train[val_idx]
        try:
            _, _, _, y_pred, y_score = _ml_fit_and_predict(
                record, opposition,
                X_fold_tr, y_fold_tr,
                X_fold_val, y_fold_val,
                tp=tp,
            )
            fold_metrics.append(_compute_metrics(y_fold_val, y_pred, y_score))
        except Exception as e:
            print("    [ML CV fold %d error] %s" % (fold_idx + 1, e))

    return _cv_mean_metrics(fold_metrics)


# -----------------------------------------------------------------------------
# ML window evaluation
# -----------------------------------------------------------------------------

def _run_ml_window(record, df, train_end, test_year, opposition, dataset_label, output_path, cv_folds=10):
    """Train/test one ML combo on a single sliding window + k-fold CV on train."""
    split = _build_window_split(df, train_end, test_year, opposition)
    if split is None:
        print("  [Skip] Empty split: train_end=%d test_year=%d" % (train_end, test_year))
        return

    X_train, y_train, X_test, y_test = split
    tp = TextProcess()

    # (a) 10-fold CV on training window — only for the 2000-2022 window
    if train_end == 2022:
        print("    Running %d-fold CV on train window (train<=%d) ..." % (cv_folds, train_end))
        cv_train_metrics = _ml_cv_train(record, X_train, y_train, opposition, cv_folds=cv_folds)
    else:
        cv_train_metrics = {}

    # (b) Full-train fit -> test
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
        "cv_train_metrics":     cv_train_metrics,
        "test_metrics":         test_metrics,
    }

    _append_json_result(result_record, output_path)
    print(
        "    OK ML [%-12s | %-10s%s] train<=%d -> test=%d  "
        "test_F1=%.4f  cv_mean_F1=%.4f" % (
            record["algo"],
            record.get("input_representation", ""),
            (" / " + embedding) if embedding else "",
            train_end, test_year,
            test_metrics["f1"],
            cv_train_metrics.get("mean_f1", float("nan")),
        )
    )


# -----------------------------------------------------------------------------
# 10-fold CV on training window (DL)
# -----------------------------------------------------------------------------

def _dl_cv_train(dl, X_train_proc, y_train, params, opposition, cv_folds=10):
    """Run k-fold CV on the DL training window using the best params."""
    from torch.nn import CrossEntropyLoss

    lr           = float(params.get("learning_rate", 2e-5))
    batch_size   = int(params.get("batch_size", 32))
    epochs       = int(params.get("epochs", 3))
    dropout      = float(params.get("dropout", 0.1))
    weight_decay = float(params.get("weight_decay", 0.0))
    warmup_steps = int(params.get("warmup_steps", 0))

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
        X_fold_tr  = X_train_proc.iloc[tr_idx].reset_index(drop=True)
        X_fold_val = X_train_proc.iloc[val_idx].reset_index(drop=True)
        y_fold_tr  = y_train[tr_idx]
        y_fold_val = y_train[val_idx]

        try:
            dl.aux_encoder = None
            fold_train_loader = dl._create_dataloader(
                X_fold_tr, y_fold_tr, is_train=True, batch_size=batch_size
            )
            fold_val_loader = dl._create_dataloader(
                X_fold_val, y_fold_val, is_train=False, batch_size=batch_size
            )

            model     = dl._build_model(dropout)
            optimizer = dl._build_optimizer(model, lr, weight_decay)
            criterion = CrossEntropyLoss()
            dl._fix_opposition_head(model, fold_train_loader)

            total_steps = len(fold_train_loader) * epochs
            scheduler   = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            for _ in range(epochs):
                dl._train_epoch(model, optimizer, fold_train_loader, criterion, scheduler)

            model.eval()
            all_preds, all_scores, all_labels = [], [], []
            with torch.no_grad():
                for batch in fold_val_loader:
                    input_ids      = batch["input_ids"].to(dl.device)
                    attention_mask = batch["attention_mask"].to(dl.device)
                    labels         = batch["labels"].to(dl.device)

                    if opposition:
                        outputs  = dl._get_encoder(model)(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        aux_feat = batch["auxiliary_features"].to(dl.device)
                        logits   = dl.custom_head(outputs.pooler_output, aux_feat)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits  = outputs.logits

                    preds  = torch.argmax(logits, dim=1)
                    probs  = torch.softmax(logits, dim=1)[:, 1]
                    all_preds.extend(preds.cpu().numpy())
                    all_scores.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            fold_metrics.append(
                _compute_metrics(
                    np.array(all_labels),
                    np.array(all_preds),
                    np.array(all_scores),
                )
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print("    [DL CV fold %d error] %s" % (fold_idx + 1, e))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _cv_mean_metrics(fold_metrics)


# -----------------------------------------------------------------------------
# DL window evaluation
# -----------------------------------------------------------------------------

def _run_dl_window(record, df, train_end, test_year, opposition, dataset_label, output_path, device, cv_folds=10):
    """Train/test one DL combo on a single sliding window + k-fold CV on train."""
    from torch.nn import CrossEntropyLoss

    split = _build_window_split(df, train_end, test_year, opposition)
    if split is None:
        print("  [Skip] Empty split: train_end=%d test_year=%d" % (train_end, test_year))
        return

    X_train, y_train, X_test, y_test = split

    model_key    = record.get("algo", "legalbert")
    hf_name      = record.get("model_name", "nlpaueb/legal-bert-base-uncased")
    max_len      = record.get("max_context_length", 512)
    params       = record.get("best_params", {})

    lr           = float(params.get("learning_rate", 2e-5))
    batch_size   = int(params.get("batch_size", 32))
    epochs       = int(params.get("epochs", 3))
    dropout      = float(params.get("dropout", 0.1))
    weight_decay = float(params.get("weight_decay", 0.0))
    warmup_steps = int(params.get("warmup_steps", 0))

    dl = DeepLearningExperiments(
        model_name=model_key,
        experiment=record.get("experiment", "1"),
        opposition=opposition,
        device=device,
        results_json_path="/dev/null",
    )

    X_train_proc = dl._preprocess_text_for_bert(X_train.reset_index(drop=True))
    X_test_proc  = dl._preprocess_text_for_bert(X_test.reset_index(drop=True))

    # (a) 10-fold CV on training window — only for the 2000-2022 window
    if train_end == 2022:
        print("    Running %d-fold DL CV on train window (train<=%d) ..." % (cv_folds, train_end))
        cv_train_metrics = _dl_cv_train(
            dl, X_train_proc, y_train, params, opposition, cv_folds=cv_folds,
        )
    else:
        cv_train_metrics = {}

    # (b) Full-train fit -> test
    dl.aux_encoder = None
    train_loader = dl._create_dataloader(X_train_proc, y_train, is_train=True,  batch_size=batch_size)
    test_loader  = dl._create_dataloader(X_test_proc,  y_test,  is_train=False, batch_size=batch_size)

    model     = dl._build_model(dropout)
    optimizer = dl._build_optimizer(model, lr, weight_decay)
    criterion = CrossEntropyLoss()
    dl._fix_opposition_head(model, train_loader)

    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        loss = dl._train_epoch(model, optimizer, train_loader, criterion, scheduler)
        print("    Window train<=%d test=%d Epoch %d/%d: loss=%.4f" % (
            train_end, test_year, epoch + 1, epochs, loss
        ))

    model.eval()
    all_preds, all_scores, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if opposition:
                outputs  = dl._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
                aux_feat = batch["auxiliary_features"].to(device)
                logits   = dl.custom_head(outputs.pooler_output, aux_feat)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits

            preds  = torch.argmax(logits, dim=1)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_metrics = _compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores),
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_record = {
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "mode":                 "sliding_window_dl",
        "algo":                 model_key,
        "experiment":           record.get("experiment", ""),
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
        "params":               {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "cv_train_metrics":     cv_train_metrics,
        "test_metrics":         test_metrics,
    }

    _append_json_result(result_record, output_path)
    print(
        "    OK DL [%-20s] train<=%d -> test=%d  test_F1=%.4f  cv_mean_F1=%.4f" % (
            model_key, train_end, test_year,
            test_metrics["f1"],
            cv_train_metrics.get("mean_f1", float("nan")),
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
        "--cv_folds", type=int, default=10,
        help="Number of folds for within-window CV on training data (default: 10)",
    )
    parser.add_argument(
        "--first_test", type=int, default=2021,
        help="First test year (default: 2021 -> first window trains on 2000-2020)",
    )
    parser.add_argument(
        "--last_test", type=int, default=2024,
        help="Last test year inclusive (default: 2024)",
    )
    parser.add_argument("--skip_ml", action="store_true", help="Skip ML models")
    parser.add_argument("--skip_dl", action="store_true", help="Skip DL models")
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

    opposition_modes = _resolve_opposition_modes(args.opposition)
    experiments      = [str(e) for e in args.experiments]
    windows = [
        (test_year - 1, test_year)
        for test_year in range(args.first_test, args.last_test + 1)
    ]

    print("\nWindow schedule: %s" % [(tr, te) for tr, te in windows])
    print("CV folds (train): %d" % args.cv_folds)
    print("Output: %s\n" % args.output)

    for dataset_label, df in datasets.items():
        for experiment in experiments:
            for opposition in opposition_modes:
                combo_tag = "%s | Exp %s | opposition=%s" % (dataset_label, experiment, opposition)
                print("\n" + "=" * 70)
                print("Combo: %s" % combo_tag)
                print("=" * 70)

                best_records = _load_best_records(
                    args.ml_results, args.dl_results, experiment, opposition,
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

                    print(
                        "\n  %s\n  %s: %s | %s | opp=%s\n  %s" % (
                            "-" * 66,
                            "DL" if is_dl else "ML",
                            record["algo"],
                            record.get("input_representation", ""),
                            opposition,
                            "-" * 66,
                        )
                    )

                    for train_end, test_year in windows:
                        if (df["Year"] == test_year).sum() == 0:
                            print("  [Skip] No data for test_year=%d" % test_year)
                            continue

                        if is_dl:
                            _run_dl_window(
                                record=record,
                                df=df,
                                train_end=train_end,
                                test_year=test_year,
                                opposition=opposition,
                                dataset_label=dataset_label,
                                output_path=args.output,
                                device=device,
                                cv_folds=args.cv_folds,
                            )
                        else:
                            _run_ml_window(
                                record=record,
                                df=df,
                                train_end=train_end,
                                test_year=test_year,
                                opposition=opposition,
                                dataset_label=dataset_label,
                                output_path=args.output,
                                cv_folds=args.cv_folds,
                            )

    print("\n" + "=" * 70)
    print("Sliding window evaluation complete.")
    print("Results written to: %s" % args.output)
    print("=" * 70)


if __name__ == "__main__":
    main()
