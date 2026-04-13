"""Patent Experiments ML.

Flat-CV experiment runner for heldout-test workflows.

Key behaviour
-------------
- Uses `RandomizedSearchCV` only (no `GridSearchCV`).
- Supports sparse text (`N-Gram`, `TF-IDF`) and dense embeddings
  (`Word2Vec`, `Law2Vec`, `Patent2Vec`, `Doc2Vec`).
- Writes each run result into a shared JSON file with file locking so parallel
  jobs can append safely.
- Optionally evaluates the best CV model on a provided heldout test set and
  records those metrics.

Last Updated: 07.04.26

Status: Done
"""

import json
import os
import re
import time
import warnings
from datetime import datetime, timezone

import fcntl
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC

from Utilities.utils import TextProcess, Word2VecTransform

warnings.filterwarnings("ignore", category=UserWarning)

class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Filters out stopwords, numbers and performs lemmatisation as configurable hyperparameters"""

    def __init__(self, stopwords=False, numbers=False, lemmatisation=False, opposition=False):
        self.stopwords = stopwords
        self.numbers = numbers
        self.lemmatisation = lemmatisation
        self.opposition = opposition

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tp = TextProcess(
            stopwords=self.stopwords,
            numbers=self.numbers,
            lemmatisation=self.lemmatisation,
        )

        if self.opposition:
            output = X.copy()
            output["New Summary Facts"] = tp.fit_transform(output["New Summary Facts"])
            return output

        return tp.fit_transform(X)

class Experiments:
    """Run flat CV experiments for N-gram/TF-IDF and embedding pipelines."""

    def __init__(
        self,
        models=None,
        experiment="1",
        opposition=False,
        case_mode=None,
        cv_num=3,
        num_iter=50,
        use_embeddings=False,
        repeat=1,
        no_grid=False,
        input_representation="N-Gram",
        results_json_path="results_main.json",
    ):
        self.models = models or []
        self.experiment = experiment
        self.opposition = opposition
        self.case_mode = (case_mode or "unknown").lower()
        self.num_iter = num_iter
        self.use_embeddings = use_embeddings
        self.no_grid = no_grid
        self.repeat = repeat
        self.input_representation = input_representation
        self.results_json_path = results_json_path

        self.set_params()

        self.params = []
        self.results = []

        if "linear" in self.models:
            self.params.append(self.linear)
        if "logistic" in self.models:
            self.params.append(self.logistic)
        if "forest" in self.models:
            self.params.append(self.forest)
        if "xgboost" in self.models:
            self.params.append(self.xgboost)

        self.cv_num = TimeSeriesSplit(n_splits=cv_num) if self.experiment == "2" else cv_num

        self.scoring = {
            "Accuracy": "accuracy",
            "F1": "f1",
            "Precision": "precision",
            "Recall": "recall",
            "MCC": make_scorer(matthews_corrcoef),
            "AUC": "roc_auc",
        }

    def _aux_feature_columns(self, X):
        """Return all non-text feature columns for structured inputs."""
        if not hasattr(X, "columns"):
            return []
        return [column for column in X.columns if column != "New Summary Facts"]

    def set_params(self):
        """Define hyperparameter spaces for all supported models."""
        self.num = "num__" if self.opposition else ""

        self.linear = {
            "clf": [LinearSVC(random_state=42)],
            "vect": [TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), Word2VecTransform],
            "scaler": [StandardScaler()],
            "prep__stopwords": [True, False],
            "prep__numbers": [True, False],
            "prep__lemmatisation": [True, False],
            f"vect__{self.num}ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
            "clf__C": [0.1, 1, 10, 100],
            f"vect__{self.num}norm": [None, "l2"],
            f"vect__{self.num}min_df": [2, 5, 10],
            f"vect__{self.num}use_idf": [True, False],
            "word2vec": ["word2vec", "law2vec", "patent2vec", "doc2vec"],
        }

        self.logistic = {
            "clf": [LogisticRegression(random_state=42)],
            "vect": [TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), Word2VecTransform],
            "scaler": [StandardScaler()],
            "prep__stopwords": [True, False],
            "prep__numbers": [True, False],
            "prep__lemmatisation": [True, False],
            f"vect__{self.num}ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
            "clf__C": [0.1, 1, 10, 100],
            "clf__solver": ["lbfgs", "sag"],
            "clf__penalty": [None, "l2"],
            "clf__max_iter": [100, 250, 500],
            f"vect__{self.num}norm": [None, "l2"],
            f"vect__{self.num}min_df": [2, 5, 10],
            f"vect__{self.num}use_idf": [True, False],
            "word2vec": ["word2vec", "law2vec", "patent2vec", "doc2vec"],
        }

        self.forest = {
            "clf": [RandomForestClassifier(random_state=42)],
            "vect": [TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), Word2VecTransform],
            "scaler": [StandardScaler()],
            "prep__stopwords": [True, False],
            "prep__numbers": [True, False],
            "prep__lemmatisation": [True, False],
            f"vect__{self.num}ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
            "clf__n_estimators": [100, 200, 300],
            "clf__max_features": ["sqrt", "log2"],
            "clf__max_depth": [10, 50, 100, None],
            f"vect__{self.num}norm": [None, "l2"],
            f"vect__{self.num}min_df": [2, 5, 10],
            f"vect__{self.num}use_idf": [True, False],
            "word2vec": ["word2vec", "law2vec", "patent2vec", "doc2vec"],
        }

        self.xgboost = {
            "clf": [xgb.XGBClassifier(random_state=42, objective="binary:logistic", tree_method="hist", device="cpu")],
            "vect": [TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), Word2VecTransform],
            "scaler": [StandardScaler()],
            "prep__stopwords": [True, False],
            "prep__numbers": [True, False],
            "prep__lemmatisation": [True, False],
            f"vect__{self.num}ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
            "clf__gamma": [0.0, 0.1, 0.2],
            "clf__max_depth": [3, 6, 9],
            f"vect__{self.num}norm": [None, "l2"],
            f"vect__{self.num}min_df": [2, 5, 10],
            f"vect__{self.num}use_idf": [True, False],
            "word2vec": ["word2vec", "law2vec", "patent2vec", "doc2vec"],
        }

    def text_preprocess(self, X_train, X_test):
        """Convert raw text to spaCy docs once before model selection for both train and test"""
        tp = TextProcess()

        if not self.opposition:
            self.X_train = [doc for doc in tp.nlp.pipe(X_train["New Summary Facts"].tolist())]
            self.X_test = [doc for doc in tp.nlp.pipe(X_test["New Summary Facts"].tolist())]
        else:
            self.X_train = X_train.copy()
            self.X_train["New Summary Facts"] = [doc for doc in tp.nlp.pipe(X_train["New Summary Facts"].tolist())]
            self.X_test = X_test.copy()
            self.X_test["New Summary Facts"] = [doc for doc in tp.nlp.pipe(X_test["New Summary Facts"].tolist())]

    def define_best_params(self, results):
        """Use best parameter set built from prior cv results."""
        param = {
            "clf": [self.model_selector(results["algo"])],
            "vect": [TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), Word2VecTransform],
            "scaler": [StandardScaler()],
            "word2vec": [results["embedding"]],
        }

        best_params = {key: [value] for key, value in results["params"].items()}
        param.update(best_params)
        self.params.append(param)

    def model_selector(self, clf):
        """Recreate estimator instance from stored algorithm name string."""
        if clf.startswith("Log"):
            return LogisticRegression(random_state=42)
        if clf.startswith("Lin"):
            return LinearSVC(random_state=42)
        if clf.startswith("Ran"):
            return RandomForestClassifier(random_state=42)
        return xgb.XGBClassifier(random_state=42, objective="binary:logistic", tree_method="hist", device="cpu")

    def training_loop(self, X_train, y_train, X_test=None, y_test=None):
        """Run flat-CV search for sparse text features (N-Gram/TF-IDF)."""
        print(f"\n{'='*70}")
        print(f"Starting sparse-feature training (Experiment {self.experiment})")
        print(f"Opposition: {self.opposition} | Input: {self.input_representation}")
        print(f"{'='*70}")
        
        self.text_preprocess(X_train, X_test)

        for param in self.params:
            param = param.copy()

            self.clf = param.pop("clf")[0]
            vect_list = param.pop("vect")
            self.vects_n = vect_list[0]
            self.vects_we = vect_list[1] if len(vect_list) > 1 else vect_list[0]
            self.scal = param.pop("scaler")[0]
            self.word2vec = param.pop("word2vec")

            if self.input_representation == "N-Gram":
                param[f"vect__{self.num}use_idf"] = [False]
                self.input_name = "N-Grams"
            elif self.input_representation == "TF-IDF":
                param[f"vect__{self.num}norm"] = ["l2"]
                param[f"vect__{self.num}use_idf"] = [True]
                self.input_name = "TF-IDF"
            else:
                self.input_name = self.input_representation

            if self.opposition:
                aux_columns = self._aux_feature_columns(self.X_train)
                preprocessor = ColumnTransformer(
                    transformers=[("num", self.vects_n, "New Summary Facts"), ("Cats", Binarizer(), aux_columns)],
                )
                steps = Pipeline([("prep", PreprocessTransformer(opposition=True)), ("vect", preprocessor), ("clf", self.clf)])
            else:
                steps = Pipeline([("prep", PreprocessTransformer(opposition=False)), ("vect", self.vects_n), ("clf", self.clf)])

            clf_name = self.name_process(self.clf)
            print(f"  [{clf_name:8s} + {self.input_name:8s}] Running CV search...")

            self.training_ngram_core(param, steps, y_train, clf_name, y_test=y_test)

            if self.use_embeddings:
                self.training_loop_we(X_train, y_train, X_test=X_test, y_test=y_test)

        print(f"\nCompleted {len(self.results)} sparse-feature experiment(s)\n")
        return self.results

    def training_ngram_core(self, param, steps, y_train, clf_name, y_test=None):
        """Run one randomized flat-CV search for sparse text features."""
        X_train_ = self.X_train.copy()

        cv = self.cv_num if self.experiment == "2" else RepeatedStratifiedKFold(
            n_splits=self.cv_num,
            n_repeats=self.repeat,
            random_state=42,
        )

        search = RandomizedSearchCV(
            steps,
            param_distributions=param,
            n_iter=self.num_iter,
            cv=cv,
            n_jobs=1,
            scoring=self.scoring,
            refit="F1",
            random_state=42,
        )

        start = time.time()
        search.fit(X_train_, self._to_1d(y_train))
        elapsed = time.time() - start

        test_metrics = None
        if self.X_test is not None and y_test is not None:
            X_test_ = self.X_test.copy()
            test_metrics = self._compute_test_metrics(search.best_estimator_, X_test_, y_test)

        preprocess = {
            "stopwords": search.best_params_.get("prep__stopwords", False),
            "numbers": search.best_params_.get("prep__numbers", False),
            "lemma": search.best_params_.get("prep__lemmatisation", False),
        }

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "sparse",
            "algo": clf_name,
            "experiment": self.experiment,
            "opposition": self.opposition,
            "case_mode": self.case_mode,
            "input_representation": self.input_name,
            "preprocess": preprocess,
            "best_score_cv": search.best_score_,
            "best_params": search.best_params_,
            "time_seconds": elapsed,
            "test_metrics": test_metrics,
        }

        self.results.append(record)
        self._append_json_result(record)
        print(f"      ✓ CV F1: {search.best_score_:.4f} ({elapsed:.1f}s)")

    def training_loop_we(self, X_train, y_train, X_test=None, y_test=None):
        """Run flat-CV search for dense embedding inputs."""
        self.text_preprocess(X_train, X_test)

        for param in self.params:
            param = param.copy()

            self.clf = param.pop("clf")[0]
            vect_list = param.pop("vect")
            self.vects_n = vect_list[0]
            self.vects_we = vect_list[1] if len(vect_list) > 1 else vect_list[0]
            self.scal = param.pop("scaler")[0]
            self.word2vec = param.pop("word2vec")
            clf_name = self.name_process(self.clf)

            for key in [
                f"vect__{self.num}min_df",
                f"vect__{self.num}norm",
                f"vect__{self.num}ngram_range",
                f"vect__{self.num}use_idf",
                "prep__stopwords",
                "prep__numbers",
                "prep__lemmatisation",
            ]:
                param.pop(key, None)

            embedding_map = {
                "Word2Vec": ["word2vec"],
                "Law2Vec": ["law2vec"],
                "Patent2Vec": ["patent2vec"],
                "Doc2Vec": ["doc2vec"],
            }
            if self.input_representation in embedding_map:
                self.word2vec = embedding_map[self.input_representation]

            steps = Pipeline([("scal", self.scal), ("clf", self.clf)])

            self.training_w2v_core(
                param,
                steps,
                y_train,
                clf_name,
                y_test=y_test,
            )

        return self.results

    def training_w2v_core(self, param, steps, y_train, clf_name, y_test=None):
        """Run one randomized flat-CV search for embedding-based features."""
        tp = TextProcess()

        if self.opposition:
            X_train_ = self.X_train.copy()
            X_train_["New Summary Facts"] = tp.fit_transform(X_train_["New Summary Facts"])
        else:
            X_train_ = tp.fit_transform(self.X_train.copy())

        for embedding in self.word2vec:
            if self.opposition:
                aux_columns = self._aux_feature_columns(X_train_)
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", Word2VecTransform(embedding=embedding, opposition=True), "New Summary Facts"),
                        ("Cats", Binarizer(), aux_columns),
                    ]
                )
                X_train_embed = preprocessor.fit_transform(X_train_)
            else:
                w2v = self.vects_we(embedding)
                X_train_embed = w2v.fit_transform(X_train_)

            cv = self.cv_num if self.experiment == "2" else RepeatedStratifiedKFold(
                n_splits=self.cv_num,
                n_repeats=self.repeat,
                random_state=42,
            )

            search = RandomizedSearchCV(
                steps,
                param_distributions=param,
                n_iter=self.num_iter,
                cv=cv,
                n_jobs=1,
                scoring=self.scoring,
                refit="F1",
                random_state=42,
            )

            start = time.time()
            search.fit(X_train_embed, self._to_1d(y_train))
            elapsed = time.time() - start

            test_metrics = None
            if self.X_test is not None and y_test is not None:
                if self.opposition:
                    X_test_ = self.X_test.copy()
                    X_test_["New Summary Facts"] = tp.fit_transform(X_test_["New Summary Facts"])
                    aux_columns = self._aux_feature_columns(X_test_)
                    preprocessor_test = ColumnTransformer(
                        transformers=[
                            ("num", Word2VecTransform(embedding=embedding, opposition=True), "New Summary Facts"),
                            ("Cats", Binarizer(), aux_columns),
                        ]
                    )
                    X_test_embed = preprocessor_test.fit_transform(X_test_)
                else:
                    X_test_ = tp.fit_transform(self.X_test.copy())
                    w2v_test = self.vects_we(embedding)
                    X_test_embed = w2v_test.fit_transform(X_test_)

                test_metrics = self._compute_test_metrics(search.best_estimator_, X_test_embed, y_test)

            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "embedding",
                "algo": clf_name,
                "experiment": self.experiment,
                "opposition": self.opposition,
                "case_mode": self.case_mode,
                "input_representation": self.input_representation,
                "embedding": embedding,
                "preprocess": {},
                "best_score_cv": search.best_score_,
                "best_params": search.best_params_,
                "time_seconds": elapsed,
                "test_metrics": test_metrics,
            }

            self.results.append(record)
            self._append_json_result(record)

    def _compute_test_metrics(self, estimator, X_test, y_test):
        y_true = self._to_1d(y_test)
        y_pred = estimator.predict(X_test)

        score = None
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                score = proba[:, 1]
        elif hasattr(estimator, "decision_function"):
            score = estimator.decision_function(X_test)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": None,
        }

        if score is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, score)
            except ValueError:
                metrics["auc"] = None

        return metrics

    def _append_json_result(self, record):
        """Append one record to the shared JSON file using an exclusive file lock."""
        path = self.results_json_path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "a+", encoding="utf-8") as file_handle:
            fcntl.flock(file_handle, fcntl.LOCK_EX)
            file_handle.seek(0)
            content = file_handle.read().strip()

            if content:
                try:
                    data = json.loads(content)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
            else:
                data = []

            data.append(self._jsonable(record))

            file_handle.seek(0)
            file_handle.truncate()
            json.dump(data, file_handle, indent=2)
            file_handle.flush()
            os.fsync(file_handle.fileno())
            fcntl.flock(file_handle, fcntl.LOCK_UN)

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {k: self._jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _to_1d(self, y):
        if isinstance(y, pd.DataFrame):
            return y.iloc[:, 0].to_numpy()
        if isinstance(y, pd.Series):
            return y.to_numpy()
        return np.asarray(y).reshape(-1)

    def name_process(self, clf):
        """Return model class name from estimator repr."""
        return re.split(r"\(", str(clf))[0]
