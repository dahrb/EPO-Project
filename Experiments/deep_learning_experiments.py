"""Patent Experiments - Deep Learning.

Transformer-based flat-CV experiment runner for heldout-test workflows.

Key behaviour
-------------
- Supports legalbert, longformer_base, longformer_large, roberta_base, roberta_large.
- Hyperparameter search via RandomizedSearchCV-style ParameterSampler over a
  defined grid; n_iter controls the number of sampled combos (default 10).
- CV strategy mirrors ml_experiments: Exp 1 = RepeatedStratifiedKFold,
  Exp 2 = TimeSeriesSplit.
- Best combo retrained on full training set; evaluated on heldout test set.
- Loss:          categorical cross-entropy (CrossEntropyLoss)
- Optimiser:     AdamW with optional weight decay and linear warmup scheduler
- Initialisation: Glorot uniform (xavier_uniform_) on all custom Linear layers.
- Supports opposition mode: CLS token fused with one-hot auxiliary features.

Last Updated: 09.04.26

Status: In Progress
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
import random
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import ParameterSampler

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader

from Utilities.utils import TextProcess

warnings.filterwarnings("ignore", category=UserWarning)

class PatentTextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for patent text with optional structured features."""

    def __init__(self, input_ids, attention_masks, labels, auxiliary_features=None):
        """
        Args:
            input_ids: Tokenized input IDs from BERT tokenizer
            attention_masks: Attention masks from tokenizer
            labels: Binary labels (0 or 1)
            auxiliary_features: Optional numpy array of one-hot encoded features (opposition mode)
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.auxiliary_features = auxiliary_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.auxiliary_features is not None:
            batch["auxiliary_features"] = torch.tensor(self.auxiliary_features[idx], dtype=torch.float32)
        return batch

class OppositionModeClassificationHead(torch.nn.Module):
    """Classification head that fuses text embeddings with one-hot encoded auxiliary features.
    
    OPPOSITION MODE HANDLING (Option A):
    - Text features: Encoded by BERT → pooled representation (CLS token)
    - Auxiliary features: Structured columns (Matches_1, Matches_2, Category, etc.) 
                         one-hot encoded and normalized
    - Fusion: Concatenate text embedding + auxiliary features → linear layer → output
    """

    def __init__(self, text_embed_dim, aux_feature_dim, num_labels=2):
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.aux_feature_dim = aux_feature_dim
        self.fused_dim = text_embed_dim + aux_feature_dim
        
        self.fusion_layer = torch.nn.Linear(self.fused_dim, 256)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(256, num_labels)

        # Glorot (Xavier) uniform initialisation on custom linear layers
        torch.nn.init.xavier_uniform_(self.fusion_layer.weight)
        torch.nn.init.zeros_(self.fusion_layer.bias)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, text_embedding, auxiliary_features):
        """
        Args:
            text_embedding: BERT CLS token [batch_size, text_embed_dim]
            auxiliary_features: One-hot features [batch_size, aux_feature_dim]
        Returns:
            logits: [batch_size, num_labels]
        """
        fused = torch.cat([text_embedding, auxiliary_features], dim=1)
        hidden = torch.relu(self.fusion_layer(fused))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits

class DeepLearningExperiments:
    """Run flat CV experiments for transformer-based models (LegalBERT)."""

    def __init__(
        self,
        model_name="legalbert",
        experiment="1",
        opposition=False,
        input_representation=None,
        results_json_path="results_deep_learning.json",
        device=None,
        cv_num=3,
        repeat=1,
        n_iter=10,
    ):
        """
        Args:
            model_name: Model identifier (e.g., 'legalbert', 'longformer_base', 'longformer_large')
            experiment: Experiment number ('1' or '2')
            opposition: Whether to include structured auxiliary features
            input_representation: Name of embedding model (e.g., 'LegalBERT')
            results_json_path: Path to JSON results file
            device: torch device (auto-detect if None)
            cv_num: Number of CV folds (default 3, matches ml_experiments)
            repeat: Number of CV repeats for Exp 1 (default 1, matches ml_experiments)
            n_iter: Number of random param combos to sample from the grid (default 10)
        """
        self.model_name = model_name
        self.experiment = experiment
        self.opposition = opposition
        self.input_representation = input_representation if input_representation is not None else model_name
        self.results_json_path = results_json_path
        self.results = []
        self.n_iter = n_iter

        # CV splitter — mirrors ml_experiments exactly:
        #   Exp 1: RepeatedStratifiedKFold (stratified, shuffled)
        #   Exp 2: TimeSeriesSplit (chronological, no shuffle)
        if experiment == "2":
            self.cv = TimeSeriesSplit(n_splits=cv_num)
        else:
            self.cv = RepeatedStratifiedKFold(
                n_splits=cv_num,
                n_repeats=repeat,
                random_state=42,
            )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Model mapping with context lengths (tokens)
        self.model_mapping = {
            "legalbert":       ("nlpaueb/legal-bert-base-uncased",   512),
            "longformer_base": ("lexlms/legal-longformer-base",      4096),
            "longformer_large":("lexlms/legal-longformer-large",     4096),
            "roberta_base":    ("lexlms/legal-roberta-base",         512),
            "roberta_large":   ("lexlms/legal-roberta-large",        512),
        }

        if model_name.lower() not in self.model_mapping:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_mapping.keys())}")

        self.hf_model_name, self.max_length = self.model_mapping[model_name.lower()]

        # Hyperparameter grid — sampled randomly via ParameterSampler (mirrors
        # RandomizedSearchCV from ml_experiments). Includes literature-grounded
        # settings (10 epochs, bs=8, wd=0.01, 500 warmup; Chalkidis et al.).
        self.param_grid = {
            "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
            "batch_size":    [8, 16, 32],
            "epochs":        [3, 4, 5, 10],
            "dropout":       [0.1, 0.2],
            "weight_decay":  [0.0, 0.01],
            "warmup_steps":  [0, 500],
        }

        # Initialize tokenizer (shared across all runs)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        except ValueError as e:
            if "backend tokenizer" in str(e):
                print(f"[Warning] Fast tokenizer unavailable for {model_name}, falling back to slow tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, use_fast=False)
            else:
                raise
        
        # Log model and context length information
        print(f"[Model Info] Using {model_name} ({self.hf_model_name}) with max_length={self.max_length}")

        self.X_test = None
        self.y_test = None
        self.aux_encoder = None

    def _get_encoder(self, model):
        """Return the base transformer encoder, handling both BERT and RoBERTa architectures.

        BERT-family models expose the encoder as ``model.bert``;
        RoBERTa-family models expose it as ``model.roberta``.
        This helper makes opposition-mode CLS extraction architecture-agnostic.
        """
        if hasattr(model, "bert"):
            return model.bert
        if hasattr(model, "roberta"):
            return model.roberta
        raise AttributeError(
            f"Cannot find a .bert or .roberta encoder on {type(model).__name__}. "
            "Add support for this architecture in _get_encoder."
        )

    def _preprocess_text_for_bert(self, X):
        """Apply fixed text preprocessing for BERT inputs.

        Always enforces:
        - stopword removal
        - lemmatization
        - number removal
        """
        if not hasattr(X, "copy"):
            return X

        X_processed = X.copy()
        if "New Summary Facts" not in X_processed.columns:
            return X_processed

        text_processor = TextProcess(stopwords=True, numbers=False, lemmatisation=True)
        docs = [str(text) for text in X_processed["New Summary Facts"].tolist()]
        processed_tokens = text_processor.fit_transform(docs)
        X_processed["New Summary Facts"] = [" ".join(tokens) for tokens in processed_tokens]
        return X_processed

    def _to_1d(self, y):
        """Convert y to 1D numpy array."""
        if isinstance(y, pd.DataFrame):
            return y.iloc[:, 0].to_numpy()
        if isinstance(y, pd.Series):
            return y.to_numpy()
        return np.asarray(y).reshape(-1)

    def _get_aux_feature_columns(self, X):
        """Return all non-text feature columns for opposition mode."""
        if not hasattr(X, "columns"):
            return []
        return [col for col in X.columns if col != "New Summary Facts"]

    def _encode_opposition_features(self, X_data, fit=False):
        """One-hot encode auxiliary features for opposition mode.
        
        Args:
            X_data: DataFrame with auxiliary columns
            fit: If True, fit encoder; if False, transform using existing encoder
        
        Returns:
            One-hot encoded features as numpy array [n_samples, n_aux_features]
        """
        aux_cols = self._get_aux_feature_columns(X_data)
        if not aux_cols:
            return None

        X_aux = X_data[aux_cols].copy()

        if fit:
            self.aux_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            aux_encoded = self.aux_encoder.fit_transform(X_aux)
        else:
            if self.aux_encoder is None:
                raise ValueError("Encoder not fitted. Call with fit=True first.")
            aux_encoded = self.aux_encoder.transform(X_aux)

        # Normalize to [0, 1]
        aux_encoded = aux_encoded.astype(np.float32)
        return aux_encoded

    def _tokenize_batch(self, texts, max_length=None):
        """Tokenize batch of texts using BERT tokenizer.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (uses model's context length if None)
        
        Returns:
            input_ids, attention_masks (both as numpy arrays)
        """
        if max_length is None:
            max_length = self.max_length
        
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return encodings["input_ids"], encodings["attention_mask"]

    def _prepare_full(self, X_train, y_train, X_test, y_test):
        """Preprocess full training and test sets (no splitting — folds handle that).

        Mirrors ml_experiments: preprocessing is applied once up front; the CV
        splitter then partitions indices into train/val folds per param combo.
        """
        y_train_1d = self._to_1d(y_train)
        y_test_1d = self._to_1d(y_test)

        X_train_proc = self._preprocess_text_for_bert(X_train.reset_index(drop=True))
        X_test_proc = self._preprocess_text_for_bert(X_test.reset_index(drop=True))

        print(
            f"[Data] Train: {len(X_train_proc)}, Test: {len(X_test_proc)} | "
            f"Exp {self.experiment} CV: {self.cv.__class__.__name__}"
        )
        return X_train_proc, y_train_1d, X_test_proc, y_test_1d

    def _create_dataloader(self, X, y, is_train=True, batch_size=32):
        """Create PyTorch DataLoader for given data.
        
        Args:
            X: Features DataFrame
            y: Labels array
            is_train: Whether this is training set (affects shuffle)
            batch_size: Batch size
        
        Returns:
            DataLoader
        """
        texts = X["New Summary Facts"].tolist()
        input_ids, attention_masks = self._tokenize_batch(texts)

        if self.opposition:
            aux_features = self._encode_opposition_features(X, fit=(is_train and self.aux_encoder is None))
            dataset = PatentTextDataset(input_ids, attention_masks, y, auxiliary_features=aux_features)
        else:
            dataset = PatentTextDataset(input_ids, attention_masks, y)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
        )
        return dataloader

    def _train_epoch(self, model, optimizer, train_loader, criterion, scheduler=None):
        """Run one epoch of training.
        
        Args:
            model: Transformer model (with custom head for opposition mode)
            optimizer: PyTorch optimizer
            train_loader: DataLoader for training
            criterion: Loss function
            scheduler: Optional learning-rate scheduler (stepped per batch)
        
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            if self.opposition:
                # Extract CLS token from transformer encoder (BERT or RoBERTa)
                outputs = self._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
                text_embedding = outputs.pooler_output  # [batch_size, hidden_size]
                auxiliary_features = batch["auxiliary_features"].to(self.device)
                logits = self.custom_head(text_embedding, auxiliary_features)
            else:
                # Standard classification
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, criterion):
        """Run validation epoch.
        
        Args:
            model: BERT model
            val_loader: DataLoader for validation
            criterion: Loss function
        
        Returns:
            (average_loss, f1_score)
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.opposition:
                    outputs = self._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
                    text_embedding = outputs.pooler_output
                    auxiliary_features = batch["auxiliary_features"].to(self.device)
                    logits = self.custom_head(text_embedding, auxiliary_features)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits

                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return avg_loss, f1

    def training_loop(self, X_train, y_train, X_test=None, y_test=None):
        """Hyperparameter search via fold-based CV matching ml_experiments exactly.

        CV strategy (mirrors ml_experiments):
          Exp 1: RepeatedStratifiedKFold — stratified, shuffled folds
          Exp 2: TimeSeriesSplit        — chronological folds, no shuffle

        For each param combo the mean val F1 across all folds is computed.
        The best combo is then retrained on the full training set and
        evaluated on the held-out test set.
        """
        # Preprocess once up front (no splitting yet — folds handle that)
        X_train_proc, y_train_1d, X_test_proc, y_test_1d = self._prepare_full(
            X_train, y_train,
            X_test if X_test is not None else X_train,
            y_test if y_test is not None else y_train,
        )
        self.X_test = X_test_proc
        self.y_test = y_test_1d

        return self._run_one_model(
            X_train_proc, y_train_1d,
            X_test_proc, y_test_1d,
            X_test is not None and y_test is not None,
        )

    def _run_one_model(self, X_train_proc, y_train_1d, X_test_proc, y_test_1d, has_test):
        """Randomised hyperparameter search via fold-based CV, then retrain + test.

        Hyperparameter combos are sampled randomly from ``self.param_grid`` using
        sklearn's ParameterSampler (mirrors RandomizedSearchCV from ml_experiments).
        ``self.n_iter`` controls the budget (default 10 combos).
        """
        # Reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Sample n_iter combos from the full grid — identical semantics to
        # RandomizedSearchCV(n_iter=self.n_iter, random_state=42)
        param_combinations = list(
            ParameterSampler(self.param_grid, n_iter=self.n_iter, random_state=42)
        )
        print(
            f"[Search] Sampling {len(param_combinations)} combos from grid "
            f"(n_iter={self.n_iter}) for model '{self.model_name}'"
        )

        start_time = time.time()
        best_mean_f1 = -1.0
        best_params = None

        # ── Phase 1: cross-validate each param combo ──────────────────────────
        for param_idx, params in enumerate(param_combinations):
            print(f"\n[{param_idx + 1}/{len(param_combinations)}] CV for params: {params}")

            fold_f1s = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_train_proc, y_train_1d)):
                X_fold_train = X_train_proc.iloc[train_idx].reset_index(drop=True)
                X_fold_val   = X_train_proc.iloc[val_idx].reset_index(drop=True)
                y_fold_train = y_train_1d[train_idx]
                y_fold_val   = y_train_1d[val_idx]

                # Reset aux encoder per fold so it fits on fold-train only
                self.aux_encoder = None

                model = self._build_model(params["dropout"])
                optimizer = self._build_optimizer(model, params["learning_rate"], params["weight_decay"])
                criterion = torch.nn.CrossEntropyLoss()

                train_loader = self._create_dataloader(
                    X_fold_train, y_fold_train, is_train=True,
                    batch_size=params["batch_size"]
                )
                val_loader = self._create_dataloader(
                    X_fold_val, y_fold_val, is_train=False,
                    batch_size=params["batch_size"]
                )
                self._fix_opposition_head(model, train_loader)

                total_steps = len(train_loader) * params["epochs"]
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=params["warmup_steps"],
                    num_training_steps=total_steps,
                )

                # Train with early stopping on val F1
                best_fold_f1 = 0.0
                patience_counter = 0
                patience = 2
                for epoch in range(params["epochs"]):
                    train_loss = self._train_epoch(model, optimizer, train_loader, criterion, scheduler)
                    val_loss, val_f1 = self._validate_epoch(model, val_loader, criterion)
                    print(
                        f"  Fold {fold_idx + 1} Epoch {epoch + 1}/{params['epochs']}: "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
                    )
                    if val_f1 > best_fold_f1:
                        best_fold_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"  Early stopping at epoch {epoch + 1}")
                            break

                fold_f1s.append(best_fold_f1)
                del model  # free GPU memory between folds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean_f1 = float(np.mean(fold_f1s))
            print(f"  → Mean CV F1: {mean_f1:.4f} (folds: {[round(f,4) for f in fold_f1s]})")

            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_params = params.copy()

        print(f"\nBest params: {best_params} | Mean CV F1: {best_mean_f1:.4f}")

        # ── Phase 2: retrain best params on full training set ─────────────────
        print("\nRetraining best params on full training set...")
        self.aux_encoder = None
        best_model = self._build_model(best_params["dropout"])
        best_optimizer = self._build_optimizer(best_model, best_params["learning_rate"], best_params["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()

        full_train_loader = self._create_dataloader(
            X_train_proc, y_train_1d, is_train=True,
            batch_size=best_params["batch_size"]
        )
        self._fix_opposition_head(best_model, full_train_loader)

        retrain_total_steps = len(full_train_loader) * best_params["epochs"]
        retrain_scheduler = get_linear_schedule_with_warmup(
            best_optimizer,
            num_warmup_steps=best_params["warmup_steps"],
            num_training_steps=retrain_total_steps,
        )

        for epoch in range(best_params["epochs"]):
            train_loss = self._train_epoch(best_model, best_optimizer, full_train_loader, criterion, retrain_scheduler)
            print(f"  Retrain Epoch {epoch + 1}/{best_params['epochs']}: train_loss={train_loss:.4f}")

        elapsed = time.time() - start_time

        # ── Phase 3: evaluate on held-out test set ────────────────────────────
        test_metrics = None
        if has_test:
            test_metrics = self._compute_test_metrics(best_model, X_test_proc, y_test_1d)

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "deep_learning",
            "algo": self.model_name,
            "model_name": self.hf_model_name,
            "max_context_length": self.max_length,
            "experiment": self.experiment,
            "cv_strategy": self.cv.__class__.__name__,
            "opposition": self.opposition,
            "input_representation": self.input_representation,
            "best_score_cv": best_mean_f1,
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
            "time_seconds": elapsed,
            "test_metrics": test_metrics,
        }

        self.results.append(record)
        self._append_json_result(record)
        return [record]

    def _build_model(self, dropout):
        """Load a fresh model instance with the given dropout, sent to device."""
        print(f"  Loading model {self.hf_model_name} (max_length={self.max_length})...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_name,
            num_labels=2,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        model.to(self.device)

        if self.opposition:
            model.classifier = None
            self.custom_head = OppositionModeClassificationHead(
                text_embed_dim=model.config.hidden_size,
                aux_feature_dim=None,
                num_labels=2,
            )
            self.custom_head.to(self.device)

        return model

    def _build_optimizer(self, model, lr, weight_decay=0.0):
        """Build AdamW optimizer over the correct parameter set."""
        if self.opposition:
            params = list(self._get_encoder(model).parameters()) + list(self.custom_head.parameters())
        else:
            params = list(model.parameters())
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    def _fix_opposition_head(self, model, train_loader):
        """Infer aux feature dim from the first batch and wire up the fusion layer."""
        if not self.opposition:
            return
        for batch in train_loader:
            aux_dim = batch["auxiliary_features"].shape[1]
            self.custom_head.aux_feature_dim = aux_dim
            self.custom_head.fused_dim = model.config.hidden_size + aux_dim
            new_layer = torch.nn.Linear(self.custom_head.fused_dim, 256).to(self.device)
            torch.nn.init.xavier_uniform_(new_layer.weight)
            torch.nn.init.zeros_(new_layer.bias)
            self.custom_head.fusion_layer = new_layer
            break

    def _compute_test_metrics(self, model, X_test, y_test):
        """Compute metrics on test set.
        
        Args:
            model: Trained BERT model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with accuracy, f1, precision, recall, mcc, auc
        """
        test_loader = self._create_dataloader(X_test, y_test, is_train=False, batch_size=32)

        model.eval()
        all_preds = []
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.opposition:
                    outputs = self._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
                    text_embedding = outputs.pooler_output
                    auxiliary_features = batch["auxiliary_features"].to(self.device)
                    logits = self.custom_head(text_embedding, auxiliary_features)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(all_labels, all_preds)),
            "auc": None,
        }

        try:
            metrics["auc"] = float(roc_auc_score(all_labels, all_scores))
        except ValueError:
            metrics["auc"] = None

        return metrics

    def _append_json_result(self, record):
        """Append result to JSON file with file locking (same pattern as ml_experiments)."""
        path = self.results_json_path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()

            if content:
                try:
                    data = json.loads(content)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
            else:
                data = []

            data.append(record)

            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
