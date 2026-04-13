"""Patent Experiments - Deep Learning.

Transformer-based experiment runner patent experiments.
Supports LegalBERT, Longformer, and Hierarchical BERT (H-BERT).

Key behaviour
-------------
- Hyperparameter search via Optuna (TPE sampler).
  Key Optuna settings exposed on DeepLearningExperiments:

    n_trials  (int,  default 10)  – number of trials Optuna runs.
    timeout   (int|None, default None) – wall-clock budget in seconds; study
                                        stops whichever comes first (n_trials
                                        or timeout).
    sampler   – any optuna.samplers.*  (default TPESampler(seed=42)).

- Each trial trains on a static train/val split and returns val-F1.
- Validation split:
    Exp 1: StratifiedShuffleSplit (random stratified, 80/20).
    Exp 2: Temporal split (last 20% of training data).
- Stores best validation performance from Optuna trials.
- Loss:          categorical cross-entropy (CrossEntropyLoss)
- Optimiser:     AdamW with optional weight decay and linear warmup scheduler
- Initialisation: Glorot uniform (xavier_uniform_) on all custom Linear layers.
- Supports opposition mode: CLS token fused with one-hot auxiliary features.

Last Updated: 13.04.26

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
from sklearn.model_selection import StratifiedShuffleSplit

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader

from Experiments.hierarchical_bert import HierarchicalPatentDataset, HierBertModel
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
    """Classification head that fuses text embeddings with one-hot encoded auxiliary features for opposition division cases.
    
    OPPOSITION MODE HANDLING:
    - Text features: Encoded by BERT → pooled representation (CLS token)
    - Auxiliary features: Structured columns One-hot encoded and normalized (The type of Opposition cases)
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

        #glorot initialisation
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
    """Run experiments for transformer-based models"""

    def __init__(
        self,
        model_name="legalbert",
        experiment="1",
        opposition=False,
        case_mode=None,
        input_representation=None,
        results_json_path="results_deep_learning.json",
        device=None,
        val_ratio=0.2,
        # ── Optuna settings ────────────────────────────────────────────────
        n_trials=10,
        timeout=None,
        sampler=None,
        # ── H-BERT settings ────────────
        chunk_size=128,
        max_chunks=64,
    ):
        """
        Args:
            model_name: Model identifier ('legalbert', 'longformer_base',
                'hbert')
            experiment: Experiment number ('1' or '2')
            opposition: Whether to include structured auxiliary features
            input_representation: Name of embedding model (e.g. 'LegalBERT')
            results_json_path: Path to JSON results file
            device: torch device (auto-detect if None)
            val_ratio: Fraction of training data used for validation (default 0.2).
                Exp 1 uses a random stratified split; Exp 2 uses a temporal
                (last val_ratio %) split.

            --- Optuna settings ---
            n_trials (int): Number of Optuna trials (default 10).
            timeout (int | None): Stop study after this many seconds regardless
                of n_trials (default None = no wall-clock limit).
            sampler: An optuna.samplers.* instance.  Defaults to
                TPESampler(seed=42) — Bayesian optimisation via Tree-structured
                Parzen Estimator.

            --- H-BERT settings---
            chunk_size (int): Tokens per chunk including special tokens (default 128).
            max_chunks (int): Maximum chunks per document (default 64).
        """
        
        self.model_name = model_name
        self.experiment = experiment
        self.opposition = opposition
        self.case_mode = (case_mode or "unknown").lower()
        self.input_representation = input_representation if input_representation is not None else model_name
        self.results_json_path = results_json_path
        self.results = []
        self.is_hbert = model_name.lower() == "hbert"
        self.is_longformer = "longformer" in model_name.lower()

        # ── H-BERT settings ────────────────────────────────────────────────
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks

        # ── Optuna settings ────────────────────────────────────────────────
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler = sampler if sampler is not None else optuna.samplers.TPESampler(seed=42)

        #validation split ratio
        #   Exp 1: random stratified split (StratifiedShuffleSplit)
        #   Exp 2: temporal split (last val_ratio of data)
        self.val_ratio = val_ratio

        #set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        #model mapping with context lengths (tokens)
        self.model_mapping = {
            "legalbert":       ("nlpaueb/legal-bert-base-uncased",   512),
            "longformer_base": ("lexlms/legal-longformer-base",      4096),
            "hbert":           ("nlpaueb/legal-bert-base-uncased",   512),
        }

        if model_name.lower() not in self.model_mapping:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_mapping.keys())}")

        self.hf_model_name, self.max_length = self.model_mapping[model_name.lower()]

        #initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

        #log model and context length information
        if self.is_hbert:
            print(
                f"[Model Info] Using H-BERT ({self.hf_model_name}) with "
                f"chunk_size={self.chunk_size}, max_chunks={self.max_chunks}"
            )
        else:
            print(f"[Model Info] Using {model_name} ({self.hf_model_name}) with max_length={self.max_length}")

        self.aux_encoder = None

    def _get_encoder(self, model):
        """Return the base transformer encoder across supported architectures.

        BERT-family models expose the encoder as ``model.bert``;
        RoBERTa-family models expose it as ``model.roberta``;
        Longformer exposes it as ``model.longformer``.
        This helper makes opposition-mode CLS extraction architecture-agnostic.
        """
        if hasattr(model, "bert"):
            return model.bert
        if hasattr(model, "roberta"):
            return model.roberta
        if hasattr(model, "longformer"):
            return model.longformer
        raise AttributeError(
            f"Cannot find a .bert, .roberta, or .longformer encoder on {type(model).__name__}. "
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
        raw = [str(text) for text in X_processed["New Summary Facts"].tolist()]
        spacy_docs = [doc for doc in text_processor.nlp.pipe(raw)]
        processed_tokens = text_processor.fit_transform(spacy_docs)
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

        #normalize to [0, 1]
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

    def _prepare_full(self, X_train, y_train):
        """Preprocess full training and test sets (no splitting — folds handle that).

        Mirrors ml_experiments: preprocessing is applied once up front; the CV
        splitter then partitions indices into train/val folds per param combo.
        """
        y_train_1d = self._to_1d(y_train)

        X_train_proc = self._preprocess_text_for_bert(X_train.reset_index(drop=True))

        print(
            f"[Data] Train: {len(X_train_proc)}"
            f"Exp {self.experiment}"
        )
        return X_train_proc, y_train_1d

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

        if self.is_hbert:
            # H-BERT: chunked hierarchical dataset
            if self.opposition:
                aux_features = self._encode_opposition_features(X, fit=(is_train and self.aux_encoder is None))
            else:
                aux_features = None
            
            dataset = HierarchicalPatentDataset(
                texts, y, self.tokenizer,
                chunk_size=self.chunk_size,
                max_chunks=self.max_chunks,
                auxiliary_features=aux_features,
            )
        else:
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
        
        For H-BERT: uses fp16 mixed-precision.
        For flat models: standard single-step backprop.
        
        Args:
            model: Transformer model (or HierBertModel for H-BERT)
            optimizer: PyTorch optimizer
            train_loader: DataLoader for training
            criterion: Loss function
            scheduler: Optional learning-rate scheduler (stepped per batch)
        
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0

        if self.is_hbert:
            scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

            for batch in train_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                cmask = batch["chunk_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                aux = batch.get("auxiliary_features")
                if aux is not None:
                    aux = aux.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = model(ids, mask, cmask, auxiliary_features=aux)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
        else:
            # Flat models training
            # Longformer: physical batch=2, accumulate 8 steps → effective batch=16
            accum_steps = 8 if self.is_longformer else 1

            for step, batch in enumerate(train_loader):
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

                loss = criterion(logits, labels) / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                total_loss += loss.item() * accum_steps 

        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, criterion):
        """Run validation epoch.
        
        Args:
            model: BERT model or HierBertModel
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
                labels = batch["labels"].to(self.device)

                if self.is_hbert:
                    ids = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    cmask = batch["chunk_mask"].to(self.device)
                    aux = batch.get("auxiliary_features")
                    if aux is not None:
                        aux = aux.to(self.device)

                    with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                        logits = model(ids, mask, cmask, auxiliary_features=aux)
                        loss = criterion(logits, labels)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

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

    def training_loop(self, X_train, y_train):
        """Hyperparameter search with a static train/val split.

        Validation strategy:
          Exp 1: StratifiedShuffleSplit — random stratified 80/20 split.
          Exp 2: Temporal split — last 20% as validation.

        This script records best validation trial performance only.
        """
        #preprocess - don't split yet
        X_train_proc, y_train_1d = self._prepare_full(
            X_train, y_train,
        )

        return self._run_model(X_train_proc, y_train_1d)

    def _run_model(self, X_train_proc, y_train_1d):
        """Optuna hyperparameter search with a static train/val split.

        Each Optuna trial:
          1. Samples a hyperparameter set via trial.suggest_*.
          2. Trains on the static training portion and evaluates on the
             static validation portion.
          3. Returns val-F1 as the objective value (maximise).

        Splitting strategy:
          Exp 1: StratifiedShuffleSplit — random stratified 80/20 split.
          Exp 2: Temporal split — last 20% as validation (preserves order).

        After n_trials (or timeout) the best trial validation metrics are
        recorded for model selection.
        """
        #reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        is_longformer = "longformer" in self.model_name.lower()

        # ── Create static train/val split indices ─────────────────────────
        n = len(X_train_proc)
       
        if self.experiment == "2":
            #temporal split: explicitly sort by time when a date-like column exists.
            date_candidates = [
                "Date", "date", "Decision Date", "decision_date",
                "DecisionDate", "decisionDate", "Year", "year",
            ]
            date_col = next((col for col in date_candidates if col in X_train_proc.columns), None)

            if date_col is not None:
                if "year" in date_col.lower() and not np.issubdtype(X_train_proc[date_col].dtype, np.datetime64):
                    date_series = pd.to_datetime(X_train_proc[date_col].astype(str), format="%Y", errors="coerce")
                else:
                    date_series = pd.to_datetime(X_train_proc[date_col], errors="coerce")

                valid_dates = date_series.notna().sum()
                if valid_dates > 0:
                    order = np.argsort(date_series.fillna(pd.Timestamp.max).to_numpy())
                    X_train_proc = X_train_proc.iloc[order].reset_index(drop=True)
                    y_train_1d = y_train_1d[order]
                    print(f"[Split] Temporal ordering by '{date_col}' ({valid_dates}/{len(date_series)} parseable dates)")
                else:
                    print("[Split] Warning: date column found but unparsable; falling back to existing row order")
            else:
                print("[Split] Warning: no date column found; temporal split uses existing row order")

            #temporal split: last val_ratio as validation
            split_idx = int(n * (1 - self.val_ratio))
            train_idx = np.arange(0, split_idx)
            val_idx = np.arange(split_idx, n)
            print(f"[Split] Temporal: train={len(train_idx)}, val={len(val_idx)}")
        
        else:
            #stratified random split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=42)
            train_idx, val_idx = next(sss.split(X_train_proc, y_train_1d))
            print(f"[Split] Stratified: train={len(train_idx)}, val={len(val_idx)}")

        X_split_train = X_train_proc.iloc[train_idx].reset_index(drop=True)
        X_split_val   = X_train_proc.iloc[val_idx].reset_index(drop=True)
        y_split_train = y_train_1d[train_idx]
        y_split_val   = y_train_1d[val_idx]

        def objective(trial):
            # ── Sample hyperparameters ─────────────────────────────────────
            epoch_choices = [10, 20, 30]
            lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            batch_size   = trial.suggest_categorical("batch_size", [2] if is_longformer else [8, 16, 32])
            epochs       = trial.suggest_categorical("epochs", epoch_choices)
            dropout      = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
            weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01])
            n_layers     = trial.suggest_categorical("n_layers", [2, 4, 6]) if self.is_hbert else None

            params = {
                "learning_rate": lr,
                "batch_size":    batch_size,
                "epochs":        epochs,
                "dropout":       dropout,
                "weight_decay":  weight_decay,
            }
            if n_layers is not None:
                params["n_layers"] = n_layers

            self.aux_encoder = None

            train_loader = self._create_dataloader(
                X_split_train, y_split_train, is_train=True,
                batch_size=params["batch_size"]
            )
            val_loader = self._create_dataloader(
                X_split_val, y_split_val, is_train=False,
                batch_size=params["batch_size"]
            )

            if self.is_hbert:
                aux_dim = {"op": 6, "both": 12}.get(self.case_mode) if self.opposition else None
                model = self._build_model(params["dropout"], aux_dim=aux_dim, n_layers=params["n_layers"])
            else:
                model = self._build_model(params["dropout"])
                self._fix_opposition_head(model, train_loader)
            optimizer = self._build_optimizer(model, params["learning_rate"], params["weight_decay"])
            criterion = torch.nn.CrossEntropyLoss()

            accum_steps = 8 if is_longformer else 1
            steps_per_epoch = -(-len(train_loader) // accum_steps)  # ceil division
            total_steps = steps_per_epoch * params["epochs"]
            warmup_steps = int(0.10 * total_steps)
            if is_longformer:
                print(f"  Gradient accumulation: {accum_steps} steps → effective batch {params['batch_size'] * accum_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            best_val_f1 = 0.0
            best_epoch = 0
            patience_counter = 0
            patience = 5
            min_epochs = 10
            print(f"  Early stopping enabled (patience={patience}, min_epochs={min_epochs})")
            for epoch in range(params["epochs"]):
                train_loss = self._train_epoch(model, optimizer, train_loader, criterion, scheduler)
                val_loss, val_f1 = self._validate_epoch(model, val_loader, criterion)
                print(
                    f"  Epoch {epoch + 1}/{params['epochs']}: "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
                )
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if (epoch + 1) >= min_epochs and patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break

            trial.set_user_attr("best_epoch", int(best_epoch))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  → Val F1: {best_val_f1:.4f} (best_epoch={best_epoch})")
            return best_val_f1

        print(
            f"[Optuna] Starting study: n_trials={self.n_trials}, timeout={self.timeout}, "
            f"sampler={type(self.sampler).__name__}"
        )
        start_time = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_val_f1 = best_trial.value
        best_epoch = int(best_trial.user_attrs.get("best_epoch", best_params.get("epochs", 0)))
        best_params["epochs"] = best_epoch
        print(f"\nBest params: {best_params} | Val F1: {best_val_f1:.4f} | best_epoch={best_epoch}")
        print(f"Completed {len(study.trials)} trials")

        elapsed = time.time() - start_time

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "deep_learning",
            "algo": self.model_name,
            "model_name": self.hf_model_name,
            "experiment": self.experiment,
            "val_strategy": "temporal" if self.experiment == "2" else "stratified_random",
            "val_ratio": self.val_ratio,
            "opposition": self.opposition,
            "case_mode": self.case_mode,
            "input_representation": self.input_representation,
            "best_score_val": best_val_f1,
            "best_epoch": best_epoch,
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
            "time_seconds": elapsed,
            "test_metrics": None,
        }
        # Add architecture-specific metadata
        if self.is_hbert:
            record["architecture"] = "hierarchical_bert"
            record["chunk_size"] = self.chunk_size
            record["max_chunks"] = self.max_chunks
        else:
            record["max_context_length"] = self.max_length

        self.results.append(record)
        self._append_json_result(record)
        return [record]

    def _build_model(self, dropout, aux_dim=None, n_layers=4):
        """Load a fresh model instance with the given dropout, sent to device.

        For H-BERT: creates HierBertModel with trainable encoder + context transformer.
        For flat models: loads AutoModelForSequenceClassification from HuggingFace.
        """
        if self.is_hbert:
            model = HierBertModel(
                hf_model_name=self.hf_model_name,
                chunk_size=self.chunk_size,
                max_chunks=self.max_chunks,
                num_labels=2,
                dropout=dropout,
                opposition=self.opposition,
                aux_dim=aux_dim,
                n_layers=n_layers,
            )
            return model.to(self.device)

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
            # aux_feature_dim is a placeholder (1); _fix_opposition_head replaces
            # fusion_layer with the correct size once the first batch is seen.
            self.custom_head = OppositionModeClassificationHead(
                text_embed_dim=model.config.hidden_size,
                aux_feature_dim=1,
                num_labels=2,
            )
            self.custom_head.to(self.device)

        return model

    def _build_optimizer(self, model, lr, weight_decay=0.0):
        """Build AdamW optimizer over the correct parameter set.

        For H-BERT: all model params (encoder + context_transformer + classifier).
        For flat models: all model params (+ custom_head for opposition mode).
        """
        if self.is_hbert:
            # Base encoder gets the small Optuna LR
            # Randomly initialized upper layers get a 10x larger LR to actually learn
            optimizer_grouped_parameters = [
                {"params": model.encoder.parameters(), "lr": lr},
                {"params": model.context_transformer.parameters(), "lr": lr * 10},
                {"params": model.position_embeddings.parameters(), "lr": lr * 10},
                {"params": model.classifier.parameters(), "lr": lr * 10},
            ]
            return AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
        elif self.opposition:
            params = list(self._get_encoder(model).parameters()) + list(self.custom_head.parameters())
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _fix_opposition_head(self, model, train_loader):
        """Set opposition head fusion width using hardcoded aux dims, validating against data."""
        if not self.opposition:
            return
        expected = {"op": 6, "both": 12}
        if self.case_mode not in expected:
            raise ValueError(
                f"Unsupported opposition case_mode '{self.case_mode}' for hardcoded aux dims. "
                "Expected one of: op, both."
            )
        aux_dim = expected[self.case_mode]
        for batch in train_loader:
            observed = batch["auxiliary_features"].shape[1]
            if observed != aux_dim:
                raise ValueError(
                    f"Hardcoded aux_dim={aux_dim} but observed {observed} for case_mode='{self.case_mode}'."
                )
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
            model: Trained model (BERT-based or HierBertModel)
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with accuracy, f1, precision, recall, mcc, auc
        """
        test_batch = 4 if self.is_hbert else 32
        test_loader = self._create_dataloader(X_test, y_test, is_train=False, batch_size=test_batch)

        model.eval()
        all_preds = []
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["labels"].to(self.device)

                if self.is_hbert:
                    ids = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    cmask = batch["chunk_mask"].to(self.device)
                    aux = batch.get("auxiliary_features")
                    if aux is not None:
                        aux = aux.to(self.device)

                    with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                        logits = model(ids, mask, cmask, auxiliary_features=aux)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    if self.opposition:
                        outputs = self._get_encoder(model)(input_ids=input_ids, attention_mask=attention_mask)
                        text_embedding = outputs.pooler_output
                        auxiliary_features = batch["auxiliary_features"].to(self.device)
                        logits = self.custom_head(text_embedding, auxiliary_features)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]

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
        """Append result to JSON file with file locking."""
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
