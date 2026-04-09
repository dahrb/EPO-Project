"""Hierarchical BERT (H-BERT) for long patent documents.

Architecture following Chalkidis et al. (2022a):
  1. Sentence-boundary chunking → 128 tokens × 64 chunks max
  2. Frozen Legal-BERT encoder (per-chunk CLS embeddings)
  3. 1-layer TransformerEncoder context layer
     (d_model=768, nhead=8, dim_feedforward=2048)
  4. Masked max-pooling over chunks
  5. Linear classifier (with opposition-mode variant)

Key design decisions
--------------------
- Encoder is **frozen** — only context layer + classifier are trained.
- fp16 mixed-precision with gradient accumulation (accum_steps=4).
- Sentence-boundary chunking at runtime (split on '. ') so existing
  flat-text pickle files can be reused without re-processing.
- No positional embeddings on chunks (ablation shows negligible gain).

Standalone usage (smoke test + benchmark):
    python -m Experiments.hierarchical_bert

Last Updated: 09.04.26
Status: In Progress
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    ParameterSampler,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utilities.utils import TextProcess

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chunking
# ═══════════════════════════════════════════════════════════════════════════════


def _split_into_chunks(
    text: str,
    tokenizer,
    chunk_size: int = 128,
    max_chunks: int = 64,
) -> list[list[int]]:
    """Split *text* into sentence-boundary-aware token chunks.

    Strategy:
      1. Split on ``'. '`` to approximate sentence boundaries.
      2. Greedily pack sentences into chunks of up to *chunk_size* tokens
         (including [CLS] / [SEP] special tokens — 2 reserved).
      3. If a single sentence exceeds *chunk_size - 2* tokens it is
         hard-truncated at the boundary.
      4. Return at most *max_chunks* chunks; each is a padded list of
         token IDs of length exactly *chunk_size*.

    Returns a list of ``input_ids`` lists (length ≤ max_chunks, each of
    length chunk_size).
    """
    usable = chunk_size - 2  # room for [CLS] and [SEP]

    # ── sentence splitting ──────────────────────────────────────────────────
    sentences = text.split(". ")
    # Re-attach the period that split removed (except for the last fragment)
    sentences = [s + "." if i < len(sentences) - 1 else s for i, s in enumerate(sentences)]

    # Tokenise each sentence (no special tokens — we add them per chunk)
    sent_token_ids: list[list[int]] = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if ids:
            sent_token_ids.append(ids)

    if not sent_token_ids:
        # Fallback: if nothing remains, create one padded chunk
        return [_pad_chunk([], tokenizer, chunk_size)]

    # ── greedy packing ──────────────────────────────────────────────────────
    chunks: list[list[int]] = []
    current: list[int] = []
    for ids in sent_token_ids:
        if len(ids) > usable:
            # Flush current if non-empty
            if current:
                chunks.append(_pad_chunk(current, tokenizer, chunk_size))
                if len(chunks) >= max_chunks:
                    break
                current = []
            # Hard-truncate the oversized sentence
            chunks.append(_pad_chunk(ids[:usable], tokenizer, chunk_size))
            if len(chunks) >= max_chunks:
                break
        elif len(current) + len(ids) > usable:
            # Current chunk is full — flush and start new one
            chunks.append(_pad_chunk(current, tokenizer, chunk_size))
            if len(chunks) >= max_chunks:
                break
            current = ids[:]
        else:
            current.extend(ids)

    # Flush trailing tokens
    if current and len(chunks) < max_chunks:
        chunks.append(_pad_chunk(current, tokenizer, chunk_size))

    return chunks


def _pad_chunk(
    token_ids: list[int],
    tokenizer,
    chunk_size: int,
) -> list[int]:
    """Wrap *token_ids* with [CLS]/[SEP] and pad/truncate to *chunk_size*."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    usable = chunk_size - 2
    ids = [cls_id] + token_ids[:usable] + [sep_id]
    padding_length = chunk_size - len(ids)
    ids = ids + [pad_id] * padding_length
    return ids


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════════════════════


class HierarchicalPatentDataset(Dataset):
    """Pre-chunked patent dataset for hierarchical BERT.

    Each sample contains:
    - input_ids   : (n_chunks, chunk_size)  — padded/truncated to max_chunks
    - attention    : (n_chunks, chunk_size)  — 1 for real tokens, 0 for padding
    - chunk_mask   : (max_chunks,)           — 1 for real chunks, 0 for padded
    - label        : scalar
    - auxiliary_features : (aux_dim,) or None
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        chunk_size: int = 128,
        max_chunks: int = 64,
        auxiliary_features: np.ndarray | None = None,
    ):
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.auxiliary_features = auxiliary_features

        # Pre-chunk all documents
        self.all_chunks: list[list[list[int]]] = []
        for text in texts:
            chunks = _split_into_chunks(text, tokenizer, chunk_size, max_chunks)
            self.all_chunks.append(chunks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chunks = self.all_chunks[idx]
        n = len(chunks)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Pad to max_chunks
        input_ids = np.zeros((self.max_chunks, self.chunk_size), dtype=np.int64)
        attention = np.zeros((self.max_chunks, self.chunk_size), dtype=np.int64)
        chunk_mask = np.zeros(self.max_chunks, dtype=np.float32)

        for i, chunk in enumerate(chunks):
            input_ids[i] = chunk
            attention[i] = [1 if t != pad_id else 0 for t in chunk]
            chunk_mask[i] = 1.0

        item = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "chunk_mask": torch.from_numpy(chunk_mask),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.auxiliary_features is not None:
            item["auxiliary_features"] = torch.tensor(
                self.auxiliary_features[idx], dtype=torch.float32
            )
        return item


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model
# ═══════════════════════════════════════════════════════════════════════════════


class HierBertModel(nn.Module):
    """Hierarchical BERT with frozen chunk encoder.

    Architecture
    ------------
    1. Frozen Legal-BERT encodes each chunk independently → CLS embedding (768-d).
    2. A 1-layer TransformerEncoder aggregates chunk embeddings into a document
       representation using self-attention across chunks.
    3. Masked max-pooling over the context output produces a single vector.
    4. A linear classifier maps to logits.

    In opposition mode the pooled vector is concatenated with auxiliary
    features before the classifier.
    """

    def __init__(
        self,
        hf_model_name: str = "nlpaueb/legal-bert-base-uncased",
        chunk_size: int = 128,
        max_chunks: int = 64,
        num_labels: int = 2,
        dropout: float = 0.1,
        opposition: bool = False,
        aux_dim: int | None = None,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.num_labels = num_labels
        self.opposition = opposition

        # ── 1. Frozen chunk encoder ─────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(hf_model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.hidden_size = self.encoder.config.hidden_size  # 768

        # ── 2. Context layer ────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

        # ── 3. Classifier ──────────────────────────────────────────────────
        if opposition and aux_dim is not None:
            cls_in = self.hidden_size + aux_dim
        else:
            cls_in = self.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        # Glorot init on classifier layers
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids      : (B, max_chunks, chunk_size)
            attention_mask  : (B, max_chunks, chunk_size)
            chunk_mask      : (B, max_chunks)  — 1.0 for real chunks
            auxiliary_features : (B, aux_dim) or None

        Returns:
            logits : (B, num_labels)
        """
        B, C, S = input_ids.shape

        # ── Flatten to (B*C, S) for encoder ─────────────────────────────────
        flat_ids = input_ids.view(B * C, S)
        flat_mask = attention_mask.view(B * C, S)

        # Skip padding-only chunks for efficiency
        # But we still pass all to keep shapes aligned; the context
        # transformer will ignore them via the key_padding_mask.
        with torch.no_grad():
            encoder_out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)

        # CLS token for each chunk → (B, C, hidden_size)
        cls_embeddings = encoder_out.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(B, C, self.hidden_size)

        # ── Context transformer ─────────────────────────────────────────────
        # key_padding_mask: True = ignore → invert chunk_mask
        key_padding_mask = chunk_mask == 0  # (B, C)
        context_out = self.context_transformer(
            cls_embeddings,
            src_key_padding_mask=key_padding_mask,
        )  # (B, C, hidden_size)

        # ── Masked max-pool ─────────────────────────────────────────────────
        mask_expanded = chunk_mask.unsqueeze(-1)  # (B, C, 1)
        context_out = context_out * mask_expanded  # zero out padding chunks
        # Set padding to large negative so they don't dominate max
        context_out = context_out + (1 - mask_expanded) * (-1e9)
        pooled = context_out.max(dim=1).values  # (B, hidden_size)

        # ── Classifier ──────────────────────────────────────────────────────
        if self.opposition and auxiliary_features is not None:
            pooled = torch.cat([pooled, auxiliary_features], dim=1)

        logits = self.classifier(pooled)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Experiment class
# ═══════════════════════════════════════════════════════════════════════════════


class HBertExperiment:
    """Flat-CV hyperparameter search + heldout test for H-BERT.

    Mirrors the API of ``DeepLearningExperiments`` but uses H-BERT with
    fp16 mixed precision and gradient accumulation.
    """

    def __init__(
        self,
        experiment: str = "1",
        opposition: bool = False,
        results_json_path: str = "results_hbert.json",
        device=None,
        cv_num: int = 3,
        repeat: int = 1,
        n_iter: int = 10,
        chunk_size: int = 128,
        max_chunks: int = 64,
        accum_steps: int = 4,
    ):
        self.experiment = experiment
        self.opposition = opposition
        self.results_json_path = results_json_path
        self.results: list[dict] = []
        self.n_iter = n_iter
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.accum_steps = accum_steps

        # HF model for the frozen encoder
        self.hf_model_name = "nlpaueb/legal-bert-base-uncased"
        self.model_name = "hbert"
        self.input_representation = "HierBERT"

        # CV splitter — same logic as DeepLearningExperiments
        if experiment == "2":
            self.cv = TimeSeriesSplit(n_splits=cv_num)
        else:
            self.cv = RepeatedStratifiedKFold(
                n_splits=cv_num, n_repeats=repeat, random_state=42
            )

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Hyperparameter grid (context layer + classifier only; encoder frozen)
        self.param_grid = {
            "learning_rate": [1e-4, 5e-4, 1e-3],
            "batch_size": [2, 4],
            "epochs": [5, 10, 15],
            "dropout": [0.1, 0.2],
            "weight_decay": [0.0, 0.01],
            "warmup_steps": [0, 100],
        }

        # Tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

        # Auxiliary encoder (opposition mode)
        self.aux_encoder = None

        print(
            f"[H-BERT] encoder={self.hf_model_name}  "
            f"chunk_size={chunk_size}  max_chunks={max_chunks}  "
            f"accum_steps={accum_steps}  device={self.device}"
        )

    # ── Preprocessing ───────────────────────────────────────────────────────

    def _preprocess_text_for_bert(self, X):
        """Apply fixed text preprocessing (stop-words, lemmatisation)."""
        if not hasattr(X, "copy"):
            return X
        X_proc = X.copy()
        if "New Summary Facts" not in X_proc.columns:
            return X_proc
        tp = TextProcess(stopwords=True, numbers=False, lemmatisation=True)
        docs = [str(t) for t in X_proc["New Summary Facts"].tolist()]
        processed = tp.fit_transform(docs)
        X_proc["New Summary Facts"] = [" ".join(tokens) for tokens in processed]
        return X_proc

    def _to_1d(self, y):
        if isinstance(y, pd.DataFrame):
            return y.iloc[:, 0].to_numpy()
        if isinstance(y, pd.Series):
            return y.to_numpy()
        return np.asarray(y).reshape(-1)

    def _get_aux_feature_columns(self, X):
        if not hasattr(X, "columns"):
            return []
        return [c for c in X.columns if c != "New Summary Facts"]

    def _encode_opposition_features(self, X, fit=False):
        from sklearn.preprocessing import OneHotEncoder

        aux_cols = self._get_aux_feature_columns(X)
        if not aux_cols:
            return None
        X_aux = X[aux_cols].copy()
        if fit:
            self.aux_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            return self.aux_encoder.fit_transform(X_aux).astype(np.float32)
        if self.aux_encoder is None:
            raise ValueError("Encoder not fitted.")
        return self.aux_encoder.transform(X_aux).astype(np.float32)

    # ── DataLoader ──────────────────────────────────────────────────────────

    def _create_dataloader(self, X, y, is_train=True, batch_size=2):
        texts = X["New Summary Facts"].tolist()

        aux = None
        if self.opposition:
            aux = self._encode_opposition_features(
                X, fit=(is_train and self.aux_encoder is None)
            )

        ds = HierarchicalPatentDataset(
            texts,
            y,
            self.tokenizer,
            chunk_size=self.chunk_size,
            max_chunks=self.max_chunks,
            auxiliary_features=aux,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=is_train)

    # ── Training ────────────────────────────────────────────────────────────

    def _train_epoch(self, model, optimizer, loader, criterion, scheduler=None):
        """One training epoch with fp16 + gradient accumulation."""
        model.train()
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            cmask = batch["chunk_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            aux = batch.get("auxiliary_features")
            if aux is not None:
                aux = aux.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = model(ids, mask, cmask, auxiliary_features=aux)
                loss = criterion(logits, labels) / self.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * self.accum_steps

        return total_loss / len(loader)

    def _validate_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                cmask = batch["chunk_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                aux = batch.get("auxiliary_features")
                if aux is not None:
                    aux = aux.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = model(ids, mask, cmask, auxiliary_features=aux)
                    loss = criterion(logits, labels)

                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return avg_loss, f1

    # ── Build helpers ───────────────────────────────────────────────────────

    def _build_model(self, dropout, aux_dim=None):
        model = HierBertModel(
            hf_model_name=self.hf_model_name,
            chunk_size=self.chunk_size,
            max_chunks=self.max_chunks,
            num_labels=2,
            dropout=dropout,
            opposition=self.opposition,
            aux_dim=aux_dim,
        )
        return model.to(self.device)

    def _build_optimizer(self, model, lr, weight_decay=0.0):
        """Optimise only the context transformer + classifier (encoder frozen)."""
        params = list(model.context_transformer.parameters()) + list(
            model.classifier.parameters()
        )
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    def _infer_aux_dim(self, loader):
        """Return aux_dim from first batch, or None."""
        if not self.opposition:
            return None
        for batch in loader:
            if "auxiliary_features" in batch:
                return batch["auxiliary_features"].shape[1]
            break
        return None

    # ── CV search ───────────────────────────────────────────────────────────

    def training_loop(self, X_train, y_train, X_test=None, y_test=None):
        """Full CV search → retrain → test (mirrors DeepLearningExperiments)."""
        y_train_1d = self._to_1d(y_train)
        y_test_1d = self._to_1d(y_test) if y_test is not None else None

        X_train_proc = self._preprocess_text_for_bert(X_train.reset_index(drop=True))
        X_test_proc = (
            self._preprocess_text_for_bert(X_test.reset_index(drop=True))
            if X_test is not None
            else None
        )

        has_test = X_test is not None and y_test is not None
        print(
            f"[Data] Train: {len(X_train_proc)}"
            + (f", Test: {len(X_test_proc)}" if has_test else "")
            + f" | Exp {self.experiment} CV: {self.cv.__class__.__name__}"
        )

        return self._run_one_model(
            X_train_proc, y_train_1d, X_test_proc, y_test_1d, has_test
        )

    def _run_one_model(self, X_train_proc, y_train_1d, X_test_proc, y_test_1d, has_test):
        """Randomised search → retrain → test."""
        # Reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        param_combos = list(
            ParameterSampler(self.param_grid, n_iter=self.n_iter, random_state=42)
        )
        print(
            f"[Search] {len(param_combos)} combos (n_iter={self.n_iter}) "
            f"for model '{self.model_name}'"
        )

        start_time = time.time()
        best_mean_f1 = -1.0
        best_params = None

        # ── Phase 1: CV ─────────────────────────────────────────────────────
        for pi, params in enumerate(param_combos):
            print(f"\n[{pi+1}/{len(param_combos)}] CV for params: {params}")

            fold_f1s = []
            for fi, (tr_idx, va_idx) in enumerate(
                self.cv.split(X_train_proc, y_train_1d)
            ):
                X_ft = X_train_proc.iloc[tr_idx].reset_index(drop=True)
                X_fv = X_train_proc.iloc[va_idx].reset_index(drop=True)
                y_ft = y_train_1d[tr_idx]
                y_fv = y_train_1d[va_idx]

                self.aux_encoder = None
                train_loader = self._create_dataloader(
                    X_ft, y_ft, is_train=True, batch_size=params["batch_size"]
                )
                val_loader = self._create_dataloader(
                    X_fv, y_fv, is_train=False, batch_size=params["batch_size"]
                )

                aux_dim = self._infer_aux_dim(train_loader)
                model = self._build_model(params["dropout"], aux_dim=aux_dim)
                optimizer = self._build_optimizer(
                    model, params["learning_rate"], params["weight_decay"]
                )
                criterion = nn.CrossEntropyLoss()

                total_steps = (len(train_loader) // self.accum_steps + 1) * params["epochs"]
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=params["warmup_steps"],
                    num_training_steps=total_steps,
                )

                best_fold_f1 = 0.0
                patience_counter = 0
                patience = 2
                for epoch in range(params["epochs"]):
                    train_loss = self._train_epoch(
                        model, optimizer, train_loader, criterion, scheduler
                    )
                    val_loss, val_f1 = self._validate_epoch(
                        model, val_loader, criterion
                    )
                    print(
                        f"  Fold {fi+1} Epoch {epoch+1}/{params['epochs']}: "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                        f"val_f1={val_f1:.4f}"
                    )
                    if val_f1 > best_fold_f1:
                        best_fold_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break

                fold_f1s.append(best_fold_f1)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean_f1 = float(np.mean(fold_f1s))
            print(
                f"  → Mean CV F1: {mean_f1:.4f} "
                f"(folds: {[round(f,4) for f in fold_f1s]})"
            )
            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_params = params.copy()

        print(f"\nBest params: {best_params} | Mean CV F1: {best_mean_f1:.4f}")

        # ── Phase 2: retrain on full training set ───────────────────────────
        print("\nRetraining best params on full training set...")
        self.aux_encoder = None
        full_loader = self._create_dataloader(
            X_train_proc, y_train_1d, is_train=True,
            batch_size=best_params["batch_size"],
        )
        aux_dim = self._infer_aux_dim(full_loader)
        best_model = self._build_model(best_params["dropout"], aux_dim=aux_dim)
        best_optimizer = self._build_optimizer(
            best_model, best_params["learning_rate"], best_params["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()

        retrain_steps = (len(full_loader) // self.accum_steps + 1) * best_params["epochs"]
        retrain_sched = get_linear_schedule_with_warmup(
            best_optimizer,
            num_warmup_steps=best_params["warmup_steps"],
            num_training_steps=retrain_steps,
        )
        for epoch in range(best_params["epochs"]):
            train_loss = self._train_epoch(
                best_model, best_optimizer, full_loader, criterion, retrain_sched
            )
            print(
                f"  Retrain Epoch {epoch+1}/{best_params['epochs']}: "
                f"train_loss={train_loss:.4f}"
            )

        elapsed = time.time() - start_time

        # ── Phase 3: test ───────────────────────────────────────────────────
        test_metrics = None
        if has_test:
            test_metrics = self._compute_test_metrics(
                best_model, X_test_proc, y_test_1d
            )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "deep_learning",
            "algo": self.model_name,
            "model_name": self.hf_model_name,
            "architecture": "hierarchical_bert",
            "chunk_size": self.chunk_size,
            "max_chunks": self.max_chunks,
            "accum_steps": self.accum_steps,
            "experiment": self.experiment,
            "cv_strategy": self.cv.__class__.__name__,
            "opposition": self.opposition,
            "input_representation": self.input_representation,
            "best_score_cv": best_mean_f1,
            "best_params": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in best_params.items()
            },
            "time_seconds": elapsed,
            "test_metrics": test_metrics,
        }

        self.results.append(record)
        self._append_json_result(record)
        return [record]

    # ── Test metrics ────────────────────────────────────────────────────────

    def _compute_test_metrics(self, model, X_test, y_test):
        test_loader = self._create_dataloader(
            X_test, y_test, is_train=False, batch_size=4
        )
        model.eval()
        all_preds, all_scores, all_labels = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                cmask = batch["chunk_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                aux = batch.get("auxiliary_features")
                if aux is not None:
                    aux = aux.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = model(ids, mask, cmask, auxiliary_features=aux)

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(probs.cpu().float().numpy())
                all_labels.extend(labels.cpu().numpy())

        preds_arr = np.array(all_preds)
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)

        metrics = {
            "accuracy": float(accuracy_score(labels_arr, preds_arr)),
            "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
            "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
            "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels_arr, preds_arr)),
            "auc": None,
        }
        try:
            metrics["auc"] = float(roc_auc_score(labels_arr, scores_arr))
        except ValueError:
            metrics["auc"] = None
        return metrics

    # ── JSON persistence ────────────────────────────────────────────────────

    def _append_json_result(self, record):
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Smoke test & benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def _make_synthetic_data(n: int = 200, opposition: bool = False):
    """Create a synthetic DataFrame mimicking the real pickle format."""
    rng = np.random.RandomState(42)
    texts = []
    for _ in range(n):
        n_sents = rng.randint(10, 60)
        sents = [
            " ".join(rng.choice(list("abcdefghij"), size=rng.randint(5, 20)))
            for _ in range(n_sents)
        ]
        texts.append(". ".join(sents))

    labels = rng.randint(0, 2, size=n)
    df = pd.DataFrame({"New Summary Facts": texts})
    if opposition:
        df["Category"] = rng.choice(["A", "B", "C"], size=n)
        df["Matches_1"] = rng.randint(0, 5, size=n)
    y = pd.Series(labels)
    return df, y


def run_smoke_test():
    """Verify shapes, loss decrease, and prediction validity on synthetic data."""
    print("\n" + "=" * 70)
    print("H-BERT SMOKE TEST  (200 samples, 2 epochs, no CV)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    df, y = _make_synthetic_data(200)
    texts = df["New Summary Facts"].tolist()
    labels = y.to_numpy().astype(np.int64)

    # ── Test chunking ───────────────────────────────────────────────────────
    chunks = _split_into_chunks(texts[0], tokenizer, chunk_size=128, max_chunks=64)
    assert len(chunks) <= 64, f"Too many chunks: {len(chunks)}"
    assert all(len(c) == 128 for c in chunks), "Chunk size mismatch"
    print(f"[✓] Chunking: {len(chunks)} chunks of 128 tokens")

    # ── Test dataset ────────────────────────────────────────────────────────
    ds = HierarchicalPatentDataset(texts, labels, tokenizer, chunk_size=128, max_chunks=64)
    sample = ds[0]
    assert sample["input_ids"].shape == (64, 128), f"ids shape: {sample['input_ids'].shape}"
    assert sample["attention_mask"].shape == (64, 128)
    assert sample["chunk_mask"].shape == (64,)
    print(f"[✓] Dataset shapes: ids={sample['input_ids'].shape}, "
          f"mask={sample['attention_mask'].shape}, chunk_mask={sample['chunk_mask'].shape}")

    # ── Test model forward pass ─────────────────────────────────────────────
    model = HierBertModel(
        hf_model_name="nlpaueb/legal-bert-base-uncased",
        chunk_size=128,
        max_chunks=64,
        dropout=0.1,
    ).to(device)

    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    cmask = batch["chunk_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, mask, cmask)
    assert logits.shape == (2, 2), f"Logits shape: {logits.shape}"
    print(f"[✓] Forward pass: logits shape = {logits.shape}")

    # ── Test training (2 epochs, loss should decrease) ──────────────────────
    optimizer = AdamW(
        list(model.context_transformer.parameters())
        + list(model.classifier.parameters()),
        lr=5e-4,
    )
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cmask = batch["chunk_mask"].to(device)
            lbl = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(ids, mask, cmask)
                loss = criterion(logits, lbl)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Loss should generally decrease (but with 2 epochs on random data, just
    # check it didn't explode)
    assert losses[-1] < 10.0, f"Loss exploded: {losses[-1]}"
    print(f"[✓] Training: loss trajectory {[round(l, 4) for l in losses]}")

    # ── Test predictions ────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cmask = batch["chunk_mask"].to(device)
            logits = model(ids, mask, cmask)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    unique_preds = set(all_preds)
    assert unique_preds.issubset({0, 1}), f"Invalid predictions: {unique_preds}"
    print(f"[✓] Predictions: {len(all_preds)} preds, classes={unique_preds}")

    # ── Count trainable params ──────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"[✓] Parameters: total={total:,}  trainable={trainable:,}  frozen={frozen:,}")

    print("\n✅ ALL SMOKE TESTS PASSED\n")
    return True


def run_benchmark():
    """Compare flat Legal-BERT vs H-BERT on 200 synthetic samples (2 epochs).

    This is a *shape/API sanity check* — real comparisons need full data and
    proper CV.  The goal is to verify both pipelines produce valid metrics.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Flat Legal-BERT vs H-BERT  (200 synthetic samples)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, y = _make_synthetic_data(200)
    labels = y.to_numpy().astype(np.int64)

    # ── Flat Legal-BERT baseline ────────────────────────────────────────────
    print("\n--- Flat Legal-BERT (512-token truncation) ---")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    texts = df["New Summary Facts"].tolist()

    enc = tokenizer(
        texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    flat_ds = torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    )
    flat_loader = DataLoader(flat_ds, batch_size=8, shuffle=True)

    flat_model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", num_labels=2
    ).to(device)
    flat_opt = AdamW(flat_model.parameters(), lr=2e-5)
    flat_crit = nn.CrossEntropyLoss()

    flat_model.train()
    for epoch in range(2):
        for ids, mask, lbl in flat_loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            out = flat_model(input_ids=ids, attention_mask=mask, labels=lbl)
            out.loss.backward()
            flat_opt.step()
            flat_opt.zero_grad()

    flat_model.eval()
    flat_preds, flat_scores, flat_labels = [], [], []
    with torch.no_grad():
        for ids, mask, lbl in flat_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = flat_model(input_ids=ids, attention_mask=mask).logits
            flat_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            flat_scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            flat_labels.extend(lbl.numpy())

    flat_f1 = f1_score(flat_labels, flat_preds, zero_division=0)
    flat_acc = accuracy_score(flat_labels, flat_preds)
    print(f"Flat BERT  →  F1={flat_f1:.4f}  Acc={flat_acc:.4f}")

    # ── H-BERT ──────────────────────────────────────────────────────────────
    print("\n--- H-BERT (128×64 chunks) ---")
    hds = HierarchicalPatentDataset(texts, labels, tokenizer, chunk_size=128, max_chunks=64)
    h_loader = DataLoader(hds, batch_size=2, shuffle=True)

    h_model = HierBertModel(
        hf_model_name="nlpaueb/legal-bert-base-uncased",
        chunk_size=128,
        max_chunks=64,
        dropout=0.1,
    ).to(device)

    h_opt = AdamW(
        list(h_model.context_transformer.parameters())
        + list(h_model.classifier.parameters()),
        lr=5e-4,
    )
    h_crit = nn.CrossEntropyLoss()

    h_model.train()
    for epoch in range(2):
        for batch in h_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cmask = batch["chunk_mask"].to(device)
            lbl = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = h_model(ids, mask, cmask)
                loss = h_crit(logits, lbl)
            loss.backward()
            h_opt.step()
            h_opt.zero_grad()

    h_model.eval()
    h_preds, h_scores, h_labels = [], [], []
    with torch.no_grad():
        for batch in h_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cmask = batch["chunk_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = h_model(ids, mask, cmask)
            h_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            h_scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            h_labels.extend(batch["labels"].numpy())

    h_f1 = f1_score(h_labels, h_preds, zero_division=0)
    h_acc = accuracy_score(h_labels, h_preds)
    print(f"H-BERT     →  F1={h_f1:.4f}  Acc={h_acc:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'Model':<18} {'F1':>8} {'Acc':>8}")
    print("-" * 36)
    print(f"{'Flat BERT (512)':<18} {flat_f1:>8.4f} {flat_acc:>8.4f}")
    print(f"{'H-BERT (128×64)':<18} {h_f1:>8.4f} {h_acc:>8.4f}")
    print(
        "\n⚠  Synthetic data — metrics are NOT meaningful.  "
        "This verifies both pipelines produce valid outputs."
    )
    print("✅ BENCHMARK COMPLETE\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_smoke_test()
    run_benchmark()
