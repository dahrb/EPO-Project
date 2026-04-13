"""Hierarchical BERT (H-BERT) components for long patent documents.

Architecture following Chalkidis et al. (2022a):
  1. Sentence-boundary chunking → 128 tokens × 64 chunks max
  2. Trainable Legal-BERT encoder (per-chunk CLS embeddings)
  3. TransformerEncoder context layer with 4 heads and tunable layer quantity
         (d_model=768, nhead=4, dim_feedforward=2048)
  4. BERT-style pooler with mean-pooling over chunks
  5. Linear classifier (with opposition-mode variant)

This module provides the **model, dataset, and chunking utilities**.  The
experiment runner lives in ``DeepLearningExperiments`` (model_name='hbert')
inside ``deep_learning_experiments.py``.

Last Updated: 13.04.26
Status: Done
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chunking
# ═══════════════════════════════════════════════════════════════════════════════


def _split_into_chunks( text: str, tokenizer, chunk_size: int = 128, max_chunks: int = 64) -> list[list[int]]:
    
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
    #re-attach the fullstop that split removed (except for the last fragment)
    sentences = [s + "." if i < len(sentences) - 1 else s for i, s in enumerate(sentences)]

    #tokenise each sentence 
    sent_token_ids: list[list[int]] = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if ids:
            sent_token_ids.append(ids)

    if not sent_token_ids:
        #fallback: if nothing remains, create one padded chunk
        return [_pad_chunk([], tokenizer, chunk_size)]

    # ── greedy packing ──────────────────────────────────────────────────────
    chunks: list[list[int]] = []
    current: list[int] = []
    for ids in sent_token_ids:
        if len(ids) > usable:
            #flush current if non-empty
            if current:
                chunks.append(_pad_chunk(current, tokenizer, chunk_size))
                if len(chunks) >= max_chunks:
                    break
                current = []
            #hard-truncate the oversized sentence
            chunks.append(_pad_chunk(ids[:usable], tokenizer, chunk_size))
            if len(chunks) >= max_chunks:
                break
        elif len(current) + len(ids) > usable:
            #current chunk is full — flush and start new one
            chunks.append(_pad_chunk(current, tokenizer, chunk_size))
            if len(chunks) >= max_chunks:
                break
            current = ids[:]
        else:
            current.extend(ids)

    #flush trailing tokens
    if current and len(chunks) < max_chunks:
        chunks.append(_pad_chunk(current, tokenizer, chunk_size))

    return chunks

def _pad_chunk(token_ids: list[int], tokenizer, chunk_size: int) -> list[int]:
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

        #chunk all documents
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

        #pad to max_chunks
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

class MaxPooler(nn.Module):
    """Masked max-pooling over sequence dimension, followed by Dense + Tanh."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            pooled_output = hidden_states.max(dim=1).values
        else:
            # mask padding positions to -inf so they are ignored by max
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, C, 1)
            masked = hidden_states + (1.0 - mask) * -1e9
            pooled_output = masked.max(dim=1).values
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class HierBertModel(nn.Module):
    """Hierarchical BERT with trainable chunk encoder.

    Architecture
    ------------
    1. Trainable Legal-BERT encodes each chunk independently → CLS embedding (768-d).
         2. Learnable chunk positional embeddings are added to CLS embeddings.
         3. A 4-layer TransformerEncoder aggregates chunk embeddings into a document
       representation using self-attention across chunks.
         4. Mean-pooling over the context output, followed by Dense + Tanh,
         produces a single vector.
        5. A linear classifier maps to logits.

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
        n_layers: int = 4,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.num_labels = num_labels
        self.opposition = opposition

        # ── 1. Chunk encoder (trainable) ─────────────────────────────────
        self.encoder = AutoModel.from_pretrained(hf_model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768

        # ── 2. Context layer ────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=3072,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        
        self.context_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        
        self.position_embeddings = nn.Embedding(self.max_chunks, self.hidden_size)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        self.pooler = MaxPooler(self.hidden_size)

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

        #glorot init on classifier layers
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, chunk_mask, auxiliary_features=None):
        """
        Args:
            input_ids       : (B, max_chunks, chunk_size)
            attention_mask  : (B, max_chunks, chunk_size)
            chunk_mask      : (B, max_chunks)            — 1 for real chunks
            auxiliary_features : (B, aux_dim) or None

        Returns:
            logits : (B, num_labels)
        """
        B, C, S = input_ids.shape

        # ── 1. Encode each chunk independently ─────────────────────────────
        ids_flat = input_ids.view(B * C, S)
        att_flat = attention_mask.view(B * C, S)

        with torch.no_grad():
            enc_out = self.encoder(input_ids=ids_flat, attention_mask=att_flat)
        cls_embeddings = enc_out.last_hidden_state[:, 0, :]   # (B*C, hidden)
        cls_embeddings = cls_embeddings.view(B, C, -1)        # (B, C, hidden)

        # Add learnable chunk positional embeddings
        position_ids = torch.arange(C, device=input_ids.device).unsqueeze(0).expand(B, C)
        cls_embeddings = cls_embeddings + self.position_embeddings(position_ids)

        # ── 2. Context transformer with key_padding_mask ───────────────────
        # TransformerEncoder expects mask where True = **ignore**
        key_padding_mask = chunk_mask == 0  # (B, C)
        context_out = self.context_transformer(
            cls_embeddings, src_key_padding_mask=key_padding_mask
        )  # (B, C, hidden)

        # ── 3. Max-pooling + Dense + Tanh ─────────────────────────────────
        pooled = self.pooler(context_out, attention_mask=chunk_mask)  # (B, hidden)

        # ── 4. Classifier ──────────────────────────────────────────────────
        if self.opposition and auxiliary_features is not None:
            pooled = torch.cat([pooled, auxiliary_features], dim=1)

        logits = self.classifier(pooled)
        return logits
