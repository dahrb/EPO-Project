"""SHAP analysis for BERT-based models (LegalBERT, PatentBERT) on EPO patent
appeal data.

Uses shap.PartitionExplainer (auto-selected) with a word-level Text masker.
Each explanation requires many masked forward passes through BERT, so this is
much slower than TreeSHAP — set --n_samples to a manageable number (50–150).

Mirrors shap_xgboost_run.py structure across the same experiment setups:
  - Loads the best-checkpoint from Results/results_deep_learning.json
  - Uses Data/Final_Processed/ train/test splits
  - Produces (all saved to --out_dir):
      1. Global bar chart of mean |SHAP| (top 20 words) on --n_samples test cases
      2. Force plots (HTML) for 3 local cases: correct_reversed, correct_affirmed, incorrect
      3. Text highlight plots (HTML) for the same 3 cases

Usage examples:
  python Results/shap_bert_run.py --model legalbert  --experiment 2 --case pf
  python Results/shap_bert_run.py --model patentbert --experiment 2 --case op --n_samples 50
  python Results/shap_bert_run.py --model legalbert  --experiment 1 --case both --n_samples 75
"""

import sys
import os
import types
import argparse
import json
import random
from pathlib import Path

import numpy as np

# ── numpy 2.0 / cupy compatibility shim (must come before any spaCy import) ──
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "AxisError"):
    np.AxisError = np.exceptions.AxisError

_cupy_stub = types.ModuleType("cupy")
_cupy_stub.ndarray = np.ndarray
for _sub in ["cuda", "cuda.stream", "testing", "_core", "_core.core"]:
    sys.modules[f"cupy.{_sub}"] = types.ModuleType(f"cupy.{_sub}")
sys.modules["cupy"] = _cupy_stub
del _cupy_stub, _sub

import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification
from Experiments.deep_learning_experiments import (
    DeepLearningExperiments,
    OppositionModeClassificationHead,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SHAP for LegalBERT / PatentBERT")
    p.add_argument("--model",       default="legalbert",
                   choices=["legalbert", "patentbert"],
                   help="Which BERT model to explain")
    p.add_argument("--experiment",  default="2", choices=["1", "2"])
    p.add_argument("--case",        default="pf", choices=["pf", "op", "both"])
    p.add_argument("--n_samples",   type=int, default=100,
                   help="Test samples for global SHAP (stratified 50/50 by label). "
                        "Set to -1 to use the full test set.")
    p.add_argument("--out_dir",     default="Results/shap_bert",
                   help="Output directory")
    p.add_argument("--dl_results",  default="Results/results_deep_learning.json")
    p.add_argument("--data_dir",    default="Data/Final_Processed")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hp_record(dl_results_path, model_name, experiment, case_mode):
    """Return best HP record from results_deep_learning.json."""
    with open(dl_results_path) as f:
        data = json.load(f)
    matches = [
        r for r in data
        if r.get("algo") == model_name
        and str(r.get("experiment")) == str(experiment)
        and r.get("case_mode") == case_mode
    ]
    if not matches:
        raise ValueError(
            f"No HP record found for {model_name} exp={experiment} case={case_mode}"
        )
    record = max(matches, key=lambda r: r.get("best_score_val", 0.0))
    print(
        f"[HP Record] {model_name} exp={experiment} case={case_mode} "
        f"val_f1={record['best_score_val']:.4f}  path={record['best_model_path']}"
    )
    return record


def load_model_from_checkpoint(record, device):
    """
    Instantiate DeepLearningExperiments, rebuild the model architecture, and
    restore weights from the saved .pt checkpoint.

    For opposition mode the custom_head dimensions are inferred from the
    checkpoint's weight shapes (avoiding any placeholder-size mismatch).
    """
    model_name = record["algo"]
    experiment = str(record["experiment"])
    case_mode  = record["case_mode"]
    opposition = record["opposition"]
    dropout    = float(record["best_params"].get("dropout", 0.1))
    ckpt_path  = record["best_model_path"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build experiment wrapper (sets up tokenizer, model_mapping, device, …)
    dl_exp = DeepLearningExperiments(
        model_name=model_name,
        experiment=experiment,
        opposition=opposition,
        case_mode=case_mode,
        device=device,
    )

    model = dl_exp._build_model(dropout=dropout)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Restore opposition head with correct dimensions from checkpoint
    if opposition and ckpt.get("custom_head_state") is not None:
        head_state  = ckpt["custom_head_state"]
        fused_dim   = head_state["fusion_layer.weight"].shape[1]   # input dim
        hidden_size = model.config.hidden_size
        aux_dim     = fused_dim - hidden_size
        dl_exp.custom_head = OppositionModeClassificationHead(
            text_embed_dim=hidden_size,
            aux_feature_dim=aux_dim,
            num_labels=2,
        ).to(device)
        dl_exp.custom_head.load_state_dict(head_state)
        dl_exp.custom_head.eval()

    print(
        f"[Model] Loaded epoch={ckpt.get('best_epoch','?')} "
        f"val_f1={ckpt.get('val_f1', 0.0):.4f}"
    )
    return model, dl_exp


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_texts(dl_exp, raw_texts):
    """Apply BERT fixed preprocessing (stopwords + lemmatisation) to a list of strings."""
    df = pd.DataFrame({"New Summary Facts": raw_texts})
    df_proc = dl_exp._preprocess_text_for_bert(df)
    return df_proc["New Summary Facts"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction wrappers
# ─────────────────────────────────────────────────────────────────────────────

def build_predict_fn(model, dl_exp, device, aux_features_fixed=None):
    """
    Return a function  f(texts: list[str]) -> np.ndarray shape (n, 2)
    suitable for shap.Explainer.

    For opposition mode, aux_features_fixed is a (1, n_aux) array of
    structured features held constant while SHAP masks different words.
    """
    def predict_fn(texts):
        processed = preprocess_texts(dl_exp, list(texts))
        input_ids, attn_masks = dl_exp._tokenize_batch(processed)
        ids_t  = torch.tensor(input_ids,   dtype=torch.long).to(device)
        mask_t = torch.tensor(attn_masks,  dtype=torch.long).to(device)

        model.eval()
        with torch.no_grad():
            if dl_exp.opposition and aux_features_fixed is not None:
                n   = ids_t.shape[0]
                aux = torch.tensor(
                    np.tile(aux_features_fixed, (n, 1)), dtype=torch.float32
                ).to(device)
                enc     = dl_exp._get_encoder(model)
                out     = enc(input_ids=ids_t, attention_mask=mask_t)
                if dl_exp.model_name == "patentbert":
                    text_emb = out.last_hidden_state[:, 0, :]
                else:
                    text_emb = (out.pooler_output
                                if out.pooler_output is not None
                                else out.last_hidden_state[:, 0, :])
                logits = dl_exp.custom_head(text_emb, aux)
            elif dl_exp.model_name == "patentbert":
                enc      = dl_exp._get_encoder(model)
                out      = enc(input_ids=ids_t, attention_mask=mask_t)
                text_emb = out.last_hidden_state[:, 0, :]
                logits   = model.classifier(text_emb)
            else:
                out    = model(input_ids=ids_t, attention_mask=mask_t)
                logits = out.logits

            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return probs

    return predict_fn


def get_all_predictions(model, dl_exp, X_test, y_test_1d, device, batch_size=32):
    """Run inference over the full test set, returning (preds, probs)."""
    all_preds, all_probs = [], []

    texts_processed = preprocess_texts(dl_exp, X_test["New Summary Facts"].tolist())

    # Fit aux encoder once on X_test (already fitted on train; just encode here)
    aux_all = None
    if dl_exp.opposition:
        aux_all = dl_exp._encode_opposition_features(X_test, fit=False)

    for i in range(0, len(texts_processed), batch_size):
        batch_texts = texts_processed[i : i + batch_size]
        ids, masks  = dl_exp._tokenize_batch(batch_texts)
        ids_t   = torch.tensor(ids,   dtype=torch.long).to(device)
        mask_t  = torch.tensor(masks, dtype=torch.long).to(device)

        model.eval()
        with torch.no_grad():
            if dl_exp.opposition:
                aux_batch = torch.tensor(
                    aux_all[i : i + batch_size], dtype=torch.float32
                ).to(device)
                enc     = dl_exp._get_encoder(model)
                out     = enc(input_ids=ids_t, attention_mask=mask_t)
                if dl_exp.model_name == "patentbert":
                    text_emb = out.last_hidden_state[:, 0, :]
                else:
                    text_emb = (out.pooler_output
                                if out.pooler_output is not None
                                else out.last_hidden_state[:, 0, :])
                logits = dl_exp.custom_head(text_emb, aux_batch)
            elif dl_exp.model_name == "patentbert":
                enc      = dl_exp._get_encoder(model)
                out      = enc(input_ids=ids_t, attention_mask=mask_t)
                text_emb = out.last_hidden_state[:, 0, :]
                logits   = model.classifier(text_emb)
            else:
                out    = model(input_ids=ids_t, attention_mask=mask_t)
                logits = out.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(PROJECT_ROOT) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_exp{args.experiment}_{args.case}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    record = load_hp_record(
        Path(PROJECT_ROOT) / args.dl_results, args.model, args.experiment, args.case
    )
    model, dl_exp = load_model_from_checkpoint(record, device)

    # ── Load data ────────────────────────────────────────────────────────────
    data_dir = Path(PROJECT_ROOT) / args.data_dir
    X_train = pd.read_pickle(data_dir / f"X_Train_{args.experiment}_{args.case}.pkl")
    y_train = pd.read_pickle(data_dir / f"y_Train_{args.experiment}_{args.case}.pkl")
    X_test  = pd.read_pickle(data_dir / f"X_test_{args.experiment}_{args.case}.pkl")
    y_test  = pd.read_pickle(data_dir / f"y_test_{args.experiment}_{args.case}.pkl")

    y_train_1d = dl_exp._to_1d(y_train)
    y_test_1d  = dl_exp._to_1d(y_test)
    print(f"[Data] Train={len(X_train):,}  Test={len(X_test):,}")

    # ── Fit aux encoder on training data (opposition mode) ───────────────────
    if dl_exp.opposition:
        dl_exp._encode_opposition_features(X_train, fit=True)
        print("[Aux] Encoder fitted on training data")

    # ── Full-test predictions (needed to select local cases) ─────────────────
    print("[Predict] Running inference on full test set...")
    y_pred, y_probs = get_all_predictions(model, dl_exp, X_test, y_test_1d, device)

    acc = (y_pred == y_test_1d).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test_1d, y_pred, zero_division=0)
    print(f"[Results] Accuracy={acc:.4f}  F1={f1:.4f}")

    # Identify cases of interest (same logic as shap_xgboost_run.py)
    misclassified    = np.where(y_test_1d != y_pred)[0]
    correct_reversed = np.where((y_test_1d == y_pred) & (y_test_1d == 0))[0]
    correct_affirmed = np.where((y_test_1d == y_pred) & (y_test_1d == 1))[0]
    all_correct      = np.where(y_test_1d == y_pred)[0]

    np.random.seed(args.seed)
    idx_corr_rev = int(np.random.choice(correct_reversed) if len(correct_reversed) > 0 else all_correct[0])
    idx_corr_aff = int(np.random.choice(correct_affirmed) if len(correct_affirmed) > 0 else all_correct[0])
    idx_incorrect = int(np.random.choice(misclassified)   if len(misclassified)    > 0 else all_correct[0])

    local_cases = {
        "correct_reversed": (idx_corr_rev, 0, "Correctly predicted Reversed"),
        "correct_affirmed": (idx_corr_aff, 1, "Correctly predicted Affirmed"),
        "incorrect":        (idx_incorrect, y_test_1d[idx_incorrect],
                             f"Incorrect  true={y_test_1d[idx_incorrect]} pred={y_pred[idx_incorrect]}"),
    }
    for scenario, (idx, true_label, desc) in local_cases.items():
        print(f"[Case] {scenario}: idx={idx}  true={true_label}  pred={y_pred[idx]}")

    # ── Preprocess all test texts once ───────────────────────────────────────
    print("\n[Preprocess] Applying BERT text preprocessing to test set...")
    texts_processed = preprocess_texts(dl_exp, X_test["New Summary Facts"].tolist())

    # ── Encode aux features for opposition test set ──────────────────────────
    aux_all = None
    if dl_exp.opposition:
        aux_all = dl_exp._encode_opposition_features(X_test, fit=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Global SHAP — bar chart of mean |SHAP| across test sample
    # ─────────────────────────────────────────────────────────────────────────
    aff_idx = np.where(y_test_1d == 1)[0]
    rev_idx = np.where(y_test_1d == 0)[0]
    np.random.seed(args.seed)
    if args.n_samples <= 0:
        # Use the full test set in random order
        global_idx = np.random.permutation(len(y_test_1d))
        print(
            f"\n[SHAP] Global analysis on full test set ({len(global_idx)} samples). "
            f"This may take a very long time..."
        )
    else:
        n_per_class = args.n_samples // 2
        global_idx  = np.concatenate([
            np.random.choice(aff_idx, size=min(n_per_class, len(aff_idx)), replace=False),
            np.random.choice(rev_idx, size=min(n_per_class, len(rev_idx)), replace=False),
        ])
        np.random.shuffle(global_idx)
        print(
            f"\n[SHAP] Global analysis on {len(global_idx)} test samples "
            f"(~{n_per_class} per class). This may take several minutes..."
        )
    global_texts = [texts_processed[i] for i in global_idx]

    # For global SHAP in opposition mode, fix aux to per-sample values
    # by building a single shared predict fn with mean aux as background.
    if dl_exp.opposition:
        aux_mean     = aux_all.mean(axis=0, keepdims=True)
        predict_fn_g = build_predict_fn(model, dl_exp, device, aux_features_fixed=aux_mean)
    else:
        predict_fn_g = build_predict_fn(model, dl_exp, device)

    masker    = shap.maskers.Text(tokenizer=r"\b\w+\b")
    explainer = shap.Explainer(
        predict_fn_g, masker,
        output_names=["Reversed", "Affirmed"],
        seed=args.seed,
    )
    shap_global = explainer(global_texts)
    print(f"[SHAP] Global explanations done. Shape: {shap_global.shape}")

    # Bar chart — mean |SHAP| per word, Affirmed class
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_global[:, :, 1], max_display=20, ax=ax, show=False)
    ax.set_title(
        f"Mean |SHAP| per Word — {args.model.upper()} Exp {args.experiment} "
        f"{args.case.upper()}\n"
        f"(n={len(global_idx)} test samples, class = Affirmed)",
        fontsize=12,
    )
    plt.tight_layout()
    bar_path = out_dir / f"{tag}_global_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {bar_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Local SHAP — 3 representative cases
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[SHAP] Local explanations for 3 representative cases...")

    for scenario, (case_idx, true_label, description) in local_cases.items():
        print(f"\n  [{scenario}] {description}")

        # Per-case predict fn (for opposition, use that case's exact aux features)
        if dl_exp.opposition:
            aux_case     = aux_all[case_idx : case_idx + 1]
            predict_fn_l = build_predict_fn(model, dl_exp, device, aux_features_fixed=aux_case)
        else:
            predict_fn_l = build_predict_fn(model, dl_exp, device)

        local_explainer = shap.Explainer(
            predict_fn_l, masker,
            output_names=["Reversed", "Affirmed"],
            seed=args.seed,
        )
        case_text = texts_processed[case_idx]
        sv        = local_explainer([case_text])   # Explanation shape (1, n_tokens, 2)
        sv_aff    = sv[0, :, 1]                    # Affirmed class, single doc

        # ── Force plot (HTML) ─────────────────────────────────────────────
        force_plot = shap.force_plot(
            base_value   = float(sv_aff.base_values),
            shap_values  = sv_aff.values,
            feature_names= list(sv_aff.data),
            show         = False,
        )
        force_path = out_dir / f"{tag}_{scenario}_force.html"
        shap.save_html(str(force_path), force_plot)
        print(f"    Saved: {force_path.name}")

        # ── Text highlight plot (HTML) ────────────────────────────────────
        try:
            text_html = shap.plots.text(sv_aff, display=False)
            text_path = out_dir / f"{tag}_{scenario}_text.html"
            with open(text_path, "w") as fh:
                fh.write(text_html)
            print(f"    Saved: {text_path.name}")
        except Exception as exc:
            print(f"    [Warning] text plot failed ({exc}); skipping text HTML")

    print(f"\n[Done] All outputs saved to: {out_dir}")
    print("\nSummary of output files:")
    for f in sorted(out_dir.glob(f"{tag}_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
