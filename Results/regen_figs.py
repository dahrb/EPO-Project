"""Regenerate patent_vs_legal.jpg and patentability_results.jpg with macro-F1."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

BASE = "/mnt/scratch/users/sgdbareh/EPO_NEW/EPO-Project/Results"
FIGS = "/mnt/scratch/users/sgdbareh/EPO_NEW/EPO-Project/Figs/eda"

# ── Load data ────────────────────────────────────────────────────────────────
with open(f"{BASE}/results_main.json") as f:
    results_main = json.load(f)
with open(f"{BASE}/results_ml_sliding_window.json") as f:
    results_ml_sw = json.load(f)
with open(f"{BASE}/results_dl_full_test.json") as f:
    results_dl_ft = json.load(f)
with open(f"{BASE}/results_dl_sliding_window.json") as f:
    results_dl_sw = json.load(f)

ALGO_NAMES = {"RandomForestClassifier": "RandomForest", "XGBClassifier": "XGBoost"}
def clean_algo(name): return ALGO_NAMES.get(name, name)

# ── Figure 1: patentability_results.jpg ──────────────────────────────────────
pi = pd.read_pickle(f"{BASE}/per_issue_predictions.pkl")
ISSUE_ORDER = ["IS-only", "Novelty-only", "IS+Novelty", "Other"]
pi["issue"] = pd.Categorical(pi["issue"], categories=ISSUE_ORDER, ordered=True)

pi2   = pi[pi["exp"] == "2"].copy()
both2 = pi2[pi2["case"] == "both"]

MODELS = ["XGBoost", "Longformer"]
mcol   = {"XGBoost": "#4C72B0", "Longformer": "#55A868"}

f1_macro = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Experiment 2 — Full-Test Performance by Patentability Issue (Combined)",
             fontweight="bold")
for ax, metric, fn in zip(axes, ["F1", "MCC"], [f1_macro, matthews_corrcoef]):
    x = np.arange(len(ISSUE_ORDER)); w = 0.38
    for mi, model in enumerate(MODELS):
        sub  = both2[both2["model"] == model]
        vals = []
        for iss in ISSUE_ORDER:
            g = sub[sub["issue"] == iss]
            vals.append(fn(g["y_true"].values, g["y_pred"].values)
                        if len(g) and g["y_true"].nunique() > 1 else np.nan)
        bars = ax.bar(x + (mi - 0.5) * w, vals, w, label=model,
                      color=mcol[model], alpha=0.88)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(ISSUE_ORDER, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0 if metric == "F1" else 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(metric)
axes[0].legend(fontsize=9)
plt.tight_layout()
out1_jpg = os.path.join(FIGS, "patentability_results.jpg")
out1_png = os.path.join(FIGS, "patentability_results.png")
fig.savefig(out1_jpg, dpi=150, bbox_inches="tight")
fig.savefig(out1_png, dpi=150, bbox_inches="tight")
print("Saved ->", out1_jpg)
plt.close()

# ── Figure 2: patent_vs_legal.jpg (CI style) ─────────────────────────────────
CASES3 = ["pf", "op", "both"]; YEARS = [2021, 2022, 2023, 2024]
EMB_CLF = "XGBClassifier"

def _dl_ft(a, c):
    return next((r["test_metrics"]["f1"] for r in results_dl_ft
                 if r["algo"]==a and str(r["experiment"])=="2" and r["case_mode"]==c), None)
def _dl_sw(a, c, y):
    return next((r["test_metrics"]["f1"] for r in results_dl_sw
                 if r["algo"]==a and r["case_mode"]==c and int(r["test_year"])==y), None)
def _emb_ft(e, c):
    return next((r["test_metrics"]["f1"] for r in results_main
                 if r.get("mode")=="embedding" and r.get("embedding")==e
                 and r["algo"]==EMB_CLF and str(r["experiment"])=="2"
                 and r["case_mode"]==c), None)
def _emb_sw(e, c, y):
    return next((r["test_metrics"]["f1"] for r in results_ml_sw
                 if r["input_representation"].lower()==e and r["algo"]==EMB_CLF
                 and r["case_mode"]==c and int(r["test_year"])==y), None)
def _dl_vec(a):
    return [_dl_ft(a,c) for c in CASES3] + [_dl_sw(a,c,y) for c in CASES3 for y in YEARS]
def _emb_vec(e):
    return [_emb_ft(e,c) for c in CASES3] + [_emb_sw(e,c,y) for c in CASES3 for y in YEARS]

DOMAIN = {"LegalBERT":    ("Legal",  "BERT"),
          "Law2Vec":       ("Legal",  "Embedding"),
          "PatentBERT":    ("Patent", "BERT"),
          "Patent2Vec":    ("Patent", "Embedding"),
          "PatentDoc2Vec": ("Patent", "Embedding")}
vecs = {"LegalBERT":    _dl_vec("legalbert"),
        "PatentBERT":   _dl_vec("patentbert"),
        "Law2Vec":      _emb_vec("law2vec"),
        "Patent2Vec":   _emb_vec("patent2vec"),
        "PatentDoc2Vec": _emb_vec("doc2vec")}

order = ["LegalBERT", "PatentBERT", "Law2Vec", "Patent2Vec", "PatentDoc2Vec"]
dcol  = {"Legal": "#4C72B0", "Patent": "#DD8452"}
xpos  = [0, 1, 2.5, 3.5, 4.5]

def _ci95(v):
    a = np.array([x for x in v if x is not None]); n = len(a)
    from scipy.stats import t
    half = t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return a.mean(), half, a

fig, ax = plt.subplots(figsize=(10, 5))
rng = np.random.default_rng(0)
for x, m in zip(xpos, order):
    mean, half, a = _ci95(vecs[m])
    c = dcol[DOMAIN[m][0]]
    ax.scatter(np.full(len(a), x) + (rng.random(len(a)) - 0.5) * 0.28, a,
               s=18, color=c, alpha=0.18, zorder=2, edgecolors="none")
    ax.errorbar(x, mean, yerr=half, fmt="o", color=c, ecolor=c,
                elinewidth=2.4, capsize=7, markersize=10, zorder=3)
    ax.text(x, mean + half + 0.012, f"{mean:.3f}\n±{half:.3f}",
            ha="center", va="bottom", fontsize=8.5)

ax.axvline(1.75, color="grey", ls="--", alpha=0.6)
y0 = ax.get_ylim()[0]
ax.text(0.5,  y0, "BERT",                 ha="center", va="top", fontsize=10, style="italic")
ax.text(3.5,  y0, "Embedding (XGBoost)",  ha="center", va="top", fontsize=10, style="italic")
ax.set_xticks(xpos); ax.set_xticklabels(order, rotation=15)
ax.set_ylabel("Macro-F1 (per Exp 2 condition)")
ax.set_title("Legal vs Patent Domain-Specific Models — Exp 2 (Full Test + Sliding Window)\nmean ± 95% CI",
             fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=[Patch(facecolor=dcol["Legal"],   label="Legal domain"),
                   Patch(facecolor=dcol["Patent"], label="Patent domain")],
          loc="lower right")
plt.tight_layout()
out2_jpg = os.path.join(FIGS, "patent_vs_legal.jpg")
out2_png = os.path.join(FIGS, "patent_vs_legal.png")
fig.savefig(out2_jpg, dpi=150, bbox_inches="tight")
fig.savefig(out2_png, dpi=150, bbox_inches="tight")
print("Saved ->", out2_jpg)
plt.close()
