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

fig, axes = plt.subplots(2, 1, figsize=(7, 9))
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

# ── Figure 3: outcome_dist (Affirmed/Reversed: Patent Refusal vs Opp. Division) ─
binary_refusal_df2  = pd.read_pickle(f"{BASE}/../Data/Train&TestData_1.0_PatentRefusal.pkl")
opposition_df2      = pd.read_pickle(f"{BASE}/../Data/Train&TestData_1.0_OppositionDivision.pkl")
br_known2 = binary_refusal_df2[binary_refusal_df2["Outcome"] != "Unknown"].copy()
op_known2 = opposition_df2[opposition_df2["Outcome"] != "Unknown"].copy()

merged = pd.concat([op_known2, br_known2], ignore_index=True)
merged["Category"] = merged["Matches"].apply(
    lambda x: "Patent Refusal" if x == "Patent Refusal" else "Opposition Division"
)
pivot = merged.pivot_table(index="Category", columns="Outcome", aggfunc="size", fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(7, 5))
pivot_pct.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#F44336"])
ax.set_xlabel("Case Type")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(pivot_pct.index, rotation=0)
ax.legend(title="Outcome")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="center", color="white", fontweight="bold", fontsize=10)
plt.tight_layout()
out3_jpg = os.path.join(FIGS, "outcome_dist.jpg")
out3_png = os.path.join(FIGS, "outcome_dist.png")
fig.savefig(out3_jpg, dpi=150, bbox_inches="tight")
fig.savefig(out3_png, dpi=150, bbox_inches="tight")
print("Saved ->", out3_jpg)
plt.close()

# ── Figure 4: conf_matrices (per-issue, XGBoost, Exp 2; 2 on top, 1 on bottom) ─
from sklearn.metrics import confusion_matrix
ISSUE3  = ["IS-only", "Novelty-only", "IS+Novelty"]
CLASSES = ["Reversed", "Affirmed"]
xgb_both = pi[(pi["model"] == "XGBoost") & (pi["case"] == "both") & (pi["exp"].astype(str) == "2")]

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
axes = axes.ravel()
for ci, iss in enumerate(ISSUE3):
    ax = axes[ci]
    g = xgb_both[xgb_both["issue"] == iss]
    cm = confusion_matrix(g["y_true"], g["y_pred"], labels=[0, 1])
    rowsum = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, rowsum, out=np.zeros_like(cm, dtype=float), where=rowsum > 0)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
    multi = g["y_true"].nunique() > 1
    f1  = f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0) if multi else float("nan")
    mcc = matthews_corrcoef(g["y_true"], g["y_pred"]) if multi else float("nan")
    ax.set_title(f"{iss}\nn={len(g)}  F1={f1:.2f}  MCC={mcc:.2f}", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
axes[3].axis("off")
plt.tight_layout()
out4_jpg = os.path.join(FIGS, "conf_matrices.jpg")
out4_png = os.path.join(FIGS, "conf_matrices.png")
fig.savefig(out4_jpg, dpi=150, bbox_inches="tight")
fig.savefig(out4_png, dpi=150, bbox_inches="tight")
print("Saved ->", out4_jpg)
plt.close()

# ── Figure 5: pairwise_conf / stat_conf (pairwise issue-bucket significance) ──
from scipy import stats
from itertools import combinations
pi["correct"] = (pi["y_true"] == pi["y_pred"]).astype(int)
pooled = pi[(pi["exp"].astype(str) == "2") & (pi["case"] == "both")]   # both models pooled
k = len(ISSUE_ORDER)
pairs = list(combinations(range(k), 2))
P = np.full((k, k), np.nan)
for a, b in pairs:
    two = pooled[pooled["issue"].isin([ISSUE_ORDER[a], ISSUE_ORDER[b]])]
    cont = pd.crosstab(two["issue"].astype(str), two["correct"])
    if cont.shape == (2, 2):
        _, p, _, _ = stats.chi2_contingency(cont)
        P[a, b] = P[b, a] = min(1.0, p * len(pairs))   # Bonferroni

M = -np.log10(P)
vmax = max(3.0, np.nanmax(M))
fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
for a in range(k):
    for b in range(k):
        if a == b:
            ax.text(b, a, "—", ha="center", va="center", color="grey")
        elif not np.isnan(P[a, b]):
            p = P[a, b]
            label = "*" if p < .05 else "ns"
            ax.text(b, a, f"{p:.1e}\n{label}", ha="center", va="center", fontsize=9.5,
                    color="white" if M[a, b] < 0.6 * vmax else "black")
ax.set_xticks(range(k)); ax.set_xticklabels(ISSUE_ORDER, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(k)); ax.set_yticklabels(ISSUE_ORDER, fontsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("$-\\log_{10}(p_{bonf})$  (higher = more significant)", fontsize=8)
ythr = -np.log10(0.05)
cbar.ax.plot([0, 1], [ythr, ythr], color="red", lw=1.2,
             transform=cbar.ax.get_yaxis_transform(), clip_on=False)
cbar.ax.text(1.6, ythr, "p = 0.05", color="red", fontsize=7, va="center",
             transform=cbar.ax.get_yaxis_transform())
plt.tight_layout()
out5_jpg = os.path.join(FIGS, "pairwise_conf.jpg")
out5_png = os.path.join(FIGS, "pairwise_conf.png")
fig.savefig(out5_jpg, dpi=150, bbox_inches="tight")
fig.savefig(out5_png, dpi=150, bbox_inches="tight")
print("Saved ->", out5_jpg)
plt.close()

# ── Figure 6: od_appellant (OD outcome share by appellant type) ───────────────
od = pd.read_pickle(f"{BASE}/../Data/Train&TestData_1.0_OppositionDivision.pkl")
od = od[od["Matches"].isin(["1", "2", "3"]) & od["Outcome"].isin(["Affirmed", "Reversed"])].copy()
lab = {"1": "Patentee\nappellant", "2": "Opponent\nappellant", "3": "Both\nparties"}
od["Appellant"] = od["Matches"].map(lab)
od_order = ["Patentee\nappellant", "Opponent\nappellant", "Both\nparties"]
ct = pd.crosstab(od["Appellant"], od["Outcome"]).reindex(od_order)
n_od = ct.sum(1)
aff = 100 * ct["Affirmed"] / n_od
rev = 100 * ct["Reversed"] / n_od
base_aff = 100 * (od["Outcome"] == "Affirmed").mean()

C_AFF, C_REV = "#4C72B0", "#DD8452"
x = np.arange(len(od_order))
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.bar(x, aff, color=C_AFF, label="Affirmed (OD upheld)")
ax.bar(x, rev, bottom=aff, color=C_REV, label="Reversed (OD overturned)")
for i in range(len(od_order)):
    ax.text(i, aff.iloc[i] / 2, f"{aff.iloc[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, aff.iloc[i] + rev.iloc[i] / 2, f"{rev.iloc[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, 101.5, f"n={int(n_od.iloc[i]):,}", ha="center", fontsize=8.5)
ax.axhline(base_aff, color="black", ls="--", lw=1.3)
ax.text(len(od_order) - 0.5, base_aff + 1.2, f"overall Affirmed = {base_aff:.1f}%", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(od_order, fontsize=10)
ax.set_ylabel("% of cases"); ax.set_ylim(0, 108)
ax.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "od_appellant.jpg"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "od_appellant.png"), dpi=150, bbox_inches="tight")
print("Saved ->", os.path.join(FIGS, "od_appellant.jpg"))
plt.close()
