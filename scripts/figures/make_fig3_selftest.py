#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

IMI = "#c1121f"
MER = "#e03131"
GREY = "#d9d9d9"

pred_sets = {
    "Hybrid": {
        "imipenem": "outputs/runs_directional/hybrid_directional_unitig79_regression/imipenem_self_test_predictions.tsv",
        "meropenem": "outputs/runs_directional/hybrid_directional_unitig79_regression/meropenem_self_test_predictions.tsv",
    },
    "AMR only": {
        "imipenem": "outputs/runs_selftest/amr_only/amr_selftest/imipenem_self_test_predictions.tsv",
        "meropenem": "outputs/runs_selftest/amr_only/amr_selftest/meropenem_self_test_predictions.tsv",
    },
}

def prep(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    def find_col(possible):
        for p in possible:
            for c in df.columns:
                if p in c.lower():
                    return c
        raise ValueError(f"Could not find column from {possible}. Columns={df.columns.tolist()}")

    pred_col = find_col(["prediction", "pred", "y_pred"])
    truth_col = find_col(["truth", "true", "y_true", "observed"])

    df["pred_log2"] = pd.to_numeric(df[pred_col], errors="coerce")
    df["true_log2"] = pd.to_numeric(df[truth_col], errors="coerce")
    df = df.dropna(subset=["pred_log2", "true_log2"])
    df["residual"] = df["pred_log2"] - df["true_log2"]
    return df

def metrics(df):
    err = df["residual"]
    return {
        "n": len(df),
        "mae": np.mean(np.abs(err)),
        "rmse": np.sqrt(np.mean(err**2)),
        "within1": np.mean(np.abs(err) <= 1),
        "within2": np.mean(np.abs(err) <= 2),
    }

rows = []
data = {}
for model, ab_files in pred_sets.items():
    data[model] = {}
    for ab, fp in ab_files.items():
        d = prep(fp)
        data[model][ab] = d
        rows.append({"model": model, "antibiotic": ab, **metrics(d)})

met = pd.DataFrame(rows)
met.to_csv(OUT / "fig3_selftest_metrics.tsv", sep="\t", index=False)
print(met.to_string(index=False))

models = list(pred_sets.keys())
antibiotics = [("imipenem", IMI), ("meropenem", MER)]

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(3, len(models), height_ratios=[1,1,0.7], hspace=0.55, wspace=0.4)

for i, model in enumerate(models):
    for j, (ab, color) in enumerate(antibiotics):
        ax = fig.add_subplot(gs[j, i])
        d = data[model][ab]

        ax.scatter(d["true_log2"], d["pred_log2"], s=14, alpha=0.65, color=color, edgecolor="none")

        lo = min(d["true_log2"].min(), d["pred_log2"].min()) - 0.5
        hi = max(d["true_log2"].max(), d["pred_log2"].max()) + 0.5
        ax.fill_between([lo, hi], [lo-1, hi-1], [lo+1, hi+1], color=GREY, alpha=0.25, zorder=0)
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1)

        m = metrics(d)
        ax.text(0.03, 0.95, f"n={m['n']}\n±1={m['within1']*100:.1f}%", transform=ax.transAxes,
                va="top", fontsize=8)

        ax.set_title(f"{model}\n{ab}", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if i == 0:
            ax.set_ylabel("Predicted MIC (log₂)")
        if j == 1:
            ax.set_xlabel("True MIC (log₂)")
        ax.grid(True, alpha=0.25)

ax = fig.add_subplot(gs[2, :])
x = np.arange(len(models))
width = 0.35

imi = [met[(met.model == m) & (met.antibiotic == "imipenem")]["within1"].iloc[0] * 100 for m in models]
mer = [met[(met.model == m) & (met.antibiotic == "meropenem")]["within1"].iloc[0] * 100 for m in models]

ax.bar(x - width/2, imi, width, color=IMI, label="Imipenem")
ax.bar(x + width/2, mer, width, color=MER, label="Meropenem")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Within ±1 dilution (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

fig.savefig(OUT / "Figure3_selftest.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT / "Figure3_selftest.svg", bbox_inches="tight")

print("Saved outputs/figures/Figure3_selftest.png/svg")
