import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# OUTPUT
# =========================
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# =========================
# COLOURS (your scheme)
# =========================
IMI = "#c1121f"
MER = "#e03131"
GREY = "#b0b0b0"

# =========================
# FILES
# =========================
truth_file = "data/phenotypes_validation/validation_dataset_MIC.csv"

pred_sets = {
    "AMR only": {
        "imipenem": "outputs/runs/amr_only_regression/validation_amr/preds_imipenem_mic.tsv",
        "meropenem": "outputs/runs/amr_only_regression/validation_amr/preds_meropenem_mic.tsv",
    },
    "GWAS only": {
        "imipenem": "outputs/runs/gwas_only_regression/validation_gwas/preds_imipenem_mic.tsv",
        "meropenem": "outputs/runs/gwas_only_regression/validation_gwas/preds_meropenem_mic.tsv",
    },
    "Hybrid": {
        "imipenem": "outputs/runs/hybrid_regression/validation_hybrid/preds_imipenem_mic.tsv",
        "meropenem": "outputs/runs/hybrid_regression/validation_hybrid/preds_meropenem_mic.tsv",
    },
}

# =========================
# LOAD TRUTH
# =========================
truth = pd.read_csv(truth_file)
truth.columns = [c.strip() for c in truth.columns]

# fix ID mismatch (dot vs underscore)
def clean_id(x):
    return str(x).strip().replace(".", "_")

truth = truth.rename(columns={"id": "sample"})
truth["sample"] = truth["sample"].map(clean_id)

# find MIC columns robustly
def find_col(df, name):
    name = name.lower()
    for c in df.columns:
        if name in c.lower():
            return c
    raise ValueError(f"No column for {name}")

mic_cols = {
    "imipenem": find_col(truth, "imipenem"),
    "meropenem": find_col(truth, "meropenem"),
}

print("Using MIC columns:", mic_cols)

# =========================
# PREP FUNCTION
# =========================
def prep(pred_path, antibiotic):
    pred = pd.read_csv(pred_path, sep="\t")
    pred.columns = [c.strip() for c in pred.columns]

    pred = pred.rename(columns={"prediction": "pred_mic"})
    pred["sample"] = pred["sample"].map(clean_id)

    df = pred.merge(
        truth[["sample", mic_cols[antibiotic]]],
        on="sample",
        how="inner"
    )

    df = df.rename(columns={mic_cols[antibiotic]: "true_mic"})

    df["true_mic"] = pd.to_numeric(df["true_mic"], errors="coerce")
    df["pred_mic"] = pd.to_numeric(df["pred_mic"], errors="coerce")

    df = df.dropna(subset=["true_mic", "pred_mic"])
    df = df[(df["true_mic"] > 0) & (df["pred_mic"] > 0)]

    df["true_log2"] = np.log2(df["true_mic"])
    df["pred_log2"] = np.log2(df["pred_mic"])
    df["residual"] = df["pred_log2"] - df["true_log2"]

    return df

# =========================
# METRICS
# =========================
def metrics(df):
    err = df["residual"]
    return {
        "n": len(df),
        "mae": np.mean(np.abs(err)),
        "rmse": np.sqrt(np.mean(err**2)),
        "within1": np.mean(np.abs(err) <= 1),
        "within2": np.mean(np.abs(err) <= 2),
    }

# =========================
# LOAD ALL DATA
# =========================
data = {}
rows = []

for model, ab_dict in pred_sets.items():
    data[model] = {}
    for ab, path in ab_dict.items():
        df = prep(path, ab)
        data[model][ab] = df
        m = metrics(df)
        rows.append({"model": model, "antibiotic": ab, **m})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT / "fig3_metrics.tsv", sep="\t", index=False)

print("\n=== METRICS ===")
print(metrics_df)

# =========================
# FIGURE
# =========================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1,1,0.7], hspace=0.5, wspace=0.4)

models = ["AMR only", "GWAS only", "Hybrid"]
antibiotics = [("imipenem", IMI), ("meropenem", MER)]

# ---- SCATTERS ----
for i, model in enumerate(models):
    for j, (ab, color) in enumerate(antibiotics):
        ax = fig.add_subplot(gs[j, i])

        df = data[model][ab]

        ax.scatter(
            df["true_log2"],
            df["pred_log2"],
            s=8,
            alpha=0.6,
            color=color
        )

        lo, hi = df["true_log2"].min(), df["true_log2"].max()
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", lw=1)

        ax.set_title(f"{model}\n{ab} (n={len(df)})", fontsize=9)

        if j == 1:
            ax.set_xlabel("True MIC (log₂)")
        if i == 0:
            ax.set_ylabel("Predicted MIC (log₂)")

# ---- BARPLOT ----
ax = fig.add_subplot(gs[2, :])

x = np.arange(len(models))
width = 0.35

imi_vals = [metrics_df[(metrics_df.model==m)&(metrics_df.antibiotic=="imipenem")]["within1"].values[0] for m in models]
mer_vals = [metrics_df[(metrics_df.model==m)&(metrics_df.antibiotic=="meropenem")]["within1"].values[0] for m in models]

ax.bar(x - width/2, imi_vals, width, color=IMI, label="Imipenem")
ax.bar(x + width/2, mer_vals, width, color=MER, label="Meropenem")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("±1 dilution accuracy")
ax.set_ylim(0, 1)
ax.legend()

# =========================
# SAVE
# =========================
plt.savefig(OUT / "Figure3.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT / "Figure3.svg", bbox_inches="tight")

print("\nSaved →", OUT / "Figure3.png")