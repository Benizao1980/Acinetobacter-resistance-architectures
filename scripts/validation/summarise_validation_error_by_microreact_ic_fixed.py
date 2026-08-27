#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

IN = Path("outputs/error_characterisation/validation_prediction_error_table.with_microreact_ic.tsv")
OUTDIR = Path("outputs/error_characterisation")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, sep="\t")

# Keep only rows with truth and inferred/reference IC
df = df[df["true_mic"].notna()].copy()
df = df[df["IC_microreact_or_reference"].notna()].copy()

# Normalise within-column names
if "within_1_dilution" in df.columns:
    w1 = "within_1_dilution"
elif "within1" in df.columns:
    w1 = "within1"
else:
    raise SystemExit("No within-1 column found")

if "within_2_dilution" in df.columns:
    w2 = "within_2_dilution"
elif "within2" in df.columns:
    w2 = "within2"
else:
    raise SystemExit("No within-2 column found")

df[w1] = df[w1].astype(bool)
df[w2] = df[w2].astype(bool)

summary = (
    df.groupby(
        [
            "IC_microreact_or_reference",
            "IC_microreact_assignment_source",
            "model",
            "antibiotic",
        ],
        dropna=False,
    )
    .agg(
        n=("sample", "count"),
        mae_log2=("abs_log2_error", "mean"),
        median_abs_log2_error=("abs_log2_error", "median"),
        within_1=(w1, "mean"),
        within_2=(w2, "mean"),
        mean_signed_error=("log2_error", "mean"),
    )
    .reset_index()
)

summary["model_label"] = summary["model"].map({
    "amr": "AMR only",
    "locus": "Locus-GWAS",
    "hybrid": "Hybrid",
}).fillna(summary["model"])

summary["support_level"] = pd.cut(
    summary["n"],
    bins=[-1, 4, 9, 999999],
    labels=["very_low_n", "low_n", "supported_n"],
)

cols = [
    "IC_microreact_or_reference",
    "IC_microreact_assignment_source",
    "support_level",
    "model_label",
    "model",
    "antibiotic",
    "n",
    "mae_log2",
    "median_abs_log2_error",
    "within_1",
    "within_2",
    "mean_signed_error",
]

summary = summary[cols].sort_values(
    ["antibiotic", "IC_microreact_or_reference", "mae_log2"],
    ascending=[True, True, True],
)

summary.to_csv(
    OUTDIR / "validation_error_by_ST_inferred_IC_all_with_support.fixed.tsv",
    sep="\t",
    index=False,
)

clean = summary[summary["n"] >= 10].copy()
clean.to_csv(
    OUTDIR / "validation_error_by_ST_inferred_IC_clean.fixed.tsv",
    sep="\t",
    index=False,
)

best = (
    summary.sort_values(["antibiotic", "IC_microreact_or_reference", "mae_log2"])
    .groupby(["antibiotic", "IC_microreact_or_reference"], as_index=False)
    .head(1)
    .copy()
)

best.to_csv(
    OUTDIR / "validation_best_model_by_ST_inferred_IC_all_with_support.fixed.tsv",
    sep="\t",
    index=False,
)

print("\n## All ICs with support flags")
print(summary.to_string(index=False))

print("\n## Clean n>=10")
print(clean.to_string(index=False))

print("\n## Best model by IC/drug")
print(best.to_string(index=False))

print("\n[OK] wrote fixed IC summary tables")
