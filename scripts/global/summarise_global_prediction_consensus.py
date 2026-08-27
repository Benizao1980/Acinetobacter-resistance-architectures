#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("outputs/global_prediction/stratified/global_predictions_enriched.tsv")
OUTDIR = Path("outputs/global_prediction/stratified")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, sep="\t", low_memory=False)

MODELS = ["amr", "locus", "hybrid"]
ANTIBIOTICS = ["imipenem", "meropenem"]
THRESHOLDS = [2, 8, 16, 32]

MODEL_LABELS = {
    "amr": "AMR only",
    "locus": "Locus-GWAS",
    "hybrid": "Hybrid",
}

AB_LABELS = {
    "imipenem": "Imipenem",
    "meropenem": "Meropenem",
}


def add_flags(dat):
    dat = dat.copy()

    for ab in ANTIBIOTICS:
        for thr in THRESHOLDS:
            cols = []
            for model in MODELS:
                col = f"{model}_{ab}_pred_mic"
                if col in dat.columns:
                    flag_col = f"{model}_{ab}_gt{thr}"
                    dat[flag_col] = pd.to_numeric(dat[col], errors="coerce") > thr
                    cols.append(flag_col)

            if cols:
                dat[f"{ab}_n_models_gt{thr}"] = dat[cols].sum(axis=1)
                dat[f"{ab}_consensus_gt{thr}"] = dat[f"{ab}_n_models_gt{thr}"] >= 2

    return dat


def summarise_by(group_cols, label, min_n=1):
    rows = []

    grouped = df.groupby(group_cols, dropna=False)

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        n = len(sub)
        if n < min_n:
            continue

        row = dict(zip(group_cols, key))
        row["n"] = n

        for ab in ANTIBIOTICS:
            for model in MODELS:
                col = f"{model}_{ab}_pred_mic"
                if col not in sub.columns:
                    continue

                vals = pd.to_numeric(sub[col], errors="coerce")
                row[f"{model}_{ab}_median_mic"] = vals.median()
                row[f"{model}_{ab}_iqr_low"] = vals.quantile(0.25)
                row[f"{model}_{ab}_iqr_high"] = vals.quantile(0.75)

            for thr in THRESHOLDS:
                c = f"{ab}_consensus_gt{thr}"
                if c in sub.columns:
                    row[f"{ab}_consensus_gt{thr}_n"] = int(sub[c].sum())
                    row[f"{ab}_consensus_gt{thr}_prop"] = sub[c].mean()

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(["n"], ascending=False)

    fp = OUTDIR / f"global_prediction_consensus_by_{label}.tsv"
    out.to_csv(fp, sep="\t", index=False)
    print("[OK] wrote", fp)

    return out


df = add_flags(df)

# Write enriched table with consensus flags.
df.to_csv(
    OUTDIR / "global_predictions_enriched.with_consensus.tsv",
    sep="\t",
    index=False,
)

# Overall summary
overall_rows = []

for ab in ANTIBIOTICS:
    for model in MODELS:
        col = f"{model}_{ab}_pred_mic"
        vals = pd.to_numeric(df[col], errors="coerce")

        overall_rows.append({
            "antibiotic": AB_LABELS[ab],
            "model": MODEL_LABELS[model],
            "n": vals.notna().sum(),
            "median_mic": vals.median(),
            "iqr_low": vals.quantile(0.25),
            "iqr_high": vals.quantile(0.75),
            "p90": vals.quantile(0.90),
            "p95": vals.quantile(0.95),
            "p99": vals.quantile(0.99),
            "max_mic": vals.max(),
            "n_gt2": int((vals > 2).sum()),
            "prop_gt2": (vals > 2).mean(),
            "n_gt8": int((vals > 8).sum()),
            "prop_gt8": (vals > 8).mean(),
            "n_gt16": int((vals > 16).sum()),
            "prop_gt16": (vals > 16).mean(),
            "n_gt32": int((vals > 32).sum()),
            "prop_gt32": (vals > 32).mean(),
        })

overall = pd.DataFrame(overall_rows)
overall.to_csv(
    OUTDIR / "global_prediction_overall_by_model.tsv",
    sep="\t",
    index=False,
)

# Consensus overall
cons_rows = []
for ab in ANTIBIOTICS:
    row = {
        "antibiotic": AB_LABELS[ab],
        "n": len(df),
    }
    for thr in THRESHOLDS:
        c = f"{ab}_consensus_gt{thr}"
        row[f"consensus_gt{thr}_n"] = int(df[c].sum())
        row[f"consensus_gt{thr}_prop"] = df[c].mean()
    cons_rows.append(row)

cons = pd.DataFrame(cons_rows)
cons.to_csv(
    OUTDIR / "global_prediction_overall_consensus.tsv",
    sep="\t",
    index=False,
)

# Stratified summaries
summarise_by(["IC"], "IC", min_n=1)
summarise_by(["ST_Pasteur"], "ST_Pasteur", min_n=5)
summarise_by(["ST_Oxford"], "ST_Oxford", min_n=5)
summarise_by(["Country"], "Country", min_n=10)
summarise_by(["amrfinder_carbapenemase_group"], "amrfinder_mechanism", min_n=1)

print("\n## Overall model summary")
print(overall.to_string(index=False))

print("\n## Overall consensus summary")
print(cons.to_string(index=False))

print("\n## IC counts")
print(df["IC"].value_counts(dropna=False).to_string())

print("\n## Mechanism counts")
print(df["amrfinder_carbapenemase_group"].value_counts(dropna=False).to_string())
