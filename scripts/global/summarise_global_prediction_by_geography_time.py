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
THR = 8

# Keep threshold flags consistent with the global prediction figure:
# consensus = at least 2 of 3 feature sets predict MIC > threshold.
for ab in ANTIBIOTICS:
    flag_cols = []
    for model in MODELS:
        col = f"{model}_{ab}_pred_mic"
        if col in df.columns:
            flag = f"{model}_{ab}_gt{THR}"
            df[flag] = pd.to_numeric(df[col], errors="coerce") > THR
            flag_cols.append(flag)
    df[f"{ab}_n_models_gt{THR}"] = df[flag_cols].sum(axis=1)
    df[f"{ab}_consensus_gt{THR}"] = df[f"{ab}_n_models_gt{THR}"] >= 2

df[f"any_consensus_gt{THR}"] = df[[f"{ab}_consensus_gt{THR}" for ab in ANTIBIOTICS]].any(axis=1)

# Basic continent mapping from country labels.
continent_map = {
    "USA": "North America",
    "Canada": "North America",
    "Brazil": "South America",
    "Argentina": "South America",
    "China": "Asia",
    "India": "Asia",
    "Japan": "Asia",
    "Singapore": "Asia",
    "Thailand": "Asia",
    "Malaysia": "Asia",
    "Pakistan": "Asia",
    "Israel": "Asia",
    "Iraq": "Asia",
    "Lebanon": "Asia",
    "Jordan": "Asia",
    "South Korea": "Asia",
    "Germany": "Europe",
    "Croatia": "Europe",
    "Czech Republic": "Europe",
    "Italy": "Europe",
    "The Netherlands": "Europe",
    "France": "Europe",
    "Poland": "Europe",
    "Denmark": "Europe",
    "Sweden": "Europe",
    "UK": "Europe",
    "Australia": "Oceania",
    "Ethiopia": "Africa",
    "Egypt": "Africa",
}

df["Country_clean"] = df["Country"].astype(str).replace({"nan": np.nan, "Unknown": np.nan})
df["Continent_inferred"] = df["Country_clean"].map(continent_map).fillna("Unknown")

df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
df["Decade"] = (df["Year_num"] // 10 * 10).astype("Int64").astype(str).replace("<NA>", "Unknown") + "s"
df.loc[df["Decade"].eq("Unknowns"), "Decade"] = "Unknown"

def summarise(group_col, min_n=10):
    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("sample", "count"),
            imi_consensus_n=(f"imipenem_consensus_gt{THR}", "sum"),
            imi_consensus_prop=(f"imipenem_consensus_gt{THR}", "mean"),
            mer_consensus_n=(f"meropenem_consensus_gt{THR}", "sum"),
            mer_consensus_prop=(f"meropenem_consensus_gt{THR}", "mean"),
            any_consensus_n=(f"any_consensus_gt{THR}", "sum"),
            any_consensus_prop=(f"any_consensus_gt{THR}", "mean"),
        )
        .reset_index()
        .sort_values(["any_consensus_prop", "n"], ascending=[False, False])
    )
    out_n = out[out["n"] >= min_n].copy()
    return out, out_n

for group in ["Country_clean", "Continent_inferred", "Decade", "Year_num"]:
    out, out_n = summarise(group, min_n=10)
    safe = group.lower().replace("_clean", "").replace("_num", "")
    out.to_csv(OUTDIR / f"global_prediction_consensus_by_{safe}.tsv", sep="\t", index=False)
    out_n.to_csv(OUTDIR / f"global_prediction_consensus_by_{safe}_n10.tsv", sep="\t", index=False)

# Time trend table: years only, n>=10.
year = pd.read_csv(OUTDIR / "global_prediction_consensus_by_year.tsv", sep="\t")
year_n10 = year[pd.to_numeric(year["Year_num"], errors="coerce").notna() & (year["n"] >= 10)].copy()
year_n10 = year_n10.sort_values("Year_num")
year_n10.to_csv(OUTDIR / "global_prediction_consensus_time_trend_year_n10.tsv", sep="\t", index=False)

print("[OK] wrote geography/time summaries")
print("\n## Continent n>=10")
print(pd.read_csv(OUTDIR / "global_prediction_consensus_by_continent_inferred_n10.tsv", sep="\t").to_string(index=False))
print("\n## Decade")
print(pd.read_csv(OUTDIR / "global_prediction_consensus_by_decade_n10.tsv", sep="\t").to_string(index=False))
print("\n## Year trend n>=10")
print(year_n10.to_string(index=False))
