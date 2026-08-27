#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd

ERROR_FP = Path("outputs/error_characterisation/validation_prediction_error_table.tsv")
MICRO_FP = Path("data/FullMicroreactWyr-with-Russian-Metadata_IC_inferred.csv")

OUTDIR = Path("outputs/error_characterisation")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_ENRICHED = OUTDIR / "validation_prediction_error_table.with_microreact_ic.tsv"
OUT_ST_PASTEUR_MAP = OUTDIR / "microreact_ST_Pasteur_to_IC_reference.tsv"
OUT_ST_OXFORD_MAP = OUTDIR / "microreact_ST_Oxford_to_IC_reference.tsv"
OUT_SUMMARY = OUTDIR / "validation_error_by_microreact_ic.tsv"

MIN_ST_SUPPORT_N = 5
MIN_ST_SUPPORT_PROP = 0.80


def norm_id(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    m = re.match(r"^(\d+)", x)
    return m.group(1) if m else x


def clean_st(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return np.nan
    x = re.sub(r"^ST", "", x, flags=re.I)
    # keep compound STs as-is, e.g. "208;1806"
    return x


def clean_ic(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "ua", "unassigned"}:
        return np.nan
    return x


def build_st_to_ic_map(micro, st_col, out_fp):
    tmp = micro[[st_col, "IC"]].copy()
    tmp[st_col] = tmp[st_col].map(clean_st)
    tmp["IC"] = tmp["IC"].map(clean_ic)
    tmp = tmp[tmp[st_col].notna() & tmp["IC"].notna()].copy()

    counts = (
        tmp.groupby([st_col, "IC"])
        .size()
        .reset_index(name="n_ic")
    )

    totals = (
        tmp.groupby(st_col)
        .size()
        .reset_index(name="n_total")
    )

    out = counts.merge(totals, on=st_col, how="left")
    out["prop"] = out["n_ic"] / out["n_total"]

    out = out.sort_values([st_col, "prop", "n_ic"], ascending=[True, False, False])
    best = out.groupby(st_col, as_index=False).head(1).copy()

    best["accepted_reference_mapping"] = (
        (best["n_total"] >= MIN_ST_SUPPORT_N)
        & (best["prop"] >= MIN_ST_SUPPORT_PROP)
    )

    best.to_csv(out_fp, sep="\t", index=False)
    return best


def main():
    err = pd.read_csv(ERROR_FP, sep="\t", dtype=str)
    micro = pd.read_csv(MICRO_FP, dtype=str, low_memory=False)

    # Make numeric IDs for exact overlap checks.
    err["numeric_id"] = err["sample"].map(norm_id)

    for c in ["id", "alternative id", "pubMLSTid", "Match id"]:
        if c in micro.columns:
            micro[f"{c}__numeric"] = micro[c].map(norm_id)

    micro["IC"] = micro["IC"].map(clean_ic)

    keep_cols = [
        "id",
        "alternative id",
        "pubMLSTid",
        "BAPS",
        "BAPS group",
        "BAPS_group",
        "lineage",
        "sublineage",
        "sublineage (Ribosomal MLST)",
        "lineage (Ribosomal MLST)",
        "Match id",
        "IC",
        "ST_Pasteur",
        "ST_Oxford",
        "hBAPS",
        "blaOXA51",
        "K",
        "OLC",
        "Country",
        "Country2",
        "City",
        "Year",
    ]
    keep_cols += [c for c in micro.columns if re.match(r"^(OXA|NDM|VIM|IMP|KPC|GES)-", c)]
    keep_cols = [c for c in keep_cols if c in micro.columns]

    # Long exact ID map across all plausible ID columns.
    exact_maps = []
    for c in ["id", "alternative id", "pubMLSTid", "Match id"]:
        nc = f"{c}__numeric"
        if nc not in micro.columns:
            continue
        sub = micro[[nc] + keep_cols].copy()
        sub = sub.rename(columns={nc: "numeric_id"})
        sub["microreact_exact_match_column"] = c
        exact_maps.append(sub)

    exact = pd.concat(exact_maps, ignore_index=True)
    exact = exact[exact["numeric_id"].notna()].drop_duplicates("numeric_id")

    enriched = err.merge(
        exact,
        on="numeric_id",
        how="left",
        suffixes=("", "_microreact"),
    )

    # Build reference ST->IC mappings from Microreact table.
    pasteur_map = build_st_to_ic_map(
        micro,
        "ST_Pasteur",
        OUT_ST_PASTEUR_MAP,
    ) if "ST_Pasteur" in micro.columns else pd.DataFrame()

    oxford_map = build_st_to_ic_map(
        micro,
        "ST_Oxford",
        OUT_ST_OXFORD_MAP,
    ) if "ST_Oxford" in micro.columns else pd.DataFrame()

    # Apply ST-reference IC where exact IC is absent.
    enriched["ST_Pasteur_clean"] = enriched.get("ST (MLST (Pasteur))", pd.Series(index=enriched.index, dtype=str)).map(clean_st)
    enriched["ST_Oxford_clean"] = enriched.get("ST (MLST (Oxford))", pd.Series(index=enriched.index, dtype=str)).map(clean_st)

    if not pasteur_map.empty:
        pm = pasteur_map[pasteur_map["accepted_reference_mapping"]].copy()
        pm = pm.rename(columns={
            "ST_Pasteur": "ST_Pasteur_clean",
            "IC": "IC_from_microreact_ST_Pasteur",
            "n_total": "IC_from_microreact_ST_Pasteur_n_total",
            "prop": "IC_from_microreact_ST_Pasteur_prop",
        })
        enriched = enriched.merge(
            pm[[
                "ST_Pasteur_clean",
                "IC_from_microreact_ST_Pasteur",
                "IC_from_microreact_ST_Pasteur_n_total",
                "IC_from_microreact_ST_Pasteur_prop",
            ]],
            on="ST_Pasteur_clean",
            how="left",
        )

    if not oxford_map.empty:
        om = oxford_map[oxford_map["accepted_reference_mapping"]].copy()
        om = om.rename(columns={
            "ST_Oxford": "ST_Oxford_clean",
            "IC": "IC_from_microreact_ST_Oxford",
            "n_total": "IC_from_microreact_ST_Oxford_n_total",
            "prop": "IC_from_microreact_ST_Oxford_prop",
        })
        enriched = enriched.merge(
            om[[
                "ST_Oxford_clean",
                "IC_from_microreact_ST_Oxford",
                "IC_from_microreact_ST_Oxford_n_total",
                "IC_from_microreact_ST_Oxford_prop",
            ]],
            on="ST_Oxford_clean",
            how="left",
        )

    # Final IC call with provenance.
    exact_ic = enriched["IC"] if "IC" in enriched.columns else pd.Series(np.nan, index=enriched.index)

    enriched["IC_microreact_or_reference"] = exact_ic
    enriched["IC_microreact_assignment_source"] = np.where(
        enriched["IC_microreact_or_reference"].notna(),
        "exact_microreact_id_match",
        np.nan,
    )

    # Fill from Pasteur ST map first, then Oxford ST map.
    if "IC_from_microreact_ST_Pasteur" in enriched.columns:
        mask = enriched["IC_microreact_or_reference"].isna() & enriched["IC_from_microreact_ST_Pasteur"].notna()
        enriched.loc[mask, "IC_microreact_or_reference"] = enriched.loc[mask, "IC_from_microreact_ST_Pasteur"]
        enriched.loc[mask, "IC_microreact_assignment_source"] = "reference_ST_Pasteur_to_IC"

    if "IC_from_microreact_ST_Oxford" in enriched.columns:
        mask = enriched["IC_microreact_or_reference"].isna() & enriched["IC_from_microreact_ST_Oxford"].notna()
        enriched.loc[mask, "IC_microreact_or_reference"] = enriched.loc[mask, "IC_from_microreact_ST_Oxford"]
        enriched.loc[mask, "IC_microreact_assignment_source"] = "reference_ST_Oxford_to_IC"

    enriched.to_csv(OUT_ENRICHED, sep="\t", index=False)

    # Summary by IC, model, antibiotic.
    metric_df = enriched.copy()
    for c in ["true_mic", "abs_log2_error", "within_1_dilution", "within_2_dilution", "log2_error"]:
        if c in metric_df.columns:
            metric_df[c] = pd.to_numeric(metric_df[c], errors="coerce")

    metric_df = metric_df[metric_df["true_mic"].notna()].copy()

    if "IC_microreact_or_reference" in metric_df.columns:
        summary = (
            metric_df.groupby(
                ["IC_microreact_or_reference", "IC_microreact_assignment_source", "model", "antibiotic"],
                dropna=False,
            )
            .agg(
                n=("sample", "count"),
                mae_log2=("abs_log2_error", "mean"),
                median_abs_log2_error=("abs_log2_error", "median"),
                within_1=("within_1_dilution", "mean"),
                within_2=("within_2_dilution", "mean"),
                mean_signed_error=("log2_error", "mean"),
            )
            .reset_index()
            .sort_values(["antibiotic", "model", "mae_log2"], ascending=[True, True, False])
        )

        summary.to_csv(OUT_SUMMARY, sep="\t", index=False)

    print("[OK] wrote", OUT_ENRICHED)
    print("[OK] wrote", OUT_ST_PASTEUR_MAP)
    print("[OK] wrote", OUT_ST_OXFORD_MAP)
    print("[OK] wrote", OUT_SUMMARY)

    print()
    print("Rows:", len(enriched))
    print("Exact Microreact ID matches:", enriched["microreact_exact_match_column"].notna().sum() if "microreact_exact_match_column" in enriched.columns else 0)

    print()
    print("IC assignment source counts:")
    print(enriched["IC_microreact_assignment_source"].value_counts(dropna=False).to_string())

    print()
    print("IC counts:")
    print(enriched["IC_microreact_or_reference"].value_counts(dropna=False).head(30).to_string())


if __name__ == "__main__":
    main()
