#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd

PRED_LONG = Path("outputs/global_prediction/global_locked_xgb_tuned_predictions_long.tsv")
PRED_WIDE = Path("outputs/global_prediction/global_locked_xgb_tuned_predictions_wide.tsv")
AMR_FP = Path("outputs/amrfinder/Global_dataset/amr_presence_absence.tsv")
META_FP = Path("data/FullMicroreactWyr-with-Russian-Metadata_IC_inferred.csv")

OUTDIR = Path("outputs/global_prediction/stratified")
OUTDIR.mkdir(parents=True, exist_ok=True)

ANTIBIOTICS = ["imipenem", "meropenem"]
MODELS = ["amr", "locus", "hybrid"]

THRESHOLDS = [2, 8, 16, 32]


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
    if x == "" or x.lower() in {"nan", "na", "none"}:
        return np.nan
    x = re.sub(r"^ST", "", x, flags=re.I)
    return x


def clean_ic(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "ua", "unassigned"}:
        return np.nan
    return x


def classify_amrfinder_mechanism(row):
    genes = []
    for c in row.index:
        if not str(c).startswith("GENE:bla"):
            continue
        try:
            if float(row[c]) > 0:
                genes.append(c)
        except Exception:
            pass

    text = " ; ".join(genes)

    if re.search(r"NDM|VIM|IMP", text, re.I):
        return "MBL"
    if re.search(r"KPC", text, re.I):
        return "KPC-like"
    if re.search(r"GES", text, re.I):
        return "GES-like"
    if re.search(r"OXA-23", text, re.I):
        return "OXA-23-like"
    if re.search(r"OXA-24|OXA-40|OXA-72|OXA-143", text, re.I):
        return "OXA-24/40-like"
    if re.search(r"OXA-58", text, re.I):
        return "OXA-58-like"
    if re.search(r"OXA-235", text, re.I):
        return "OXA-235-like"
    if re.search(r"OXA-51|OXA-64|OXA-65|OXA-66|OXA-68|OXA-69|OXA-70|OXA-71|OXA-82|OXA-90|OXA-91|OXA-94|OXA-95|OXA-98|OXA-100|OXA-104|OXA-106|OXA-109|OXA-132|OXA-208|OXA-317|OXA-523|OXA-528|OXA-531|OXA-532|OXA-855|OXA-1093|OXA-1241", text, re.I):
        return "Intrinsic OXA-51-like only/other"
    if text.strip():
        return "Other beta-lactamase"
    return "No detected carbapenemase"


def load_metadata():
    if not META_FP.exists():
        print(f"[WARN] Missing metadata file: {META_FP}")
        return pd.DataFrame(columns=["sample", "numeric_id"])

    meta = pd.read_csv(META_FP, dtype=str, low_memory=False)

    # Build sample/numeric ID. Global prediction sample names appear to be the first AMRFinder matrix column.
    id_candidates = ["id", "alternative id", "pubMLSTid", "Match id"]
    for c in id_candidates:
        if c in meta.columns:
            meta[f"{c}__numeric"] = meta[c].map(norm_id)

    keep = [
        "id",
        "alternative id",
        "pubMLSTid",
        "Match id",
        "IC",
        "ST_Pasteur",
        "ST_Oxford",
        "hBAPS",
        "Country",
        "Country2",
        "City",
        "Year",
        "SiteofInfection",
        "Specimen",
        "K",
        "OLC",
        "blaOXA51",
    ]
    keep = [c for c in keep if c in meta.columns]

    meta = meta[keep + [c for c in meta.columns if c.endswith("__numeric")]].copy()

    # We will later merge using several candidate ID columns.
    meta["IC"] = meta["IC"].map(clean_ic) if "IC" in meta.columns else np.nan
    if "ST_Pasteur" in meta.columns:
        meta["ST_Pasteur"] = meta["ST_Pasteur"].map(clean_st)
    if "ST_Oxford" in meta.columns:
        meta["ST_Oxford"] = meta["ST_Oxford"].map(clean_st)

    return meta


def merge_metadata(wide):
    meta = load_metadata()
    wide = wide.copy()
    wide["numeric_id"] = wide["sample"].map(norm_id)

    if meta.empty:
        wide["_metadata_match_type"] = np.nan
        return wide

    id_cols = [c for c in ["id", "alternative id", "pubMLSTid", "Match id"] if c in meta.columns]

    # Build a long-form metadata crosswalk using both exact IDs and numeric IDs.
    records = []

    for _, row in meta.iterrows():
        base = row.to_dict()

        for c in id_cols:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() != "":
                rec = base.copy()
                rec["_join_key"] = str(val).strip()
                rec["_metadata_match_type"] = f"direct_{c}"
                records.append(rec)

            num_col = f"{c}__numeric"
            val_num = row.get(num_col)
            if pd.notna(val_num) and str(val_num).strip() != "":
                rec = base.copy()
                rec["_join_key"] = str(val_num).strip()
                rec["_metadata_match_type"] = f"numeric_{num_col}"
                records.append(rec)

    cross = pd.DataFrame(records)

    if cross.empty:
        wide["_metadata_match_type"] = np.nan
        return wide

    # Prefer direct matches over numeric matches, and rows with IC/hBAPS.
    cross["_rank"] = 10
    cross.loc[cross["_metadata_match_type"].str.startswith("direct_", na=False), "_rank"] = 0
    cross.loc[cross["IC"].notna() if "IC" in cross.columns else False, "_rank"] -= 1
    cross.loc[cross["hBAPS"].notna() if "hBAPS" in cross.columns else False, "_rank"] -= 1

    cross = (
        cross.sort_values(["_join_key", "_rank"])
        .drop_duplicates("_join_key", keep="first")
        .copy()
    )

    # First try exact sample string.
    m1 = wide.merge(
        cross,
        left_on="sample",
        right_on="_join_key",
        how="left",
        suffixes=("", "_meta")
    )

    matched = m1["_metadata_match_type"].notna()

    # Then try numeric ID for still-unmatched rows.
    need = m1.loc[~matched, wide.columns].copy()
    if len(need):
        m2 = need.merge(
            cross,
            left_on="numeric_id",
            right_on="_join_key",
            how="left",
            suffixes=("", "_meta")
        )

        # Recombine.
        m1.loc[~matched, :] = m2.reindex(columns=m1.columns).values

    return m1

def load_amrfinder():
    amr = pd.read_csv(AMR_FP, sep="\t")
    if "sample" not in amr.columns:
        amr = amr.rename(columns={amr.columns[0]: "sample"})
    amr["sample"] = amr["sample"].astype(str)
    amr["numeric_id"] = amr["sample"].map(norm_id)

    gene_cols = [c for c in amr.columns if str(c).startswith("GENE:bla")]
    amr["amrfinder_carbapenemase_group"] = amr[gene_cols].apply(classify_amrfinder_mechanism, axis=1)

    keep = ["sample", "numeric_id", "amrfinder_carbapenemase_group"] + gene_cols
    return amr[keep].copy()


def add_threshold_flags(df):
    for model in MODELS:
        for ab in ANTIBIOTICS:
            col = f"{model}_{ab}_pred_mic"
            if col not in df.columns:
                continue
            for t in THRESHOLDS:
                df[f"{model}_{ab}_gt{t}"] = pd.to_numeric(df[col], errors="coerce") > t
    return df


def summarise_long(enriched):
    rows = []

    strata = [
        "IC",
        "hBAPS",
        "ST_Pasteur",
        "ST_Oxford",
        "Country",
        "Country2",
        "Year",
        "amrfinder_carbapenemase_group",
    ]

    for stratum in strata:
        if stratum not in enriched.columns:
            continue

        for model in MODELS:
            for ab in ANTIBIOTICS:
                mic_col = f"{model}_{ab}_pred_mic"
                if mic_col not in enriched.columns:
                    continue

                tmp = enriched[["sample", stratum, mic_col]].copy()
                tmp[mic_col] = pd.to_numeric(tmp[mic_col], errors="coerce")
                tmp = tmp[tmp[mic_col].notna()].copy()

                if tmp.empty:
                    continue

                grouped = (
                    tmp.groupby(stratum, dropna=False)
                    .agg(
                        n=("sample", "count"),
                        median_mic=(mic_col, "median"),
                        mean_mic=(mic_col, "mean"),
                        q25=(mic_col, lambda x: np.percentile(x, 25)),
                        q75=(mic_col, lambda x: np.percentile(x, 75)),
                        p90=(mic_col, lambda x: np.percentile(x, 90)),
                        p95=(mic_col, lambda x: np.percentile(x, 95)),
                        max_mic=(mic_col, "max"),
                        n_gt2=(mic_col, lambda x: (x > 2).sum()),
                        prop_gt2=(mic_col, lambda x: (x > 2).mean()),
                        n_gt8=(mic_col, lambda x: (x > 8).sum()),
                        prop_gt8=(mic_col, lambda x: (x > 8).mean()),
                        n_gt16=(mic_col, lambda x: (x > 16).sum()),
                        prop_gt16=(mic_col, lambda x: (x > 16).mean()),
                        n_gt32=(mic_col, lambda x: (x > 32).sum()),
                        prop_gt32=(mic_col, lambda x: (x > 32).mean()),
                    )
                    .reset_index()
                    .rename(columns={stratum: "group"})
                )

                grouped.insert(0, "stratum", stratum)
                grouped.insert(1, "model", model)
                grouped.insert(2, "antibiotic", ab)
                rows.append(grouped)

    out = pd.concat(rows, ignore_index=True)
    return out


def main():
    wide = pd.read_csv(PRED_WIDE, sep="\t")
    wide["sample"] = wide["sample"].astype(str)

    enriched = merge_metadata(wide)

    amr = load_amrfinder()
    enriched = enriched.merge(
        amr,
        on=["sample", "numeric_id"],
        how="left",
        suffixes=("", "_amrfinder"),
    )

    enriched = add_threshold_flags(enriched)

    enriched_fp = OUTDIR / "global_predictions_enriched.tsv"
    enriched.to_csv(enriched_fp, sep="\t", index=False)

    summary = summarise_long(enriched)
    summary_fp = OUTDIR / "global_predictions_stratified_summary.tsv"
    summary.to_csv(summary_fp, sep="\t", index=False)

    # Compact supported summaries for likely manuscript use.
    compact = summary[summary["n"] >= 10].copy()
    compact = compact.sort_values(
        ["stratum", "antibiotic", "model", "prop_gt16", "median_mic"],
        ascending=[True, True, True, False, False],
    )
    compact_fp = OUTDIR / "global_predictions_stratified_summary_n10.tsv"
    compact.to_csv(compact_fp, sep="\t", index=False)

    # Consensus table: how often models agree on >2, >8, >16, >32 per drug.
    consensus = enriched[["sample"]].copy()
    for ab in ANTIBIOTICS:
        for t in THRESHOLDS:
            cols = [f"{m}_{ab}_gt{t}" for m in MODELS if f"{m}_{ab}_gt{t}" in enriched.columns]
            consensus[f"{ab}_n_models_gt{t}"] = enriched[cols].sum(axis=1)
            consensus[f"{ab}_consensus_gt{t}"] = consensus[f"{ab}_n_models_gt{t}"] >= 2

    consensus_fp = OUTDIR / "global_predictions_consensus_flags.tsv"
    consensus.to_csv(consensus_fp, sep="\t", index=False)

    print("[OK] wrote", enriched_fp)
    print("[OK] wrote", summary_fp)
    print("[OK] wrote", compact_fp)
    print("[OK] wrote", consensus_fp)

    print("\nMetadata match types:")
    if "_metadata_match_type" in enriched.columns:
        print(enriched["_metadata_match_type"].value_counts(dropna=False).to_string())

    print("\nAMRFinder mechanism counts:")
    print(enriched["amrfinder_carbapenemase_group"].value_counts(dropna=False).to_string())

    for col in ["IC", "hBAPS", "ST_Pasteur", "Country", "Year"]:
        if col in enriched.columns:
            print(f"\nTop {col}:")
            print(enriched[col].value_counts(dropna=False).head(20).to_string())

    print("\nOverall consensus counts:")
    for ab in ANTIBIOTICS:
        for t in THRESHOLDS:
            c = f"{ab}_consensus_gt{t}"
            if c in consensus.columns:
                print(f"{c}: {int(consensus[c].sum())}/{len(consensus)}")


if __name__ == "__main__":
    main()
