#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd

OUTDIR = Path("outputs/error_characterisation")
OUTDIR.mkdir(parents=True, exist_ok=True)

PRED_ROOT = Path("preds/validation_locked_xgb_tuned_evalnames")
TRUTH_FP = Path("data/phenotypes_validation/validation_dataset_MIC.cleaned.tsv")
AMR_FP = Path("outputs/amrfinder/validation/amr_presence_absence.tsv")
IC_FP = Path("outputs/error_characterisation/validation_metadata_with_icassigner.tsv")
RICH_FP = Path("outputs/error_characterisation/validation_metadata_for_error_characterisation.tsv")

MODELS = ["amr", "locus", "hybrid"]
ANTIBIOTICS = ["imipenem", "meropenem"]


def read_table(fp):
    return pd.read_csv(fp, sep="\t")


def normalise_sample_col(df):
    if "sample" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample"})
    df["sample"] = df["sample"].astype(str)
    return df


def numeric_id_from_sample(x):
    m = re.match(r"^(\d+)", str(x))
    return m.group(1) if m else str(x)


def load_truth():
    truth = read_table(TRUTH_FP)
    truth = truth.rename(columns={"id": "sample"})
    truth["sample"] = truth["sample"].astype(str)
    truth["numeric_id"] = truth["sample"].map(numeric_id_from_sample)

    keep = ["sample", "numeric_id", "country", "year", "resistance_profile"] + [
        a for a in ANTIBIOTICS if a in truth.columns
    ]

    extra = [
        "ST (MLST (Oxford))",
        "species (MLST (Oxford))",
        "ST (MLST (Pasteur))",
        "species (MLST (Pasteur))",
        "bioproject_accession",
        "biosample_accession",
        "run_accession",
    ]
    keep += [c for c in extra if c in truth.columns]

    return truth[keep].copy()


def load_amrfinder():
    if not AMR_FP.exists():
        return pd.DataFrame(columns=["sample", "numeric_id"])

    amr = read_table(AMR_FP)
    amr = normalise_sample_col(amr)
    amr["numeric_id"] = amr["sample"].map(numeric_id_from_sample)

    # Keep carbapenemase/beta-lactamase columns only, to avoid an unwieldy table.
    gene_cols = [
        c for c in amr.columns
        if c.startswith("GENE:bla")
        or c in ["GENE:ble"]
    ]

    keep = ["sample", "numeric_id"] + gene_cols
    return amr[keep].copy()


def load_metadata():
    # Prefer ICassigner-enriched metadata if present.
    if IC_FP.exists():
        meta = read_table(IC_FP)
        if "sample_id" in meta.columns and "sample" not in meta.columns:
            meta = meta.rename(columns={"sample_id": "sample"})
        if "sample" not in meta.columns and "id" in meta.columns:
            meta = meta.rename(columns={"id": "sample"})
        meta["sample"] = meta["sample"].astype(str)
        meta["numeric_id"] = meta["sample"].map(numeric_id_from_sample)
        return meta

    if RICH_FP.exists():
        meta = read_table(RICH_FP)
        if "id" in meta.columns and "sample" not in meta.columns:
            meta = meta.rename(columns={"id": "sample"})
        if "sample" not in meta.columns:
            meta["sample"] = meta.iloc[:, 0].astype(str)
        meta["sample"] = meta["sample"].astype(str)
        if "numeric_id" not in meta.columns:
            meta["numeric_id"] = meta["sample"].map(numeric_id_from_sample)
        return meta

    return pd.DataFrame(columns=["sample", "numeric_id"])


def present(row, pattern):
    cols = [c for c in row.index if re.search(pattern, c, re.I)]
    for c in cols:
        try:
            if float(row[c]) > 0:
                return True
        except Exception:
            if str(row[c]).strip() not in ["", "0", "0.0", "nan", "NaN"]:
                return True
    return False


def classify_amrfinder_mechanism(row):
    if present(row, r"bla(NDM|VIM|IMP)"):
        return "MBL"
    if present(row, r"blaKPC"):
        return "KPC-like"
    if present(row, r"blaGES"):
        return "GES-like"
    if present(row, r"blaOXA-23\b"):
        return "OXA-23-like"
    if present(row, r"blaOXA-(24|40|72|143)\b"):
        return "OXA-24/40-like"
    if present(row, r"blaOXA-58\b"):
        return "OXA-58-like"
    if present(row, r"blaOXA-235\b"):
        return "OXA-235-like"

    # Intrinsic A. baumannii OXA-51 family only.
    if present(row, r"blaOXA-(51|64|65|66|68|69|71|82|90|91|94|95|98|100|104|106|109|113|120|121|126|132|208|312|317|523|525|528|529|531|532|855)\b"):
        return "Intrinsic OXA-51-like only/other"

    return "No detected carbapenemase"


def load_predictions():
    frames = []

    for model in MODELS:
        for ab in ANTIBIOTICS:
            fp = PRED_ROOT / model / f"{ab}.tsv"
            if not fp.exists():
                print(f"[MISSING] {fp}")
                continue

            pred = read_table(fp)
            pred = normalise_sample_col(pred)

            if "prediction" not in pred.columns:
                raise SystemExit(f"No 'prediction' column in {fp}: {list(pred.columns)}")

            sub = pred[["sample", "prediction"]].copy()
            sub = sub.rename(columns={"prediction": "pred_mic"})
            sub["model"] = model
            sub["antibiotic"] = ab
            sub["numeric_id"] = sub["sample"].map(numeric_id_from_sample)
            frames.append(sub)

    if not frames:
        raise SystemExit("No prediction files found.")

    return pd.concat(frames, ignore_index=True)


def main():
    truth = load_truth()
    amr = load_amrfinder()
    meta = load_metadata()
    pred = load_predictions()

    # Long truth table.
    truth_long = truth.melt(
        id_vars=[c for c in truth.columns if c not in ANTIBIOTICS],
        value_vars=[a for a in ANTIBIOTICS if a in truth.columns],
        var_name="antibiotic",
        value_name="true_mic",
    )

    merged = pred.merge(
        truth_long,
        on=["sample", "numeric_id", "antibiotic"],
        how="left"
    )

    # Add metadata, avoiding duplicate truth columns.
    if not meta.empty:
        drop_cols = [c for c in meta.columns if c in merged.columns and c not in ["sample", "numeric_id"]]
        meta2 = meta.drop(columns=drop_cols, errors="ignore")
        merged = merged.merge(meta2, on=["sample", "numeric_id"], how="left")

    # Add AMRFinder gene calls.
    if not amr.empty:
        amr2 = amr.drop(columns=[c for c in ["sample"] if c in amr.columns], errors="ignore")
        merged = merged.merge(amr2, on="numeric_id", how="left")

    gene_cols = [c for c in merged.columns if c.startswith("GENE:")]
    for c in gene_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    merged["amrfinder_carbapenemase_group"] = merged.apply(classify_amrfinder_mechanism, axis=1)

    merged["true_mic"] = pd.to_numeric(merged["true_mic"], errors="coerce")
    merged["pred_mic"] = pd.to_numeric(merged["pred_mic"], errors="coerce")

    merged["log2_true_mic"] = np.log2(merged["true_mic"])
    merged["log2_pred_mic"] = np.log2(merged["pred_mic"])
    merged["log2_error"] = merged["log2_pred_mic"] - merged["log2_true_mic"]
    merged["abs_log2_error"] = merged["log2_error"].abs()
    merged["within_1_dilution"] = merged["abs_log2_error"] <= 1
    merged["within_2_dilution"] = merged["abs_log2_error"] <= 2

    merged.loc[merged["true_mic"].isna(), ["within_1_dilution", "within_2_dilution"]] = np.nan

    merged["direction"] = np.select(
        [
            merged["log2_error"] > 0,
            merged["log2_error"] < 0,
            merged["log2_error"] == 0,
        ],
        [
            "over-predicted",
            "under-predicted",
            "exact",
        ],
        default=np.nan,
    )

    table_out = OUTDIR / "validation_prediction_error_table.tsv"
    merged.to_csv(table_out, sep="\t", index=False)

    summary = (
        merged.groupby(["model", "antibiotic"], dropna=False)
        .agg(
            n_pred=("sample", "count"),
            n_with_truth=("true_mic", lambda x: x.notna().sum()),
            mae_log2=("abs_log2_error", "mean"),
            median_abs_log2_error=("abs_log2_error", "median"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
        )
        .reset_index()
    )

    summary_out = OUTDIR / "validation_prediction_error_summary.tsv"
    summary.to_csv(summary_out, sep="\t", index=False)

    mech_summary = (
        merged[merged["true_mic"].notna()]
        .groupby(["amrfinder_carbapenemase_group", "model", "antibiotic"], dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
        )
        .reset_index()
        .sort_values(["antibiotic", "model", "mae_log2"], ascending=[True, True, False])
    )

    mech_out = OUTDIR / "validation_prediction_error_by_mechanism.tsv"
    mech_summary.to_csv(mech_out, sep="\t", index=False)

    print("[OK] wrote", table_out)
    print("[OK] wrote", summary_out)
    print("[OK] wrote", mech_out)
    print()
    print(summary.to_string(index=False))
    print()
    print("Mechanism counts:")
    print(merged.drop_duplicates("sample")["amrfinder_carbapenemase_group"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
