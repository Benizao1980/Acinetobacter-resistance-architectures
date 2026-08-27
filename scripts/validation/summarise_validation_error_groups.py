#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

IN = Path("outputs/error_characterisation/validation_prediction_error_table.tsv")
OUTDIR = Path("outputs/error_characterisation")
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_N = 10

df = pd.read_csv(IN, sep="\t")

# Keep only rows with truth
df = df[df["true_mic"].notna()].copy()

def summarise(group_cols, name, min_n=MIN_N):
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            median_abs_log2_error=("abs_log2_error", "median"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
            mean_signed_error=("log2_error", "mean"),
            n_overpredicted=("direction", lambda x: (x == "over-predicted").sum()),
            n_underpredicted=("direction", lambda x: (x == "under-predicted").sum()),
            n_exact=("direction", lambda x: (x == "exact").sum()),
        )
        .reset_index()
    )

    out = out[out["n"] >= min_n].copy()
    out = out.sort_values(
        ["antibiotic", "model", "mae_log2"],
        ascending=[True, True, False],
    )

    fp = OUTDIR / f"validation_error_by_{name}.tsv"
    out.to_csv(fp, sep="\t", index=False)
    print(f"\n## {name}")
    print(out.to_string(index=False))
    print("[OK] wrote", fp)


# 1. AMRfinder carbapenemase mechanism
summarise(
    ["amrfinder_carbapenemase_group", "model", "antibiotic"],
    "amrfinder_mechanism",
)

# 2. Pasteur ST
if "ST (MLST (Pasteur))" in df.columns:
    summarise(
        ["ST (MLST (Pasteur))", "model", "antibiotic"],
        "pasteur_st",
    )

# 3. Oxford ST
if "ST (MLST (Oxford))" in df.columns:
    summarise(
        ["ST (MLST (Oxford))", "model", "antibiotic"],
        "oxford_st",
    )

# 4. ICassigner / conservative IC if available
ic_cols = [
    c for c in [
        "IC_tree_conservative",
        "IC_seed",
        "lineage (Ribosomal MLST)",
        "sublineage (Ribosomal MLST)",
    ]
    if c in df.columns
]

for c in ic_cols:
    safe = (
        c.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    summarise([c, "model", "antibiotic"], safe)

# 5. Combined compact table: mechanism x antibiotic, only best/worst model patterns
compact_cols = [
    "amrfinder_carbapenemase_group",
    "model",
    "antibiotic",
    "n",
    "mae_log2",
    "within_1",
    "within_2",
    "mean_signed_error",
]

mech_fp = OUTDIR / "validation_error_by_amrfinder_mechanism.tsv"
if mech_fp.exists():
    mech = pd.read_csv(mech_fp, sep="\t")
    compact = mech[compact_cols].copy()
    compact.to_csv(
        OUTDIR / "validation_error_by_amrfinder_mechanism_compact.tsv",
        sep="\t",
        index=False,
    )
    print("\n[OK] wrote", OUTDIR / "validation_error_by_amrfinder_mechanism_compact.tsv")
