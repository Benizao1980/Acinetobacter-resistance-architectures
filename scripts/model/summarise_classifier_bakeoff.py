from pathlib import Path
import pandas as pd
import re

base = Path("outputs/runs_classifier_bakeoff")
outdir = Path("outputs/figure3_model_comparison/tables")
outdir.mkdir(parents=True, exist_ok=True)

summary_files = sorted(base.glob("*/all_bootstrap_summary.tsv"))

if not summary_files:
    raise SystemExit(
        f"No all_bootstrap_summary.tsv files found under {base}. "
        "Check whether the classifier bake-off job completed and where --plot-dir was set."
    )

frames = []

for fp in summary_files:
    run = fp.parent.name
    df = pd.read_csv(fp, sep="\t")
    df.insert(0, "run_name", run)

    # Try to infer classifier from run name if not already present
    # Expected examples: xgb_amr, rf_locus, ridge_amrsel_plus_locus, etc.
    classifier = None
    for clf in ["xgb", "rf", "ridge", "lasso", "elasticnet", "svr", "knn"]:
        if re.search(rf"(^|[_-]){clf}([_-]|$)", run):
            classifier = clf
            break

    if "classifier" not in df.columns:
        df.insert(1, "classifier", classifier if classifier else "unknown")

    frames.append(df)

long = pd.concat(frames, ignore_index=True)
long.to_csv(outdir / "classifier_bakeoff_long.tsv", sep="\t", index=False)

wide = (
    long.pivot_table(
        index=["run_name", "classifier", "antibiotic", "feature_set"],
        columns="metric",
        values="mean",
        aggfunc="first"
    )
    .reset_index()
)

wanted = [
    "run_name", "classifier", "antibiotic", "feature_set",
    "mae", "rmse", "r2", "within_1_dilution", "within_2_dilution"
]
wide = wide[[c for c in wanted if c in wide.columns]]
wide = wide.sort_values(["antibiotic", "mae"])

wide.to_csv(outdir / "classifier_bakeoff_wide.tsv", sep="\t", index=False)

print(wide.to_string(index=False))
print()
print(f"[OK] wrote {outdir / 'classifier_bakeoff_long.tsv'}")
print(f"[OK] wrote {outdir / 'classifier_bakeoff_wide.tsv'}")
