#!/bin/bash
#SBATCH --job-name=BAMPS-clf
#SBATCH --account=cooperma
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/BAMPS_classifier_bakeoff_%j.out
#SBATCH --error=slurm_logs/BAMPS_classifier_bakeoff_%j.err

echo "Starting classifier bake-off"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAMPY

cd ~/Acinetobacter/BAMPS_ML

mkdir -p slurm_logs
mkdir -p outputs/runs_classifier_bakeoff
mkdir -p outputs/models_classifier_bakeoff
mkdir -p outputs/figure3_model_comparison/tables

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

MIC_FILE="data/mic_values.norm.csv"
AMR_FULL="outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv"
LOCUS_FULL="outputs/figure3_model_comparison/inputs/gwas_features_russia280_locus_presence_absence.sample.tsv"
AMR_SELECTED="outputs/figure3_model_comparison/inputs/amrfinder_selected_union.tsv"

# This was the current best-performing biological feature set:
# selected AMR features + full/pruned locus-GWAS matrix
# We will also test AMR-only and locus-GWAS only for comparison.

echo "Checking inputs..."
for f in "$MIC_FILE" "$AMR_FULL" "$LOCUS_FULL" "$AMR_SELECTED"; do
    if [[ ! -s "$f" ]]; then
        echo "[ERROR] Missing or empty input: $f"
        exit 1
    fi
    echo "[OK] $f"
done

# ------------------------------------------------------------
# Classifiers
# ------------------------------------------------------------
# These are the classifiers we know are likely to be supported
# from your current scripts. Add lightgbm/catboost/extra_trees only
# if train_model.py --help lists them.
# ------------------------------------------------------------

CLASSIFIERS=("xgb" "rf" "ridge")

echo
echo "Available classifier help line:"
python scripts/train_model.py --help | grep -A2 -i classifier || true
echo

# ------------------------------------------------------------
# Function to run one model
# ------------------------------------------------------------

run_model () {
    local RUN_NAME="$1"
    local CLASSIFIER="$2"
    shift 2
    local FEATURE_ARGS=("$@")

    echo
    echo "============================================================"
    echo "Running: ${RUN_NAME}"
    echo "Classifier: ${CLASSIFIER}"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/train_model.py \
      "${FEATURE_ARGS[@]}" \
      --mic-file "$MIC_FILE" \
      --mic-id-col sample \
      --antibiotics imipenem meropenem \
      --task regression \
      --classifier "$CLASSIFIER" \
      --log2 \
      --self-test \
      --fit-full \
      --bootstrap-reps 200 \
      --n-jobs 4 \
      --model-dir outputs/models_classifier_bakeoff \
      --plot-dir outputs/runs_classifier_bakeoff \
      --run-name "$RUN_NAME"

    echo "Finished: $(date)"
}

# ------------------------------------------------------------
# Main bake-off
# ------------------------------------------------------------

for CLF in "${CLASSIFIERS[@]}"; do

    # 1. AMR-only baseline
    run_model "01_amr_only__${CLF}" "$CLF" \
      --feature-table "$AMR_FULL" \
      --feature-prefix AMR

    # 2. Locus-GWAS only
    run_model "02_locus_gwas__${CLF}" "$CLF" \
      --feature-table "$LOCUS_FULL" \
      --feature-prefix LOCUS

    # 3. Selected AMR + locus-GWAS hybrid
    run_model "03_amr_selected_plus_locus__${CLF}" "$CLF" \
      --feature-table "$AMR_SELECTED" "$LOCUS_FULL" \
      --feature-prefix AMRSEL LOCUS

done

# ------------------------------------------------------------
# Build combined summary table
# ------------------------------------------------------------

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("outputs/runs_classifier_bakeoff")
outdir = Path("outputs/figure3_model_comparison/tables")
outdir.mkdir(parents=True, exist_ok=True)

frames = []

for summary in sorted(base.glob("*/all_bootstrap_summary.tsv")):
    run_name = summary.parent.name

    parts = run_name.split("__")
    if len(parts) == 2:
        model_set, classifier = parts
    else:
        model_set, classifier = run_name, "unknown"

    df = pd.read_csv(summary, sep="\t")
    df.insert(0, "run_name", run_name)
    df.insert(1, "model_set", model_set)
    df.insert(2, "classifier", classifier)
    df.insert(3, "source_file", str(summary))
    frames.append(df)

if not frames:
    raise SystemExit("No classifier bake-off summaries found.")

long = pd.concat(frames, ignore_index=True)
long_path = outdir / "classifier_bakeoff_long.tsv"
long.to_csv(long_path, sep="\t", index=False)

wide = (
    long.pivot_table(
        index=["model_set", "classifier", "antibiotic", "feature_set"],
        columns="metric",
        values="mean",
        aggfunc="first"
    )
    .reset_index()
)

wanted = [
    "model_set", "classifier", "antibiotic", "feature_set",
    "mae", "rmse", "r2",
    "within_1_dilution", "within_2_dilution"
]
wide = wide[[c for c in wanted if c in wide.columns]]

wide = wide.sort_values(["antibiotic", "mae", "rmse"])
wide_path = outdir / "classifier_bakeoff_wide.tsv"
wide.to_csv(wide_path, sep="\t", index=False)

print("\n[CLASSIFIER BAKE-OFF SUMMARY]")
print(wide.to_string(index=False))

print("\n[OK] Wrote:")
print(long_path)
print(wide_path)