#!/usr/bin/env bash
set -euo pipefail

# BAMPS: AMRFinder feature selection for carbapenem MIC prediction
# Run from the BAMPS repository root after AMRFinder feature extraction is complete.

AMR_TABLE=${AMR_TABLE:-outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv}
MIC_FILE=${MIC_FILE:-data/mic_values.norm.csv}
OUTDIR=${OUTDIR:-outputs/feature_selection/amrfinder}
THREADS=${THREADS:-8}

mkdir -p "${OUTDIR}"

python scripts/select_predictive_features.py \
  --candidate-feature-table "${AMR_TABLE}" \
  --baseline-feature-table "${AMR_TABLE}" \
  --mic-file "${MIC_FILE}" \
  --antibiotics imipenem meropenem \
  --task regression \
  --classifier xgb \
  --log2 \
  --repeats 30 \
  --screen-repeats 10 \
  --greedy \
  --max-features 25 \
  --min-prevalence 0.02 \
  --id-col sample \
  --n-jobs "${THREADS}" \
  --outdir "${OUTDIR}"

printf '\n[OK] AMRFinder feature selection complete. Outputs written to: %s\n' "${OUTDIR}"
