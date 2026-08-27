#!/usr/bin/env python3

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor


BASE = Path(".")
OUTDIR = Path("outputs/tuning_locked_xgb")
PARAM_DIR = OUTDIR / "tuned_params"
CV_DIR = OUTDIR / "cv_results"
PARAM_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)

MIC_FILE = Path("data/mic_values.norm.csv")
ID_COL = "sample"
ANTIBIOTICS = ["imipenem", "meropenem"]

RUNS = {
    "01_amr_only_xgb": {
        "feature_tables": [
            ("outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv", "AMR")
        ]
    },
    "02_locus_gwas_xgb": {
        "feature_tables": [
            ("outputs/figure3_model_comparison/inputs/gwas_features_russia280_locus_presence_absence.sample.tsv", "LOCUS")
        ]
    },
    "03_hybrid_xgb": {
        "feature_tables": [
            ("outputs/figure3_model_comparison/inputs/amrfinder_selected_union.tsv", "AMRSEL"),
            ("outputs/figure3_model_comparison/inputs/gwas_features_russia280_locus_presence_absence.sample.tsv", "LOCUS")
        ]
    },
}


PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1, 3, 10],
}

N_RANDOM = 80
RANDOM_STATE = 1
N_SPLITS = 5


def read_table(path):
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def load_feature_table(path, prefix):
    df = read_table(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{path} missing ID column: {ID_COL}")

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str)

    feature_cols = [c for c in df.columns if c != ID_COL]
    renamed = {c: f"{prefix}__{c}" for c in feature_cols}
    df = df.rename(columns=renamed)

    for c in renamed.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def load_features(feature_tables):
    merged = None

    for path, prefix in feature_tables:
        df = load_feature_table(path, prefix)
        merged = df if merged is None else merged.merge(df, on=ID_COL, how="outer")

    feature_cols = [c for c in merged.columns if c != ID_COL]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return merged


def load_xy(feature_tables, antibiotic):
    Xdf = load_features(feature_tables)

    mic = pd.read_csv(MIC_FILE)
    mic[ID_COL] = mic[ID_COL].astype(str)

    if antibiotic not in mic.columns:
        raise ValueError(f"{MIC_FILE} missing antibiotic column: {antibiotic}")

    df = Xdf.merge(mic[[ID_COL, antibiotic]], on=ID_COL, how="inner")
    df[antibiotic] = pd.to_numeric(df[antibiotic], errors="coerce")
    df = df.dropna(subset=[antibiotic]).copy()

    feature_cols = [c for c in df.columns if c not in [ID_COL, antibiotic]]
    X = df[feature_cols].astype(float)
    y = np.log2(df[antibiotic].astype(float))

    return X, y, feature_cols


def sample_param_grid():
    keys = list(PARAM_GRID)
    all_combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    rng = np.random.default_rng(RANDOM_STATE)
    if len(all_combos) > N_RANDOM:
        idx = rng.choice(len(all_combos), size=N_RANDOM, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    for combo in combos:
        yield dict(zip(keys, combo))


def evaluate_params(X, y, params):
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **params,
    )

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
    mae_scores = -scores

    return {
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_sd": float(np.std(mae_scores)),
    }


def main():
    for run_name, spec in RUNS.items():
        print(f"\n## {run_name}")

        run_param_dir = PARAM_DIR / run_name
        run_cv_dir = CV_DIR / run_name
        run_param_dir.mkdir(parents=True, exist_ok=True)
        run_cv_dir.mkdir(parents=True, exist_ok=True)

        for antibiotic in ANTIBIOTICS:
            print(f"  - tuning {antibiotic}")

            X, y, feature_cols = load_xy(spec["feature_tables"], antibiotic)

            rows = []
            best = None

            for i, params in enumerate(sample_param_grid(), start=1):
                metrics = evaluate_params(X, y, params)
                row = {
                    "run_name": run_name,
                    "antibiotic": antibiotic,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "candidate": i,
                    **params,
                    **metrics,
                }
                rows.append(row)

                if best is None or row["cv_mae_mean"] < best["cv_mae_mean"]:
                    best = row

                if i % 10 == 0:
                    print(f"    candidate {i}: current best MAE={best['cv_mae_mean']:.4f}")

            cv = pd.DataFrame(rows).sort_values("cv_mae_mean")
            cv_path = run_cv_dir / f"{antibiotic}_cv_results.tsv"
            cv.to_csv(cv_path, sep="\t", index=False)

            best_params = {
                k: best[k]
                for k in PARAM_GRID
            }

            param_path = run_param_dir / f"{antibiotic}_params.json"
            with open(param_path, "w") as f:
                json.dump(best_params, f, indent=2)

            print(f"    best CV MAE: {best['cv_mae_mean']:.4f}")
            print(f"    wrote {param_path}")

    print("\n[OK] tuning complete")


if __name__ == "__main__":
    main()
