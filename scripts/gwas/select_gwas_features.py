#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def log(msg: str) -> None:
    print(msg, flush=True)


def read_table(path: Path, id_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if id_col is None:
        id_col = df.columns[0]
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in {path}. Columns: {df.columns.tolist()[:20]}")

    df = df.rename(columns={id_col: "sample"})
    df["sample"] = df["sample"].astype(str)
    df = df.set_index("sample", drop=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def apply_prefix(df: pd.DataFrame, prefix: Optional[str]) -> pd.DataFrame:
    if prefix is None or str(prefix).strip() == "":
        return df
    out = df.copy()
    prefix = str(prefix).strip()
    out.columns = [
        f"{prefix}__{c}" if not str(c).startswith(f"{prefix}__") else str(c)
        for c in out.columns
    ]
    return out


def load_mic(path: Path, mic_id_col: str, antibiotic: str, log2_values: bool) -> pd.Series:
    ph = pd.read_csv(path, sep=None, engine="python", dtype=str)
    ph.columns = [str(c).replace("\ufeff", "").strip() for c in ph.columns]

    if mic_id_col not in ph.columns:
        raise ValueError(f"MIC ID column '{mic_id_col}' not found in {path}. Columns: {ph.columns.tolist()}")

    ab_col = None
    target = antibiotic.lower().replace("_mic", "")
    for c in ph.columns:
        cl = c.lower().replace("_mic", "")
        if cl == target or target in cl:
            ab_col = c
            break
    if ab_col is None:
        raise ValueError(f"Could not find antibiotic column for '{antibiotic}' in {path}. Columns: {ph.columns.tolist()}")

    y = pd.Series(
        pd.to_numeric(ph[ab_col], errors="coerce").values,
        index=ph[mic_id_col].astype(str),
        name=antibiotic,
    )
    y = y.dropna()
    y = y[y > 0]

    if log2_values:
        y = np.log2(y)

    return y


def make_model(classifier: str, random_state: int, n_jobs: int):
    classifier = classifier.lower()

    if classifier == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=2,
        )

    if classifier == "ridge":
        return Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0)),
        ])

    if classifier == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise RuntimeError("xgboost is not installed. Use --classifier rf or ridge.") from e

        return XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )

    raise ValueError(f"Unknown classifier: {classifier}")


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "within1": float(np.mean(np.abs(err) <= 1)),
        "within2": float(np.mean(np.abs(err) <= 2)),
        "bias": float(np.mean(err)),
    }


def eval_feature_set(
    X_amr: pd.DataFrame,
    X_gwas: Optional[pd.DataFrame],
    y: pd.Series,
    gwas_features: Sequence[str],
    classifier: str,
    repeats: int,
    test_size: float,
    random_state: int,
    n_jobs: int,
) -> pd.DataFrame:
    X_parts = [X_amr]
    if X_gwas is not None and len(gwas_features) > 0:
        X_parts.append(X_gwas.loc[:, list(gwas_features)])

    X = pd.concat(X_parts, axis=1)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    yy = y.loc[common]

    rows = []
    for i in range(repeats):
        seed = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            yy,
            test_size=test_size,
            random_state=seed,
        )

        model = make_model(classifier, random_state=seed, n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rows.append({"repeat": i, "seed": seed, "n_train": len(X_train), "n_test": len(X_test), **calc_metrics(y_test.values, pred)})

    return pd.DataFrame(rows)


def summarise_metrics(df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in ["mae", "rmse", "r2", "within1", "within2", "bias"]:
        out[f"{prefix}{metric}_mean"] = float(df[metric].mean())
        out[f"{prefix}{metric}_sd"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def screen_features(
    X_amr: pd.DataFrame,
    X_gwas: pd.DataFrame,
    y: pd.Series,
    baseline_summary: Dict[str, float],
    classifier: str,
    repeats: int,
    test_size: float,
    random_state: int,
    n_jobs: int,
    min_feature_prevalence: float,
    max_feature_prevalence: float,
) -> pd.DataFrame:
    rows = []
    features = list(X_gwas.columns)

    for idx, feat in enumerate(features, start=1):
        vals = X_gwas[feat]
        prev = float((vals > 0).mean())

        if prev < min_feature_prevalence or prev > max_feature_prevalence:
            rows.append({"feature": feat, "status": "skipped_prevalence", "prevalence": prev})
            continue

        if vals.nunique(dropna=False) <= 1:
            rows.append({"feature": feat, "status": "skipped_constant", "prevalence": prev})
            continue

        res = eval_feature_set(
            X_amr=X_amr,
            X_gwas=X_gwas,
            y=y,
            gwas_features=[feat],
            classifier=classifier,
            repeats=repeats,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        summ = summarise_metrics(res)

        row = {"feature": feat, "status": "tested", "prevalence": prev, **summ}

        row["delta_mae"] = baseline_summary["mae_mean"] - row["mae_mean"]
        row["delta_rmse"] = baseline_summary["rmse_mean"] - row["rmse_mean"]
        row["delta_r2"] = row["r2_mean"] - baseline_summary["r2_mean"]
        row["delta_within1"] = row["within1_mean"] - baseline_summary["within1_mean"]
        row["delta_within2"] = row["within2_mean"] - baseline_summary["within2_mean"]

        row["predictive_gain_score"] = (
            row["delta_rmse"]
            + 0.5 * row["delta_mae"]
            + 0.25 * row["delta_r2"]
            + 0.5 * row["delta_within1"]
        )

        rows.append(row)

        if idx % 10 == 0 or idx == len(features):
            log(f"[INFO] Screened {idx}/{len(features)} GWAS features")

    out = pd.DataFrame(rows)
    tested = out[out["status"] == "tested"].copy()
    skipped = out[out["status"] != "tested"].copy()

    if not tested.empty:
        tested = tested.sort_values(
            ["predictive_gain_score", "delta_rmse", "delta_within1"],
            ascending=[False, False, False],
            kind="stable",
        )

    return pd.concat([tested, skipped], ignore_index=True)


def greedy_forward_selection(
    X_amr: pd.DataFrame,
    X_gwas: pd.DataFrame,
    y: pd.Series,
    candidate_features: Sequence[str],
    baseline_summary: Dict[str, float],
    classifier: str,
    repeats: int,
    test_size: float,
    random_state: int,
    n_jobs: int,
    max_selected: int,
    min_delta_rmse: float,
    min_delta_within1: float,
    stop_on_no_gain: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    selected: List[str] = []
    remaining = list(candidate_features)
    path_rows = []
    current_summary = baseline_summary.copy()

    step = 0
    while remaining and len(selected) < max_selected:
        step += 1
        best_row = None

        for feat in remaining:
            trial_features = selected + [feat]
            res = eval_feature_set(
                X_amr=X_amr,
                X_gwas=X_gwas,
                y=y,
                gwas_features=trial_features,
                classifier=classifier,
                repeats=repeats,
                test_size=test_size,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            summ = summarise_metrics(res)
            row = {"step": step, "candidate_feature": feat, "n_selected_if_added": len(trial_features), **summ}

            row["delta_mae_vs_current"] = current_summary["mae_mean"] - row["mae_mean"]
            row["delta_rmse_vs_current"] = current_summary["rmse_mean"] - row["rmse_mean"]
            row["delta_r2_vs_current"] = row["r2_mean"] - current_summary["r2_mean"]
            row["delta_within1_vs_current"] = row["within1_mean"] - current_summary["within1_mean"]
            row["delta_within2_vs_current"] = row["within2_mean"] - current_summary["within2_mean"]

            row["delta_mae_vs_baseline"] = baseline_summary["mae_mean"] - row["mae_mean"]
            row["delta_rmse_vs_baseline"] = baseline_summary["rmse_mean"] - row["rmse_mean"]
            row["delta_r2_vs_baseline"] = row["r2_mean"] - baseline_summary["r2_mean"]
            row["delta_within1_vs_baseline"] = row["within1_mean"] - baseline_summary["within1_mean"]
            row["delta_within2_vs_baseline"] = row["within2_mean"] - baseline_summary["within2_mean"]

            row["greedy_score"] = (
                row["delta_rmse_vs_current"]
                + 0.5 * row["delta_mae_vs_current"]
                + 0.25 * row["delta_r2_vs_current"]
                + 0.5 * row["delta_within1_vs_current"]
            )

            if best_row is None or row["greedy_score"] > best_row["greedy_score"]:
                best_row = row

        if best_row is None:
            break

        accept = (
            best_row["delta_rmse_vs_current"] >= min_delta_rmse
            and best_row["delta_within1_vs_current"] >= min_delta_within1
        )

        best_row["accepted"] = bool(accept)
        best_row["selected_features_after_step"] = ";".join(selected + ([best_row["candidate_feature"]] if accept else []))
        path_rows.append(best_row)

        log(
            f"[GREEDY] step={step} best={best_row['candidate_feature']} "
            f"delta_RMSE={best_row['delta_rmse_vs_current']:.4f} "
            f"delta_within1={best_row['delta_within1_vs_current']:.4f} "
            f"accepted={accept}"
        )

        if accept:
            feat = best_row["candidate_feature"]
            selected.append(feat)
            remaining.remove(feat)
            for metric in ["mae", "rmse", "r2", "within1", "within2", "bias"]:
                current_summary[f"{metric}_mean"] = best_row[f"{metric}_mean"]
                current_summary[f"{metric}_sd"] = best_row[f"{metric}_sd"]
        else:
            if stop_on_no_gain:
                break
            remaining.remove(best_row["candidate_feature"])

    return pd.DataFrame(path_rows), selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select GWAS features by predictive gain on top of AMR features.")
    p.add_argument("--amr-table", required=True, type=Path)
    p.add_argument("--gwas-table", required=True, type=Path)
    p.add_argument("--mic-file", required=True, type=Path)
    p.add_argument("--mic-id-col", default="sample")
    p.add_argument("--antibiotic", required=True)

    p.add_argument("--id-col-amr", default=None)
    p.add_argument("--id-col-gwas", default=None)
    p.add_argument("--feature-prefix-amr", default="AMR")
    p.add_argument("--feature-prefix-gwas", default="GWAS")

    p.add_argument("--classifier", choices=["xgb", "rf", "ridge"], default="xgb")
    p.add_argument("--log2", action="store_true", help="Use log2 MIC values. Use this if MIC file is raw mg/L. If already log2, omit.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--screen-repeats", type=int, default=None)
    p.add_argument("--greedy-repeats", type=int, default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)

    p.add_argument("--min-feature-prevalence", type=float, default=0.02)
    p.add_argument("--max-feature-prevalence", type=float, default=0.98)

    p.add_argument("--top-screen", type=int, default=30)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--max-selected", type=int, default=20)
    p.add_argument("--min-delta-rmse", type=float, default=0.0)
    p.add_argument("--min-delta-within1", type=float, default=-0.01)
    p.add_argument("--no-stop-on-no-gain", action="store_true")
    p.add_argument("--out-prefix", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    screen_repeats = args.screen_repeats if args.screen_repeats is not None else args.repeats
    greedy_repeats = args.greedy_repeats if args.greedy_repeats is not None else args.repeats

    log("[INFO] Loading feature matrices")
    X_amr = apply_prefix(read_table(args.amr_table, args.id_col_amr), args.feature_prefix_amr)
    X_gwas_raw = read_table(args.gwas_table, args.id_col_gwas)
    X_gwas = apply_prefix(X_gwas_raw, args.feature_prefix_gwas)
    y = load_mic(args.mic_file, args.mic_id_col, args.antibiotic, args.log2)

    common = X_amr.index.intersection(X_gwas.index).intersection(y.index)
    X_amr = X_amr.loc[common]
    X_gwas = X_gwas.loc[common]
    X_gwas_raw = X_gwas_raw.loc[common]
    y = y.loc[common]

    log(f"[INFO] Matched samples: {len(common)}")
    log(f"[INFO] AMR features:    {X_amr.shape[1]}")
    log(f"[INFO] GWAS features:   {X_gwas.shape[1]}")
    log(f"[INFO] Antibiotic:      {args.antibiotic}")

    baseline_res = eval_feature_set(
        X_amr=X_amr,
        X_gwas=None,
        y=y,
        gwas_features=[],
        classifier=args.classifier,
        repeats=args.repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    baseline_path = Path(str(args.out_prefix) + "_baseline.tsv")
    baseline_res.to_csv(baseline_path, sep="\t", index=False)
    baseline_summary = summarise_metrics(baseline_res)

    log(
        "[BASELINE] "
        f"MAE={baseline_summary['mae_mean']:.3f}, "
        f"RMSE={baseline_summary['rmse_mean']:.3f}, "
        f"R2={baseline_summary['r2_mean']:.3f}, "
        f"within1={baseline_summary['within1_mean']:.3f}"
    )

    screened = screen_features(
        X_amr=X_amr,
        X_gwas=X_gwas,
        y=y,
        baseline_summary=baseline_summary,
        classifier=args.classifier,
        repeats=screen_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        min_feature_prevalence=args.min_feature_prevalence,
        max_feature_prevalence=args.max_feature_prevalence,
    )
    screened_path = Path(str(args.out_prefix) + "_screened_features.tsv")
    screened.to_csv(screened_path, sep="\t", index=False)

    tested = screened[screened["status"] == "tested"].copy()
    top_features = tested["feature"].head(args.top_screen).tolist()

    greedy_path = Path(str(args.out_prefix) + "_greedy_path.tsv")
    if args.greedy and top_features:
        greedy_df, selected_features = greedy_forward_selection(
            X_amr=X_amr,
            X_gwas=X_gwas,
            y=y,
            candidate_features=top_features,
            baseline_summary=baseline_summary,
            classifier=args.classifier,
            repeats=greedy_repeats,
            test_size=args.test_size,
            random_state=args.random_state + 10000,
            n_jobs=args.n_jobs,
            max_selected=args.max_selected,
            min_delta_rmse=args.min_delta_rmse,
            min_delta_within1=args.min_delta_within1,
            stop_on_no_gain=not args.no_stop_on_no_gain,
        )
        greedy_df.to_csv(greedy_path, sep="\t", index=False)
    else:
        selected_features = top_features[: args.max_selected]
        pd.DataFrame().to_csv(greedy_path, sep="\t", index=False)

    prefix = f"{args.feature_prefix_gwas}__" if args.feature_prefix_gwas else ""
    selected_raw = [f[len(prefix):] if prefix and f.startswith(prefix) else f for f in selected_features]

    selected_df = pd.DataFrame({
        "selected_feature_prefixed": selected_features,
        "selected_feature_raw": selected_raw,
        "rank": range(1, len(selected_features) + 1),
    })

    if not tested.empty and selected_features:
        selected_df = selected_df.merge(
            tested,
            left_on="selected_feature_prefixed",
            right_on="feature",
            how="left",
        )

    selected_path = Path(str(args.out_prefix) + "_selected_features.tsv")
    selected_df.to_csv(selected_path, sep="\t", index=False)

    matrix_path = Path(str(args.out_prefix) + "_selected_gwas_matrix.tsv")
    if selected_raw:
        out_matrix = X_gwas_raw.loc[:, selected_raw].copy()
        out_matrix.insert(0, "sample", out_matrix.index)
    else:
        out_matrix = pd.DataFrame({"sample": X_gwas_raw.index})
    out_matrix.to_csv(matrix_path, sep="\t", index=False)

    meta = {
        "antibiotic": args.antibiotic,
        "matched_samples": int(len(common)),
        "amr_features": int(X_amr.shape[1]),
        "gwas_features": int(X_gwas.shape[1]),
        "selected_features": int(len(selected_features)),
        "classifier": args.classifier,
        "repeats": args.repeats,
        "screen_repeats": screen_repeats,
        "greedy": bool(args.greedy),
        "baseline_summary": baseline_summary,
    }
    meta_path = Path(str(args.out_prefix) + "_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    log("[OK] Wrote:")
    for p in [baseline_path, screened_path, greedy_path, selected_path, matrix_path, meta_path]:
        log(f"  {p}")
    log(f"[INFO] Selected features: {len(selected_features)}")


if __name__ == "__main__":
    main()
