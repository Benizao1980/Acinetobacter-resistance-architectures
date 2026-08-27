#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keys = ["pvalue", "neglog10", "beta", "h2", "score", "n_", "consistency"]
    for col in df.columns:
        if any(k in col for k in keys):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def infer_feature_id(df: pd.DataFrame) -> pd.Series:
    for col in ["feature_group", "locus", "variant", "feature_id"]:
        if col in df.columns:
            return df[col].astype(str)
    raise ValueError("Input table must contain feature_group, locus, variant or feature_id.")


def standardise(df: pd.DataFrame, source_label: str, source_file: Path) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["feature_id"] = infer_feature_id(df)
    if "feature_type" not in df.columns:
        df["feature_type"] = "unknown"
    df["evidence_source"] = source_label
    df["evidence_file"] = str(source_file)
    df = numeric_columns(df)

    if "max_priority_score_any" not in df.columns:
        cols = [c for c in df.columns if c.startswith("priority_score")]
        df["max_priority_score_any"] = df[cols].max(axis=1) if cols else np.nan
    if "min_lrt_pvalue_any" not in df.columns:
        cols = [c for c in df.columns if c.startswith("min_lrt_pvalue")]
        df["min_lrt_pvalue_any"] = df[cols].min(axis=1) if cols else np.nan
    if "max_variant_h2_any" not in df.columns:
        cols = [c for c in df.columns if c.startswith("max_variant_h2")]
        df["max_variant_h2_any"] = df[cols].max(axis=1) if cols else np.nan
    if "max_abs_beta_any" not in df.columns:
        cols = [c for c in df.columns if c.startswith("beta_at_min_p")]
        df["max_abs_beta_any"] = df[cols].abs().max(axis=1) if cols else np.nan
    return df


def select_features(
    df: pd.DataFrame,
    top_k: Optional[int],
    feature_types: Optional[List[str]],
    min_h2: Optional[float],
    require_shared: bool,
    require_same_direction: bool,
    exclude_unmapped_unitigs: bool,
) -> pd.DataFrame:
    sel = df.copy()
    if feature_types:
        sel = sel[sel["feature_type"].isin(feature_types)].copy()
    if exclude_unmapped_unitigs:
        sel = sel[
            ~(sel["feature_type"].eq("unitig") & sel["feature_id"].astype(str).str.startswith("UNMAPPED_UNITIG__"))
        ].copy()
    if min_h2 is not None:
        sel = sel[sel["max_variant_h2_any"].fillna(0) >= min_h2].copy()
    if require_shared and "shared_across_drugs" in sel.columns:
        sel = sel[sel["shared_across_drugs"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if require_same_direction and "same_direction_across_drugs" in sel.columns:
        sel = sel[sel["same_direction_across_drugs"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    sel = sel.sort_values(["max_priority_score_any", "min_lrt_pvalue_any"], ascending=[False, True], kind="stable")
    return sel.head(top_k).copy() if top_k else sel


def parse_args():
    p = argparse.ArgumentParser(description="Combine direction-aware GWAS evidence tables into one master evidence table.")
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out-prefix", required=True, type=Path)
    p.add_argument("--top-k", type=int)
    p.add_argument("--feature-types", nargs="*", choices=["unitig", "snp", "gene_pa", "unknown"])
    p.add_argument("--min-h2", type=float)
    p.add_argument("--require-shared-drugs", action="store_true")
    p.add_argument("--require-same-direction", action="store_true")
    p.add_argument("--exclude-unmapped-unitigs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must match --inputs length.")

    frames = []
    for i, path in enumerate(args.inputs):
        label = args.labels[i] if args.labels else path.stem
        df = read_table(path)
        frames.append(standardise(df, label, path))
        print(f"[INFO] Loaded {label}: {path} | rows={len(df)}")

    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["max_priority_score_any", "min_lrt_pvalue_any"], ascending=[False, True], kind="stable")
    master = master.drop_duplicates(subset=["feature_id", "feature_type", "evidence_source"], keep="first")

    master_path = Path(str(args.out_prefix) + "_master_evidence.tsv")
    master.to_csv(master_path, sep="\t", index=False)

    for ft in ["unitig", "snp", "gene_pa"]:
        sub = master[master["feature_type"].eq(ft)].copy()
        sub.to_csv(Path(str(args.out_prefix) + f"_{ft}_evidence.tsv"), sep="\t", index=False)

    selected = select_features(
        master, args.top_k, args.feature_types, args.min_h2,
        args.require_shared_drugs, args.require_same_direction, args.exclude_unmapped_unitigs
    )
    selected_path = Path(str(args.out_prefix) + "_selected_features.tsv")
    selected.to_csv(selected_path, sep="\t", index=False)

    print("[OK] Wrote:")
    print(f"  {master_path}")
    print(f"  {Path(str(args.out_prefix) + '_unitig_evidence.tsv')}")
    print(f"  {Path(str(args.out_prefix) + '_snp_evidence.tsv')}")
    print(f"  {Path(str(args.out_prefix) + '_gene_pa_evidence.tsv')}")
    print(f"  {selected_path}")
    print(f"[INFO] Master rows:   {len(master)}")
    print(f"[INFO] Selected rows: {len(selected)}")


if __name__ == "__main__":
    main()
