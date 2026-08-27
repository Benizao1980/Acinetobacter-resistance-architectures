#!/usr/bin/env python3
"""
build_raw_pyseer_feature_ranking.py

Parse raw pyseer outputs from unitigs, SNPs and gene presence/absence,
add directionality and optional h2, collapse unitigs to loci, and produce
ranked feature/locus tables for model feature selection.

This script is deliberately conservative:
  - It does not assume every pyseer output has variant_h2.
  - It keeps 'bad-chisq' rows by default but annotates them.
  - It uses beta direction explicitly:
        beta > 0: feature presence/allele increases MIC
        beta < 0: feature presence/allele decreases MIC
  - For unitigs, it can collapse variants to previously defined loci using
    a locus summary table with 'locus' and 'unitig_ids' columns.

Example: IMI-only raw outputs

  python scripts/build_raw_pyseer_feature_ranking.py \
    --unitig-imi GWAS_prep/pyseer_IPM_lmm_unitigs.zip \
    --snp-imi GWAS_prep/Ipm_pyseer_SNPs_output.zip \
    --pa-imi GWAS_prep/PA_Ipm_COGs.txt \
    --locus-summary GWAS_prep/gwas_combined_locus_summary.tsv \
    --out-prefix outputs/gwas_raw_rankings/imi_raw

Example: IMI + MER when both are available

  python scripts/build_raw_pyseer_feature_ranking.py \
    --unitig-imi GWAS_prep/pyseer_IPM_lmm_unitigs.zip \
    --unitig-mer GWAS_prep/pyseer_MER_lmm_unitigs.zip \
    --snp-imi GWAS_prep/Ipm_pyseer_SNPs_output.zip \
    --snp-mer GWAS_prep/Mer_pyseer_SNPs_output.zip \
    --pa-imi GWAS_prep/PA_Ipm_COGs.txt \
    --pa-mer GWAS_prep/PA_Mer_COGs.txt \
    --locus-summary GWAS_prep/gwas_combined_locus_summary.tsv \
    --exclude-bad-chisq \
    --top-k 50 \
    --out-prefix outputs/gwas_raw_rankings/imi_mer_directional
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------

def read_pyseer_table(path: Path) -> pd.DataFrame:
    """Read a pyseer TSV/CSV, including a zip containing one text-like output file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            candidates = [
                n for n in z.namelist()
                if not n.endswith("/")
                and not Path(n).name.startswith("._")
                and "__MACOSX" not in n
                and Path(n).suffix.lower() in {".txt", ".tsv", ".csv", ".tab", ""}
            ]
            if not candidates:
                raise ValueError(f"No readable table found inside zip: {path}")
            # Prefer files that look like pyseer outputs
            candidates = sorted(candidates, key=lambda x: (0 if "pyseer" in x.lower() or "output" in x.lower() else 1, len(x)))
            member = candidates[0]
            with z.open(member) as fh:
                df = pd.read_csv(fh, sep=None, engine="python", dtype=str)
            df.attrs["source_member"] = member
            return df

    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def safe_float(x: object) -> float:
    try:
        val = float(x)
        if math.isnan(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def neg_log10(p: object) -> float:
    p = safe_float(p)
    if math.isnan(p) or p <= 0:
        return float("nan")
    return -math.log10(p)


def split_ids(val: object) -> List[str]:
    if pd.isna(val):
        return []
    return [x.strip() for x in re.split(r"[;,|]", str(val)) if x.strip()]


# -----------------------------
# pyseer parsing
# -----------------------------

def normalise_pyseer(
    df: pd.DataFrame,
    drug: str,
    feature_type: str,
    source_path: Path,
    variant_col: str = "variant",
    pvalue_col: str = "lrt-pvalue",
    filter_pvalue_col: str = "filter-pvalue",
    beta_col: str = "beta",
    beta_se_col: str = "beta-std-err",
    h2_col: str = "variant_h2",
    notes_col: str = "notes",
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = [variant_col, pvalue_col, beta_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_path} missing required columns {missing}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    out = pd.DataFrame()
    out["variant"] = df[variant_col].astype(str)
    out["drug"] = drug.upper()
    out["feature_type"] = feature_type
    out["source_file"] = str(source_path)
    out["source_member"] = df.attrs.get("source_member", "")

    out["lrt_pvalue"] = df[pvalue_col].apply(safe_float)
    out["neglog10_lrt_pvalue"] = out["lrt_pvalue"].apply(neg_log10)
    if filter_pvalue_col in df.columns:
        out["filter_pvalue"] = df[filter_pvalue_col].apply(safe_float)
        out["neglog10_filter_pvalue"] = out["filter_pvalue"].apply(neg_log10)
    else:
        out["filter_pvalue"] = np.nan
        out["neglog10_filter_pvalue"] = np.nan

    out["beta"] = df[beta_col].apply(safe_float)
    if beta_se_col in df.columns:
        out["beta_std_err"] = df[beta_se_col].apply(safe_float)
    else:
        out["beta_std_err"] = np.nan

    if h2_col in df.columns:
        out["variant_h2"] = df[h2_col].apply(safe_float)
    else:
        out["variant_h2"] = np.nan

    if "af" in df.columns:
        out["af"] = df["af"].apply(safe_float)
    else:
        out["af"] = np.nan

    if notes_col in df.columns:
        out["notes"] = df[notes_col].fillna("").astype(str)
    else:
        out["notes"] = ""

    out["bad_chisq"] = out["notes"].str.contains("bad-chisq", case=False, na=False)
    out["abs_beta"] = out["beta"].abs()
    out["direction"] = np.select(
        [out["beta"] > 0, out["beta"] < 0],
        ["presence_increases_MIC", "presence_decreases_MIC"],
        default="zero_or_unknown",
    )

    # Basic prioritisation score.
    # h2 may be missing for PA; use 1.0 as neutral weight when missing.
    h2_weight = out["variant_h2"].fillna(1.0)
    out["priority_score"] = out["neglog10_lrt_pvalue"].fillna(0) * h2_weight * out["abs_beta"].fillna(0)

    out = out.dropna(subset=["lrt_pvalue", "beta"])
    return out


# -----------------------------
# Unitig-to-locus mapping
# -----------------------------

def build_unitig_to_locus_from_summary(
    locus_summary: Path,
    locus_col: str = "locus",
    unitig_ids_col: str = "unitig_ids",
) -> Dict[str, str]:
    summary = read_table(locus_summary)
    if locus_col not in summary.columns:
        raise ValueError(f"'{locus_col}' not found in {locus_summary}")
    if unitig_ids_col not in summary.columns:
        raise ValueError(f"'{unitig_ids_col}' not found in {locus_summary}")

    mapping: Dict[str, str] = {}
    for _, row in summary.iterrows():
        locus = str(row[locus_col])
        for uid in split_ids(row[unitig_ids_col]):
            mapping[uid] = locus
    return mapping


def add_feature_group(
    long_df: pd.DataFrame,
    unitig_to_locus: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = long_df.copy()
    df["feature_group"] = df["variant"]

    # Collapse unitigs to locus where possible.
    if unitig_to_locus:
        mask = df["feature_type"].eq("unitig")
        df.loc[mask, "feature_group"] = df.loc[mask, "variant"].map(unitig_to_locus)
        # For unmapped unitigs, keep variant ID but mark as unmapped.
        unmapped = mask & df["feature_group"].isna()
        df.loc[unmapped, "feature_group"] = "UNMAPPED_UNITIG__" + df.loc[unmapped, "variant"].astype(str)

    return df


# -----------------------------
# Collapse and scoring
# -----------------------------

def direction_summary(beta_values: Iterable[float]) -> Tuple[str, int, int, float]:
    vals = [v for v in beta_values if not math.isnan(v)]
    n_pos = sum(v > 0 for v in vals)
    n_neg = sum(v < 0 for v in vals)
    n = n_pos + n_neg
    if n == 0:
        return "zero_or_unknown", 0, 0, float("nan")
    if n_pos >= n_neg:
        direction = "presence_increases_MIC"
        consistency = n_pos / n
    else:
        direction = "presence_decreases_MIC"
        consistency = n_neg / n
    return direction, n_pos, n_neg, float(consistency)


def collapse_feature_groups(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["feature_group", "feature_type", "drug"]
    for (feature_group, feature_type, drug), sub in long_df.groupby(group_cols, dropna=False):
        sub = sub.copy()
        sub = sub.sort_values(["lrt_pvalue", "filter_pvalue"], ascending=[True, True], kind="stable")
        best = sub.iloc[0]

        direction, n_pos, n_neg, direction_consistency = direction_summary(sub["beta"].tolist())

        row = {
            "feature_group": feature_group,
            "feature_type": feature_type,
            "drug": drug,
            "n_variants": int(sub["variant"].nunique()),
            "n_bad_chisq": int(sub["bad_chisq"].sum()),
            "min_lrt_pvalue": float(sub["lrt_pvalue"].min()),
            "max_neglog10_lrt_pvalue": float(sub["neglog10_lrt_pvalue"].max()),
            "min_filter_pvalue": float(sub["filter_pvalue"].min()) if sub["filter_pvalue"].notna().any() else np.nan,
            "beta_at_min_p": float(best["beta"]),
            "abs_beta_at_min_p": float(abs(best["beta"])),
            "max_abs_beta": float(sub["abs_beta"].max()),
            "max_variant_h2": float(sub["variant_h2"].max()) if sub["variant_h2"].notna().any() else np.nan,
            "mean_variant_h2": float(sub["variant_h2"].mean()) if sub["variant_h2"].notna().any() else np.nan,
            "dominant_direction": direction,
            "n_positive_beta": int(n_pos),
            "n_negative_beta": int(n_neg),
            "direction_consistency": direction_consistency,
            "best_variant": str(best["variant"]),
            "variant_examples": ";".join(map(str, sub["variant"].head(10).tolist())),
        }

        # Score: p-value strength x effect size x h2/stability x direction consistency.
        h2_weight = row["max_variant_h2"] if not math.isnan(row["max_variant_h2"]) else 1.0
        row["priority_score"] = (
            row["max_neglog10_lrt_pvalue"]
            * row["abs_beta_at_min_p"]
            * h2_weight
            * (row["direction_consistency"] if not math.isnan(row["direction_consistency"]) else 1.0)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def combine_across_drugs(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return group_df

    pivots = []
    keep_cols = [
        "n_variants", "n_bad_chisq", "min_lrt_pvalue", "max_neglog10_lrt_pvalue",
        "min_filter_pvalue", "beta_at_min_p", "abs_beta_at_min_p", "max_abs_beta",
        "max_variant_h2", "mean_variant_h2", "dominant_direction",
        "n_positive_beta", "n_negative_beta", "direction_consistency",
        "best_variant", "variant_examples", "priority_score"
    ]

    base = group_df[["feature_group", "feature_type"]].drop_duplicates()
    out = base.copy()

    for drug in sorted(group_df["drug"].dropna().unique()):
        sub = group_df[group_df["drug"] == drug][["feature_group"] + keep_cols].copy()
        sub = sub.rename(columns={c: f"{c}_{drug.lower()}" for c in keep_cols})
        out = out.merge(sub, on="feature_group", how="left")

    drug_names = sorted(group_df["drug"].dropna().unique())
    p_cols = [f"min_lrt_pvalue_{d.lower()}" for d in drug_names if f"min_lrt_pvalue_{d.lower()}" in out.columns]
    score_cols = [f"priority_score_{d.lower()}" for d in drug_names if f"priority_score_{d.lower()}" in out.columns]
    h2_cols = [f"max_variant_h2_{d.lower()}" for d in drug_names if f"max_variant_h2_{d.lower()}" in out.columns]
    beta_cols = [f"beta_at_min_p_{d.lower()}" for d in drug_names if f"beta_at_min_p_{d.lower()}" in out.columns]

    out["min_lrt_pvalue_any"] = out[p_cols].min(axis=1) if p_cols else np.nan
    out["max_priority_score_any"] = out[score_cols].max(axis=1) if score_cols else np.nan
    out["max_variant_h2_any"] = out[h2_cols].max(axis=1) if h2_cols else np.nan
    out["max_abs_beta_any"] = out[beta_cols].abs().max(axis=1) if beta_cols else np.nan

    # shared across drugs
    out["n_drugs_hit"] = out[p_cols].notna().sum(axis=1) if p_cols else 0
    out["shared_across_drugs"] = out["n_drugs_hit"] >= 2

    # Direction agreement across drugs where both are present
    dir_cols = [f"dominant_direction_{d.lower()}" for d in drug_names if f"dominant_direction_{d.lower()}" in out.columns]
    def direction_agreement(row):
        vals = [row[c] for c in dir_cols if pd.notna(row[c])]
        vals = [v for v in vals if v != "zero_or_unknown"]
        if len(vals) < 2:
            return np.nan
        return len(set(vals)) == 1
    out["same_direction_across_drugs"] = out.apply(direction_agreement, axis=1)

    out = out.sort_values(["max_priority_score_any", "min_lrt_pvalue_any"], ascending=[False, True], kind="stable")
    return out


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build direction-aware rankings from raw pyseer outputs.")
    p.add_argument("--out-prefix", required=True, type=Path)

    # IMI
    p.add_argument("--unitig-imi", type=Path, default=None)
    p.add_argument("--snp-imi", type=Path, default=None)
    p.add_argument("--pa-imi", type=Path, default=None)

    # MER
    p.add_argument("--unitig-mer", type=Path, default=None)
    p.add_argument("--snp-mer", type=Path, default=None)
    p.add_argument("--pa-mer", type=Path, default=None)

    # Optional unitig collapsing
    p.add_argument("--locus-summary", type=Path, default=None,
                   help="Collapsed locus summary with locus and unitig_ids columns.")
    p.add_argument("--locus-col", default="locus")
    p.add_argument("--unitig-ids-col", default="unitig_ids")

    # Columns
    p.add_argument("--variant-col", default="variant")
    p.add_argument("--pvalue-col", default="lrt-pvalue")
    p.add_argument("--filter-pvalue-col", default="filter-pvalue")
    p.add_argument("--beta-col", default="beta")
    p.add_argument("--beta-se-col", default="beta-std-err")
    p.add_argument("--h2-col", default="variant_h2")
    p.add_argument("--notes-col", default="notes")

    # Filtering / selection
    p.add_argument("--exclude-bad-chisq", action="store_true")
    p.add_argument("--pvalue-threshold", type=float, default=None)
    p.add_argument("--min-h2", type=float, default=None)
    p.add_argument("--require-direction-consistency", type=float, default=None,
                   help="Minimum within-group direction consistency, e.g. 0.8.")
    p.add_argument("--require-shared-drugs", action="store_true")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--feature-types", nargs="*", default=None,
                   choices=["unitig", "snp", "gene_pa"],
                   help="Optional feature types to keep in selected output.")

    return p.parse_args()


def add_input(records: List[Tuple[Path, str, str]], path: Optional[Path], drug: str, feature_type: str) -> None:
    if path is not None:
        records.append((path, drug, feature_type))


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    inputs: List[Tuple[Path, str, str]] = []
    add_input(inputs, args.unitig_imi, "IMI", "unitig")
    add_input(inputs, args.snp_imi, "IMI", "snp")
    add_input(inputs, args.pa_imi, "IMI", "gene_pa")
    add_input(inputs, args.unitig_mer, "MER", "unitig")
    add_input(inputs, args.snp_mer, "MER", "snp")
    add_input(inputs, args.pa_mer, "MER", "gene_pa")

    if not inputs:
        raise SystemExit("[ERROR] No pyseer input files supplied.")

    long_parts = []
    for path, drug, feature_type in inputs:
        raw = read_pyseer_table(path)
        norm = normalise_pyseer(
            raw, drug=drug, feature_type=feature_type, source_path=path,
            variant_col=args.variant_col,
            pvalue_col=args.pvalue_col,
            filter_pvalue_col=args.filter_pvalue_col,
            beta_col=args.beta_col,
            beta_se_col=args.beta_se_col,
            h2_col=args.h2_col,
            notes_col=args.notes_col,
        )
        long_parts.append(norm)
        print(f"[INFO] Loaded {drug} {feature_type}: {path} | rows={len(norm)}")

    long_df = pd.concat(long_parts, axis=0, ignore_index=True)

    if args.exclude_bad_chisq:
        before = len(long_df)
        long_df = long_df[~long_df["bad_chisq"]].copy()
        print(f"[INFO] Excluded bad-chisq rows: {before - len(long_df)}")

    unitig_to_locus = None
    if args.locus_summary is not None:
        unitig_to_locus = build_unitig_to_locus_from_summary(
            args.locus_summary,
            locus_col=args.locus_col,
            unitig_ids_col=args.unitig_ids_col,
        )
        print(f"[INFO] Loaded unitig→locus mappings: {len(unitig_to_locus)}")

    long_df = add_feature_group(long_df, unitig_to_locus=unitig_to_locus)

    # Optional raw-hit p-value threshold before collapsing
    if args.pvalue_threshold is not None:
        before = len(long_df)
        long_df = long_df[long_df["lrt_pvalue"] <= args.pvalue_threshold].copy()
        print(f"[INFO] Applied raw p-value threshold {args.pvalue_threshold}: kept {len(long_df)} / {before}")

    raw_path = Path(str(args.out_prefix) + "_raw_long.tsv")
    long_df.to_csv(raw_path, sep="\t", index=False)

    per_drug_group = collapse_feature_groups(long_df)
    per_drug_path = Path(str(args.out_prefix) + "_per_drug_feature_groups.tsv")
    per_drug_group.to_csv(per_drug_path, sep="\t", index=False)

    combined = combine_across_drugs(per_drug_group)
    combined_path = Path(str(args.out_prefix) + "_combined_feature_ranking.tsv")
    combined.to_csv(combined_path, sep="\t", index=False)

    selected = combined.copy()
    if args.feature_types:
        selected = selected[selected["feature_type"].isin(args.feature_types)].copy()
    if args.min_h2 is not None:
        selected = selected[selected["max_variant_h2_any"].fillna(0) >= args.min_h2].copy()
    if args.require_direction_consistency is not None:
        # keep if any drug has sufficient direction consistency
        dcols = [c for c in selected.columns if c.startswith("direction_consistency_")]
        if dcols:
            selected = selected[selected[dcols].max(axis=1, skipna=True) >= args.require_direction_consistency].copy()
    if args.require_shared_drugs:
        selected = selected[selected["shared_across_drugs"]].copy()
    if args.top_k is not None:
        selected = selected.head(args.top_k).copy()

    selected_path = Path(str(args.out_prefix) + "_selected_features.tsv")
    selected.to_csv(selected_path, sep="\t", index=False)

    # Convenience: unitig loci only, useful for filtering locus matrices.
    unitig_loci = combined[combined["feature_type"].eq("unitig")].copy()
    unitig_loci_path = Path(str(args.out_prefix) + "_unitig_locus_ranking.tsv")
    unitig_loci.to_csv(unitig_loci_path, sep="\t", index=False)

    print("[OK] Wrote:")
    print(f"  {raw_path}")
    print(f"  {per_drug_path}")
    print(f"  {combined_path}")
    print(f"  {selected_path}")
    print(f"  {unitig_loci_path}")
    print(f"[INFO] Raw rows:              {len(long_df)}")
    print(f"[INFO] Feature groups/drug:   {len(per_drug_group)}")
    print(f"[INFO] Combined feature rows: {len(combined)}")
    print(f"[INFO] Selected rows:         {len(selected)}")


if __name__ == "__main__":
    main()
