#!/usr/bin/env python3
"""
filter_gwas_loci.py

Filter a locus-level GWAS matrix to the top-k loci, using either:
  1) an existing ranking table, or
  2) raw IMI / MER GWAS hit tables collapsed to loci via a locus summary table.

This is designed for BAMPS-ML workflows where loci have already been defined
by collapsing unitig hits to locus-level features.

Examples
--------
Build top-25 loci ranked by minimum p-value across IMI and MER:

  python scripts/filter_gwas_loci.py \
    --matrix GWAS_prep/gwas_features_russia280_locus_presence_absence.tsv \
    --locus-summary GWAS_prep/gwas_combined_locus_summary.tsv \
    --imi-hits GWAS_prep/IMI_hits.csv \
    --mer-hits GWAS_prep/MER_hits.csv \
    --top-k 25 \
    --source ANY \
    --out outputs/filtered_gwas/russia_gwas_top25.tsv \
    --selected-out outputs/filtered_gwas/russia_gwas_top25_selected.tsv

Restrict to IMI-ranked loci only:

  python scripts/filter_gwas_loci.py \
    --matrix GWAS_prep/gwas_features_validation_locus_presence_absence.tsv \
    --locus-summary GWAS_prep/gwas_combined_locus_summary.tsv \
    --imi-hits GWAS_prep/IMI_hits.csv \
    --top-k 50 \
    --source IMI \
    --out outputs/filtered_gwas/validation_gwas_IMI_top50.tsv

Use an existing ranking file directly (must contain a locus column):

  python scripts/filter_gwas_loci.py \
    --matrix GWAS_prep/gwas_features_russia280_locus_presence_absence.tsv \
    --ranking GWAS_prep/gwas_locus_ranking.tsv \
    --ranking-col min_p_any \
    --top-k 25 \
    --out outputs/filtered_gwas/russia_gwas_top25.tsv
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter a locus-level GWAS matrix to the top-k loci."
    )
    p.add_argument("--matrix", required=True, type=Path,
                   help="Input locus presence/absence TSV. First column should be sample ID.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output filtered TSV.")
    p.add_argument("--top-k", required=True, type=int,
                   help="Number of loci to retain.")
    p.add_argument("--id-col", default=None,
                   help="ID column in matrix. Default: first column.")
    p.add_argument("--locus-col", default="locus",
                   help="Locus column name in ranking / summary files (default: locus).")
    p.add_argument("--selected-out", type=Path, default=None,
                   help="Optional TSV of retained loci and ranking metadata.")
    p.add_argument("--ranking-out", type=Path, default=None,
                   help="Optional TSV of the full derived locus ranking.")

    # Direct ranking mode
    p.add_argument("--ranking", type=Path, default=None,
                   help="Optional existing ranking TSV with at least a locus column.")
    p.add_argument("--ranking-col", default=None,
                   help="Ranking column to sort on if using --ranking.")
    p.add_argument("--ascending", action="store_true",
                   help="Sort the ranking column ascending instead of descending.")

    # Raw-hit collapse mode
    p.add_argument("--locus-summary", type=Path, default=None,
                   help="Locus summary TSV from collapsed unitigs. Must include locus and unitig_ids.")
    p.add_argument("--unitig-ids-col", default="unitig_ids",
                   help="Column in locus summary containing semicolon-separated unitig IDs.")
    p.add_argument("--sources-col", default="sources",
                   help="Column in locus summary containing source labels (default: sources).")
    p.add_argument("--imi-hits", type=Path, default=None,
                   help="Raw IMI GWAS hits CSV/TSV.")
    p.add_argument("--mer-hits", type=Path, default=None,
                   help="Raw MER GWAS hits CSV/TSV.")
    p.add_argument("--imi-unitig-col", default="Unitig_Ipm_identifier",
                   help="Unitig identifier column in IMI hits.")
    p.add_argument("--mer-unitig-col", default="Unitig_identifier",
                   help="Unitig identifier column in MER hits.")
    p.add_argument("--pval-col", default="Pvalue",
                   help="P-value column in raw hit files (default: Pvalue).")
    p.add_argument("--source", choices=["ANY", "IMI", "MER"], default="ANY",
                   help="Which source to rank/filter on (default: ANY).")
    p.add_argument("--require-source-in-summary", action="store_true",
                   help="When using raw hits, restrict to loci whose summary 'sources' includes the requested source.")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def _safe_float(x: object) -> float:
    try:
        val = float(x)
        if math.isnan(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def _split_unitig_ids(val: object) -> List[str]:
    if pd.isna(val):
        return []
    parts = re.split(r"[;,|]", str(val))
    return [p.strip() for p in parts if p.strip()]


def build_unitig_to_locus(summary_df: pd.DataFrame, locus_col: str, unitig_ids_col: str) -> Dict[str, str]:
    if locus_col not in summary_df.columns:
        raise ValueError(f"Locus column '{locus_col}' not found in locus summary.")
    if unitig_ids_col not in summary_df.columns:
        raise ValueError(f"Unitig IDs column '{unitig_ids_col}' not found in locus summary.")

    mapping: Dict[str, str] = {}
    for _, row in summary_df.iterrows():
        locus = str(row[locus_col])
        for unitig_id in _split_unitig_ids(row[unitig_ids_col]):
            mapping[unitig_id] = locus
    return mapping


def collapse_hits_to_loci(
    hits_df: pd.DataFrame,
    unitig_col: str,
    pval_col: str,
    unitig_to_locus: Dict[str, str],
    source_name: str,
) -> pd.DataFrame:
    if unitig_col not in hits_df.columns:
        raise ValueError(
            f"Unitig column '{unitig_col}' not found in {source_name} hits. "
            f"Available columns: {', '.join(hits_df.columns)}"
        )
    if pval_col not in hits_df.columns:
        raise ValueError(
            f"P-value column '{pval_col}' not found in {source_name} hits. "
            f"Available columns: {', '.join(hits_df.columns)}"
        )

    tmp = hits_df[[unitig_col, pval_col]].copy()
    tmp["unitig_id"] = tmp[unitig_col].astype(str)
    tmp["pvalue"] = tmp[pval_col].apply(_safe_float)
    tmp = tmp.dropna(subset=["pvalue"])
    tmp["locus"] = tmp["unitig_id"].map(unitig_to_locus)
    tmp = tmp.dropna(subset=["locus"])

    if tmp.empty:
        return pd.DataFrame(columns=["locus", f"min_p_{source_name.lower()}", f"n_hits_{source_name.lower()}"])

    out = (
        tmp.groupby("locus", as_index=False)
           .agg(
               **{
                   f"min_p_{source_name.lower()}": ("pvalue", "min"),
                   f"n_hits_{source_name.lower()}": ("unitig_id", "count"),
               }
           )
    )
    return out


def build_ranking_from_raw_hits(args: argparse.Namespace) -> pd.DataFrame:
    if args.locus_summary is None:
        raise ValueError("Raw-hit mode requires --locus-summary.")
    if args.imi_hits is None and args.mer_hits is None:
        raise ValueError("Raw-hit mode requires at least one of --imi-hits or --mer-hits.")

    summary_df = read_table(args.locus_summary)
    unitig_to_locus = build_unitig_to_locus(summary_df, args.locus_col, args.unitig_ids_col)

    base_cols = [c for c in [args.locus_col, args.sources_col] if c in summary_df.columns]
    ranking = summary_df[base_cols].drop_duplicates(subset=[args.locus_col]).copy()

    if args.imi_hits is not None:
        imi_df = read_table(args.imi_hits)
        imi_rank = collapse_hits_to_loci(
            hits_df=imi_df,
            unitig_col=args.imi_unitig_col,
            pval_col=args.pval_col,
            unitig_to_locus=unitig_to_locus,
            source_name="IMI",
        )
        ranking = ranking.merge(
            imi_rank,
            left_on=args.locus_col,
            right_on="locus",
            how="left",
            suffixes=("", "_imi_tmp"),
        )
        if "locus_imi_tmp" in ranking.columns:
            ranking = ranking.drop(columns=["locus_imi_tmp"])
        elif "locus" in ranking.columns and args.locus_col != "locus":
            ranking = ranking.drop(columns=["locus"])

    if args.mer_hits is not None:
        mer_df = read_table(args.mer_hits)
        mer_rank = collapse_hits_to_loci(
            hits_df=mer_df,
            unitig_col=args.mer_unitig_col,
            pval_col=args.pval_col,
            unitig_to_locus=unitig_to_locus,
            source_name="MER",
        )
        ranking = ranking.merge(
            mer_rank,
            left_on=args.locus_col,
            right_on="locus",
            how="left",
            suffixes=("", "_mer_tmp"),
        )
        if "locus_mer_tmp" in ranking.columns:
            ranking = ranking.drop(columns=["locus_mer_tmp"])
        elif "locus" in ranking.columns and args.locus_col != "locus":
            ranking = ranking.drop(columns=["locus"])

    # Combined ranking columns
    def combine_min_p(row: pd.Series) -> float:
        vals = []
        for col in ["min_p_imi", "min_p_mer"]:
            if col in row.index:
                v = _safe_float(row[col])
                if not math.isnan(v):
                    vals.append(v)
        return min(vals) if vals else float("nan")

    ranking["min_p_any"] = ranking.apply(combine_min_p, axis=1)

    def combine_sources(row: pd.Series) -> str:
        s = []
        if not math.isnan(_safe_float(row.get("min_p_imi", float("nan")))):
            s.append("IMI")
        if not math.isnan(_safe_float(row.get("min_p_mer", float("nan")))):
            s.append("MER")
        return ";".join(s)

    ranking["rank_sources"] = ranking.apply(combine_sources, axis=1)

    # Optional source-based restriction
    if args.source != "ANY":
        pcol = f"min_p_{args.source.lower()}"
        ranking = ranking[~ranking[pcol].apply(_safe_float).apply(math.isnan)].copy()
        if args.require_source_in_summary and args.sources_col in ranking.columns:
            ranking = ranking[
                ranking[args.sources_col].fillna("").astype(str).str.contains(rf"(^|;){args.source}(;|$)")
            ].copy()
        ranking = ranking.sort_values(pcol, ascending=True, kind="stable")
    else:
        ranking = ranking[~ranking["min_p_any"].apply(math.isnan)].copy()
        ranking = ranking.sort_values("min_p_any", ascending=True, kind="stable")

    return ranking.reset_index(drop=True)


def choose_loci_from_ranking(
    ranking_df: pd.DataFrame,
    locus_col: str,
    top_k: int,
    ranking_col: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    if locus_col not in ranking_df.columns:
        raise ValueError(
            f"Locus column '{locus_col}' not found in ranking file. "
            f"Available columns: {', '.join(ranking_df.columns)}"
        )

    df = ranking_df.copy()
    df = df.dropna(subset=[locus_col])
    df[locus_col] = df[locus_col].astype(str)
    df = df.loc[~df[locus_col].duplicated(keep="first")].copy()

    if ranking_col:
        if ranking_col not in df.columns:
            raise ValueError(
                f"Ranking column '{ranking_col}' not found in ranking file. "
                f"Available columns: {', '.join(df.columns)}"
            )
        numeric = pd.to_numeric(df[ranking_col], errors="coerce")
        if numeric.isna().all():
            raise ValueError(f"Ranking column '{ranking_col}' could not be interpreted numerically.")
        df["_rank_value"] = numeric
        df = df.sort_values("_rank_value", ascending=ascending, kind="stable")
    else:
        df["_rank_order"] = range(len(df))
        df = df.sort_values("_rank_order", kind="stable")

    return df.head(top_k).copy()


def main() -> None:
    args = parse_args()

    matrix_df = read_table(args.matrix)
    if matrix_df.empty:
        raise ValueError(f"Matrix file is empty: {args.matrix}")

    id_col = args.id_col if args.id_col else matrix_df.columns[0]
    if id_col not in matrix_df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in matrix file. "
            f"Available columns: {', '.join(matrix_df.columns)}"
        )

    # Build or load ranking
    if args.ranking is not None:
        ranking_df = read_table(args.ranking)
    else:
        ranking_df = build_ranking_from_raw_hits(args)

    if args.ranking_out:
        args.ranking_out.parent.mkdir(parents=True, exist_ok=True)
        ranking_df.to_csv(args.ranking_out, sep="\t", index=False)

    # Choose top-k loci
    # For raw-hit mode, sorting is already applied; preserve that order.
    selected_df = choose_loci_from_ranking(
        ranking_df=ranking_df,
        locus_col=args.locus_col,
        top_k=args.top_k,
        ranking_col=args.ranking_col if args.ranking is not None else None,
        ascending=args.ascending,
    )

    selected_loci = selected_df[args.locus_col].astype(str).tolist()
    matrix_cols = set(map(str, matrix_df.columns))

    present_loci = [l for l in selected_loci if l in matrix_cols]
    missing_loci = [l for l in selected_loci if l not in matrix_cols]

    out_cols = [id_col] + present_loci
    filtered_df = matrix_df[out_cols].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.out, sep="\t", index=False)

    if args.selected_out:
        args.selected_out.parent.mkdir(parents=True, exist_ok=True)
        selected_df = selected_df.copy()
        selected_df["in_matrix"] = selected_df[args.locus_col].astype(str).isin(matrix_cols)
        selected_df.to_csv(args.selected_out, sep="\t", index=False)

    print(f"[OK] Wrote filtered matrix: {args.out}")
    if args.ranking_out:
        print(f"[OK] Wrote full ranking:   {args.ranking_out}")
    if args.selected_out:
        print(f"[OK] Wrote selected loci:  {args.selected_out}")
    print(f"[INFO] Matrix samples:        {filtered_df.shape[0]}")
    print(f"[INFO] Requested top-k loci: {args.top_k}")
    print(f"[INFO] Selected loci rows:   {len(selected_loci)}")
    print(f"[INFO] Loci found in matrix: {len(present_loci)}")
    print(f"[INFO] Loci missing:         {len(missing_loci)}")

    if missing_loci:
        preview = ", ".join(missing_loci[:10])
        print(f"[WARN] Example missing loci: {preview}")
        if len(missing_loci) > 10:
            print(f"[WARN] ... and {len(missing_loci) - 10} more")

    if len(present_loci) == 0:
        print("[ERROR] None of the selected loci were found in the matrix columns.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
