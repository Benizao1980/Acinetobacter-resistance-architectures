#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_pyseer_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            candidates = [
                n for n in z.namelist()
                if not n.endswith("/")
                and "__MACOSX" not in n
                and not Path(n).name.startswith("._")
                and Path(n).suffix.lower() in {".txt", ".tsv", ".csv", ".tab", ""}
            ]
            if not candidates:
                raise ValueError(f"No readable table found inside zip: {path}")
            candidates = sorted(candidates, key=lambda x: (0 if ("pyseer" in x.lower() or "output" in x.lower()) else 1, len(x)))
            member = candidates[0]
            with z.open(member) as fh:
                df = pd.read_csv(fh, sep="\t", dtype=str, index_col=False)
            df.attrs["source_member"] = member
            return clean_columns(df)
    return clean_columns(pd.read_csv(path, sep="\t", dtype=str, index_col=False))


def read_table(path: Path) -> pd.DataFrame:
    return clean_columns(pd.read_csv(path, sep="\t", dtype=str, index_col=False))


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if not math.isnan(v) else float("nan")
    except Exception:
        return float("nan")


def neg_log10(x) -> float:
    p = safe_float(x)
    if math.isnan(p) or p <= 0:
        return float("nan")
    return -math.log10(p)


def split_ids(x) -> List[str]:
    if pd.isna(x):
        return []
    return [v.strip() for v in re.split(r"[;,|]", str(x)) if v.strip()]


def unitig_id_to_locus(summary_path: Path, locus_col: str, unitig_ids_col: str) -> Dict[str, str]:
    s = read_table(summary_path)
    if locus_col not in s.columns or unitig_ids_col not in s.columns:
        raise ValueError(f"{summary_path} must contain {locus_col} and {unitig_ids_col}")
    out = {}
    for _, row in s.iterrows():
        locus = str(row[locus_col])
        for uid in split_ids(row[unitig_ids_col]):
            out[str(uid)] = locus
    return out


def infer_annotation_cols(df: pd.DataFrame, drug_hint: str) -> Tuple[str, str]:
    df = clean_columns(df)
    id_candidates = [
        f"Unitig_{drug_hint}_identifier",
        "Unitig_Ipm_identifier",
        "Unitig_Mer_identifier",
        "Unitig_identifier",
        "unitig_id",
        "unitig_identifier",
    ]
    seq_candidates = [
        f"Unitig {drug_hint}",
        "Unitig Ipm",
        "Unitig Mer",
        "Unitig",
        "variant",
        "sequence",
    ]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    seq_col = next((c for c in seq_candidates if c in df.columns), None)
    if id_col is None:
        matches = [c for c in df.columns if "identifier" in c.lower()]
        id_col = matches[0] if matches else None
    if seq_col is None:
        non_id = [c for c in df.columns if c != id_col]
        seq_col = non_id[0] if non_id else None
    if id_col is None or seq_col is None:
        raise ValueError(f"Could not infer annotation columns. Columns={list(df.columns)}")
    return seq_col, id_col


def sequence_to_locus(annotation_path: Path, id_to_locus: Dict[str, str], drug_hint: str) -> Dict[str, str]:
    ann = clean_columns(pd.read_csv(annotation_path, sep=None, engine="python", dtype=str))
    seq_col, id_col = infer_annotation_cols(ann, drug_hint)
    out = {}
    for _, row in ann.iterrows():
        seq = str(row[seq_col]).strip()
        uid = str(row[id_col]).strip()
        if seq and uid and seq != "nan" and uid != "nan" and uid in id_to_locus:
            out[seq] = id_to_locus[uid]
    print(f"[INFO] Annotation bridge {annotation_path}: seq_col={seq_col}, id_col={id_col}, mapped={len(out)}")
    return out


def normalise_pyseer(
    df: pd.DataFrame,
    drug: str,
    feature_type: str,
    source_path: Path,
    variant_col: str,
    pvalue_col: str,
    filter_pvalue_col: str,
    beta_col: str,
    beta_se_col: str,
    h2_col: str,
    notes_col: str,
) -> pd.DataFrame:
    df = clean_columns(df)
    for c in [variant_col, pvalue_col, beta_col]:
        if c not in df.columns:
            raise ValueError(f"{source_path} missing {c}; columns={list(df.columns)}")
    out = pd.DataFrame()
    out["variant"] = df[variant_col].astype(str)
    out["drug"] = drug.upper()
    out["feature_type"] = feature_type
    out["source_file"] = str(source_path)
    out["source_member"] = df.attrs.get("source_member", "")
    out["lrt_pvalue"] = df[pvalue_col].apply(safe_float)
    out["neglog10_lrt_pvalue"] = out["lrt_pvalue"].apply(neg_log10)
    out["filter_pvalue"] = df[filter_pvalue_col].apply(safe_float) if filter_pvalue_col in df.columns else np.nan
    out["neglog10_filter_pvalue"] = out["filter_pvalue"].apply(neg_log10)
    out["beta"] = df[beta_col].apply(safe_float)
    out["beta_std_err"] = df[beta_se_col].apply(safe_float) if beta_se_col in df.columns else np.nan
    out["variant_h2"] = df[h2_col].apply(safe_float) if h2_col in df.columns else np.nan
    out["af"] = df["af"].apply(safe_float) if "af" in df.columns else np.nan
    out["notes"] = df[notes_col].fillna("").astype(str) if notes_col in df.columns else ""
    out["bad_chisq"] = out["notes"].str.contains("bad-chisq", case=False, na=False)
    out["abs_beta"] = out["beta"].abs()
    out["direction"] = np.select(
        [out["beta"] > 0, out["beta"] < 0],
        ["presence_increases_MIC", "presence_decreases_MIC"],
        default="zero_or_unknown",
    )
    h2_weight = out["variant_h2"].fillna(1.0)
    out["priority_score_variant"] = out["neglog10_lrt_pvalue"].fillna(0) * out["abs_beta"].fillna(0) * h2_weight
    return out.dropna(subset=["lrt_pvalue", "beta"])


def add_feature_group(df: pd.DataFrame, seq_maps: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    df = df.copy()
    df["feature_group"] = df["variant"]
    df["mapped_to_locus"] = False
    for drug, mp in seq_maps.items():
        mask = df["feature_type"].eq("unitig") & df["drug"].eq(drug.upper())
        mapped = df.loc[mask, "variant"].map(mp)
        hit = mask & mapped.notna()
        df.loc[hit, "feature_group"] = mapped[hit]
        df.loc[hit, "mapped_to_locus"] = True
    unmapped = df["feature_type"].eq("unitig") & ~df["mapped_to_locus"]
    df.loc[unmapped, "feature_group"] = "UNMAPPED_UNITIG__" + df.loc[unmapped, "variant"].astype(str)
    return df


def direction_summary(vals: Iterable[float]) -> Tuple[str, int, int, float]:
    vals = [v for v in vals if not math.isnan(v)]
    npos, nneg = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    n = npos + nneg
    if n == 0:
        return "zero_or_unknown", 0, 0, float("nan")
    if npos >= nneg:
        return "presence_increases_MIC", npos, nneg, npos / n
    return "presence_decreases_MIC", npos, nneg, nneg / n


def collapse_feature_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fg, ft, drug), sub in df.groupby(["feature_group", "feature_type", "drug"], dropna=False):
        sub = sub.sort_values(["lrt_pvalue", "filter_pvalue"], ascending=[True, True], kind="stable")
        best = sub.iloc[0]
        direction, npos, nneg, consistency = direction_summary(sub["beta"].tolist())
        h2_present = sub["variant_h2"].notna().any()
        max_h2 = float(sub["variant_h2"].max()) if h2_present else np.nan
        h2_weight = max_h2 if not math.isnan(max_h2) else 1.0
        row = {
            "feature_group": fg,
            "feature_type": ft,
            "drug": drug,
            "n_variants": int(sub["variant"].nunique()),
            "n_mapped_to_locus": int(sub["mapped_to_locus"].sum()) if "mapped_to_locus" in sub.columns else 0,
            "n_bad_chisq": int(sub["bad_chisq"].sum()),
            "min_lrt_pvalue": float(sub["lrt_pvalue"].min()),
            "max_neglog10_lrt_pvalue": float(sub["neglog10_lrt_pvalue"].max()),
            "min_filter_pvalue": float(sub["filter_pvalue"].min()) if sub["filter_pvalue"].notna().any() else np.nan,
            "beta_at_min_p": float(best["beta"]),
            "abs_beta_at_min_p": float(abs(best["beta"])),
            "max_abs_beta": float(sub["abs_beta"].max()),
            "max_variant_h2": max_h2,
            "mean_variant_h2": float(sub["variant_h2"].mean()) if h2_present else np.nan,
            "dominant_direction": direction,
            "n_positive_beta": int(npos),
            "n_negative_beta": int(nneg),
            "direction_consistency": consistency,
            "best_variant": str(best["variant"]),
            "variant_examples": ";".join(map(str, sub["variant"].head(10).tolist())),
        }
        row["priority_score"] = row["max_neglog10_lrt_pvalue"] * row["abs_beta_at_min_p"] * h2_weight * (consistency if not math.isnan(consistency) else 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def combine_across_drugs(g: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "n_variants", "n_mapped_to_locus", "n_bad_chisq", "min_lrt_pvalue",
        "max_neglog10_lrt_pvalue", "min_filter_pvalue", "beta_at_min_p",
        "abs_beta_at_min_p", "max_abs_beta", "max_variant_h2", "mean_variant_h2",
        "dominant_direction", "n_positive_beta", "n_negative_beta",
        "direction_consistency", "best_variant", "variant_examples", "priority_score"
    ]
    out = g[["feature_group", "feature_type"]].drop_duplicates().copy()
    drugs = sorted(g["drug"].dropna().unique())
    for drug in drugs:
        sub = g[g["drug"] == drug][["feature_group"] + keep].copy()
        sub = sub.rename(columns={c: f"{c}_{drug.lower()}" for c in keep})
        out = out.merge(sub, on="feature_group", how="left")
    pcols = [f"min_lrt_pvalue_{d.lower()}" for d in drugs if f"min_lrt_pvalue_{d.lower()}" in out.columns]
    scols = [f"priority_score_{d.lower()}" for d in drugs if f"priority_score_{d.lower()}" in out.columns]
    hcols = [f"max_variant_h2_{d.lower()}" for d in drugs if f"max_variant_h2_{d.lower()}" in out.columns]
    bcols = [f"beta_at_min_p_{d.lower()}" for d in drugs if f"beta_at_min_p_{d.lower()}" in out.columns]
    out["min_lrt_pvalue_any"] = out[pcols].min(axis=1) if pcols else np.nan
    out["max_priority_score_any"] = out[scols].max(axis=1) if scols else np.nan
    out["max_variant_h2_any"] = out[hcols].max(axis=1) if hcols else np.nan
    out["max_abs_beta_any"] = out[bcols].abs().max(axis=1) if bcols else np.nan
    out["n_drugs_hit"] = out[pcols].notna().sum(axis=1) if pcols else 0
    out["shared_across_drugs"] = out["n_drugs_hit"] >= 2
    dcols = [f"dominant_direction_{d.lower()}" for d in drugs if f"dominant_direction_{d.lower()}" in out.columns]
    def same_dir(row):
        vals = [row[c] for c in dcols if pd.notna(row[c]) and row[c] != "zero_or_unknown"]
        return np.nan if len(vals) < 2 else len(set(vals)) == 1
    out["same_direction_across_drugs"] = out.apply(same_dir, axis=1)
    return out.sort_values(["max_priority_score_any", "min_lrt_pvalue_any"], ascending=[False, True], kind="stable")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", required=True, type=Path)
    p.add_argument("--unitig-imi", type=Path)
    p.add_argument("--unitig-annotation-imi", type=Path)
    p.add_argument("--snp-imi", type=Path)
    p.add_argument("--pa-imi", type=Path)
    p.add_argument("--unitig-mer", type=Path)
    p.add_argument("--unitig-annotation-mer", type=Path)
    p.add_argument("--snp-mer", type=Path)
    p.add_argument("--pa-mer", type=Path)
    p.add_argument("--locus-summary", type=Path)
    p.add_argument("--locus-col", default="locus")
    p.add_argument("--unitig-ids-col", default="unitig_ids")
    p.add_argument("--variant-col", default="variant")
    p.add_argument("--pvalue-col", default="lrt-pvalue")
    p.add_argument("--filter-pvalue-col", default="filter-pvalue")
    p.add_argument("--beta-col", default="beta")
    p.add_argument("--beta-se-col", default="beta-std-err")
    p.add_argument("--h2-col", default="variant_h2")
    p.add_argument("--notes-col", default="notes")
    p.add_argument("--exclude-bad-chisq", action="store_true")
    p.add_argument("--pvalue-threshold", type=float)
    p.add_argument("--min-h2", type=float)
    p.add_argument("--require-direction-consistency", type=float)
    p.add_argument("--require-shared-drugs", action="store_true")
    p.add_argument("--top-k", type=int)
    p.add_argument("--feature-types", nargs="*", choices=["unitig", "snp", "gene_pa"])
    p.add_argument("--mapped-unitigs-only", action="store_true")
    return p.parse_args()


def add_input(lst, path, drug, ft):
    if path is not None:
        lst.append((path, drug, ft))


def main():
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    inputs = []
    add_input(inputs, args.unitig_imi, "IMI", "unitig")
    add_input(inputs, args.snp_imi, "IMI", "snp")
    add_input(inputs, args.pa_imi, "IMI", "gene_pa")
    add_input(inputs, args.unitig_mer, "MER", "unitig")
    add_input(inputs, args.snp_mer, "MER", "snp")
    add_input(inputs, args.pa_mer, "MER", "gene_pa")
    if not inputs:
        raise SystemExit("[ERROR] No input files supplied.")

    parts = []
    for path, drug, ft in inputs:
        raw = read_pyseer_table(path)
        norm = normalise_pyseer(raw, drug, ft, path, args.variant_col, args.pvalue_col, args.filter_pvalue_col, args.beta_col, args.beta_se_col, args.h2_col, args.notes_col)
        parts.append(norm)
        print(f"[INFO] Loaded {drug} {ft}: {path} | rows={len(norm)}")
    long = pd.concat(parts, ignore_index=True)

    if args.exclude_bad_chisq:
        before = len(long)
        long = long[~long["bad_chisq"]].copy()
        print(f"[INFO] Excluded bad-chisq rows: {before - len(long)}")

    seq_maps = {}
    if args.locus_summary:
        id_map = unitig_id_to_locus(args.locus_summary, args.locus_col, args.unitig_ids_col)
        print(f"[INFO] Loaded numeric unitig→locus mappings: {len(id_map)}")
        if args.unitig_annotation_imi:
            seq_maps["IMI"] = sequence_to_locus(args.unitig_annotation_imi, id_map, "Ipm")
        if args.unitig_annotation_mer:
            seq_maps["MER"] = sequence_to_locus(args.unitig_annotation_mer, id_map, "Mer")
        if not seq_maps:
            print("[WARN] Locus summary supplied but no unitig annotation bridge supplied; unitigs will remain unmapped.")

    long = add_feature_group(long, seq_maps)

    if args.pvalue_threshold is not None:
        before = len(long)
        long = long[long["lrt_pvalue"] <= args.pvalue_threshold].copy()
        print(f"[INFO] Applied p threshold {args.pvalue_threshold}: kept {len(long)} / {before}")

    raw_path = Path(str(args.out_prefix) + "_raw_long.tsv")
    long.to_csv(raw_path, sep="\t", index=False)

    per = collapse_feature_groups(long)
    per_path = Path(str(args.out_prefix) + "_per_drug_feature_groups.tsv")
    per.to_csv(per_path, sep="\t", index=False)

    combined = combine_across_drugs(per)
    combined_path = Path(str(args.out_prefix) + "_combined_feature_ranking.tsv")
    combined.to_csv(combined_path, sep="\t", index=False)

    unitig_loci = combined[
        combined["feature_type"].eq("unitig")
        & ~combined["feature_group"].astype(str).str.startswith("UNMAPPED_UNITIG__")
    ].copy()
    unitig_loci_path = Path(str(args.out_prefix) + "_unitig_locus_ranking.tsv")
    unitig_loci.to_csv(unitig_loci_path, sep="\t", index=False)

    selected = combined.copy()
    if args.feature_types:
        selected = selected[selected["feature_type"].isin(args.feature_types)].copy()
    if args.mapped_unitigs_only:
        selected = selected[
            ~(selected["feature_type"].eq("unitig") & selected["feature_group"].astype(str).str.startswith("UNMAPPED_UNITIG__"))
        ].copy()
    if args.min_h2 is not None:
        selected = selected[selected["max_variant_h2_any"].fillna(0) >= args.min_h2].copy()
    if args.require_direction_consistency is not None:
        dcols = [c for c in selected.columns if c.startswith("direction_consistency_")]
        selected = selected[selected[dcols].max(axis=1, skipna=True) >= args.require_direction_consistency].copy()
    if args.require_shared_drugs:
        selected = selected[selected["shared_across_drugs"]].copy()
    if args.top_k:
        selected = selected.head(args.top_k).copy()

    selected_path = Path(str(args.out_prefix) + "_selected_features.tsv")
    selected.to_csv(selected_path, sep="\t", index=False)

    print("[OK] Wrote:")
    for p in [raw_path, per_path, combined_path, selected_path, unitig_loci_path]:
        print(f"  {p}")
    print(f"[INFO] Raw rows:                {len(long)}")
    print(f"[INFO] Feature groups/drug:     {len(per)}")
    print(f"[INFO] Combined feature rows:   {len(combined)}")
    print(f"[INFO] Unitig loci mapped rows: {len(unitig_loci)}")
    print(f"[INFO] Selected rows:           {len(selected)}")


if __name__ == "__main__":
    main()
