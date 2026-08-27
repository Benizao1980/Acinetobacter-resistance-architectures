#!/usr/bin/env python3
"""
parse_gwas_annotations.py

Parse mapped pyseer/unitig annotation files into one tidy GWAS mapping table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


MAP_RE = re.compile(r"(?P<ref>[^:;,]+):(?P<start>\d+)-(?P<end>\d+)")


def infer_drug(path: str, user_drug: Optional[str] = None) -> str:
    if user_drug:
        return user_drug
    low = path.lower()
    if "mem" in low or "mer" in low:
        return "MER"
    if "ipm" in low or "imi" in low:
        return "IMI"
    return "UNKNOWN"


def read_loose_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        engine="python",
        on_bad_lines="skip"   # ← key fix
    )
    df.columns = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in df.columns]
    if "variant" in lower or "snp" in lower:
        return df
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        engine="python",
        header=None,
        on_bad_lines="skip"
    )


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = list(df.columns)
    low_map = {str(c).lower().strip(): c for c in original_cols}

    if "variant" in low_map:
        variant_col = low_map["variant"]
    elif "snp" in low_map:
        variant_col = low_map["snp"]
    else:
        variant_col = original_cols[0]

    def pick(names, idx=None):
        for n in names:
            if n in low_map:
                return low_map[n]
        if idx is not None and idx < len(original_cols):
            return original_cols[idx]
        return None

    out = pd.DataFrame()
    out["variant"] = df[variant_col].astype(str)
    out["af"] = df[pick(["af"], 1)] if pick(["af"], 1) is not None else np.nan
    out["filter_pvalue"] = df[pick(["filter-pvalue", "filter_pvalue"], 2)] if pick(["filter-pvalue", "filter_pvalue"], 2) is not None else np.nan
    out["lrt_pvalue"] = df[pick(["lrt-pvalue", "lrt_pvalue", "p"], 3)] if pick(["lrt-pvalue", "lrt_pvalue", "p"], 3) is not None else np.nan
    out["beta"] = df[pick(["beta"], 4)] if pick(["beta"], 4) is not None else np.nan
    out["beta_std_err"] = df[pick(["beta-std-err", "beta_std_err"], 5)] if pick(["beta-std-err", "beta_std_err"], 5) is not None else np.nan
    out["variant_h2"] = df[pick(["variant_h2", "h2"], 6)] if pick(["variant_h2", "h2"], 6) is not None else np.nan

    ann_col = None
    for c in reversed(original_cols):
        vals = df[c].astype(str)
        if vals.str.contains(r":[0-9]+-[0-9]+", regex=True, na=False).any():
            ann_col = c
            break
    if ann_col is None:
        ann_col = original_cols[-1]

    out["mapping_raw"] = df[ann_col].astype(str)

    notes_col = pick(["notes"], None)
    if notes_col is None:
        for c in original_cols:
            if c == ann_col:
                continue
            if df[c].astype(str).str.contains("bad-chisq", na=False).any():
                notes_col = c
                break
    out["notes"] = df[notes_col].astype(str) if notes_col is not None else ""

    for c in ["af", "filter_pvalue", "lrt_pvalue", "beta", "beta_std_err", "variant_h2"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def parse_one_mapping(mapping: str) -> List[Dict[str, object]]:
    if mapping is None or str(mapping).strip() in {"", "nan", "NaN"}:
        return []

    results = []
    parts = [p.strip() for p in str(mapping).split(",") if p.strip()]
    for part in parts:
        m = MAP_RE.search(part)
        if not m:
            continue

        reference = m.group("ref")
        start = int(m.group("start"))
        end = int(m.group("end"))

        fields = part.split(";")
        ann = fields[1:] + ["", "", ""]

        labels = [x for x in ann[:3] if x and x.lower() != "nan"]
        gene_label = ""
        for x in labels:
            if not x.startswith("cds-"):
                gene_label = x
                break
        if not gene_label and labels:
            gene_label = labels[0].replace("cds-", "")

        results.append({
            "mapping_raw_one": part,
            "reference": reference,
            "start": start,
            "end": end,
            "midpoint": (start + end) / 2,
            "gene_label": gene_label,
            "annotation_1": ann[0],
            "annotation_2": ann[1],
            "annotation_3": ann[2],
        })

    return results


def classify_feature(row: pd.Series) -> str:
    v = str(row.get("variant", ""))
    ref = str(row.get("reference", ""))
    gene = str(row.get("gene_label", ""))
    if re.match(r"^[A-Za-z0-9_.]+_\d+_[ACGT]+_[ACGT]+$", v):
        return "snp"
    if gene.startswith("g") and "_" in gene:
        return "gene_pa"
    if ref:
        return "mapped_unitig"
    return "unitig"


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse mapped GWAS annotation files into a tidy table.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Annotation files to parse.")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels/drugs for inputs, same length as --inputs.")
    ap.add_argument("--out", required=True, help="Output TSV.")
    ap.add_argument("--drop-bad-chisq", action="store_true", help="Exclude rows with bad-chisq in notes.")
    args = ap.parse_args()

    rows = []

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must be same length as --inputs")

    for i, fp in enumerate(args.inputs):
        path = Path(fp)
        drug = infer_drug(str(path), args.labels[i] if args.labels else None)

        df = read_loose_table(path)
        norm = normalise_columns(df)

        if args.drop_bad_chisq:
            norm = norm[~norm["notes"].astype(str).str.contains("bad-chisq", na=False)]

        for _, r in norm.iterrows():
            maps = parse_one_mapping(r["mapping_raw"])
            if not maps:
                rows.append({
                    "source_file": str(path),
                    "drug": drug,
                    "variant": r["variant"],
                    "af": r["af"],
                    "filter_pvalue": r["filter_pvalue"],
                    "lrt_pvalue": r["lrt_pvalue"],
                    "beta": r["beta"],
                    "beta_std_err": r["beta_std_err"],
                    "variant_h2": r["variant_h2"],
                    "notes": r["notes"],
                    "mapping_raw": r["mapping_raw"],
                    "reference": "",
                    "start": np.nan,
                    "end": np.nan,
                    "midpoint": np.nan,
                    "gene_label": "",
                    "annotation_1": "",
                    "annotation_2": "",
                    "annotation_3": "",
                })
            else:
                for mp in maps:
                    rows.append({
                        "source_file": str(path),
                        "drug": drug,
                        "variant": r["variant"],
                        "af": r["af"],
                        "filter_pvalue": r["filter_pvalue"],
                        "lrt_pvalue": r["lrt_pvalue"],
                        "beta": r["beta"],
                        "beta_std_err": r["beta_std_err"],
                        "variant_h2": r["variant_h2"],
                        "notes": r["notes"],
                        "mapping_raw": r["mapping_raw"],
                        **mp,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows parsed.")

    out["neglog10_p"] = -np.log10(pd.to_numeric(out["lrt_pvalue"], errors="coerce"))
    out["feature_class"] = out.apply(classify_feature, axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] Wrote {out_path}")
    print(f"[INFO] rows={len(out)}")
    print(f"[INFO] references={out['reference'].nunique(dropna=True)}")
    print(out["drug"].value_counts(dropna=False).to_string())
    print(out["feature_class"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
