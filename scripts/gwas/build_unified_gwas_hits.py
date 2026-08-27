#!/usr/bin/env python3
import argparse, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

MAP_RE = re.compile(r"(?P<ref>[^:;,]+):(?P<start>\d+)-(?P<end>\d+)")
SNP_RE = re.compile(r"^(?P<ref>.+?)_(?P<pos>\d+)_(?P<refbase>[ACGT]+)_(?P<altbase>[ACGT]+)$")

def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def neglog10(p):
    p = safe_float(p)
    if not np.isfinite(p) or p <= 0:
        return np.nan
    return -math.log10(p)

def parse_gene_label(mapping):
    fields = str(mapping).split(";")
    for x in fields[1:]:
        if x and x.lower() != "nan" and not x.startswith("cds-"):
            return x
    if len(fields) > 1 and fields[1]:
        return fields[1].replace("cds-", "")
    return ""

def parse_annotation_file(path, drug, feature_type="unitig"):
    rows = []
    path = Path(path)
    with path.open(errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            variant = parts[0]
            af = safe_float(parts[1]) if len(parts) > 1 else np.nan
            filter_p = safe_float(parts[2]) if len(parts) > 2 else np.nan
            lrt_p = safe_float(parts[3]) if len(parts) > 3 else np.nan
            beta = safe_float(parts[4]) if len(parts) > 4 else np.nan
            note = parts[5] if len(parts) > 5 else ""
            mappings = [p for p in parts[6:] if MAP_RE.search(str(p))]
            for mapping in mappings:
                for one in str(mapping).split(","):
                    m = MAP_RE.search(one)
                    if not m:
                        continue
                    start, end = int(m.group("start")), int(m.group("end"))
                    gene = parse_gene_label(one)
                    rows.append({
                        "drug": drug, "feature_type": feature_type,
                        "feature_id": gene if gene else variant, "variant": variant,
                        "reference": m.group("ref"), "start": start, "end": end,
                        "midpoint": (start + end) / 2, "gene_label": gene,
                        "af": af, "filter_pvalue": filter_p, "lrt_pvalue": lrt_p,
                        "beta": beta, "beta_std_err": np.nan, "variant_h2": np.nan,
                        "note": note, "source_file": str(path)
                    })
    return pd.DataFrame(rows)

def read_table_or_zip(path):
    path = Path(path)
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if not m.endswith("/")]
            if not members:
                raise ValueError(f"No files in zip: {path}")
            member = next((m for m in members if m.lower().endswith((".txt",".tsv",".csv",".out"))), members[0])
            with z.open(member) as fh:
                return pd.read_csv(fh, sep="\t", dtype=str, engine="python", on_bad_lines="skip")
    return pd.read_csv(path, sep="\t", dtype=str, engine="python", on_bad_lines="skip")

def standardise_pyseer_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    def get_col(names, fallback_idx=None):
        for n in names:
            if n in lower:
                return lower[n]
        if fallback_idx is not None and fallback_idx < len(df.columns):
            return df.columns[fallback_idx]
        return None
    variant_col = get_col(["variant", "snp"], 0)
    af_col = get_col(["af"], 1)
    filter_p_col = get_col(["filter-pvalue", "filter_pvalue"], 2)
    lrt_p_col = get_col(["lrt-pvalue", "lrt_pvalue", "pvalue", "p"], 3)
    beta_col = get_col(["beta"], 4)
    beta_se_col = get_col(["beta-std-err", "beta_std_err"], 5)
    h2_col = get_col(["variant_h2", "h2"], 6)
    notes_col = get_col(["notes", "note"], None)
    out = pd.DataFrame()
    out["variant"] = df[variant_col].astype(str)
    out["af"] = pd.to_numeric(df[af_col], errors="coerce") if af_col else np.nan
    out["filter_pvalue"] = pd.to_numeric(df[filter_p_col], errors="coerce") if filter_p_col else np.nan
    out["lrt_pvalue"] = pd.to_numeric(df[lrt_p_col], errors="coerce") if lrt_p_col else np.nan
    out["beta"] = pd.to_numeric(df[beta_col], errors="coerce") if beta_col else np.nan
    out["beta_std_err"] = pd.to_numeric(df[beta_se_col], errors="coerce") if beta_se_col else np.nan
    out["variant_h2"] = pd.to_numeric(df[h2_col], errors="coerce") if h2_col else np.nan
    out["note"] = df[notes_col].astype(str) if notes_col else ""
    return out

def parse_snp_file(path, drug):
    df = standardise_pyseer_columns(read_table_or_zip(path))
    rows = []
    for _, r in df.iterrows():
        var = str(r["variant"])
        m = SNP_RE.match(var)
        if m:
            ref, pos = m.group("ref"), int(m.group("pos"))
            start = end = midpoint = pos
        else:
            ref, start, end, midpoint = "", np.nan, np.nan, np.nan
        rows.append({
            "drug": drug, "feature_type": "snp", "feature_id": var, "variant": var,
            "reference": ref, "start": start, "end": end, "midpoint": midpoint,
            "gene_label": "", "af": r["af"], "filter_pvalue": r["filter_pvalue"],
            "lrt_pvalue": r["lrt_pvalue"], "beta": r["beta"],
            "beta_std_err": r["beta_std_err"], "variant_h2": r["variant_h2"],
            "note": r["note"], "source_file": str(path)
        })
    return pd.DataFrame(rows)

def parse_gene_pa_file(path, drug):
    df = standardise_pyseer_columns(read_table_or_zip(path))
    rows = []
    for _, r in df.iterrows():
        var = str(r["variant"])
        rows.append({
            "drug": drug, "feature_type": "gene_pa", "feature_id": var, "variant": var,
            "reference": "", "start": np.nan, "end": np.nan, "midpoint": np.nan,
            "gene_label": var, "af": r["af"], "filter_pvalue": r["filter_pvalue"],
            "lrt_pvalue": r["lrt_pvalue"], "beta": r["beta"],
            "beta_std_err": r["beta_std_err"], "variant_h2": r["variant_h2"],
            "note": r["note"], "source_file": str(path)
        })
    return pd.DataFrame(rows)

def add_summary_fields(df):
    df = df.copy()
    df["lrt_pvalue"] = pd.to_numeric(df["lrt_pvalue"], errors="coerce")
    df["neglog10_p"] = df["lrt_pvalue"].apply(neglog10)
    df["overlap_key"] = df["gene_label"].fillna("").astype(str)
    missing = df["overlap_key"].isin(["", "nan", "None"])
    df.loc[missing, "overlap_key"] = df.loc[missing, "feature_id"].astype(str)
    return df

def main():
    ap = argparse.ArgumentParser(description="Build unified IMI/MER GWAS hit table.")
    ap.add_argument("--imi-unitig", nargs="*", default=[])
    ap.add_argument("--mer-unitig", nargs="*", default=[])
    ap.add_argument("--imi-snp", nargs="*", default=[])
    ap.add_argument("--mer-snp", nargs="*", default=[])
    ap.add_argument("--imi-pa", nargs="*", default=[])
    ap.add_argument("--mer-pa", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-bad-chisq", action="store_true")
    ap.add_argument("--p-threshold", type=float, default=None)
    args = ap.parse_args()
    frames = []
    for f in args.imi_unitig: frames.append(parse_annotation_file(f, "IMI", "unitig"))
    for f in args.mer_unitig: frames.append(parse_annotation_file(f, "MER", "unitig"))
    for f in args.imi_snp: frames.append(parse_snp_file(f, "IMI"))
    for f in args.mer_snp: frames.append(parse_snp_file(f, "MER"))
    for f in args.imi_pa: frames.append(parse_gene_pa_file(f, "IMI"))
    for f in args.mer_pa: frames.append(parse_gene_pa_file(f, "MER"))
    if not frames:
        raise ValueError("No input files supplied.")
    df = add_summary_fields(pd.concat(frames, ignore_index=True))
    if args.drop_bad_chisq:
        df = df[~df["note"].astype(str).str.contains("bad-chisq", case=False, na=False)]
    if args.p_threshold is not None:
        df = df[pd.to_numeric(df["lrt_pvalue"], errors="coerce") < args.p_threshold]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[OK] Wrote {out}")
    print(f"[INFO] rows={len(df)}")
    print("\n[drug]")
    print(df["drug"].value_counts(dropna=False).to_string())
    print("\n[feature_type]")
    print(df["feature_type"].value_counts(dropna=False).to_string())
    print("\n[mapped rows]")
    print(df["reference"].replace("", np.nan).notna().value_counts().to_string())

if __name__ == "__main__":
    main()
