#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import re
import math
from pathlib import Path
import zipfile

MAP_RE = re.compile(r"(?P<ref>[^:;,]+):(?P<start>\d+)-(?P<end>\d+)")
SNP_RE = re.compile(r"^(?P<ref>.+?)_(?P<pos>\d+)_")

# ----------------------------
# helpers
# ----------------------------

def neglog10(p):
    try:
        return -math.log10(float(p))
    except:
        return np.nan

def classify_replicon(ref):
    if pd.isna(ref) or ref == "":
        return "unmapped"
    if "CP043953" in ref:
        return "chromosome"
    if "plasmid" in ref.lower() or ref.startswith("CP") and ref != "CP043953.1":
        return "plasmid"
    return "pangenome"

# ----------------------------
# unitig parser
# ----------------------------

def parse_unitigs(path, drug):
    rows = []

    with open(path, errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            variant = parts[0]
            pval = parts[3]
            beta = parts[4]
            note = parts[5]

            mappings = parts[6:]

            for m in mappings:
                match = MAP_RE.search(m)
                if not match:
                    continue

                ref = match.group("ref")
                start = int(match.group("start"))
                end = int(match.group("end"))

                rows.append({
                    "drug": drug,
                    "feature_type": "unitig",
                    "feature_id": variant,
                    "reference": ref,
                    "start": start,
                    "end": end,
                    "midpoint": (start + end) / 2,
                    "pvalue": pval,
                    "beta": beta,
                    "note": note
                })

    return pd.DataFrame(rows)

# ----------------------------
# SNP parser
# ----------------------------

def parse_snp(path, drug):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            f = z.open(z.namelist()[0])
            df = pd.read_csv(f, sep="\t", dtype=str, on_bad_lines="skip")
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip")

    rows = []

    for _, r in df.iterrows():
        var = str(r.iloc[0])
        match = SNP_RE.match(var)

        if match:
            ref = match.group("ref")
            pos = int(match.group("pos"))
        else:
            ref = ""
            pos = np.nan

        rows.append({
            "drug": drug,
            "feature_type": "snp",
            "feature_id": var,
            "reference": ref,
            "start": pos,
            "end": pos,
            "midpoint": pos,
            "pvalue": r.get("lrt-pvalue", np.nan),
            "beta": r.get("beta", np.nan),
            "note": r.get("notes", "")
        })

    return pd.DataFrame(rows)

# ----------------------------
# gene PA parser
# ----------------------------

def parse_gene_pa(path, drug):
    df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip")

    rows = []
    for _, r in df.iterrows():
        gene = str(r.iloc[0])

        rows.append({
            "drug": drug,
            "feature_type": "gene_pa",
            "feature_id": gene,
            "reference": "",
            "start": np.nan,
            "end": np.nan,
            "midpoint": np.nan,
            "pvalue": r.get("lrt-pvalue", np.nan),
            "beta": r.get("beta", np.nan),
            "note": r.get("notes", "")
        })

    return pd.DataFrame(rows)

# ----------------------------
# collapse to loci
# ----------------------------

def collapse_loci(df):
    df["locus_id"] = (
        df["reference"].fillna("") + "_" +
        (df["start"].fillna(-1).astype(int) // 1000).astype(str)
    )

    grouped = df.groupby(["drug", "locus_id"]).agg({
        "reference": "first",
        "midpoint": "mean",
        "pvalue": lambda x: np.min(pd.to_numeric(x, errors="coerce")),
        "beta": lambda x: np.mean(pd.to_numeric(x, errors="coerce")),
        "feature_type": lambda x: ",".join(set(x))
    }).reset_index()

    grouped["neglog10_p"] = grouped["pvalue"].apply(neglog10)
    grouped["replicon"] = grouped["reference"].apply(classify_replicon)

    return grouped

# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--imi-unitig", nargs="*", default=[])
    ap.add_argument("--mer-unitig", nargs="*", default=[])
    ap.add_argument("--imi-snp", nargs="*", default=[])
    ap.add_argument("--mer-snp", nargs="*", default=[])
    ap.add_argument("--imi-pa", nargs="*", default=[])
    ap.add_argument("--mer-pa", nargs="*", default=[])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--p-threshold", type=float, default=1e-7)

    args = ap.parse_args()

    dfs = []

    for f in args.imi_unitig:
        dfs.append(parse_unitigs(f, "IMI"))

    for f in args.mer_unitig:
        dfs.append(parse_unitigs(f, "MER"))

    for f in args.imi_snp:
        dfs.append(parse_snp(f, "IMI"))

    for f in args.mer_snp:
        dfs.append(parse_snp(f, "MER"))

    for f in args.imi_pa:
        dfs.append(parse_gene_pa(f, "IMI"))

    for f in args.mer_pa:
        dfs.append(parse_gene_pa(f, "MER"))

    df = pd.concat(dfs, ignore_index=True)

    # filter
    df = df[~df["note"].astype(str).str.contains("bad-chisq", na=False)]
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    df = df[df["pvalue"] < args.p_threshold]

    # collapse
    loci = collapse_loci(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "gwas_raw.tsv", sep="\t", index=False)
    loci.to_csv(outdir / "gwas_loci.tsv", sep="\t", index=False)

    print("\n[SUMMARY]")
    print(df["drug"].value_counts())
    print(df["feature_type"].value_counts())
    print(loci["replicon"].value_counts())

if __name__ == "__main__":
    main()