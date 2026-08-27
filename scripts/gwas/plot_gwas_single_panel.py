#!/usr/bin/env python3

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMI = "#E67E22"      # burnt orange
MER = "#F4C542"      # mustard yellow
BOTH = "#C1121F"     # carbapenem red
GREY = "#B8B8B8"
BG = "#FAF7F2"
INK = "#2B2B2B"

MAP_RE = re.compile(r"(?P<ref>[^:;,]+):(?P<start>\d+)-(?P<end>\d+)")

def parse_annotation_file(path, drug):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            variant = parts[0]
            nums = parts[1:5]
            note = parts[5] if len(parts) > 5 else ""
            mapping = parts[-1] if len(parts) > 6 else ""

            m = MAP_RE.search(mapping)
            if not m:
                continue

            try:
                af = float(nums[0])
                filter_p = float(nums[1])
                lrt_p = float(nums[2])
                beta = float(nums[3])
            except Exception:
                continue

            fields = mapping.split(";")
            gene = ""
            for x in fields[1:]:
                if x and not x.startswith("cds-"):
                    gene = x
                    break
            if not gene and len(fields) > 1:
                gene = fields[1].replace("cds-", "")

            start, end = int(m.group("start")), int(m.group("end"))
            rows.append({
                "drug": drug,
                "variant": variant,
                "reference": m.group("ref"),
                "start": start,
                "end": end,
                "midpoint": (start + end) / 2,
                "af": af,
                "filter_pvalue": filter_p,
                "lrt_pvalue": lrt_p,
                "beta": beta,
                "note": note,
                "gene_label": gene,
                "source_file": str(path),
            })
    return pd.DataFrame(rows)

def load_mer(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    keep = ["drug","variant","reference","start","end","midpoint","af",
            "filter_pvalue","lrt_pvalue","beta","notes","gene_label","source_file"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].rename(columns={"notes":"note"})
    df["drug"] = "MER"
    return df

def load_plasmid_refs(path):
    refs = set()
    if not path:
        return refs
    with open(path) as f:
        for line in f:
            x = line.strip().split()[0]
            x = x.replace(".fas", "")
            if x:
                refs.add(x)
    return refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imi-files", nargs="+", required=True)
    ap.add_argument("--mer-mapped", required=True)
    ap.add_argument("--plasmid-refs", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--drop-bad-chisq", action="store_true")
    ap.add_argument("--top-labels", type=int, default=12)
    args = ap.parse_args()

    imi = pd.concat([parse_annotation_file(p, "IMI") for p in args.imi_files], ignore_index=True)
    mer = load_mer(args.mer_mapped)
    df = pd.concat([imi, mer], ignore_index=True)

    if args.drop_bad_chisq:
        df = df[~df["note"].astype(str).str.contains("bad-chisq", na=False)]

    df["lrt_pvalue"] = pd.to_numeric(df["lrt_pvalue"], errors="coerce")
    df["midpoint"] = pd.to_numeric(df["midpoint"], errors="coerce")
    df = df.dropna(subset=["reference", "midpoint", "lrt_pvalue"])
    df = df[df["lrt_pvalue"] > 0]
    df["neglog10_p"] = -np.log10(df["lrt_pvalue"])

    plasmids = load_plasmid_refs(args.plasmid_refs)

    def block(ref):
        if ref == "CP043953.1":
            return "Chromosome"
        if ref in plasmids or ref == "CP043954.1":
            return "Plasmids"
        return "Rest of pangenome"

    df["block"] = df["reference"].map(block)

    # collapse near-identical same-reference/same-position hits
    rows = []
    for _, g in df.groupby(["reference", "midpoint"], dropna=False):
        best = g.sort_values("lrt_pvalue").iloc[0].copy()
        drugs = set(g["drug"])
        if "IMI" in drugs and "MER" in drugs:
            best["signal"] = "BOTH"
        elif "IMI" in drugs:
            best["signal"] = "IMI"
        elif "MER" in drugs:
            best["signal"] = "MER"
        else:
            best["signal"] = "OTHER"
        best["neglog10_p"] = g["neglog10_p"].max()
        rows.append(best)
    plot = pd.DataFrame(rows)

    block_order = ["Chromosome", "Plasmids", "Rest of pangenome"]
    offsets = {}
    cursor = 0
    sep_positions = []
    tick_positions = []
    tick_labels = []

    for b in block_order:
        sub = plot[plot["block"] == b]
        if sub.empty:
            continue
        refs = ["CP043953.1"] if b == "Chromosome" else sorted(sub["reference"].unique())
        start_cursor = cursor
        for ref in refs:
            rsub = sub[sub["reference"] == ref]
            if rsub.empty:
                continue
            offsets[ref] = cursor
            maxpos = rsub["midpoint"].max()
            tick_positions.append(cursor + maxpos / 2)
            tick_labels.append(ref if b != "Rest of pangenome" else "")
            cursor += maxpos + 50000
        sep_positions.append(cursor)
        cursor += 150000

    plot["x"] = plot.apply(lambda r: r["midpoint"] + offsets.get(r["reference"], 0), axis=1)

    colors = {"IMI": IMI, "MER": MER, "BOTH": BOTH, "OTHER": GREY}

    fig, ax = plt.subplots(figsize=(13,4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for sig in ["IMI", "MER", "BOTH", "OTHER"]:
        sub = plot[plot["signal"] == sig]
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["neglog10_p"], s=18 if sig != "BOTH" else 28,
                   color=colors[sig], alpha=0.82, edgecolor="none", label=sig)

    ax.axhline(-np.log10(1e-7), ls="--", lw=1, color=INK, alpha=0.6)

    for x in sep_positions[:-1]:
        ax.axvline(x, ls="--", lw=1, color=INK, alpha=0.45)

    ymax = max(8, plot["neglog10_p"].max() * 1.12)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(r"$-\log_{10}(P)$")
    ax.set_xlabel("Genomic context")

    # block labels
    for b in block_order:
        sub = plot[plot["block"] == b]
        if sub.empty:
            continue
        ax.text((sub["x"].min()+sub["x"].max())/2, -0.18*ymax, b,
                ha="center", va="top", fontsize=11, color=INK)

    # label top hits, but keep sparse
    lab = plot.sort_values("neglog10_p", ascending=False).head(args.top_labels)
    for _, r in lab.iterrows():
        label = str(r.get("gene_label", "")).replace("cds-", "")
        if not label or label == "nan":
            label = str(r["reference"])
        ax.text(r["x"], r["neglog10_p"] + 0.4, label, fontsize=8,
                rotation=25, ha="center", va="bottom", color=INK)

    handles = [
        plt.Line2D([0],[0], marker="o", ls="", color=IMI, label="IMI only"),
        plt.Line2D([0],[0], marker="o", ls="", color=MER, label="MER only"),
        plt.Line2D([0],[0], marker="o", ls="", color=BOTH, label="IMI + MER"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=3)

    ax.grid(axis="y", color="#E6E1D8", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([])

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.to_csv(str(out) + "_plot_data.tsv", sep="\t", index=False)
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", facecolor=BG)
    fig.savefig(str(out) + ".svg", bbox_inches="tight", facecolor=BG)

    print(f"[OK] Wrote {out}.png/svg")
    print(f"[OK] Wrote {out}_plot_data.tsv")
    print(plot["signal"].value_counts().to_string())
    print(plot["block"].value_counts().to_string())

if __name__ == "__main__":
    main()
PY