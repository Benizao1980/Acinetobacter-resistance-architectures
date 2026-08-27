#!/usr/bin/env python3
"""
plot_gwas_manhattan.py

Create a Wes-ish, replicon-aware Manhattan plot from parsed GWAS mappings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLOR_IMI = "#E67E22"
COLOR_MER = "#F4C542"
COLOR_BOTH = "#C1121F"
COLOR_OTHER = "#B8B8B8"
CHARCOAL = "#2B2B2B"
GRID = "#E6E1D8"
BACKGROUND = "#FAF7F2"


def classify_signal(drugs):
    s = set(str(x).upper() for x in drugs if str(x) != "nan")
    has_imi = "IMI" in s or "IPM" in s
    has_mer = "MER" in s or "MEM" in s
    if has_imi and has_mer:
        return "BOTH"
    if has_imi:
        return "IMI"
    if has_mer:
        return "MER"
    return "OTHER"


def signal_color(signal):
    return {"IMI": COLOR_IMI, "MER": COLOR_MER, "BOTH": COLOR_BOTH, "OTHER": COLOR_OTHER}.get(signal, COLOR_OTHER)


def label_text(row):
    for c in ["gene_label", "annotation_1", "annotation_2", "annotation_3", "variant"]:
        v = str(row.get(c, "")).strip()
        if v and v.lower() != "nan":
            return v.replace("cds-", "")
    return ""


def simplify_reference(ref):
    ref = str(ref)
    if ref in {"", "nan"}:
        return "unmapped"
    return ref


def make_collapsed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["midpoint", "lrt_pvalue"])
    df["reference"] = df["reference"].map(simplify_reference)
    df["midpoint"] = pd.to_numeric(df["midpoint"], errors="coerce")
    df["lrt_pvalue"] = pd.to_numeric(df["lrt_pvalue"], errors="coerce")
    df["neglog10_p"] = -np.log10(df["lrt_pvalue"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["midpoint", "neglog10_p"])

    rows = []
    for key, g in df.groupby(["variant", "reference", "midpoint"], dropna=False):
        best = g.sort_values("lrt_pvalue", ascending=True).iloc[0].copy()
        best["signal"] = classify_signal(g["drug"])
        best["drugs_combined"] = ";".join(sorted(set(g["drug"].astype(str))))
        best["neglog10_p"] = g["neglog10_p"].max()
        rows.append(best)

    return pd.DataFrame(rows)


def choose_references(df, max_refs=8, include=None):
    include = include or []
    refs = []
    for r in include:
        if r in set(df["reference"]):
            refs.append(r)

    tmp = (
        df.groupby("reference")
        .agg(n=("variant", "count"), max_sig=("neglog10_p", "max"))
        .reset_index()
        .sort_values(["max_sig", "n"], ascending=[False, False])
    )
    for r in tmp["reference"]:
        if r not in refs:
            refs.append(r)
        if len(refs) >= max_refs:
            break
    return refs


def plot(df, out_prefix, max_refs=8, include_refs=None, top_labels=12, title=None):
    df = make_collapsed(df)
    refs = choose_references(df, max_refs=max_refs, include=include_refs)

    plot_df = df[df["reference"].isin(refs)].copy()
    if plot_df.empty:
        raise ValueError("No mapped points available for selected references.")

    n_refs = len(refs)
    fig_h = max(3.2, 1.15 * n_refs + 1.3)
    fig, axes = plt.subplots(n_refs, 1, figsize=(12, fig_h), sharey=True)
    if n_refs == 1:
        axes = [axes]

    fig.patch.set_facecolor(BACKGROUND)

    ymax = max(8, plot_df["neglog10_p"].max() * 1.08)

    for ax, ref in zip(axes, refs):
        sub = plot_df[plot_df["reference"] == ref].copy().sort_values("midpoint")
        ax.set_facecolor(BACKGROUND)

        for signal in ["OTHER", "IMI", "MER", "BOTH"]:
            ss = sub[sub["signal"] == signal]
            if ss.empty:
                continue
            size = 18 if signal != "BOTH" else 28
            alpha = 0.78 if signal != "OTHER" else 0.45
            ax.scatter(
                ss["midpoint"],
                ss["neglog10_p"],
                s=size,
                c=signal_color(signal),
                alpha=alpha,
                edgecolor="none",
                label=signal,
                rasterized=True,
            )

        ax.axhline(-np.log10(1e-7), color=CHARCOAL, lw=0.8, ls="--", alpha=0.65)
        ax.set_ylabel(ref, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(CHARCOAL)
        ax.spines["bottom"].set_color(CHARCOAL)
        ax.tick_params(axis="both", labelsize=8, colors=CHARCOAL)
        ax.set_ylim(0, ymax)

        lab = sub.sort_values("neglog10_p", ascending=False).head(max(1, top_labels // n_refs + 1))
        for _, r in lab.iterrows():
            txt = label_text(r)
            if not txt:
                continue
            ax.text(
                r["midpoint"],
                r["neglog10_p"] + 0.35,
                txt,
                fontsize=7,
                ha="center",
                va="bottom",
                color=CHARCOAL,
                rotation=25,
            )

    axes[-1].set_xlabel("Genomic position within reference / plasmid", fontsize=10, color=CHARCOAL)
    fig.text(0.015, 0.5, r"$-\log_{10}(P)$", rotation=90, va="center", fontsize=11, color=CHARCOAL)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLOR_IMI, label="IMI only", markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLOR_MER, label="MER only", markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLOR_BOTH, label="IMI + MER", markersize=7),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", ncol=3, fontsize=9)

    if title:
        fig.suptitle(title, y=0.995, fontsize=13, color=CHARCOAL)

    fig.tight_layout(rect=[0.04, 0.02, 1, 0.98])

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix) + ".png", dpi=600, bbox_inches="tight", facecolor=BACKGROUND)
    fig.savefig(str(out_prefix) + ".svg", bbox_inches="tight", facecolor=BACKGROUND)
    print(f"[OK] Wrote {out_prefix}.png")
    print(f"[OK] Wrote {out_prefix}.svg")


def main():
    ap = argparse.ArgumentParser(description="Plot replicon-aware GWAS Manhattan from parsed mappings.")
    ap.add_argument("--mapped-hits", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-refs", type=int, default=8)
    ap.add_argument("--include-refs", nargs="*", default=["CP043953.1", "CP043954.1"])
    ap.add_argument("--top-labels", type=int, default=14)
    ap.add_argument("--title", default="Carbapenem-associated GWAS signals")
    ap.add_argument("--drop-bad-chisq", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.mapped_hits, sep="\t", low_memory=False)
    if args.drop_bad_chisq and "notes" in df.columns:
        df = df[~df["notes"].astype(str).str.contains("bad-chisq", na=False)]

    plot(
        df,
        out_prefix=args.out_prefix,
        max_refs=args.max_refs,
        include_refs=args.include_refs,
        top_labels=args.top_labels,
        title=args.title,
    )


if __name__ == "__main__":
    main()
