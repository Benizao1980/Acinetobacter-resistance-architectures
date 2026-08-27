#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("outputs/global_prediction/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_PREFIX = OUTDIR / "Figure_global_predictions_geography_time"

COL = {
    "IMI": "#E67E22",
    "MER": "#F4C542",
    "IMI+MER": "#C1121F",
    "grey": "#B8B8B8",
    "light_grey": "#E6E1D8",
    "dark": "#2D2A32",
    "cream": "#F7F1E3",
    "grid": "#D7CEC1",
}

def style_ax(ax):
    ax.set_facecolor(COL["cream"])
    ax.grid(axis="x", color=COL["grid"], lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(COL["dark"])
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=COL["dark"])

continent = pd.read_csv(
    "outputs/global_prediction/stratified/global_prediction_consensus_by_continent_inferred_n10.tsv",
    sep="\t"
)

decade = pd.read_csv(
    "outputs/global_prediction/stratified/global_prediction_consensus_by_decade_n10.tsv",
    sep="\t"
)

year = pd.read_csv(
    "outputs/global_prediction/stratified/global_prediction_consensus_time_trend_year_n10.tsv",
    sep="\t"
)

country = pd.read_csv(
    "outputs/global_prediction/stratified/global_prediction_consensus_by_country_n10.tsv",
    sep="\t"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.patch.set_facecolor("white")

# A continent
ax = axes[0, 0]
continent = continent.sort_values("any_consensus_prop")
y = np.arange(len(continent))
ax.barh(y, continent["imi_consensus_prop"], color=COL["IMI"], alpha=0.9, label="Imipenem")
ax.barh(y + 0.34, continent["mer_consensus_prop"], color=COL["MER"], alpha=0.9, label="Meropenem", height=0.34)
ax.set_yticks(y + 0.17)
ax.set_yticklabels([f"{r.Continent_inferred} (n={int(r.n)})" for _, r in continent.iterrows()])
ax.set_xlim(0, 1)
ax.set_xlabel("Consensus proportion predicted >8 mg/L")
ax.set_title("A  Predicted high-MIC burden by continent")
style_ax(ax)
ax.legend(frameon=False, fontsize=8, loc="lower right")

# B country top 12
ax = axes[0, 1]
country = country.sort_values("any_consensus_prop", ascending=False).head(12)
country = country.sort_values("any_consensus_prop")
y = np.arange(len(country))
ax.barh(y, country["any_consensus_prop"], color=COL["IMI+MER"], alpha=0.9)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.Country_clean} (n={int(r.n)})" for _, r in country.iterrows()])
ax.set_xlim(0, 1)
ax.set_xlabel("Any carbapenem consensus proportion >8 mg/L")
ax.set_title("B  Countries with highest predicted burden")
style_ax(ax)

# C decade
ax = axes[1, 0]
order = ["1990s", "2000s", "2010s", "2020s", "Unknown"]
decade["Decade"] = pd.Categorical(decade["Decade"], categories=order, ordered=True)
decade = decade.sort_values("Decade")
x = np.arange(len(decade))
w = 0.36
ax.bar(x - w/2, decade["imi_consensus_prop"], width=w, color=COL["IMI"], label="Imipenem")
ax.bar(x + w/2, decade["mer_consensus_prop"], width=w, color=COL["MER"], label="Meropenem")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.Decade}\n(n={int(r.n)})" for _, r in decade.iterrows()])
ax.set_ylim(0, 1)
ax.set_ylabel("Consensus proportion predicted >8 mg/L")
ax.set_title("C  Temporal enrichment by decade")
style_ax(ax)
ax.legend(frameon=False, fontsize=8)

# D year trend
ax = axes[1, 1]
year = year.sort_values("Year_num")
ax.plot(year["Year_num"], year["imi_consensus_prop"], marker="o", color=COL["IMI"], lw=1.8, label="Imipenem")
ax.plot(year["Year_num"], year["mer_consensus_prop"], marker="o", color=COL["MER"], lw=1.8, label="Meropenem")
ax.set_ylim(0, 1)
ax.set_xlabel("Sampling year, n ≥ 10")
ax.set_ylabel("Consensus proportion predicted >8 mg/L")
ax.set_title("D  Year-by-year trend in sampled genomes")
style_ax(ax)
ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Global distribution of predicted carbapenem high-MIC burden",
    fontsize=14,
    fontweight="bold",
    color=COL["dark"],
)

fig.text(
    0.01,
    0.01,
    "Consensus = at least two of three feature sets (AMR only, Locus-GWAS, Hybrid) predict MIC >8 mg/L. "
    "Temporal/geographic patterns reflect available sampled genomes, not incidence estimates.",
    fontsize=8,
    color=COL["dark"],
)

fig.subplots_adjust(top=0.90, bottom=0.10, left=0.12, right=0.98, hspace=0.42, wspace=0.38)

for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT_PREFIX}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")

print("[OK] wrote", f"{OUT_PREFIX}.png/svg/pdf")
