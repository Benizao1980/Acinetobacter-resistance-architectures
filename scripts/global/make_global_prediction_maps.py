#!/usr/bin/env python3
"""
make_global_prediction_maps.py

Builds a true-vector SVG figure from BAMPS global prediction outputs.

Panels:
A. Global predicted high-MIC burden map
B. Change in predicted high-MIC burden since 2010
C. Country-level burden + resistance architecture composition

Inputs expected:
  - global_predictions_enriched.with_consensus.tsv
  - global_prediction_consensus_by_country_n10.tsv
  - Natural Earth countries shapefile

Example:
python make_global_prediction_maps.py \
  --enriched global_prediction/stratified/global_predictions_enriched.with_consensus.tsv \
  --country-summary global_prediction/stratified/global_prediction_consensus_by_country_n10.tsv \
  --shapefile /path/to/naturalearth_lowres.shp \
  --out-prefix outputs/figures/global_prediction_architecture
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--enriched", required=True,
                   help="Per-genome prediction table with consensus flags and AMRFinder mechanism.")
    p.add_argument("--country-summary", required=True,
                   help="Country-level consensus prediction summary, usually n>=10.")
    p.add_argument("--shapefile", default=None,
                   help="Natural Earth countries shapefile.")
    p.add_argument("--out-prefix", required=True,
                   help="Output prefix for svg/pdf/png and tables.")
    p.add_argument("--split-year", type=int, default=2010,
                   help="Year used to compare earlier vs later burden.")
    p.add_argument("--min-period-n", type=int, default=5,
                   help="Minimum genomes before and after split year for temporal-change map.")
    p.add_argument("--top-countries", type=int, default=10,
                   help="Number of highest-burden countries to show in architecture panel.")
    p.add_argument("--map-unknown-to-russia", action="store_true",
                   help="Map Country=Unknown/NA summary row to Russia. Use only if verified.")
    return p.parse_args()


def clean_mechanism(x):
    x = str(x)
    if "OXA-23" in x:
        return "OXA-23-like"
    if "OXA-24/40" in x or "OXA-72" in x:
        return "OXA-72/OXA-24/40-like"
    if x == "MBL" or "NDM" in x or "VIM" in x or "IMP" in x:
        return "MBL"
    if "GES" in x:
        return "GES-like"
    if "OXA-58" in x:
        return "OXA-58-like"
    if "Intrinsic" in x or "OXA-51" in x:
        return "No recognised carbapenemase"
    return "Other / mixed"


def harmonise_country_column(df, map_unknown_to_russia=False):
    df = df.copy()
    if "Country_clean" not in df.columns:
        if "Country" in df.columns:
            df["Country_clean"] = df["Country"]
        elif "Country2" in df.columns:
            df["Country_clean"] = df["Country2"]
        else:
            raise ValueError("No Country/Country2/Country_clean column found.")

    df["Country_clean"] = df["Country_clean"].astype("object")
    if map_unknown_to_russia:
        df["Country_clean"] = df["Country_clean"].fillna("Russia")
        df.loc[df["Country_clean"].astype(str).isin(["Unknown", "nan", "None", ""]), "Country_clean"] = "Russia"
    else:
        df.loc[df["Country_clean"].astype(str).isin(["nan", "None", ""]), "Country_clean"] = np.nan

    df["map_name"] = df["Country_clean"].replace({
        "USA": "United States of America",
        "UK": "United Kingdom",
        "Czech Republic": "Czechia",
        "The Netherlands": "Netherlands",
        "Russia": "Russia",
    })
    return df


def main():
    args = parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    enriched = pd.read_csv(args.enriched, sep="\t", low_memory=False)
    country = pd.read_csv(args.country_summary, sep="\t")

    # Build country column in enriched.
    if "Country_clean" not in enriched.columns:
        enriched["Country_clean"] = enriched.get("Country")
        if "Country2" in enriched.columns:
            enriched["Country_clean"] = enriched["Country_clean"].fillna(enriched["Country2"])

    enriched = harmonise_country_column(enriched, args.map_unknown_to_russia)
    country = harmonise_country_column(country, args.map_unknown_to_russia)

    # Consensus any-carbapenem high-MIC flag.
    enriched["any_consensus_gt8"] = (
        enriched["imipenem_consensus_gt8"].astype(bool)
        | enriched["meropenem_consensus_gt8"].astype(bool)
    )

    # Mechanism classes.
    enriched["mechanism_group"] = enriched["amrfinder_carbapenemase_group"].apply(clean_mechanism)

    # Burden summary.
    burden = country.rename(columns={"any_consensus_prop": "burden_prop"}).copy()

    # Time-change summary.
    time_df = enriched.dropna(subset=["Country_clean", "Year"]).copy()
    time_df["Year"] = pd.to_numeric(time_df["Year"], errors="coerce")
    time_df = time_df.dropna(subset=["Year"])
    time_df["period"] = np.where(time_df["Year"] < args.split_year, "before", "since")

    time_summary = (
        time_df.groupby(["Country_clean", "period"])
        .agg(n=("sample", "count"), prop=("any_consensus_gt8", "mean"))
        .reset_index()
    )

    wide = time_summary.pivot(index="Country_clean", columns="period", values=["n", "prop"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for col in ["n_before", "n_since", "prop_before", "prop_since"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["rise_since_split"] = wide["prop_since"] - wide["prop_before"]
    wide["has_rise_data"] = (
        (wide["n_before"] >= args.min_period_n)
        & (wide["n_since"] >= args.min_period_n)
    )
    rise = wide[wide["has_rise_data"]].copy()
    rise = harmonise_country_column(rise, False)

    # Architecture composition for top countries.
    top_countries = (
        burden.sort_values("burden_prop", ascending=False)
        .head(args.top_countries)["Country_clean"]
        .dropna()
        .tolist()
    )

    mech_counts = (
        enriched[enriched["Country_clean"].isin(top_countries)]
        .groupby(["Country_clean", "mechanism_group"])
        .size()
        .reset_index(name="count")
    )
    mech_totals = mech_counts.groupby("Country_clean")["count"].sum().rename("total").reset_index()
    mech_counts = mech_counts.merge(mech_totals, on="Country_clean")
    mech_counts["prop"] = mech_counts["count"] / mech_counts["total"]

    bar_df = burden[burden["Country_clean"].isin(top_countries)][
        ["Country_clean", "n", "burden_prop"]
    ].merge(mech_counts, on="Country_clean", how="left")
    bar_df["segment_height"] = bar_df["burden_prop"] * bar_df["prop"]

    # Geometry.
    if args.shapefile:
        world = gpd.read_file(args.shapefile)
    else:
        world = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        )
        if "ADMIN" in world.columns and "name" not in world.columns:
            world = world.rename(columns={"ADMIN": "name"})
    world_burden = world.merge(burden, left_on="name", right_on="map_name", how="left")
    world_rise = world.merge(rise, left_on="name", right_on="map_name", how="left")

    # Palettes matched to manuscript.
    burden_cmap = LinearSegmentedColormap.from_list(
        "burden_paper",
        [
            (0.00, "#FFFFFF"),
            (0.10, "#F1D2CA"),
            (0.35, "#CC7A69"),
            (0.65, "#B52A1F"),
            (1.00, "#7F1D13"),
        ],
    )

    rise_cmap = LinearSegmentedColormap.from_list(
        "rise_paper",
        [
            (0.00, "#2F5D62"),
            (0.50, "#FFFFFF"),
            (0.72, "#CC7A69"),
            (1.00, "#7F1D13"),
        ],
    )

    mech_colours = {
        "OXA-23-like": "#7F1D13",
        "OXA-72/OXA-24/40-like": "#CC7A69",
        "MBL": "#5F0F40",
        "GES-like": "#B52A1F",
        "OXA-58-like": "#D99873",
        "Other / mixed": "#7A687F",
        "No recognised carbapenemase": "#D8D8D8",
    }
    mech_order = [
        "OXA-23-like",
        "OXA-72/OXA-24/40-like",
        "MBL",
        "GES-like",
        "OXA-58-like",
        "Other / mixed",
        "No recognised carbapenemase",
    ]

    # Figure.
    fig = plt.figure(figsize=(13, 8.2))
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.25], hspace=0.20, wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax in [ax1, ax2]:
        ax.set_facecolor("none")
        ax.set_axis_off()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)

    # A. Burden map.
    world_burden.plot(ax=ax1, color="#FFFFFF", edgecolor="#D5D5D5", linewidth=0.22)
    world_burden[world_burden["burden_prop"].notna()].plot(
        ax=ax1,
        column="burden_prop",
        cmap=burden_cmap,
        vmin=0,
        vmax=1,
        edgecolor="#666666",
        linewidth=0.34,
    )
    ax1.set_title("A  Predicted high-MIC burden", loc="left", fontsize=14, fontweight="bold")

    # Add Singapore point if present because low-res Natural Earth misses it.
    sg = burden[burden["Country_clean"].eq("Singapore")]
    if len(sg):
        ax1.scatter(
            103.82, 1.35,
            s=28,
            color=burden_cmap(float(sg["burden_prop"].iloc[0])),
            edgecolor="#444444",
            linewidth=0.5,
            zorder=5,
        )

    # B. Rise map.
    world_rise.plot(ax=ax2, color="#FFFFFF", edgecolor="#D5D5D5", linewidth=0.22)
    if len(world_rise[world_rise["rise_since_split"].notna()]):
        world_rise[world_rise["rise_since_split"].notna()].plot(
            ax=ax2,
            column="rise_since_split",
            cmap=rise_cmap,
            norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.8),
            edgecolor="#666666",
            linewidth=0.34,
        )
    ax2.set_title(
        f"B  Change since {args.split_year}",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    # C. Country burden + mechanism composition.
    ax3.set_facecolor("none")
    bar_countries = burden[burden["Country_clean"].isin(top_countries)].sort_values(
        "burden_prop", ascending=False
    )
    countries = bar_countries["Country_clean"].tolist()
    x = np.arange(len(countries))
    bottom = np.zeros(len(countries))

    for mech in mech_order:
        vals = []
        for c in countries:
            row = bar_df[(bar_df["Country_clean"] == c) & (bar_df["mechanism_group"] == mech)]
            vals.append(float(row["segment_height"].sum()) if len(row) else 0)
        vals = np.array(vals)
        if vals.sum() > 0:
            ax3.bar(
                x,
                vals,
                bottom=bottom,
                width=0.72,
                color=mech_colours[mech],
                edgecolor="white",
                linewidth=0.5,
                label=mech,
            )
            bottom += vals

    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Predicted high-MIC burden", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [
            f"{c}\n(n={int(bar_countries[bar_countries['Country_clean'] == c]['n'].iloc[0])})"
            for c in countries
        ],
        rotation=0,
        fontsize=8,
    )
    ax3.set_title(
        "C  Mechanism composition among highest-burden countries",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax3.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")

    # Colorbars.
    cax1 = fig.add_axes([0.43, 0.535, 0.012, 0.28])
    sm1 = ScalarMappable(norm=Normalize(0, 1), cmap=burden_cmap)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cax1)
    cb1.set_label("Burden", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    cax2 = fig.add_axes([0.92, 0.535, 0.012, 0.28])
    sm2 = ScalarMappable(norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.8), cmap=rise_cmap)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cax2)
    cb2.set_label("Δ burden", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Global predictions suggest geographically structured carbapenem resistance burden and architecture",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight", transparent=True)

    rise.to_csv(f"{out_prefix}.rise_since_{args.split_year}.tsv", sep="\t", index=False)
    bar_df.to_csv(f"{out_prefix}.mechanism_bars.tsv", sep="\t", index=False)

    print("[OK] wrote:")
    print(f"{out_prefix}.svg")
    print(f"{out_prefix}.pdf")
    print(f"{out_prefix}.png")
    print(f"{out_prefix}.rise_since_{args.split_year}.tsv")
    print(f"{out_prefix}.mechanism_bars.tsv")


if __name__ == "__main__":
    main()
