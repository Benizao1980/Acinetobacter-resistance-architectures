#!/usr/bin/env python3

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# OUTPUTS
# ============================================================

OUTDIR = Path("outputs/figures")
TABLEDIR = Path("outputs/figure3_model_comparison/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)
TABLEDIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUTDIR / "Figure3_final"

# ============================================================
# WES-ISH / UPDATED COLOUR SCHEME
# ============================================================
# IMI and MER are kept visually distinct.
# IMI+MER gets its own combined colour rather than being treated
# as just another antibiotic class.

COL = {
    "IMI": "#E67E22",       # burnt orange
    "MER": "#F4C542",       # mustard yellow
    "IMI+MER": "#C1121F",   # carbapenem red
    "grey": "#B8B8B8",
    "light_grey": "#E6E1D8",
    "dark": "#2D2A32",
    "cream": "#F7F1E3",
    "white": "#FFFFFF",
    "grid": "#D7CEC1",
    "highlight": "#1F1F1F",
}

ANTIBIOTIC_LABELS = {
    "imipenem": "Imipenem",
    "meropenem": "Meropenem",
}

ANTIBIOTIC_SHORT = {
    "imipenem": "IMI",
    "meropenem": "MER",
}

# ============================================================
# MODEL CHOICE FOR MAIN FIGURE
# ============================================================
# Keep this clean: AMR-only vs locus-level GWAS vs hybrid.
# Hybrid here = selected AMR + full locus-level GWAS, based on the
# current strongest/clearest comparison.

MODELS = {
    "AMR only": {
        "run": "outputs/runs_rebuilt/01_amr_only",
        "label": "AMR only",
    },
    "Locus-GWAS": {
        "run": "outputs/runs_rebuilt/04_locus_gwas",
        "label": "Locus-GWAS",
    },
    "Hybrid": {
        "run": "outputs/runs_rebuilt/07_amr_selected_plus_locus_full",
        "label": "Hybrid",
    },
}

ANTIBIOTICS = ["imipenem", "meropenem"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_feature_id(x):
    """Remove model prefixes and tidy feature/locus IDs."""
    x = str(x)
    for prefix in [
        "LOCUS__", "LOCUSSEL__", "GWAS__", "AMR__", "AMRSEL__",
        "feature=", "locus=", "gene="
    ]:
        x = x.replace(prefix, "")
    return x.strip()


def find_col(df, candidates):
    """Find a column by partial matching."""
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    for cand in candidates:
        cand = cand.lower()
        for c, lc in lower.items():
            if cand == lc:
                return c

    for cand in candidates:
        cand = cand.lower()
        for c, lc in lower.items():
            if cand in lc:
                return c

    raise ValueError(
        f"Could not find column matching {candidates}. "
        f"Available columns: {cols}"
    )


def read_prediction_file(path):
    """Read self-test prediction table and infer true/prediction columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    pred_col = find_col(df, [
        "prediction", "pred", "y_pred", "pred_log2", "predicted"
    ])
    true_col = find_col(df, [
        "truth", "true", "observed", "y_true", "true_log2", "actual"
    ])

    d = df.copy()
    d["pred_log2"] = pd.to_numeric(d[pred_col], errors="coerce")
    d["true_log2"] = pd.to_numeric(d[true_col], errors="coerce")
    d = d.dropna(subset=["pred_log2", "true_log2"])

    d["residual"] = d["pred_log2"] - d["true_log2"]
    d["abs_error"] = d["residual"].abs()
    return d


def calc_metrics(df):
    """Metrics in log2 MIC dilution space."""
    err = df["residual"]
    abs_err = err.abs()

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((df["true_log2"] - df["true_log2"].mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "n": int(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2),
        "within_1_dilution": float((abs_err <= 1).mean()),
        "within_2_dilution": float((abs_err <= 2).mean()),
    }


def load_all_predictions():
    data = {}
    rows = []

    for model_name, meta in MODELS.items():
        run = Path(meta["run"])
        data[model_name] = {}

        for ab in ANTIBIOTICS:
            pred_path = run / f"{ab}_self_test_predictions.tsv"
            d = read_prediction_file(pred_path)
            data[model_name][ab] = d

            m = calc_metrics(d)
            rows.append({
                "model": model_name,
                "antibiotic": ab,
                **m,
                "source_file": str(pred_path),
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLEDIR / "Figure3_final_metrics.tsv", sep="\t", index=False)
    return data, metrics


def load_model_loci():
    """Load loci used in the locus/hybrid model for Manhattan highlighting."""
    candidate_files = [
        "outputs/runs_rebuilt/04_locus_gwas/feature_columns.tsv",
        "outputs/runs_rebuilt/07_amr_selected_plus_locus_full/feature_columns.tsv",
        "outputs/runs_rebuilt/09_amr_selected_plus_locus_selected/feature_columns.tsv",
    ]

    loci = set()

    for fp in candidate_files:
        p = Path(fp)
        if not p.exists():
            continue

        df = pd.read_csv(p, sep="\t")
        if "feature" not in df.columns:
            continue

        for f in df["feature"].dropna():
            fs = str(f)
            if fs.startswith("LOCUS__") or fs.startswith("LOCUSSEL__") or fs.startswith("GWAS__"):
                loci.add(clean_feature_id(fs))

    return loci


def classify_gwas_signal(drugs):
    drugs = set(str(x).upper() for x in drugs if pd.notna(x))
    has_imi = any("IMI" in d or "IMIPENEM" in d for d in drugs)
    has_mer = any("MER" in d or "MEROPENEM" in d for d in drugs)

    if has_imi and has_mer:
        return "IMI+MER"
    if has_imi:
        return "IMI"
    if has_mer:
        return "MER"
    return "Other"


def load_gwas_for_manhattan():
    """Create a robust Manhattan-style table from integrated GWAS loci."""
    gwas_file = Path("outputs/gwas_integrated/gwas_loci_mapped_clean.tsv")

    if not gwas_file.exists():
        print(f"[WARN] Missing {gwas_file}; Manhattan panel will be empty.")
        return pd.DataFrame()

    g = pd.read_csv(gwas_file, sep="\t")
    g.columns = [c.strip() for c in g.columns]

    required = {"drug", "locus_id", "pvalue"}
    missing = required - set(g.columns)
    if missing:
        print(f"[WARN] GWAS file missing required columns: {missing}")
        return pd.DataFrame()

    g["pvalue"] = pd.to_numeric(g["pvalue"], errors="coerce")
    g = g.dropna(subset=["pvalue"])
    g = g[g["pvalue"] > 0].copy()

    if "reference" not in g.columns:
        g["reference"] = "Unknown"
    if "replicon" not in g.columns:
        g["replicon"] = "genomic context"

    g["locus_clean"] = g["locus_id"].map(clean_feature_id)

    # Collapse duplicate rows per locus/reference/replicon and classify drug signal.
    collapsed = (
        g.groupby(["locus_clean", "reference", "replicon"], dropna=False)
         .agg(
             pvalue=("pvalue", "min"),
             drugs=("drug", lambda x: ";".join(sorted(set(map(str, x))))),
         )
         .reset_index()
    )

    collapsed["signal"] = collapsed["drugs"].str.split(";").map(classify_gwas_signal)
    collapsed["neglog10p"] = -np.log10(collapsed["pvalue"])

    # Arrange x-axis by broad genomic context, then reference, then p-value.
    replicon_order = {
        "chromosome": 0,
        "plasmid": 1,
        "pangenome": 2,
        "unmapped": 3,
    }

    collapsed["replicon_key"] = (
        collapsed["replicon"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .map(lambda x: replicon_order.get(x, 9))
    )

    collapsed = collapsed.sort_values(
        ["replicon_key", "reference", "pvalue", "locus_clean"]
    ).reset_index(drop=True)

    collapsed["x"] = np.arange(len(collapsed))

    model_loci = load_model_loci()
    collapsed["in_model"] = collapsed["locus_clean"].isin(model_loci)

    # Fallback fuzzy highlight: useful if IDs differ by prefixes/versions.
    if collapsed["in_model"].sum() == 0 and model_loci:
        model_nover = {m.split(".")[0] for m in model_loci}
        collapsed["in_model"] = collapsed["locus_clean"].map(
            lambda x: str(x).split(".")[0] in model_nover
        )

    collapsed.to_csv(
        TABLEDIR / "Figure3_final_manhattan_plot_data.tsv",
        sep="\t",
        index=False
    )

    print(
        f"[INFO] GWAS loci for Manhattan: {len(collapsed)}; "
        f"model-highlighted: {collapsed['in_model'].sum()}"
    )

    return collapsed


def style_ax(ax):
    ax.set_facecolor(COL["cream"])
    ax.grid(True, color=COL["grid"], linewidth=0.6, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL["dark"])
        ax.spines[spine].set_linewidth(0.8)


# ============================================================
# PLOTTING
# ============================================================

def plot_panel_a(fig, gs, data):
    """Prediction scatterplots."""
    model_names = list(MODELS.keys())

    for col_i, model in enumerate(model_names):
        for row_i, ab in enumerate(ANTIBIOTICS):
            ax = fig.add_subplot(gs[row_i, col_i])
            style_ax(ax)

            d = data[model][ab]
            short = ANTIBIOTIC_SHORT[ab]
            colour = COL[short]

            lo = min(d["true_log2"].min(), d["pred_log2"].min()) - 0.5
            hi = max(d["true_log2"].max(), d["pred_log2"].max()) + 0.5

            # ±2 and ±1 dilution bands
            ax.fill_between(
                [lo, hi],
                [lo - 2, hi - 2],
                [lo + 2, hi + 2],
                color=COL["light_grey"],
                alpha=0.65,
                zorder=0,
                linewidth=0,
            )
            ax.fill_between(
                [lo, hi],
                [lo - 1, hi - 1],
                [lo + 1, hi + 1],
                color=COL["grey"],
                alpha=0.35,
                zorder=1,
                linewidth=0,
            )

            ax.scatter(
                d["true_log2"],
                d["pred_log2"],
                s=16,
                alpha=0.72,
                color=colour,
                edgecolor=COL["white"],
                linewidth=0.25,
                zorder=3,
            )

            ax.plot([lo, hi], [lo, hi], "--", color=COL["dark"], lw=1.0, zorder=4)

            m = calc_metrics(d)
            ax.text(
                0.04, 0.96,
                f"n={m['n']}\nMAE={m['mae']:.2f}\n±1={m['within_1_dilution']*100:.1f}%\n±2={m['within_2_dilution']*100:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=COL["dark"],
                bbox=dict(
                    facecolor=COL["white"],
                    edgecolor="none",
                    alpha=0.78,
                    boxstyle="round,pad=0.25"
                )
            )

            if row_i == 0:
                ax.set_title(model, fontsize=10.5, fontweight="bold", color=COL["dark"])

            if col_i == 0:
                ax.set_ylabel(
                    f"{ANTIBIOTIC_LABELS[ab]}\nPredicted MIC (log₂)",
                    fontsize=9,
                    color=COL["dark"]
                )

            if row_i == 1:
                ax.set_xlabel("Observed MIC (log₂)", fontsize=9, color=COL["dark"])

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=8, colors=COL["dark"])

    fig.text(
        0.015, 0.965,
        "A",
        fontsize=16,
        fontweight="bold",
        color=COL["dark"]
    )


def plot_panel_b(fig, gs):
    """Overlapping Manhattan-style GWAS plot."""
    ax = fig.add_subplot(gs[2, :])
    style_ax(ax)

    g = load_gwas_for_manhattan()

    if g.empty:
        ax.text(
            0.5, 0.5,
            "GWAS loci unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=COL["dark"]
        )
        ax.set_axis_off()
        return

    # Background points first
    for signal in ["IMI", "MER", "IMI+MER"]:
        sub = g[g["signal"] == signal]
        if sub.empty:
            continue

        ax.scatter(
            sub["x"],
            sub["neglog10p"],
            s=10 if signal != "IMI+MER" else 16,
            color=COL[signal],
            alpha=0.72 if signal != "IMI+MER" else 0.95,
            edgecolor="none",
            label=signal,
            zorder=2 if signal != "IMI+MER" else 3,
        )

    # Highlight model loci
    hm = g[g["in_model"]].copy()
    if not hm.empty:
        ax.scatter(
            hm["x"],
            hm["neglog10p"],
            s=44,
            facecolor="none",
            edgecolor=COL["highlight"],
            linewidth=1.1,
            zorder=5,
            label="Model loci",
        )

        # Label only a few highest-confidence model loci to avoid clutter.
        label_df = hm.sort_values("neglog10p", ascending=False).head(10)
        for _, r in label_df.iterrows():
            ax.text(
                r["x"],
                r["neglog10p"] + 0.25,
                str(r["locus_clean"]),
                fontsize=6.5,
                rotation=35,
                ha="left",
                va="bottom",
                color=COL["dark"],
                zorder=6,
            )

    # Significance guide line
    ax.axhline(
        -math.log10(1e-7),
        color=COL["dark"],
        linestyle=":",
        lw=1.0,
        alpha=0.8
    )

    # Broad context separators
    context = g.groupby("replicon", dropna=False)["x"].agg(["min", "max"]).reset_index()
    for _, r in context.iterrows():
        mid = (r["min"] + r["max"]) / 2
        ax.text(
            mid,
            -0.12,
            str(r["replicon"]).capitalize(),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=COL["dark"]
        )
        ax.axvline(r["max"] + 0.5, color=COL["grid"], lw=0.8, alpha=0.8)

    ax.set_ylabel("−log₁₀(P)", fontsize=9, color=COL["dark"])
    ax.set_xlabel("Genomic context", fontsize=9, color=COL["dark"])
    ax.set_xticks([])
    ax.tick_params(labelsize=8, colors=COL["dark"])

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["IMI"],
               markersize=6, label="IMI only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["MER"],
               markersize=6, label="MER only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["IMI+MER"],
               markersize=6, label="IMI+MER"),
    ]

    if not hm.empty:
        legend_handles.append(
            Line2D([0], [0], marker="o", color=COL["highlight"], markerfacecolor="none",
                   markersize=7, label="Model loci")
        )

    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=4,
        loc="upper right",
        fontsize=8
    )

    fig.text(
        0.015, 0.50,
        "B",
        fontsize=16,
        fontweight="bold",
        color=COL["dark"]
    )


def plot_panel_c(fig, gs, metrics):
    """Metric comparison: MAE, ±1, ±2."""
    metric_specs = [
        ("mae", "MAE\n(log₂ dilutions)", False),
        ("within_1_dilution", "Within ±1\ndilution (%)", True),
        ("within_2_dilution", "Within ±2\ndilutions (%)", True),
    ]

    model_names = list(MODELS.keys())
    x = np.arange(len(model_names))
    width = 0.36

    for i, (metric, ylabel, as_percent) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[3, i])
        style_ax(ax)

        imi_vals = []
        mer_vals = []

        for model in model_names:
            imi = metrics[
                (metrics["model"] == model) &
                (metrics["antibiotic"] == "imipenem")
            ][metric].iloc[0]

            mer = metrics[
                (metrics["model"] == model) &
                (metrics["antibiotic"] == "meropenem")
            ][metric].iloc[0]

            if as_percent:
                imi *= 100
                mer *= 100

            imi_vals.append(imi)
            mer_vals.append(mer)

        ax.bar(
            x - width / 2,
            imi_vals,
            width,
            color=COL["IMI"],
            edgecolor=COL["dark"],
            linewidth=0.4,
            label="Imipenem",
        )
        ax.bar(
            x + width / 2,
            mer_vals,
            width,
            color=COL["MER"],
            edgecolor=COL["dark"],
            linewidth=0.4,
            label="Meropenem",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9, color=COL["dark"])

        if as_percent:
            ax.set_ylim(0, 100)
        else:
            ymax = max(max(imi_vals), max(mer_vals)) * 1.25
            ax.set_ylim(0, ymax)

        ax.tick_params(labelsize=8, colors=COL["dark"])

        if i == 2:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.text(
        0.015, 0.255,
        "C",
        fontsize=16,
        fontweight="bold",
        color=COL["dark"]
    )


def main():
    data, metrics = load_all_predictions()

    print("\n[METRICS]")
    print(metrics[[
        "model", "antibiotic", "n", "mae", "rmse", "r2",
        "within_1_dilution", "within_2_dilution"
    ]].to_string(index=False))

    # Save a clean wide table as well.
    metrics_wide = metrics.copy()
    metrics_wide["within_1_dilution_pct"] = metrics_wide["within_1_dilution"] * 100
    metrics_wide["within_2_dilution_pct"] = metrics_wide["within_2_dilution"] * 100
    metrics_wide.to_csv(TABLEDIR / "Figure3_final_metrics_clean.tsv", sep="\t", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": COL["dark"],
        "xtick.color": COL["dark"],
        "ytick.color": COL["dark"],
        "figure.facecolor": COL["white"],
        "savefig.facecolor": COL["white"],
    })

    fig = plt.figure(figsize=(13.2, 15.0), facecolor=COL["white"])

    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.05, 1.05, 1.15, 0.85],
        hspace=0.55,
        wspace=0.35,
    )

    plot_panel_a(fig, gs, data)
    plot_panel_b(fig, gs)
    plot_panel_c(fig, gs, metrics)

    fig.suptitle(
        "Genomic loci refine carbapenem MIC prediction in Acinetobacter baumannii",
        fontsize=15,
        fontweight="bold",
        color=COL["dark"],
        y=0.995,
    )

    fig.savefig(f"{OUT_PREFIX}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")

    print(f"\n[OK] Wrote:")
    print(f"  {OUT_PREFIX}.png")
    print(f"  {OUT_PREFIX}.svg")
    print(f"  {OUT_PREFIX}.pdf")
    print(f"  {TABLEDIR / 'Figure3_final_metrics.tsv'}")
    print(f"  {TABLEDIR / 'Figure3_final_manhattan_plot_data.tsv'}")


if __name__ == "__main__":
    main()