#!/usr/bin/env python3

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# OUTPUTS
# ============================================================

OUTDIR = Path("outputs/figures")
TABLEDIR = Path("outputs/figure3_ml_validation/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)
TABLEDIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUTDIR / "Figure3_ML_selftest_portability"

# ============================================================
# INPUTS
# ============================================================

SELFTEST_RUNS = {
    "amr": {
        "label": "AMR only",
        "imipenem": "outputs/runs_tuned_locked_xgb_real_final/01_amr_only_xgb_tuned_imipenem/imipenem_self_test_predictions.tsv",
        "meropenem": "outputs/runs_tuned_locked_xgb_real_final/01_amr_only_xgb_tuned_meropenem/meropenem_self_test_predictions.tsv",
    },
    "locus": {
        "label": "Locus-GWAS",
        "imipenem": "outputs/runs_tuned_locked_xgb_real_final/02_locus_gwas_xgb_tuned_imipenem/imipenem_self_test_predictions.tsv",
        "meropenem": "outputs/runs_tuned_locked_xgb_real_final/02_locus_gwas_xgb_tuned_meropenem/meropenem_self_test_predictions.tsv",
    },
    "hybrid": {
        "label": "Hybrid",
        "imipenem": "outputs/runs_tuned_locked_xgb_real_final/03_hybrid_xgb_tuned_imipenem/imipenem_self_test_predictions.tsv",
        "meropenem": "outputs/runs_tuned_locked_xgb_real_final/03_hybrid_xgb_tuned_meropenem/meropenem_self_test_predictions.tsv",
    },
}

EXTERNAL_ERROR_FP = Path(
    "outputs/error_characterisation/validation_prediction_error_table.with_microreact_ic.tsv"
)

MICROREACT_FP = Path(
    "data/FullMicroreactWyr-with-Russian-Metadata_IC_inferred.csv"
)

# ============================================================
# STYLE
# ============================================================

COL = {
    "IMI": "#E67E22",       # burnt orange
    "MER": "#F4C542",       # mustard yellow
    "IMI+MER": "#C1121F",   # combined carbapenem red
    "dark": "#2D2A32",
    "cream": "#F7F1E3",
    "pale": "#FBF7EE",
    "grid": "#D7CEC1",
    "grey": "#B8B8B8",
    "light_grey": "#E6E1D8",
    "white": "#FFFFFF",
}

MODEL_ORDER = ["amr", "locus", "hybrid"]
AB_ORDER = ["imipenem", "meropenem"]

MODEL_LABEL = {
    "amr": "AMR only",
    "locus": "Locus-GWAS",
    "hybrid": "Hybrid",
}

AB_LABEL = {
    "imipenem": "Imipenem",
    "meropenem": "Meropenem",
}

AB_SHORT = {
    "imipenem": "IMI",
    "meropenem": "MER",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "wes_error",
    [COL["pale"], COL["MER"], COL["IMI"], COL["IMI+MER"]]
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": COL["white"],
    "savefig.facecolor": COL["white"],
    "axes.facecolor": COL["cream"],
    "axes.edgecolor": COL["dark"],
    "axes.labelcolor": COL["dark"],
    "xtick.color": COL["dark"],
    "ytick.color": COL["dark"],
    "text.color": COL["dark"],
})

# ============================================================
# HELPERS
# ============================================================

def style_ax(ax, grid=True):
    ax.set_facecolor(COL["cream"])
    if grid:
        ax.grid(True, color=COL["grid"], linewidth=0.6, alpha=0.70)
    else:
        ax.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL["dark"])
        ax.spines[spine].set_linewidth(0.8)


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for cand in candidates:
        cand = cand.lower()
        for c in df.columns:
            if cand in c.lower():
                return c

    raise ValueError(f"Could not find column from {candidates}. Columns: {df.columns.tolist()}")


def clean_st(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none"}:
        return np.nan

    x = re.sub(r"^ST", "", x, flags=re.I)

    if re.fullmatch(r"\d+\.0", x):
        x = str(int(float(x)))

    return x


def clean_group(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none", "unassigned", "ua"}:
        return np.nan

    return x


def calc_metrics(df):
    err = pd.to_numeric(df["residual"], errors="coerce")
    abs_err = err.abs()

    ss_res = float(np.nansum(err ** 2))
    ss_tot = float(np.nansum((df["true_log2"] - df["true_log2"].mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "n": int(len(df)),
        "mae_log2": float(abs_err.mean()),
        "rmse_log2": float(np.sqrt(np.nanmean(err ** 2))),
        "r2": float(r2),
        "within_1": float((abs_err <= 1).mean()),
        "within_2": float((abs_err <= 2).mean()),
        "mean_signed_error": float(err.mean()),
    }


def load_selftest_prediction(fp):
    fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    pred_col = find_col(df, ["prediction", "pred", "y_pred", "predicted"])
    true_col = find_col(df, ["truth", "true", "observed", "actual", "y_true"])

    out = df.copy()
    out["pred_log2"] = pd.to_numeric(out[pred_col], errors="coerce")
    out["true_log2"] = pd.to_numeric(out[true_col], errors="coerce")
    out = out.dropna(subset=["pred_log2", "true_log2"]).copy()
    out["residual"] = out["pred_log2"] - out["true_log2"]
    out["abs_log2_error"] = out["residual"].abs()

    return out


def load_selftest_data():
    data = {}
    rows = []

    for model in MODEL_ORDER:
        data[model] = {}
        for ab in AB_ORDER:
            fp = SELFTEST_RUNS[model][ab]
            d = load_selftest_prediction(fp)
            data[model][ab] = d

            rows.append({
                "dataset": "self_test",
                "model": model,
                "model_label": MODEL_LABEL[model],
                "antibiotic": ab,
                **calc_metrics(d),
                "source_file": fp,
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLEDIR / "figure3_selftest_metrics.tsv", sep="\t", index=False)

    return data, metrics


def build_st_to_hbaps_map():
    if not MICROREACT_FP.exists():
        return {}, {}

    micro = pd.read_csv(MICROREACT_FP, dtype=str, low_memory=False)
    micro.columns = [c.strip() for c in micro.columns]

    if "hBAPS" not in micro.columns:
        return {}, {}

    def make_map(st_col):
        if st_col not in micro.columns:
            return {}

        tmp = micro[[st_col, "hBAPS"]].copy()
        tmp[st_col] = tmp[st_col].map(clean_st)
        tmp["hBAPS"] = tmp["hBAPS"].map(clean_group)
        tmp = tmp[tmp[st_col].notna() & tmp["hBAPS"].notna()].copy()

        if tmp.empty:
            return {}

        counts = (
            tmp.groupby([st_col, "hBAPS"])
            .size()
            .reset_index(name="n_group")
        )
        totals = (
            tmp.groupby(st_col)
            .size()
            .reset_index(name="n_total")
        )

        ref = counts.merge(totals, on=st_col, how="left")
        ref["prop"] = ref["n_group"] / ref["n_total"]
        ref = ref.sort_values([st_col, "prop", "n_group"], ascending=[True, False, False])

        best = ref.groupby(st_col, as_index=False).head(1).copy()

        # Conservative enough for plotting substructure; avoids very noisy singleton mappings.
        best = best[(best["n_total"] >= 5) & (best["prop"] >= 0.80)].copy()

        return best.set_index(st_col)["hBAPS"].to_dict()

    return make_map("ST_Pasteur"), make_map("ST_Oxford")


def load_external_data():
    if not EXTERNAL_ERROR_FP.exists():
        raise FileNotFoundError(EXTERNAL_ERROR_FP)

    df = pd.read_csv(EXTERNAL_ERROR_FP, sep="\t", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df["model"] = df["model"].astype(str).str.lower().str.strip()
    df["antibiotic"] = df["antibiotic"].astype(str).str.lower().str.strip()

    # Prediction/observed log2 values
    if "log2_true_mic" in df.columns:
        df["true_log2"] = pd.to_numeric(df["log2_true_mic"], errors="coerce")
    elif "true_log2" in df.columns:
        df["true_log2"] = pd.to_numeric(df["true_log2"], errors="coerce")
    elif "true_mic" in df.columns:
        tmp = pd.to_numeric(df["true_mic"], errors="coerce")
        df["true_log2"] = np.where(tmp > 0, np.log2(tmp), np.nan)
    else:
        raise ValueError("Could not find true MIC/log2 column in external validation table.")

    if "log2_pred_mic" in df.columns:
        df["pred_log2"] = pd.to_numeric(df["log2_pred_mic"], errors="coerce")
    elif "pred_log2" in df.columns:
        df["pred_log2"] = pd.to_numeric(df["pred_log2"], errors="coerce")
    elif "pred_mic" in df.columns:
        tmp = pd.to_numeric(df["pred_mic"], errors="coerce")
        df["pred_log2"] = np.where(tmp > 0, np.log2(tmp), np.nan)
    else:
        raise ValueError("Could not find predicted MIC/log2 column in external validation table.")

    if "log2_error" in df.columns:
        df["residual"] = pd.to_numeric(df["log2_error"], errors="coerce")
    else:
        df["residual"] = df["pred_log2"] - df["true_log2"]

    df["abs_log2_error"] = df["residual"].abs()

    # IC
    ic_col = None
    for c in ["IC_microreact_or_reference", "IC", "IC_tree_conservative"]:
        if c in df.columns:
            ic_col = c
            break

    if ic_col is not None:
        df["IC_plot"] = df[ic_col].map(clean_group)
    else:
        df["IC_plot"] = np.nan

    # ST and hBAPS
    stp_col = None
    sto_col = None

    for c in df.columns:
        if c.lower() in {"st (mlst (pasteur))", "st_pasteur"}:
            stp_col = c
        if c.lower() in {"st (mlst (oxford))", "st_oxford"}:
            sto_col = c

    if stp_col is not None:
        df["ST_Pasteur_clean"] = df[stp_col].map(clean_st)
    else:
        df["ST_Pasteur_clean"] = np.nan

    if sto_col is not None:
        df["ST_Oxford_clean"] = df[sto_col].map(clean_st)
    else:
        df["ST_Oxford_clean"] = np.nan

    hbaps_col = None
    for c in df.columns:
        if c.lower() == "hbaps":
            hbaps_col = c
            break

    if hbaps_col is not None:
        df["hBAPS_plot"] = df[hbaps_col].map(clean_group)
    else:
        pasteur_map, oxford_map = build_st_to_hbaps_map()
        df["hBAPS_plot"] = df["ST_Pasteur_clean"].map(pasteur_map)
        df["hBAPS_plot"] = df["hBAPS_plot"].fillna(df["ST_Oxford_clean"].map(oxford_map))

    df = df[
        df["model"].isin(MODEL_ORDER)
        & df["antibiotic"].isin(AB_ORDER)
        & df["true_log2"].notna()
        & df["pred_log2"].notna()
    ].copy()

    rows = []
    for model in MODEL_ORDER:
        for ab in AB_ORDER:
            d = df[(df["model"] == model) & (df["antibiotic"] == ab)].copy()
            rows.append({
                "dataset": "external_validation",
                "model": model,
                "model_label": MODEL_LABEL[model],
                "antibiotic": ab,
                **calc_metrics(d),
                "source_file": str(EXTERNAL_ERROR_FP),
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLEDIR / "figure3_external_validation_metrics.tsv", sep="\t", index=False)

    return df, metrics


def summarise_group(df, group_col):
    d = df[df[group_col].notna()].copy()

    if d.empty:
        return pd.DataFrame()

    out = (
        d.groupby([group_col, "model", "antibiotic"], dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            within_1=("abs_log2_error", lambda x: (x <= 1).mean()),
            within_2=("abs_log2_error", lambda x: (x <= 2).mean()),
            mean_signed_error=("residual", "mean"),
        )
        .reset_index()
    )

    out["support_level"] = pd.cut(
        out["n"],
        bins=[-1, 4, 9, 999999],
        labels=["very_low_n", "low_n", "supported_n"],
    )

    return out


def sort_ic(labels):
    def key(x):
        m = re.search(r"IC(\d+)", str(x))
        if m:
            return (0, int(m.group(1)))
        return (1, str(x))
    return sorted(labels, key=key)


# ============================================================
# PLOTS
# ============================================================

def plot_prediction_grid(fig, gs, data, title, panel_letter):
    for row_i, ab in enumerate(AB_ORDER):
        for col_i, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[row_i, col_i])
            style_ax(ax)

            d = data[model][ab]
            colour = COL[AB_SHORT[ab]]

            lo = min(d["true_log2"].min(), d["pred_log2"].min()) - 0.5
            hi = max(d["true_log2"].max(), d["pred_log2"].max()) + 0.5

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
                s=14,
                alpha=0.70,
                color=colour,
                edgecolor=COL["white"],
                linewidth=0.25,
                zorder=3,
            )

            ax.plot([lo, hi], [lo, hi], "--", color=COL["dark"], lw=1.0, zorder=4)

            m = calc_metrics(d)
            ax.text(
                0.04, 0.96,
                f"n={m['n']}\nMAE={m['mae_log2']:.2f}\n±1={m['within_1']*100:.1f}%\n±2={m['within_2']*100:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.1,
                color=COL["dark"],
                bbox=dict(
                    facecolor=COL["white"],
                    edgecolor="none",
                    alpha=0.78,
                    boxstyle="round,pad=0.25",
                ),
            )

            if row_i == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=10, fontweight="bold")

            if col_i == 0:
                ax.set_ylabel(f"{AB_LABEL[ab]}\nPredicted MIC (log₂)", fontsize=8.5)

            if row_i == 1:
                ax.set_xlabel("Observed MIC (log₂)", fontsize=8.5)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=7.5)

    first_ax = fig.axes[-6]
    first_ax.text(
        -0.32, 1.17,
        panel_letter,
        transform=first_ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=COL["dark"],
    )
    first_ax.text(
        0.00, 1.17,
        title,
        transform=first_ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COL["dark"],
    )


def plot_metric_summary(ax, metrics, title, panel_letter):
    style_ax(ax)

    metric_specs = [
        ("mae_log2", "MAE", False),
        ("within_1", "±1", True),
        ("within_2", "±2", True),
    ]

    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    ax2 = None

    # Plot ±1 and ±2 as bars on left axis; annotate MAE above.
    for offset, ab in [(-width / 2, "imipenem"), (width / 2, "meropenem")]:
        vals = []
        maes = []

        for model in MODEL_ORDER:
            r = metrics[(metrics["model"] == model) & (metrics["antibiotic"] == ab)].iloc[0]
            vals.append(r["within_1"] * 100)
            maes.append(r["mae_log2"])

        ax.bar(
            x + offset,
            vals,
            width,
            color=COL[AB_SHORT[ab]],
            edgecolor=COL["dark"],
            linewidth=0.4,
            label=AB_LABEL[ab],
        )

        for xi, val, mae in zip(x + offset, vals, maes):
            ax.text(
                xi,
                val + 2,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Within ±1 dilution (%)")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)

    ax.text(
        -0.15,
        1.08,
        panel_letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.50,
        1.01,
        "MAE shown above bars",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


def plot_external_scatter_grid(fig, gs, external_df):
    data = {m: {} for m in MODEL_ORDER}

    for model in MODEL_ORDER:
        for ab in AB_ORDER:
            data[model][ab] = external_df[
                (external_df["model"] == model)
                & (external_df["antibiotic"] == ab)
            ].copy()

    plot_prediction_grid(
        fig,
        gs,
        data,
        title="External validation: observed vs predicted MIC",
        panel_letter="C",
    )


def plot_heatmap(ax, summary, group_col, title, panel_letter=None, max_groups=None):
    style_ax(ax, grid=False)

    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    # group order
    totals = (
        summary.groupby(group_col, dropna=False)["n"]
        .sum()
        .sort_values(ascending=False)
    )

    groups = list(totals.index)

    if group_col == "IC_plot":
        groups = sort_ic(groups)

    if max_groups is not None:
        groups = groups[:max_groups]

    cols = [(ab, model) for ab in AB_ORDER for model in MODEL_ORDER]

    mat = np.full((len(groups), len(cols)), np.nan)
    ann = np.full((len(groups), len(cols)), "", dtype=object)

    for i, grp in enumerate(groups):
        for j, (ab, model) in enumerate(cols):
            sub = summary[
                (summary[group_col] == grp)
                & (summary["antibiotic"] == ab)
                & (summary["model"] == model)
            ]

            if sub.empty:
                continue

            r = sub.iloc[0]
            mat[i, j] = r["mae_log2"]

            n = int(r["n"])
            flag = "*" if n < 10 else ""
            ann[i, j] = f"{r['mae_log2']:.2f}\n{r['within_1']*100:.0f}%\nn={n}{flag}"

    vmax = max(2.0, np.nanmax(mat) if np.isfinite(mat).any() else 2.0)

    im = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=HEAT_CMAP,
        norm=Normalize(vmin=0, vmax=vmax),
        aspect="auto",
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=6.1)

    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups, fontsize=7.5)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["AMR", "GWAS", "Hybrid", "AMR", "GWAS", "Hybrid"],
                       rotation=35, ha="right", fontsize=7.5)

    ax.axvline(2.5, color=COL["dark"], lw=1.0)
    ax.text(1.0, 1.03, "IMI", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=COL["IMI"])
    ax.text(4.0, 1.03, "MER", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=COL["MER"])

    ax.set_title(title, fontsize=10, fontweight="bold")

    if panel_letter:
        ax.text(
            -0.18,
            1.08,
            panel_letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    return im


# ============================================================
# MAIN
# ============================================================

def main():
    print("[INFO] Loading self-test predictions")
    selftest_data, selftest_metrics = load_selftest_data()

    print("[INFO] Loading external validation predictions/error table")
    external_df, external_metrics = load_external_data()

    ic_summary = summarise_group(external_df, "IC_plot")

    # hBAPS only within IC2, because this is the useful sublineage portability view.
    hbaps_ic2_summary = summarise_group(
        external_df[external_df["IC_plot"] == "IC2"].copy(),
        "hBAPS_plot",
    )

    ic_summary.to_csv(TABLEDIR / "figure3_external_validation_by_IC.tsv", sep="\t", index=False)
    hbaps_ic2_summary.to_csv(TABLEDIR / "figure3_external_validation_by_hBAPS_within_IC2.tsv", sep="\t", index=False)

    all_metrics = pd.concat([selftest_metrics, external_metrics], ignore_index=True)
    all_metrics.to_csv(TABLEDIR / "figure3_selftest_vs_external_metrics.tsv", sep="\t", index=False)

    print("\n[SELF-TEST METRICS]")
    print(selftest_metrics[[
        "model_label", "antibiotic", "n", "mae_log2", "within_1", "within_2"
    ]].to_string(index=False))

    print("\n[EXTERNAL VALIDATION METRICS]")
    print(external_metrics[[
        "model_label", "antibiotic", "n", "mae_log2", "within_1", "within_2"
    ]].to_string(index=False))

    # ========================================================
    # FIGURE LAYOUT
    # ========================================================

    fig = plt.figure(figsize=(16.5, 17.0), facecolor=COL["white"])

    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.05, 0.75, 2.05, 1.15],
        hspace=0.42,
    )

    # A: self-test scatter grid
    gs_a = outer[0].subgridspec(2, 3, hspace=0.28, wspace=0.25)
    plot_prediction_grid(
        fig,
        gs_a,
        selftest_data,
        title="Self-test: observed vs predicted MIC",
        panel_letter="A",
    )

    # B: self-test summary
    gs_b = outer[1].subgridspec(1, 1)
    ax_b = fig.add_subplot(gs_b[0, 0])
    plot_metric_summary(
        ax_b,
        selftest_metrics,
        title="Self-test model performance",
        panel_letter="B",
    )

    # C: external validation scatter grid
    gs_c = outer[2].subgridspec(2, 3, hspace=0.28, wspace=0.25)
    plot_external_scatter_grid(fig, gs_c, external_df)

    # D: portability stratification
    gs_d = outer[3].subgridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.32)

    ax_d1 = fig.add_subplot(gs_d[0, 0])
    im1 = plot_heatmap(
        ax_d1,
        ic_summary,
        group_col="IC_plot",
        title="External portability by inferred IC",
        panel_letter="D",
    )

    ax_d2 = fig.add_subplot(gs_d[0, 1])
    im2 = plot_heatmap(
        ax_d2,
        hbaps_ic2_summary,
        group_col="hBAPS_plot",
        title="External portability by hBAPS within IC2",
        panel_letter=None,
        max_groups=8,
    )

    use_im = im2 if im2 is not None else im1
    if use_im is not None:
        cbar = fig.colorbar(use_im, ax=[ax_d1, ax_d2], fraction=0.020, pad=0.015)
        cbar.set_label("MAE (log₂ MIC dilutions)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Machine learning prediction and portability of carbapenem MICs in Acinetobacter baumannii",
        fontsize=15,
        fontweight="bold",
        y=0.995,
        color=COL["dark"],
    )

    fig.text(
        0.012,
        0.010,
        "Panels A–B show self-test performance after locked XGBoost tuning. "
        "Panels C–D assess portability in the external validation dataset. "
        "Scatter bands show ±1 and ±2 log₂ MIC dilutions. "
        "Heatmap cells show MAE / within ±1 dilution (%) / n; * indicates n < 10.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=COL["dark"],
    )

    for ext in ["png", "svg", "pdf"]:
        out = f"{OUT_PREFIX}.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")

    plt.close(fig)

    print("\n[OK] wrote:")
    print(f"  {OUT_PREFIX}.png")
    print(f"  {OUT_PREFIX}.svg")
    print(f"  {OUT_PREFIX}.pdf")

    print("\n[OK] tables:")
    for fp in sorted(TABLEDIR.glob("figure3_*.tsv")):
        print(f"  {fp}")


if __name__ == "__main__":
    main()