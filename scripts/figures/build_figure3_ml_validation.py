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
# PATHS
# ============================================================

def pick_path(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return Path(candidates[0])

ERROR_FP = pick_path([
    "outputs/paper_validation_package/error_characterisation/validation_prediction_error_table.with_microreact_ic.tsv",
    "outputs/error_characterisation/validation_prediction_error_table.with_microreact_ic.tsv",
])

IC_FP = pick_path([
    "outputs/paper_validation_package/tables/validation_error_by_ST_inferred_IC_all_with_support.fixed.tsv",
    "outputs/error_characterisation/validation_error_by_ST_inferred_IC_all_with_support.fixed.tsv",
])

MICRO_FP = pick_path([
    "data/FullMicroreactWyr-with-Russian-Metadata_IC_inferred.csv",
])

OUTDIR = Path("outputs/figures")
TABDIR = Path("outputs/figure3_ml_validation/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUTDIR / "Figure3_ml_validation"

# ============================================================
# STYLE / COLOURS
# ============================================================

COL = {
    "paper": "#FAF8F2",
    "cream": "#F7F1E3",
    "light_cream": "#FBF7EE",
    "grid": "#D7CEC1",
    "ink": "#2D2A32",
    "white": "#FFFFFF",
    "grey": "#B8B8B8",
    "light_grey": "#E6E1D8",

    # antibiotic colours (keep close to existing scheme)
    "IMI": "#E67E22",      # burnt orange
    "MER": "#F4C542",      # mustard
    "CARB": "#C1121F",     # carbapenem red

    # model colours
    "amr": "#7C9A92",      # muted teal
    "locus": "#D9A441",    # muted gold
    "hybrid": "#C8553D",   # muted red
}

MODEL_LABEL = {
    "amr": "AMR only",
    "locus": "Locus-GWAS",
    "hybrid": "Hybrid",
}

MODEL_ORDER = ["amr", "locus", "hybrid"]
AB_ORDER = ["imipenem", "meropenem"]
AB_SHORT = {"imipenem": "IMI", "meropenem": "MER"}
AB_LABEL = {"imipenem": "Imipenem", "meropenem": "Meropenem"}

plt.rcParams.update({
    "figure.facecolor": COL["paper"],
    "axes.facecolor": COL["cream"],
    "savefig.facecolor": COL["paper"],
    "font.size": 9,
    "axes.labelcolor": COL["ink"],
    "xtick.color": COL["ink"],
    "ytick.color": COL["ink"],
    "text.color": COL["ink"],
    "axes.edgecolor": COL["ink"],
})

MAE_CMAP = LinearSegmentedColormap.from_list(
    "wes_mae",
    [COL["light_cream"], COL["MER"], COL["IMI"], COL["CARB"]]
)

# ============================================================
# HELPERS
# ============================================================

def style_ax(ax, grid=True):
    ax.set_facecolor(COL["cream"])
    if grid:
        ax.grid(True, color=COL["grid"], linewidth=0.6, alpha=0.7)
    else:
        ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL["ink"])
        ax.spines[spine].set_linewidth(0.8)

def clean_st(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na"}:
        return np.nan
    x = re.sub(r"^ST", "", x, flags=re.I).strip()
    if re.fullmatch(r"\d+\.0", x):
        x = str(int(float(x)))
    return x

def clean_group(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none", "unassigned"}:
        return np.nan
    return x

def as_binary01(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    tmp = series.astype(str).str.strip().str.lower()
    out = tmp.map({
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        "1.0": 1.0,
        "0.0": 0.0,
    })
    num = pd.to_numeric(series, errors="coerce")
    out = out.where(out.notna(), num)
    return out.astype(float)

def find_first_existing_col(df, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        cand_low = cand.lower()
        for c in cols:
            if cand_low in c.lower():
                return c
    return None

def calc_metrics(d):
    err = d["log2_error"].astype(float)
    abs_err = err.abs()
    return {
        "n": int(len(d)),
        "mae_log2": float(abs_err.mean()),
        "within_1": float((abs_err <= 1).mean()),
        "within_2": float((abs_err <= 2).mean()),
        "mean_signed_error": float(err.mean()),
    }

def build_majority_map(micro, st_col, target_col, min_n=5, min_prop=0.80):
    tmp = micro[[st_col, target_col]].copy()
    tmp[st_col] = tmp[st_col].map(clean_st)
    tmp[target_col] = tmp[target_col].map(clean_group)
    tmp = tmp[tmp[st_col].notna() & tmp[target_col].notna()].copy()

    if tmp.empty:
        return {}, pd.DataFrame(columns=[st_col, target_col, "n_target", "n_total", "prop", "accepted"])

    counts = (
        tmp.groupby([st_col, target_col], dropna=False)
        .size()
        .reset_index(name="n_target")
    )
    totals = (
        tmp.groupby(st_col, dropna=False)
        .size()
        .reset_index(name="n_total")
    )
    ref = counts.merge(totals, on=st_col, how="left")
    ref["prop"] = ref["n_target"] / ref["n_total"]
    ref = ref.sort_values([st_col, "prop", "n_target"], ascending=[True, False, False])

    best = ref.groupby(st_col, as_index=False).head(1).copy()
    best["accepted"] = (best["n_total"] >= min_n) & (best["prop"] >= min_prop)

    mapping = (
        best.loc[best["accepted"], [st_col, target_col]]
        .drop_duplicates()
        .set_index(st_col)[target_col]
        .to_dict()
    )

    return mapping, best

def sort_ic_labels(labels):
    def keyfunc(x):
        x = str(x)
        m = re.search(r"IC(\d+)", x)
        if m:
            return (0, int(m.group(1)))
        return (1, x)
    return sorted(labels, key=keyfunc)

# ============================================================
# LOAD / PREP DATA
# ============================================================

def load_error_table():
    df = pd.read_csv(ERROR_FP, sep="\t", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # standardise key columns
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip().str.lower()
    if "antibiotic" in df.columns:
        df["antibiotic"] = df["antibiotic"].astype(str).str.strip().str.lower()

    # true / pred MIC columns
    true_mic_col = find_first_existing_col(df, ["true_mic", "truth_mic", "observed_mic", "actual_mic"])
    pred_mic_col = find_first_existing_col(df, ["pred_mic", "predicted_mic", "prediction", "predicted"])

    true_log2_col = find_first_existing_col(df, ["true_log2", "truth_log2", "observed_log2"])
    pred_log2_col = find_first_existing_col(df, ["pred_log2", "prediction_log2", "predicted_log2"])

    if true_log2_col is not None:
        df["true_log2"] = pd.to_numeric(df[true_log2_col], errors="coerce")
    elif true_mic_col is not None:
        tmp = pd.to_numeric(df[true_mic_col], errors="coerce")
        df["true_log2"] = np.where(tmp > 0, np.log2(tmp), np.nan)
    else:
        raise ValueError("Could not find true MIC/log2 column in validation error table.")

    if pred_log2_col is not None:
        df["pred_log2"] = pd.to_numeric(df[pred_log2_col], errors="coerce")
    elif pred_mic_col is not None:
        tmp = pd.to_numeric(df[pred_mic_col], errors="coerce")
        df["pred_log2"] = np.where(tmp > 0, np.log2(tmp), np.nan)
    else:
        raise ValueError("Could not find predicted MIC/log2 column in validation error table.")

    if "log2_error" in df.columns:
        df["log2_error"] = pd.to_numeric(df["log2_error"], errors="coerce")
    else:
        df["log2_error"] = df["pred_log2"] - df["true_log2"]

    if "abs_log2_error" in df.columns:
        df["abs_log2_error"] = pd.to_numeric(df["abs_log2_error"], errors="coerce")
    else:
        df["abs_log2_error"] = df["log2_error"].abs()

    w1_col = find_first_existing_col(df, ["within_1_dilution", "within1"])
    w2_col = find_first_existing_col(df, ["within_2_dilution", "within2"])

    if w1_col is not None:
        df["within_1_dilution"] = as_binary01(df[w1_col])
    else:
        df["within_1_dilution"] = (df["abs_log2_error"] <= 1).astype(float)

    if w2_col is not None:
        df["within_2_dilution"] = as_binary01(df[w2_col])
    else:
        df["within_2_dilution"] = (df["abs_log2_error"] <= 2).astype(float)

    # ST columns
    stp_col = find_first_existing_col(df, ["ST (MLST (Pasteur))", "ST_Pasteur", "pasteur"])
    sto_col = find_first_existing_col(df, ["ST (MLST (Oxford))", "ST_Oxford", "oxford"])
    if stp_col is not None:
        df["ST_Pasteur_clean"] = df[stp_col].map(clean_st)
    else:
        df["ST_Pasteur_clean"] = np.nan
    if sto_col is not None:
        df["ST_Oxford_clean"] = df[sto_col].map(clean_st)
    else:
        df["ST_Oxford_clean"] = np.nan

    # hBAPS inference from Microreact if not already present
    hbaps_col = find_first_existing_col(df, ["hBAPS", "hbaps"])
    if hbaps_col is not None:
        df["hBAPS_inferred"] = df[hbaps_col].map(clean_group)
    else:
        micro = pd.read_csv(MICRO_FP, dtype=str, low_memory=False)
        micro.columns = [c.strip() for c in micro.columns]
        if "hBAPS" not in micro.columns:
            raise ValueError(f"hBAPS column not found in {MICRO_FP}")

        map_p, ref_p = build_majority_map(micro, "ST_Pasteur", "hBAPS")
        map_o, ref_o = build_majority_map(micro, "ST_Oxford", "hBAPS")

        df["hBAPS_inferred"] = df["ST_Pasteur_clean"].map(map_p)
        df["hBAPS_inferred"] = df["hBAPS_inferred"].fillna(df["ST_Oxford_clean"].map(map_o))

        ref_p.to_csv(TABDIR / "reference_ST_Pasteur_to_hBAPS.tsv", sep="\t", index=False)
        ref_o.to_csv(TABDIR / "reference_ST_Oxford_to_hBAPS.tsv", sep="\t", index=False)

    # final cleanup
    df = df[
        df["model"].isin(MODEL_ORDER)
        & df["antibiotic"].isin(AB_ORDER)
        & df["true_log2"].notna()
        & df["pred_log2"].notna()
    ].copy()

    return df

def load_ic_summary(err_df):
    if IC_FP.exists():
        ic = pd.read_csv(IC_FP, sep="\t", low_memory=False)
        ic.columns = [c.strip() for c in ic.columns]
        ic["model"] = ic["model"].astype(str).str.strip().str.lower()
        ic["antibiotic"] = ic["antibiotic"].astype(str).str.strip().str.lower()
        ic["IC_microreact_or_reference"] = ic["IC_microreact_or_reference"].astype(str).str.strip()

        for c in ["n", "mae_log2", "within_1", "within_2", "mean_signed_error"]:
            if c in ic.columns:
                ic[c] = pd.to_numeric(ic[c], errors="coerce")

        if "support_level" not in ic.columns:
            ic["support_level"] = pd.cut(
                ic["n"],
                bins=[-1, 4, 9, 999999],
                labels=["very_low_n", "low_n", "supported_n"],
            )
        return ic

    # fallback: build from raw error table
    tmp = err_df[err_df["IC_microreact_or_reference"].notna()].copy()
    ic = (
        tmp.groupby(["IC_microreact_or_reference", "model", "antibiotic"], dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
            mean_signed_error=("log2_error", "mean"),
        )
        .reset_index()
    )
    ic["support_level"] = pd.cut(
        ic["n"],
        bins=[-1, 4, 9, 999999],
        labels=["very_low_n", "low_n", "supported_n"],
    )
    return ic

def overall_summary(err):
    out = (
        err.groupby(["model", "antibiotic"], dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
            mean_signed_error=("log2_error", "mean"),
        )
        .reset_index()
    )
    return out

def hbaps_ic2_summary(err):
    if "IC_microreact_or_reference" not in err.columns:
        return pd.DataFrame()

    tmp = err[
        (err["IC_microreact_or_reference"] == "IC2")
        & err["hBAPS_inferred"].notna()
    ].copy()

    if tmp.empty:
        return pd.DataFrame()

    out = (
        tmp.groupby(["hBAPS_inferred", "model", "antibiotic"], dropna=False)
        .agg(
            n=("sample", "count"),
            mae_log2=("abs_log2_error", "mean"),
            within_1=("within_1_dilution", "mean"),
            within_2=("within_2_dilution", "mean"),
            mean_signed_error=("log2_error", "mean"),
        )
        .reset_index()
    )
    out["support_level"] = pd.cut(
        out["n"],
        bins=[-1, 4, 9, 999999],
        labels=["very_low_n", "low_n", "supported_n"],
    )
    return out

# ============================================================
# PLOTTING
# ============================================================

def plot_scatter_grid(fig, gs, err):
    axes = []
    for r, ab in enumerate(AB_ORDER):
        for c, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)
            style_ax(ax, grid=True)

            d = err[(err["antibiotic"] == ab) & (err["model"] == model)].copy()
            colour = COL[AB_SHORT[ab]]

            lo = min(d["true_log2"].min(), d["pred_log2"].min()) - 0.5
            hi = max(d["true_log2"].max(), d["pred_log2"].max()) + 0.5

            # ±2 and ±1 bands
            ax.fill_between(
                [lo, hi],
                [lo - 2, hi - 2],
                [lo + 2, hi + 2],
                color=COL["light_grey"],
                alpha=0.75,
                zorder=0,
                linewidth=0,
            )
            ax.fill_between(
                [lo, hi],
                [lo - 1, hi - 1],
                [lo + 1, hi + 1],
                color=COL["grey"],
                alpha=0.30,
                zorder=1,
                linewidth=0,
            )

            ax.scatter(
                d["true_log2"],
                d["pred_log2"],
                s=18,
                alpha=0.72,
                color=colour,
                edgecolor=COL["white"],
                linewidth=0.25,
                zorder=3,
            )
            ax.plot([lo, hi], [lo, hi], "--", color=COL["ink"], lw=1.0, zorder=4)

            m = calc_metrics(d)
            ax.text(
                0.04, 0.96,
                f"n={m['n']}\nMAE={m['mae_log2']:.2f}\n±1={m['within_1']*100:.1f}%\n±2={m['within_2']*100:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.3,
                bbox=dict(
                    facecolor=COL["white"],
                    edgecolor="none",
                    alpha=0.82,
                    boxstyle="round,pad=0.25",
                ),
            )

            if r == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=10.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{AB_LABEL[ab]}\nPredicted MIC (log₂)")
            if r == 1:
                ax.set_xlabel("Observed MIC (log₂)")

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=8)

    axes[0].text(
        -0.28, 1.15, "A",
        transform=axes[0].transAxes,
        fontsize=16, fontweight="bold", color=COL["ink"]
    )
    fig.text(
        0.08, 0.965,
        "Machine learning prediction of carbapenem MICs",
        ha="left", va="center",
        fontsize=12, fontweight="bold", color=COL["ink"]
    )

def plot_overall_bar(ax, overall):
    style_ax(ax, grid=True)

    y = np.arange(len(MODEL_ORDER))
    height = 0.32

    imi = []
    mer = []
    imi_mae = []
    mer_mae = []

    for m in MODEL_ORDER:
        x1 = overall[(overall["model"] == m) & (overall["antibiotic"] == "imipenem")].iloc[0]
        x2 = overall[(overall["model"] == m) & (overall["antibiotic"] == "meropenem")].iloc[0]
        imi.append(x1["within_1"] * 100)
        mer.append(x2["within_1"] * 100)
        imi_mae.append(x1["mae_log2"])
        mer_mae.append(x2["mae_log2"])

    ax.barh(y - height/2, imi, height=height, color=COL["IMI"], label="Imipenem")
    ax.barh(y + height/2, mer, height=height, color=COL["MER"], label="Meropenem")

    for i in range(len(y)):
        ax.text(imi[i] + 1.2, y[i] - height/2, f"{imi[i]:.1f}%  |  MAE {imi_mae[i]:.2f}",
                va="center", ha="left", fontsize=7.4)
        ax.text(mer[i] + 1.2, y[i] + height/2, f"{mer[i]:.1f}%  |  MAE {mer_mae[i]:.2f}",
                va="center", ha="left", fontsize=7.4)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Within ±1 dilution (%)")
    ax.set_title("Overall validation performance", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax.text(
        -0.22, 1.08, "B",
        transform=ax.transAxes,
        fontsize=16, fontweight="bold", color=COL["ink"]
    )

def plot_group_heatmap(ax, df, group_col, title, group_order=None):
    style_ax(ax, grid=False)

    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    if group_order is None:
        totals = (
            df.groupby(group_col, dropna=False)["n"]
            .sum()
            .sort_values(ascending=False)
        )
        group_order = list(totals.index)

    col_keys = [(ab, model) for ab in AB_ORDER for model in MODEL_ORDER]
    nrow, ncol = len(group_order), len(col_keys)

    mat = np.full((nrow, ncol), np.nan, dtype=float)
    ann = np.empty((nrow, ncol), dtype=object)
    ann[:] = ""

    for i, grp in enumerate(group_order):
        for j, (ab, model) in enumerate(col_keys):
            sub = df[
                (df[group_col] == grp)
                & (df["antibiotic"] == ab)
                & (df["model"] == model)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            mat[i, j] = float(r["mae_log2"])
            n = int(r["n"])
            lowflag = "*" if n < 10 else ""
            ann[i, j] = f"{r['mae_log2']:.2f}\n{r['within_1']*100:.0f}%\n(n={n}{lowflag})"

    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 2.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    im = ax.imshow(np.ma.masked_invalid(mat), cmap=MAE_CMAP, norm=norm, aspect="auto")

    for i in range(nrow):
        for j in range(ncol):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j, i, ann[i, j],
                    ha="center", va="center",
                    fontsize=6.3, color=COL["ink"]
                )

    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(group_order)

    xticklabels = [
        "AMR", "GWAS", "Hybrid",
        "AMR", "GWAS", "Hybrid",
    ]
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(xticklabels, fontsize=8)

    ax.axvline(2.5, color=COL["ink"], lw=1.0)
    ax.text(1.0, 1.03, "IMI", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=COL["IMI"])
    ax.text(4.0, 1.03, "MER", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=COL["MER"])

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    return im

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"[INFO] Using error table: {ERROR_FP}")
    print(f"[INFO] Using IC summary:  {IC_FP}")
    print(f"[INFO] Using Microreact:  {MICRO_FP}")

    err = load_error_table()
    ic = load_ic_summary(err)
    overall = overall_summary(err)
    hb_ic2 = hbaps_ic2_summary(err)

    overall.to_csv(TABDIR / "figure3_overall_validation_summary.tsv", sep="\t", index=False)
    ic.to_csv(TABDIR / "figure3_validation_by_IC.tsv", sep="\t", index=False)
    hb_ic2.to_csv(TABDIR / "figure3_validation_by_hBAPS_within_IC2.tsv", sep="\t", index=False)

    # choose IC order
    ic_groups = sort_ic_labels(ic["IC_microreact_or_reference"].dropna().unique().tolist())

    # choose top hBAPS groups within IC2
    if not hb_ic2.empty:
        hb_groups = (
            hb_ic2.groupby("hBAPS_inferred", dropna=False)["n"]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .index
            .tolist()
        )
    else:
        hb_groups = []

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------
    fig = plt.figure(figsize=(16.5, 12.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.35], hspace=0.30)

    # top = 2x3 scatter
    gs_top = outer[0].subgridspec(2, 3, wspace=0.24, hspace=0.28)
    plot_scatter_grid(fig, gs_top, err)

    # bottom = summary bar + IC heatmap + hBAPS heatmap
    gs_bottom = outer[1].subgridspec(1, 3, width_ratios=[0.90, 1.25, 1.25], wspace=0.35)

    ax_b = fig.add_subplot(gs_bottom[0, 0])
    plot_overall_bar(ax_b, overall)

    ax_c = fig.add_subplot(gs_bottom[0, 1])
    im1 = plot_group_heatmap(
        ax_c,
        ic,
        group_col="IC_microreact_or_reference",
        title="C  Validation performance by inferred IC",
        group_order=ic_groups,
    )

    ax_d = fig.add_subplot(gs_bottom[0, 2])
    im2 = plot_group_heatmap(
        ax_d,
        hb_ic2,
        group_col="hBAPS_inferred",
        title="D  hBAPS structure within IC2",
        group_order=hb_groups,
    )

    # colorbar
    use_im = im2 if im2 is not None else im1
    if use_im is not None:
        cbar = fig.colorbar(use_im, ax=[ax_c, ax_d], fraction=0.025, pad=0.02)
        cbar.set_label("Mean absolute error (log₂ MIC)")

    # footnote
    fig.text(
        0.012, 0.012,
        "Heatmap cells show MAE / within ±1 dilution (%) / n. "
        "Asterisks indicate low-support cells (n < 10). "
        "Lower-right panel uses hBAPS groups inferred from Microreact reference data within IC2.",
        ha="left", va="bottom", fontsize=8, color=COL["ink"]
    )

    for ext in ["png", "pdf", "svg"]:
        out = f"{OUT_PREFIX}.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")

    plt.close(fig)

    print("\n[OK] wrote:")
    print(" ", f"{OUT_PREFIX}.png")
    print(" ", f"{OUT_PREFIX}.pdf")
    print(" ", f"{OUT_PREFIX}.svg")
    print("\n[OK] summary tables:")
    for fp in sorted(TABDIR.glob("*.tsv")):
        print(" ", fp)

if __name__ == "__main__":
    main()