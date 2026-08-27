#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# INPUTS / OUTPUTS
# ============================================================

IN = Path("outputs/global_prediction/stratified/global_predictions_enriched.tsv")
OUTDIR = Path("outputs/global_prediction/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUTDIR / "Figure_global_predictions"

# ============================================================
# WES-ISH / UPDATED COLOUR SCHEME
# ============================================================

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

MODEL_COL = {
    "AMR only": COL["dark"],
    "Locus-GWAS": COL["grey"],
    "Hybrid": COL["IMI+MER"],
}

AB_COL = {
    "imipenem": COL["IMI"],
    "meropenem": COL["MER"],
}

ANTIBIOTIC_LABELS = {
    "imipenem": "Imipenem",
    "meropenem": "Meropenem",
}

ANTIBIOTIC_SHORT = {
    "imipenem": "IMI",
    "meropenem": "MER",
}

MODEL_LABELS = {
    "amr": "AMR only",
    "locus": "Locus-GWAS",
    "hybrid": "Hybrid",
}

MODELS = ["amr", "locus", "hybrid"]
ANTIBIOTICS = ["imipenem", "meropenem"]
THRESHOLDS = [2, 8, 16, 32]

# Main manuscript threshold to emphasise
MAIN_THR = 8


def force_clean_st_label(x):
    if pd.isna(x):
        return "Unassigned"
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none", "unassigned"}:
        return "Unassigned"
    if x.endswith(".0"):
        x = x[:-2]
    if not x.upper().startswith("ST"):
        x = "ST" + x
    return x


def force_clean_mechanism_label(x):
    if pd.isna(x):
        return "Unassigned"
    x = str(x).strip()
    return {
        "Intrinsic OXA-51-like only/other": "OXA-51-like only/other",
        "Other beta-lactamase": "Other β-lactamase",
        "No detected carbapenemase": "No carbapenemase",
    }.get(x, x)


# ============================================================
# HELPERS
# ============================================================

def clean_st_for_plot(x):
    """Render Pasteur ST labels cleanly, e.g. 15.0 -> ST15."""
    if pd.isna(x):
        return "Unassigned"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "na", "none"}:
        return "Unassigned"
    if s.endswith(".0"):
        s = s[:-2]
    if not s.upper().startswith("ST"):
        s = "ST" + s
    return s


def clean_mechanism_for_plot(x):
    """Shorten long mechanism labels for plotting."""
    if pd.isna(x):
        return "Unassigned"
    s = str(x).strip()
    replacements = {
        "Intrinsic OXA-51-like only/other": "OXA-51-like only/other",
        "Other beta-lactamase": "Other β-lactamase",
        "No detected carbapenemase": "No carbapenemase",
    }
    return replacements.get(s, s)


def clean_label(x):
    if pd.isna(x):
        return "Unassigned"
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none"}:
        return "Unassigned"
    return x


def add_prediction_flags(df):
    df = df.copy()

    for model in MODELS:
        for ab in ANTIBIOTICS:
            mic_col = f"{model}_{ab}_pred_mic"
            if mic_col not in df.columns:
                continue
            df[mic_col] = pd.to_numeric(df[mic_col], errors="coerce")

            for thr in THRESHOLDS:
                df[f"{model}_{ab}_gt{thr}"] = df[mic_col] > thr

    for ab in ANTIBIOTICS:
        for thr in THRESHOLDS:
            flag_cols = [
                f"{model}_{ab}_gt{thr}"
                for model in MODELS
                if f"{model}_{ab}_gt{thr}" in df.columns
            ]
            df[f"{ab}_n_models_gt{thr}"] = df[flag_cols].sum(axis=1)
            df[f"{ab}_consensus_gt{thr}"] = df[f"{ab}_n_models_gt{thr}"] >= 2

    return df


def summarise_model_thresholds(df):
    rows = []
    for ab in ANTIBIOTICS:
        for model in MODELS:
            mic_col = f"{model}_{ab}_pred_mic"
            vals = pd.to_numeric(df[mic_col], errors="coerce")
            row = {
                "antibiotic": ab,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n": vals.notna().sum(),
                "median_mic": vals.median(),
                "iqr_low": vals.quantile(0.25),
                "iqr_high": vals.quantile(0.75),
            }
            for thr in THRESHOLDS:
                row[f"prop_gt{thr}"] = (vals > thr).mean()
                row[f"n_gt{thr}"] = int((vals > thr).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_consensus(df, group_col, min_n=10, threshold=MAIN_THR):
    tmp = df.copy()
    tmp[group_col] = tmp[group_col].map(clean_label)

    rows = []
    for group, sub in tmp.groupby(group_col, dropna=False):
        if len(sub) < min_n:
            continue

        row = {
            group_col: group,
            "n": len(sub),
        }

        for ab in ANTIBIOTICS:
            row[f"{ab}_consensus_prop"] = sub[f"{ab}_consensus_gt{threshold}"].mean()
            row[f"{ab}_consensus_n"] = int(sub[f"{ab}_consensus_gt{threshold}"].sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["any_consensus_prop"] = (
        out["imipenem_consensus_prop"] + out["meropenem_consensus_prop"]
    ) / 2

    out = out.sort_values("any_consensus_prop", ascending=True)
    return out


def style_ax(ax):
    ax.set_facecolor(COL["cream"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COL["dark"])
    ax.spines["bottom"].set_color(COL["dark"])
    ax.tick_params(colors=COL["dark"], labelsize=8)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        color=COL["dark"],
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(IN, sep="\t", low_memory=False)
df = add_prediction_flags(df)

# Clean grouping columns
if "IC" not in df.columns:
    df["IC"] = "Unassigned"
if "ST_Pasteur" not in df.columns:
    df["ST_Pasteur"] = "Unassigned"
if "amrfinder_carbapenemase_group" not in df.columns:
    df["amrfinder_carbapenemase_group"] = "Unassigned"

df["IC_clean"] = df["IC"].map(clean_label)
df["ST_Pasteur_clean"] = df["ST_Pasteur"].map(clean_label)
df["mechanism_clean"] = df["amrfinder_carbapenemase_group"].map(clean_label)

model_summary = summarise_model_thresholds(df)

ic_summary = summarise_consensus(df, "IC_clean", min_n=10, threshold=MAIN_THR)
mech_summary = summarise_consensus(df, "mechanism_clean", min_n=10, threshold=MAIN_THR)
st_summary = summarise_consensus(df, "ST_Pasteur_clean", min_n=10, threshold=MAIN_THR)

# Limit STs to top informative groups for figure readability
if not st_summary.empty:
    st_summary = st_summary.sort_values("n", ascending=False).head(12)
    st_summary = st_summary.sort_values("any_consensus_prop", ascending=True)

# Save figure-source tables
model_summary.to_csv(OUTDIR / "global_prediction_figure_model_threshold_source.tsv", sep="\t", index=False)
ic_summary.to_csv(OUTDIR / "global_prediction_figure_IC_source.tsv", sep="\t", index=False)
mech_summary.to_csv(OUTDIR / "global_prediction_figure_mechanism_source.tsv", sep="\t", index=False)
st_summary.to_csv(OUTDIR / "global_prediction_figure_ST_source.tsv", sep="\t", index=False)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": COL["white"],
    "savefig.facecolor": COL["white"],
})


# ============================================================
# FINAL LABEL CLEANING BEFORE PLOTTING / EXPORT
# ============================================================

for _name in ["st_source", "mechanism_source", "ic_source"]:
    if _name in globals():
        _df = globals()[_name]

        if _name == "st_source" and "ST_Pasteur_clean" in _df.columns:
            _df["ST_Pasteur_clean"] = _df["ST_Pasteur_clean"].map(clean_st_for_plot)

        if _name == "mechanism_source" and "mechanism_clean" in _df.columns:
            _df["mechanism_clean"] = _df["mechanism_clean"].map(clean_mechanism_for_plot)

        if _name == "ic_source" and "IC_clean" in _df.columns:
            _df["IC_clean"] = _df["IC_clean"].replace({
                "nan": "Unassigned",
                "NaN": "Unassigned",
                "None": "Unassigned",
            })


fig = plt.figure(figsize=(14, 12), constrained_layout=False)
fig.patch.set_facecolor(COL["white"])

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1.0, 1.15, 1.2],
    width_ratios=[1, 1],
    hspace=0.52,
    wspace=0.34,
)

# ------------------------------------------------------------
# A. Predicted MIC distributions by model and drug
# ------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
style_ax(axA)
panel_label(axA, "A")

box_data = []
box_positions = []
box_colors = []
box_labels = []

pos = 1
for ab in ANTIBIOTICS:
    for model in MODELS:
        col = f"{model}_{ab}_pred_mic"
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = np.log2(vals.clip(lower=0.03125))
        box_data.append(vals)
        box_positions.append(pos)
        box_colors.append(AB_COL[ab])
        box_labels.append(MODEL_LABELS[model])
        pos += 1
    pos += 0.7

bp = axA.boxplot(
    box_data,
    positions=box_positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color=COL["highlight"], linewidth=1.3),
    boxprops=dict(linewidth=0.8, color=COL["dark"]),
    whiskerprops=dict(linewidth=0.8, color=COL["dark"]),
    capprops=dict(linewidth=0.8, color=COL["dark"]),
)

for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)

axA.axhline(np.log2(MAIN_THR), color=COL["IMI+MER"], linewidth=1.1, linestyle="--")
axA.text(
    0.98, np.log2(MAIN_THR) + 0.1,
    f"{MAIN_THR} mg/L",
    transform=axA.get_yaxis_transform(),
    ha="right",
    va="bottom",
    color=COL["IMI+MER"],
    fontsize=8,
)

axA.set_xticks(box_positions)
axA.set_xticklabels(box_labels, rotation=35, ha="right")
axA.set_ylabel("Predicted MIC (log2 mg/L)")
axA.set_title("Predicted carbapenem MIC distributions")

# Drug labels
axA.text(2, axA.get_ylim()[1], "Imipenem", ha="center", va="bottom", color=COL["dark"], fontsize=9)
axA.text(5.7, axA.get_ylim()[1], "Meropenem", ha="center", va="bottom", color=COL["dark"], fontsize=9)

# ------------------------------------------------------------
# B. Model-level predicted resistance proportions
# ------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
style_ax(axB)
panel_label(axB, "B")

plot_df = model_summary.copy()
x_labels = []
x = np.arange(len(MODELS))
width = 0.35

for i, ab in enumerate(ANTIBIOTICS):
    sub = plot_df[plot_df["antibiotic"] == ab].set_index("model").loc[MODELS]
    vals = sub[f"prop_gt{MAIN_THR}"].values
    offset = -width / 2 if i == 0 else width / 2
    axB.bar(
        x + offset,
        vals,
        width=width,
        color=AB_COL[ab],
        edgecolor=COL["dark"],
        linewidth=0.7,
        label=ANTIBIOTIC_LABELS[ab],
        alpha=0.9,
    )

    for xi, v, n in zip(x + offset, vals, sub[f"n_gt{MAIN_THR}"].values):
        axB.text(xi, v + 0.018, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8, color=COL["dark"])

axB.set_xticks(x)
axB.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=25, ha="right")
axB.set_ylim(0, 0.75)
axB.set_ylabel(f"Proportion predicted >{MAIN_THR} mg/L")
axB.set_title("Predicted high-MIC burden by model")
axB.legend(frameon=False, loc="upper right")

# ------------------------------------------------------------
# C. Consensus predictions by IC
# ------------------------------------------------------------
axC = fig.add_subplot(gs[1, 0])
style_ax(axC)
panel_label(axC, "C")

if not ic_summary.empty:
    y = np.arange(len(ic_summary))
    h = 0.36

    axC.barh(
        y - h / 2,
        ic_summary["imipenem_consensus_prop"],
        height=h,
        color=COL["IMI"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Imipenem",
    )
    axC.barh(
        y + h / 2,
        ic_summary["meropenem_consensus_prop"],
        height=h,
        color=COL["MER"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Meropenem",
    )

    labels = [f"{g} (n={n})" for g, n in zip(ic_summary["IC_clean"], ic_summary["n"])]
    axC.set_yticks(y)
    axC.set_yticklabels(labels)
    axC.set_xlim(0, 1)
    axC.set_xlabel(f"Consensus proportion predicted >{MAIN_THR} mg/L")
    axC.set_title("Global predictions stratified by international clone")
    axC.legend(frameon=False, loc="lower right")
else:
    axC.text(0.5, 0.5, "No IC groups available", ha="center", va="center")

# ------------------------------------------------------------
# D. Consensus predictions by carbapenemase mechanism
# ------------------------------------------------------------
axD = fig.add_subplot(gs[1, 1])
style_ax(axD)
panel_label(axD, "D")

if not mech_summary.empty:
    y = np.arange(len(mech_summary))
    h = 0.36

    axD.barh(
        y - h / 2,
        mech_summary["imipenem_consensus_prop"],
        height=h,
        color=COL["IMI"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Imipenem",
    )
    axD.barh(
        y + h / 2,
        mech_summary["meropenem_consensus_prop"],
        height=h,
        color=COL["MER"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Meropenem",
    )

    labels = [f"{g} (n={n})" for g, n in zip(mech_summary["mechanism_clean"], mech_summary["n"])]
    axD.set_yticks(y)
    axD.set_yticklabels(labels)
    axD.set_xlim(0, 1)
    axD.set_xlabel(f"Consensus proportion predicted >{MAIN_THR} mg/L")
    axD.set_title("Global predictions stratified by AMRFinderPlus mechanism")
    axD.legend(frameon=False, loc="lower right")
else:
    axD.text(0.5, 0.5, "No mechanism groups available", ha="center", va="center")

# ------------------------------------------------------------
# E. Consensus predictions by common Pasteur ST
# ------------------------------------------------------------
axE = fig.add_subplot(gs[2, 0])
style_ax(axE)
panel_label(axE, "E")

if not st_summary.empty:
    y = np.arange(len(st_summary))
    h = 0.36

    axE.barh(
        y - h / 2,
        st_summary["imipenem_consensus_prop"],
        height=h,
        color=COL["IMI"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Imipenem",
    )
    axE.barh(
        y + h / 2,
        st_summary["meropenem_consensus_prop"],
        height=h,
        color=COL["MER"],
        edgecolor=COL["dark"],
        linewidth=0.5,
        label="Meropenem",
    )

    labels = [f"ST{g} (n={n})" if g != "Unassigned" else f"Unassigned (n={n})"
              for g, n in zip(st_summary["ST_Pasteur_clean"], st_summary["n"])]
    axE.set_yticks(y)
    axE.set_yticklabels(labels)
    axE.set_xlim(0, 1)
    axE.set_xlabel(f"Consensus proportion predicted >{MAIN_THR} mg/L")
    axE.set_title("Global predictions stratified by common Pasteur ST")
    axE.legend(frameon=False, loc="lower right")
else:
    axE.text(0.5, 0.5, "No ST groups available", ha="center", va="center")

# ------------------------------------------------------------
# F. Consensus across thresholds
# ------------------------------------------------------------
axF = fig.add_subplot(gs[2, 1])
style_ax(axF)
panel_label(axF, "F")

threshold_rows = []
for ab in ANTIBIOTICS:
    for thr in THRESHOLDS:
        threshold_rows.append({
            "antibiotic": ab,
            "threshold": thr,
            "prop": df[f"{ab}_consensus_gt{thr}"].mean(),
            "n": int(df[f"{ab}_consensus_gt{thr}"].sum()),
        })

thr_df = pd.DataFrame(threshold_rows)

x = np.arange(len(THRESHOLDS))
width = 0.35

for i, ab in enumerate(ANTIBIOTICS):
    sub = thr_df[thr_df["antibiotic"] == ab].set_index("threshold").loc[THRESHOLDS]
    offset = -width / 2 if i == 0 else width / 2
    axF.bar(
        x + offset,
        sub["prop"],
        width=width,
        color=AB_COL[ab],
        edgecolor=COL["dark"],
        linewidth=0.7,
        label=ANTIBIOTIC_LABELS[ab],
    )
    for xi, v, n in zip(x + offset, sub["prop"], sub["n"]):
        axF.text(xi, v + 0.018, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)

axF.set_xticks(x)
axF.set_xticklabels([f">{t}" for t in THRESHOLDS])
axF.set_ylim(0, 0.75)
axF.set_ylabel("Consensus proportion")
axF.set_xlabel("Predicted MIC threshold (mg/L)")
axF.set_title("Consensus predictions across MIC thresholds")
axF.legend(frameon=False, loc="upper right")

# ------------------------------------------------------------
# Final layout / export
# ------------------------------------------------------------

fig.suptitle(
    "Global prediction of carbapenem MICs across A. baumannii population diversity",
    fontsize=14,
    fontweight="bold",
    color=COL["dark"],
    y=0.985,
)

fig.text(
    0.01, 0.012,
    "Consensus = at least two of three feature sets (AMR only, Locus-GWAS, Hybrid) exceed threshold. "
    "Predictions are model-derived MIC estimates; global genomes do not have measured MICs.",
    fontsize=8,
    color=COL["dark"],
)

plt.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.07)

# Force-clean labels after source tables are assembled
if "st_source" in globals() and "ST_Pasteur_clean" in st_source.columns:
    st_source["ST_Pasteur_clean"] = st_source["ST_Pasteur_clean"].map(force_clean_st_label)

if "mechanism_source" in globals() and "mechanism_clean" in mechanism_source.columns:
    mechanism_source["mechanism_clean"] = mechanism_source["mechanism_clean"].map(force_clean_mechanism_label)

for ext in ["png", "svg", "pdf"]:
    out = f"{OUT_PREFIX}.{ext}"
    fig.savefig(out, dpi=400 if ext == "png" else None)
    print("[OK] wrote", out)

plt.close(fig)

print("[OK] source tables written to", OUTDIR)
