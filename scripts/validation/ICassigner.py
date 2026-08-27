#!/usr/bin/env python3
"""
ICassigner.py

Conservative phylogeny-guided assignment of Acinetobacter baumannii
International Clones (ICs) using a core genome tree and metadata.

Key behaviour
- Existing IC labels are never overwritten (unless --force_overwrite is set).
- Unassigned isolates (default label: "UA") are assigned only if:
    * >= MIN_SUPPORT labelled neighbours in the nearest ancestor clade, AND
    * >= MIN_PROP majority support for a single IC among those labelled neighbours.
- Otherwise, isolates remain "UA".

Optional extras (recommended)
- Summary report (counts before/after, inferred vs remaining UA)
- Simple plots (PNG + PDF):
    1) IC counts before vs after
    2) Histogram of support_n for inferred calls
    3) Histogram of support_prop for inferred calls
- Cross-checks and confusion matrices (CSV + plot):
    * ST-expected IC vs assigned IC (Pasteur and/or Oxford MLST ST columns)
    * Any user-supplied grouping columns (e.g., cgMLST clonal group, BAPS)

Dependencies
- pandas
- ete3
- matplotlib

Example
python ICassigner.py --tree RAxML-result.Acinetobacter-coreML.nwk \
  --metadata metadata.csv --tip_col sample_id --ic_col IC \
  --plots --outdir outputs_icassigner \
  --pasteur_st_col ST_Pasteur --oxford_st_col ST_Oxford \
  --group_cols cgMLST_group,BAPS
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import pandas as pd
from ete3 import Tree


# --- Canonical ST → IC anchors (conservative / commonly used) ---
# Notes:
# - These are intended as a sanity-check, not as a primary lineage definition.
# - Real datasets include exceptions due to recombination, nomenclature differences, and incomplete typing.
PASTEUR_ST_TO_IC = {
    "ST1": "IC1",
    "ST2": "IC2",
    "ST3": "IC3",
    "ST15": "IC4",
    "ST79": "IC5",
    "ST78": "IC6",
    "ST25": "IC7",
    "ST10": "IC8",
    "ST85": "IC9",
}

# Oxford ST mapping is less “one-to-one” in the literature; the following are common anchors used as checks.
OXFORD_ST_TO_IC = {
    "ST231": "IC1",
    "ST208": "IC2",
    "ST451": "IC2",
    "ST348": "IC3",
    "ST15": "IC4",
    "ST79": "IC5",
    "ST78": "IC6",
    "ST25": "IC7",
    "ST85": "IC9",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conservative IC assignment from a phylogenetic tree.")
    p.add_argument("--tree", required=True, help="Newick tree file (e.g. RAxML core genome tree)")
    p.add_argument("--metadata", required=True, help="Metadata CSV containing sample IDs and IC labels")
    p.add_argument("--tip_col", default="sample_id", help="Column for tree tip/sample IDs")
    p.add_argument("--ic_col", default="IC", help="Column for International Clone labels")
    p.add_argument("--ua_label", default="UA", help='Label used for "unassigned" in IC column (default: UA)')

    p.add_argument("--min_support", type=int, default=10, help="Minimum labelled neighbours required (default: 10)")
    p.add_argument("--min_prop", type=float, default=0.90, help="Minimum majority proportion required (default: 0.90)")

    p.add_argument("--output", default="metadata_with_conservative_IC.csv", help="Output CSV filename")
    p.add_argument("--outdir", default="outputs_icassigner", help="Output directory for plots/reports (default: outputs_icassigner)")
    p.add_argument("--plots", action="store_true", help="Generate simple plots (PNG+PDF) into --outdir")

    p.add_argument("--pasteur_st_col", default=None, help="Column containing Pasteur MLST ST (e.g., ST_Pasteur)")
    p.add_argument("--oxford_st_col", default=None, help="Column containing Oxford MLST ST (e.g., ST_Oxford)")
    p.add_argument("--group_cols", default=None,
                   help="Comma-separated list of additional group columns for confusion matrices (e.g., cgMLST,BAPS)")
    p.add_argument("--max_groups_plot", type=int, default=30,
                   help="Max categories to show on confusion plots for group_cols (default: 30). Full CSV is always written.")

    p.add_argument("--force_overwrite", action="store_true",
                   help="Overwrite existing non-UA IC labels with tree-inferred labels (NOT recommended).")

    p.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: INFO)")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def normalize_st(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Accept forms like "2" -> "ST2", "ST2" -> "ST2"
    if s.upper().startswith("ST"):
        return "ST" + s[2:].strip()
    # If it's numeric-ish
    if s.isdigit():
        return "ST" + s
    return s  # fallback (still might map if user already has ST### strings)


@dataclass
class InferenceResult:
    ic: str
    support_n: int
    majority_n: int
    majority_prop: float
    node_size: int


def infer_ic_from_tree(
    tree: Tree,
    tip: str,
    ic_map: Dict[str, Optional[str]],
    ua_label: str,
    min_support: int,
    min_prop: float,
) -> InferenceResult:
    """
    Walk up from tip to ancestors; at each node compute labels among descendants.
    Assign if there are at least min_support labelled descendants and majority_prop >= min_prop.
    """
    node = tree & tip
    while node is not None:
        tips = node.get_leaf_names()
        labels = [ic_map.get(t) for t in tips if ic_map.get(t) not in [None, ua_label]]
        if len(labels) >= min_support:
            counts = Counter(labels)
            ic, majority_n = counts.most_common(1)[0]
            total = sum(counts.values())
            majority_prop = majority_n / total if total else 0.0
            if majority_prop >= min_prop:
                return InferenceResult(ic=ic, support_n=total, majority_n=majority_n, majority_prop=majority_prop, node_size=len(tips))
        node = node.up
    return InferenceResult(ic=ua_label, support_n=0, majority_n=0, majority_prop=0.0, node_size=0)


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_summary(meta: pd.DataFrame, ic_col: str, out_ic_col: str, ua_label: str, outdir: str) -> None:
    safe_makedirs(outdir)
    before = meta[ic_col].fillna("NA").astype(str)
    after = meta[out_ic_col].fillna("NA").astype(str)

    total = len(meta)
    ua_before = (before == ua_label).sum()
    ua_after = (after == ua_label).sum()
    inferred = ((before == ua_label) & (after != ua_label)).sum()

    lines = []
    lines.append(f"Total isolates: {total}")
    lines.append(f"UA before: {ua_before}")
    lines.append(f"UA after: {ua_after}")
    lines.append(f"Inferred (UA -> IC): {inferred}")
    lines.append("")
    lines.append("Counts before (top 20):")
    lines.extend([f"  {k}: {v}" for k, v in before.value_counts().head(20).items()])
    lines.append("")
    lines.append("Counts after (top 20):")
    lines.extend([f"  {k}: {v}" for k, v in after.value_counts().head(20).items()])

    with open(os.path.join(outdir, "ICassigner_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_ic_before_after(meta: pd.DataFrame, ic_col: str, out_ic_col: str, outdir: str) -> None:
    import matplotlib.pyplot as plt

    before = meta[ic_col].fillna("NA").astype(str)
    after = meta[out_ic_col].fillna("NA").astype(str)

    all_ics = sorted(set(before.unique()).union(set(after.unique())))
    before_counts = before.value_counts().reindex(all_ics, fill_value=0)
    after_counts = after.value_counts().reindex(all_ics, fill_value=0)

    x = list(range(len(all_ics)))
    plt.figure()
    plt.bar([i - 0.2 for i in x], before_counts.values, width=0.4, label="Before")
    plt.bar([i + 0.2 for i in x], after_counts.values, width=0.4, label="After")
    plt.xticks(x, all_ics, rotation=90)
    plt.ylabel("Isolates")
    plt.title("IC counts before vs after conservative tree inference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "IC_counts_before_after.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "IC_counts_before_after.pdf"))
    plt.close()


def plot_histograms(meta: pd.DataFrame, ic_col: str, out_ic_col: str, ua_label: str, outdir: str) -> None:
    import matplotlib.pyplot as plt

    inferred_mask = (meta[ic_col].astype(str) == ua_label) & (meta[out_ic_col].astype(str) != ua_label)
    inferred = meta.loc[inferred_mask].copy()

    if len(inferred) == 0:
        logging.info("No inferred isolates found for histogram plots (UA -> IC). Skipping histograms.")
        return

    if "IC_tree_support_n" in inferred.columns:
        plt.figure()
        plt.hist(pd.to_numeric(inferred["IC_tree_support_n"], errors="coerce").dropna(), bins=30)
        plt.xlabel("Labelled neighbours used (support_n)")
        plt.ylabel("Count")
        plt.title("Support used for inferred IC calls")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "IC_inference_support_hist.png"), dpi=300)
        plt.savefig(os.path.join(outdir, "IC_inference_support_hist.pdf"))
        plt.close()

    if "IC_tree_support_prop" in inferred.columns:
        plt.figure()
        plt.hist(pd.to_numeric(inferred["IC_tree_support_prop"], errors="coerce").dropna(), bins=30)
        plt.xlabel("Majority support proportion (support_prop)")
        plt.ylabel("Count")
        plt.title("Majority support for inferred IC calls")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "IC_inference_prop_hist.png"), dpi=300)
        plt.savefig(os.path.join(outdir, "IC_inference_prop_hist.pdf"))
        plt.close()


def save_confusion_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path)


def plot_confusion_matrix(df: pd.DataFrame, title: str, out_png: str, out_pdf: str, max_rows: int = 30, max_cols: int = 30) -> None:
    """
    Plot a confusion matrix (counts) using matplotlib imshow.
    For very large matrices, plot a truncated view (top categories by marginal totals),
    but still write the full CSV separately.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Truncate for plotting
    if df.shape[0] > max_rows:
        top_rows = df.sum(axis=1).sort_values(ascending=False).head(max_rows).index
        dfp = df.loc[top_rows, :]
    else:
        dfp = df.copy()

    if dfp.shape[1] > max_cols:
        top_cols = dfp.sum(axis=0).sort_values(ascending=False).head(max_cols).index
        dfp = dfp.loc[:, top_cols]

    data = dfp.values.astype(float)

    plt.figure(figsize=(max(6, 0.3 * dfp.shape[1]), max(4, 0.3 * dfp.shape[0])))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xlabel(dfp.columns.name if dfp.columns.name else "Predicted/Assigned")
    plt.ylabel(dfp.index.name if dfp.index.name else "Expected/Group")
    plt.xticks(range(dfp.shape[1]), dfp.columns.tolist(), rotation=90)
    plt.yticks(range(dfp.shape[0]), dfp.index.tolist())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def build_st_expected_ic(meta: pd.DataFrame, st_col: str, mapping: Dict[str, str], out_col: str) -> pd.Series:
    st_norm = meta[st_col].apply(normalize_st)
    return st_norm.map(mapping)


def main() -> None:
    args = parse_args()
    setup_logging(args.loglevel)

    safe_makedirs(args.outdir)

    logging.info("Reading tree: %s", args.tree)
    tree = Tree(args.tree, format=1)

    logging.info("Reading metadata: %s", args.metadata)
    meta = pd.read_csv(args.metadata)

    if args.tip_col not in meta.columns:
        raise ValueError(f"--tip_col '{args.tip_col}' not found in metadata columns.")

    if args.ic_col not in meta.columns:
        raise ValueError(f"--ic_col '{args.ic_col}' not found in metadata columns.")

    # Build IC map keyed by tip/sample id
    ic_map: Dict[str, Optional[str]] = dict(zip(meta[args.tip_col].astype(str), meta[args.ic_col]))

    # Track which metadata tips are missing in tree
    tree_tips = set(tree.get_leaf_names())
    meta_tips = set(meta[args.tip_col].astype(str))
    missing_in_tree = sorted(list(meta_tips - tree_tips))
    if missing_in_tree:
        logging.warning("Found %d metadata tips not present in tree. These cannot be inferred and will remain as-is.", len(missing_in_tree))
        with open(os.path.join(args.outdir, "tips_missing_in_tree.txt"), "w") as f:
            f.write("\n".join(missing_in_tree) + "\n")

    inferred_ic: List[str] = []
    support_n: List[Optional[int]] = []
    majority_n: List[Optional[int]] = []
    support_prop: List[Optional[float]] = []
    node_size: List[Optional[int]] = []

    out_ic_col = "IC_tree_conservative"

    for tip in meta[args.tip_col].astype(str):
        original_ic = ic_map.get(tip)

        if (not args.force_overwrite) and (original_ic not in [None, args.ua_label]):
            # Keep known labels
            inferred_ic.append(str(original_ic))
            support_n.append(None)
            majority_n.append(None)
            support_prop.append(None)
            node_size.append(None)
            continue

        # Only infer if tip is actually in tree; otherwise keep as-is
        if tip not in tree_tips:
            inferred_ic.append(str(original_ic) if original_ic is not None else args.ua_label)
            support_n.append(0)
            majority_n.append(0)
            support_prop.append(0.0)
            node_size.append(0)
            continue

        res = infer_ic_from_tree(
            tree=tree,
            tip=tip,
            ic_map=ic_map,
            ua_label=args.ua_label,
            min_support=args.min_support,
            min_prop=args.min_prop,
        )

        inferred_ic.append(res.ic)
        support_n.append(res.support_n)
        majority_n.append(res.majority_n)
        support_prop.append(res.majority_prop)
        node_size.append(res.node_size)

    meta[out_ic_col] = inferred_ic
    meta["IC_tree_support_n"] = support_n
    meta["IC_tree_majority_n"] = majority_n
    meta["IC_tree_support_prop"] = support_prop
    meta["IC_tree_node_size"] = node_size

    logging.info("Writing output CSV: %s", args.output)
    meta.to_csv(args.output, index=False)

    # Write summary
    write_summary(meta, args.ic_col, out_ic_col, args.ua_label, args.outdir)

    # Cross-check: Pasteur/Oxford ST expected IC
    # Output both per-sample flags and confusion matrix plots/CSVs
    if args.pasteur_st_col and args.pasteur_st_col in meta.columns:
        meta["IC_expected_from_PasteurST"] = build_st_expected_ic(meta, args.pasteur_st_col, PASTEUR_ST_TO_IC, "IC_expected_from_PasteurST")
        # conflicts (only where both expected and assigned are present and assigned != UA)
        mask = meta["IC_expected_from_PasteurST"].notna() & (meta[out_ic_col].astype(str) != args.ua_label)
        meta["IC_PasteurST_conflict"] = False
        meta.loc[mask, "IC_PasteurST_conflict"] = (meta.loc[mask, "IC_expected_from_PasteurST"].astype(str) != meta.loc[mask, out_ic_col].astype(str))

        ct = pd.crosstab(
            meta.loc[mask, "IC_expected_from_PasteurST"].astype(str),
            meta.loc[mask, out_ic_col].astype(str),
            rownames=["Expected_IC (Pasteur ST)"],
            colnames=["Assigned_IC (tree)"]
        )
        save_confusion_csv(ct, os.path.join(args.outdir, "confusion_expectedIC_PasteurST_vs_assignedIC.csv"))
        if args.plots:
            plot_confusion_matrix(
                ct,
                title="Pasteur ST-expected IC vs Tree-assigned IC",
                out_png=os.path.join(args.outdir, "confusion_expectedIC_PasteurST_vs_assignedIC.png"),
                out_pdf=os.path.join(args.outdir, "confusion_expectedIC_PasteurST_vs_assignedIC.pdf"),
                max_rows=min(args.max_groups_plot, 50),
                max_cols=min(args.max_groups_plot, 50)
            )
    elif args.pasteur_st_col:
        logging.warning("--pasteur_st_col provided but column not found: %s", args.pasteur_st_col)

    if args.oxford_st_col and args.oxford_st_col in meta.columns:
        meta["IC_expected_from_OxfordST"] = build_st_expected_ic(meta, args.oxford_st_col, OXFORD_ST_TO_IC, "IC_expected_from_OxfordST")
        mask = meta["IC_expected_from_OxfordST"].notna() & (meta[out_ic_col].astype(str) != args.ua_label)
        meta["IC_OxfordST_conflict"] = False
        meta.loc[mask, "IC_OxfordST_conflict"] = (meta.loc[mask, "IC_expected_from_OxfordST"].astype(str) != meta.loc[mask, out_ic_col].astype(str))

        ct = pd.crosstab(
            meta.loc[mask, "IC_expected_from_OxfordST"].astype(str),
            meta.loc[mask, out_ic_col].astype(str),
            rownames=["Expected_IC (Oxford ST)"],
            colnames=["Assigned_IC (tree)"]
        )
        save_confusion_csv(ct, os.path.join(args.outdir, "confusion_expectedIC_OxfordST_vs_assignedIC.csv"))
        if args.plots:
            plot_confusion_matrix(
                ct,
                title="Oxford ST-expected IC vs Tree-assigned IC",
                out_png=os.path.join(args.outdir, "confusion_expectedIC_OxfordST_vs_assignedIC.png"),
                out_pdf=os.path.join(args.outdir, "confusion_expectedIC_OxfordST_vs_assignedIC.pdf"),
                max_rows=min(args.max_groups_plot, 50),
                max_cols=min(args.max_groups_plot, 50)
            )
    elif args.oxford_st_col:
        logging.warning("--oxford_st_col provided but column not found: %s", args.oxford_st_col)

    # Group column confusion matrices (e.g., cgMLST group, BAPS)
    if args.group_cols:
        cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
        for gc in cols:
            if gc not in meta.columns:
                logging.warning("Group column not found (skipping): %s", gc)
                continue
            mask = meta[gc].notna() & (meta[out_ic_col].astype(str) != args.ua_label)
            if mask.sum() == 0:
                logging.info("No rows to compare for group column %s (all NA or assigned=UA).", gc)
                continue

            ct = pd.crosstab(
                meta.loc[mask, gc].astype(str),
                meta.loc[mask, out_ic_col].astype(str),
                rownames=[f"{gc}"],
                colnames=["Assigned_IC (tree)"]
            )
            safe_name = "".join([ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in gc])
            save_confusion_csv(ct, os.path.join(args.outdir, f"confusion_{safe_name}_vs_assignedIC.csv"))
            if args.plots:
                plot_confusion_matrix(
                    ct,
                    title=f"{gc} vs Tree-assigned IC (counts)",
                    out_png=os.path.join(args.outdir, f"confusion_{safe_name}_vs_assignedIC.png"),
                    out_pdf=os.path.join(args.outdir, f"confusion_{safe_name}_vs_assignedIC.pdf"),
                    max_rows=args.max_groups_plot,
                    max_cols=min(args.max_groups_plot, 50)
                )

    # If we added expected IC columns, re-write the enriched metadata next to outdir too (optional convenience)
    enriched_path = os.path.join(args.outdir, os.path.basename(args.output).replace(".csv", "_enriched.csv"))
    meta.to_csv(enriched_path, index=False)
    logging.info("Wrote enriched output (with cross-check columns) to: %s", enriched_path)

    # Plots
    if args.plots:
        logging.info("Generating plots into: %s", args.outdir)
        plot_ic_before_after(meta, args.ic_col, out_ic_col, args.outdir)
        plot_histograms(meta, args.ic_col, out_ic_col, args.ua_label, args.outdir)

    logging.info("Done.")


if __name__ == "__main__":
    main()