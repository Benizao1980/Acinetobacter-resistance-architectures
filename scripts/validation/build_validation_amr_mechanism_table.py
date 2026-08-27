from pathlib import Path
import pandas as pd
import re

amr_fp = Path("outputs/amrfinder/validation/amr_presence_absence.tsv")
outdir = Path("outputs/error_characterisation")
outdir.mkdir(parents=True, exist_ok=True)

amr = pd.read_csv(amr_fp, sep="\t")

if "sample" not in amr.columns:
    amr = amr.rename(columns={amr.columns[0]: "sample"})

gene_cols = [c for c in amr.columns if c.startswith("GENE:")]

def present_genes(row):
    genes = []
    for c in gene_cols:
        try:
            val = float(row[c])
        except Exception:
            val = 0
        if val > 0:
            genes.append(c.replace("GENE:", ""))
    return genes

def has_any(genes, patterns):
    text = ";".join(genes)
    return any(re.search(p, text, re.I) for p in patterns)

rows = []

for _, row in amr.iterrows():
    genes = present_genes(row)

    oxa_genes = [g for g in genes if re.search(r"blaOXA", g, re.I)]
    mbl_genes = [g for g in genes if re.search(r"bla(NDM|VIM|IMP|SIM|GIM|SPM)", g, re.I)]
    carb_genes = [
        g for g in genes
        if re.search(r"bla(OXA|NDM|VIM|IMP|KPC|GES|PER|CTX|CMY|ADC)", g, re.I)
    ]

    has_o23 = has_any(genes, [r"blaOXA-23"])
    has_o24_40 = has_any(genes, [r"blaOXA-24", r"blaOXA-40", r"blaOXA-72"])
    has_o58 = has_any(genes, [r"blaOXA-58"])
    has_o235 = has_any(genes, [r"blaOXA-235"])
    has_mbl = len(mbl_genes) > 0
    has_o51 = has_any(genes, [r"blaOXA-51", r"blaOXA-64", r"blaOXA-65", r"blaOXA-66", r"blaOXA-68", r"blaOXA-69", r"blaOXA-71", r"blaOXA-82", r"blaOXA-90", r"blaOXA-91", r"blaOXA-94", r"blaOXA-95", r"blaOXA-98", r"blaOXA-100", r"blaOXA-104", r"blaOXA-106", r"blaOXA-109", r"blaOXA-113", r"blaOXA-120", r"blaOXA-121", r"blaOXA-126", r"blaOXA-132"])

    if has_mbl and (has_o23 or has_o24_40 or has_o58 or has_o235):
        mechanism = "MBL + acquired OXA"
    elif has_mbl:
        mechanism = "MBL"
    elif has_o23:
        mechanism = "OXA-23-like"
    elif has_o24_40:
        mechanism = "OXA-24/40-like"
    elif has_o58:
        mechanism = "OXA-58-like"
    elif has_o235:
        mechanism = "OXA-235-like"
    elif has_o51:
        mechanism = "OXA-51-like only"
    elif carb_genes:
        mechanism = "Other beta-lactamase"
    else:
        mechanism = "No detected carbapenemase"

    rows.append({
        "sample": row["sample"],
        "amrfinder_mechanism_group": mechanism,
        "has_mbl": int(has_mbl),
        "has_oxa23_like": int(has_o23),
        "has_oxa24_40_like": int(has_o24_40),
        "has_oxa58_like": int(has_o58),
        "has_oxa235_like": int(has_o235),
        "has_oxa51_like": int(has_o51),
        "oxa_genes": ";".join(sorted(oxa_genes)),
        "mbl_genes": ";".join(sorted(mbl_genes)),
        "carbapenemase_or_beta_lactamase_genes": ";".join(sorted(carb_genes)),
        "n_amr_genes": len(genes),
    })

out = pd.DataFrame(rows)

out_fp = outdir / "validation_amrfinder_mechanism_groups.tsv"
out.to_csv(out_fp, sep="\t", index=False)

print("Wrote:", out_fp)
print("Rows:", len(out))
print()
print(out["amrfinder_mechanism_group"].value_counts(dropna=False).to_string())
