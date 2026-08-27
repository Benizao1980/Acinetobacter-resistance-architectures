from pathlib import Path
import pandas as pd

outdir = Path("outputs/error_characterisation")
outdir.mkdir(parents=True, exist_ok=True)

mic_fp = Path("data/phenotypes_validation/validation_dataset_MIC.cleaned.tsv")
rich_fp = Path("data/phenotypes_validation/data.csv")
amr_mech_fp = Path("outputs/error_characterisation/validation_amrfinder_mechanism_groups.tsv")

mic = pd.read_csv(mic_fp, sep="\t")
rich = pd.read_csv(rich_fp, sep="\t")
amr_mech = pd.read_csv(amr_mech_fp, sep="\t")

mic["numeric_id"] = mic["id"].astype(str).str.extract(r"^(\d+)")
rich["numeric_id"] = rich["id"].astype(str)
amr_mech["numeric_id"] = amr_mech["sample"].astype(str).str.extract(r"^(\d+)")

wanted = [
    "numeric_id",
    "id",
    "isolate",
    "country",
    "continent",
    "region",
    "town or city",
    "origin",
    "year",
    "species",
    "source",
    "detailed source",
    "resistance profile",
    "OXA family",
    "OXA class",
    "OXA betalactamase",
    "MBL family",
    "MBL class",
    "MBL betalactamase",
    "class B D betalactamase",
    "ST (MLST (Oxford))",
    "species (MLST (Oxford))",
    "ST (MLST (Pasteur))",
    "species (MLST (Pasteur))",
    "cgST (A. baumannii cgMLST)",
    "rST (Ribosomal MLST)",
    "lineage (Ribosomal MLST)",
    "sublineage (Ribosomal MLST)",
    "bioproject accession",
    "biosample accession",
    "run accession",
]

wanted = [c for c in wanted if c in rich.columns]
rich_sub = rich[wanted].copy()

rich_sub = rich_sub.rename(columns={
    "id": "numeric_id_original",
    "country": "country_rich",
    "year": "year_rich",
    "resistance profile": "resistance_profile_rich",
    "bioproject accession": "bioproject_accession_rich",
    "biosample accession": "biosample_accession_rich",
    "run accession": "run_accession_rich",
})

merged = mic.merge(rich_sub, on="numeric_id", how="left")

amr_cols = [
    "numeric_id",
    "amrfinder_mechanism_group",
    "has_mbl",
    "has_oxa23_like",
    "has_oxa24_40_like",
    "has_oxa58_like",
    "has_oxa235_like",
    "has_oxa51_like",
    "oxa_genes",
    "mbl_genes",
    "carbapenemase_or_beta_lactamase_genes",
    "n_amr_genes",
]
amr_cols = [c for c in amr_cols if c in amr_mech.columns]

merged = merged.merge(amr_mech[amr_cols], on="numeric_id", how="left")

out_fp = outdir / "validation_metadata_for_error_characterisation.tsv"
merged.to_csv(out_fp, sep="\t", index=False)

print("Wrote:", out_fp)
print("Rows:", len(merged))
print("Matched rich metadata:", merged["isolate"].notna().sum() if "isolate" in merged.columns else "NA")
print("Matched AMRFinderPlus mechanism:", merged["amrfinder_mechanism_group"].notna().sum())
print()
print("AMRFinderPlus mechanism groups:")
print(merged["amrfinder_mechanism_group"].value_counts(dropna=False).to_string())
print()
if "ST (MLST (Pasteur))" in merged.columns:
    print("Top Pasteur STs:")
    print(merged["ST (MLST (Pasteur))"].value_counts(dropna=False).head(15).to_string())
