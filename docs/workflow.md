# Analysis workflow

This document separates reusable BAMPS steps from project-specific processing.

1. **Study metadata and assemblies** – harmonise isolate IDs and phenotype data.
2. **AMR features** – run AMRFinderPlus and build presence/absence matrices.
3. **GWAS** – generate/parse pyseer outputs, map significant unitigs/features, and integrate overlapping evidence.
4. **Feature sets** – construct AMR-only, locus-GWAS and hybrid matrices without using external-validation outcomes for feature selection.
5. **Model training** – train per-antibiotic quantitative MIC models on the regional study collection.
6. **External validation** – transform the independent collection using the locked feature definitions and evaluate without retraining.
7. **Global application** – apply locked models to the global collection and summarise model consensus.
8. **Resistance Architecture evidence** – integrate carbapenemase, mobile-element, GWAS and genomic-context evidence; do not equate ISAba1 presence with physical promoter linkage.
9. **Temporal analysis** – use recombination-filtered lineage trees and documented BactDating/SkyGrowth settings.
10. **Figures/source data** – generate paper panels from deposited source tables.

### Required release metadata still to lock

- AMRFinderPlus executable and database versions.
- BactDating and SkyGrowth versions/settings and date-randomisation provenance.
- Exact external-validation consensus high-MIC endpoint outputs.
- Final NCBI BioProject/BioSample/SRA accession mapping.
