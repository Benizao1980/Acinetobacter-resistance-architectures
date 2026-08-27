# Analysis workflow

This repository freezes the **project-specific** analysis used for the *Acinetobacter baumannii* resistance-architecture study. Reusable model training and prediction functionality is maintained in [BAMPS](https://github.com/Benizao1980/BAMPS). The scripts retained here are manuscript-era snapshots so that future BAMPS changes do not silently alter the published analysis.

## 1. Harmonise study inputs

Required inputs are isolate metadata, assemblies, quantitative MIC measurements and stable isolate identifiers. Large/private inputs are not committed to GitHub; see `docs/data_manifest.md`.

## 2. AMR determinant features

Run AMRFinderPlus on the study assemblies and retain both the raw per-isolate output and the presence/absence feature matrix. Generic feature construction is available in BAMPS; the manuscript-specific snapshot is retained under `scripts/model/`.

**Release requirement:** record the AMRFinderPlus executable version and database version/date in `software_versions.tsv`.

## 3. GWAS evidence

Project scripts in `scripts/gwas/` parse pyseer output, map unitigs/features to annotated loci, integrate imipenem and meropenem evidence and construct the prioritised evidence tables used for the Resistance Architecture interpretation.

Key scripts include:

- `build_master_gwas_evidence.py`
- `build_raw_pyseer_feature_ranking_v2.py`
- `integrate_gwas_evidence.py`
- `filter_gwas_loci.py`
- `parse_gwas_annotations.py`
- `select_gwas_features.py`

External-validation outcomes must **not** be used for feature selection.

## 4. Model training

The regional study collection is used to train per-antibiotic quantitative MIC models. `scripts/model/` retains the exact project snapshots, including classifier comparison and locked-XGBoost tuning. BAMPS is the maintained generic implementation.

For a publication release retain the exact feature matrix, phenotype table, random seeds, split definitions, fitted model metadata and command lines.

## 5. Independent validation

Scripts in `scripts/validation/` harmonise validation metadata, assign international-clone context and characterise prediction error. Validation data are transformed with the **locked training feature definition** and are not used to retrain or select features.

The final manuscript release should deposit sample-level observed/predicted values and the exact consensus high-MIC endpoint used in Fig. 4.

## 6. Global application

Scripts in `scripts/global/` apply/summarise locked model predictions across the global collection and produce geography/time and model-consensus summaries. The global collection is opportunistically sampled; outputs should be described as patterns among sampled genomes rather than population burden.

## 7. ISAba1 / OXA genomic context

`scripts/context/isaaba1_oxa_context.py` separates whole-genome ISAba1 detection from physical linkage to OXA loci. For the focal short-read assemblies, absence of same-contig evidence is treated as **unresolved linkage**, particularly where the promoter-facing OXA locus lies near a contig edge. ISAba1 presence must not be presented as demonstrated promoter activation.

## 8. Resistance Architecture evidence

Resistance Architectures integrate defining carbapenemases with reproducible GWAS/mobile-element/context evidence. Association is distinguished from mechanism: labels such as activation, stabilisation or altered efflux require direct supporting evidence and should not be inferred from GWAS alone.

## 9. Temporal analysis

Time-scaled lineage analyses use recombination-filtered trees and sampling dates. Before manuscript release, archive the exact BactDating/SkyGrowth versions, R scripts, MCMC/settings, posterior summaries and date-randomisation/QC outputs. Do not present acquisition/loss events as reconstructed ancestral events unless the reconstruction method is explicitly documented.

## 10. Figures and source data

See `FIGURE_PROVENANCE.md` for the figure-to-script/source-data map. Numerical source data and large derived outputs should be deposited externally (for example Figshare/Zenodo/NCBI as appropriate), with stable links recorded in `docs/data_manifest.md`.

## Release gate

The repository is ready for a manuscript-linked Zenodo release when:

- all rows in `software_versions.tsv` are locked;
- every manuscript figure has a provenance mapping;
- external validation sample-level outputs are deposited;
- BioProject/BioSample/SRA accessions are final;
- no private paths, credentials or unresolved `TO_LOCK` items remain in publication-facing documentation.
