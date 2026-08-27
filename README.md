# Acinetobacter resistance architectures

Reproducible analysis code accompanying the study of recurrent carbapenem Resistance Architectures in *Acinetobacter baumannii*.

This repository contains **project-specific** analysis glue, GWAS processing, validation, global prediction, genomic-context analysis and figure-generation scripts. The reusable modelling framework is maintained separately as [BAMPS](https://github.com/Benizao1980/BAMPS) (Bacterial AMR Modelling and Prediction Suite).

## Data scope

The analysis integrates:
- the Eastern European clinical isolate collection;
- an independent *A. baumannii* validation collection;
- a global comparative genome collection;
- AMRFinderPlus resistance determinants;
- pangenome/GWAS-derived features;
- quantitative carbapenem MICs;
- time-scaled phylogenetic analyses.

Large source datasets and model outputs should be deposited on Figshare/NCBI rather than committed to GitHub. Raw reads belong in SRA.

## Repository layout

- `scripts/model/` – exact modelling/prediction scripts used by the project.
- `scripts/gwas/` – GWAS integration, mapping, filtering and annotation.
- `scripts/validation/` – external validation and IC-assignment utilities.
- `scripts/global/` – global prediction summaries and maps.
- `scripts/context/` – genomic-context analyses, including ISAba1/OXA analysis.
- `scripts/figures/` – paper-specific figure builders.
- `archive/` – dated snapshot of the latest working scripts as received, retained for provenance only.
- `source_data/` – lightweight source-data manifests only; large data are external.

## Important evidence convention

ISAba1 whole-genome detection is kept separate from physical ISAba1–blaOXA linkage. The short-read assembly-context analysis does not establish adjacency for the focal OXA-23 loci; manuscript and figure code should therefore use association/context language rather than promoter-activation claims unless independently demonstrated.

## Reproducibility

See `docs/workflow.md`, `docs/data_manifest.md`, `FIGURE_PROVENANCE.md` and `software_versions.tsv`. Exact release versions of BAMPS, AMRFinderPlus, pyseer, BactDating and SkyGrowth should be recorded in the manuscript and release metadata.

## Licence

GPL-3.0.

## Manuscript-era modelling snapshots

`scripts/model/` intentionally retains the project versions of modelling scripts used during manuscript development. These may overlap with BAMPS. They are preserved here for analysis provenance; new/general development should occur in BAMPS rather than by independently extending these copies.

See `FIGURE_PROVENANCE.md` and `software_versions.tsv` for the remaining release-lock items.
