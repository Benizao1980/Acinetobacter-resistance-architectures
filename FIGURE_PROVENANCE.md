# Figure provenance

This table is the release-facing map between manuscript figures, project scripts and deposited source data. File names should be updated only when the final manuscript figure set is locked.

| Figure | Analysis / content | Primary script(s) | Required source data / outputs | Release status |
|---|---|---|---|---|
| Fig. 1 | Study population structure and sampling | project-specific phylogeny/plotting workflow; final script to identify | study metadata, recombination-filtered phylogeny, IC/BAPS assignments | **TO LOCK** |
| Fig. 2 | Carbapenem phenotype/genotype relationships | `scripts/model/plot_predicted_mic_panel.py` plus final phenotype plotting source | isolate-level MICs, AMRFinderPlus calls | **TO LOCK exact panel builder** |
| Fig. 3 | GWAS and Resistance Architecture evidence | `scripts/gwas/build_master_gwas_evidence.py`; `scripts/gwas/integrate_gwas_evidence.py`; `scripts/figures/build_figure3_final.py` | pyseer outputs, mapped annotations, prioritised evidence tables | mapped |
| Fig. 4 | Global BAMPS prediction / consensus | `scripts/figures/build_global_prediction_figure.py`; `scripts/global/make_global_prediction_maps.py`; `scripts/global/summarise_global_prediction_consensus.py` | locked models, global feature matrix, sample-level predictions and metadata | mapped; external consensus endpoint **TO LOCK** |
| Fig. 5 | Dated lineage analyses | final R/BactDating/SkyGrowth scripts to identify | recombination-filtered lineage trees, sampling dates, posterior summaries | **TO LOCK** |
| Supp. Fig. S4 | OXA-23 / ISAba1 MIC context | `scripts/context/isaaba1_oxa_context.py` plus phenotype plotting source | assemblies, AMRFinderPlus calls, MICs | context script mapped; plot script **TO LOCK** |
| Supp. Fig. S5 | GWAS breakdown | `scripts/gwas/plot_gwas_manhattan.py`; `scripts/gwas/plot_gwas_single_panel.py` | pyseer/GWAS source tables | mapped |
| Supp. Fig. S6 | Global prediction summaries | `scripts/global/build_global_prediction_geography_time_figure.py`; `scripts/global/summarise_global_prediction_by_geography_time.py` | global sample-level predictions and metadata | mapped |
| Supp. Fig. S7 | Dating QC | final R dating/QC scripts to identify | dating posterior/QC/date-randomisation outputs | **TO LOCK** |

## Rule for release

Every manuscript panel should have (1) a named generating script or notebook, (2) a deposited numerical source table or documented upstream file, and (3) the software/version information required to rerun it. Rows marked **TO LOCK** should be resolved before the Zenodo manuscript release.
