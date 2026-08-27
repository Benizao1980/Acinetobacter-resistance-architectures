# Data manifest

Large clinical/genomic datasets are not committed to GitHub. This repository contains code, lightweight manifests and provenance only.

| Dataset | Public destination | Identifier / status | GitHub content |
|---|---|---|---|
| Eastern European raw reads | NCBI SRA | BioProject **PRJNA1518973**; BioSample/SRA accessions pending | accession/manifest only |
| Eastern European BioSamples | NCBI BioSample | submission in preparation | metadata/accession manifest only |
| Study assemblies / metadata | NCBI/pubMLST/Figshare as appropriate | final links to lock | lightweight manifest / source-data tables |
| Independent validation collection | public repository cited in manuscript | exact accession/link to lock | provenance/link only |
| Global comparative collection | Figshare/public genome accessions | final Figshare DOI to lock | genome/accession manifest only |
| Full GWAS evidence | Figshare | DOI to lock | scripts + prioritised summary |
| Sample-level ML predictions | Figshare | DOI to lock | scripts + compact source tables if practical |
| BactDating/SkyGrowth outputs | Figshare/Zenodo | DOI to lock | scripts/configuration + summaries |
| Figure numerical source data | Figshare | DOI to lock | compact copies may also be retained here |

## Sensitive-data rule

Do not commit raw patient identifiers, credentials, private download tokens or non-public clinical metadata. Public manifests should use the study isolate identifiers used in the manuscript and public archives.
