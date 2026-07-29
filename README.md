# PlantEssentialGenePredictor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21387076.svg)](https://doi.org/10.5281/zenodo.21387076)

PlantEssentialGenePredictor is a reproducible plant essential-gene prioritization framework for *Arabidopsis thaliana* and rice (*Oryza sativa*). It uses a shared 6,751-dimensional feature space composed of 95 biological features and 6,656 protein language model embeddings from ESM2, ProtBERT and ProtT5.

The repository provides:

- processed feature matrices, not raw GO/PPI/expression/database downloads;
- trained Arabidopsis, rice and joint Arabidopsis-rice models;
- deployable feature-profile models for different uploaded annotation combinations;
- fixed train/validation/test labels and model-evaluation outputs;
- scripts for feature processing, model training, prediction and manuscript analyses;
- a Streamlit web app for prediction from processed feature matrices or user-uploaded raw biological files.

## Main Released Predictions

The original four model-output tables remain in the archive for exact model
reproduction. They are **not** interchangeable with a pure unknown-gene
resource because the historical Arabidopsis feature-covered table contains
genes later classified as modelling labels or phenotype-recorded genes.

For biological interpretation and candidate selection, use the audited release
tables under `predictions/publication_release/`. Every feature-covered gene is
assigned exactly one status:

- `known_label_used_in_study`;
- `pseudo_label_used_in_study`;
- `phenotype_recorded_but_excluded`;
- `true_unknown_candidate`.

Only `true_unknown_candidate` is eligible for the frozen candidate resource.
The registry used for this exclusion is
`results/tpc_candidate_resource/study_label_and_phenotype_registry.tsv` and
the frozen file checksums are in
`results/tpc_candidate_resource/frozen_submission_inputs.json`.

| Audited file | Scope | Genes with `true_unknown_candidate` status |
|---|---|---:|
| `arabidopsis_all_feature_covered_genes_reclassified.tsv` | Arabidopsis, single-species + joint + annotation-light probabilities | 17,522 |
| `rice_all_feature_covered_genes_reclassified.tsv` | Rice, single-species + joint + annotation-light probabilities | 17,457 |

The release also contains two frozen 10-gene core candidate panels, their
ensemble-component robustness, closest-labelled-homolog audit, and
provenance-aware independent-evidence card templates. The `homology_isolated`
and `homology_supported_by_essential` categories are presentation strata, not
new phenotype labels.

### Historical model-output files

| File | Model | Genes predicted |
|---|---|---:|
| `arabidopsis_unknown20460_single_model_predictions.tsv` | Arabidopsis single-species strict2601 model | 20,460 Arabidopsis unknown genes |
| `arabidopsis_unknown20460_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | 20,460 Arabidopsis unknown genes |
| `rice_unknown_all_single_model_predictions.tsv` | Rice strict399 + Tos17 N4 single-species model | rice genome-scale prediction set |
| `rice_unknown_all_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | rice genome-scale prediction set |

Summary counts are in `predictions/prediction_summary.tsv`.

## Independent validation protocol

No post hoc candidate hit rate is reported as an unbiased performance metric.
The repository includes a locked external phenotype-cohort protocol:

- `docs/independent_validation_protocol.md`;
- `data/external_validation/independent_phenotype_cohort_template.tsv`;
- `scripts/publication/evaluate_external_phenotype_cohort.py`;
- `docs/candidate_evidence_card_guidance.md`.

The evaluator rejects study-label and pseudo-label genes, requires a stable
source/date/assay/phenotype record for each included gene, and reports frozen-
threshold AUC, AUPRC and classification metrics with stratified bootstrap 95%
confidence intervals. Candidate evidence cards are discovery-oriented and
clearly separated from the locked quantitative cohort.

### Current independent-evidence status

The public release now includes an auditable Europe PMC source-screening ledger
(312 articles; 919 gene-level source records), 19 curator-checked direct
Arabidopsis LoF records and their zero-overlap audit. Sixteen records (nine
essential and seven viable/non-essential) meet the phenotype-adjudication rules
for a prelocked Arabidopsis cohort. This is below the pre-registered minimum of
30 genes per species and 10 genes per class; no external AUC, AUPRC or
threshold metric is reported. No rice record has yet completed curator locking.

The lightweight release tables are in `data/external_validation/release/`; the
complete source-screening policy and reproduction instructions are in
`docs/external_validation_release_notes.md`. The manuscript therefore treats
the current evidence as qualitative candidate context, not a second unbiased
performance test.

## Code and Data Availability

Code, documentation and the Streamlit web application are hosted on GitHub:

```bash
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
```

Large artifacts are archived on Zenodo:

```text
DOI: 10.5281/zenodo.21387076
```

The prepared `v1.1-candidate-audit` supplement is staged for a Zenodo version
update before manuscript submission. It contains the audited publication tables
and independent-validation protocol, without duplicating raw databases or PLM
weights. See `docs/zenodo_release.md`.

Download the Zenodo artifact and extract it into the repository root so that
`models/`, `data/processed_features/`, `data/labels/` and `predictions/` are
present.

## Two Ways To Use This Release

### Option 1: Use processed features and trained models

After downloading the Zenodo artifact, processed feature matrices are available
in `data/processed_features/` as compressed NumPy `.npz` files. Each `.npz`
contains:

- `X`: feature matrix;
- `gene_id`: gene IDs;
- optional `transcript_id` or `sequence_id`;
- optional `feature_names`;
- optional `n_bio`.

Example prediction command:

```bash
python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/rice_common6751_all_genes.npz \
  --model rice_single \
  --out predictions/rice_single_rerun.tsv
```

Available full models:

- `arabidopsis_single`;
- `rice_single`;
- `joint`.

### Option 2: Use raw-upload web prediction

The Streamlit app can accept raw biological input files and construct the matching deployable feature profile. Users do not need to manually build a 6751-dimensional table.

Required or optional uploads:

- `protein.fasta`, required for protein sequence features and PLM embeddings;
- `cds.fasta`, strongly recommended for CDS composition features;
- `annotation.gff3`, optional for gene structure and transcript mapping;
- `go_annotation.tsv`, optional;
- `ppi_edges.tsv`, optional;
- `expression_matrix.tsv`, optional;
- `domain_annotation.tsv`, optional.

The joint deployable model provides feature-profile choices matching available annotations:

- sequence + PLM;
- sequence + PLM + GO;
- sequence + PLM + PPI;
- sequence + PLM + expression;
- sequence + PLM + GO + PPI;
- sequence + PLM + GO + expression;
- sequence + PLM + PPI + expression;
- sequence + PLM + GO + PPI + expression;
- advanced full uploaded-feature profile.

This design avoids treating missing GO, PPI or expression annotations as true biological zeros.

## Web App

Run locally:

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

On a server, run the app behind Nginx or another reverse proxy and point the domain DNS A record to the server IP. See `docs/webapp_deployment.md` and `docs/local_server_quickstart.md`.

For an Ubuntu production server with systemd, Nginx and HTTPS, use
`docs/linux_server_deployment.md` and the files under `deploy/linux/`.

The app supports:

- full-model prediction from processed `.npz` matrices;
- raw-upload prediction from FASTA plus optional annotation files;
- browsing and downloading released Arabidopsis and rice prediction tables;
- downloading known-label tables and input templates;
- public caching of final species-level prediction results;
- temporary job directories for uploaded data, with cleanup scripts in `scripts/webapp/`.

The server does not download GO, PPI, expression or domain annotations for users. These files must be uploaded by the user when those feature blocks are selected. Protein language model embeddings can be extracted locally when the bundled PLM weights are present under `../plm_model_weights`.

For a server that accepts raw FASTA from new species, first download the PLM
weights once as the server administrator. The download is intentionally manual
because ESM2, ProtBERT and ProtT5 require substantial disk space:

```bash
python scripts/feature_extraction/download_plm_weights.py \
  --weights-root ../plm_model_weights
```

The application automatically detects `../plm_model_weights`. To use a
different location, set `PLANT_EG_PLM_WEIGHTS` before starting Streamlit.
`PLANT_EG_PRECOMPUTED_PLM` and `PLANT_EG_GO_OBO` optionally configure cached
Arabidopsis/rice embeddings and a GO ontology file, respectively.

## Feature and Label Notes

The full common model feature space is:

- 95 shared biological features;
- 2,560 ESM2 embedding features;
- 2,048 ProtBERT embedding features;
- 2,048 ProtT5 embedding features.

Raw GO, PPI, expression, phenotype database dumps and large PLM intermediate files are intentionally separated from the lightweight GitHub code release. The portable server package may contain bundled PLM model weights and reference assets for local deployment. Use the scripts in `scripts/feature_extraction/` to rebuild features from official sources.

## Large Files

GitHub is used for code and lightweight documentation. Zenodo is used for large
research artifacts:

- trained `.joblib` model bundles;
- processed `.npz` feature matrices;
- fixed label tables;
- genome-scale prediction tables;
- file manifests and SHA256 checksums.

See `docs/zenodo_release.md` for the release plan and upload checklist.

## Citation

If you use this resource, cite the associated manuscript and the source databases and PLM models described in the paper.
