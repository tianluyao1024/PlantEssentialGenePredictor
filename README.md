# PlantEssentialGenePredictor

PlantEssentialGenePredictor is a reproducible plant essential-gene prioritization framework for *Arabidopsis thaliana* and rice (*Oryza sativa*). It uses a shared 6,751-dimensional feature space composed of 95 biological features and 6,656 protein language model embeddings from ESM2, ProtBERT and ProtT5.

The repository provides:

- processed feature matrices, not raw GO/PPI/expression/database downloads;
- trained Arabidopsis, rice and joint Arabidopsis-rice models;
- deployable feature-profile models for different uploaded annotation combinations;
- fixed train/validation/test labels and model-evaluation outputs;
- scripts for feature processing, model training, prediction and manuscript analyses;
- a Streamlit web app for prediction from processed feature matrices or user-uploaded raw biological files.

## Main Released Predictions

The four released prediction tables are archived on Zenodo under `predictions/`.

| File | Model | Genes predicted |
|---|---|---:|
| `arabidopsis_unknown20460_single_model_predictions.tsv` | Arabidopsis single-species strict2601 model | 20,460 Arabidopsis unknown genes |
| `arabidopsis_unknown20460_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | 20,460 Arabidopsis unknown genes |
| `rice_unknown_all_single_model_predictions.tsv` | Rice strict399 + Tos17 N4 single-species model | rice genome-scale prediction set |
| `rice_unknown_all_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | rice genome-scale prediction set |

Summary counts are in `predictions/prediction_summary.tsv`.

## Code and Data Availability

Code, documentation and the Streamlit web application are hosted on GitHub:

```bash
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
```

Large artifacts are archived on Zenodo:

```text
DOI: 10.5281/zenodo.XXXXXXX
```

Replace the placeholder DOI after the Zenodo deposition is published. Download
the Zenodo artifact and extract it into the repository root so that `models/`,
`data/processed_features/`, `data/labels/` and `predictions/` are present.

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

The app supports:

- full-model prediction from processed `.npz` matrices;
- raw-upload prediction from FASTA plus optional annotation files;
- browsing and downloading released Arabidopsis and rice prediction tables;
- downloading known-label tables and input templates;
- public caching of final species-level prediction results;
- temporary job directories for uploaded data, with cleanup scripts in `scripts/webapp/`.

The server does not download GO, PPI, expression or domain annotations for users. These files must be uploaded by the user when those feature blocks are selected. Protein language model embeddings can be extracted locally when the bundled PLM weights are present under `../plm_model_weights`.

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
