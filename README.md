# PlantEssentialGenePredictor

PlantEssentialGenePredictor is a reproducible plant essential-gene prioritization framework for *Arabidopsis thaliana* and rice (*Oryza sativa*). It uses a shared 6,751-dimensional feature space composed of 95 biological features and 6,656 protein language model embeddings from ESM2, ProtBERT and ProtT5.

The repository provides:

- processed feature matrices, not raw GO/PPI/expression/database downloads;
- trained Arabidopsis, rice and joint Arabidopsis-rice models;
- fixed train/validation/test labels and model-evaluation outputs;
- scripts for feature processing, model training, prediction and manuscript analyses;
- a Streamlit web app for prediction from processed feature matrices.

## Main Released Predictions

The four released prediction tables are in `predictions/`.

| File | Model | Genes predicted |
|---|---|---:|
| `arabidopsis_unknown20460_single_model_predictions.tsv` | Arabidopsis single-species strict2601 model | 20,460 Arabidopsis unknown genes |
| `arabidopsis_unknown20460_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | 20,460 Arabidopsis unknown genes |
| `rice_unknown_all_single_model_predictions.tsv` | Rice strict399 + Tos17 N4 single-species model | 34,215 rice genes |
| `rice_unknown_all_joint_model_predictions.tsv` | Joint Arabidopsis-rice model | 34,215 rice genes |

Summary counts are in `predictions/prediction_summary.tsv`.

## Two Ways To Use This Release

Before using the models or processed feature matrices, clone the repository with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
git lfs pull
```

### Option 1: Use the processed features and train or predict yourself

Processed feature matrices are stored in `data/processed_features/` as compressed NumPy `.npz` files:

- `rice_common6751_all_genes.npz`
- `arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz`
- `common6751_feature_names.tsv`

Each `.npz` contains:

- `X`: feature matrix with 6,751 columns;
- `gene_id`: gene IDs;
- `transcript_id` or `sequence_id` when available;
- `feature_names`;
- `n_bio = 95`.

Example prediction command:

```bash
python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/rice_common6751_all_genes.npz \
  --model rice_single \
  --out predictions/rice_single_rerun.tsv
```

Available models:

- `arabidopsis_single`
- `rice_single`
- `joint`

The released training scripts are provided for reproducibility and inspection. Some full retraining workflows require source database downloads or intermediate feature files that are not redistributed in this repository.

### Option 2: Use the trained models directly

The trained model folders are:

- `models/arabidopsis_single_strict2601_common6751`
- `models/rice_single_strict399_Tos17N4_common6751`
- `models/joint_arabidopsis_rice_common6751`

Use the same `predict_from_processed_features.py` script with a processed feature `.npz` file. The script applies the training-time preprocessing, base learners and stacking or model-selection rule.

## Web App

Run locally:

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

The app accepts a processed `.npz` feature matrix matching the released common6751 schema. It does not download raw GO/PPI/expression data or run PLM embedding extraction online.

## Feature and Label Notes

The common model feature space is:

- 95 shared biological features;
- 2,560 ESM2 embedding features;
- 2,048 ProtBERT embedding features;
- 2,048 ProtT5 embedding features.

For Arabidopsis unknown genes, annotation-derived biological features that were not available were left as missing values and handled by the trained model imputers. This mirrors the released unknown-gene prediction pipeline.

Raw GO, PPI, expression, phenotype database dumps and PLM intermediate files are intentionally not included. Use the scripts in `scripts/feature_extraction/` to rebuild features from official sources.

## Large Files

Processed feature matrices and model bundles are stored with Git LFS:

```bash
git lfs install
git lfs pull
```

If GitHub LFS quota becomes limiting, future releases should deposit large `.npz` and `.joblib` files on Zenodo, Figshare or OSF and keep GitHub as the code repository.

## Citation

If you use this resource, cite the associated manuscript and the source databases and PLM models described in the paper.
