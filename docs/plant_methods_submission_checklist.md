# Plant Methods submission checklist

## Website and software

- [x] Streamlit web interface is included in `webapp/app.py`.
- [x] Processed `.npz` prediction mode is available.
- [x] Raw-upload mode is available for FASTA plus optional GO, PPI, expression, domain and GFF3 files.
- [x] Input templates are included under `docs/input_templates/`.
- [x] Model card is included.
- [x] Deployment instructions are included.
- [ ] Public domain and HTTPS reverse proxy are configured on the final server.
- [ ] Final public URL is added to README and manuscript.

## Data and model release

- [x] Arabidopsis single-model unknown-gene predictions are included in the Zenodo artifact.
- [x] Arabidopsis joint-model unknown-gene predictions are included in the Zenodo artifact.
- [x] Rice single-model genome-scale predictions are included in the Zenodo artifact.
- [x] Rice joint-model genome-scale predictions are included in the Zenodo artifact.
- [x] Known-label tables are included in the Zenodo artifact.
- [x] Main model directories are included in the Zenodo artifact.
- [x] Deployable feature-profile models are included in the Zenodo artifact.
- [ ] GitHub code repository is pushed.
- [x] Zenodo deposition is published: version DOI `10.5281/zenodo.21387076`;
  concept DOI `10.5281/zenodo.21387075`.

## Manuscript reproducibility

- [x] Fixed train/validation/test split tables are included.
- [x] Training scripts are included under `scripts/training/`.
- [x] Prediction scripts are included under `scripts/prediction/`.
- [x] Feature-extraction scripts are included under `scripts/feature_extraction/`.
- [x] Source data and input preparation notes are included under `docs/`.
- [ ] Final repository commit hash and archived DOI are added to the manuscript.
