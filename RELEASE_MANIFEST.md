# Release manifest

This manifest separates the lightweight GitHub code release from large artifacts deposited on Zenodo.

## Code and documentation kept on GitHub

These files should be committed directly to GitHub:

- `README.md`
- `LICENSE`
- `requirements.txt`
- `.zenodo.json`
- `prediction_manifest.json`
- `release_metadata.json`
- `docs/`
- `scripts/`
- `webapp/`
- `.streamlit/`

## Artifacts deposited on Zenodo

These directories contain trained model files, processed feature matrices, labels and prediction outputs:

- `models/arabidopsis_single_strict2601_common6751/`
- `models/rice_single_strict399_Tos17N4_common6751/`
- `models/joint_arabidopsis_rice_common6751/`
- `models/deployable_feature_profiles/`
- `data/processed_features/`
- `data/labels/`
- `predictions/`
- `results/tpc_candidate_resource/` (audited candidate registry, frozen-input manifest, stability and homology screens, and evidence-card templates)
- `data/external_validation/` (locked independent phenotype-cohort schema)

The historical model-output tables are:

- `predictions/arabidopsis_unknown20460_single_model_predictions.tsv`
- `predictions/arabidopsis_unknown20460_joint_model_predictions.tsv`
- `predictions/rice_unknown_all_single_model_predictions.tsv`
- `predictions/rice_unknown_all_joint_model_predictions.tsv`

The publication-facing audited tables are:

- `predictions/publication_release/arabidopsis_all_feature_covered_genes_reclassified.tsv`
- `predictions/publication_release/rice_all_feature_covered_genes_reclassified.tsv`
- `results/tpc_candidate_resource/study_label_and_phenotype_registry.tsv`
- `results/tpc_candidate_resource/frozen_submission_inputs.json`

Only records with `candidate_status=true_unknown_candidate` should be used as
novel candidate genes. Historical model-output filenames are retained for
reproducibility and should not be interpreted as a pure unknown-gene release.

## Portable server-only artifacts

These are useful for a local or institutional server but should not usually be pushed to GitHub or included in the main Zenodo deposition:

- `../plm_model_weights/`
- `precomputed_plm_embeddings/`
- `raw_data/`
- `.venv/`
- `webapp_data/jobs/`
- `webapp_data/logs/`

The `webapp_data/jobs/` and `webapp_data/logs/` directories are runtime outputs and can be deleted safely after results have been archived.
