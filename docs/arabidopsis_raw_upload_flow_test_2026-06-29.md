# Arabidopsis Raw Upload Flow Test, 2026-06-29

This test simulates a real website user workflow with Arabidopsis raw input files.

## Input Preparation

Script:

```powershell
python scripts\webapp\prepare_arabidopsis_raw_upload_test.py --out-dir webapp_data\jobs\arabidopsis_raw_upload_test_20260629 --n-genes 120
```

Raw source files used:

- Araport11 CDS FASTA
- Araport11 protein FASTA
- Ensembl Plants release 63 Arabidopsis GFF3
- TAIR ATH_GO_GOSLIM annotation
- STRING Arabidopsis protein links
- TAIR-to-UniProt mapping table

Generated website-style upload files:

- `protein.fasta`: 120 records
- `cds.fasta`: 120 records
- `annotation.gff3`: 5,293 non-comment rows
- `go_annotation.tsv`: 1,186 rows
- `ppi_edges.tsv`: 88,462 edges

Expression matrix and domain annotation were not found in the local raw Arabidopsis sources, so they were not included in this upload-flow test.

## Feature Extraction

Command:

```powershell
python scripts\feature_extraction\raw_upload_to_profile_features.py `
  --input-dir webapp_data\jobs\arabidopsis_raw_upload_test_20260629 `
  --profile-dir models\deployable_feature_profiles\sequence_plm_go_ppi `
  --out-prefix webapp_data\jobs\arabidopsis_raw_upload_test_20260629\ath_raw_sequence_plm_go_ppi `
  --plm-dir E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\plm_embeddings `
  --go-obo E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\external_raw_stable\go-basic.obo
```

Output:

- genes: 120
- features: 6,727
- missing fraction: 0.0000074

## Prediction

Command:

```powershell
python scripts\prediction\predict_from_profile_features.py `
  --features webapp_data\jobs\arabidopsis_raw_upload_test_20260629\ath_raw_sequence_plm_go_ppi.features.npz `
  --model models\deployable_feature_profiles\sequence_plm_go_ppi\model.joblib `
  --out webapp_data\jobs\arabidopsis_raw_upload_test_20260629\ath_raw_sequence_plm_go_ppi.predictions.tsv
```

Prediction summary:

- genes predicted: 120
- classification threshold: 0.4771289667113617
- predicted essential: 15
- predicted non-essential: 105
- mean essential probability: 0.1745
- median essential probability: 0.0827
- max essential probability: 0.8788

Only LightGBM feature-name warnings were emitted. They are expected because the stored LightGBM estimators receive NumPy arrays after trained preprocessing transforms.

## Current Limitation

This flow used precomputed Arabidopsis PLM embeddings to simulate the PLM extraction stage. For arbitrary uploaded species, the server still needs a production PLM embedding backend or an external embedding-upload option.
