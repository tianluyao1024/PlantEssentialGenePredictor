# Arabidopsis raw-upload flow test

This documented test creates realistic website upload files from Arabidopsis
source data, constructs the joint `sequence_plm_go_ppi` profile and predicts
essentiality. It is a reproducibility example; paths are supplied by the user
and no local machine paths are embedded in the script.

## Prepare website-style input files

```powershell
python scripts\webapp\prepare_arabidopsis_raw_upload_test.py `
  --out-dir webapp_data\jobs\arabidopsis_raw_upload_test `
  --n-genes 120 `
  --cds <Araport11_cds.fasta> `
  --protein <Araport11_pep.fasta> `
  --gff <Arabidopsis.gff3.gz> `
  --go <TAIR_GO_annotation.txt> `
  --string-links <STRING_links.txt> `
  --tair-uniprot <TAIR_to_UniProt.tsv>
```

The output directory contains `protein.fasta`, `cds.fasta`,
`annotation.gff3`, `go_annotation.tsv`, `ppi_edges.tsv` and an
`upload_manifest.json`. These are the same file types accepted by the raw-data
prediction page.

## Create PLM embeddings

For a fresh server, download checkpoints once:

```powershell
python scripts\feature_extraction\download_plm_weights.py --weights-root ..\plm_model_weights
```

Then extract the three PLM blocks for the raw protein FASTA:

```powershell
python scripts\feature_extraction\extract_plm_embeddings_from_fasta.py `
  --protein-fasta webapp_data\jobs\arabidopsis_raw_upload_test\protein.fasta `
  --out-dir webapp_data\jobs\arabidopsis_raw_upload_test\online_plm_embeddings `
  --weights-root ..\plm_model_weights `
  --device auto `
  --batch-size 4
```

## Construct features and predict

```powershell
python scripts\feature_extraction\raw_upload_to_profile_features.py `
  --input-dir webapp_data\jobs\arabidopsis_raw_upload_test `
  --profile-dir models\deployable_feature_profiles\sequence_plm_go_ppi `
  --out-prefix webapp_data\jobs\arabidopsis_raw_upload_test\ath_raw_sequence_plm_go_ppi `
  --plm-dir webapp_data\jobs\arabidopsis_raw_upload_test\online_plm_embeddings `
  --go-obo <go-basic.obo>

python scripts\prediction\predict_from_profile_features.py `
  --features webapp_data\jobs\arabidopsis_raw_upload_test\ath_raw_sequence_plm_go_ppi.features.npz `
  --model models\deployable_feature_profiles\sequence_plm_go_ppi\model.joblib `
  --out webapp_data\jobs\arabidopsis_raw_upload_test\ath_raw_sequence_plm_go_ppi.predictions.tsv
```

The Streamlit raw-upload page performs these same operations after validating
the uploaded files. Cached Arabidopsis/rice embeddings may be used only for
matching reference records; use online extraction for new species.
