# Website deployment

The Streamlit application supports two production workflows:

1. prediction from a released or user-supplied 6,751-column processed `.npz` matrix;
2. raw biological uploads, from which the server derives the matching joint-model feature profile before prediction.

## Install the code and released artifacts

```bash
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
pip install -r requirements.txt
```

Download the Zenodo release (version DOI `10.5281/zenodo.21387076`) and extract its contents into this repository root. The directories `models/`,
`data/processed_features/`, `data/labels/` and `predictions/` must then exist.

## Configure raw FASTA prediction

Raw prediction uses ESM2, ProtBERT and ProtT5. The checkpoint files are not
redistributed in the Zenodo artifact. Download them once on the server:

```bash
python scripts/feature_extraction/download_plm_weights.py \
  --weights-root ../plm_model_weights
```

This creates the directory layout expected by the app:

```text
../plm_model_weights/
  esm2/esm2_t33_650M_UR50D.pt
  huggingface_hub/models--Rostlab--prot_bert/
  huggingface_hub/models--Rostlab--prot_t5_xl_uniref50/
```

The initial PLM download can require more than 10 GB of storage. It is an
administrator task, not a per-user download. Set these optional environment
variables when assets live elsewhere:

```text
PLANT_EG_PLM_WEIGHTS=/path/to/plm_model_weights
PLANT_EG_PRECOMPUTED_PLM=/path/to/precomputed_plm_embeddings/ath_rice
PLANT_EG_GO_OBO=/path/to/go-basic.obo
```

`PLANT_EG_PRECOMPUTED_PLM` is only appropriate for matching cached
Arabidopsis/rice records. For a new species, select online PLM extraction in
the web app.

## Start the server

```bash
streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
```

Place Nginx or another reverse proxy in front of Streamlit for a public
domain. Configure HTTPS at the reverse proxy and set an upload-size limit that
matches the expected FASTA and annotation files.

## Upload requirements

`protein.fasta` is required. `cds.fasta`, `annotation.gff3`,
`go_annotation.tsv`, `ppi_edges.tsv`, `expression_matrix.tsv` and
`domain_annotation.tsv` are optional. The app validates each file and selects
the joint feature-profile model that matches the uploaded annotation blocks.
It never represents unavailable GO, PPI or expression data as biological zero.

Exact templates and field definitions are available in the app and in
`docs/input_feature_preparation.md`.

## Privacy and cleanup

Private uploads are written to `webapp_data/jobs/`. Remove expired jobs using:

```bash
python scripts/webapp/cleanup_jobs.py --max-age-hours 24
```

Final species-level prediction tables enter `webapp_data/public_predictions/`
only when a user explicitly opts into public sharing.
