# Website deployment notes

The Streamlit website is designed for a local or institutional server. It has
two layers:

1. Full-model prediction from processed `common6751` feature matrices.
2. A drafted FASTA validation interface for the future annotation-light model.

The current release does not run ESM2, ProtBERT or ProtT5 online. It predicts
from processed `.npz` feature matrices and the released trained models.

## Run locally

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

## Input modes

### Full-model mode

Upload a compressed NumPy `.npz` file with:

- `X`: numeric matrix with shape `n_genes x 6751`;
- `gene_id`: gene identifiers;
- optional `transcript_id` or `sequence_id`;
- optional `feature_names` and `n_bio`.

The app checks that the matrix has 6,751 columns, runs the selected model and
returns a table with:

- `gene_id`;
- optional transcript or sequence identifier;
- `essential_probability`;
- `classification_threshold`;
- `predicted_label`;
- `predicted_class`;
- `model_name`.

### FASTA mode

The FASTA mode currently validates CDS and protein FASTA files and reports:

- record count;
- duplicate IDs;
- invalid records;
- sequence length summary;
- SHA-256 checksum.

Probability prediction from raw FASTA should be enabled only after training and
releasing the annotation-light model. That model should use only features that
can be extracted from FASTA and PLM embeddings, avoiding missing GO/PPI values
being treated as true biological absence.

## Temporary files

Private uploads are written to:

```text
webapp_data/jobs/
```

They can be removed manually or with:

```bash
python scripts/webapp/cleanup_jobs.py --max-age-hours 24
```

Use `--dry-run` to inspect which paths would be removed.

## Public species-level cache

The app can save final predictions to:

```text
webapp_data/public_predictions/
```

This happens only when the user explicitly selects the public-cache consent box.
Only final prediction tables and metadata are cached. Raw uploads and
intermediate embeddings should not be stored publicly.

The public-cache key uses:

- species name;
- assembly version;
- annotation version;
- model;
- input feature-file SHA-256 checksum.

If a matching cache already exists, the app provides the cached result instead
of requiring recomputation.

## Practical server size

For the current full-feature `.npz` prediction mode, a CPU server with enough
RAM to load the selected feature matrix is sufficient. For future raw FASTA
prediction with PLM extraction, batch processing should be used. A practical
local server target is 32-64 GB RAM, a 12-16 GB NVIDIA GPU if PLM extraction is
performed online, and 50-200 GB free disk depending on job volume.
