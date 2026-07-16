# Zenodo release plan

The GitHub repository contains code, documentation, input templates and small
label files. Large artifacts are deposited on Zenodo.

## Zenodo account

Zenodo creator/account name: `tianluyao1024`.

## Recommended Zenodo deposition title

`PlantEssentialGenePredictor: models, processed features and genome-scale predictions`

## Files to deposit

Create a Zenodo upload from the contents of:

```text
E:\PlantEssentialGenePredictor_Zenodo\PlantEssentialGenePredictor_Zenodo_v1
```

The deposition should include:

- `models/`
- `data/processed_features/`
- `data/labels/`
- `predictions/`
- `prediction_manifest.json`
- `release_metadata.json`
- `RELEASE_MANIFEST.md`
- `zenodo_file_manifest.tsv`
- `zenodo_sha256sums.txt`

Do not upload:

- `.venv/`
- `webapp_data/jobs/`
- `webapp_data/logs/`
- raw database dumps;
- bundled ESM2, ProtBERT or ProtT5 pretrained weights, unless a separate
  redistribution policy check is completed.

## DOI handling

Zenodo will mint a DOI after the deposition is published. Replace the placeholder
in `README.md`:

```text
10.5281/zenodo.XXXXXXX
```

with the final DOI, then commit and push the README update to GitHub.

## Manuscript wording

Suggested Data availability wording:

> Source code, documentation and the Streamlit web application are available at
> GitHub: https://github.com/tianluyao1024/PlantEssentialGenePredictor.
> Processed feature matrices, trained model objects, fixed label tables and
> genome-scale prediction tables are archived on Zenodo under DOI:
> 10.5281/zenodo.XXXXXXX.

