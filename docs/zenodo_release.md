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

The published Zenodo DOI is:

```text
10.5281/zenodo.21387076
```

## Required v1.1 candidate-audit update

Before manuscript submission, create a new version of the existing Zenodo
record and retain the original base artifact. Add the prepared supplement:

```text
E:\PlantEssentialGenePredictor_Zenodo\PlantEssentialGenePredictor_audited_candidate_resource_v1_1.zip
```

The supplement contains the audited publication-release prediction tables,
candidate-exclusion audit, frozen-input manifest, ten-gene core panels,
independent-validation template and the scripts required to reproduce the
candidate audit. Its SHA256 checksum is recorded in the generated manifest.
After publishing the new version, update the README, manuscript and release
notes with both the immutable version DOI and the concept DOI.

This DOI is referenced in `README.md` and in the manuscript data availability
statement.

## Manuscript wording

Suggested Data availability wording:

> Source code, documentation and the Streamlit web application are available at
> GitHub: https://github.com/tianluyao1024/PlantEssentialGenePredictor.
> Processed feature matrices, trained model objects, fixed label tables and
> genome-scale prediction tables are archived on Zenodo under DOI:
> 10.5281/zenodo.21387076.


