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

The release was publicly verified through DOI.org and the Zenodo API on
2026-07-30. Cite the immutable version DOI in the manuscript; use the concept
DOI only when a link to the latest version is desired.

```text
Concept DOI: 10.5281/zenodo.21387075
Version DOI: 10.5281/zenodo.21387076
```

## Required v1.2 independent-evidence update

The published v1.2.3 record contains the base artifact and independent-evidence
supplement. For future changes, create a new version of the existing record and
retain the current version as an immutable archive. Generate any updated
supplement with:

```text
python scripts/release/create_independent_validation_supplement.py
```

The generated `PlantEssentialGenePredictor_independent_validation_v1_2_3.zip`
contains the source-screening ledger, curator-checked direct loss-of-function
records, zero-overlap audit, prelocked external-cohort status, evidence-card
tables, Figure 7 source data and the scripts required to reproduce the audit.
It does not duplicate raw databases, PLM weights, large matrices or model
binaries. Its SHA256 checksum and per-file manifest are written into the zip.
After publishing the new version, update the README, manuscript and release
notes with both the immutable version DOI and the concept DOI.

The lightweight supplement is additionally mirrored in the GitHub
[v1.2.3 release](https://github.com/tianluyao1024/PlantEssentialGenePredictor/releases/tag/v1.2.3-independent-validation).

## Manuscript wording

Suggested Data availability wording:

> Source code, documentation and the Streamlit web application are available at
> GitHub: https://github.com/tianluyao1024/PlantEssentialGenePredictor.
> Processed feature matrices, trained model objects, fixed label tables and
> genome-scale prediction tables are archived on Zenodo, version v1.2.3,
> https://doi.org/10.5281/zenodo.21387076.


