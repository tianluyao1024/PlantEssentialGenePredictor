# Data Sources and Processed Release Scope

This release contains processed labels, processed feature matrices, trained models and prediction outputs.

It does not redistribute raw database downloads such as GO annotation dumps, PPI source files, expression matrices, phenotype database exports or PLM checkpoint files. The feature-extraction scripts document how the processed feature tables were generated from public source databases.

## Label Sets

Arabidopsis primary labels use the strict2601 label set:

- 508 essential genes;
- 2,093 non-essential genes;
- conflicts removed;
- fixed validation and test sets derived from the high-confidence core split.

Rice primary labels use strict Oryzabase essential evidence plus Tos17 N4 non-essential evidence:

- strict Oryzabase essential genes;
- Tos17 non-essential genes with zero essential records, ES < 0.1 and at least four non-essential observations;
- conflicts removed from the non-essential set;
- 1,168 feature-matched genes used for fixed evaluation.

## Feature Matrices

The released common6751 feature space contains:

- 95 shared biological features;
- 2,560 ESM2 features;
- 2,048 ProtBERT features;
- 2,048 ProtT5 features.

Processed feature matrices are stored as `.npz` files under `data/processed_features/`.

