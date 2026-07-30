# Candidate Audit Release v1.1 Notes

## Purpose

This release corrects the interpretation of historical genome-scale prediction
tables. Earlier files named `unknown` were generated from feature-covered sets
and were not guaranteed to exclude every later study label or raw phenotype
record. They remain reproducibility artifacts, not novel-candidate resources.

## Audited status classes

Every gene in the new publication-release tables receives one mutually
exclusive status:

| Status | Meaning | Eligible for novel-candidate analyses |
|---|---|---:|
| `known_label_used_in_study` | Used as a strict model label | No |
| `pseudo_label_used_in_study` | Used as a study pseudo-label | No |
| `phenotype_recorded_but_excluded` | Present in a reconciled phenotype source but excluded from the primary model | No |
| `true_unknown_candidate` | No overlap with the above sources | Yes |

## Core candidate panels

The release includes ten candidates per species. These are not confirmed
essential genes. They were selected only after a fully automated exclusion
audit, agreement across frozen model families, component-model robustness and a
closest-labelled-homolog screen. The panels include both candidates with a
nearby labelled essential gene and candidates without one to avoid a purely
nearest-neighbour presentation.

## Required citation and versioning

The current immutable archive DOI is `10.5281/zenodo.21387076` (v1.2.3); the
concept DOI is `10.5281/zenodo.21387075`. The matching independent-validation
GitHub tag is `v1.2.3-independent-validation` at commit
`e1a2442c1140eed1f46674c7b05eb7faadc65e5f`.
