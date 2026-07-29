# Locked independent-validation protocol

## Purpose

This protocol evaluates frozen plant essential-gene predictions against an
external phenotype cohort without reusing any study-label, pseudo-label, or
feature-derived evidence. It applies to the Arabidopsis and rice candidate
resource released with PlantEssentialGenePredictor.

## Freeze point

The frozen inputs and their SHA256 digests are recorded in
`results/tpc_candidate_resource/frozen_submission_inputs.json`. The candidate
registry is `results/tpc_candidate_resource/study_label_and_phenotype_registry.tsv`.
No model, threshold, feature set, split, or candidate-ranking rule may be
altered after an external cohort is opened.

## Eligible external phenotype evidence

One row represents one gene-level experimental observation. A row is eligible
only when all of the following are true:

1. The gene does not appear in the study label registry as
   `known_label_used_in_study` or `pseudo_label_used_in_study`.
2. The observation was not used to make any phenotype source archive, label,
   pseudo-label, train/validation/test split, model weight, or candidate rank.
3. The evidence is a published or publicly released loss-of-function,
   knockout, insertional-mutant, CRISPR, RNAi, or equivalent genotype-to-
   phenotype observation. Functional annotation alone is not phenotype
   evidence.
4. The source, stable accession/URL, release or publication date, assay type,
   phenotype stage, and binary adjudication rule are recorded.
5. The source is not one of the phenotype tables used to create the study
   registry. A source may only be considered if it provides a demonstrably
   independent record that was not ingested during label construction.

## Essentiality adjudication

`essential_label=1` requires developmental arrest at the gametophytic,
embryonic, seedling, or pre-reproductive vegetative stage, or complete
sterility/severe defect expected to preclude normal laboratory growth.
`essential_label=0` requires a viable mutant record without the preceding
essentiality criteria. Ambiguous, environment-conditional, heterozygous-only,
or conflicting records are excluded from the locked quantitative cohort and
may be retained only as qualitative evidence cards.

## Separation of analyses

* **Quantitative external cohort:** locked before scoring; used once for AUC,
  AUPRC, threshold metrics, and 10,000-replicate stratified bootstrap 95% CIs.
* **Candidate evidence cards:** curated after predictions, used only as
  discovery-oriented biological context. They must never be summarized as an
  unbiased success rate.
* **Model explanation:** GO, PPI, expression, paralogy and related training
  features are not independent validation. Any external evidence card records
  whether a source overlaps those feature families.

## Required cohort schema

Use `data/external_validation/independent_phenotype_cohort_template.tsv`.
Set `include_in_locked_cohort=yes` only after all eligibility checks are
complete. The evaluator refuses rows with missing provenance, non-independent
flags, or a gene that overlaps the study-label registry.

## Reporting

Report cohort size, class counts, source distribution, date range, all
exclusions, AUC, AUPRC, sensitivity, specificity, precision and F1. AUC and
AUPRC use probabilities; the remaining metrics use the pre-locked single
species threshold. Do not tune a threshold on this cohort.
