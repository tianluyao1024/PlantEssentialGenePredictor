# The Plant Cell Submission Checklist

This checklist tracks the evidence and administrative items needed to convert
the current computational resource into a submission-ready manuscript. It
separates completed audit work from requirements that cannot be inferred or
completed without author confirmation.

## Completed computational record

- [x] Lock strict phenotype-label inputs, fixed train/validation/test splits,
  feature list and model artifacts in `frozen_submission_inputs.json`.
- [x] Reclassify every feature-covered gene as a study label, pseudo-label,
  phenotype-recorded-but-excluded gene or true unknown candidate.
- [x] Automatically verify zero overlap between each core candidate and all
  prohibited model-label and phenotype-source sets.
- [x] Release ten core candidates for each species with component-model
  robustness and closest-labelled-homolog strata.
- [x] Provide an external phenotype cohort schema and evaluator that refuses
  invalid or non-independent records.
- [x] Screen 312 Europe PMC articles (919 gene-level records) with a
  machine-readable source ledger, then perform targeted DOI/PMCID curation.
- [x] Verify that all 19 curator-checked direct LoF records have zero overlap
  with frozen study labels, pseudo-labels and raw phenotype archives.
- [x] Apply the pre-registered external-cohort gate. The current Arabidopsis
  cohort contains 16 records (9 essential, 7 viable/non-essential), and rice
  contains no curator-locked record. Both species are therefore qualitative
  only; no external AUC/AUPRC or threshold metric is reported.
- [x] Create long-form independent-evidence cards and a summary that labels
  unsupported entries as prediction-only rather than validated candidates.
- [x] Archive the current computational release at Zenodo DOI
  `10.5281/zenodo.21387076` and host code at GitHub.

## Must be completed before submission

- [ ] Create a versioned Zenodo update containing the audited candidate tables,
  external-validation template/evaluator, source data for Figure 6 and the
  final release manifest. Record the version DOI and archive DOI in the
  manuscript.
- [ ] Tag the matching GitHub release and record the immutable commit hash.
- [ ] Expand the independently curated cohort to the pre-registered minimum
  for either species (n >= 30 and at least 10 records per class) before adding
  any external AUC/AUPRC or threshold result. Do not use records that
  contributed to labels, pseudo-labels or phenotype-derived feature encoding.
- [x] Run `scripts/publication/evaluate_external_phenotype_cohort.py` on the
  prelocked cohort. The actual output is an underpowered status report, which
  is now reflected in the manuscript without a performance claim.
- [ ] Confirm the corresponding author, contact email, author order, ORCID
  identifiers, funding, competing interests and material-distribution contact.
- [ ] Add the journal-required AI-use disclosure, if any generative AI was used
  for writing, coding, figure drafting or language editing. Human authors must
  verify all scientific claims and references.
- [ ] Confirm that every reference, software version, database release and web
  URL matches the final manuscript.

## Evidence boundary for the current draft

The current manuscript may claim locked internal performance, feature ablation,
homology-aware evaluation, audited genome-scale candidate nomination,
qualitative direct evidence for the explicitly documented Arabidopsis cards and
public resource availability. It must not claim independent external
discrimination performance or biological confirmation of all twenty core
candidates until the pre-registered cohort threshold or wet-lab evidence is
met.
