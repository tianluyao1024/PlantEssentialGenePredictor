# Independent Evidence and External-Validation Release Notes

**Release:** `v1.2.1-independent-validation`. The package manifest records the
Git commit, tag description and SHA256 checksum of every included file.

## Scope

This release separates three activities that must not be conflated:

1. **Literature screening:** Europe PMC search results are source leads, not
   phenotype labels.
2. **Curated direct LoF records:** a curator records a stable source, allele or
   assay, stage, binary adjudication and source-provenance check.
3. **External performance evaluation:** the frozen evaluator runs only when a
   species has at least 30 records, including at least 10 essential and 10
   viable/non-essential records.

## Current result

The first screening pass queried 312 articles and produced 919 gene-level
screening records. Automated exclusion left 108 true-unknown records for
manual review. Nineteen Arabidopsis direct LoF records passed the zero-overlap
audit; sixteen met the phenotype-adjudication criteria for a prelocked cohort
(nine essential and seven viable/non-essential). Three viable but severely
growth-retarded records are retained as qualitative secondary evidence, not as
non-essential labels. No rice row has yet completed curator locking.

The minimum is therefore not met for either species. The release writes an
underpowered status report and intentionally does not calculate external AUC,
AUPRC, sensitivity, specificity, precision or F1.

## Reproduction

```powershell
python scripts/publication/screen_external_phenotype_sources.py --max-results 80
powershell -ExecutionPolicy Bypass -File scripts/publication/run_external_validation_release_pipeline.ps1
```

The optional Europe PMC refresh may change search results over time; the frozen
curated records, audit outputs and SHA256 manifest are the reproducibility
record for this release.

## Main artifacts

* `europe_pmc_source_screening_ledger.tsv`: automated search and exclusion
  ledger.
* `curated_external_source_screening_ledger.tsv`: target DOI/PMCID curator
  ledger with gene-ID normalization and inclusion/exclusion reason.
* `prelocked_external_phenotype_cohort.tsv`: direct LoF rows that passed
  provenance and phenotype adjudication checks.
* `locked_external_cohort_metrics.json`: current gate result. A `false`
  eligibility field means no quantitative external performance was calculated.
* `*_evidence_cards_with_independent_evidence.tsv`: long-form core-candidate
  evidence records.
