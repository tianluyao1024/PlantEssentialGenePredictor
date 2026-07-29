# Candidate evidence cards: curation guidance

Each core candidate receives two or more **non-training-source** evidence rows
before being presented as a representative biological case in the manuscript.
Use the templates in `results/tpc_candidate_resource/*_final_core10_evidence_cards.tsv`.

The release summary reports a `material_availability_status` for every frozen
candidate. `not_curated_not_evidence_of_unavailability` is deliberately
different from a negative material search: it means that a curator has not yet
verified a stock or germplasm catalogue. It must never be interpreted as
evidence that no material exists.

The accompanying `*_final_core10_uniprot_context.tsv` files provide protein
names, functional descriptions and localization context to guide literature
curation. They are explicitly marked `requires_independence_review`; they are
not evidence-card rows and cannot count toward the two-category requirement
without a curator locating the underlying independent experimental source.

## Accepted evidence categories

* Independent loss-of-function phenotype or new public mutant record.
* Developmental or reproductive expression atlas not used as a model feature.
* Experimentally supported subcellular localization or protein-complex evidence
  from an independent study.
* Ortholog knockout phenotype in another plant species, with the orthology
  source explicitly recorded.
* Public seed, insertion, mutant, CRISPR line, or germplasm accession showing
  that a direct follow-up experiment is feasible.

## Not accepted as independent validation

* Any phenotype source used to derive the study labels or pseudo-labels.
* GO terms, PPI degrees, expression summaries, paralog counts, or homology
  variables that were supplied to the model.
* Computational functional predictions without an external experimental source.
* A post hoc internet search result without a stable accession, paper, or date.

## Evidence strength

* `A_direct`: independent loss-of-function phenotype with a clear essential or
  viable conclusion.
* `B_functional`: independent experimental localization, complex membership,
  or ortholog knockout phenotype that supports the proposed mechanism.
* `C_material`: verified mutant/germplasm availability for a planned direct
  test; useful for follow-up but not confirmation.

At least two evidence categories are needed for a manuscript evidence card.
Only a category-A row may be used in the locked quantitative external phenotype
cohort.
