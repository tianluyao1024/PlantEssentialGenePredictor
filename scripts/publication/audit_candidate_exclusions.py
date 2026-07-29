"""Assert that every released candidate is disjoint from all study-label records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tpc_candidate_resource"
REGISTRY = OUT / "study_label_and_phenotype_registry.tsv"
PUBLISHED = ROOT / "predictions" / "publication_release"


def main() -> None:
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    prohibited = registry[["species", "gene_id_key", "candidate_status"]].rename(
        columns={"candidate_status": "registry_status"}
    )
    candidate_sets = []
    for species in ("arabidopsis", "rice"):
        release = pd.read_csv(PUBLISHED / f"{species}_all_feature_covered_genes_reclassified.tsv", sep="\t", dtype=str, keep_default_na=False)
        candidate_sets.append((species, "publication_true_unknown_candidate", release.loc[release["candidate_status"].eq("true_unknown_candidate"), ["gene_id_key"]]))
        for name in ("provisional_true_unknown_candidates", "final_core10_candidates"):
            frame = pd.read_csv(OUT / f"{species}_{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
            candidate_sets.append((species, name, frame[["gene_id_key"]]))

    rows = []
    failures = []
    for species, name, candidates in candidate_sets:
        checked = candidates.assign(species=species).merge(prohibited, on=["species", "gene_id_key"], how="left")
        overlap = checked.loc[checked["registry_status"].fillna("").ne("")]
        status_counts = overlap["registry_status"].value_counts().to_dict()
        rows.append({
            "species": species,
            "candidate_set": name,
            "candidate_rows": len(candidates),
            "overlap_with_any_study_label_or_phenotype_record": len(overlap),
            "overlap_status_counts": ";".join(f"{key}:{value}" for key, value in sorted(status_counts.items())),
            "audit_result": "PASS" if overlap.empty else "FAIL",
        })
        if not overlap.empty:
            failures.append(f"{species}/{name}: {overlap['gene_id_key'].head(10).tolist()}")
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "candidate_exclusion_audit.tsv", sep="\t", index=False)
    print(audit.to_string(index=False))
    if failures:
        raise RuntimeError("Candidate exclusion audit failed: " + " | ".join(failures))


if __name__ == "__main__":
    main()
