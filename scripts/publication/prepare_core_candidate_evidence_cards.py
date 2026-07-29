"""Create blank, provenance-aware evidence cards for final frozen candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tpc_candidate_resource"


def main() -> None:
    for species in ("arabidopsis", "rice"):
        path = OUT / f"{species}_final_core10_candidates.tsv"
        candidates = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        candidates["candidate_rank"] = range(1, len(candidates) + 1)
        columns = [
            "species", "candidate_rank", "gene_id", "gene_id_key",
            "single_species_probability", "joint_probability", "annotation_light_probability",
            "single_component_count", "single_component_mean", "single_component_sd",
            "joint_component_count", "joint_component_mean", "joint_component_sd",
            "homology_category", "closest_labelled_gene", "closest_labelled_label",
            "identity_percent", "query_coverage_percent", "subject_coverage_percent", "selection_group",
        ]
        card = candidates.reindex(columns=columns).copy()
        for column in [
            "evidence_id", "evidence_category", "evidence_source", "source_url_or_accession",
            "publication_or_release_date", "evidence_summary", "supports_essentiality",
            "independent_of_training_labels", "independent_of_pseudo_labels",
            "independent_of_model_features", "evidence_strength", "material_accession_or_availability",
            "curator", "evidence_status", "notes",
        ]:
            card[column] = ""
        card["evidence_status"] = "pending_manual_independent_curation"
        card.to_csv(OUT / f"{species}_final_core10_evidence_cards.tsv", sep="\t", index=False)
    print(f"Wrote final-core evidence-card templates to {OUT}")


if __name__ == "__main__":
    main()
