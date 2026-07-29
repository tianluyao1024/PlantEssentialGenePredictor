"""Audit curated external phenotype records and generate release-ready assets.

This tool never retrains or scores a model. It verifies that curated evidence
is restricted to feature-covered `true_unknown_candidate` genes and has zero
intersection with every frozen label or archived phenotype status.  It also
appends independently curated candidate evidence to the frozen core-10 cards.
"""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "predictions" / "publication_release"
REGISTRY = ROOT / "results" / "tpc_candidate_resource" / "study_label_and_phenotype_registry.tsv"
CARDS = ROOT / "results" / "tpc_candidate_resource"
DATA = ROOT / "data" / "external_validation"
OUT = CARDS / "external_validation"
RELEASE = DATA / "release"
PROHIBITED = {"known_label_used_in_study", "pseudo_label_used_in_study", "phenotype_recorded_but_excluded"}


def canonical(value: object) -> str:
    value = str(value).strip().upper()
    return value[4:] if value.startswith("LOC_") else value


def load_prediction_status() -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    for species in ("arabidopsis", "rice"):
        frame = pd.read_csv(PUBLISHED / f"{species}_all_feature_covered_genes_reclassified.tsv", sep="\t", dtype=str, keep_default_na=False)
        for row in frame[["gene_id_key", "candidate_status"]].itertuples(index=False):
            values[(species, canonical(row.gene_id_key))] = row.candidate_status
    return values


def status_audit(frame: pd.DataFrame, prediction_status: dict[tuple[str, str], str], registry_status: dict[tuple[str, str], str]) -> pd.DataFrame:
    audited = frame.copy()
    audited["species"] = audited["species"].str.lower()
    audited["gene_id_key"] = audited["gene_id"].map(canonical)
    audited["released_candidate_status"] = [prediction_status.get((s, g), "not_feature_covered_or_identifier_unresolved") for s, g in zip(audited["species"], audited["gene_id_key"])]
    audited["registry_status"] = [registry_status.get((s, g), "") for s, g in zip(audited["species"], audited["gene_id_key"])]
    audited["zero_prohibited_overlap"] = (~audited["registry_status"].isin(PROHIBITED)).map({True: "yes", False: "no"})
    audited["is_true_unknown_candidate"] = audited["released_candidate_status"].eq("true_unknown_candidate").map({True: "yes", False: "no"})
    audited["automated_audit_decision"] = "eligible_for_curator_decision"
    audited.loc[audited["zero_prohibited_overlap"].eq("no"), "automated_audit_decision"] = "rejected_prohibited_registry_overlap"
    audited.loc[audited["is_true_unknown_candidate"].eq("no"), "automated_audit_decision"] = "rejected_not_feature_covered_true_unknown"
    return audited


def append_candidate_evidence() -> dict[str, int]:
    evidence = pd.read_csv(DATA / "candidate_independent_evidence.tsv", sep="\t", dtype=str, keep_default_na=False)
    counts: dict[str, int] = {}
    for species in ("arabidopsis", "rice"):
        cards_path = CARDS / f"{species}_final_core10_evidence_cards.tsv"
        cards = pd.read_csv(cards_path, sep="\t", dtype=str, keep_default_na=False)
        additions = evidence.loc[evidence["species"].eq(species)].copy()
        if additions.empty:
            cards.to_csv(CARDS / f"{species}_final_core10_evidence_cards_with_independent_evidence.tsv", sep="\t", index=False)
            counts[species] = 0
            continue
        base = cards.iloc[0:0].copy()
        rows: list[pd.Series] = []
        for item in additions.itertuples(index=False):
            matched = cards.loc[cards["gene_id_key"].map(canonical).eq(canonical(item.gene_id))]
            if matched.empty:
                raise ValueError(f"Evidence gene is not a frozen {species} core candidate: {item.gene_id}")
            row = matched.iloc[0].copy()
            for column in [
                "evidence_id", "evidence_category", "evidence_source", "source_url_or_accession",
                "publication_or_release_date", "evidence_summary", "supports_essentiality",
                "independent_of_training_labels", "independent_of_pseudo_labels",
                "independent_of_model_features", "evidence_strength", "material_accession_or_availability",
                "curator", "evidence_status", "notes",
            ]:
                row[column] = getattr(item, column)
            rows.append(row)
        append = pd.DataFrame(rows, columns=cards.columns)
        # Retain one blank/prediction-only row for candidates without evidence and
        # replace populated candidates by their long-form evidence records.
        evidenced = set(additions["gene_id"].map(canonical))
        remaining = cards.loc[~cards["gene_id_key"].map(canonical).isin(evidenced)].copy()
        remaining["evidence_status"] = "prediction_only_no_independent_evidence_curated"
        combined = pd.concat([append, remaining], ignore_index=True)
        combined.to_csv(CARDS / f"{species}_final_core10_evidence_cards_with_independent_evidence.tsv", sep="\t", index=False)
        counts[species] = int(len(additions))
    return counts


def build_candidate_card_summaries() -> dict[str, int]:
    evidence = pd.read_csv(DATA / "candidate_independent_evidence.tsv", sep="\t", dtype=str, keep_default_na=False)
    counts: dict[str, int] = {}
    for species in ("arabidopsis", "rice"):
        candidates = pd.read_csv(CARDS / f"{species}_final_core10_candidates.tsv", sep="\t", dtype=str, keep_default_na=False)
        candidates["gene_id_key"] = candidates["gene_id_key"].map(canonical)
        subset = evidence.loc[evidence["species"].eq(species)].copy()
        subset["gene_id_key"] = subset["gene_id"].map(canonical) if not subset.empty else pd.Series(dtype=str)
        rows: list[dict[str, object]] = []
        for candidate in candidates.itertuples(index=False):
            records = subset.loc[subset["gene_id_key"].eq(candidate.gene_id_key)]
            categories = sorted(set(records["evidence_category"])) if not records.empty else []
            direct = records.loc[records["evidence_category"].eq("A_direct")] if not records.empty else records
            if direct.empty:
                direction = "prediction_only"
                direct_status = "no_independent_direct_LoF_record_curated"
            elif direct["supports_essentiality"].eq("yes").any():
                direction = "direct_essential_support"
                direct_status = "independent_direct_LoF_supports_essentiality"
            else:
                direction = "direct_viable_or_counterexample"
                direct_status = "independent_direct_LoF_viable_or_counterexample"
            material = records.loc[records["evidence_category"].eq("C_material")] if not records.empty else records
            reported_material = records.loc[records["material_accession_or_availability"].ne("")] if not records.empty else records
            if not material.empty:
                material_status = "verified_public_material_reported"
                material_detail = ";".join(sorted(set(material["material_accession_or_availability"]) - {""}))
            elif not reported_material.empty:
                material_status = "source_reviewed_material_not_stated_or_not_accessioned"
                material_detail = ";".join(sorted(set(reported_material["material_accession_or_availability"]) - {""}))
            else:
                material_status = "not_curated_not_evidence_of_unavailability"
                material_detail = ""
            rows.append({
                "species": species,
                "gene_id": candidate.gene_id,
                "candidate_rank": candidate.candidate_rank,
                "single_species_probability": candidate.single_species_probability,
                "joint_probability": candidate.joint_probability,
                "annotation_light_probability": candidate.annotation_light_probability,
                "homology_category": candidate.homology_category,
                "independent_evidence_categories": ";".join(categories),
                "n_independent_evidence_categories": len(categories),
                "direct_evidence_direction": direction,
                "direct_LoF_evidence_status": direct_status,
                "material_availability_status": material_status,
                "material_accession_or_availability": material_detail,
                "main_text_evidence_card_eligible": "yes" if len(categories) >= 2 else "no",
                "evidence_ids": ";".join(records["evidence_id"].tolist()),
                "candidate_resource_status": "prediction_only" if not categories else "independent_evidence_curated",
            })
        result = pd.DataFrame(rows)
        result.to_csv(CARDS / f"{species}_final_core10_evidence_card_summary.tsv", sep="\t", index=False)
        counts[species] = int(result["main_text_evidence_card_eligible"].eq("yes").sum())
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    prediction_status = load_prediction_status()
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    registry_status = {
        (row.species.lower(), canonical(row.gene_id_key)): row.candidate_status
        for row in registry[["species", "gene_id_key", "candidate_status"]].itertuples(index=False)
    }
    curated = pd.read_csv(DATA / "curated_external_phenotype_records.tsv", sep="\t", dtype=str, keep_default_na=False)
    audited = status_audit(curated, prediction_status, registry_status)
    audited.to_csv(OUT / "curated_external_phenotype_records_audited.tsv", sep="\t", index=False)
    rejected = audited.loc[~((audited["zero_prohibited_overlap"].eq("yes")) & (audited["is_true_unknown_candidate"].eq("yes")))]
    if not rejected.empty:
        raise ValueError("Curated external evidence failed frozen status audit: " + ", ".join(rejected["gene_id"].tolist()))
    locked = audited.loc[audited["include_in_locked_cohort"].str.lower().eq("yes")].copy()
    locked.to_csv(OUT / "prelocked_external_phenotype_cohort.tsv", sep="\t", index=False)
    source_ledger = audited.copy()
    source_ledger["searched_source"] = "targeted DOI/PMCID manual full-text verification"
    source_ledger["query_date"] = "2026-07-29"
    source_ledger["gene_id_normalization"] = source_ledger["gene_id_key"]
    source_ledger["evidence_category"] = "A_direct"
    source_ledger["screening_inclusion_or_exclusion_reason"] = source_ledger["exclusion_reason"]
    source_ledger.loc[source_ledger["screening_inclusion_or_exclusion_reason"].eq(""), "screening_inclusion_or_exclusion_reason"] = "included_as_prelocked_direct_LoF_record"
    source_ledger.to_csv(OUT / "curated_external_source_screening_ledger.tsv", sep="\t", index=False)
    card_counts = append_candidate_evidence()
    card_summary_counts = build_candidate_card_summaries()
    input_files = {
        "curated_external_phenotype_records": DATA / "curated_external_phenotype_records.tsv",
        "candidate_independent_evidence": DATA / "candidate_independent_evidence.tsv",
        "study_label_and_phenotype_registry": REGISTRY,
        "released_arabidopsis_predictions": PUBLISHED / "arabidopsis_all_feature_covered_genes_reclassified.tsv",
        "released_rice_predictions": PUBLISHED / "rice_all_feature_covered_genes_reclassified.tsv",
    }
    summary = {
        "curated_records": int(len(audited)),
        "prelocked_records": int(len(locked)),
        "prelocked_by_species_and_label": (
            locked.groupby(["species", "essential_label"], dropna=False).size().rename("n").reset_index().to_dict(orient="records")
        ),
        "zero_prohibited_overlap": bool(audited["zero_prohibited_overlap"].eq("yes").all()),
        "all_true_unknown_candidates": bool(audited["is_true_unknown_candidate"].eq("yes").all()),
        "candidate_evidence_rows_appended": card_counts,
        "core_candidate_cards_eligible_for_main_text": card_summary_counts,
        "quantitative_evaluation_policy": "The frozen evaluator withholds AUC/AUPRC and threshold metrics below n=30 per species with >=10 per class.",
        "frozen_input_sha256": {name: sha256(path) for name, path in input_files.items()},
    }
    (OUT / "external_validation_release_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    release_files = [
        OUT / "curated_external_phenotype_records_audited.tsv",
        OUT / "prelocked_external_phenotype_cohort.tsv",
        OUT / "curated_external_source_screening_ledger.tsv",
        OUT / "external_validation_release_summary.json",
        CARDS / "arabidopsis_final_core10_evidence_cards_with_independent_evidence.tsv",
        CARDS / "rice_final_core10_evidence_cards_with_independent_evidence.tsv",
        CARDS / "arabidopsis_final_core10_evidence_card_summary.tsv",
        CARDS / "rice_final_core10_evidence_card_summary.tsv",
    ]
    source_ledger = ROOT / "results" / "tpc_candidate_resource" / "external_source_screening" / "europe_pmc_source_screening_ledger.tsv"
    if source_ledger.exists():
        release_files.append(source_ledger)
    for source in release_files:
        shutil.copy2(source, RELEASE / source.name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
