"""Evaluate a locked, independently curated phenotype cohort.

The tool is intentionally strict: it refuses genes used as study labels,
pseudo-labels, or any archived phenotype source, and requires
provenance/independence fields before reporting metrics. It never retrains or
selects a threshold.  Quantitative metrics are deliberately withheld unless a
pre-registered minimum cohort size and class balance are met.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results" / "tpc_candidate_resource" / "study_label_and_phenotype_registry.tsv"
PUBLISHED = ROOT / "predictions" / "publication_release"
OUT = ROOT / "results" / "tpc_candidate_resource" / "external_validation"
RELEASE = ROOT / "data" / "external_validation" / "release"
REQUIRED = {
    "species", "gene_id", "essential_label", "evidence_source",
    "source_url_or_accession", "publication_or_release_date",
    "experimental_system", "mutant_or_assay", "phenotype_stage",
    "adjudication_rule", "independent_of_training_labels",
    "independent_of_pseudo_labels", "independent_of_model_features",
    "include_in_locked_cohort", "evidence_curator", "exclusion_reason",
}
PROHIBITED_REGISTRY_STATUS = {
    "known_label_used_in_study",
    "pseudo_label_used_in_study",
    "phenotype_recorded_but_excluded",
}


def canonical(value: object) -> str:
    return str(value).strip().upper()


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def bootstrap(y: np.ndarray, p: np.ndarray, threshold: float, n: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    values: dict[str, list[float]] = {}
    for _ in range(n):
        indices = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        for name, value in metrics(y[indices], p[indices], threshold).items():
            values.setdefault(name, []).append(value)
    return {
        key: {"lower_95": float(np.nanpercentile(item, 2.5)), "upper_95": float(np.nanpercentile(item, 97.5))}
        for key, item in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path, help="TSV following the locked cohort schema")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--minimum-total", type=int, default=30)
    parser.add_argument("--minimum-per-class", type=int, default=10)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, sep="\t", dtype=str, keep_default_na=False)
    missing = REQUIRED - set(cohort.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    cohort = cohort.loc[cohort["include_in_locked_cohort"].str.lower().eq("yes")].copy()
    if cohort.empty:
        raise ValueError("No rows are marked include_in_locked_cohort=yes; no metrics were calculated.")
    for field in ["independent_of_training_labels", "independent_of_pseudo_labels", "independent_of_model_features"]:
        invalid = ~cohort[field].str.lower().eq("yes")
        if invalid.any():
            raise ValueError(f"{field} must be yes for every included row: {cohort.loc[invalid, 'gene_id'].tolist()}")
    provenance = ["evidence_source", "source_url_or_accession", "publication_or_release_date", "experimental_system", "mutant_or_assay", "phenotype_stage", "adjudication_rule", "evidence_curator"]
    missing_provenance = cohort[provenance].eq("").any(axis=1)
    if missing_provenance.any():
        raise ValueError("Included rows with incomplete provenance: " + ", ".join(cohort.loc[missing_provenance, "gene_id"]))

    cohort["species"] = cohort["species"].str.lower()
    cohort["gene_id_key"] = cohort["gene_id"].map(canonical)
    cohort["essential_label"] = pd.to_numeric(cohort["essential_label"], errors="raise").astype(int)
    if not cohort["essential_label"].isin([0, 1]).all():
        raise ValueError("essential_label must be 0 or 1")
    if cohort.duplicated(["species", "gene_id_key"]).any():
        raise ValueError("Duplicate gene rows are not allowed in a locked cohort")

    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    prohibited = registry.loc[
        registry["candidate_status"].isin(PROHIBITED_REGISTRY_STATUS),
        ["species", "gene_id_key", "candidate_status"],
    ]
    overlap = cohort.merge(prohibited.assign(_study_overlap=True), on=["species", "gene_id_key"], how="left")
    overlap = overlap.loc[overlap["_study_overlap"].eq(True)]
    if not overlap.empty:
        details = ", ".join(
            f"{row.gene_id} ({row.candidate_status})" for row in overlap.itertuples(index=False)
        )
        raise ValueError("External cohort overlaps a prohibited study phenotype record: " + details)

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "cohort": str(args.cohort),
        "bootstrap_replicates": args.bootstrap,
        "minimum_total": args.minimum_total,
        "minimum_per_class": args.minimum_per_class,
        "threshold_or_model_selection_performed_on_external_cohort": False,
        "quantitative_metrics_withheld_when_minimum_not_met": True,
        "species": {},
    }
    scored = []
    for species, subset in cohort.groupby("species", sort=True):
        pred_path = PUBLISHED / f"{species}_all_feature_covered_genes_reclassified.tsv"
        predictions = pd.read_csv(pred_path, sep="\t", dtype=str, keep_default_na=False)
        predictions["gene_id_key"] = predictions["gene_id_key"].map(canonical)
        numeric = ["single_species_probability", "single_species_threshold"]
        predictions[numeric] = predictions[numeric].astype(float)
        joined = subset.merge(predictions[["gene_id_key", *numeric]], on="gene_id_key", how="left")
        if joined["single_species_probability"].isna().any():
            raise ValueError(f"No released prediction for: {joined.loc[joined['single_species_probability'].isna(), 'gene_id'].tolist()}")
        y = joined["essential_label"].to_numpy(int)
        p = joined["single_species_probability"].to_numpy(float)
        n_essential = int(y.sum())
        n_nonessential = int((y == 0).sum())
        eligible = (
            len(np.unique(y)) == 2
            and len(joined) >= args.minimum_total
            and n_essential >= args.minimum_per_class
            and n_nonessential >= args.minimum_per_class
        )
        if not eligible:
            summary["species"][species] = {
                "n": int(len(joined)),
                "essential": n_essential,
                "nonessential": n_nonessential,
                "eligible_for_quantitative_evaluation": False,
                "reason": (
                    "Pre-registered external-cohort minimum not met: requires "
                    f"n >= {args.minimum_total} and >= {args.minimum_per_class} per class."
                ),
            }
            scored.append(joined)
            continue
        thresholds = joined["single_species_threshold"].unique()
        if len(thresholds) != 1:
            raise ValueError(f"{species} has inconsistent frozen thresholds: {thresholds.tolist()}")
        threshold = float(thresholds[0])
        point = metrics(y, p, threshold)
        intervals = bootstrap(y, p, threshold, args.bootstrap, args.seed)
        summary["species"][species] = {
            "n": int(len(joined)), "essential": n_essential, "nonessential": n_nonessential,
            "eligible_for_quantitative_evaluation": True,
            "frozen_single_species_threshold": threshold, "point_estimates": point, "bootstrap_95_ci": intervals,
        }
        scored.append(joined)
    pd.concat(scored, ignore_index=True).to_csv(OUT / "locked_external_cohort_scored.tsv", sep="\t", index=False)
    (OUT / "locked_external_cohort_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RELEASE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "locked_external_cohort_scored.tsv", RELEASE / "locked_external_cohort_scored.tsv")
    shutil.copy2(OUT / "locked_external_cohort_metrics.json", RELEASE / "locked_external_cohort_metrics.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
