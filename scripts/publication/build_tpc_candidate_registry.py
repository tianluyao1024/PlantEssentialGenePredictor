"""Build a frozen, auditable candidate registry for the Plant Cell submission.

This script deliberately does not retrain a model.  It reconciles the released
genome-scale prediction tables with every locally archived phenotype source,
then labels each gene as a study label, a historical pseudo-label, a recorded
but excluded phenotype gene, or a true candidate.  It also derives a
sequence/PLM-only robustness score from the released deployable profile model.

Run from the repository root:
    python scripts/publication/build_tpc_candidate_registry.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "predictions"
LABELS = ROOT / "data" / "labels"
FEATURES = ROOT / "data" / "processed_features"
MODELS = ROOT / "models"
OUT = ROOT / "results" / "tpc_candidate_resource"
PUBLISHED = PREDICTIONS / "publication_release"

# These are source archives used during this project.  Only gene IDs and source
# provenance are emitted by this script; raw phenotype descriptions are not
# copied into the public release.
ATH_SOURCE_FILES = {
    "TAIR_Locus_Germplasm_Phenotype_20130122": Path(
        r"C:\Users\tly\Desktop\植物\拟南芥\Locus_Germplasm_Phenotype.txt"
    ),
    "phenotype_screen_csv": Path(r"C:\Users\tly\Desktop\植物\拟南芥\拟南芥表型筛选.csv"),
    "literature_classification": Path(r"C:\Users\tly\Desktop\植物\拟南芥\文献分类.tsv"),
    "ogee_essentiality": Path(r"C:\Users\tly\Desktop\植物\拟南芥\ogee必需ID.csv"),
    "embryo_defect_xlsx": Path(r"C:\Users\tly\Desktop\植物\拟南芥\拟南芥胚胎缺陷基因列表.xlsx"),
}
RICE_SOURCE_FILES = {
    "Tos17_processed_evidence": Path(
        r"E:\CodexMoved\Desktop\水稻\rice_mutant_sources\processed\tos17_source_evidence.csv"
    ),
    "Oryzabase_trait_gene": Path(
        r"E:\CodexMoved\Desktop\水稻\rice_mutant_sources\oryzabase_all_essentiality_20260620\oryzabase_trait_gene_all_normalized.tsv"
    ),
    "TRIM_TDNA_processed_evidence": Path(
        r"E:\CodexMoved\Desktop\水稻\rice_mutant_sources\processed_trim_tdna_es\trim_tdna_rap_source_evidence_for_combining.tsv"
    ),
    "RiceData_strict_LOF": Path(
        r"E:\CodexMoved\Desktop\水稻\rice_list_output0\essentiality_processed_strict_lof\rice_documented_gene_essentiality_classification_strict_lof.csv"
    ),
}
ATH_3359_SPLIT = Path(
    r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\096style_aug3359_pseudo14731_ricedata_only_cross\arabidopsis_3359_train_validation_test_split.tsv"
)

ATH_RE = re.compile(r"\bAT[1-5MC]G\d{5}\b", re.I)
# Tos17 commonly reports MSU locus identifiers as ``LOC_Os...`` while the
# RAP-native feature matrix and released prediction tables use ``Os...``.
# Treat both as the same stable gene key during the exclusion audit.
RICE_RE = re.compile(r"\b(?:LOC_)?Os\d{2}g\d+\b", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifiers_in_file(path: Path, pattern: re.Pattern[str]) -> set[str]:
    """Extract stable gene IDs without retaining raw descriptions."""
    if not path.exists():
        return set()
    if path.suffix.lower() == ".xlsx":
        with zipfile.ZipFile(path) as archive:
            text = "\n".join(
                archive.read(name).decode("utf-8", errors="ignore")
                for name in archive.namelist()
                if name.endswith(".xml")
            )
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    identifiers = {match.upper() for match in pattern.findall(text)}
    if pattern is RICE_RE:
        identifiers = {re.sub(r"^LOC_", "", gene, flags=re.I) for gene in identifiers}
    return identifiers


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def status_registry() -> tuple[pd.DataFrame, dict[str, object]]:
    if not ATH_3359_SPLIT.exists():
        raise FileNotFoundError(f"Missing frozen 3359 label split: {ATH_3359_SPLIT}")
    ath_3359 = read_tsv(ATH_3359_SPLIT)
    required = {"gene_id", "label_source"}
    if not required.issubset(ath_3359.columns):
        raise ValueError(f"Unexpected 3359 columns: {ath_3359.columns.tolist()}")
    ath_3359["gene_id"] = ath_3359["gene_id"].str.upper()
    ath_known = set(
        ath_3359.loc[ath_3359["label_source"].eq("true_consensus_2plus"), "gene_id"]
    )
    ath_pseudo = set(
        ath_3359.loc[ath_3359["label_source"].ne("true_consensus_2plus"), "gene_id"]
    )
    ath_sources = {
        name: identifiers_in_file(path, ATH_RE) for name, path in ATH_SOURCE_FILES.items()
    }
    ath_recorded = set().union(*ath_sources.values()) if ath_sources else set()

    rice_labels = read_tsv(LABELS / "rice_raw_strict399_Tos17N4_labels.tsv")
    rice_known = set(rice_labels["gene_id"].str.upper())
    rice_sources = {
        name: identifiers_in_file(path, RICE_RE) for name, path in RICE_SOURCE_FILES.items()
    }
    rice_recorded = set().union(*rice_sources.values()) if rice_sources else set()

    ath_label = dict(zip(ath_3359["gene_id"], ath_3359["label"].astype(str)))
    rice_label = dict(zip(rice_labels["gene_id"].str.upper(), rice_labels["label"].astype(str)))
    rows: list[dict[str, str]] = []
    for species, known, pseudo, recorded, sources, label_map in [
        ("arabidopsis", ath_known, ath_pseudo, ath_recorded, ath_sources, ath_label),
        ("rice", rice_known, set(), rice_recorded, rice_sources, rice_label),
    ]:
        all_ids = known | pseudo | recorded
        for gene in sorted(all_ids):
            if gene in known:
                status = "known_label_used_in_study"
            elif gene in pseudo:
                status = "pseudo_label_used_in_study"
            else:
                status = "phenotype_recorded_but_excluded"
            source_names = sorted(name for name, ids in sources.items() if gene in ids)
            rows.append(
                {
                    "species": species,
                    "gene_id_key": gene,
                    "candidate_status": status,
                    "study_label": label_map.get(gene, ""),
                    "recorded_source_count": str(len(source_names)),
                    "recorded_sources": ";".join(source_names),
                }
            )
    registry = pd.DataFrame(rows)
    metadata = {
        "created_on": date.today().isoformat(),
        "ath_known_consensus_2plus": len(ath_known),
        "ath_historical_pseudo_labels": len(ath_pseudo),
        "ath_raw_phenotype_union": len(ath_recorded),
        "rice_known_strict399_Tos17N4": len(rice_known),
        "rice_raw_phenotype_union": len(rice_recorded),
        "source_availability": {
            name: {"path": str(path), "exists": path.exists(), "ids": len(ath_sources[name])}
            for name, path in ATH_SOURCE_FILES.items()
        }
        | {
            name: {"path": str(path), "exists": path.exists(), "ids": len(rice_sources[name])}
            for name, path in RICE_SOURCE_FILES.items()
        },
    }
    return registry, metadata


def import_profile_predictor():
    sys.path.insert(0, str(ROOT / "scripts" / "prediction"))
    from predict_from_profile_features import predict_profile  # noqa: PLC0415

    return predict_profile


def profile_predictions(npz_path: Path, profile: str) -> pd.DataFrame:
    """Use the released sequence+PLM profile as annotation-light robustness test."""
    model_dir = MODELS / "deployable_feature_profiles" / profile
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    names = data["feature_names"].astype(str).tolist()
    indices = np.asarray(manifest["selected_common6751_indices"], dtype=int)
    selected_x = x[:, indices]
    selected_names = [names[index] for index in indices]
    probability, threshold, method = import_profile_predictor()(model_dir / "model.joblib", selected_x, selected_names)
    return pd.DataFrame(
        {
            "gene_id_key": pd.Series(data["gene_id"].astype(str)).str.upper(),
            "annotation_light_probability": probability,
            "annotation_light_threshold": threshold,
            "annotation_light_method": method,
        }
    )


def load_prediction_pair(species: str) -> pd.DataFrame:
    single_name = (
        "arabidopsis_unknown20460_single_model_predictions.tsv"
        if species == "arabidopsis"
        else "rice_unknown_all_single_model_predictions.tsv"
    )
    joint_name = (
        "arabidopsis_unknown20460_joint_model_predictions.tsv"
        if species == "arabidopsis"
        else "rice_unknown_all_joint_model_predictions.tsv"
    )
    single = read_tsv(PREDICTIONS / single_name).rename(
        columns={
            "essential_probability": "single_species_probability",
            "classification_threshold": "single_species_threshold",
            "predicted_label": "single_species_predicted_label",
            "predicted_class": "single_species_predicted_class",
            "model_name": "single_species_model",
        }
    )
    joint = read_tsv(PREDICTIONS / joint_name).rename(
        columns={
            "essential_probability": "joint_probability",
            "classification_threshold": "joint_threshold",
            "predicted_label": "joint_predicted_label",
            "predicted_class": "joint_predicted_class",
            "model_name": "joint_model",
        }
    )
    join_cols = ["gene_id", "joint_probability", "joint_threshold", "joint_predicted_label", "joint_predicted_class", "joint_model"]
    merged = single.merge(joint[join_cols], on="gene_id", validate="one_to_one")
    merged["gene_id_key"] = merged["gene_id"].str.upper()
    return merged


def classify_predictions(frame: pd.DataFrame, registry: pd.DataFrame, species: str) -> pd.DataFrame:
    registry_species = registry.loc[registry["species"].eq(species)].drop(columns="species")
    result = frame.merge(registry_species, on="gene_id_key", how="left")
    result["candidate_status"] = result["candidate_status"].fillna("true_unknown_candidate")
    result["recorded_source_count"] = result["recorded_source_count"].fillna("0").astype(int)
    result["recorded_sources"] = result["recorded_sources"].fillna("")
    result["model_consensus_essential"] = (
        result["single_species_predicted_label"].astype(int).eq(1)
        & result["joint_predicted_label"].astype(int).eq(1)
    )
    return result


def rank_candidates(frame: pd.DataFrame, annotation_light: pd.DataFrame, species: str) -> pd.DataFrame:
    result = frame.merge(annotation_light, on="gene_id_key", how="left", validate="one_to_one")
    result["single_rank_percentile"] = result["single_species_probability"].rank(pct=True)
    result["joint_rank_percentile"] = result["joint_probability"].rank(pct=True)
    result["annotation_light_rank_percentile"] = result["annotation_light_probability"].rank(pct=True)
    result["candidate_priority_score"] = (
        0.45 * result["single_rank_percentile"]
        + 0.35 * result["joint_rank_percentile"]
        + 0.20 * result["annotation_light_rank_percentile"].fillna(0)
    )
    result["robust_to_annotation_light_model"] = (
        result["annotation_light_probability"] >= result["annotation_light_threshold"]
    )
    candidates = result.loc[
        result["candidate_status"].eq("true_unknown_candidate")
        & result["model_consensus_essential"]
        & result["robust_to_annotation_light_model"].fillna(False)
    ].copy()
    candidates["species"] = species
    candidates = candidates.sort_values(
        ["candidate_priority_score", "single_species_probability", "joint_probability"],
        ascending=False,
    )
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
    return candidates


def evidence_template(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in candidates.head(10).itertuples(index=False):
        rows.append(
            {
                "species": record.species,
                "gene_id": record.gene_id,
                "gene_id_key": record.gene_id_key,
                "candidate_rank": record.candidate_rank,
                "single_species_probability": record.single_species_probability,
                "joint_probability": record.joint_probability,
                "annotation_light_probability": record.annotation_light_probability,
                "evidence_source": "",
                "source_url_or_accession": "",
                "publication_or_release_date": "",
                "evidence_type": "",  # phenotype, expression, ortholog knockout, complex, stock
                "phenotype_stage_or_context": "",
                "supports_essentiality": "",  # yes, no, unclear
                "independent_of_training_labels": "",
                "independent_of_model_features": "",
                "evidence_strength": "",  # strong, moderate, weak
                "curator_notes": "",
            }
        )
    return pd.DataFrame(rows)


def frozen_manifest(paths: list[Path], metadata: dict[str, object]) -> dict[str, object]:
    return {
        "freeze_date": date.today().isoformat(),
        "purpose": "Locked inputs for independent candidate discovery; do not use candidate evidence to retrain these models.",
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
            if path.exists()
        ],
        "label_registry": metadata,
        "candidate_rules": {
            "true_unknown": "Absent from the frozen true-label, pseudo-label and locally archived phenotype-record union.",
            "primary_model": "Species-specific locked model",
            "supporting_model": "Locked Arabidopsis-rice joint model",
            "annotation_light_check": "Released sequence+PLM joint profile must classify the gene as essential.",
            "selection": "Rank by weighted within-species percentile: 45% single model, 35% joint model, 20% annotation-light model.",
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    registry, registry_meta = status_registry()
    registry.to_csv(OUT / "study_label_and_phenotype_registry.tsv", sep="\t", index=False)

    ath = classify_predictions(load_prediction_pair("arabidopsis"), registry, "arabidopsis")
    rice = classify_predictions(load_prediction_pair("rice"), registry, "rice")
    ath_light = profile_predictions(
        FEATURES / "arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz", "sequence_plm"
    )
    rice_light = profile_predictions(FEATURES / "rice_common6751_all_genes.npz", "sequence_plm")
    ath_candidates = rank_candidates(ath, ath_light, "arabidopsis")
    rice_candidates = rank_candidates(rice, rice_light, "rice")

    for species, table, candidates in [
        ("arabidopsis", ath, ath_candidates),
        ("rice", rice, rice_candidates),
    ]:
        enriched = table.merge(
            candidates[["gene_id_key", "annotation_light_probability", "annotation_light_threshold", "annotation_light_method", "candidate_priority_score", "candidate_rank"]],
            on="gene_id_key", how="left",
        )
        enriched.sort_values(["single_species_probability", "gene_id"], ascending=[False, True]).to_csv(
            PUBLISHED / f"{species}_all_feature_covered_genes_reclassified.tsv", sep="\t", index=False
        )
        candidates.to_csv(OUT / f"{species}_provisional_true_unknown_candidates.tsv", sep="\t", index=False)
        evidence_template(candidates).to_csv(OUT / f"{species}_top10_independent_evidence_template.tsv", sep="\t", index=False)

    pd.DataFrame(
        [
            {
                "species": "arabidopsis",
                "feature_covered_genes": len(ath),
                "true_unknown_candidates": int(ath["candidate_status"].eq("true_unknown_candidate").sum()),
                "consensus_annotation_light_candidates": len(ath_candidates),
                "top10_exported": min(10, len(ath_candidates)),
            },
            {
                "species": "rice",
                "feature_covered_genes": len(rice),
                "true_unknown_candidates": int(rice["candidate_status"].eq("true_unknown_candidate").sum()),
                "consensus_annotation_light_candidates": len(rice_candidates),
                "top10_exported": min(10, len(rice_candidates)),
            },
        ]
    ).to_csv(OUT / "candidate_release_summary.tsv", sep="\t", index=False)

    freeze_paths = [
        LABELS / "arabidopsis_strict2601_fixed_split_labels.tsv",
        LABELS / "rice_strict399_Tos17N4_fixed_split_labels.tsv",
        PREDICTIONS / "arabidopsis_unknown20460_single_model_predictions.tsv",
        PREDICTIONS / "arabidopsis_unknown20460_joint_model_predictions.tsv",
        PREDICTIONS / "rice_unknown_all_single_model_predictions.tsv",
        PREDICTIONS / "rice_unknown_all_joint_model_predictions.tsv",
        FEATURES / "common6751_feature_names.tsv",
        MODELS / "arabidopsis_single_strict2601_common6751" / "selected_model_and_manifest.joblib",
        MODELS / "rice_single_strict399_Tos17N4_common6751" / "model.joblib",
        MODELS / "joint_arabidopsis_rice_common6751" / "model.joblib",
    ]
    (OUT / "frozen_submission_inputs.json").write_text(
        json.dumps(frozen_manifest(freeze_paths, registry_meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote candidate registry to {OUT}")


if __name__ == "__main__":
    main()
