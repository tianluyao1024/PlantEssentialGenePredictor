"""Quantify prediction dispersion across released ensemble components.

This is an ensemble-stability analysis, not an external validation result.  It
is kept separate from the candidate registry so the frozen model artefacts are
never modified while candidate priorities are updated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tpc_candidate_resource"
FEATURES = ROOT / "data" / "processed_features"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "scripts" / "prediction"))
from predict_from_processed_features import patch_sklearn_compat, transform_with  # noqa: E402


def load_matrix(species: str) -> tuple[np.ndarray, pd.Series]:
    path = (
        FEATURES / "arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz"
        if species == "arabidopsis"
        else FEATURES / "rice_common6751_all_genes.npz"
    )
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), pd.Series(data["gene_id"].astype(str)).str.upper()


def package_component_probabilities(model_path: Path, x: np.ndarray) -> np.ndarray:
    """Return one mean base-learner probability per stored deployment bundle."""
    package = patch_sklearn_compat(joblib.load(model_path))
    component = []
    if "deployment_base_models" in package:
        bundles = package["deployment_base_models"]
        n_bio = int(package["n_bio"])
    elif "base_model_bundle_paths" in package:
        bundles = [patch_sklearn_compat(joblib.load(path)) for path in package["base_model_bundle_paths"]]
        n_bio = None
    else:
        raise ValueError(f"Unsupported model package: {model_path}")
    for bundle in bundles:
        transformed = transform_with(bundle["transforms"], x, int(bundle.get("n_bio", n_bio)))
        probabilities = [
            model.predict_proba(transformed)[:, 1]
            for model in bundle["models"].values()
        ]
        component.append(np.mean(probabilities, axis=0))
    return np.column_stack(component)


def annotate(species: str, candidates: pd.DataFrame) -> pd.DataFrame:
    x, genes = load_matrix(species)
    index = pd.Series(np.arange(len(genes)), index=genes).groupby(level=0).first()
    candidate_index = candidates["gene_id_key"].map(index)
    if candidate_index.isna().any():
        missing = candidates.loc[candidate_index.isna(), "gene_id"].head(10).tolist()
        raise ValueError(f"Feature rows missing for {species}: {missing}")
    selected = x[candidate_index.astype(int).to_numpy()]
    single_model = (
        MODELS / "arabidopsis_single_strict2601_common6751" / "selected_model_and_manifest.joblib"
        if species == "arabidopsis"
        else MODELS / "rice_single_strict399_Tos17N4_common6751" / "model.joblib"
    )
    joint_model = MODELS / "joint_arabidopsis_rice_common6751" / "model.joblib"
    single = package_component_probabilities(single_model, selected)
    joint = package_component_probabilities(joint_model, selected)
    result = candidates[["species", "gene_id", "gene_id_key", "candidate_rank"]].copy()
    for prefix, values in [("single_component", single), ("joint_component", joint)]:
        result[f"{prefix}_count"] = values.shape[1]
        result[f"{prefix}_mean"] = values.mean(axis=1)
        result[f"{prefix}_sd"] = values.std(axis=1, ddof=0)
        result[f"{prefix}_min"] = values.min(axis=1)
        result[f"{prefix}_max"] = values.max(axis=1)
    return result


def update_candidate_table(species: str, stability: pd.DataFrame) -> None:
    candidate_path = OUT / f"{species}_provisional_true_unknown_candidates.tsv"
    candidates = pd.read_csv(candidate_path, sep="\t")
    result = candidates.merge(stability, on=["species", "gene_id", "gene_id_key", "candidate_rank"], validate="one_to_one")
    # Lower dispersion is preferable.  Percentile ranks make the scale comparable
    # between species and models without treating the components as independent CVs.
    result["single_stability_score"] = 1 - result["single_component_sd"].rank(pct=True)
    result["joint_stability_score"] = 1 - result["joint_component_sd"].rank(pct=True)
    result["stability_adjusted_priority_score"] = (
        0.80 * result["candidate_priority_score"]
        + 0.10 * result["single_stability_score"]
        + 0.10 * result["joint_stability_score"]
    )
    result = result.sort_values(
        ["stability_adjusted_priority_score", "candidate_priority_score"], ascending=False
    ).reset_index(drop=True)
    result["stability_adjusted_rank"] = np.arange(1, len(result) + 1)
    result.to_csv(candidate_path, sep="\t", index=False)
    result.head(10).to_csv(OUT / f"{species}_top10_after_ensemble_stability.tsv", sep="\t", index=False)


def main() -> None:
    for species in ["arabidopsis", "rice"]:
        candidates = pd.read_csv(OUT / f"{species}_provisional_true_unknown_candidates.tsv", sep="\t")
        stability = annotate(species, candidates)
        stability.to_csv(OUT / f"{species}_candidate_ensemble_stability.tsv", sep="\t", index=False)
        update_candidate_table(species, stability)
    manifest = {
        "analysis": "Ensemble-component prediction dispersion",
        "interpretation": "Lower standard deviation indicates higher agreement among stored deployment bundles; this is not a new training run or an external validation cohort.",
        "models": {
            "arabidopsis_single": "arabidopsis_single_strict2601_common6751",
            "rice_single": "rice_single_strict399_Tos17N4_common6751",
            "joint": "joint_arabidopsis_rice_common6751",
        },
    }
    (OUT / "ensemble_stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
