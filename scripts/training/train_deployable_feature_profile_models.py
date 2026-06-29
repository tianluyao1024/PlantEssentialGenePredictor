from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "feature_extraction"))

import train_joint_ath2601_rice_strict399_common6751 as joint
import train_rice_E0_Nge6_common6751_fixed80_10_10 as trainer


OUT = ROOT / "models" / "deployable_feature_profiles"

BASE_SEQUENCE_FEATURES = [
    "protein_length",
    "cds_length",
    "gc_content",
    "at_content",
    "gc_skew",
    "at_skew",
    "gc3_content",
    "nt_freq_A",
    "nt_freq_C",
    "nt_freq_G",
    "nt_freq_T",
    "aa_freq_A",
    "aa_freq_C",
    "aa_freq_D",
    "aa_freq_E",
    "aa_freq_F",
    "aa_freq_G",
    "aa_freq_H",
    "aa_freq_I",
    "aa_freq_K",
    "aa_freq_L",
    "aa_freq_M",
    "aa_freq_N",
    "aa_freq_P",
    "aa_freq_Q",
    "aa_freq_R",
    "aa_freq_S",
    "aa_freq_T",
    "aa_freq_V",
    "aa_freq_W",
    "aa_freq_Y",
    "aa_group_hydrophobic",
    "aa_group_polar",
    "aa_group_positive",
    "aa_group_negative",
    "aa_group_small",
    "aa_group_aromatic",
    "aa_group_sulfur",
    "protein_molecular_weight",
    "protein_gravy",
    "protein_isoelectric_point",
    "protein_aromaticity",
    "protein_instability_index",
]
GO_FEATURES = [
    "go_cellular_component_organization",
    "go_rna_binding",
    "go_cell_cycle",
    "go_response_to_stress",
    "go_dna_binding_transcription_factor_activity",
    "go_translation",
    "go_response_to_abiotic_stimulus",
    "go_nucleic_acid_binding",
    "go_pollination",
    "go_dna_binding",
    "go_response_to_biotic_stimulus",
    "go_response_to_light_stimulus",
    "go_ribosome",
    "go_nucleolus",
    "go_structural_molecule_activity",
    "go_signal_transduction",
    "go_nucleobase_containing_compound_metabolic_process",
    "go_chloroplast",
    "go_extracellular_region",
    "go_response_to_endogenous_stimulus",
    "go_embryo_development",
    "go_multicellular_organism_development",
    "go_post_embryonic_development",
    "go_anatomical_structure_development",
    "go_response_to_external_stimulus",
    "go_response_to_chemical",
]
PPI_FEATURES = ["string_network_connections_400", "string_network_connections_700"]
EXPRESSION_FEATURES = [
    "median_expression",
    "expression_variation",
    "expression_breadth",
    "expression_module_size",
]
GENE_STRUCTURE_FEATURES = ["gene_span_bp"]
PARALOG_DOMAIN_HOMOLOG_FEATURES = [
    "gene_family_size",
    "singleton_status",
    "paralog_percentage_identity",
    "top_paralog_bitscore",
    "tandem_duplicate",
    "domain_number",
    "pfam_domain_number",
    "ensembl_compara_paralog_count",
    "ensembl_compara_max_paralog_percent_identity",
    "alyrata_homolog_found",
    "alyrata_homolog_percent_identity",
    "ptrichocarpa_homolog_found",
    "ptrichocarpa_homolog_percent_identity",
    "vvinifera_homolog_found",
    "vvinifera_homolog_percent_identity",
    "ppatens_homolog_found",
    "ppatens_homolog_percent_identity",
    "percentage_identity_in_plants",
    "ensembl_compara_paralog_lca_type_count",
]

PROFILES = {
    "sequence_plm": BASE_SEQUENCE_FEATURES,
    "sequence_plm_go": BASE_SEQUENCE_FEATURES + GO_FEATURES,
    "sequence_plm_ppi": BASE_SEQUENCE_FEATURES + PPI_FEATURES,
    "sequence_plm_expression": BASE_SEQUENCE_FEATURES + EXPRESSION_FEATURES,
    "sequence_plm_go_ppi": BASE_SEQUENCE_FEATURES + GO_FEATURES + PPI_FEATURES,
    "sequence_plm_go_expression": BASE_SEQUENCE_FEATURES + GO_FEATURES + EXPRESSION_FEATURES,
    "sequence_plm_ppi_expression": BASE_SEQUENCE_FEATURES + PPI_FEATURES + EXPRESSION_FEATURES,
    "sequence_plm_go_ppi_expression": BASE_SEQUENCE_FEATURES + GO_FEATURES + PPI_FEATURES + EXPRESSION_FEATURES,
    "full_uploadable_without_cross_species_homologs": (
        BASE_SEQUENCE_FEATURES
        + GENE_STRUCTURE_FEATURES
        + PPI_FEATURES
        + GO_FEATURES
        + EXPRESSION_FEATURES
        + PARALOG_DOMAIN_HOMOLOG_FEATURES
    ),
}


def plm_indices(names: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(names) if name.startswith(("esm2_", "protbert_", "prott5_"))]


def profile_indices(names: list[str], profile: str) -> tuple[np.ndarray, int, list[str]]:
    lookup = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in PROFILES[profile] if name not in lookup]
    if missing:
        raise RuntimeError(f"{profile} missing feature names: {missing}")
    bio_names = list(dict.fromkeys(PROFILES[profile]))
    selected_names = bio_names + [names[idx] for idx in plm_indices(names)]
    selected = np.array([lookup[name] for name in selected_names], dtype=int)
    return selected, len(bio_names), selected_names


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probability >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "precision": tp / max(1, tp + fp),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train_profile(profile: str, force: bool = False) -> list[dict]:
    out_dir = OUT / profile
    result_file = out_dir / "result.tsv"
    model_file = out_dir / "model.joblib"
    if result_file.exists() and model_file.exists() and not force:
        return pd.read_csv(result_file, sep="\t").to_dict("records")

    out_dir.mkdir(parents=True, exist_ok=True)
    rice_x, names, rice_meta, rice_split = joint.load_rice()
    ath_x, ath_names, ath_meta, ath_split = joint.load_ath()
    if names != ath_names:
        raise RuntimeError("Feature names differ between rice and Arabidopsis")
    selected, n_bio, selected_names = profile_indices(names, profile)

    rice_idx = {key: joint.index_genes(rice_meta, value) for key, value in rice_split.items()}
    ath_idx = {key: joint.index_genes(ath_meta, value) for key, value in ath_split.items()}
    joint_labels = joint.build_weighted_joint_labels(rice_split["train"], ath_split["train"])
    train_x = np.vstack([rice_x[rice_idx["train"]][:, selected], ath_x[ath_idx["train"]][:, selected]]).astype(np.float32)
    train_y = joint_labels["label"].to_numpy(np.int8)
    weights = joint_labels["sample_weight"].to_numpy(np.float32)

    targets = {
        "rice_validation": rice_x[rice_idx["validation"]][:, selected],
        "rice_test": rice_x[rice_idx["test"]][:, selected],
        "ath_validation": ath_x[ath_idx["validation"]][:, selected],
        "ath_test": ath_x[ath_idx["test"]][:, selected],
    }
    _, predictions, folds, meta, base_models, column_names = trainer.fit_library(
        train_x,
        train_y,
        weights,
        targets,
        n_bio,
    )
    pd.DataFrame(folds).to_csv(out_dir / "inner_oof_fold_scores.tsv", sep="\t", index=False)

    candidates = []
    thresholds = {}
    for method in ["meta", "mean", "logit_mean"]:
        selected_threshold, search = joint.select_joint_threshold(
            rice_split["validation"]["label"].to_numpy(np.int8),
            predictions["rice_validation"][method],
            ath_split["validation"]["label"].to_numpy(np.int8),
            predictions["ath_validation"][method],
        )
        thresholds[method] = float(selected_threshold["threshold"])
        search.to_csv(out_dir / f"{method}_joint_threshold_search.tsv", sep="\t", index=False)
        rice_auc = roc_auc_score(rice_split["validation"]["label"], predictions["rice_validation"][method])
        ath_auc = roc_auc_score(ath_split["validation"]["label"], predictions["ath_validation"][method])
        candidates.append(
            {
                "method": method,
                "threshold": thresholds[method],
                "rice_validation_auc": float(rice_auc),
                "ath_validation_auc": float(ath_auc),
                "min_validation_auc": float(min(rice_auc, ath_auc)),
                "mean_validation_auc": float(np.mean([rice_auc, ath_auc])),
            }
        )
    model_selection = pd.DataFrame(candidates).sort_values(
        ["min_validation_auc", "mean_validation_auc"], ascending=False
    )
    model_selection.to_csv(out_dir / "validation_model_selection.tsv", sep="\t", index=False)
    method = str(model_selection.iloc[0]["method"])
    threshold = float(model_selection.iloc[0]["threshold"])

    rows = []
    for species, split_name, y, probability in [
        ("rice", "validation", rice_split["validation"]["label"].to_numpy(np.int8), predictions["rice_validation"][method]),
        ("rice", "test", rice_split["test"]["label"].to_numpy(np.int8), predictions["rice_test"][method]),
        ("arabidopsis", "validation", ath_split["validation"]["label"].to_numpy(np.int8), predictions["ath_validation"][method]),
        ("arabidopsis", "test", ath_split["test"]["label"].to_numpy(np.int8), predictions["ath_test"][method]),
    ]:
        score = metrics(y, probability, threshold)
        rows.append(
            {
                "profile": profile,
                "evaluation_species": species,
                "split": split_name,
                "selected_method": method,
                "threshold": threshold,
                "feature_count": int(len(selected)),
                "bio_feature_count": int(n_bio),
                **score,
            }
        )
    pd.DataFrame(rows).to_csv(result_file, sep="\t", index=False)
    (out_dir / "feature_names.tsv").write_text(
        "feature_name\n" + "\n".join(selected_names) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "profile": profile,
        "selected_method": method,
        "threshold": threshold,
        "n_bio": n_bio,
        "feature_count": len(selected),
        "source_schema": "common6751",
        "selected_common6751_indices": selected.tolist(),
        "feature_names": selected_names,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "profile": profile,
            "n_bio": n_bio,
            "feature_count": len(selected),
            "selected_common6751_indices": selected,
            "feature_names": selected_names,
            "selected_method": method,
            "threshold": threshold,
            "deployment_base_models": base_models,
            "meta_model": meta,
            "meta_feature_names": column_names,
        },
        model_file,
        compress=3,
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for profile in PROFILES:
        print(f"\n=== Training deployable profile: {profile} ===", flush=True)
        all_rows.extend(train_profile(profile))
        pd.DataFrame(all_rows).to_csv(OUT / "profile_model_comparison_partial.tsv", sep="\t", index=False)
    pd.DataFrame(all_rows).to_csv(OUT / "profile_model_comparison.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
