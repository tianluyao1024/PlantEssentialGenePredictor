from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_ath_high_confidence_models as base
import train_ath_strict_inclusive_paired_fixed_core_split as runner


RICE_FEATURE_NAMES = Path(
    "E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models/"
    "rice_v2_high_quality_label_tier_comparison/E_hq_plus_N_E0_Nge3/feature_names.tsv"
)
CORE_SPLIT = Path(
    "D:/拟南芥/模型/strict_vs_inclusive_paired_core1623_fixed80_10_10/"
    "shared_core1623_fixed80_10_10_split.tsv"
)
STRICT_LABELS = Path(
    "D:/拟南芥/模型/essential_gene_prediction_1623_plus_prediction_experiment_concordant/"
    "high_confidence_1623_plus_concordant_strict_labels.tsv"
)
LABELS_3359 = Path(
    "D:/拟南芥/模型/essential_gene_prediction_consensus1623_plus_pseudo06_predict_all_unknown/"
    "training_labels_true1623_plus_pseudo06.tsv"
)
OUT_ROOT = Path(
    "D:/拟南芥/模型/ath_three_labelsets_common6751_fixed_core1623_80_10_10"
)

DROP_DIRECTIONAL_HOMOLOG_FEATURES = {
    "rice_homolog_found",
    "rice_homolog_percent_identity",
    "homolog_not_found_in_rice",
}


def load_label_table(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, sep="\t")
    labels["gene_id"] = labels["gene_id"].astype(str).str.upper()
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(np.int8)
    if labels["gene_id"].duplicated().any():
        raise RuntimeError(f"Duplicated genes in {path}")
    return labels


def build_common_6751(
    X_all: np.ndarray,
    ath_feature_names: list[str],
    ath_n_bio: int,
) -> tuple[np.ndarray, list[str], int, pd.DataFrame]:
    rice = pd.read_csv(RICE_FEATURE_NAMES, sep="\t")
    common_bio = rice.loc[
        rice["block"].eq("bio")
        & ~rice["feature_name"].isin(DROP_DIRECTIONAL_HOMOLOG_FEATURES),
        "feature_name",
    ].tolist()
    if len(common_bio) != 95:
        raise RuntimeError(f"Expected 95 common bio features, got {len(common_bio)}")

    ath_name_to_idx = {name: idx for idx, name in enumerate(ath_feature_names)}
    missing = [name for name in common_bio if name not in ath_name_to_idx]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} common features in Arabidopsis: {missing}")

    plm_names = ath_feature_names[ath_n_bio:]
    expected_prefix_counts = {
        "esm2": 2560,
        "protbert": 2048,
        "prott5": 2048,
    }
    observed_prefix_counts = {
        prefix: sum(name.startswith(prefix + "_") for name in plm_names)
        for prefix in expected_prefix_counts
    }
    if observed_prefix_counts != expected_prefix_counts:
        raise RuntimeError(
            f"Unexpected PLM dimensions: {observed_prefix_counts}"
        )

    selected_names = common_bio + plm_names
    selected_indices = [ath_name_to_idx[name] for name in selected_names]
    X_common = X_all[:, selected_indices].astype(np.float32)
    if X_common.shape[1] != 6751:
        raise RuntimeError(f"Expected 6751 features, got {X_common.shape[1]}")

    mapping_rows = []
    for output_idx, name in enumerate(selected_names):
        mapping_rows.append(
            {
                "output_index": output_idx,
                "feature_name": name,
                "block": "bio" if output_idx < 95 else "plm",
                "original_ath_index": ath_name_to_idx[name],
            }
        )
    return X_common, selected_names, 95, pd.DataFrame(mapping_rows)


def prepare_training_sets(core_split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core_train = core_split[core_split["split"].eq("train")].copy()
    core_train["training_component"] = "core1623_train80"

    strict = load_label_table(STRICT_LABELS)
    strict_added = strict[
        strict["label_source"].ne("original_consensus_2plus")
    ].copy()
    strict_added["training_component"] = "strict_concordant_addition"
    strict_train = pd.concat([core_train, strict_added], ignore_index=True)

    labels_3359 = load_label_table(LABELS_3359)
    pseudo_1736 = labels_3359[
        labels_3359["label_source"].eq("pseudo_0.60_0.40_from_2216")
    ].copy()
    pseudo_1736["training_component"] = "teacher_pseudo_0.60_0.40"
    teacher_train = pd.concat([core_train, pseudo_1736], ignore_index=True)

    training_sets = {
        "core1623_common6751": core_train,
        "strict2601_common6751": strict_train,
        "teacher3359_common6751": teacher_train,
    }
    expected = {
        "core1623_common6751": (1297, 318, 979),
        "strict2601_common6751": (2275, 428, 1847),
        "teacher3359_common6751": (3033, 1012, 2021),
    }
    for name, labels in training_sets.items():
        if labels["gene_id"].duplicated().any():
            duplicated = labels.loc[
                labels["gene_id"].duplicated(False), "gene_id"
            ].tolist()
            raise RuntimeError(f"{name}: duplicated genes {duplicated[:5]}")
        observed = (
            len(labels),
            int(labels["label"].sum()),
            int((labels["label"] == 0).sum()),
        )
        if observed != expected[name]:
            raise RuntimeError(f"{name}: expected {expected[name]}, got {observed}")
    return training_sets


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    core_split = pd.read_csv(CORE_SPLIT, sep="\t")
    core_split["gene_id"] = core_split["gene_id"].astype(str).str.upper()
    core_split["label"] = pd.to_numeric(
        core_split["label"], errors="raise"
    ).astype(np.int8)

    validation = core_split[core_split["split"].eq("validation")]
    test = core_split[core_split["split"].eq("test")]
    if len(validation) != 163 or len(test) != 163:
        raise RuntimeError(
            f"Expected 163 validation and 163 test genes, got {len(validation)}, {len(test)}"
        )

    X_full, _ids_all, genes_all, ath_feature_names, ath_n_bio = base.load_matrix()
    genes_all = np.array([str(gene).upper() for gene in genes_all])
    X_common, selected_names, common_n_bio, mapping = build_common_6751(
        X_full,
        ath_feature_names,
        ath_n_bio,
    )
    mapping.to_csv(
        OUT_ROOT / "common6751_feature_mapping.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"feature_name": selected_names}).to_csv(
        OUT_ROOT / "common6751_feature_names.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {"removed_ath_specific_or_directional_feature": sorted(DROP_DIRECTIONAL_HOMOLOG_FEATURES)}
    ).to_csv(
        OUT_ROOT / "removed_directional_homolog_features.tsv",
        sep="\t",
        index=False,
    )

    training_sets = prepare_training_sets(core_split)
    scores = []
    runner.OUT_ROOT = OUT_ROOT
    for version, labels in training_sets.items():
        version_dir = OUT_ROOT / version
        version_dir.mkdir(parents=True, exist_ok=True)

        core_train_count = int(
            labels["training_component"].eq("core1623_train80").sum()
        )
        additions = labels[
            ~labels["training_component"].eq("core1623_train80")
        ].copy()
        if core_train_count != 1297:
            raise RuntimeError(f"{version}: core train count {core_train_count}")

        score = runner.run_version(
            version,
            core_split,
            additions,
            X_common,
            genes_all,
            selected_names,
            common_n_bio,
        )
        scores.append(score)

    comparison = pd.DataFrame(scores).sort_values(
        ["test_auc", "test_auprc"], ascending=False
    )
    comparison.to_csv(
        OUT_ROOT / "three_models_common6751_shared_test_comparison.tsv",
        sep="\t",
        index=False,
    )
    manifest = {
        "feature_definition": (
            "95 cross-species common biological features plus 6656 PLM features; "
            "three directional Arabidopsis-rice homolog features removed"
        ),
        "removed_features": sorted(DROP_DIRECTIONAL_HOMOLOG_FEATURES),
        "feature_count": 6751,
        "bio_feature_count": 95,
        "plm_feature_count": 6656,
        "shared_split_source": str(CORE_SPLIT),
        "validation_genes": 163,
        "validation_essential": 40,
        "validation_nonessential": 123,
        "test_genes": 163,
        "test_essential": 40,
        "test_nonessential": 123,
        "training_design": {
            "core1623_common6751": (
                "1297 core training genes; held-out validation/test excluded"
            ),
            "strict2601_common6751": (
                "1297 core training genes + 978 strict concordant additions = 2275"
            ),
            "teacher3359_common6751": (
                "1297 core training genes + 1736 teacher pseudo labels = 3033; "
                "the 326 shared validation/test core genes are excluded to prevent leakage"
            ),
        },
        "scores": scores,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
