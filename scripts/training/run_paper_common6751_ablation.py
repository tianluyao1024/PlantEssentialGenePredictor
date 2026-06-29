from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import train_joint_ath2601_rice_strict399_common6751 as joint
import train_rice_E0_Nge6_common6751_fixed80_10_10 as trainer


ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/"
    "cross_species_ath_rice_common_features_models"
)
OUT = ROOT / "paper_submission_experiments/common6751_ablation"

BLOCKS = {
    "bio95_only": np.arange(0, 95),
    "esm2_only": np.arange(95, 95 + 2560),
    "protbert_only": np.arange(95 + 2560, 95 + 2560 + 2048),
    "prott5_only": np.arange(95 + 2560 + 2048, 6751),
    "all_plm_6656": np.arange(95, 6751),
    "bio95_plus_esm2": np.r_[np.arange(0, 95), np.arange(95, 95 + 2560)],
    "bio95_plus_protbert": np.r_[
        np.arange(0, 95),
        np.arange(95 + 2560, 95 + 2560 + 2048),
    ],
    "bio95_plus_prott5": np.r_[
        np.arange(0, 95),
        np.arange(95 + 2560 + 2048, 6751),
    ],
    "full_without_esm2": np.r_[
        np.arange(0, 95),
        np.arange(95 + 2560, 6751),
    ],
    "full_without_protbert": np.r_[
        np.arange(0, 95 + 2560),
        np.arange(95 + 2560 + 2048, 6751),
    ],
    "full_without_prott5": np.arange(0, 95 + 2560 + 2048),
    "full_without_GO": np.setdiff1d(
        np.arange(0, 6751),
        np.r_[np.arange(51, 71), np.arange(79, 85)],
    ),
    "full_without_PPI": np.setdiff1d(
        np.arange(0, 6751), np.array([49, 50])
    ),
    "full_without_GO_PPI": np.setdiff1d(
        np.arange(0, 6751),
        np.r_[np.array([49, 50]), np.arange(51, 71), np.arange(79, 85)],
    ),
    "full_common6751": np.arange(0, 6751),
}


def fit_one(
    dataset: str,
    variant: str,
    train_matrix: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    validation_matrix: np.ndarray,
    validation_y: np.ndarray,
    test_matrix: np.ndarray,
    test_y: np.ndarray,
    n_bio: int,
) -> dict:
    version_out = OUT / dataset / variant
    version_out.mkdir(parents=True, exist_ok=True)
    result_path = version_out / "result.tsv"
    if result_path.exists():
        print(f"reuse {dataset}/{variant}", flush=True)
        return pd.read_csv(result_path, sep="\t").iloc[0].to_dict()

    _, predictions, folds, _, _, _ = trainer.fit_library(
        train_matrix,
        train_y,
        weights,
        {"validation": validation_matrix, "test": test_matrix},
        n_bio,
    )
    pd.DataFrame(folds).to_csv(
        version_out / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    candidates = []
    for method in ["meta", "mean", "logit_mean"]:
        val_probability = predictions["validation"][method]
        threshold = joint.select_single_species_threshold(
            validation_y, val_probability
        )
        candidates.append(
            {
                "method": method,
                "validation_auc": roc_auc_score(
                    validation_y, val_probability
                ),
                "validation_auprc": average_precision_score(
                    validation_y, val_probability
                ),
                "threshold": threshold["threshold"],
            }
        )
    candidate_table = pd.DataFrame(candidates).sort_values(
        ["validation_auc", "validation_auprc"], ascending=False
    )
    candidate_table.to_csv(
        version_out / "validation_model_selection.tsv",
        sep="\t",
        index=False,
    )
    method = str(candidate_table.iloc[0]["method"])
    threshold = joint.select_single_species_threshold(
        validation_y, predictions["validation"][method]
    )
    val_metrics = joint.metrics(
        validation_y,
        predictions["validation"][method],
        threshold["threshold"],
    )
    test_metrics = joint.metrics(
        test_y,
        predictions["test"][method],
        threshold["threshold"],
    )
    pd.DataFrame(
        {
            "label": test_y,
            "probability": predictions["test"][method],
            "threshold": threshold["threshold"],
        }
    ).to_csv(version_out / "test_predictions.tsv", sep="\t", index=False)
    result = {
        "dataset": dataset,
        "variant": variant,
        "feature_count": int(train_matrix.shape[1]),
        "bio_feature_count": int(n_bio),
        "selected_method": method,
        **{f"validation_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    pd.DataFrame([result]).to_csv(result_path, sep="\t", index=False)
    return result


def fit_joint_one(
    variant: str,
    train_matrix: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    rice_validation: np.ndarray,
    rice_validation_y: np.ndarray,
    rice_test: np.ndarray,
    rice_test_y: np.ndarray,
    ath_validation: np.ndarray,
    ath_validation_y: np.ndarray,
    ath_test: np.ndarray,
    ath_test_y: np.ndarray,
    n_bio: int,
) -> list[dict]:
    dataset = "joint"
    version_out = OUT / dataset / variant
    version_out.mkdir(parents=True, exist_ok=True)
    result_path = version_out / "result.tsv"
    if result_path.exists():
        print(f"reuse joint/{variant}", flush=True)
        return pd.read_csv(result_path, sep="\t").to_dict("records")

    _, predictions, folds, _, _, _ = trainer.fit_library(
        train_matrix,
        train_y,
        weights,
        {
            "rice_validation": rice_validation,
            "rice_test": rice_test,
            "ath_validation": ath_validation,
            "ath_test": ath_test,
        },
        n_bio,
    )
    pd.DataFrame(folds).to_csv(
        version_out / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    methods = []
    thresholds = {}
    for method in ["meta", "mean", "logit_mean"]:
        selected, search = joint.select_joint_threshold(
            rice_validation_y,
            predictions["rice_validation"][method],
            ath_validation_y,
            predictions["ath_validation"][method],
        )
        thresholds[method] = selected
        search.to_csv(
            version_out / f"{method}_joint_threshold_search.tsv",
            sep="\t",
            index=False,
        )
        rice_auc = roc_auc_score(
            rice_validation_y, predictions["rice_validation"][method]
        )
        ath_auc = roc_auc_score(
            ath_validation_y, predictions["ath_validation"][method]
        )
        methods.append(
            {
                "method": method,
                "rice_validation_auc": rice_auc,
                "ath_validation_auc": ath_auc,
                "min_validation_auc": min(rice_auc, ath_auc),
                "mean_validation_auc": np.mean([rice_auc, ath_auc]),
                "threshold": selected["threshold"],
                "min_validation_sp": selected["min_sp"],
            }
        )
    method_table = pd.DataFrame(methods).sort_values(
        ["min_validation_auc", "mean_validation_auc", "min_validation_sp"],
        ascending=False,
    )
    method_table.to_csv(
        version_out / "validation_model_selection.tsv",
        sep="\t",
        index=False,
    )
    method = str(method_table.iloc[0]["method"])
    threshold = thresholds[method]["threshold"]
    rows = []
    for species, y, probability in [
        ("rice", rice_test_y, predictions["rice_test"][method]),
        ("arabidopsis", ath_test_y, predictions["ath_test"][method]),
    ]:
        score = joint.metrics(y, probability, threshold)
        rows.append(
            {
                "dataset": dataset,
                "evaluation_species": species,
                "variant": variant,
                "feature_count": int(train_matrix.shape[1]),
                "bio_feature_count": int(n_bio),
                "selected_method": method,
                **{f"test_{k}": v for k, v in score.items()},
            }
        )
        pd.DataFrame(
            {
                "label": y,
                "probability": probability,
                "threshold": threshold,
            }
        ).to_csv(
            version_out / f"{species}_test_predictions.tsv",
            sep="\t",
            index=False,
        )
    pd.DataFrame(rows).to_csv(result_path, sep="\t", index=False)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rice_x, names, rice_meta, rice_split = joint.load_rice()
    ath_x, ath_names, ath_meta, ath_split = joint.load_ath()
    if names != ath_names:
        raise RuntimeError("Rice and Arabidopsis feature names differ")
    rice_idx = {
        key: joint.index_genes(rice_meta, value)
        for key, value in rice_split.items()
    }
    ath_idx = {
        key: joint.index_genes(ath_meta, value)
        for key, value in ath_split.items()
    }
    joint_labels = joint.build_weighted_joint_labels(
        rice_split["train"], ath_split["train"]
    )
    joint_full = np.vstack(
        [rice_x[rice_idx["train"]], ath_x[ath_idx["train"]]]
    ).astype(np.float32)

    results = []
    for variant, columns in BLOCKS.items():
        print(f"\n=== Ablation {variant} ===", flush=True)
        n_bio = {
            "bio95_only": 0,
            "esm2_only": 0,
            "protbert_only": 0,
            "prott5_only": 0,
            "all_plm_6656": 0,
            "bio95_plus_esm2": 95,
            "bio95_plus_protbert": 95,
            "bio95_plus_prott5": 95,
            "full_without_esm2": 95,
            "full_without_protbert": 95,
            "full_without_prott5": 95,
            "full_without_GO": 69,
            "full_without_PPI": 93,
            "full_without_GO_PPI": 67,
            "full_common6751": 95,
        }[variant]
        results.append(
            fit_one(
                "rice",
                variant,
                rice_x[rice_idx["train"]][:, columns],
                rice_split["train"]["label"].to_numpy(np.int8),
                np.ones(len(rice_split["train"]), dtype=np.float32),
                rice_x[rice_idx["validation"]][:, columns],
                rice_split["validation"]["label"].to_numpy(np.int8),
                rice_x[rice_idx["test"]][:, columns],
                rice_split["test"]["label"].to_numpy(np.int8),
                n_bio,
            )
        )
        ath_weights = np.where(
            ath_split["train"]["training_component"].eq(
                "core1623_train80"
            ),
            1.0,
            0.5,
        ).astype(np.float32)
        results.append(
            fit_one(
                "arabidopsis",
                variant,
                ath_x[ath_idx["train"]][:, columns],
                ath_split["train"]["label"].to_numpy(np.int8),
                ath_weights,
                ath_x[ath_idx["validation"]][:, columns],
                ath_split["validation"]["label"].to_numpy(np.int8),
                ath_x[ath_idx["test"]][:, columns],
                ath_split["test"]["label"].to_numpy(np.int8),
                n_bio,
            )
        )
        results.extend(
            fit_joint_one(
                variant,
                joint_full[:, columns],
                joint_labels["label"].to_numpy(np.int8),
                joint_labels["sample_weight"].to_numpy(np.float32),
                rice_x[rice_idx["validation"]][:, columns],
                rice_split["validation"]["label"].to_numpy(np.int8),
                rice_x[rice_idx["test"]][:, columns],
                rice_split["test"]["label"].to_numpy(np.int8),
                ath_x[ath_idx["validation"]][:, columns],
                ath_split["validation"]["label"].to_numpy(np.int8),
                ath_x[ath_idx["test"]][:, columns],
                ath_split["test"]["label"].to_numpy(np.int8),
                n_bio,
            )
        )
        pd.DataFrame(results).to_csv(
            OUT / "ablation_results_partial.tsv", sep="\t", index=False
        )

    table = pd.DataFrame(results)
    table.to_csv(OUT / "complete_ablation_results.tsv", sep="\t", index=False)
    manifest = {
        "variants": {key: int(len(value)) for key, value in BLOCKS.items()},
        "fixed_splits": {
            "rice": {key: len(value) for key, value in rice_split.items()},
            "arabidopsis": {
                key: len(value) for key, value in ath_split.items()
            },
        },
        "training": (
            "Each ablation refits imputation, PCA, supervised feature "
            "selection, base learners, and stacking on the fixed training set."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
