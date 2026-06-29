from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

import train_ath_high_confidence_models as ath_features
import train_ath_three_labelsets_common6751_fixed_split as ath_common
import train_ath_unknown20460_pseudo_train_validate3359_50_50 as ath_stack
import train_rice_E0_Nge6_common6751_fixed80_10_10 as common
import train_rice_N6_four_feature_variants_fixed_split as rice_features


ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/"
    "cross_species_ath_rice_common_features_models"
)
OUT = ROOT / "joint_ath2601_rice_strict399_common6751"
RICE_SPLIT_ROOT = ROOT / "rice_strict399_N4_new80_10_10_OOF_bootstrap"
ATH_ROOT = Path(
    "D:/\u62df\u5357\u82a5/\u6a21\u578b/"
    "ath_three_labelsets_common6751_fixed_core1623_80_10_10"
)
ATH_MODEL_ROOT = ATH_ROOT / "strict2601_common6751"
ATH_MODEL = ATH_MODEL_ROOT / "selected_model_and_manifest.joblib"
ATH_CORE_SPLIT = Path(
    "D:/\u62df\u5357\u82a5/\u6a21\u578b/"
    "strict_vs_inclusive_paired_core1623_fixed80_10_10/"
    "shared_core1623_fixed80_10_10_split.tsv"
)
RANDOM_STATE = 20260622
METHODS = ["meta", "mean", "logit_mean"]


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sn = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "threshold": float(threshold),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "precision": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "youden": float(sn + sp - 1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def select_single_species_threshold(
    y: np.ndarray, probability: np.ndarray
) -> dict:
    candidates = np.unique(
        np.r_[0.0, 1.0, probability, np.nextafter(probability, np.inf)]
    )
    rows = [metrics(y, probability, float(t)) for t in candidates]
    table = pd.DataFrame(rows)
    feasible = table[table["sensitivity"].ge(0.80)]
    if not feasible.empty:
        selected = feasible.sort_values(
            ["specificity", "sensitivity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rule = "SN>=0.80, then max SP, SN, F1"
    else:
        selected = table.sort_values(
            ["sensitivity", "specificity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rule = "fallback max SN, then SP, F1"
    result = selected.to_dict()
    result["selection_rule"] = rule
    return result


def select_joint_threshold(
    rice_y: np.ndarray,
    rice_probability: np.ndarray,
    ath_y: np.ndarray,
    ath_probability: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    candidates = np.unique(
        np.r_[
            0.0,
            1.0,
            rice_probability,
            ath_probability,
            np.nextafter(rice_probability, np.inf),
            np.nextafter(ath_probability, np.inf),
        ]
    )
    rows = []
    for threshold in candidates:
        rice = metrics(rice_y, rice_probability, float(threshold))
        ath = metrics(ath_y, ath_probability, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "rice_sn": rice["sensitivity"],
                "rice_sp": rice["specificity"],
                "rice_f1": rice["f1"],
                "ath_sn": ath["sensitivity"],
                "ath_sp": ath["specificity"],
                "ath_f1": ath["f1"],
                "min_sn": min(rice["sensitivity"], ath["sensitivity"]),
                "min_sp": min(rice["specificity"], ath["specificity"]),
                "mean_sp": np.mean(
                    [rice["specificity"], ath["specificity"]]
                ),
                "mean_sn": np.mean(
                    [rice["sensitivity"], ath["sensitivity"]]
                ),
                "mean_f1": np.mean([rice["f1"], ath["f1"]]),
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[
        table["rice_sn"].ge(0.80) & table["ath_sn"].ge(0.80)
    ]
    if not feasible.empty:
        selected = feasible.sort_values(
            ["min_sp", "mean_sp", "mean_sn", "mean_f1", "threshold"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        rule = (
            "both species validation SN>=0.80; maximize min SP, then mean SP, "
            "mean SN, mean F1"
        )
    else:
        selected = table.sort_values(
            ["min_sn", "min_sp", "mean_sp", "mean_f1", "threshold"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        rule = (
            "fallback: maximize min species SN, then min SP, mean SP, mean F1"
        )
    result = selected.to_dict()
    result["selection_rule"] = rule
    result["both_sn_feasible"] = bool(not feasible.empty)
    return result, table


def index_genes(meta: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    lookup = dict(
        zip(meta["gene_id"].astype(str), np.arange(len(meta), dtype=int))
    )
    missing = [gene for gene in labels["gene_id"] if gene not in lookup]
    if missing:
        raise RuntimeError(
            f"{len(missing)} genes missing from matrix: {missing[:5]}"
        )
    return np.array([lookup[gene] for gene in labels["gene_id"]], dtype=int)


def load_rice() -> tuple[np.ndarray, list[str], pd.DataFrame, dict]:
    variants, meta, _, _, _, _ = rice_features.load_matrices()
    matrix, names, n_bio = variants["common6751"]
    if matrix.shape[1] != 6751 or n_bio != 95:
        raise RuntimeError("Rice common6751 schema mismatch")
    split = {}
    for name in ["train", "validation", "test"]:
        part = pd.read_csv(RICE_SPLIT_ROOT / f"{name}_labels.tsv", sep="\t")
        part["gene_id"] = part["gene_id"].astype(str)
        part["label"] = part["label"].astype(np.int8)
        part["species"] = "rice"
        part["split"] = name
        split[name] = part
    return matrix.astype(np.float32), names, meta, split


def load_ath() -> tuple[np.ndarray, list[str], pd.DataFrame, dict]:
    full, _, genes, full_names, n_bio = ath_features.load_matrix()
    genes = np.array([str(gene).upper() for gene in genes])
    matrix, names, common_n_bio, _ = ath_common.build_common_6751(
        full, full_names, n_bio
    )
    meta = pd.DataFrame({"gene_id": genes})
    core = pd.read_csv(ATH_CORE_SPLIT, sep="\t")
    core["gene_id"] = core["gene_id"].astype(str).str.upper()
    core["label"] = core["label"].astype(np.int8)
    training = pd.read_csv(
        ATH_MODEL_ROOT / "training_labels.tsv", sep="\t"
    )
    training["gene_id"] = training["gene_id"].astype(str).str.upper()
    training["label"] = training["label"].astype(np.int8)
    split = {
        "train": training.copy(),
        "validation": core[core["split"].eq("validation")].copy(),
        "test": core[core["split"].eq("test")].copy(),
    }
    for name, part in split.items():
        part["species"] = "arabidopsis"
        part["split"] = name
        split[name] = part
    if matrix.shape[1] != 6751 or common_n_bio != 95:
        raise RuntimeError("Arabidopsis common6751 schema mismatch")
    return matrix, names, meta, split


def build_weighted_joint_labels(
    rice_train: pd.DataFrame, ath_train: pd.DataFrame
) -> pd.DataFrame:
    rice = rice_train.copy()
    rice["raw_weight"] = 1.0
    ath = ath_train.copy()
    if "training_component" in ath:
        ath["raw_weight"] = np.where(
            ath["training_component"].eq("core1623_train80"), 1.0, 0.5
        )
    else:
        ath["raw_weight"] = 1.0
    rice_scale = 1.0 / rice["raw_weight"].sum()
    ath_scale = 1.0 / ath["raw_weight"].sum()
    rice["sample_weight"] = rice["raw_weight"] * rice_scale
    ath["sample_weight"] = ath["raw_weight"] * ath_scale
    joint = pd.concat([rice, ath], ignore_index=True, sort=False)
    joint["sample_weight"] *= len(joint) / joint["sample_weight"].sum()
    return joint


def save_predictions(
    out: Path,
    labels: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
) -> None:
    frame = labels.copy()
    frame["probability"] = probability
    frame["classification_threshold"] = threshold
    frame["predicted_label"] = (probability >= threshold).astype(np.int8)
    frame.to_csv(out, sep="\t", index=False)


def fit_common_library(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    targets: dict[str, np.ndarray],
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "target_predictions.npz"
    package_path = output / "model.joblib"
    if cache_path.exists() and package_path.exists():
        saved = np.load(cache_path)
        target = {
            target_name: {
                method: saved[f"{target_name}__{method}"]
                for method in METHODS
            }
            for target_name in targets
        }
        print(f"reuse completed library: {name}", flush=True)
        return {}, target
    source, target, folds, meta_model, deployment, meta_names = (
        common.fit_library(
            X_train, y_train, weights, targets, 95
        )
    )
    pd.DataFrame(folds).to_csv(
        output / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    pd.DataFrame({"meta_feature_name": meta_names}).to_csv(
        output / "meta_feature_names.tsv", sep="\t", index=False
    )
    package = {
        "name": name,
        "meta_model": meta_model,
        "deployment_base_models": deployment,
        "meta_feature_names": meta_names,
        "feature_count": 6751,
        "n_bio": 95,
    }
    joblib.dump(package, package_path, compress=3)
    np.savez_compressed(
        cache_path,
        **{
            f"{target_name}__{method}": probability
            for target_name, predictions in target.items()
            for method, probability in predictions.items()
        },
    )
    return source, target


def deploy_existing_ath(
    target: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    warnings.filterwarnings("ignore", category=UserWarning)
    package = joblib.load(ATH_MODEL)
    columns = []
    for path in package["base_model_bundle_paths"]:
        bundle = joblib.load(path)
        imputer = bundle["transforms"][0]
        if not hasattr(imputer, "_fill_dtype") and hasattr(
            imputer, "_fit_dtype"
        ):
            imputer._fill_dtype = imputer._fit_dtype
        transformed = ath_stack.transform_with(
            bundle["transforms"], target, int(bundle["n_bio"])
        )
        by_model = {
            name: model.predict_proba(transformed)[:, 1]
            for name, model in bundle["models"].items()
        }
        by_model["mean_all"] = np.mean(list(by_model.values()), axis=0)
        tree_names = [
            "extra_sqrt",
            "extra_log2",
            "rf_sqrt",
            "lgbm_gbdt",
            "xgb_depth3",
        ]
        by_model["mean_tree"] = np.mean(
            [by_model[name] for name in tree_names], axis=0
        )
        for short_name in [
            "extra_sqrt",
            "extra_log2",
            "rf_sqrt",
            "lgbm_gbdt",
            "xgb_depth3",
            "logistic",
            "mean_all",
            "mean_tree",
        ]:
            columns.append(by_model[short_name])
    meta = np.column_stack(columns).astype(np.float32)
    selected = package["selected_candidate"]
    candidate = selected["candidate"]
    if candidate == "global_mean_all_40_columns":
        probability = meta.mean(axis=1)
    elif candidate == "global_mean_six_base_models":
        names = package["base_prediction_feature_names"]
        idx = [
            i
            for i, value in enumerate(names)
            if value.rsplit("__", 1)[-1]
            in {
                "extra_sqrt",
                "extra_log2",
                "rf_sqrt",
                "lgbm_gbdt",
                "xgb_depth3",
                "logistic",
            }
        ]
        probability = meta[:, idx].mean(axis=1)
    elif candidate == "logistic_stacking_5fold_validation_oof":
        probability = package["stacking_model"].predict_proba(meta)[:, 1]
    else:
        names = package["base_prediction_feature_names"]
        if candidate not in names:
            raise RuntimeError(f"Unsupported ATH selected candidate: {candidate}")
        probability = meta[:, names.index(candidate)]
    return (
        probability.astype(float),
        float(selected["validation_threshold"]),
        str(candidate),
    )


class MultiHeadNetwork(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
        )
        self.general = torch.nn.Linear(96, 1)
        self.rice = torch.nn.Linear(96, 1)
        self.ath = torch.nn.Linear(96, 1)

    def forward(self, x):
        hidden = self.shared(x)
        return (
            self.general(hidden).squeeze(1),
            self.rice(hidden).squeeze(1),
            self.ath(hidden).squeeze(1),
        )


def fit_auxiliary_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    species: np.ndarray,
    weights: np.ndarray,
    targets: dict[str, np.ndarray],
    out: Path,
) -> dict[str, np.ndarray]:
    # Reuse the established supervised PCA/selection preprocessing.
    transformed_train, _, transforms = (
        common.metrics.full.opt.make_fold_features(
            X_train,
            X_train,
            95,
            y_train,
            k=700,
            n_pca_limit=512,
        )
    )
    transformed_targets = {
        key: common.transform_with(transforms, value, 95)
        for key, value in targets.items()
    }
    x = torch.tensor(transformed_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    rice_mask = torch.tensor(species == "rice")
    ath_mask = torch.tensor(species == "arabidopsis")
    predictions = {key: [] for key in targets}
    models = []
    for seed in [20260622, 20260623, 20260624]:
        torch.manual_seed(seed)
        model = MultiHeadNetwork(x.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=8e-4, weight_decay=1e-4
        )
        best_state = None
        best_loss = np.inf
        patience = 0
        for _ in range(180):
            model.train()
            optimizer.zero_grad()
            general, rice_head, ath_head = model(x)
            loss_general = torch.nn.functional.binary_cross_entropy_with_logits(
                general, y, weight=w
            )
            loss_rice = torch.nn.functional.binary_cross_entropy_with_logits(
                rice_head[rice_mask], y[rice_mask], weight=w[rice_mask]
            )
            loss_ath = torch.nn.functional.binary_cross_entropy_with_logits(
                ath_head[ath_mask], y[ath_mask], weight=w[ath_mask]
            )
            loss = loss_general + 0.25 * loss_rice + 0.25 * loss_ath
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_loss - 1e-5:
                best_loss = value
                best_state = {
                    key: tensor.detach().clone()
                    for key, tensor in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            if patience >= 25:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for key, matrix in transformed_targets.items():
                general, _, _ = model(
                    torch.tensor(matrix, dtype=torch.float32)
                )
                predictions[key].append(torch.sigmoid(general).numpy())
        models.append(model.state_dict())
    joblib.dump(
        {"transforms": transforms, "states": models, "input_dim": x.shape[1]},
        out / "auxiliary_species_heads_model.joblib",
        compress=3,
    )
    return {
        key: np.mean(values, axis=0) for key, values in predictions.items()
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rice_X, rice_names, rice_meta, rice_split = load_rice()
    ath_X, ath_names, ath_meta, ath_split = load_ath()
    if rice_names != ath_names:
        mismatch = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(rice_names, ath_names))
            if a != b
        ]
        raise RuntimeError(f"Feature schema mismatch: {mismatch[:5]}")
    pd.DataFrame({"feature_name": rice_names}).to_csv(
        OUT / "common6751_feature_names.tsv", sep="\t", index=False
    )

    rice_indices = {
        key: index_genes(rice_meta, value)
        for key, value in rice_split.items()
    }
    ath_indices = {
        key: index_genes(ath_meta, value)
        for key, value in ath_split.items()
    }
    for species_name, split in [
        ("rice", rice_split),
        ("arabidopsis", ath_split),
    ]:
        pd.concat(split.values(), ignore_index=True, sort=False).to_csv(
            OUT / f"{species_name}_fixed_split_labels.tsv",
            sep="\t",
            index=False,
        )

    targets = {
        "rice_validation": rice_X[rice_indices["validation"]],
        "rice_test": rice_X[rice_indices["test"]],
        "ath_validation": ath_X[ath_indices["validation"]],
        "ath_test": ath_X[ath_indices["test"]],
    }

    # Rice common6751 baseline plus rice-to-Arabidopsis transfer.
    rice_dir = OUT / "rice_strict399_N4_common6751_baseline"
    rice_train_y = rice_split["train"]["label"].to_numpy(np.int8)
    _, rice_predictions = fit_common_library(
        "rice_strict399_N4_common6751",
        rice_X[rice_indices["train"]],
        rice_train_y,
        np.ones(len(rice_train_y), dtype=np.float32),
        targets,
        rice_dir,
    )
    rice_candidates = []
    for method in METHODS:
        val_p = rice_predictions["rice_validation"][method]
        threshold = select_single_species_threshold(
            rice_split["validation"]["label"].to_numpy(np.int8), val_p
        )
        rice_candidates.append(
            {
                "method": method,
                "threshold": threshold["threshold"],
                "validation_auc": roc_auc_score(
                    rice_split["validation"]["label"], val_p
                ),
                "validation_auprc": average_precision_score(
                    rice_split["validation"]["label"], val_p
                ),
            }
        )
    rice_candidate_table = pd.DataFrame(rice_candidates).sort_values(
        ["validation_auc", "validation_auprc"], ascending=False
    )
    rice_method = str(rice_candidate_table.iloc[0]["method"])
    rice_threshold = select_single_species_threshold(
        rice_split["validation"]["label"].to_numpy(np.int8),
        rice_predictions["rice_validation"][rice_method],
    )
    rice_test_metrics = metrics(
        rice_split["test"]["label"].to_numpy(np.int8),
        rice_predictions["rice_test"][rice_method],
        rice_threshold["threshold"],
    )
    save_predictions(
        rice_dir / "fixed_test_predictions.tsv",
        rice_split["test"],
        rice_predictions["rice_test"][rice_method],
        rice_threshold["threshold"],
    )
    rice_candidate_table.to_csv(
        rice_dir / "validation_model_selection.tsv", sep="\t", index=False
    )

    # Existing Arabidopsis strict2601 model directly predicts rice.
    ath_to_rice_probability, ath_existing_threshold, ath_candidate = (
        deploy_existing_ath(targets["rice_test"])
    )
    ath_to_rice_metrics = metrics(
        rice_split["test"]["label"].to_numpy(np.int8),
        ath_to_rice_probability,
        ath_existing_threshold,
    )
    save_predictions(
        OUT / "leave_one_ath2601_to_rice_test_predictions.tsv",
        rice_split["test"],
        ath_to_rice_probability,
        ath_existing_threshold,
    )

    # Rice-only model predicts the fixed Arabidopsis test set.
    rice_to_ath_metrics = metrics(
        ath_split["test"]["label"].to_numpy(np.int8),
        rice_predictions["ath_test"][rice_method],
        rice_threshold["threshold"],
    )
    save_predictions(
        OUT / "leave_one_rice_to_ath_test_predictions.tsv",
        ath_split["test"],
        rice_predictions["ath_test"][rice_method],
        rice_threshold["threshold"],
    )

    # Joint weighted stack.
    joint_labels = build_weighted_joint_labels(
        rice_split["train"], ath_split["train"]
    )
    rice_train_matrix = rice_X[rice_indices["train"]]
    ath_train_matrix = ath_X[ath_indices["train"]]
    joint_X = np.vstack([rice_train_matrix, ath_train_matrix]).astype(np.float32)
    joint_y = joint_labels["label"].to_numpy(np.int8)
    joint_w = joint_labels["sample_weight"].to_numpy(np.float32)
    joint_labels.to_csv(
        OUT / "joint_weighted_training_labels.tsv", sep="\t", index=False
    )
    joint_dir = OUT / "joint_common6751_stacking"
    _, joint_predictions = fit_common_library(
        "joint_ath2601_rice_strict399_common6751",
        joint_X,
        joint_y,
        joint_w,
        targets,
        joint_dir,
    )
    method_rows = []
    method_thresholds = {}
    for method in METHODS:
        threshold, search = select_joint_threshold(
            rice_split["validation"]["label"].to_numpy(np.int8),
            joint_predictions["rice_validation"][method],
            ath_split["validation"]["label"].to_numpy(np.int8),
            joint_predictions["ath_validation"][method],
        )
        search.to_csv(
            joint_dir / f"{method}_joint_threshold_search.tsv",
            sep="\t",
            index=False,
        )
        method_thresholds[method] = threshold
        method_rows.append(
            {
                "method": method,
                "rice_validation_auc": roc_auc_score(
                    rice_split["validation"]["label"],
                    joint_predictions["rice_validation"][method],
                ),
                "ath_validation_auc": roc_auc_score(
                    ath_split["validation"]["label"],
                    joint_predictions["ath_validation"][method],
                ),
                "min_validation_auc": min(
                    roc_auc_score(
                        rice_split["validation"]["label"],
                        joint_predictions["rice_validation"][method],
                    ),
                    roc_auc_score(
                        ath_split["validation"]["label"],
                        joint_predictions["ath_validation"][method],
                    ),
                ),
                "mean_validation_auc": np.mean(
                    [
                        roc_auc_score(
                            rice_split["validation"]["label"],
                            joint_predictions["rice_validation"][method],
                        ),
                        roc_auc_score(
                            ath_split["validation"]["label"],
                            joint_predictions["ath_validation"][method],
                        ),
                    ]
                ),
                "threshold": threshold["threshold"],
                "min_validation_sp": threshold["min_sp"],
            }
        )
    method_table = pd.DataFrame(method_rows).sort_values(
        ["min_validation_auc", "mean_validation_auc", "min_validation_sp"],
        ascending=False,
    )
    method_table.to_csv(
        joint_dir / "joint_validation_method_selection.tsv",
        sep="\t",
        index=False,
    )
    joint_method = str(method_table.iloc[0]["method"])
    joint_threshold = method_thresholds[joint_method]
    joint_rice_test = metrics(
        rice_split["test"]["label"].to_numpy(np.int8),
        joint_predictions["rice_test"][joint_method],
        joint_threshold["threshold"],
    )
    joint_ath_test = metrics(
        ath_split["test"]["label"].to_numpy(np.int8),
        joint_predictions["ath_test"][joint_method],
        joint_threshold["threshold"],
    )
    save_predictions(
        joint_dir / "rice_fixed_test_predictions.tsv",
        rice_split["test"],
        joint_predictions["rice_test"][joint_method],
        joint_threshold["threshold"],
    )
    save_predictions(
        joint_dir / "ath_fixed_test_predictions.tsv",
        ath_split["test"],
        joint_predictions["ath_test"][joint_method],
        joint_threshold["threshold"],
    )

    # Training-only auxiliary species heads; deployment uses the general head.
    head_dir = OUT / "joint_common6751_auxiliary_species_heads"
    head_dir.mkdir(parents=True, exist_ok=True)
    head_predictions = fit_auxiliary_heads(
        joint_X,
        joint_y,
        joint_labels["species"].to_numpy(str),
        joint_w,
        targets,
        head_dir,
    )
    head_threshold, head_search = select_joint_threshold(
        rice_split["validation"]["label"].to_numpy(np.int8),
        head_predictions["rice_validation"],
        ath_split["validation"]["label"].to_numpy(np.int8),
        head_predictions["ath_validation"],
    )
    head_search.to_csv(
        head_dir / "joint_threshold_search.tsv", sep="\t", index=False
    )
    head_rice_test = metrics(
        rice_split["test"]["label"].to_numpy(np.int8),
        head_predictions["rice_test"],
        head_threshold["threshold"],
    )
    head_ath_test = metrics(
        ath_split["test"]["label"].to_numpy(np.int8),
        head_predictions["ath_test"],
        head_threshold["threshold"],
    )
    save_predictions(
        head_dir / "rice_fixed_test_predictions.tsv",
        rice_split["test"],
        head_predictions["rice_test"],
        head_threshold["threshold"],
    )
    save_predictions(
        head_dir / "ath_fixed_test_predictions.tsv",
        ath_split["test"],
        head_predictions["ath_test"],
        head_threshold["threshold"],
    )

    ath_existing_score = pd.read_csv(
        ATH_MODEL_ROOT / "final_shared_validation_test_scores.tsv", sep="\t"
    ).iloc[0].to_dict()
    summary_rows = [
        {
            "model": "rice_strict399_N4_single_species_common6751",
            "evaluation_species": "rice",
            **rice_test_metrics,
        },
        {
            "model": "ath_strict2601_existing_single_species_common6751",
            "evaluation_species": "arabidopsis",
            "auc": float(ath_existing_score["test_auc"]),
            "auprc": float(ath_existing_score["test_auprc"]),
            "threshold": float(ath_existing_score["test_threshold"]),
            "sensitivity": float(ath_existing_score["test_recall"]),
            "specificity": float(ath_existing_score["test_specificity"]),
            "precision": float(ath_existing_score["test_precision"]),
            "f1": float(ath_existing_score["test_f1"]),
            "tp": int(ath_existing_score["test_tp"]),
            "fp": int(ath_existing_score["test_fp"]),
            "tn": int(ath_existing_score["test_tn"]),
            "fn": int(ath_existing_score["test_fn"]),
        },
        {
            "model": "joint_common6751_stacking",
            "evaluation_species": "rice",
            **joint_rice_test,
        },
        {
            "model": "joint_common6751_stacking",
            "evaluation_species": "arabidopsis",
            **joint_ath_test,
        },
        {
            "model": "joint_common6751_auxiliary_species_heads_general_output",
            "evaluation_species": "rice",
            **head_rice_test,
        },
        {
            "model": "joint_common6751_auxiliary_species_heads_general_output",
            "evaluation_species": "arabidopsis",
            **head_ath_test,
        },
        {
            "model": "leave_one_rice_only_to_arabidopsis",
            "evaluation_species": "arabidopsis",
            **rice_to_ath_metrics,
        },
        {
            "model": "leave_one_existing_ath2601_to_rice",
            "evaluation_species": "rice",
            **ath_to_rice_metrics,
        },
    ]
    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(
        OUT / "all_models_fixed_test_comparison.tsv", sep="\t", index=False
    )
    manifest = {
        "feature_definition": "95 shared biological + 6656 PLM = 6751",
        "rice_fixed_split": {
            key: len(value) for key, value in rice_split.items()
        },
        "ath_fixed_split": {
            key: len(value) for key, value in ath_split.items()
        },
        "joint_training": {
            "total": len(joint_labels),
            "rice": int(joint_labels["species"].eq("rice").sum()),
            "arabidopsis": int(
                joint_labels["species"].eq("arabidopsis").sum()
            ),
            "species_total_weight_equalized": True,
            "ath_strict_addition_raw_weight": 0.5,
        },
        "rice_baseline_method": rice_method,
        "rice_baseline_threshold": rice_threshold,
        "joint_method": joint_method,
        "joint_threshold": joint_threshold,
        "auxiliary_heads_threshold": head_threshold,
        "ath_existing_candidate": ath_candidate,
        "ath_existing_threshold": ath_existing_threshold,
        "comparison": summary_rows,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
