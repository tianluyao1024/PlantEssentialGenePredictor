from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

import train_ath_high_confidence_models as base
import train_ath_no_conflict_optimized as opt
import train_pseudo06_predict_all_longest_unknown as prev


SOURCE_ROOT = Path(
    "D:/拟南芥/模型/essential_gene_prediction_3359_fixed10_valthreshold_threshold_retrain"
)
OUT_DIR = Path(
    "D:/拟南芥/模型/unknown14731_train80_teacher_thr065_035_validate3359_fixed50_50"
)
UNKNOWN_PREDICTIONS = SOURCE_ROOT / "unknown_20460_baseline_train80_predictions.tsv"
POSITIVE_PSEUDO_THRESHOLD = 0.65
NEGATIVE_PSEUDO_THRESHOLD = 0.35
RANDOM_STATE = 20260619

CONFIGS = [
    {"name": "seed20260610_pca512_k700", "seed": 20260610, "pca": 512, "k": 700},
    {"name": "seed20260611_pca512_k700", "seed": 20260611, "pca": 512, "k": 700},
    {"name": "seed20260612_pca768_k900", "seed": 20260612, "pca": 768, "k": 900},
    {"name": "seed20260613_pca1024_k1100", "seed": 20260613, "pca": 1024, "k": 1100},
    {"name": "seed20260614_pca384_k550", "seed": 20260614, "pca": 384, "k": 550},
]
BASE_MODEL_NAMES = [
    "extra_sqrt",
    "extra_log2",
    "rf_sqrt",
    "lgbm_gbdt",
    "xgb_depth3",
    "logistic",
]
TREE_MODEL_NAMES = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]


def transform_with(transforms, X: np.ndarray, n_bio: int) -> np.ndarray:
    imp, scaler, pca, selector = transforms
    X_imp = imp.transform(X)
    X_emb = scaler.transform(X_imp[:, n_bio:])
    X_pca = pca.transform(X_emb)
    merged = np.hstack([X_imp[:, :n_bio], X_pca]).astype(np.float32)
    return selector.transform(merged)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> dict:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), probability])
    best = None
    for threshold in candidates:
        prediction = (probability >= threshold).astype(np.int8)
        balanced_accuracy = balanced_accuracy_score(y_true, prediction)
        f1 = f1_score(y_true, prediction, zero_division=0)
        accuracy = accuracy_score(y_true, prediction)
        key = (balanced_accuracy, f1, accuracy)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy),
                "f1": float(f1),
                "accuracy": float(accuracy),
            }
    assert best is not None
    return best


def load_high_quality_3359(genes_all: np.ndarray, X_all: np.ndarray):
    matrix_idx, y, labels = prev.build_train_labels(genes_all)
    labels = labels.reset_index(drop=True)
    X = X_all[matrix_idx].astype(np.float32)
    if len(labels) != 3359 or len(y) != 3359:
        raise RuntimeError(f"Expected 3359 high-quality genes, got labels={len(labels)}, y={len(y)}")
    return X, y.astype(np.int8), labels


def make_or_load_50_50_split(X: np.ndarray, y: np.ndarray, labels: pd.DataFrame):
    path = OUT_DIR / "fixed_stratified_validation_test_50_50_3359.tsv"
    if path.exists():
        split = pd.read_csv(path, sep="\t")
        validation_idx = split.index[split["split"].eq("validation")].to_numpy()
        test_idx = split.index[split["split"].eq("test")].to_numpy()
        return validation_idx, test_idx, split

    validation_idx, test_idx = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=RANDOM_STATE,
        ).split(X, y)
    )
    split = labels.copy()
    split["split"] = ""
    split.loc[validation_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(path, sep="\t", index=False)
    return validation_idx, test_idx, split


def load_unknown_pseudo_training(feature_names: list[str], n_bio: int):
    X_unknown, sequence_ids, genes = prev.build_unknown_matrix(feature_names, n_bio)
    predictions = pd.read_csv(UNKNOWN_PREDICTIONS, sep="\t")
    predictions["gene_id"] = predictions["gene_id"].astype(str).str.upper()
    prediction_map = predictions.drop_duplicates("gene_id").set_index("gene_id")

    missing = [gene for gene in genes if gene not in prediction_map.index]
    if missing:
        raise RuntimeError(f"Missing previous teacher predictions for {len(missing)} unknown genes: {missing[:5]}")

    ordered = prediction_map.loc[genes].reset_index()
    probability = pd.to_numeric(ordered["baseline_mean_probability"], errors="raise").to_numpy(float)
    selected_mask = (probability >= POSITIVE_PSEUDO_THRESHOLD) | (
        probability <= NEGATIVE_PSEUDO_THRESHOLD
    )
    X_unknown = X_unknown[selected_mask]
    sequence_ids = sequence_ids[selected_mask]
    genes = genes[selected_mask]
    probability = probability[selected_mask]
    pseudo_y = (probability >= POSITIVE_PSEUDO_THRESHOLD).astype(np.int8)
    pseudo_table = pd.DataFrame(
        {
            "seq_id": sequence_ids,
            "gene_id": genes,
            "teacher_train80_baseline_mean_probability": probability,
            "positive_pseudo_threshold": POSITIVE_PSEUDO_THRESHOLD,
            "negative_pseudo_threshold": NEGATIVE_PSEUDO_THRESHOLD,
            "pseudo_label": pseudo_y,
            "pseudo_classification": np.where(pseudo_y == 1, "essential", "nonessential"),
            "pseudo_label_rule": (
                f"essential if probability >= {POSITIVE_PSEUDO_THRESHOLD}; "
                f"nonessential if probability <= {NEGATIVE_PSEUDO_THRESHOLD}; "
                "middle probabilities excluded"
            ),
            "teacher_training_labels": (
                "80% training subset of the 3359 genes; prior split was "
                "80% train / 10% validation / 10% fixed test"
            ),
        }
    )
    expected = (14731, 464, 14267)
    observed = (
        len(pseudo_table),
        int((pseudo_y == 1).sum()),
        int((pseudo_y == 0).sum()),
    )
    if observed != expected:
        raise RuntimeError(f"Expected pseudo counts {expected}, got {observed}")
    return X_unknown.astype(np.float32), pseudo_y, pseudo_table


def fit_base_library(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    n_bio: int,
):
    validation_columns = []
    test_columns = []
    column_names = []
    model_paths = []
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    for config in CONFIGS:
        config_name = config["name"]
        model_path = OUT_DIR / f"base_models_{config_name}.joblib"
        prediction_path = OUT_DIR / f"base_predictions_{config_name}.npz"
        names_path = OUT_DIR / f"base_prediction_names_{config_name}.tsv"

        if prediction_path.exists() and names_path.exists():
            saved = np.load(prediction_path)
            config_validation = saved["validation"].astype(np.float32)
            config_test = saved["test"].astype(np.float32)
            config_names = pd.read_csv(names_path, sep="\t")["meta_feature_name"].tolist()
            print(f"reuse predictions: {config_name}", flush=True)
        else:
            seed = int(config["seed"])
            model_defs = opt.make_models(pos_weight, seed)
            X_train_selected, X_validation_selected, transforms = opt.make_fold_features(
                X_train,
                X_validation,
                n_bio,
                y_train,
                k=int(config["k"]),
                n_pca_limit=int(config["pca"]),
            )
            X_test_selected = transform_with(transforms, X_test, n_bio)

            val_by_model = {}
            test_by_model = {}
            fitted_models = {}
            for model_name in BASE_MODEL_NAMES:
                print(f"{config_name}: fitting {model_name}", flush=True)
                model = clone(model_defs[model_name])
                model.fit(X_train_selected, y_train)
                val_by_model[model_name] = model.predict_proba(X_validation_selected)[:, 1]
                test_by_model[model_name] = model.predict_proba(X_test_selected)[:, 1]
                fitted_models[model_name] = model

            val_by_model["mean_all"] = np.mean(
                [val_by_model[name] for name in BASE_MODEL_NAMES], axis=0
            )
            test_by_model["mean_all"] = np.mean(
                [test_by_model[name] for name in BASE_MODEL_NAMES], axis=0
            )
            val_by_model["mean_tree"] = np.mean(
                [val_by_model[name] for name in TREE_MODEL_NAMES], axis=0
            )
            test_by_model["mean_tree"] = np.mean(
                [test_by_model[name] for name in TREE_MODEL_NAMES], axis=0
            )

            short_names = BASE_MODEL_NAMES + ["mean_all", "mean_tree"]
            config_names = [f"{config_name}__{name}" for name in short_names]
            config_validation = np.column_stack([val_by_model[name] for name in short_names]).astype(
                np.float32
            )
            config_test = np.column_stack([test_by_model[name] for name in short_names]).astype(
                np.float32
            )
            np.savez_compressed(
                prediction_path,
                validation=config_validation,
                test=config_test,
            )
            pd.DataFrame({"meta_feature_name": config_names}).to_csv(
                names_path, sep="\t", index=False
            )
            joblib.dump(
                {
                    "config": config,
                    "models": fitted_models,
                    "transforms": transforms,
                    "n_bio": n_bio,
                    "training_label_rule": (
                        "14731 high-confidence unknown pseudo labels selected from prior "
                        "3359 train80 predictions at >=0.65 or <=0.35"
                    ),
                },
                model_path,
                compress=3,
            )
            print(f"saved {model_path}", flush=True)

        validation_columns.append(config_validation)
        test_columns.append(config_test)
        column_names.extend(config_names)
        model_paths.append(str(model_path))

    return (
        np.hstack(validation_columns).astype(np.float32),
        np.hstack(test_columns).astype(np.float32),
        column_names,
        model_paths,
    )


def oof_stacking(
    X_validation_meta: np.ndarray,
    y_validation: np.ndarray,
    X_test_meta: np.ndarray,
):
    c_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    rows = []
    best = None

    for c_value in c_values:
        oof = np.zeros(len(y_validation), dtype=np.float64)
        for train_idx, holdout_idx in folds.split(X_validation_meta, y_validation):
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "quantile",
                        QuantileTransformer(
                            n_quantiles=min(512, len(train_idx)),
                            output_distribution="normal",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c_value,
                            class_weight="balanced",
                            solver="liblinear",
                            max_iter=10000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            model.fit(X_validation_meta[train_idx], y_validation[train_idx])
            oof[holdout_idx] = model.predict_proba(X_validation_meta[holdout_idx])[:, 1]

        threshold_metrics = choose_threshold(y_validation, oof)
        row = {
            "C": c_value,
            "oof_auc": float(roc_auc_score(y_validation, oof)),
            "oof_auprc": float(average_precision_score(y_validation, oof)),
            "selected_threshold": threshold_metrics["threshold"],
            "oof_balanced_accuracy": threshold_metrics["balanced_accuracy"],
            "oof_f1": threshold_metrics["f1"],
        }
        rows.append(row)
        key = (row["oof_auc"], row["oof_auprc"], row["oof_balanced_accuracy"])
        if best is None or key > best["key"]:
            best = {"key": key, "row": row, "oof": oof.copy()}

    assert best is not None
    best_c = float(best["row"]["C"])
    final_model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=min(512, len(y_validation)),
                    output_distribution="normal",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=best_c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    final_model.fit(X_validation_meta, y_validation)
    test_probability = final_model.predict_proba(X_test_meta)[:, 1]
    return (
        best["oof"],
        test_probability,
        final_model,
        best["row"],
        pd.DataFrame(rows),
    )


def evaluate_candidates(
    validation_meta: np.ndarray,
    test_meta: np.ndarray,
    meta_names: list[str],
    y_validation: np.ndarray,
    y_test: np.ndarray,
):
    candidate_rows = []
    predictions = {}

    for column_idx, name in enumerate(meta_names):
        validation_probability = validation_meta[:, column_idx]
        test_probability = test_meta[:, column_idx]
        threshold_metrics = choose_threshold(y_validation, validation_probability)
        val_metrics = binary_metrics(
            y_validation, validation_probability, threshold_metrics["threshold"]
        )
        candidate_rows.append(
            {
                "candidate": name,
                "candidate_type": "base_or_within_config_mean",
                "validation_auc": val_metrics["auc"],
                "validation_auprc": val_metrics["auprc"],
                "validation_threshold": threshold_metrics["threshold"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_f1": val_metrics["f1"],
            }
        )
        predictions[name] = (validation_probability, test_probability)

    blend_definitions = {
        "global_mean_all_40_columns": np.arange(validation_meta.shape[1]),
        "global_mean_six_base_models": np.array(
            [i for i, name in enumerate(meta_names) if name.rsplit("__", 1)[-1] in BASE_MODEL_NAMES]
        ),
        "global_mean_tree_models": np.array(
            [i for i, name in enumerate(meta_names) if name.rsplit("__", 1)[-1] in TREE_MODEL_NAMES]
        ),
        "global_mean_config_mean_all": np.array(
            [i for i, name in enumerate(meta_names) if name.endswith("__mean_all")]
        ),
        "global_mean_config_mean_tree": np.array(
            [i for i, name in enumerate(meta_names) if name.endswith("__mean_tree")]
        ),
    }
    for name, indices in blend_definitions.items():
        validation_probability = validation_meta[:, indices].mean(axis=1)
        test_probability = test_meta[:, indices].mean(axis=1)
        threshold_metrics = choose_threshold(y_validation, validation_probability)
        val_metrics = binary_metrics(
            y_validation, validation_probability, threshold_metrics["threshold"]
        )
        candidate_rows.append(
            {
                "candidate": name,
                "candidate_type": "fixed_blend",
                "validation_auc": val_metrics["auc"],
                "validation_auprc": val_metrics["auprc"],
                "validation_threshold": threshold_metrics["threshold"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_f1": val_metrics["f1"],
            }
        )
        predictions[name] = (validation_probability, test_probability)

    stack_oof, stack_test, stack_model, stack_best, stack_search = oof_stacking(
        validation_meta, y_validation, test_meta
    )
    stack_threshold_metrics = choose_threshold(y_validation, stack_oof)
    stack_val_metrics = binary_metrics(
        y_validation, stack_oof, stack_threshold_metrics["threshold"]
    )
    candidate_rows.append(
        {
            "candidate": "logistic_stacking_5fold_validation_oof",
            "candidate_type": "stacking",
            "validation_auc": stack_val_metrics["auc"],
            "validation_auprc": stack_val_metrics["auprc"],
            "validation_threshold": stack_threshold_metrics["threshold"],
            "validation_balanced_accuracy": stack_val_metrics["balanced_accuracy"],
            "validation_f1": stack_val_metrics["f1"],
        }
    )
    predictions["logistic_stacking_5fold_validation_oof"] = (stack_oof, stack_test)

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["validation_auc", "validation_auprc", "validation_balanced_accuracy"],
        ascending=False,
    )
    selected = candidates.iloc[0].to_dict()
    selected_name = selected["candidate"]
    validation_probability, test_probability = predictions[selected_name]
    selected_threshold = float(selected["validation_threshold"])
    validation_metrics = binary_metrics(y_validation, validation_probability, selected_threshold)
    test_metrics = binary_metrics(y_test, test_probability, selected_threshold)
    return {
        "candidates": candidates,
        "selected": selected,
        "validation_probability": validation_probability,
        "test_probability": test_probability,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "stack_model": stack_model,
        "stack_best": stack_best,
        "stack_search": stack_search,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_all, _sequence_ids_all, genes_all, feature_names, n_bio = base.load_matrix()
    genes_all = np.array([str(g).upper() for g in genes_all])

    X_3359, y_3359, labels_3359 = load_high_quality_3359(genes_all, X_all)
    validation_idx, test_idx, split = make_or_load_50_50_split(X_3359, y_3359, labels_3359)
    X_validation = X_3359[validation_idx]
    y_validation = y_3359[validation_idx]
    X_test = X_3359[test_idx]
    y_test = y_3359[test_idx]

    X_train, y_train, pseudo_training = load_unknown_pseudo_training(feature_names, n_bio)
    pseudo_training.to_csv(
        OUT_DIR / "unknown14731_pseudo_training_labels_train80_teacher_thr065_035.tsv",
        sep="\t",
        index=False,
    )

    validation_meta, test_meta, meta_names, model_paths = fit_base_library(
        X_train,
        y_train,
        X_validation,
        X_test,
        n_bio,
    )
    np.save(OUT_DIR / "validation_base_prediction_matrix.npy", validation_meta)
    np.save(OUT_DIR / "test_base_prediction_matrix.npy", test_meta)
    pd.DataFrame({"meta_feature_name": meta_names}).to_csv(
        OUT_DIR / "base_prediction_feature_names.tsv", sep="\t", index=False
    )

    result = evaluate_candidates(
        validation_meta,
        test_meta,
        meta_names,
        y_validation,
        y_test,
    )
    result["candidates"].to_csv(
        OUT_DIR / "validation_model_selection_candidates.tsv", sep="\t", index=False
    )
    result["stack_search"].to_csv(
        OUT_DIR / "validation_stacking_C_search.tsv", sep="\t", index=False
    )

    validation_predictions = split.loc[validation_idx].copy()
    validation_predictions["true_label"] = y_validation
    validation_predictions["selected_model_probability"] = result["validation_probability"]
    validation_predictions["selected_classification_threshold"] = result["selected"][
        "validation_threshold"
    ]
    validation_predictions["predicted_label"] = (
        validation_predictions["selected_model_probability"]
        >= validation_predictions["selected_classification_threshold"]
    ).astype(int)
    validation_predictions.to_csv(
        OUT_DIR / "validation_predictions_selected_model.tsv", sep="\t", index=False
    )

    test_predictions = split.loc[test_idx].copy()
    test_predictions["true_label"] = y_test
    test_predictions["selected_model_probability"] = result["test_probability"]
    test_predictions["selected_classification_threshold"] = result["selected"][
        "validation_threshold"
    ]
    test_predictions["predicted_label"] = (
        test_predictions["selected_model_probability"]
        >= test_predictions["selected_classification_threshold"]
    ).astype(int)
    test_predictions.to_csv(
        OUT_DIR / "fixed50_test_predictions_selected_model.tsv", sep="\t", index=False
    )

    score_row = {
        "selected_candidate": result["selected"]["candidate"],
        "selected_candidate_type": result["selected"]["candidate_type"],
        "classification_threshold_selected_on_validation": result["selected"][
            "validation_threshold"
        ],
        **{f"validation_{key}": value for key, value in result["validation_metrics"].items()},
        **{f"test_{key}": value for key, value in result["test_metrics"].items()},
    }
    pd.DataFrame([score_row]).to_csv(
        OUT_DIR / "final_validation_and_fixed50_test_scores.tsv", sep="\t", index=False
    )
    joblib.dump(
        {
            "stacking_model_fitted_on_validation": result["stack_model"],
            "stacking_best_configuration": result["stack_best"],
            "selected_candidate": result["selected"],
            "base_prediction_feature_names": meta_names,
            "base_model_bundle_paths": model_paths,
            "feature_names": feature_names,
            "n_bio": n_bio,
        },
        OUT_DIR / "selected_ensemble_and_stacking_model.joblib",
        compress=3,
    )

    manifest = {
        "experiment": (
            "train only on 14731 high-confidence unknown pseudo labels selected at "
            "0.65/0.35; validate/test only on 3359 high-quality genes"
        ),
        "teacher_for_unknown": (
            "model fitted on the fixed 80% training subset of the 3359 genes"
        ),
        "teacher_probability_used": "baseline_mean_probability from the prior train80 teacher",
        "unknown_pseudo_rule": (
            "essential if probability >= 0.65; nonessential if probability <= 0.35; "
            "genes in (0.35, 0.65) excluded"
        ),
        "n_pseudo_train": int(len(y_train)),
        "pseudo_train_essential": int((y_train == 1).sum()),
        "pseudo_train_nonessential": int((y_train == 0).sum()),
        "n_high_quality_total": int(len(y_3359)),
        "n_validation": int(len(y_validation)),
        "validation_essential": int((y_validation == 1).sum()),
        "validation_nonessential": int((y_validation == 0).sum()),
        "n_fixed_test": int(len(y_test)),
        "test_essential": int((y_test == 1).sum()),
        "test_nonessential": int((y_test == 0).sum()),
        "split_random_state": RANDOM_STATE,
        "feature_count": int(len(feature_names)),
        "bio_feature_count": int(n_bio),
        "plm_feature_count": int(len(feature_names) - n_bio),
        "selected_candidate": result["selected"],
        "validation_metrics": result["validation_metrics"],
        "fixed_test_metrics": result["test_metrics"],
        "base_model_bundle_paths": model_paths,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
