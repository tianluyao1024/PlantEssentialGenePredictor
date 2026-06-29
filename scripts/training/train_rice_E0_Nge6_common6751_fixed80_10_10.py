from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

import train_rice_internal_enhanced_replacement_v2 as features
import train_rice_rap_es_robust_lite_models as robust
import train_rice_species_specific_strict_highconf_repeated as metrics


ROOT = Path("E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models")
LABEL_FILE = (
    ROOT
    / "rice_v2_high_quality_label_tier_comparison"
    / "labels"
    / "E_hq_plus_N_E0_Nge3_labels.tsv"
)
OUT_DIR = ROOT / "rice_E_hq_plus_N_E0_Nge6_common6751_fixed80_10_10"
RANDOM_STATE = 20260619
DROP_FEATURES = {
    "rice_homolog_found",
    "rice_homolog_percent_identity",
    "homolog_not_found_in_rice",
}


def transform_with(transforms, X: np.ndarray, n_bio: int) -> np.ndarray:
    return metrics.transform_with(transforms, X, n_bio)


def build_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABEL_FILE, sep="\t")
    labels["rap_gene_id"] = labels["rap_gene_id"].astype(str)
    labels["label"] = pd.to_numeric(labels["final_label"], errors="raise").astype(np.int8)
    e = pd.to_numeric(labels["essential_evidence_count_E"], errors="coerce")
    n = pd.to_numeric(labels["nonessential_evidence_count_N"], errors="coerce")
    essential = labels[labels["label"].eq(1)].copy()
    nonessential = labels[labels["label"].eq(0) & e.fillna(-1).eq(0) & n.fillna(-1).ge(6)].copy()
    selected = pd.concat([essential, nonessential], ignore_index=True)
    selected["label_rule"] = np.where(
        selected["label"].eq(1),
        "HQ essential: Oryzabase_trait_gene plus Tos17 ES>0.9 and E>=2",
        "HQ nonessential: Tos17 E=0 and N>=6 after all-source essential-conflict removal",
    )
    if selected["rap_gene_id"].duplicated().any():
        raise RuntimeError("Duplicated RAP genes in selected labels")
    if (len(selected), int(selected["label"].sum())) != (1048, 592):
        raise RuntimeError(
            f"Expected raw labels 1048 total / 592 essential, got "
            f"{len(selected)} / {int(selected['label'].sum())}"
        )
    return selected


def load_common6751():
    X, meta, names, n_bio, coverage, status = features.load_feature_matrix()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = sorted(DROP_FEATURES - set(name_to_idx))
    if missing:
        raise RuntimeError(f"Directional features missing from rice matrix: {missing}")
    keep_names = [name for name in names if name not in DROP_FEATURES]
    keep_idx = [name_to_idx[name] for name in keep_names]
    X = X[:, keep_idx].astype(np.float32)
    new_n_bio = n_bio - len(DROP_FEATURES)
    if (new_n_bio, X.shape[1] - new_n_bio, X.shape[1]) != (95, 6656, 6751):
        raise RuntimeError(
            f"Expected 95+6656=6751, got {new_n_bio}+{X.shape[1]-new_n_bio}={X.shape[1]}"
        )
    return X, meta.copy(), keep_names, new_n_bio, coverage, status


def fit_library(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    targets: dict[str, np.ndarray],
    n_bio: int,
):
    oof_columns = []
    target_columns = {name: [] for name in targets}
    column_names = []
    deployment_base_models = []
    fold_rows = []

    for config in robust.CONFIGS:
        seed = int(config["seed"])
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pos_weight = float(
            w_train[y_train == 0].sum() / max(1e-6, w_train[y_train == 1].sum())
        )
        model_defs = robust.make_models(pos_weight, seed)
        oof = {name: np.zeros(len(y_train), dtype=np.float32) for name in model_defs}
        oof["mean_all"] = np.zeros(len(y_train), dtype=np.float32)
        target_folds = {
            target: {name: [] for name in robust.PRED_COLS} for target in targets
        }

        for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train), 1):
            Xtr, Xval, transforms = metrics.full.opt.make_fold_features(
                X_train[train_idx],
                X_train[val_idx],
                n_bio,
                y_train[train_idx],
                k=int(config["k"]),
                n_pca_limit=int(config["pca"]),
            )
            transformed_targets = {
                name: transform_with(transforms, matrix, n_bio)
                for name, matrix in targets.items()
            }
            val_predictions = {}
            target_predictions = {name: {} for name in targets}
            for model_name, model_def in model_defs.items():
                model = clone(model_def)
                robust.fit_with_optional_weight(
                    model,
                    Xtr,
                    y_train[train_idx],
                    w_train[train_idx],
                )
                val_probability = model.predict_proba(Xval)[:, 1]
                oof[model_name][val_idx] = val_probability
                val_predictions[model_name] = val_probability
                for target_name, target_matrix in transformed_targets.items():
                    target_predictions[target_name][model_name] = model.predict_proba(
                        target_matrix
                    )[:, 1]
                    target_folds[target_name][model_name].append(
                        target_predictions[target_name][model_name]
                    )
            oof["mean_all"][val_idx] = np.mean(
                [val_predictions[name] for name in model_defs], axis=0
            )
            for target_name in targets:
                target_folds[target_name]["mean_all"].append(
                    np.mean(
                        [
                            target_predictions[target_name][name]
                            for name in model_defs
                        ],
                        axis=0,
                    )
                )
            fold_rows.append(
                {
                    "config": config["name"],
                    "fold": fold,
                    "mean_all_auc": float(
                        metrics.roc_auc_score(
                            y_train[val_idx], oof["mean_all"][val_idx]
                        )
                    ),
                }
            )
            print(
                f"{config['name']} fold {fold}: "
                f"mean_all_auc={fold_rows[-1]['mean_all_auc']:.4f}",
                flush=True,
            )

        for model_name in robust.PRED_COLS:
            oof_columns.append(oof[model_name])
            column_names.append(f"{config['name']}__{model_name}")
            for target_name in targets:
                target_columns[target_name].append(
                    np.mean(target_folds[target_name][model_name], axis=0)
                )

        Xfit, _, transforms = metrics.full.opt.make_fold_features(
            X_train,
            X_train,
            n_bio,
            y_train,
            k=int(config["k"]),
            n_pca_limit=int(config["pca"]),
        )
        fitted = {}
        for model_name, model_def in model_defs.items():
            model = clone(model_def)
            robust.fit_with_optional_weight(model, Xfit, y_train, w_train)
            fitted[model_name] = model
        deployment_base_models.append(
            {"config": config, "transforms": transforms, "models": fitted}
        )

    X_oof = np.column_stack(oof_columns).astype(np.float32)
    X_targets = {
        name: np.column_stack(columns).astype(np.float32)
        for name, columns in target_columns.items()
    }
    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=min(512, len(y_train)),
                    output_distribution="normal",
                    random_state=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.02,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=7,
                ),
            ),
        ]
    )
    meta.fit(X_oof, y_train, classifier__sample_weight=w_train)
    source_predictions = {
        "meta": meta.predict_proba(X_oof)[:, 1],
        "mean": X_oof.mean(axis=1),
        "logit_mean": robust.logit_mean(X_oof),
    }
    target_predictions = {}
    for target_name, matrix in X_targets.items():
        target_predictions[target_name] = {
            "meta": meta.predict_proba(matrix)[:, 1],
            "mean": matrix.mean(axis=1),
            "logit_mean": robust.logit_mean(matrix),
        }
    return (
        source_predictions,
        target_predictions,
        fold_rows,
        meta,
        deployment_base_models,
        column_names,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = build_labels()
    X_all, meta_all, feature_names, n_bio, coverage, status = load_common6751()
    meta_all["matrix_row"] = np.arange(len(meta_all), dtype=int)
    labeled = meta_all.merge(
        labels,
        left_on="gene_id",
        right_on="rap_gene_id",
        how="inner",
    ).reset_index(drop=True)
    X = X_all[labeled["matrix_row"].to_numpy(int)].astype(np.float32)
    y = labeled["label"].to_numpy(np.int8)
    if (len(y), int(y.sum()), int((y == 0).sum())) != (1014, 587, 427):
        raise RuntimeError(
            f"Expected feature intersection 1014/587/427, got "
            f"{len(y)}/{int(y.sum())}/{int((y==0).sum())}"
        )

    weights = np.ones(len(y), dtype=np.float32)
    train_idx, val_idx, test_idx = metrics.split_80_10_10(y, RANDOM_STATE)
    split = labeled[
        [
            "gene_id",
            "transcript_id",
            "rap_gene_id",
            "label",
            "sources",
            "essential_evidence_count_E",
            "nonessential_evidence_count_N",
            "label_rule",
        ]
    ].copy()
    split["split"] = ""
    split.loc[train_idx, "split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(OUT_DIR / "fixed80_10_10_split_labels.tsv", sep="\t", index=False)
    labels.to_csv(OUT_DIR / "raw_E_hq_plus_N_E0_Nge6_labels_1048.tsv", sep="\t", index=False)
    labeled.to_csv(OUT_DIR / "labeled_feature_intersection_1014.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        OUT_DIR / "common6751_feature_names.tsv", sep="\t", index=False
    )

    (
        oof_predictions,
        target_predictions,
        fold_rows,
        meta_model,
        deployment_models,
        meta_feature_names,
    ) = fit_library(
        X[train_idx],
        y[train_idx],
        weights[train_idx],
        {"validation": X[val_idx], "test": X[test_idx]},
        n_bio,
    )

    candidate_rows = []
    for model_name in ["meta", "mean", "logit_mean"]:
        val_probability = target_predictions["validation"][model_name]
        test_probability = target_predictions["test"][model_name]
        threshold = metrics.best_threshold(y[val_idx], val_probability)
        val_metrics = metrics.binary_metrics(
            y[val_idx], val_probability, threshold["threshold"]
        )
        test_metrics = metrics.binary_metrics(
            y[test_idx], test_probability, threshold["threshold"]
        )
        candidate_rows.append(
            {
                "model": model_name,
                "validation_threshold": threshold["threshold"],
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["validation_auc", "validation_auprc"], ascending=False
    )
    candidates.to_csv(OUT_DIR / "validation_model_selection.tsv", sep="\t", index=False)
    best = candidates.iloc[0].to_dict()
    best_name = str(best["model"])
    best_threshold = float(best["validation_threshold"])

    validation_predictions = split.loc[val_idx].copy()
    validation_predictions["probability"] = target_predictions["validation"][best_name]
    validation_predictions["threshold"] = best_threshold
    validation_predictions["predicted_label"] = (
        validation_predictions["probability"] >= best_threshold
    ).astype(np.int8)
    validation_predictions.to_csv(
        OUT_DIR / "validation_predictions.tsv", sep="\t", index=False
    )
    test_predictions = split.loc[test_idx].copy()
    test_predictions["probability"] = target_predictions["test"][best_name]
    test_predictions["threshold"] = best_threshold
    test_predictions["predicted_label"] = (
        test_predictions["probability"] >= best_threshold
    ).astype(np.int8)
    test_predictions.to_csv(OUT_DIR / "fixed_test_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "inner_oof_fold_scores.tsv", sep="\t", index=False)
    pd.DataFrame({"meta_feature_name": meta_feature_names}).to_csv(
        OUT_DIR / "meta_feature_names.tsv", sep="\t", index=False
    )
    joblib.dump(
        {
            "selected_prediction_method": best_name,
            "classification_threshold": best_threshold,
            "meta_model": meta_model,
            "deployment_base_models": deployment_models,
            "meta_feature_names": meta_feature_names,
            "feature_names": feature_names,
            "n_bio": n_bio,
            "label_rule": "HQ essential plus Tos17 E=0,N>=6 nonessential",
        },
        OUT_DIR / "final_rice_E0_Nge6_common6751_model.joblib",
        compress=3,
    )

    manifest = {
        "raw_label_file": str(LABEL_FILE),
        "raw_selected_labels": 1048,
        "raw_essential": 592,
        "raw_nonessential_E0_Nge6": 456,
        "feature_intersection": 1014,
        "feature_intersection_essential": 587,
        "feature_intersection_nonessential": 427,
        "split_random_state": RANDOM_STATE,
        "train_n": int(len(train_idx)),
        "train_essential": int(y[train_idx].sum()),
        "train_nonessential": int((y[train_idx] == 0).sum()),
        "validation_n": int(len(val_idx)),
        "validation_essential": int(y[val_idx].sum()),
        "validation_nonessential": int((y[val_idx] == 0).sum()),
        "test_n": int(len(test_idx)),
        "test_essential": int(y[test_idx].sum()),
        "test_nonessential": int((y[test_idx] == 0).sum()),
        "feature_count": 6751,
        "bio_feature_count": 95,
        "plm_feature_count": 6656,
        "removed_features": sorted(DROP_FEATURES),
        "selected_model": best,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
