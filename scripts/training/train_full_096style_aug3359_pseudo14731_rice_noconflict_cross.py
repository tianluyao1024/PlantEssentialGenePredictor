from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

sys.path.insert(0, str(Path(__file__).parent))
import train_ath_high_confidence_models as ath_base
import train_ath_no_conflict_optimized as opt
import train_pseudo06_predict_all_longest_unknown as ath_prev
import train_096style_consensus1623_vs_ricedata as rice_096


ATH_MODEL_ROOT = Path("D:/\u62df\u5357\u82a5/\u6a21\u578b")
ATH_FIXED10_ROOT = ATH_MODEL_ROOT / "essential_gene_prediction_3359_fixed10_valthreshold_threshold_retrain"
ATH_UNKNOWN_065 = ATH_FIXED10_ROOT / "unknown_selected_for_retrain_thr_0.65_0.35.tsv"
RICE_LABELS = (
    Path.home()
    / "Desktop"
    / "\u6c34\u7a3b"
    / "rice_mutant_sources"
    / "processed"
    / "rice_multi_source_no_conflict_gene_essentiality.csv"
)
OUT_DIR = (
    Path.home()
    / "Desktop"
    / "\u6c34\u7a3b"
    / "cross_species_ath_rice_common_features_models"
    / "096style_aug3359_pseudo14731_rice_noconflict_cross"
)

RANDOM_STATE = 20260613
CONFIGS = rice_096.CONFIGS
PRED_COLS = rice_096.PRED_COLS


def split_80_10_10(y: np.ndarray, random_state: int):
    all_idx = np.arange(len(y))
    trainval_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=random_state).split(all_idx, y)
    )
    train_local, val_local = next(
        StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=random_state + 1).split(
            trainval_idx, y[trainval_idx]
        )
    )
    return trainval_idx[train_local], trainval_idx[val_local], test_idx


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float):
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def best_validation_threshold(y_true: np.ndarray, prob: np.ndarray):
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), prob])
    best = None
    for threshold in candidates:
        metrics = binary_metrics(y_true, prob, float(threshold))
        row = {"threshold": float(threshold), **metrics}
        if best is None or (row["balanced_accuracy"], row["f1"], row["auc"]) > (
            best["balanced_accuracy"],
            best["f1"],
            best["auc"],
        ):
            best = row
    return best


def logit_mean(x: np.ndarray):
    p = np.clip(x, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def transform_with(transforms, X: np.ndarray, n_bio: int):
    imp, scaler, pca, selector = transforms
    x_imp = imp.transform(X)
    x_emb = scaler.transform(x_imp[:, n_bio:])
    x_pca = pca.transform(x_emb)
    x_fold = np.hstack([x_imp[:, :n_bio], x_pca]).astype(np.float32)
    return selector.transform(x_fold)


def fit_cv_stack_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    targets: dict[str, np.ndarray],
    n_bio: int,
    tag: str,
):
    oof_cols = []
    target_cols = {name: [] for name in targets}
    meta_names = []
    fold_rows = []
    fold_models = []

    for config in CONFIGS:
        seed = int(config["seed"])
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pos_weight = float((y_train == 0).sum() / max(1, y_train.sum()))
        model_defs = opt.make_models(pos_weight, seed)
        oof = {name: np.zeros(len(y_train), dtype=np.float32) for name in model_defs}
        oof["mean_all"] = np.zeros(len(y_train), dtype=np.float32)
        oof["mean_tree"] = np.zeros(len(y_train), dtype=np.float32)
        target_fold = {name: {model_name: [] for model_name in PRED_COLS} for name in targets}

        for fold, (tr, va) in enumerate(folds.split(X_train, y_train), 1):
            Xtr, Xva, transforms = opt.make_fold_features(
                X_train[tr],
                X_train[va],
                n_bio,
                y_train[tr],
                k=int(config["k"]),
                n_pca_limit=int(config["pca"]),
            )
            transformed_targets = {
                target_name: transform_with(transforms, X_target, n_bio)
                for target_name, X_target in targets.items()
            }
            val_preds = {}
            target_preds = {name: {} for name in targets}
            for model_name, model_def in model_defs.items():
                model = clone(model_def)
                model.fit(Xtr, y_train[tr])
                val_prob = model.predict_proba(Xva)[:, 1]
                oof[model_name][va] = val_prob
                val_preds[model_name] = val_prob
                for target_name, Xt in transformed_targets.items():
                    target_prob = model.predict_proba(Xt)[:, 1]
                    target_preds[target_name][model_name] = target_prob
                    target_fold[target_name][model_name].append(target_prob)
                fold_models.append(
                    {
                        "tag": tag,
                        "config": config,
                        "fold": fold,
                        "model_name": model_name,
                        "model": model,
                        "transforms": transforms,
                    }
                )

            tree_names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
            oof["mean_all"][va] = np.mean([val_preds[n] for n in model_defs], axis=0)
            oof["mean_tree"][va] = np.mean([val_preds[n] for n in tree_names], axis=0)
            for target_name in targets:
                target_fold[target_name]["mean_all"].append(
                    np.mean([target_preds[target_name][n] for n in model_defs], axis=0)
                )
                target_fold[target_name]["mean_tree"].append(
                    np.mean([target_preds[target_name][n] for n in tree_names], axis=0)
                )
            fold_auc = roc_auc_score(y_train[va], oof["mean_all"][va])
            fold_rows.append({"tag": tag, "config": config["name"], "fold": fold, "mean_all_auc": float(fold_auc)})
            print(f"{tag} {config['name']} fold {fold}: mean_all_auc={fold_auc:.4f}", flush=True)

        for model_name in PRED_COLS:
            oof_cols.append(oof[model_name])
            meta_names.append(f"{config['name']}__{model_name}")
            for target_name in targets:
                target_cols[target_name].append(np.mean(target_fold[target_name][model_name], axis=0))

    X_oof = np.vstack(oof_cols).T.astype(np.float32)
    X_targets = {name: np.vstack(cols).T.astype(np.float32) for name, cols in target_cols.items()}
    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "qt",
                QuantileTransformer(
                    n_quantiles=min(512, len(y_train)),
                    output_distribution="normal",
                    random_state=1,
                ),
            ),
            (
                "clf",
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
    meta.fit(X_oof, y_train)
    source_preds = {
        "meta": meta.predict_proba(X_oof)[:, 1],
        "mean": X_oof.mean(axis=1),
        "logit_mean": logit_mean(X_oof),
    }
    target_preds = {}
    for name, X_meta in X_targets.items():
        target_preds[name] = {
            "meta": meta.predict_proba(X_meta)[:, 1],
            "mean": X_meta.mean(axis=1),
            "logit_mean": logit_mean(X_meta),
        }
    return source_preds, target_preds, meta_names, fold_rows, {"meta": meta, "fold_models": fold_models}


def load_ath_data():
    X_all, ids_all, genes_all, feature_names, n_bio = ath_base.load_matrix()
    genes_all = np.array([str(g).upper() for g in genes_all])
    ids_all = np.array([str(i) for i in ids_all])

    label_idx, y_3359, labels_3359 = ath_prev.build_train_labels(genes_all)
    X_3359 = X_all[label_idx].astype(np.float32)
    labels_3359 = labels_3359.reset_index(drop=True).copy()
    labels_3359["seq_id"] = ids_all[label_idx]
    labels_3359["gene_id"] = labels_3359["gene_id"].astype(str).str.upper()
    labels_3359["label"] = labels_3359["label"].astype(int)

    X_unknown, ids_unknown, genes_unknown = ath_prev.build_unknown_matrix(feature_names, n_bio)
    genes_unknown = np.array([str(g).upper() for g in genes_unknown])
    ids_unknown = np.array([str(i) for i in ids_unknown])
    pseudo = pd.read_csv(ATH_UNKNOWN_065, sep="\t")
    pseudo["gene_id"] = pseudo["gene_id"].astype(str).str.upper()
    pseudo["pseudo_label"] = pseudo["pseudo_label"].astype(int)
    unknown_lookup = {g: i for i, g in enumerate(genes_unknown)}
    pseudo = pseudo[pseudo["gene_id"].isin(unknown_lookup)].drop_duplicates("gene_id").reset_index(drop=True)
    pseudo_idx = np.array([unknown_lookup[g] for g in pseudo["gene_id"]], dtype=int)
    pseudo_meta = pd.DataFrame(
        {
            "seq_id": ids_unknown[pseudo_idx],
            "gene_id": genes_unknown[pseudo_idx],
            "label": pseudo["pseudo_label"].to_numpy(dtype=np.int8),
            "classification": np.where(pseudo["pseudo_label"].to_numpy(dtype=np.int8) == 1, "essential", "nonessential"),
            "label_source": "unknown_20460_pseudo_thr_0.65_0.35",
            "baseline_mean_probability": pseudo["baseline_mean_probability"].to_numpy(),
        }
    )
    return X_3359, y_3359.astype(np.int8), labels_3359, X_unknown[pseudo_idx].astype(np.float32), pseudo_meta, feature_names, n_bio


def load_rice_noconflict(feature_names, n_bio):
    X_rice_all, rice_meta_all, _bio_names = rice_096.load_rice_matrix(feature_names, n_bio)
    labels = pd.read_csv(RICE_LABELS)
    labels = labels.rename(columns={"final_label": "label"})
    labels["gene_id"] = labels["gene_id"].astype(str)
    labels["label"] = labels["label"].astype(int)
    rice_meta_all = rice_meta_all.copy()
    rice_meta_all["matrix_row"] = np.arange(len(rice_meta_all), dtype=int)
    merged = rice_meta_all.merge(
        labels[
            [
                "gene_id",
                "label",
                "final_classification",
                "source_count",
                "sources",
                "essential_source_count",
                "nonessential_source_count",
            ]
        ],
        on="gene_id",
        how="inner",
    )
    idx = merged["matrix_row"].to_numpy(dtype=int)
    return X_rice_all[idx].astype(np.float32), merged.drop(columns=["matrix_row"]).reset_index(drop=True)


def write_split(path: Path, meta: pd.DataFrame, train_idx, val_idx, test_idx):
    split = meta.copy()
    split["split"] = ""
    split.loc[train_idx, "split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(path, sep="\t", index=False)
    return split


def score_split(prefix: str, y_train, source_preds, target_preds, y_val, y_test):
    rows = []
    val_best = {}
    for model_name in ["meta", "mean", "logit_mean"]:
        val_prob = target_preds["validation"][model_name]
        test_prob = target_preds["test"][model_name]
        best_thr = best_validation_threshold(y_val, val_prob)
        threshold = best_thr["threshold"]
        val_metrics = binary_metrics(y_val, val_prob, threshold)
        test_metrics = binary_metrics(y_test, test_prob, threshold)
        rows.append(
            {
                "dataset": prefix,
                "model": model_name,
                "train_n": int(len(y_train)),
                "train_positive": int(y_train.sum()),
                "train_oof_auc": float(roc_auc_score(y_train, source_preds[model_name])),
                "train_oof_auprc": float(average_precision_score(y_train, source_preds[model_name])),
                "validation_threshold": float(threshold),
                "validation_auc": val_metrics["auc"],
                "validation_auprc": val_metrics["auprc"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_f1": val_metrics["f1"],
                "test_auc": test_metrics["auc"],
                "test_auprc": test_metrics["auprc"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_specificity": test_metrics["specificity"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],
            }
        )
        val_best[model_name] = threshold
    return pd.DataFrame(rows), val_best


def write_prediction_table(path: Path, meta: pd.DataFrame, pred_dict: dict[str, np.ndarray], label_col: bool = True):
    out = meta.copy()
    if label_col and "label" in out.columns:
        out["label"] = out["label"].astype(int)
    out["meta_probability"] = pred_dict["meta"]
    out["mean_probability"] = pred_dict["mean"]
    out["logit_mean_probability"] = pred_dict["logit_mean"]
    out.to_csv(path, sep="\t", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_3359, y_3359, ath3359_meta, X_ath_pseudo, ath_pseudo_meta, feature_names, n_bio = load_ath_data()
    X_rice, rice_meta = load_rice_noconflict(feature_names, n_bio)
    y_rice = rice_meta["label"].to_numpy(dtype=np.int8)

    ath_train_idx, ath_val_idx, ath_test_idx = split_80_10_10(y_3359, RANDOM_STATE)
    rice_train_idx, rice_val_idx, rice_test_idx = split_80_10_10(y_rice, RANDOM_STATE)

    ath_split = write_split(OUT_DIR / "arabidopsis_3359_train_validation_test_split.tsv", ath3359_meta, ath_train_idx, ath_val_idx, ath_test_idx)
    rice_split = write_split(OUT_DIR / "rice_noconflict_train_validation_test_split.tsv", rice_meta, rice_train_idx, rice_val_idx, rice_test_idx)

    X_ath_train = np.vstack([X_3359[ath_train_idx], X_ath_pseudo]).astype(np.float32)
    y_ath_train = np.concatenate([y_3359[ath_train_idx], ath_pseudo_meta["label"].to_numpy(dtype=np.int8)])
    ath_train_meta = pd.concat(
        [ath3359_meta.iloc[ath_train_idx].copy(), ath_pseudo_meta.copy()],
        ignore_index=True,
    )
    ath_train_meta.to_csv(OUT_DIR / "arabidopsis_training_80pct3359_plus_pseudo14731.tsv", sep="\t", index=False)

    X_rice_train = X_rice[rice_train_idx]
    y_rice_train = y_rice[rice_train_idx]

    ath_targets = {
        "validation": X_3359[ath_val_idx],
        "test": X_3359[ath_test_idx],
        "cross_rice_all_noconflict": X_rice,
    }
    rice_targets = {
        "validation": X_rice[rice_val_idx],
        "test": X_rice[rice_test_idx],
        "cross_ath3359": X_3359,
    }

    ath_source_preds, ath_target_preds, ath_meta_names, ath_fold_rows, ath_model = fit_cv_stack_predict(
        X_ath_train,
        y_ath_train,
        ath_targets,
        n_bio,
        "arabidopsis_80pct3359_plus_pseudo14731",
    )
    rice_source_preds, rice_target_preds, rice_meta_names, rice_fold_rows, rice_model = fit_cv_stack_predict(
        X_rice_train,
        y_rice_train,
        rice_targets,
        n_bio,
        "rice_noconflict_80pct",
    )

    ath_scores, ath_thresholds = score_split(
        "arabidopsis_augmented",
        y_ath_train,
        ath_source_preds,
        ath_target_preds,
        y_3359[ath_val_idx],
        y_3359[ath_test_idx],
    )
    rice_scores, rice_thresholds = score_split(
        "rice_noconflict",
        y_rice_train,
        rice_source_preds,
        rice_target_preds,
        y_rice[rice_val_idx],
        y_rice[rice_test_idx],
    )

    cross_rows = []
    for model_name in ["meta", "mean", "logit_mean"]:
        rice_prob = ath_target_preds["cross_rice_all_noconflict"][model_name]
        ath_prob = rice_target_preds["cross_ath3359"][model_name]
        cross_rows.append(
            {
                "direction": "arabidopsis_augmented_train_predict_rice_all_noconflict",
                "model": model_name,
                "target_n": int(len(y_rice)),
                "target_positive": int(y_rice.sum()),
                "target_auc": float(roc_auc_score(y_rice, rice_prob)),
                "target_auprc": float(average_precision_score(y_rice, rice_prob)),
            }
        )
        cross_rows.append(
            {
                "direction": "rice_noconflict_train_predict_arabidopsis_3359",
                "model": model_name,
                "target_n": int(len(y_3359)),
                "target_positive": int(y_3359.sum()),
                "target_auc": float(roc_auc_score(y_3359, ath_prob)),
                "target_auprc": float(average_precision_score(y_3359, ath_prob)),
            }
        )

    ath_scores.to_csv(OUT_DIR / "arabidopsis_internal_validation_test_scores.tsv", sep="\t", index=False)
    rice_scores.to_csv(OUT_DIR / "rice_internal_validation_test_scores.tsv", sep="\t", index=False)
    pd.DataFrame(cross_rows).to_csv(OUT_DIR / "cross_species_prediction_scores.tsv", sep="\t", index=False)
    pd.DataFrame(ath_fold_rows).to_csv(OUT_DIR / "arabidopsis_training_oof_fold_scores.tsv", sep="\t", index=False)
    pd.DataFrame(rice_fold_rows).to_csv(OUT_DIR / "rice_training_oof_fold_scores.tsv", sep="\t", index=False)
    pd.DataFrame({"meta_feature_name": ath_meta_names}).to_csv(OUT_DIR / "arabidopsis_meta_feature_names.tsv", sep="\t", index=False)
    pd.DataFrame({"meta_feature_name": rice_meta_names}).to_csv(OUT_DIR / "rice_meta_feature_names.tsv", sep="\t", index=False)

    write_prediction_table(
        OUT_DIR / "arabidopsis_validation_predictions.tsv",
        ath_split.iloc[ath_val_idx].reset_index(drop=True),
        ath_target_preds["validation"],
    )
    write_prediction_table(
        OUT_DIR / "arabidopsis_test_predictions.tsv",
        ath_split.iloc[ath_test_idx].reset_index(drop=True),
        ath_target_preds["test"],
    )
    write_prediction_table(
        OUT_DIR / "rice_validation_predictions.tsv",
        rice_split.iloc[rice_val_idx].reset_index(drop=True),
        rice_target_preds["validation"],
    )
    write_prediction_table(
        OUT_DIR / "rice_test_predictions.tsv",
        rice_split.iloc[rice_test_idx].reset_index(drop=True),
        rice_target_preds["test"],
    )
    write_prediction_table(
        OUT_DIR / "arabidopsis_model_predictions_on_rice_all_noconflict.tsv",
        rice_meta,
        ath_target_preds["cross_rice_all_noconflict"],
    )
    write_prediction_table(
        OUT_DIR / "rice_model_predictions_on_arabidopsis_3359.tsv",
        ath3359_meta,
        rice_target_preds["cross_ath3359"],
    )

    joblib.dump(
        {
            "feature_names": feature_names,
            "n_bio": n_bio,
            "configs": CONFIGS,
            "arabidopsis_model": ath_model,
            "rice_model": rice_model,
            "arabidopsis_validation_thresholds": ath_thresholds,
            "rice_validation_thresholds": rice_thresholds,
        },
        OUT_DIR / "model_packages.joblib",
    )

    manifest = {
        "feature_protocol": "0.96-style 6839 columns: 183 bio features + 6656 global PLM embeddings. Same schema for Arabidopsis and rice.",
        "n_features": int(len(feature_names)),
        "n_bio": int(n_bio),
        "arabidopsis_base_3359": int(len(y_3359)),
        "arabidopsis_base_positive": int(y_3359.sum()),
        "arabidopsis_unknown_pseudo_0.65_0.35": int(len(ath_pseudo_meta)),
        "arabidopsis_unknown_pseudo_positive": int(ath_pseudo_meta["label"].sum()),
        "arabidopsis_unknown_pseudo_negative": int((ath_pseudo_meta["label"] == 0).sum()),
        "arabidopsis_train_actual": int(len(y_ath_train)),
        "arabidopsis_train_note": "80% of 3359 + 14731 pseudo labels; validation/test are held out from the 3359 set to avoid leakage.",
        "arabidopsis_validation": int(len(ath_val_idx)),
        "arabidopsis_test": int(len(ath_test_idx)),
        "rice_label_source": str(RICE_LABELS),
        "rice_noconflict_total": int(len(y_rice)),
        "rice_positive": int(y_rice.sum()),
        "rice_negative": int((y_rice == 0).sum()),
        "rice_train": int(len(y_rice_train)),
        "rice_validation": int(len(rice_val_idx)),
        "rice_test": int(len(rice_test_idx)),
        "scores": {
            "arabidopsis_internal": ath_scores.to_dict(orient="records"),
            "rice_internal": rice_scores.to_dict(orient="records"),
            "cross_species": cross_rows,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
