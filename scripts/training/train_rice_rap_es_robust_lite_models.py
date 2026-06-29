from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

import train_rice_rap_es_no_conflict_species_model as rap_base
import train_rice_species_specific_strict_highconf_repeated as base


ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
COMMON_DIR = ROOT / "cross_species_ath_rice_common_features_models"
OUT_ROOT = COMMON_DIR / "rice_RAP_ES_robust_lite_repeated10"
SEEDS = list(range(20260700, 20260710))
CONFIGS = [
    {"name": "seed20260700_pca512_k700", "seed": 20260700, "pca": 512, "k": 700},
    {"name": "seed20260701_pca768_k900", "seed": 20260701, "pca": 768, "k": 900},
    {"name": "seed20260702_pca384_k550", "seed": 20260702, "pca": 384, "k": 550},
]
PRED_COLS = ["lgbm_gbdt", "xgb_depth3", "logistic", "mean_all"]


def make_models(pos_weight: float, seed: int):
    return {
        "lgbm_gbdt": LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.92,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=2.2,
            scale_pos_weight=pos_weight,
            random_state=seed + 4,
            n_jobs=4,
            verbosity=-1,
        ),
        "xgb_depth3": XGBClassifier(
            n_estimators=900,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=pos_weight,
            random_state=seed + 5,
            n_jobs=4,
        ),
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.15,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed + 6,
                    ),
                ),
            ]
        ),
    }


def fit_with_optional_weight(model, X, y, w):
    try:
        if isinstance(model, Pipeline):
            model.fit(X, y, clf__sample_weight=w)
        else:
            model.fit(X, y, sample_weight=w)
    except TypeError:
        model.fit(X, y)
    return model


def logit_mean(x: np.ndarray):
    p = np.clip(x, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def fit_cv_predict(X_train, y_train, w_train, targets: dict[str, np.ndarray], n_numeric: int, tag: str):
    oof_cols = []
    target_cols = {name: [] for name in targets}
    fold_rows = []
    for cfg in CONFIGS:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(cfg["seed"]))
        pos_weight = float(w_train[y_train == 0].sum() / max(1e-6, w_train[y_train == 1].sum()))
        models = make_models(pos_weight, int(cfg["seed"]))
        oof = {name: np.zeros(len(y_train), dtype=np.float32) for name in models}
        oof["mean_all"] = np.zeros(len(y_train), dtype=np.float32)
        target_fold = {name: {model_name: [] for model_name in PRED_COLS} for name in targets}
        for fold, (tr, va) in enumerate(folds.split(X_train, y_train), 1):
            Xtr, Xva, transforms = base.full.opt.make_fold_features(
                X_train[tr],
                X_train[va],
                n_numeric,
                y_train[tr],
                k=int(cfg["k"]),
                n_pca_limit=int(cfg["pca"]),
            )
            transformed_targets = {name: base.transform_with(transforms, X, n_numeric) for name, X in targets.items()}
            val_preds = {}
            target_preds = {name: {} for name in targets}
            for model_name, model_def in models.items():
                model = clone(model_def)
                fit_with_optional_weight(model, Xtr, y_train[tr], w_train[tr])
                val_prob = model.predict_proba(Xva)[:, 1]
                oof[model_name][va] = val_prob
                val_preds[model_name] = val_prob
                for target_name, Xt in transformed_targets.items():
                    prob = model.predict_proba(Xt)[:, 1]
                    target_preds[target_name][model_name] = prob
                    target_fold[target_name][model_name].append(prob)
            oof["mean_all"][va] = np.mean([val_preds[n] for n in models], axis=0)
            for target_name in targets:
                target_fold[target_name]["mean_all"].append(np.mean([target_preds[target_name][n] for n in models], axis=0))
            auc = roc_auc_score(y_train[va], oof["mean_all"][va])
            fold_rows.append({"tag": tag, "config": cfg["name"], "fold": fold, "mean_all_auc": float(auc)})
            print(f"{tag} {cfg['name']} fold {fold}: mean_all_auc={auc:.4f}", flush=True)
        for model_name in PRED_COLS:
            oof_cols.append(oof[model_name])
            for target_name in targets:
                target_cols[target_name].append(np.mean(target_fold[target_name][model_name], axis=0))
    X_oof = np.vstack(oof_cols).T.astype(np.float32)
    X_targets = {name: np.vstack(cols).T.astype(np.float32) for name, cols in target_cols.items()}
    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(512, len(y_train)), output_distribution="normal", random_state=1)),
            ("clf", LogisticRegression(C=0.02, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=7)),
        ]
    )
    meta.fit(X_oof, y_train, clf__sample_weight=w_train)
    source_preds = {
        "meta": meta.predict_proba(X_oof)[:, 1],
        "mean": X_oof.mean(axis=1),
        "logit_mean": logit_mean(X_oof),
    }
    target_preds = {}
    for target_name, X_meta in X_targets.items():
        target_preds[target_name] = {
            "meta": meta.predict_proba(X_meta)[:, 1],
            "mean": X_meta.mean(axis=1),
            "logit_mean": logit_mean(X_meta),
        }
    return source_preds, target_preds, fold_rows


def run_dataset(label_path: Path, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_rap, rap_meta, feature_names, n_numeric, _coverage, _dropped = rap_base.load_rap_feature_matrix()
    labels = pd.read_csv(label_path, sep="\t").rename(columns={"final_label": "label"})
    rap_meta = rap_meta.copy()
    rap_meta["matrix_row"] = np.arange(len(rap_meta), dtype=int)
    meta = rap_meta.merge(labels, on="rap_gene_id", how="inner").reset_index(drop=True)
    X = X_rap[meta["matrix_row"].to_numpy(dtype=int)].astype(np.float32)
    meta = meta.drop(columns=["matrix_row"])
    y = meta["label"].astype(int).to_numpy(dtype=np.int8)
    weights = base.compute_sample_weights(meta)
    meta.assign(sample_weight=weights).to_csv(out_dir / f"{tag}_labeled_feature_intersection.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / f"{tag}_feature_names.tsv", sep="\t", index=False)

    all_scores = []
    all_folds = []
    all_test_predictions = []
    for seed in SEEDS:
        train_idx, val_idx, test_idx = base.split_80_10_10(y, seed)
        source_preds, target_preds, folds = fit_cv_predict(
            X[train_idx],
            y[train_idx],
            weights[train_idx],
            {"validation": X[val_idx], "test": X[test_idx]},
            n_numeric,
            f"{tag}_seed_{seed}",
        )
        all_folds.extend(folds)
        for model_name in ["meta", "mean", "logit_mean"]:
            val_prob = target_preds["validation"][model_name]
            test_prob = target_preds["test"][model_name]
            threshold = base.best_threshold(y[val_idx], val_prob)
            test_metrics = base.binary_metrics(y[test_idx], test_prob, threshold["threshold"])
            all_scores.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_n": int(len(train_idx)),
                    "validation_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                    "train_positive": int(y[train_idx].sum()),
                    "validation_positive": int(y[val_idx].sum()),
                    "test_positive": int(y[test_idx].sum()),
                    "validation_threshold": float(threshold["threshold"]),
                    "train_oof_auc": float(roc_auc_score(y[train_idx], source_preds[model_name])),
                    "train_oof_auprc": float(average_precision_score(y[train_idx], source_preds[model_name])),
                    "validation_auc": float(roc_auc_score(y[val_idx], val_prob)),
                    "validation_auprc": float(average_precision_score(y[val_idx], val_prob)),
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
            all_test_predictions.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "model": model_name,
                        "rap_gene_id": meta.iloc[test_idx]["rap_gene_id"].to_numpy(),
                        "msu_gene_id": meta.iloc[test_idx]["msu_gene_id"].to_numpy(),
                        "label": y[test_idx],
                        "probability": test_prob,
                        "threshold": threshold["threshold"],
                    }
                )
            )
        score_df = pd.DataFrame(all_scores)
        score_df.to_csv(out_dir / f"{tag}_robust_lite_repeated10_holdout_scores.tsv", sep="\t", index=False)
        base.summarize_scores(score_df).to_csv(out_dir / f"{tag}_robust_lite_repeated10_holdout_summary.tsv", sep="\t", index=False)
        pd.DataFrame(all_folds).to_csv(out_dir / f"{tag}_robust_lite_repeated10_fold_scores.tsv", sep="\t", index=False)
        pd.concat(all_test_predictions, ignore_index=True).to_csv(out_dir / f"{tag}_robust_lite_repeated10_test_predictions.tsv", sep="\t", index=False)
        print(base.summarize_scores(score_df).to_string(index=False), flush=True)

    manifest = {
        "label_file": str(label_path),
        "model_note": "Robust lite stack avoids sklearn ExtraTrees/RandomForest because local sklearn inspect intermittently crashes during forest clone.",
        "base_models": list(make_models(1.0, 1).keys()),
        "configs": CONFIGS,
        "seeds": SEEDS,
        "labeled_feature_intersection": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": int(n_numeric),
        "plm_feature_count": int(X.shape[1] - n_numeric),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    run_dataset(rap_base.NO_CONFLICT_LABELS, OUT_ROOT / "no_conflict_binary", "rap_es_no_conflict")
    run_dataset(rap_base.STRICT_2SOURCE_LABELS, OUT_ROOT / "strict_2source_binary", "rap_es_strict_2source")


if __name__ == "__main__":
    main()
