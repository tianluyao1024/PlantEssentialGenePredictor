from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_ath_high_confidence_models as base


OUT_DIR = Path("D:/拟南芥/模型/essential_gene_prediction_no_conflict_optimized")


def make_fold_features(X_train, X_test, n_bio: int, y_train, k: int, n_pca_limit: int):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    scaler = StandardScaler()
    Xtr_emb = scaler.fit_transform(Xtr[:, n_bio:])
    Xte_emb = scaler.transform(Xte[:, n_bio:])
    n_pca = min(n_pca_limit, Xtr_emb.shape[0] - 1, Xtr_emb.shape[1])
    pca = PCA(n_components=n_pca, random_state=17)
    Xtr_pca = pca.fit_transform(Xtr_emb)
    Xte_pca = pca.transform(Xte_emb)

    Xtr_fold = np.hstack([Xtr[:, :n_bio], Xtr_pca]).astype(np.float32)
    Xte_fold = np.hstack([Xte[:, :n_bio], Xte_pca]).astype(np.float32)
    selector = SelectKBest(score_func=f_classif, k=min(k, Xtr_fold.shape[1]))
    Xtr_sel = selector.fit_transform(Xtr_fold, y_train)
    Xte_sel = selector.transform(Xte_fold)
    return Xtr_sel, Xte_sel, (imp, scaler, pca, selector)


def make_models(pos_weight: float, seed: int):
    return {
        "extra_sqrt": ExtraTreesClassifier(
            n_estimators=1600,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed + 1,
            n_jobs=-1,
        ),
        "extra_log2": ExtraTreesClassifier(
            n_estimators=1600,
            max_features="log2",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed + 2,
            n_jobs=-1,
        ),
        "rf_sqrt": RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed + 3,
            n_jobs=-1,
        ),
        "lgbm_gbdt": LGBMClassifier(
            objective="binary",
            n_estimators=1600,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.92,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=2.2,
            scale_pos_weight=pos_weight,
            random_state=seed + 4,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb_depth3": XGBClassifier(
            n_estimators=1200,
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
            n_jobs=-1,
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


def evaluate_config(X, y, genes, ids, n_bio: int, config: dict):
    seed = int(config["seed"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pos_weight = float((y == 0).sum() / y.sum())
    models = make_models(pos_weight, seed)
    oof = {name: np.zeros(len(y), dtype=np.float32) for name in models}
    oof["mean_all"] = np.zeros(len(y), dtype=np.float32)
    oof["mean_tree"] = np.zeros(len(y), dtype=np.float32)
    fold_rows = []

    for fold, (tr, te) in enumerate(folds.split(X, y), 1):
        Xtr, Xte, _ = make_fold_features(
            X[tr],
            X[te],
            n_bio,
            y[tr],
            k=int(config["k"]),
            n_pca_limit=int(config["pca"]),
        )
        preds = {}
        row = {"config": config["name"], "seed": seed, "fold": fold, "selected_features": int(Xtr.shape[1])}
        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(Xtr, y[tr])
            pred = fitted.predict_proba(Xte)[:, 1]
            preds[model_name] = pred
            oof[model_name][te] = pred
            row[f"{model_name}_auc"] = float(roc_auc_score(y[te], pred))

        all_stack = np.mean(list(preds.values()), axis=0)
        tree_stack = np.mean(
            [preds[name] for name in ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]],
            axis=0,
        )
        oof["mean_all"][te] = all_stack
        oof["mean_tree"][te] = tree_stack
        row["mean_all_auc"] = float(roc_auc_score(y[te], all_stack))
        row["mean_tree_auc"] = float(roc_auc_score(y[te], tree_stack))
        fold_rows.append(row)
        print(
            f"{config['name']} fold {fold}: "
            f"mean_tree AUC={row['mean_tree_auc']:.4f} mean_all AUC={row['mean_all_auc']:.4f}"
        )

    scores = []
    for model_name, pred in oof.items():
        scores.append(
            {
                "config": config["name"],
                "seed": seed,
                "pca": int(config["pca"]),
                "k": int(config["k"]),
                "model": model_name,
                "oof_auc": float(roc_auc_score(y, pred)),
                "oof_auprc": float(average_precision_score(y, pred)),
                "n_genes": int(len(y)),
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
            }
        )

    pred_df = pd.DataFrame({"seq_id": ids, "gene_id": genes, "label": y})
    for model_name, pred in oof.items():
        pred_df[f"{model_name}_oof_probability"] = pred
    pred_df.to_csv(OUT_DIR / f"{config['name']}_oof_predictions.tsv", sep="\t", index=False)
    return scores, fold_rows


def fit_final(X, y, n_bio: int, best: dict):
    X_fit, _, transforms = make_fold_features(
        X, X, n_bio, y, k=int(best["k"]), n_pca_limit=int(best["pca"])
    )
    pos_weight = float((y == 0).sum() / y.sum())
    models = make_models(pos_weight, int(best["seed"]))
    model_name = str(best["model"])
    if model_name == "mean_tree":
        names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
    elif model_name == "mean_all":
        names = list(models)
    else:
        names = [model_name]

    fitted = {}
    for name in names:
        model = clone(models[name])
        model.fit(X_fit, y)
        fitted[name] = model
    return {
        "transforms": transforms,
        "models": fitted,
        "model_name": model_name,
        "prediction_rule": "mean probability if more than one fitted model, otherwise single model probability",
        "pca": int(best["pca"]),
        "k": int(best["k"]),
        "seed": int(best["seed"]),
    }


def predict_all(model_obj, X, n_bio: int):
    imp, scaler, pca, selector = model_obj["transforms"]
    X_imp = imp.transform(X)
    X_emb = scaler.transform(X_imp[:, n_bio:])
    X_pca = pca.transform(X_emb)
    X_fold = np.hstack([X_imp[:, :n_bio], X_pca]).astype(np.float32)
    X_sel = selector.transform(X_fold)
    probs = [model.predict_proba(X_sel)[:, 1] for model in model_obj["models"].values()]
    return np.mean(probs, axis=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, ids, genes, feature_names, n_bio = base.load_matrix()
    label_df = base.load_labels(genes)
    mask = label_df["has_conflict"].eq("no").to_numpy()
    y = label_df.loc[mask, "label_from_evidence"].to_numpy(np.int8)

    configs = [
        {"name": "seed20260610_pca512_k700", "seed": 20260610, "pca": 512, "k": 700},
        {"name": "seed20260611_pca512_k700", "seed": 20260611, "pca": 512, "k": 700},
        {"name": "seed20260612_pca512_k700", "seed": 20260612, "pca": 512, "k": 700},
        {"name": "seed20260613_pca768_k900", "seed": 20260613, "pca": 768, "k": 900},
        {"name": "seed20260614_pca384_k550", "seed": 20260614, "pca": 384, "k": 550},
    ]

    all_scores = []
    all_folds = []
    for config in configs:
        scores, folds = evaluate_config(X[mask], y, genes[mask], ids[mask], n_bio, config)
        all_scores.extend(scores)
        all_folds.extend(folds)

    score_df = pd.DataFrame(all_scores).sort_values("oof_auc", ascending=False)
    fold_df = pd.DataFrame(all_folds)
    score_df.to_csv(OUT_DIR / "no_conflict_optimized_oof_scores.tsv", sep="\t", index=False)
    fold_df.to_csv(OUT_DIR / "no_conflict_optimized_fold_scores.tsv", sep="\t", index=False)

    best = score_df.iloc[0].to_dict()
    final = fit_final(X[mask], y, n_bio, best)
    joblib.dump(final, OUT_DIR / "final_no_conflict_optimized_model.joblib")

    probs = predict_all(final, X, n_bio)
    pred = pd.DataFrame(
        {
            "seq_id": ids,
            "gene_id": genes,
            "final_classification": label_df["final_classification"].to_numpy(),
            "has_conflict": label_df["has_conflict"].to_numpy(),
            "essential_probability": probs,
        }
    ).sort_values("essential_probability", ascending=False)
    pred.to_csv(OUT_DIR / "all_3839_gene_predictions_by_no_conflict_optimized_model.tsv", sep="\t", index=False)

    manifest = {
        "label_rule": "remove all genes with has_conflict=yes from training/evaluation; keep no-conflict essential and nonessential genes",
        "best": best,
        "n_no_conflict_genes": int(mask.sum()),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "raw_input_features": int(X.shape[1]),
        "bio_features": int(n_bio),
        "embedding_features": int(X.shape[1] - n_bio),
        "configs_tested": configs,
        "note": "OOF AUC is measured only on no-conflict genes. Conflicted genes are scored by the final model but not used in CV.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"raw_input_feature_name": feature_names}).to_csv(
        OUT_DIR / "raw_input_feature_names.tsv", sep="\t", index=False
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
