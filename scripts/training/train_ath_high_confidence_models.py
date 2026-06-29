from __future__ import annotations

import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_ROOT = Path("D:/拟南芥/特征/Araport11_综合必需非必需_features")
PAPER_ROOT = Path("D:/拟南芥/文献特征复现/paper_style_features_araport11_latest_modern_replacements")
ENHANCED_ROOT = Path("D:/拟南芥/增强特征数据/processed")
LABEL_ROOT = Path("C:/Users/tly/Desktop/植物/拟南芥/综合必需非必需去重")
OUT_DIR = Path("D:/拟南芥/模型/essential_gene_prediction_high_confidence")


def source_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    return len([x for x in text.split(";") if x.strip()])


def load_matrix():
    paper = pd.read_csv(PAPER_ROOT / "paper_style_features_all_modern_replacements.tsv", sep="\t")
    status = pd.read_csv(PAPER_ROOT / "feature_status_modern_replacements.tsv", sep="\t")
    enhanced = pd.read_csv(ENHANCED_ROOT / "enhanced_downloaded_gene_features.tsv", sep="\t")

    paper["gene_id"] = paper["gene_id"].astype(str).str.upper()
    enhanced["gene_id"] = enhanced["gene_id"].astype(str).str.upper()
    merged = paper.merge(
        enhanced.drop(columns=["seq_id", "label"], errors="ignore"),
        on="gene_id",
        how="left",
        suffixes=("", "_enh"),
    )

    drop = set(status.loc[status["status"].isin(["unavailable", "placeholder_nan"]), "feature_name"])
    meta = {"seq_id", "gene_id", "label", "source_fasta", "top_paralog_gene"}
    bio_cols = []
    for col in merged.columns:
        if col in meta or col in drop:
            continue
        vals = pd.to_numeric(merged[col], errors="coerce")
        if vals.notna().sum() == 0 or vals.nunique(dropna=True) <= 1:
            continue
        bio_cols.append(col)

    ids = merged["seq_id"].astype(str).to_numpy()
    genes = merged["gene_id"].astype(str).str.upper().to_numpy()
    X_bio = merged[bio_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)

    emb_blocks = []
    emb_names = []
    for sub, prefix in [
        ("esm2_global", "esm2"),
        ("protbert_global", "protbert"),
        ("prott5_global", "prott5"),
    ]:
        emb_dir = FEATURE_ROOT / sub
        emb_ids = np.load(emb_dir / "all_ids.npy", allow_pickle=True).astype(str)
        emb = np.load(emb_dir / "all_emb.npy").astype(np.float32)
        idx = {sid: i for i, sid in enumerate(emb_ids)}
        emb_blocks.append(emb[[idx[sid] for sid in ids]])
        emb_names.extend([f"{prefix}_{i}" for i in range(emb.shape[1])])

    X = np.hstack([X_bio] + emb_blocks).astype(np.float32)
    return X, ids, genes, bio_cols + emb_names, len(bio_cols)


def load_labels(genes: np.ndarray) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_ROOT / "final_gene_classification_summary.tsv", sep="\t")
    labels["gene_id"] = labels["gene_id"].astype(str).str.upper()
    labels["essential_source_count"] = labels["essential_or_conditional_sources"].map(source_count)
    labels["nonessential_source_count"] = labels["nonessential_sources"].map(source_count)
    ess_text = labels["essential_or_conditional_sources"].fillna("").astype(str)
    labels["has_ogee"] = ess_text.str.contains("ogee", case=False, regex=False)
    labels["has_embryo"] = ess_text.str.contains("embryo", case=False, regex=False)
    labels["label_from_evidence"] = (labels["final_classification"] == "essential").astype(int)
    lookup = labels.set_index("gene_id")
    matched = lookup.loc[pd.Index(genes)]
    matched = matched.reset_index().rename(columns={"index": "gene_id"})
    return matched


def subset_masks(label_df: pd.DataFrame) -> dict[str, np.ndarray]:
    no_conflict = label_df["has_conflict"].eq("no")
    essential = label_df["final_classification"].eq("essential")
    nonessential = label_df["final_classification"].eq("nonessential")
    strong_essential = essential & (
        label_df["essential_source_count"].ge(2) | label_df["has_ogee"] | label_df["has_embryo"]
    )

    return {
        "no_conflict": no_conflict.to_numpy(),
        "medium_high_confidence": (
            no_conflict
            & ((essential & label_df["essential_source_count"].ge(1)) | (nonessential & label_df["nonessential_source_count"].ge(2)))
        ).to_numpy(),
        "strict_high_confidence": (
            no_conflict & (strong_essential | (nonessential & label_df["nonessential_source_count"].ge(2)))
        ).to_numpy(),
        "very_strict_high_confidence": (
            no_conflict & (strong_essential | (nonessential & label_df["nonessential_source_count"].ge(3)))
        ).to_numpy(),
    }


def make_fold_features(X_train, X_test, n_bio: int, y_train, k: int):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    scaler = StandardScaler()
    Xtr_emb = scaler.fit_transform(Xtr[:, n_bio:])
    Xte_emb = scaler.transform(Xte[:, n_bio:])
    n_pca = min(512, Xtr_emb.shape[0] - 1, Xtr_emb.shape[1])
    pca = PCA(n_components=n_pca, random_state=17)
    Xtr_pca = pca.fit_transform(Xtr_emb)
    Xte_pca = pca.transform(Xte_emb)

    Xtr_fold = np.hstack([Xtr[:, :n_bio], Xtr_pca]).astype(np.float32)
    Xte_fold = np.hstack([Xte[:, :n_bio], Xte_pca]).astype(np.float32)
    selector = SelectKBest(score_func=f_classif, k=min(k, Xtr_fold.shape[1]))
    Xtr_sel = selector.fit_transform(Xtr_fold, y_train)
    Xte_sel = selector.transform(Xte_fold)
    return Xtr_sel, Xte_sel, (imp, scaler, pca, selector)


def make_models(pos_weight: float):
    return {
        "extra": ExtraTreesClassifier(
            n_estimators=1200,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=21,
            n_jobs=-1,
        ),
        "lgbm": LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=pos_weight,
            random_state=22,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=900,
            learning_rate=0.025,
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
            random_state=23,
            n_jobs=-1,
        ),
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=24,
                    ),
                ),
            ]
        ),
    }


def evaluate_subset(name: str, X, y, genes, ids, n_bio: int, out_dir: Path, k: int = 700):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260610)
    pos_weight = float((y == 0).sum() / y.sum())
    models = make_models(pos_weight)
    oof = {model_name: np.zeros(len(y), dtype=np.float32) for model_name in models}
    oof["stack_mean"] = np.zeros(len(y), dtype=np.float32)
    fold_rows = []

    for fold, (tr, te) in enumerate(folds.split(X, y), 1):
        Xtr, Xte, _ = make_fold_features(X[tr], X[te], n_bio, y[tr], k=k)
        fold_preds = []
        row = {"subset": name, "fold": fold, "selected_features": int(Xtr.shape[1])}
        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(Xtr, y[tr])
            pred = fitted.predict_proba(Xte)[:, 1]
            oof[model_name][te] = pred
            fold_preds.append(pred)
            row[f"{model_name}_auc"] = float(roc_auc_score(y[te], pred))
        mean_pred = np.mean(fold_preds, axis=0)
        oof["stack_mean"][te] = mean_pred
        row["stack_mean_auc"] = float(roc_auc_score(y[te], mean_pred))
        row["stack_mean_auprc"] = float(average_precision_score(y[te], mean_pred))
        fold_rows.append(row)
        print(f"{name} fold {fold}: stack_mean AUC={row['stack_mean_auc']:.4f}")

    scores = []
    for model_name, pred in oof.items():
        scores.append(
            {
                "subset": name,
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
    pred_df.to_csv(out_dir / f"{name}_oof_predictions.tsv", sep="\t", index=False)
    return scores, fold_rows, oof


def fit_final_best(X, y, n_bio: int, k: int, model_name: str):
    X_fit, _, transforms = make_fold_features(X, X, n_bio, y, k=k)
    pos_weight = float((y == 0).sum() / y.sum())
    models = make_models(pos_weight)
    if model_name == "stack_mean":
        fitted_models = {}
        for base_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_fit, y)
            fitted_models[base_name] = fitted
        return {
            "transforms": transforms,
            "models": fitted_models,
            "model_name": model_name,
            "selected_k": k,
            "prediction_rule": "mean of base model positive-class probabilities",
        }
    model = models[model_name]
    fitted = clone(model)
    fitted.fit(X_fit, y)
    return {"transforms": transforms, "model": fitted, "model_name": model_name, "selected_k": k}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, ids, genes, feature_names, n_bio = load_matrix()
    labels = load_labels(genes)
    masks = subset_masks(labels)
    y_all = labels["label_from_evidence"].to_numpy(np.int8)

    subset_summary = []
    all_scores = []
    all_folds = []
    best = None
    for subset_name, mask in masks.items():
        y = y_all[mask]
        if y.sum() < 40 or (y == 0).sum() < 40:
            continue
        subset_summary.append(
            {
                "subset": subset_name,
                "n_genes": int(mask.sum()),
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
            }
        )
        scores, folds, _ = evaluate_subset(
            subset_name, X[mask], y, genes[mask], ids[mask], n_bio, OUT_DIR, k=700
        )
        all_scores.extend(scores)
        all_folds.extend(folds)
        for row in scores:
            if best is None or row["oof_auc"] > best["oof_auc"]:
                best = {**row, "mask": mask}

    score_df = pd.DataFrame(all_scores).sort_values("oof_auc", ascending=False)
    fold_df = pd.DataFrame(all_folds)
    subset_df = pd.DataFrame(subset_summary)
    score_df.to_csv(OUT_DIR / "high_confidence_oof_scores.tsv", sep="\t", index=False)
    fold_df.to_csv(OUT_DIR / "high_confidence_fold_scores.tsv", sep="\t", index=False)
    subset_df.to_csv(OUT_DIR / "high_confidence_subset_summary.tsv", sep="\t", index=False)

    if best is not None:
        mask = best.pop("mask")
        y = y_all[mask]
        final = fit_final_best(X[mask], y, n_bio, k=700, model_name=best["model"])
        joblib.dump(final, OUT_DIR / "final_best_high_confidence_model.joblib")
        best["note"] = (
            "AUC is 5-fold out-of-fold performance on the named high-confidence label subset. "
            "It is not directly comparable to full conflicted-label AUC."
        )

    manifest = {
        "best": best,
        "subsets": subset_summary,
        "raw_input_features": int(X.shape[1]),
        "bio_features": int(n_bio),
        "embedding_features": int(X.shape[1] - n_bio),
        "feature_selection_after_fold_safe_pca": 700,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"raw_input_feature_name": feature_names}).to_csv(
        OUT_DIR / "raw_input_feature_names.tsv", sep="\t", index=False
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
