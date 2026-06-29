from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

import train_ath_high_confidence_models as base
import train_ath_no_conflict_optimized as opt
from extract_literature_sequence_features_araport11 import all_longest_records, sequence_features


PSEUDO_ROOT = Path("D:/拟南芥/模型/essential_gene_prediction_consensus_2plus_pseudolabel_holdout10")
CONSENSUS_ROOT = Path("D:/拟南芥/模型/essential_gene_prediction_consensus_2plus")
UNKNOWN_ROOT = Path("D:/拟南芥/特征_unknown")
OUT_DIR = Path("D:/拟南芥/模型/essential_gene_prediction_consensus1623_plus_pseudo06_predict_all_unknown")

PRED_COLS = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3", "logistic", "mean_all", "mean_tree"]
CONFIGS = [
    {"name": "seed20260610_pca512_k700", "seed": 20260610, "pca": 512, "k": 700},
    {"name": "seed20260611_pca512_k700", "seed": 20260611, "pca": 512, "k": 700},
    {"name": "seed20260612_pca768_k900", "seed": 20260612, "pca": 768, "k": 900},
    {"name": "seed20260613_pca1024_k1100", "seed": 20260613, "pca": 1024, "k": 1100},
    {"name": "seed20260614_pca384_k550", "seed": 20260614, "pca": 384, "k": 550},
]
THRESHOLDS = [
    ("thr_0.95_0.05", 0.95, 0.05),
    ("thr_0.90_0.10", 0.90, 0.10),
    ("thr_0.85_0.15", 0.85, 0.15),
    ("thr_0.80_0.20", 0.80, 0.20),
    ("thr_0.75_0.25", 0.75, 0.25),
    ("thr_0.70_0.30", 0.70, 0.30),
    ("thr_0.65_0.35", 0.65, 0.35),
    ("thr_0.60_0.40", 0.60, 0.40),
    ("thr_0.55_0.45", 0.55, 0.45),
    ("thr_0.50_0.50", 0.50, 0.50),
]


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def transform_with(transforms, X, n_bio: int):
    imp, scaler, pca, selector = transforms
    X_imp = imp.transform(X)
    X_emb = scaler.transform(X_imp[:, n_bio:])
    X_pca = pca.transform(X_emb)
    return selector.transform(np.hstack([X_imp[:, :n_bio], X_pca]).astype(np.float32))


def build_train_labels(genes: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels = pd.read_csv(CONSENSUS_ROOT / "consensus_2plus_gene_labels.tsv", sep="\t")
    labels["gene_id"] = labels["gene_id"].astype(str).str.upper()
    if "classification" not in labels.columns and "consensus_classification" in labels.columns:
        labels["classification"] = labels["consensus_classification"]
    true_labels = labels[["gene_id", "label", "classification", "source_count", "sources"]].copy()
    true_labels["label_source"] = "true_consensus_2plus"
    true_labels["teacher_meta_probability"] = np.nan

    pseudo = pd.read_csv(PSEUDO_ROOT / "unlabeled_gene_teacher_predictions.tsv", sep="\t")
    pseudo["gene_id"] = pseudo["gene_id"].astype(str).str.upper()
    pseudo_sel = pseudo[(pseudo["teacher_meta_probability"].ge(0.60)) | (pseudo["teacher_meta_probability"].le(0.40))].copy()
    pseudo_sel["label"] = (pseudo_sel["teacher_meta_probability"].ge(0.50)).astype(int)
    pseudo_sel["classification"] = np.where(pseudo_sel["label"].eq(1), "pseudo_essential", "pseudo_nonessential")
    pseudo_sel["source_count"] = np.nan
    pseudo_sel["sources"] = "teacher_meta_probability_0.60_0.40"
    pseudo_sel["label_source"] = "pseudo_0.60_0.40_from_2216"
    pseudo_sel = pseudo_sel[
        ["gene_id", "label", "classification", "source_count", "sources", "label_source", "teacher_meta_probability"]
    ]

    combined = pd.concat([true_labels, pseudo_sel], ignore_index=True)
    combined = combined.drop_duplicates("gene_id", keep="first")
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    combined = combined[combined["gene_id"].isin(gene_to_idx)].copy()
    train_idx = combined["gene_id"].map(gene_to_idx).to_numpy(int)
    y_train = combined["label"].to_numpy(np.int8)
    return train_idx, y_train, combined


def build_unknown_matrix(feature_names: list[str], n_bio: int):
    records = all_longest_records()
    unknown_ids = np.load(UNKNOWN_ROOT / "ESM2" / "unknown_ids.npy", allow_pickle=True).astype(str)
    unknown_genes = np.array([sid.split(".", 1)[0].upper() for sid in unknown_ids])

    rows = []
    for sid, gene in zip(unknown_ids, unknown_genes):
        rec = records.get(gene)
        feats = sequence_features(rec) if rec is not None else {}
        rows.append(feats)
    bio_cols = feature_names[:n_bio]
    bio_df = pd.DataFrame(rows)
    for col in bio_cols:
        if col not in bio_df.columns:
            bio_df[col] = np.nan
    X_bio = bio_df[bio_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)

    blocks = []
    for subdir, expected_dim in [("ESM2", 2560), ("ProtBERT", 2048), ("ProtT5", 2048)]:
        ids = np.load(UNKNOWN_ROOT / subdir / "unknown_ids.npy", allow_pickle=True).astype(str)
        emb = np.load(UNKNOWN_ROOT / subdir / "unknown_emb.npy").astype(np.float32)
        idx = {sid: i for i, sid in enumerate(ids)}
        if not all(sid in idx for sid in unknown_ids):
            missing = [sid for sid in unknown_ids if sid not in idx][:5]
            raise RuntimeError(f"{subdir} missing ids: {missing}")
        block = emb[[idx[sid] for sid in unknown_ids]]
        if block.shape[1] != expected_dim:
            raise RuntimeError(f"{subdir} dim mismatch: {block.shape}")
        blocks.append(block)
    X_unknown = np.hstack([X_bio] + blocks).astype(np.float32)
    return X_unknown, unknown_ids, unknown_genes


def fit_predict_library(X_train, y_train, X_unknown, n_bio: int):
    unknown_cols = []
    names = []
    final_models = []
    pos_weight = float((y_train == 0).sum() / y_train.sum())
    for config in CONFIGS:
        seed = int(config["seed"])
        model_defs = opt.make_models(pos_weight, seed)
        Xtr, Xu, transforms = opt.make_fold_features(
            X_train, X_unknown, n_bio, y_train, k=int(config["k"]), n_pca_limit=int(config["pca"])
        )
        model_preds = {}
        for model_name, model_def in model_defs.items():
            model = clone(model_def)
            model.fit(Xtr, y_train)
            pred = model.predict_proba(Xu)[:, 1]
            model_preds[model_name] = pred
            final_models.append({"config": config, "model_name": model_name, "model": model, "transforms": transforms})
        tree_names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
        model_preds["mean_all"] = np.mean([model_preds[n] for n in model_defs], axis=0)
        model_preds["mean_tree"] = np.mean([model_preds[n] for n in tree_names], axis=0)
        for model_name in PRED_COLS:
            unknown_cols.append(model_preds[model_name])
            names.append(f"{config['name']}__{model_name}")
        print(f"trained {config['name']} predict_unknown_mean_all={model_preds['mean_all'].mean():.4f}")
    return np.vstack(unknown_cols).T.astype(np.float32), names, final_models


def make_threshold_outputs(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, pos_thr, neg_thr in THRESHOLDS:
        mask_pos = pred_df["final_meta_probability"].ge(pos_thr)
        mask_neg = pred_df["final_meta_probability"].le(neg_thr)
        selected = pred_df[mask_pos | mask_neg].copy()
        selected["threshold_set"] = name
        selected["threshold_positive"] = pos_thr
        selected["threshold_negative"] = neg_thr
        selected["pseudo_label"] = np.where(selected["final_meta_probability"].ge(0.5), 1, 0)
        selected["pseudo_classification"] = np.where(selected["pseudo_label"].eq(1), "essential", "nonessential")
        selected.to_csv(OUT_DIR / f"unknown_selected_{name}.tsv", sep="\t", index=False)
        rows.append(
            {
                "threshold_set": name,
                "positive_threshold": pos_thr,
                "negative_threshold": neg_thr,
                "selected_total": int(len(selected)),
                "selected_essential": int((selected["pseudo_label"] == 1).sum()),
                "selected_nonessential": int((selected["pseudo_label"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_all, ids_all, genes_all, feature_names, n_bio = base.load_matrix()
    genes_all = np.array([str(g).upper() for g in genes_all])
    train_idx, y_train, train_labels = build_train_labels(genes_all)
    X_train = X_all[train_idx].astype(np.float32)

    X_unknown, unknown_ids, unknown_genes = build_unknown_matrix(feature_names, n_bio)
    X_unknown_meta, meta_names, final_models = fit_predict_library(X_train, y_train, X_unknown, n_bio)

    train_meta_features = []
    for item in final_models:
        Xtr_sel = transform_with(item["transforms"], X_train, n_bio)
        train_meta_features.append(item["model"].predict_proba(Xtr_sel)[:, 1])
    train_meta_base = np.vstack(train_meta_features).T.astype(np.float32)

    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(512, len(y_train)), output_distribution="normal", random_state=1)),
            ("clf", LogisticRegression(C=0.02, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=7)),
        ]
    )
    meta.fit(train_meta_base, y_train)
    base_model_cols = [i for i, name in enumerate(meta_names) if not name.endswith(("mean_all", "mean_tree"))]
    final_meta = meta.predict_proba(X_unknown_meta[:, base_model_cols])[:, 1]
    final_mean = X_unknown_meta.mean(axis=1)

    pred_df = pd.DataFrame(
        {
            "seq_id": unknown_ids,
            "gene_id": unknown_genes,
            "final_meta_probability": final_meta,
            "final_mean_probability": final_mean,
            "classification_0.5": np.where(final_meta >= 0.5, "essential", "nonessential"),
        }
    )
    pred_df.to_csv(OUT_DIR / "all_unknown_longest_gene_predictions.tsv", sep="\t", index=False)
    threshold_summary = make_threshold_outputs(pred_df)
    threshold_summary.to_csv(OUT_DIR / "unknown_threshold_summary.tsv", sep="\t", index=False)
    train_labels.to_csv(OUT_DIR / "training_labels_true1623_plus_pseudo06.tsv", sep="\t", index=False)
    pd.DataFrame({"meta_feature_name": meta_names}).to_csv(OUT_DIR / "unknown_prediction_library_feature_names.tsv", sep="\t", index=False)
    joblib.dump(
        {
            "meta_model": meta,
            "final_models": final_models,
            "meta_feature_names": meta_names,
            "feature_names": feature_names,
            "n_bio": n_bio,
        },
        OUT_DIR / "final_pseudo06_unknown_prediction_models.joblib",
    )
    manifest = {
        "longest_transcript_only": True,
        "true_consensus_genes": int((train_labels["label_source"] == "true_consensus_2plus").sum()),
        "pseudo_rule_from_2216": "teacher_meta_probability >= 0.60 essential, <= 0.40 nonessential",
        "pseudo_added": int((train_labels["label_source"] == "pseudo_0.60_0.40_from_2216").sum()),
        "pseudo_added_essential": int(((train_labels["label_source"] == "pseudo_0.60_0.40_from_2216") & (train_labels["label"] == 1)).sum()),
        "pseudo_added_nonessential": int(((train_labels["label_source"] == "pseudo_0.60_0.40_from_2216") & (train_labels["label"] == 0)).sum()),
        "training_total": int(len(train_labels)),
        "unknown_predicted": int(len(pred_df)),
        "threshold_summary": threshold_summary.to_dict(orient="records"),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
