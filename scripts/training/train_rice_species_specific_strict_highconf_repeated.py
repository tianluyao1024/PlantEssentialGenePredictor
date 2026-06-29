from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

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
import train_full_096style_aug3359_pseudo14731_rice_noconflict_cross as full


ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
COMMON_DIR = ROOT / "cross_species_ath_rice_common_features_models"
LABELS = (
    COMMON_DIR
    / "096style_rice_strict_highconf_2source_model"
    / "rice_strict_highconf_2source_gene_labels.tsv"
)
OUT_DIR = COMMON_DIR / "rice_species_specific_strict_highconf_repeated20"

RICE_FEATURE_TABLES = [
    ("common", COMMON_DIR / "rice_cross_species_common_features_all_genes.tsv"),
    ("stable", COMMON_DIR / "rice_stable_external_features.tsv"),
    (
        "paper",
        ROOT
        / "paper_2015_lloyd_essential_gene"
        / "paper_style_features"
        / "lloyd2015_rice_paper_style_features.tsv",
    ),
]
PLM_DIRS = [
    ("esm2", COMMON_DIR / "plm_embeddings" / "esm2" / "rice"),
    ("protbert", COMMON_DIR / "plm_embeddings" / "protbert" / "rice"),
    ("prott5", COMMON_DIR / "plm_embeddings" / "prott5" / "rice"),
]

SEEDS = list(range(20260640, 20260660))
CONFIGS = full.CONFIGS
PRED_COLS = full.PRED_COLS
MAX_MODEL_JOBS = 4
ID_COLUMNS = {
    "gene_id",
    "species",
    "transcript_id",
    "representative_transcript",
    "seq_id",
    "label",
    "final_label",
    "final_classification",
    "classification",
    "class",
}

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Features .* are constant.*")
warnings.filterwarnings("ignore", message="invalid value encountered in divide.*")


def load_numeric_feature_table(tag: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "gene_id" not in df.columns:
        raise ValueError(f"{path} has no gene_id column")
    df["gene_id"] = df["gene_id"].astype(str)
    df = df.drop_duplicates("gene_id", keep="first")
    numeric = pd.DataFrame({"gene_id": df["gene_id"]})
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        out_col = col if tag == "common" else f"{tag}__{col}"
        numeric[out_col] = values
    return numeric


def merge_numeric_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = None
    coverage_rows = []
    used_names = {"gene_id"}
    for tag, path in RICE_FEATURE_TABLES:
        numeric = load_numeric_feature_table(tag, path)
        rename = {}
        for col in numeric.columns:
            if col == "gene_id":
                continue
            name = col
            if name in used_names:
                name = f"{tag}__{col}"
            suffix = 2
            unique_name = name
            while unique_name in used_names:
                unique_name = f"{name}__dup{suffix}"
                suffix += 1
            rename[col] = unique_name
            used_names.add(unique_name)
        numeric = numeric.rename(columns=rename)
        feature_cols = [c for c in numeric.columns if c != "gene_id"]
        coverage_rows.append(
            {
                "source": tag,
                "path": str(path),
                "gene_count": int(len(numeric)),
                "numeric_feature_count": int(len(feature_cols)),
            }
        )
        if base is None:
            base = numeric
        else:
            base = base.merge(numeric, on="gene_id", how="outer")
    assert base is not None
    return base, pd.DataFrame(coverage_rows)


def load_plm_embeddings(gene_order: pd.Series) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    order = gene_order.astype(str).to_numpy()
    pieces = []
    names = []
    coverage = []
    for tag, path in PLM_DIRS:
        ids = np.load(path / "all_ids.npy", allow_pickle=True).astype(str)
        emb = np.load(path / "all_emb.npy", mmap_mode="r")
        lookup = {gene_id: i for i, gene_id in enumerate(ids)}
        dim = int(emb.shape[1])
        arr = np.full((len(order), dim), np.nan, dtype=np.float32)
        hit = np.zeros(len(order), dtype=bool)
        for row, gene_id in enumerate(order):
            idx = lookup.get(gene_id)
            if idx is not None:
                arr[row] = emb[idx]
                hit[row] = True
        pieces.append(arr)
        names.extend([f"{tag}_{i:04d}" for i in range(dim)])
        coverage.append(
            {
                "source": tag,
                "path": str(path),
                "gene_count": int(hit.sum()),
                "numeric_feature_count": dim,
            }
        )
    return np.hstack(pieces).astype(np.float32), names, pd.DataFrame(coverage)


def drop_uninformative_features(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[int], pd.DataFrame]:
    rows = []
    keep = []
    for i, name in enumerate(names):
        col = X[:, i]
        non_na = int(np.isfinite(col).sum())
        if non_na == 0:
            rows.append({"feature": name, "drop_reason": "all_missing"})
            continue
        finite = col[np.isfinite(col)]
        if finite.size > 0 and np.nanstd(finite) == 0:
            rows.append({"feature": name, "drop_reason": "zero_variance"})
            continue
        keep.append(i)
    kept_names = [names[i] for i in keep]
    return X[:, keep].astype(np.float32), kept_names, keep, pd.DataFrame(rows)


def load_rice_species_specific_matrix():
    numeric, numeric_coverage = merge_numeric_features()
    numeric = numeric.sort_values("gene_id").reset_index(drop=True)
    numeric_names = [c for c in numeric.columns if c != "gene_id"]
    X_num = numeric[numeric_names].to_numpy(dtype=np.float32)
    X_plm, plm_names, plm_coverage = load_plm_embeddings(numeric["gene_id"])
    X = np.hstack([X_num, X_plm]).astype(np.float32)
    names = numeric_names + plm_names
    X, names, kept_indices, dropped = drop_uninformative_features(X, names)
    n_numeric_after_drop = int(sum(i < len(numeric_names) for i in kept_indices))
    meta = numeric[["gene_id"]].copy()
    coverage = pd.concat([numeric_coverage, plm_coverage], ignore_index=True)
    return X, meta, names, n_numeric_after_drop, coverage, dropped


def compute_sample_weights(meta: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(meta), dtype=np.float32)
    source_count = pd.to_numeric(meta["source_count"], errors="coerce").fillna(2).to_numpy()
    weights *= np.clip(1.0 + 0.18 * (source_count - 2), 0.85, 1.55)

    labels = meta["label"].astype(int).to_numpy()
    ess_types = meta.get("essential_evidence_types", pd.Series("", index=meta.index)).fillna("").astype(str).str.lower()
    lof_types = meta.get("lof_evidence_types", pd.Series("", index=meta.index)).fillna("").astype(str).str.lower()

    core_essential = ess_types.str.contains("lethal|seed_abortion|embryo|gametophyte|seedling", regex=True).to_numpy()
    sterility_only = ess_types.str.contains("sterility|pollen", regex=True).to_numpy() & ~core_essential
    strong_nonessential = lof_types.str.contains("insertion_loss|knockout", regex=True).to_numpy()

    weights[(labels == 1) & core_essential] *= 1.35
    weights[(labels == 1) & sterility_only] *= 0.90
    weights[(labels == 0) & strong_nonessential] *= 1.20

    weights = np.clip(weights, 0.75, 2.20)
    return (weights / weights.mean()).astype(np.float32)


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


def fit_with_optional_weight(model, X, y, sample_weight):
    try:
        if isinstance(model, Pipeline):
            last = model.steps[-1][0]
            model.fit(X, y, **{f"{last}__sample_weight": sample_weight})
        else:
            model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def limit_model_jobs(model):
    try:
        params = model.get_params(deep=True)
    except Exception:
        return model
    updates = {}
    for key in params:
        if key == "n_jobs" or key.endswith("__n_jobs"):
            updates[key] = MAX_MODEL_JOBS
    if updates:
        try:
            model.set_params(**updates)
        except Exception:
            pass
    return model


def transform_with(transforms, X: np.ndarray, n_bio: int):
    imp, scaler, pca, selector = transforms
    x_imp = imp.transform(X)
    x_emb = scaler.transform(x_imp[:, n_bio:])
    x_pca = pca.transform(x_emb)
    x_fold = np.hstack([x_imp[:, :n_bio], x_pca]).astype(np.float32)
    return selector.transform(x_fold)


def logit_mean(x: np.ndarray):
    p = np.clip(x, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def fit_weighted_cv_predict(X_train, y_train, w_train, targets: dict[str, np.ndarray], n_bio: int, seed_tag: str):
    oof_cols = []
    target_cols = {name: [] for name in targets}
    fold_rows = []

    for config in CONFIGS:
        seed = int(config["seed"])
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pos_weight = float((w_train[y_train == 0].sum()) / max(1e-6, w_train[y_train == 1].sum()))
        model_defs = full.opt.make_models(pos_weight, seed)
        oof = {name: np.zeros(len(y_train), dtype=np.float32) for name in model_defs}
        oof["mean_all"] = np.zeros(len(y_train), dtype=np.float32)
        oof["mean_tree"] = np.zeros(len(y_train), dtype=np.float32)
        target_fold = {name: {model_name: [] for model_name in PRED_COLS} for name in targets}

        for fold, (tr, va) in enumerate(folds.split(X_train, y_train), 1):
            Xtr, Xva, transforms = full.opt.make_fold_features(
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
                model = limit_model_jobs(clone(model_def))
                fit_with_optional_weight(model, Xtr, y_train[tr], w_train[tr])
                val_prob = model.predict_proba(Xva)[:, 1]
                oof[model_name][va] = val_prob
                val_preds[model_name] = val_prob
                for target_name, Xt in transformed_targets.items():
                    prob = model.predict_proba(Xt)[:, 1]
                    target_preds[target_name][model_name] = prob
                    target_fold[target_name][model_name].append(prob)

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
            fold_rows.append({"seed_tag": seed_tag, "config": config["name"], "fold": fold, "mean_all_auc": float(fold_auc)})
            print(f"{seed_tag} {config['name']} fold {fold}: mean_all_auc={fold_auc:.4f}", flush=True)

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


def binary_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
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


def best_threshold(y_true, prob):
    best = None
    for threshold in np.unique(np.r_[np.linspace(0.01, 0.99, 99), prob]):
        row = {"threshold": float(threshold), **binary_metrics(y_true, prob, float(threshold))}
        if best is None or (row["balanced_accuracy"], row["f1"]) > (best["balanced_accuracy"], best["f1"]):
            best = row
    return best


def summarize_scores(df: pd.DataFrame):
    rows = []
    for model, sub in df.groupby("model"):
        rows.append(
            {
                "model": model,
                "n_runs": int(len(sub)),
                "test_auc_mean": float(sub["test_auc"].mean()),
                "test_auc_std": float(sub["test_auc"].std(ddof=1)),
                "test_auc_min": float(sub["test_auc"].min()),
                "test_auc_max": float(sub["test_auc"].max()),
                "test_auprc_mean": float(sub["test_auprc"].mean()),
                "test_f1_mean": float(sub["test_f1"].mean()),
                "validation_auc_mean": float(sub["validation_auc"].mean()),
                "train_oof_auc_mean": float(sub["train_oof_auc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_auc_mean", "test_auprc_mean"], ascending=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_all, feature_meta, feature_names, n_numeric, feature_coverage, dropped_features = load_rice_species_specific_matrix()
    labels = pd.read_csv(LABELS, sep="\t").rename(columns={"final_label": "label"})
    labels["gene_id"] = labels["gene_id"].astype(str)
    feature_meta["matrix_row"] = np.arange(len(feature_meta), dtype=int)
    rice_meta = feature_meta.merge(labels, on="gene_id", how="inner").reset_index(drop=True)
    X = X_all[rice_meta["matrix_row"].to_numpy(dtype=int)].astype(np.float32)
    rice_meta = rice_meta.drop(columns=["matrix_row"])
    y = rice_meta["label"].astype(int).to_numpy(dtype=np.int8)
    weights = compute_sample_weights(rice_meta)

    pd.DataFrame({"feature_name": feature_names}).to_csv(OUT_DIR / "rice_species_specific_feature_names.tsv", sep="\t", index=False)
    feature_coverage.to_csv(OUT_DIR / "rice_species_specific_feature_source_coverage.tsv", sep="\t", index=False)
    dropped_features.to_csv(OUT_DIR / "rice_species_specific_dropped_features.tsv", sep="\t", index=False)
    rice_meta.assign(sample_weight=weights).to_csv(OUT_DIR / "rice_species_specific_labeled_genes.tsv", sep="\t", index=False)

    score_path = OUT_DIR / "rice_species_specific_repeated20_holdout_scores.tsv"
    fold_path = OUT_DIR / "rice_species_specific_repeated20_fold_scores.tsv"
    pred_path = OUT_DIR / "rice_species_specific_repeated20_test_predictions.tsv"
    all_scores = []
    all_folds = []
    all_test_predictions = []
    completed_seeds = set()
    if score_path.exists():
        previous_scores = pd.read_csv(score_path, sep="\t")
        all_scores = previous_scores.to_dict("records")
        completed = previous_scores.groupby("seed")["model"].nunique()
        completed_seeds = set(completed[completed >= 3].index.astype(int))
    if fold_path.exists():
        all_folds = pd.read_csv(fold_path, sep="\t").to_dict("records")
    if pred_path.exists():
        all_test_predictions = [pd.read_csv(pred_path, sep="\t")]

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"skip completed seed {seed}", flush=True)
            continue
        train_idx, val_idx, test_idx = split_80_10_10(y, seed)
        source_preds, target_preds, fold_rows = fit_weighted_cv_predict(
            X[train_idx],
            y[train_idx],
            weights[train_idx],
            {"validation": X[val_idx], "test": X[test_idx]},
            n_numeric,
            f"rice_species_seed_{seed}",
        )
        all_folds.extend(fold_rows)
        for model_name in ["meta", "mean", "logit_mean"]:
            val_prob = target_preds["validation"][model_name]
            test_prob = target_preds["test"][model_name]
            threshold = best_threshold(y[val_idx], val_prob)
            test_metrics = binary_metrics(y[test_idx], test_prob, threshold["threshold"])
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
                        "gene_id": rice_meta.iloc[test_idx]["gene_id"].to_numpy(),
                        "label": y[test_idx],
                        "probability": test_prob,
                        "threshold": threshold["threshold"],
                    }
                )
            )

        score_df = pd.DataFrame(all_scores)
        score_df.to_csv(score_path, sep="\t", index=False)
        summarize_scores(score_df).to_csv(OUT_DIR / "rice_species_specific_repeated20_holdout_summary.tsv", sep="\t", index=False)
        pd.DataFrame(all_folds).to_csv(fold_path, sep="\t", index=False)
        pd.concat(all_test_predictions, ignore_index=True).to_csv(
            pred_path, sep="\t", index=False
        )
        print(summarize_scores(score_df).to_string(index=False), flush=True)

    manifest = {
        "task": "rice species-specific essential-gene model, no Arabidopsis generalization evaluation",
        "label_file": str(LABELS),
        "label_rule": "strict high-confidence rice labels: remove conflicts; essential and nonessential require at least two concordant independent sources",
        "labeled_genes_with_features": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "all_rice_feature_genes": int(len(feature_meta)),
        "raw_feature_count_after_drop": int(X_all.shape[1]),
        "numeric_feature_count_before_pca": int(n_numeric),
        "plm_feature_count_before_pca": int(X_all.shape[1] - n_numeric),
        "seeds": SEEDS,
        "modeling": "20 repeated 80/10/10 holdouts; 5 CONFIGS x 5-fold OOF stacked ensemble per holdout; validation set selects classification threshold; test set is held out",
        "configs": CONFIGS,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
