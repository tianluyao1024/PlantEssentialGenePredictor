from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_rice_species_specific_strict_highconf_repeated as base


ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
COMMON_DIR = ROOT / "cross_species_ath_rice_common_features_models"
LABEL_DIR = ROOT / "rice_mutant_sources" / "processed_rap_es_combined"
NO_CONFLICT_LABELS = LABEL_DIR / "rap_no_conflict_binary_gene_labels.tsv"
STRICT_2SOURCE_LABELS = LABEL_DIR / "rap_strict_2source_binary_gene_labels.tsv"
OUT_DIR = COMMON_DIR / "rice_RAP_ES_no_conflict_species_model_repeated10"
SEEDS = list(range(20260670, 20260680))


def rap_from_transcript(transcript_id: object) -> str:
    text = str(transcript_id).split()[0]
    text = text.split("-")[0]
    if text.startswith("Os") and "t" in text[:5]:
        return text.replace("t", "g", 1)
    return ""


def load_rap_feature_matrix():
    X_all, loc_meta, feature_names, n_numeric, coverage, dropped = base.load_rice_species_specific_matrix()
    common = pd.read_csv(
        COMMON_DIR / "rice_cross_species_common_features_all_genes.tsv",
        sep="\t",
        usecols=["gene_id", "transcript_id"],
        dtype=str,
    ).drop_duplicates("gene_id", keep="first")
    common["rap_gene_id"] = common["transcript_id"].map(rap_from_transcript)
    loc_meta = loc_meta.copy()
    loc_meta["matrix_row"] = np.arange(len(loc_meta), dtype=int)
    rap_meta = loc_meta.merge(common, on="gene_id", how="inner")
    rap_meta = rap_meta[rap_meta["rap_gene_id"].astype(str).str.len().gt(0)].copy()
    rap_meta = rap_meta.sort_values(["rap_gene_id", "gene_id"]).drop_duplicates("rap_gene_id", keep="first")
    X_rap = X_all[rap_meta["matrix_row"].to_numpy(dtype=int)].astype(np.float32)
    rap_meta = rap_meta.rename(columns={"gene_id": "msu_gene_id"}).drop(columns=["matrix_row"]).reset_index(drop=True)
    return X_rap, rap_meta, feature_names, n_numeric, coverage, dropped


def run_dataset(label_path: Path, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_rap, rap_meta, feature_names, n_numeric, coverage, dropped = load_rap_feature_matrix()
    labels = pd.read_csv(label_path, sep="\t").rename(columns={"final_label": "label"})
    labels["rap_gene_id"] = labels["rap_gene_id"].astype(str)
    rap_meta = rap_meta.copy()
    rap_meta["matrix_row"] = np.arange(len(rap_meta), dtype=int)
    meta = rap_meta.merge(labels, on="rap_gene_id", how="inner").reset_index(drop=True)
    X = X_rap[meta["matrix_row"].to_numpy(dtype=int)].astype(np.float32)
    meta = meta.drop(columns=["matrix_row"])
    y = meta["label"].astype(int).to_numpy(dtype=np.int8)
    weights = base.compute_sample_weights(meta)

    pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / f"{tag}_feature_names.tsv", sep="\t", index=False)
    meta.assign(sample_weight=weights).to_csv(out_dir / f"{tag}_labeled_feature_intersection.tsv", sep="\t", index=False)
    coverage.to_csv(out_dir / f"{tag}_feature_source_coverage.tsv", sep="\t", index=False)
    dropped.to_csv(out_dir / f"{tag}_dropped_features.tsv", sep="\t", index=False)

    all_scores = []
    all_folds = []
    all_test_predictions = []
    for seed in SEEDS:
        train_idx, val_idx, test_idx = base.split_80_10_10(y, seed)
        source_preds, target_preds, fold_rows = base.fit_weighted_cv_predict(
            X[train_idx],
            y[train_idx],
            weights[train_idx],
            {"validation": X[val_idx], "test": X[test_idx]},
            n_numeric,
            f"{tag}_seed_{seed}",
        )
        all_folds.extend(fold_rows)
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
                    "train_oof_auc": float(base.roc_auc_score(y[train_idx], source_preds[model_name])),
                    "train_oof_auprc": float(base.average_precision_score(y[train_idx], source_preds[model_name])),
                    "validation_auc": float(base.roc_auc_score(y[val_idx], val_prob)),
                    "validation_auprc": float(base.average_precision_score(y[val_idx], val_prob)),
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
        score_df.to_csv(out_dir / f"{tag}_repeated10_holdout_scores.tsv", sep="\t", index=False)
        base.summarize_scores(score_df).to_csv(out_dir / f"{tag}_repeated10_holdout_summary.tsv", sep="\t", index=False)
        pd.DataFrame(all_folds).to_csv(out_dir / f"{tag}_repeated10_fold_scores.tsv", sep="\t", index=False)
        pd.concat(all_test_predictions, ignore_index=True).to_csv(
            out_dir / f"{tag}_repeated10_test_predictions.tsv", sep="\t", index=False
        )
        print(base.summarize_scores(score_df).to_string(index=False), flush=True)

    manifest = {
        "label_file": str(label_path),
        "label_rule": "RAP primary key; old Tos17 removed; new Tos17_RAP_ES uses ES=E^2/T^2 with conditional genes excluded from binary labels",
        "feature_rule": "Existing rice species-specific feature matrix reindexed to RAP by longest IRGSP transcript_id; feature row selected by RAP then first mapped MSU LOC",
        "rap_feature_genes": int(len(rap_meta)),
        "labeled_feature_intersection": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": int(n_numeric),
        "plm_feature_count": int(X.shape[1] - n_numeric),
        "seeds": SEEDS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    run_dataset(NO_CONFLICT_LABELS, OUT_DIR / "no_conflict_binary", "rap_es_no_conflict")
    run_dataset(STRICT_2SOURCE_LABELS, OUT_DIR / "strict_2source_binary", "rap_es_strict_2source")


if __name__ == "__main__":
    main()
