from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import train_ath_high_confidence_models as base
import train_ath_unknown20460_pseudo_train_validate3359_50_50 as paired


LABEL_ROOT = Path(
    "D:/拟南芥/模型/essential_gene_prediction_1623_plus_prediction_experiment_concordant"
)
OUT_ROOT = Path(
    "D:/拟南芥/模型/strict_vs_inclusive_paired_core1623_fixed80_10_10"
)
STRICT_LABELS = LABEL_ROOT / "high_confidence_1623_plus_concordant_strict_labels.tsv"
INCLUSIVE_LABELS = LABEL_ROOT / "high_confidence_1623_plus_concordant_inclusive_labels.tsv"
RANDOM_STATE = 20260619


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, sep="\t")
    labels["gene_id"] = labels["gene_id"].astype(str).str.upper()
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(np.int8)
    if labels["gene_id"].duplicated().any():
        raise RuntimeError(f"Duplicated genes in {path}")
    return labels


def make_or_load_core_split(core: pd.DataFrame) -> pd.DataFrame:
    path = OUT_ROOT / "shared_core1623_fixed80_10_10_split.tsv"
    if path.exists():
        split = pd.read_csv(path, sep="\t")
        split["gene_id"] = split["gene_id"].astype(str).str.upper()
        return split

    y = core["label"].to_numpy(np.int8)
    train_validation_idx, test_idx = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.10,
            random_state=RANDOM_STATE,
        ).split(core, y)
    )
    train_idx_local, validation_idx_local = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 / 9,
            random_state=RANDOM_STATE + 1,
        ).split(core.iloc[train_validation_idx], y[train_validation_idx])
    )
    train_idx = train_validation_idx[train_idx_local]
    validation_idx = train_validation_idx[validation_idx_local]

    split = core.copy()
    split["split"] = ""
    split.loc[train_idx, "split"] = "train"
    split.loc[validation_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(path, sep="\t", index=False)
    return split


def make_or_load_matched_additions(
    strict: pd.DataFrame,
    inclusive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_path = OUT_ROOT / "strict_matched_additions_978.tsv"
    inclusive_path = OUT_ROOT / "inclusive_matched_additions_978.tsv"
    if strict_path.exists() and inclusive_path.exists():
        return (
            pd.read_csv(strict_path, sep="\t"),
            pd.read_csv(inclusive_path, sep="\t"),
        )

    strict_added = strict[
        strict["label_source"].ne("original_consensus_2plus")
    ].copy()
    inclusive_added = inclusive[
        inclusive["label_source"].ne("original_consensus_2plus")
    ].copy()

    target_counts = strict_added["label"].value_counts().sort_index().to_dict()
    sampled_parts = []
    for label, count in sorted(target_counts.items()):
        candidates = inclusive_added[inclusive_added["label"].eq(label)].copy()
        if len(candidates) < count:
            raise RuntimeError(
                f"Inclusive candidates for label {label}: {len(candidates)} < {count}"
            )
        sampled_parts.append(
            candidates.sample(
                n=int(count),
                replace=False,
                random_state=RANDOM_STATE + 100 + int(label),
            )
        )
    inclusive_matched = (
        pd.concat(sampled_parts, ignore_index=True)
        .sort_values("gene_id")
        .reset_index(drop=True)
    )
    strict_added = strict_added.sort_values("gene_id").reset_index(drop=True)

    strict_counts = strict_added["label"].value_counts().sort_index().to_dict()
    inclusive_counts = inclusive_matched["label"].value_counts().sort_index().to_dict()
    if len(strict_added) != len(inclusive_matched) or strict_counts != inclusive_counts:
        raise RuntimeError(
            f"Matched addition mismatch: strict={strict_counts}, inclusive={inclusive_counts}"
        )

    strict_added.to_csv(strict_path, sep="\t", index=False)
    inclusive_matched.to_csv(inclusive_path, sep="\t", index=False)
    return strict_added, inclusive_matched


def index_for_genes(genes_all: np.ndarray, gene_ids: pd.Series) -> np.ndarray:
    index = {gene: idx for idx, gene in enumerate(genes_all)}
    missing = [gene for gene in gene_ids if gene not in index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} genes from feature matrix: {missing[:5]}")
    return np.array([index[gene] for gene in gene_ids], dtype=int)


def run_version(
    version: str,
    core_split: pd.DataFrame,
    additions: pd.DataFrame,
    X_all: np.ndarray,
    genes_all: np.ndarray,
    feature_names: list[str],
    n_bio: int,
) -> dict:
    out_dir = OUT_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    core_train = core_split[core_split["split"].eq("train")].copy()
    validation = core_split[core_split["split"].eq("validation")].copy()
    test = core_split[core_split["split"].eq("test")].copy()
    training_labels = pd.concat(
        [
            core_train.assign(training_component="core1623_train80"),
            additions.assign(training_component=f"{version}_matched_additions"),
        ],
        ignore_index=True,
    )
    if training_labels["gene_id"].duplicated().any():
        raise RuntimeError(f"{version}: duplicated training genes")

    train_idx = index_for_genes(genes_all, training_labels["gene_id"])
    validation_idx = index_for_genes(genes_all, validation["gene_id"])
    test_idx = index_for_genes(genes_all, test["gene_id"])
    X_train = X_all[train_idx].astype(np.float32)
    y_train = training_labels["label"].to_numpy(np.int8)
    X_validation = X_all[validation_idx].astype(np.float32)
    y_validation = validation["label"].to_numpy(np.int8)
    X_test = X_all[test_idx].astype(np.float32)
    y_test = test["label"].to_numpy(np.int8)

    training_labels.to_csv(out_dir / "training_labels.tsv", sep="\t", index=False)
    validation.to_csv(out_dir / "shared_validation_labels.tsv", sep="\t", index=False)
    test.to_csv(out_dir / "shared_test_labels.tsv", sep="\t", index=False)

    paired.OUT_DIR = out_dir
    validation_meta, test_meta, meta_names, model_paths = paired.fit_base_library(
        X_train,
        y_train,
        X_validation,
        X_test,
        n_bio,
    )
    np.save(out_dir / "validation_base_prediction_matrix.npy", validation_meta)
    np.save(out_dir / "test_base_prediction_matrix.npy", test_meta)
    pd.DataFrame({"meta_feature_name": meta_names}).to_csv(
        out_dir / "base_prediction_feature_names.tsv",
        sep="\t",
        index=False,
    )

    result = paired.evaluate_candidates(
        validation_meta,
        test_meta,
        meta_names,
        y_validation,
        y_test,
    )
    result["candidates"].to_csv(
        out_dir / "validation_model_selection_candidates.tsv",
        sep="\t",
        index=False,
    )
    result["stack_search"].to_csv(
        out_dir / "validation_stacking_C_search.tsv",
        sep="\t",
        index=False,
    )

    selected_threshold = float(result["selected"]["validation_threshold"])
    validation_predictions = validation.copy()
    validation_predictions["probability"] = result["validation_probability"]
    validation_predictions["classification_threshold"] = selected_threshold
    validation_predictions["predicted_label"] = (
        validation_predictions["probability"] >= selected_threshold
    ).astype(np.int8)
    validation_predictions.to_csv(
        out_dir / "shared_validation_predictions.tsv",
        sep="\t",
        index=False,
    )

    test_predictions = test.copy()
    test_predictions["probability"] = result["test_probability"]
    test_predictions["classification_threshold"] = selected_threshold
    test_predictions["predicted_label"] = (
        test_predictions["probability"] >= selected_threshold
    ).astype(np.int8)
    test_predictions.to_csv(
        out_dir / "shared_test_predictions.tsv",
        sep="\t",
        index=False,
    )

    score = {
        "version": version,
        "selected_candidate": result["selected"]["candidate"],
        "selected_candidate_type": result["selected"]["candidate_type"],
        "classification_threshold_selected_on_validation": selected_threshold,
        "n_train": int(len(y_train)),
        "train_essential": int(y_train.sum()),
        "train_nonessential": int((y_train == 0).sum()),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        **{
            f"validation_{key}": value
            for key, value in result["validation_metrics"].items()
        },
        **{f"test_{key}": value for key, value in result["test_metrics"].items()},
    }
    pd.DataFrame([score]).to_csv(
        out_dir / "final_shared_validation_test_scores.tsv",
        sep="\t",
        index=False,
    )
    joblib.dump(
        {
            "version": version,
            "selected_candidate": result["selected"],
            "stacking_model": result["stack_model"],
            "stacking_best_configuration": result["stack_best"],
            "base_prediction_feature_names": meta_names,
            "base_model_bundle_paths": model_paths,
            "feature_names": feature_names,
            "n_bio": n_bio,
        },
        out_dir / "selected_model_and_manifest.joblib",
        compress=3,
    )
    return score


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    strict = load_labels(STRICT_LABELS)
    inclusive = load_labels(INCLUSIVE_LABELS)

    strict_core = strict[
        strict["label_source"].eq("original_consensus_2plus")
    ].copy()
    inclusive_core = inclusive[
        inclusive["label_source"].eq("original_consensus_2plus")
    ].copy()
    if len(strict_core) != 1623:
        raise RuntimeError(f"Expected 1623 core genes, got {len(strict_core)}")
    if not strict_core[["gene_id", "label"]].sort_values("gene_id").reset_index(
        drop=True
    ).equals(
        inclusive_core[["gene_id", "label"]]
        .sort_values("gene_id")
        .reset_index(drop=True)
    ):
        raise RuntimeError("Strict and inclusive core 1623 labels differ")

    core_split = make_or_load_core_split(strict_core.reset_index(drop=True))
    strict_added, inclusive_added = make_or_load_matched_additions(strict, inclusive)

    X_all, _ids_all, genes_all, feature_names, n_bio = base.load_matrix()
    genes_all = np.array([str(gene).upper() for gene in genes_all])

    scores = []
    scores.append(
        run_version(
            "strict_matched",
            core_split,
            strict_added,
            X_all,
            genes_all,
            feature_names,
            n_bio,
        )
    )
    scores.append(
        run_version(
            "inclusive_matched_to_strict_counts",
            core_split,
            inclusive_added,
            X_all,
            genes_all,
            feature_names,
            n_bio,
        )
    )
    comparison = pd.DataFrame(scores).sort_values("test_auc", ascending=False)
    comparison.to_csv(
        OUT_ROOT / "strict_vs_inclusive_shared_test_comparison.tsv",
        sep="\t",
        index=False,
    )

    conflict_composition = (
        inclusive_added.groupby(["label", "experimental_has_conflict"])
        .size()
        .reset_index(name="gene_count")
    )
    conflict_composition.to_csv(
        OUT_ROOT / "inclusive_matched_conflict_composition.tsv",
        sep="\t",
        index=False,
    )

    split_summary = (
        core_split.groupby(["split", "label"])
        .size()
        .reset_index(name="gene_count")
    )
    split_summary.to_csv(
        OUT_ROOT / "shared_core_split_summary.tsv",
        sep="\t",
        index=False,
    )
    manifest = {
        "comparison_design": (
            "Both versions use the same core-1623 train/validation/test genes. "
            "Inclusive additions are downsampled to the strict addition counts by class."
        ),
        "core_split": "80% train / 10% validation / 10% test, stratified",
        "random_state": RANDOM_STATE,
        "core_total": int(len(core_split)),
        "strict_additions": int(len(strict_added)),
        "inclusive_matched_additions": int(len(inclusive_added)),
        "strict_addition_counts": {
            str(key): int(value)
            for key, value in strict_added["label"].value_counts().sort_index().items()
        },
        "inclusive_addition_counts": {
            str(key): int(value)
            for key, value in inclusive_added["label"].value_counts().sort_index().items()
        },
        "feature_count": int(len(feature_names)),
        "bio_feature_count": int(n_bio),
        "plm_feature_count": int(len(feature_names) - n_bio),
        "scores": scores,
        "caution": (
            "The added genes were selected using a teacher trained from the core 1623. "
            "This is a paired internal comparison, not a fully independent external test."
        ),
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
