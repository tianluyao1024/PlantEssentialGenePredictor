from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_rice_E0_Nge6_common6751_fixed80_10_10 as n6
import train_rice_internal_enhanced_replacement_v2 as feature_loader
import train_rice_species_specific_strict_highconf_repeated as metric_utils


ROOT = Path("E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models")
BASELINE_DIR = ROOT / "rice_E_hq_plus_N_E0_Nge6_common6751_fixed80_10_10"
OUT_ROOT = ROOT / "rice_N6_four_feature_variants_fixed_split"
FIXED_SPLIT = BASELINE_DIR / "fixed80_10_10_split_labels.tsv"
ATH_HOMOLOG = Path(
    "D:/拟南芥/文献特征复现/downloads/ensembl_plants_selected_homologs.tsv"
)
ATH_PREDICTIONS = Path(
    "D:/拟南芥/模型/essential_gene_prediction_consensus1623_plus_pseudo06_predict_all_unknown/"
    "all_unknown_longest_gene_predictions.tsv"
)
RICE_HOMOLOG_COLUMN = "Oryza sativa Japonica Group gene stable ID"
DROP_DIRECTIONAL = {
    "rice_homolog_found",
    "rice_homolog_percent_identity",
    "homolog_not_found_in_rice",
}


def load_ath_prior(rice_genes: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict]:
    predictions = pd.read_csv(ATH_PREDICTIONS, sep="\t", dtype=str)
    predictions["gene_id"] = predictions["gene_id"].astype(str).str.upper()
    predictions["final_mean_probability"] = pd.to_numeric(
        predictions["final_mean_probability"], errors="raise"
    )
    ath_probability = dict(
        zip(predictions["gene_id"], predictions["final_mean_probability"])
    )

    homolog = pd.read_csv(
        ATH_HOMOLOG,
        sep="\t",
        usecols=["Gene stable ID", RICE_HOMOLOG_COLUMN],
        dtype=str,
    ).fillna("")
    homolog["Gene stable ID"] = homolog["Gene stable ID"].str.strip().str.upper()
    homolog[RICE_HOMOLOG_COLUMN] = homolog[RICE_HOMOLOG_COLUMN].str.strip()
    homolog = homolog[
        homolog["Gene stable ID"].ne("")
        & homolog[RICE_HOMOLOG_COLUMN].ne("")
    ].copy()

    # Reproduce the existing Mimocode prior definition.
    ath_to_first_rice: dict[str, str] = {}
    for ath_gene, rice_gene in homolog[
        ["Gene stable ID", RICE_HOMOLOG_COLUMN]
    ].itertuples(index=False):
        if ath_gene not in ath_to_first_rice:
            ath_to_first_rice[ath_gene] = rice_gene

    rice_to_probability: dict[str, float] = {}
    rice_to_ath: dict[str, str] = {}
    for ath_gene, rice_gene in ath_to_first_rice.items():
        if ath_gene in ath_probability:
            rice_to_probability[rice_gene] = float(ath_probability[ath_gene])
            rice_to_ath[rice_gene] = ath_gene

    rows = []
    prior = np.full(len(rice_genes), 0.5, dtype=np.float32)
    for idx, rice_gene in enumerate(rice_genes):
        mapped = rice_gene in rice_to_probability
        if mapped:
            prior[idx] = rice_to_probability[rice_gene]
        rows.append(
            {
                "gene_id": rice_gene,
                "ath_prior_probability": float(prior[idx]),
                "has_ath_teacher_prior": int(mapped),
                "mapped_ath_gene": rice_to_ath.get(rice_gene, ""),
            }
        )
    mapping = pd.DataFrame(rows)
    summary = {
        "rice_genes_total": int(len(rice_genes)),
        "rice_genes_with_ath_prior": int(mapping["has_ath_teacher_prior"].sum()),
        "rice_genes_default_0.5": int(
            (mapping["has_ath_teacher_prior"] == 0).sum()
        ),
        "coverage_fraction": float(mapping["has_ath_teacher_prior"].mean()),
        "prior_min": float(prior.min()),
        "prior_mean": float(prior.mean()),
        "prior_max": float(prior.max()),
    }
    return prior, mapping, summary


def load_matrices():
    X_6754, meta, names_6754, n_bio_6754, coverage, status = (
        feature_loader.load_feature_matrix()
    )
    if (n_bio_6754, len(names_6754) - n_bio_6754, len(names_6754)) != (
        98,
        6656,
        6754,
    ):
        raise RuntimeError(
            f"Expected rice matrix 98+6656=6754, got "
            f"{n_bio_6754}+{len(names_6754)-n_bio_6754}={len(names_6754)}"
        )
    name_to_idx = {name: idx for idx, name in enumerate(names_6754)}
    missing = sorted(DROP_DIRECTIONAL - set(name_to_idx))
    if missing:
        raise RuntimeError(f"Missing directional features: {missing}")

    names_6751 = [name for name in names_6754 if name not in DROP_DIRECTIONAL]
    idx_6751 = [name_to_idx[name] for name in names_6751]
    X_6751 = X_6754[:, idx_6751].astype(np.float32)
    if X_6751.shape[1] != 6751:
        raise RuntimeError(f"Expected 6751 matrix, got {X_6751.shape}")

    genes = meta["gene_id"].astype(str).to_numpy()
    prior, prior_mapping, prior_summary = load_ath_prior(genes)
    X_6752 = np.column_stack([X_6751[:, :95], prior, X_6751[:, 95:]]).astype(
        np.float32
    )
    names_6752 = names_6751[:95] + ["ath_teacher_prior_probability"] + names_6751[95:]
    X_6755 = np.column_stack(
        [X_6754[:, :98], prior, X_6754[:, 98:]]
    ).astype(np.float32)
    names_6755 = names_6754[:98] + ["ath_teacher_prior_probability"] + names_6754[98:]

    variants = {
        "common6751": (X_6751, names_6751, 95),
        "full6754": (X_6754.astype(np.float32), names_6754, 98),
        "common6752_plus_ath_prior": (X_6752, names_6752, 96),
        "full6755_plus_ath_prior": (X_6755, names_6755, 99),
    }
    for name, (matrix, feature_names, n_bio) in variants.items():
        if matrix.shape[1] != len(feature_names):
            raise RuntimeError(f"{name}: matrix/name mismatch")
        expected = {
            "common6751": (6751, 95),
            "full6754": (6754, 98),
            "common6752_plus_ath_prior": (6752, 96),
            "full6755_plus_ath_prior": (6755, 99),
        }[name]
        if (matrix.shape[1], n_bio) != expected:
            raise RuntimeError(
                f"{name}: expected dimensions {expected}, got "
                f"{matrix.shape[1], n_bio}"
            )
    return variants, meta.copy(), prior_mapping, prior_summary, coverage, status


def load_fixed_indices(meta: pd.DataFrame):
    split = pd.read_csv(FIXED_SPLIT, sep="\t")
    split["gene_id"] = split["gene_id"].astype(str)
    split["label"] = pd.to_numeric(split["label"], errors="raise").astype(np.int8)
    meta = meta.copy()
    meta["matrix_row"] = np.arange(len(meta), dtype=int)
    aligned = split.merge(
        meta[["gene_id", "matrix_row"]],
        on="gene_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["matrix_row"].isna().any():
        missing = aligned.loc[aligned["matrix_row"].isna(), "gene_id"].tolist()
        raise RuntimeError(f"Fixed split genes missing from matrix: {missing[:5]}")
    indices = {}
    for split_name in ["train", "validation", "test"]:
        part = aligned[aligned["split"].eq(split_name)]
        indices[split_name] = part["matrix_row"].to_numpy(int)
    expected = {"train": 810, "validation": 102, "test": 102}
    observed = {name: len(idx) for name, idx in indices.items()}
    if observed != expected:
        raise RuntimeError(f"Expected fixed split {expected}, got {observed}")
    return aligned, indices


def train_variant(
    name: str,
    matrix: np.ndarray,
    feature_names: list[str],
    n_bio: int,
    split: pd.DataFrame,
    indices: dict[str, np.ndarray],
):
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    y_all = split["label"].to_numpy(np.int8)
    # split is ordered independently from the full matrix, so labels are selected by split rows.
    y_train = split.loc[split["split"].eq("train"), "label"].to_numpy(np.int8)
    y_validation = split.loc[
        split["split"].eq("validation"), "label"
    ].to_numpy(np.int8)
    y_test = split.loc[split["split"].eq("test"), "label"].to_numpy(np.int8)
    X_train = matrix[indices["train"]]
    X_validation = matrix[indices["validation"]]
    X_test = matrix[indices["test"]]
    weights = np.ones(len(y_train), dtype=np.float32)

    (
        _oof_predictions,
        target_predictions,
        fold_rows,
        meta_model,
        deployment_models,
        meta_feature_names,
    ) = n6.fit_library(
        X_train,
        y_train,
        weights,
        {"validation": X_validation, "test": X_test},
        n_bio,
    )

    candidate_rows = []
    for model_name in ["meta", "mean", "logit_mean"]:
        validation_probability = target_predictions["validation"][model_name]
        test_probability = target_predictions["test"][model_name]
        threshold = metric_utils.best_threshold(
            y_validation, validation_probability
        )
        validation_metrics = metric_utils.binary_metrics(
            y_validation,
            validation_probability,
            threshold["threshold"],
        )
        test_metrics = metric_utils.binary_metrics(
            y_test,
            test_probability,
            threshold["threshold"],
        )
        candidate_rows.append(
            {
                "variant": name,
                "model": model_name,
                "validation_threshold": threshold["threshold"],
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["validation_auc", "validation_auprc"], ascending=False
    )
    best = candidates.iloc[0].to_dict()
    best_name = str(best["model"])
    threshold = float(best["validation_threshold"])

    candidates.to_csv(
        out_dir / "validation_model_selection.tsv", sep="\t", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        out_dir / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        out_dir / "feature_names.tsv", sep="\t", index=False
    )
    pd.DataFrame({"meta_feature_name": meta_feature_names}).to_csv(
        out_dir / "meta_feature_names.tsv", sep="\t", index=False
    )

    for split_name, probabilities, labels in [
        ("validation", target_predictions["validation"][best_name], y_validation),
        ("test", target_predictions["test"][best_name], y_test),
    ]:
        rows = split[split["split"].eq(split_name)].copy()
        rows["probability"] = probabilities
        rows["threshold"] = threshold
        rows["predicted_label"] = (probabilities >= threshold).astype(np.int8)
        rows.to_csv(
            out_dir / f"{split_name}_predictions.tsv", sep="\t", index=False
        )

    joblib.dump(
        {
            "variant": name,
            "selected_prediction_method": best_name,
            "classification_threshold": threshold,
            "meta_model": meta_model,
            "deployment_base_models": deployment_models,
            "meta_feature_names": meta_feature_names,
            "feature_names": feature_names,
            "n_bio": n_bio,
        },
        out_dir / "final_model.joblib",
        compress=3,
    )
    pd.DataFrame([best]).to_csv(
        out_dir / "final_fixed_test_score.tsv", sep="\t", index=False
    )
    return best


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    variants, meta, prior_mapping, prior_summary, coverage, status = load_matrices()
    split, indices = load_fixed_indices(meta)
    split.to_csv(OUT_ROOT / "shared_fixed_split_1014.tsv", sep="\t", index=False)
    prior_mapping.to_csv(
        OUT_ROOT / "rice_all_gene_ath_teacher_prior_mapping.tsv",
        sep="\t",
        index=False,
    )
    prior_mapping[
        prior_mapping["gene_id"].isin(set(split["gene_id"]))
    ].to_csv(
        OUT_ROOT / "rice_N6_1014_ath_teacher_prior_mapping.tsv",
        sep="\t",
        index=False,
    )

    scores = []
    for name, (matrix, feature_names, n_bio) in variants.items():
        existing_score = (
            BASELINE_DIR / "validation_model_selection.tsv"
            if name == "common6751"
            else None
        )
        # Reuse the existing 6751 result because it used this exact split and fit_library.
        if name == "common6751" and existing_score and existing_score.exists():
            candidates = pd.read_csv(existing_score, sep="\t").sort_values(
                ["validation_auc", "validation_auprc"], ascending=False
            )
            best = candidates.iloc[0].to_dict()
            best["variant"] = name
            out_dir = OUT_ROOT / name
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([best]).to_csv(
                out_dir / "final_fixed_test_score.tsv", sep="\t", index=False
            )
            pd.DataFrame({"feature_name": feature_names}).to_csv(
                out_dir / "feature_names.tsv", sep="\t", index=False
            )
            print("reused exact common6751 baseline result", flush=True)
        else:
            best = train_variant(
                name,
                matrix,
                feature_names,
                n_bio,
                split,
                indices,
            )
        scores.append(best)

    comparison = pd.DataFrame(scores).sort_values(
        ["test_auc", "test_auprc"], ascending=False
    )
    comparison.to_csv(
        OUT_ROOT / "four_feature_variants_shared_test_comparison.tsv",
        sep="\t",
        index=False,
    )
    n6_prior = prior_mapping[
        prior_mapping["gene_id"].isin(set(split["gene_id"]))
    ]
    manifest = {
        "fixed_split_source": str(FIXED_SPLIT),
        "same_train_validation_test_genes_for_all_variants": True,
        "train_n": 810,
        "validation_n": 102,
        "test_n": 102,
        "variants": {
            name: {
                "feature_count": len(feature_names),
                "bio_feature_count": n_bio,
                "plm_feature_count": len(feature_names) - n_bio,
            }
            for name, (_matrix, feature_names, n_bio) in variants.items()
        },
        "ath_prior_definition": (
            "Mimocode-compatible: map Arabidopsis genes to the first listed rice "
            "homolog, use final_mean_probability from the 20460 Arabidopsis teacher "
            "prediction table, and assign 0.5 when unavailable"
        ),
        "ath_prior_all_rice_coverage": prior_summary,
        "ath_prior_N6_1014_coverage": {
            "genes": int(len(n6_prior)),
            "mapped": int(n6_prior["has_ath_teacher_prior"].sum()),
            "default_0.5": int((n6_prior["has_ath_teacher_prior"] == 0).sum()),
            "coverage_fraction": float(n6_prior["has_ath_teacher_prior"].mean()),
        },
        "scores": scores,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
