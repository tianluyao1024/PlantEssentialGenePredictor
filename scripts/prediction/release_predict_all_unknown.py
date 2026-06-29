from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_joint_ath2601_rice_strict399_common6751 as joint
import train_ath_unknown20460_pseudo_train_validate3359_50_50 as ath_stack
import train_pseudo06_predict_all_longest_unknown as ath_unknown
import train_rice_E0_Nge6_common6751_fixed80_10_10 as rice_common
import train_rice_rap_es_robust_lite_models as robust


RELEASE_ROOT = Path("E:/PlantEssentialGenePredictor")
PRED_DIR = RELEASE_ROOT / "predictions"
FEATURE_DIR = RELEASE_ROOT / "data" / "processed_features"
MODEL_DIR = RELEASE_ROOT / "models"

ATH_SINGLE_MODEL = Path(
    "D:/拟南芥/模型/ath_three_labelsets_common6751_fixed_core1623_80_10_10/"
    "strict2601_common6751/selected_model_and_manifest.joblib"
)
RICE_SINGLE_MODEL = Path(
    "E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models/"
    "joint_ath2601_rice_strict399_common6751/"
    "rice_strict399_N4_common6751_baseline/model.joblib"
)
JOINT_MODEL = Path(
    "E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models/"
    "joint_ath2601_rice_strict399_common6751/"
    "joint_common6751_stacking/model.joblib"
)
JOINT_MANIFEST = Path(
    "E:/CodexMoved/Desktop/水稻/cross_species_ath_rice_common_features_models/"
    "joint_ath2601_rice_strict399_common6751/manifest.json"
)

ATH_SINGLE_THRESHOLD = 0.3
RICE_SINGLE_THRESHOLD = 0.4015871584415436
JOINT_THRESHOLD = 0.45597791914222807


def ensure_dirs() -> None:
    for path in [PRED_DIR, FEATURE_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def patch_sklearn_compat(obj):
    """Patch small sklearn 1.8 -> 1.7 pickle compatibility gaps in-place."""
    if obj is None:
        return obj
    if obj.__class__.__name__ == "LogisticRegression" and not hasattr(
        obj, "multi_class"
    ):
        obj.multi_class = "auto"
    if isinstance(obj, dict):
        for value in obj.values():
            patch_sklearn_compat(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            patch_sklearn_compat(value)
    else:
        for attr in ["steps", "models", "named_steps"]:
            if hasattr(obj, attr):
                patch_sklearn_compat(getattr(obj, attr))
    return obj


def deploy_common_model(model_path: Path, x: np.ndarray) -> dict[str, np.ndarray]:
    """Deploy rice/joint common6751 package saved by fit_common_library."""
    package = patch_sklearn_compat(joblib.load(model_path))
    columns = []
    expected_names = []
    for bundle in package["deployment_base_models"]:
        transformed = rice_common.transform_with(
            bundle["transforms"], x, int(package["n_bio"])
        )
        model_predictions = {}
        for model_name, model in bundle["models"].items():
            model_predictions[model_name] = model.predict_proba(transformed)[:, 1]
        model_predictions["mean_all"] = np.mean(
            [model_predictions[name] for name in bundle["models"]], axis=0
        )
        for model_name in robust.PRED_COLS:
            columns.append(model_predictions[model_name])
            expected_names.append(f"{bundle['config']['name']}__{model_name}")
    if expected_names != list(package["meta_feature_names"]):
        raise RuntimeError("Meta feature order mismatch while deploying common model")
    meta = np.column_stack(columns).astype(np.float32)
    return {
        "meta": package["meta_model"].predict_proba(meta)[:, 1].astype(float),
        "mean": meta.mean(axis=1).astype(float),
        "logit_mean": robust.logit_mean(meta).astype(float),
    }


def deploy_ath_single_model(x: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Deploy selected Arabidopsis strict2601 common6751 model."""
    package = patch_sklearn_compat(joblib.load(ATH_SINGLE_MODEL))
    columns = []
    for path in package["base_model_bundle_paths"]:
        bundle = patch_sklearn_compat(joblib.load(path))
        imputer = bundle["transforms"][0]
        if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
            imputer._fill_dtype = imputer._fit_dtype
        transformed = ath_stack.transform_with(
            bundle["transforms"], x, int(bundle["n_bio"])
        )
        by_model = {
            name: model.predict_proba(transformed)[:, 1]
            for name, model in bundle["models"].items()
        }
        by_model["mean_all"] = np.mean(list(by_model.values()), axis=0)
        tree_names = [
            "extra_sqrt",
            "extra_log2",
            "rf_sqrt",
            "lgbm_gbdt",
            "xgb_depth3",
        ]
        by_model["mean_tree"] = np.mean([by_model[name] for name in tree_names], axis=0)
        for short_name in [
            "extra_sqrt",
            "extra_log2",
            "rf_sqrt",
            "lgbm_gbdt",
            "xgb_depth3",
            "logistic",
            "mean_all",
            "mean_tree",
        ]:
            columns.append(by_model[short_name])
    meta = np.column_stack(columns).astype(np.float32)
    selected = package["selected_candidate"]
    candidate = selected["candidate"]
    if candidate == "global_mean_all_40_columns":
        probability = meta.mean(axis=1)
    elif candidate == "global_mean_six_base_models":
        names = package["base_prediction_feature_names"]
        idx = [
            i
            for i, value in enumerate(names)
            if value.rsplit("__", 1)[-1]
            in {
                "extra_sqrt",
                "extra_log2",
                "rf_sqrt",
                "lgbm_gbdt",
                "xgb_depth3",
                "logistic",
            }
        ]
        probability = meta[:, idx].mean(axis=1)
    elif candidate == "logistic_stacking_5fold_validation_oof":
        probability = package["stacking_model"].predict_proba(meta)[:, 1]
    else:
        names = package["base_prediction_feature_names"]
        if candidate not in names:
            raise RuntimeError(f"Unsupported ATH selected candidate: {candidate}")
        probability = meta[:, names.index(candidate)]
    return (
        probability.astype(float),
        float(selected["validation_threshold"]),
        str(candidate),
    )


def get_known_gene_set(parts: dict[str, pd.DataFrame]) -> set[str]:
    genes: set[str] = set()
    for part in parts.values():
        genes.update(part["gene_id"].astype(str))
    return genes


def write_prediction_table(
    out: Path,
    genes: np.ndarray | list[str],
    probability: np.ndarray,
    threshold: float,
    species: str,
    model_name: str,
    transcript_ids: np.ndarray | list[str] | None = None,
    known_genes: set[str] | None = None,
    sequence_ids: np.ndarray | list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "species": species,
            "gene_id": np.asarray(genes).astype(str),
            "essential_probability": probability.astype(float),
            "classification_threshold": float(threshold),
        }
    )
    if transcript_ids is not None:
        frame.insert(2, "transcript_id", np.asarray(transcript_ids).astype(str))
    if sequence_ids is not None:
        frame.insert(2, "sequence_id", np.asarray(sequence_ids).astype(str))
    frame["predicted_label"] = (frame["essential_probability"] >= threshold).astype(int)
    frame["predicted_class"] = np.where(
        frame["predicted_label"].eq(1), "essential", "nonessential"
    )
    if known_genes is not None:
        frame["label_status"] = np.where(
            frame["gene_id"].isin(known_genes), "known_label_used_in_study", "unknown"
        )
    else:
        frame["label_status"] = "unknown"
    frame["model_name"] = model_name
    frame.sort_values(
        ["essential_probability", "gene_id"], ascending=[False, True]
    ).to_csv(out, sep="\t", index=False)
    return frame


def write_minimal_feature_package(
    rice_meta: pd.DataFrame,
    rice_x: np.ndarray,
    rice_names: list[str],
    ath_unknown_genes: np.ndarray,
    ath_unknown_ids: np.ndarray,
    ath_unknown_x: np.ndarray,
    ath_common_names: list[str],
) -> None:
    """Save compressed processed features for GitHub/Zenodo release.

    Full 6751-column TSV files would be unwieldy. The release keeps numerical
    matrices as compressed NumPy arrays plus gene metadata and feature names.
    """
    np.savez_compressed(
        FEATURE_DIR / "rice_common6751_all_genes.npz",
        X=rice_x.astype(np.float32),
        gene_id=rice_meta["gene_id"].astype(str).to_numpy(),
        transcript_id=rice_meta.get("transcript_id", pd.Series([""] * len(rice_meta))).astype(str).to_numpy(),
        feature_names=np.asarray(rice_names, dtype=object),
        n_bio=np.array([95], dtype=np.int16),
    )
    np.savez_compressed(
        FEATURE_DIR / "arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz",
        X=ath_unknown_x.astype(np.float32),
        gene_id=ath_unknown_genes.astype(str),
        sequence_id=ath_unknown_ids.astype(str),
        feature_names=np.asarray(ath_common_names, dtype=object),
        n_bio=np.array([95], dtype=np.int16),
    )
    pd.DataFrame({"feature_name": rice_names}).to_csv(
        FEATURE_DIR / "common6751_feature_names.tsv", sep="\t", index=False
    )


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, frame in tables.items():
        rows.append(
            {
                "prediction_file": name,
                "species": frame["species"].iloc[0],
                "model_name": frame["model_name"].iloc[0],
                "total_genes": int(len(frame)),
                "predicted_essential": int(frame["predicted_label"].sum()),
                "predicted_nonessential": int((frame["predicted_label"] == 0).sum()),
                "threshold": float(frame["classification_threshold"].iloc[0]),
                "mean_probability": float(frame["essential_probability"].mean()),
                "median_probability": float(frame["essential_probability"].median()),
            }
        )
    pd.DataFrame(rows).to_csv(PRED_DIR / "prediction_summary.tsv", sep="\t", index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("Loading rice common6751 matrix")
    rice_x, rice_names, rice_meta, rice_split = joint.load_rice()
    rice_known = get_known_gene_set(rice_split)
    if rice_x.shape[1] != 6751 or len(rice_names) != 6751:
        raise RuntimeError(f"Rice matrix shape mismatch: {rice_x.shape}")

    print("Loading Arabidopsis unknown20460 common6751-like matrix")
    ath_common_names = pd.read_csv(
        "D:/拟南芥/模型/ath_three_labelsets_common6751_fixed_core1623_80_10_10/"
        "common6751_feature_names.tsv",
        sep="\t",
    )["feature_name"].astype(str).tolist()
    ath_unknown_x, ath_unknown_ids, ath_unknown_genes = ath_unknown.build_unknown_matrix(
        ath_common_names, 95
    )
    if ath_unknown_x.shape[1] != 6751:
        raise RuntimeError(f"Arabidopsis unknown matrix shape mismatch: {ath_unknown_x.shape}")

    print("Deploying rice single-species model on all rice genes")
    rice_single = deploy_common_model(RICE_SINGLE_MODEL, rice_x)["logit_mean"]
    print("Deploying joint model on all rice genes")
    rice_joint = deploy_common_model(JOINT_MODEL, rice_x)["meta"]

    print("Deploying Arabidopsis single-species model on unknown20460")
    ath_single, ath_threshold, ath_candidate = deploy_ath_single_model(ath_unknown_x)
    print(f"Arabidopsis single candidate={ath_candidate} threshold={ath_threshold}")
    print("Deploying joint model on Arabidopsis unknown20460")
    ath_joint = deploy_common_model(JOINT_MODEL, ath_unknown_x)["meta"]

    tables = {}
    tables["rice_unknown_all_single_model_predictions.tsv"] = write_prediction_table(
        PRED_DIR / "rice_unknown_all_single_model_predictions.tsv",
        rice_meta["gene_id"].astype(str).to_numpy(),
        rice_single,
        RICE_SINGLE_THRESHOLD,
        "rice",
        "rice_single_strict399_Tos17N4_common6751_logit_mean",
        transcript_ids=rice_meta["transcript_id"].astype(str).to_numpy()
        if "transcript_id" in rice_meta
        else None,
        known_genes=rice_known,
    )
    tables["rice_unknown_all_joint_model_predictions.tsv"] = write_prediction_table(
        PRED_DIR / "rice_unknown_all_joint_model_predictions.tsv",
        rice_meta["gene_id"].astype(str).to_numpy(),
        rice_joint,
        JOINT_THRESHOLD,
        "rice",
        "joint_arabidopsis_rice_common6751_meta",
        transcript_ids=rice_meta["transcript_id"].astype(str).to_numpy()
        if "transcript_id" in rice_meta
        else None,
        known_genes=rice_known,
    )
    tables["arabidopsis_unknown20460_single_model_predictions.tsv"] = write_prediction_table(
        PRED_DIR / "arabidopsis_unknown20460_single_model_predictions.tsv",
        ath_unknown_genes,
        ath_single,
        ATH_SINGLE_THRESHOLD,
        "arabidopsis",
        "arabidopsis_single_strict2601_common6751_global_mean",
        sequence_ids=ath_unknown_ids,
    )
    tables["arabidopsis_unknown20460_joint_model_predictions.tsv"] = write_prediction_table(
        PRED_DIR / "arabidopsis_unknown20460_joint_model_predictions.tsv",
        ath_unknown_genes,
        ath_joint,
        JOINT_THRESHOLD,
        "arabidopsis",
        "joint_arabidopsis_rice_common6751_meta",
        sequence_ids=ath_unknown_ids,
    )

    print("Writing compressed processed feature package")
    write_minimal_feature_package(
        rice_meta,
        rice_x,
        rice_names,
        ath_unknown_genes,
        ath_unknown_ids,
        ath_unknown_x,
        ath_common_names,
    )
    write_summary(tables)

    manifest = {
        "release_root": str(RELEASE_ROOT),
        "feature_space": "common6751 = 95 shared biological features + 6656 PLM embeddings",
        "arabidopsis_unknown_note": (
            "Arabidopsis unknown20460 matrices contain sequence-derived biological "
            "features plus PLM embeddings; annotation-derived biological features "
            "that were unavailable for unknown genes are left as missing values for "
            "the trained model imputers."
        ),
        "rice_total_genes_predicted": int(len(rice_meta)),
        "rice_known_label_genes": int(len(rice_known)),
        "arabidopsis_unknown_genes_predicted": int(len(ath_unknown_genes)),
        "thresholds": {
            "arabidopsis_single": ATH_SINGLE_THRESHOLD,
            "rice_single": RICE_SINGLE_THRESHOLD,
            "joint": JOINT_THRESHOLD,
        },
        "model_paths_used": {
            "arabidopsis_single": str(ATH_SINGLE_MODEL),
            "rice_single": str(RICE_SINGLE_MODEL),
            "joint": str(JOINT_MODEL),
            "joint_manifest": str(JOINT_MANIFEST),
        },
        "outputs": {
            "prediction_summary": str(PRED_DIR / "prediction_summary.tsv"),
            "predictions": list(tables.keys()),
            "features": [
                "data/processed_features/rice_common6751_all_genes.npz",
                "data/processed_features/arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz",
                "data/processed_features/common6751_feature_names.tsv",
            ],
        },
    }
    (RELEASE_ROOT / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
