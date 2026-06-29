from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = {
    "arabidopsis_single": ROOT / "models" / "arabidopsis_single_strict2601_common6751",
    "rice_single": ROOT / "models" / "rice_single_strict399_Tos17N4_common6751",
    "joint": ROOT / "models" / "joint_arabidopsis_rice_common6751",
}
DEFAULT_THRESHOLDS = {
    "arabidopsis_single": 0.3,
    "rice_single": 0.4015871584415436,
    "joint": 0.45597791914222807,
}
PRED_COLS_RICE = ["lgbm_gbdt", "xgb_depth3", "logistic", "mean_all"]
PRED_COLS_ATH = [
    "extra_sqrt",
    "extra_log2",
    "rf_sqrt",
    "lgbm_gbdt",
    "xgb_depth3",
    "logistic",
    "mean_all",
    "mean_tree",
]


def patch_sklearn_compat(obj):
    if obj is None:
        return obj
    if obj.__class__.__name__ == "LogisticRegression" and not hasattr(obj, "multi_class"):
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


def transform_with(transforms, x: np.ndarray, n_bio: int) -> np.ndarray:
    imputer, scaler, pca, selector = transforms
    if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
        imputer._fill_dtype = imputer._fit_dtype
    x_imp = imputer.transform(x)
    x_emb = scaler.transform(x_imp[:, n_bio:])
    x_pca = pca.transform(x_emb)
    return selector.transform(np.hstack([x_imp[:, :n_bio], x_pca]).astype(np.float32))


def logit_mean(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(x, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    return 1 / (1 + np.exp(-logits.mean(axis=1)))


def load_npz_features(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    meta = pd.DataFrame({"gene_id": data["gene_id"].astype(str)})
    if "transcript_id" in data.files:
        meta["transcript_id"] = data["transcript_id"].astype(str)
    if "sequence_id" in data.files:
        meta["sequence_id"] = data["sequence_id"].astype(str)
    return x, meta


def deploy_common_model(model_path: Path, x: np.ndarray, method: str) -> np.ndarray:
    package = patch_sklearn_compat(joblib.load(model_path))
    columns = []
    names = []
    for bundle in package["deployment_base_models"]:
        transformed = transform_with(bundle["transforms"], x, int(package["n_bio"]))
        predictions = {}
        for model_name, model in bundle["models"].items():
            predictions[model_name] = model.predict_proba(transformed)[:, 1]
        predictions["mean_all"] = np.mean([predictions[name] for name in bundle["models"]], axis=0)
        for model_name in PRED_COLS_RICE:
            columns.append(predictions[model_name])
            names.append(f"{bundle['config']['name']}__{model_name}")
    if names != list(package["meta_feature_names"]):
        raise RuntimeError("Meta-feature order mismatch")
    meta = np.column_stack(columns).astype(np.float32)
    if method == "meta":
        return package["meta_model"].predict_proba(meta)[:, 1]
    if method == "mean":
        return meta.mean(axis=1)
    if method == "logit_mean":
        return logit_mean(meta)
    raise ValueError(f"Unknown method: {method}")


def deploy_arabidopsis_single(model_dir: Path, x: np.ndarray) -> np.ndarray:
    package = patch_sklearn_compat(joblib.load(model_dir / "selected_model_and_manifest.joblib"))
    columns = []
    for path in package["base_model_bundle_paths"]:
        bundle_path = Path(path)
        if not bundle_path.exists():
            bundle_path = model_dir / bundle_path.name
        bundle = patch_sklearn_compat(joblib.load(bundle_path))
        transformed = transform_with(bundle["transforms"], x, int(bundle["n_bio"]))
        predictions = {name: model.predict_proba(transformed)[:, 1] for name, model in bundle["models"].items()}
        predictions["mean_all"] = np.mean(list(predictions.values()), axis=0)
        predictions["mean_tree"] = np.mean(
            [predictions[name] for name in ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]],
            axis=0,
        )
        for name in PRED_COLS_ATH:
            columns.append(predictions[name])
    meta = np.column_stack(columns).astype(np.float32)
    selected = package["selected_candidate"]["candidate"]
    if selected == "global_mean_all_40_columns":
        return meta.mean(axis=1)
    if selected == "logistic_stacking_5fold_validation_oof":
        return patch_sklearn_compat(package["stacking_model"]).predict_proba(meta)[:, 1]
    names = package["base_prediction_feature_names"]
    if selected not in names:
        raise RuntimeError(f"Unsupported Arabidopsis candidate: {selected}")
    return meta[:, names.index(selected)]


def predict(model: str, x: np.ndarray) -> np.ndarray:
    if model == "arabidopsis_single":
        return deploy_arabidopsis_single(MODEL_DIRS[model], x)
    if model == "rice_single":
        return deploy_common_model(MODEL_DIRS[model] / "model.joblib", x, "logit_mean")
    if model == "joint":
        return deploy_common_model(MODEL_DIRS[model] / "model.joblib", x, "meta")
    raise ValueError(f"Unknown model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict plant essential-gene probability from processed 6751-dim features.")
    parser.add_argument("--features", required=True, type=Path, help="Processed feature .npz file with X and gene_id arrays.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_DIRS), help="Model to use.")
    parser.add_argument("--out", required=True, type=Path, help="Output TSV file.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional classification threshold.")
    args = parser.parse_args()

    x, meta = load_npz_features(args.features)
    probability = predict(args.model, x)
    threshold = DEFAULT_THRESHOLDS[args.model] if args.threshold is None else args.threshold
    out = meta.copy()
    out["essential_probability"] = probability
    out["classification_threshold"] = threshold
    out["predicted_label"] = (out["essential_probability"] >= threshold).astype(int)
    out["predicted_class"] = np.where(out["predicted_label"].eq(1), "essential", "nonessential")
    out["model_name"] = args.model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["essential_probability", "gene_id"], ascending=[False, True]).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
