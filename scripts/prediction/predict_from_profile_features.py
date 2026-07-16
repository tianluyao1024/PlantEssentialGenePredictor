from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "prediction"))

from predict_from_processed_features import logit_mean, patch_sklearn_compat, transform_with  # noqa: E402


PRED_COLS = ["lgbm_gbdt", "xgb_depth3", "logistic", "mean_all"]


def load_profile_npz(path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    meta = pd.DataFrame({"gene_id": data["gene_id"].astype(str)})
    names = data["feature_names"].astype(str).tolist()
    return x, meta, names


def predict_profile(model_path: Path, x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, float, str]:
    package = patch_sklearn_compat(joblib.load(model_path))
    expected = list(package["feature_names"])
    if expected != feature_names:
        missing = [name for name in expected if name not in feature_names]
        extra = [name for name in feature_names if name not in expected]
        raise RuntimeError(f"Feature-name mismatch. Missing={missing[:10]} Extra={extra[:10]}")

    columns = []
    names = []
    for bundle in package["deployment_base_models"]:
        transformed = transform_with(bundle["transforms"], x, int(package["n_bio"]))
        predictions = {}
        for model_name, model in bundle["models"].items():
            predictions[model_name] = model.predict_proba(transformed)[:, 1]
        predictions["mean_all"] = np.mean([predictions[name] for name in bundle["models"]], axis=0)
        for model_name in PRED_COLS:
            columns.append(predictions[model_name])
            names.append(f"{bundle['config']['name']}__{model_name}")
    if names != list(package["meta_feature_names"]):
        raise RuntimeError("Meta-feature order mismatch")
    meta = np.column_stack(columns).astype(np.float32)
    method = str(package["selected_method"])
    if method == "meta":
        probability = package["meta_model"].predict_proba(meta)[:, 1]
    elif method == "mean":
        probability = meta.mean(axis=1)
    elif method == "logit_mean":
        probability = logit_mean(meta)
    else:
        raise ValueError(f"Unknown selected method: {method}")
    return probability, float(package["threshold"]), method


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict from raw-upload-derived deployable profile features.")
    parser.add_argument("--features", required=True, type=Path, help="Feature .npz from raw_upload_to_profile_features.py")
    parser.add_argument("--model", required=True, type=Path, help="Deployable profile model.joblib")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    x, meta, feature_names = load_profile_npz(args.features)
    probability, model_threshold, method = predict_profile(args.model, x, feature_names)
    threshold = model_threshold if args.threshold is None else float(args.threshold)
    out = meta.copy()
    out["essential_probability"] = probability
    out["classification_threshold"] = threshold
    out["predicted_label"] = (out["essential_probability"] >= threshold).astype(int)
    out["predicted_class"] = np.where(out["predicted_label"].eq(1), "essential", "nonessential")
    out["profile_model_path"] = str(args.model)
    out["selected_method"] = method
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["essential_probability", "gene_id"], ascending=[False, True]).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} profile predictions to {args.out}")


if __name__ == "__main__":
    main()
