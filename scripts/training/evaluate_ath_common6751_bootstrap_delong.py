from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(
    "D:/拟南芥/模型/ath_three_labelsets_common6751_fixed_core1623_80_10_10"
)
OUT_DIR = ROOT / "paired_bootstrap_10000_delong"
N_BOOTSTRAP = 10_000
RANDOM_STATE = 20260619

MODEL_DIRS = {
    "core1623": "core1623_common6751",
    "strict2601": "strict2601_common6751",
    "teacher3359": "teacher3359_common6751",
}
METRIC_NAMES = [
    "auc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
]


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def load_predictions() -> tuple[np.ndarray, dict[str, dict]]:
    loaded = {}
    reference_genes = None
    reference_y = None
    for model_name, directory in MODEL_DIRS.items():
        path = ROOT / directory / "shared_test_predictions.tsv"
        frame = pd.read_csv(path, sep="\t")
        frame["gene_id"] = frame["gene_id"].astype(str).str.upper()
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8)
        frame["probability"] = pd.to_numeric(
            frame["probability"], errors="raise"
        ).astype(float)
        thresholds = pd.to_numeric(
            frame["classification_threshold"], errors="raise"
        ).unique()
        if len(thresholds) != 1:
            raise RuntimeError(f"{model_name}: expected one threshold, got {thresholds}")
        genes = frame["gene_id"].to_numpy()
        y = frame["label"].to_numpy(np.int8)
        if reference_genes is None:
            reference_genes = genes
            reference_y = y
        else:
            if not np.array_equal(genes, reference_genes):
                raise RuntimeError(f"{model_name}: test genes are not identically ordered")
            if not np.array_equal(y, reference_y):
                raise RuntimeError(f"{model_name}: test labels differ")
        loaded[model_name] = {
            "probability": frame["probability"].to_numpy(float),
            "threshold": float(thresholds[0]),
            "path": str(path),
        }
    assert reference_y is not None
    return reference_y, loaded


def delong_components(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    comparison = (
        (positive_scores[:, None] > negative_scores[None, :]).astype(float)
        + 0.5
        * (positive_scores[:, None] == negative_scores[None, :]).astype(float)
    )
    v10 = comparison.mean(axis=1)
    v01 = comparison.mean(axis=0)
    return float(v10.mean()), v10, v01


def paired_delong(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
) -> dict[str, float]:
    positive = y == 1
    negative = y == 0
    auc_a, v10_a, v01_a = delong_components(
        probability_a[positive], probability_a[negative]
    )
    auc_b, v10_b, v01_b = delong_components(
        probability_b[positive], probability_b[negative]
    )
    difference = auc_a - auc_b
    variance = (
        np.var(v10_a - v10_b, ddof=1) / positive.sum()
        + np.var(v01_a - v01_b, ddof=1) / negative.sum()
    )
    standard_error = float(np.sqrt(max(variance, 0.0)))
    if standard_error == 0:
        z = np.inf if difference != 0 else 0.0
        p_value = 0.0 if difference != 0 else 1.0
    else:
        z = float(difference / standard_error)
        p_value = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_difference_a_minus_b": difference,
        "standard_error": standard_error,
        "z_statistic": z,
        "p_value_two_sided": p_value,
    }


def bootstrap_p_value(differences: np.ndarray) -> float:
    less_or_equal_zero = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    greater_or_equal_zero = (np.sum(differences >= 0) + 1) / (
        len(differences) + 1
    )
    return float(min(1.0, 2.0 * min(less_or_equal_zero, greater_or_equal_zero)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y, models = load_predictions()
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)
    if (len(positive_idx), len(negative_idx)) != (40, 123):
        raise RuntimeError(
            f"Expected 40 positive and 123 negative test genes, got "
            f"{len(positive_idx)}, {len(negative_idx)}"
        )

    point_estimates = {
        name: metrics(y, item["probability"], item["threshold"])
        for name, item in models.items()
    }
    bootstrap = {
        name: {
            metric_name: np.full(N_BOOTSTRAP, np.nan, dtype=np.float64)
            for metric_name in METRIC_NAMES
        }
        for name in models
    }

    rng = np.random.default_rng(RANDOM_STATE)
    checkpoint_path = OUT_DIR / "bootstrap_checkpoint.npz"
    start_iteration = 0
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        start_iteration = int(checkpoint["completed_iterations"])
        for model_name in models:
            for metric_name in METRIC_NAMES:
                bootstrap[model_name][metric_name][:start_iteration] = checkpoint[
                    f"{model_name}__{metric_name}"
                ][:]
        # Advance the deterministic generator to the next uncomputed replicate.
        for _ in range(start_iteration):
            rng.choice(positive_idx, size=len(positive_idx), replace=True)
            rng.choice(negative_idx, size=len(negative_idx), replace=True)
        print(f"resume bootstrap from {start_iteration}/{N_BOOTSTRAP}", flush=True)

    for iteration in range(start_iteration, N_BOOTSTRAP):
        sampled_positive = rng.choice(
            positive_idx, size=len(positive_idx), replace=True
        )
        sampled_negative = rng.choice(
            negative_idx, size=len(negative_idx), replace=True
        )
        sampled_idx = np.concatenate([sampled_positive, sampled_negative])
        sampled_y = y[sampled_idx]
        for model_name, item in models.items():
            sampled_probability = item["probability"][sampled_idx]
            result = metrics(
                sampled_y,
                sampled_probability,
                item["threshold"],
            )
            for metric_name in METRIC_NAMES:
                bootstrap[model_name][metric_name][iteration] = result[metric_name]
        if (iteration + 1) % 1000 == 0:
            np.savez_compressed(
                checkpoint_path,
                completed_iterations=np.array(iteration + 1, dtype=np.int64),
                **{
                    f"{model_name}__{metric_name}": values[: iteration + 1]
                    for model_name, model_metrics in bootstrap.items()
                    for metric_name, values in model_metrics.items()
                },
            )
            print(f"bootstrap {iteration + 1}/{N_BOOTSTRAP}", flush=True)

    ci_rows = []
    for model_name in models:
        for metric_name in METRIC_NAMES:
            values = bootstrap[model_name][metric_name]
            ci_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "point_estimate": point_estimates[model_name][metric_name],
                    "bootstrap_mean": float(np.mean(values)),
                    "ci95_lower": float(np.percentile(values, 2.5)),
                    "ci95_upper": float(np.percentile(values, 97.5)),
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(
        OUT_DIR / "model_metric_bootstrap_95ci.tsv",
        sep="\t",
        index=False,
    )

    delong_rows = []
    bootstrap_difference_rows = []
    for model_a, model_b in itertools.combinations(models, 2):
        delong = paired_delong(
            y,
            models[model_a]["probability"],
            models[model_b]["probability"],
        )
        delong_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                **delong,
                "significant_p_lt_0.05": delong["p_value_two_sided"] < 0.05,
            }
        )
        for metric_name in METRIC_NAMES:
            differences = (
                bootstrap[model_a][metric_name]
                - bootstrap[model_b][metric_name]
            )
            bootstrap_difference_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric_name,
                    "point_difference_a_minus_b": (
                        point_estimates[model_a][metric_name]
                        - point_estimates[model_b][metric_name]
                    ),
                    "bootstrap_mean_difference": float(np.mean(differences)),
                    "difference_ci95_lower": float(
                        np.percentile(differences, 2.5)
                    ),
                    "difference_ci95_upper": float(
                        np.percentile(differences, 97.5)
                    ),
                    "paired_bootstrap_p_value_two_sided": bootstrap_p_value(
                        differences
                    ),
                    "significant_p_lt_0.05": bootstrap_p_value(differences)
                    < 0.05,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )

    delong_table = pd.DataFrame(delong_rows)
    delong_table.to_csv(
        OUT_DIR / "paired_delong_auc_comparisons.tsv",
        sep="\t",
        index=False,
    )
    difference_table = pd.DataFrame(bootstrap_difference_rows)
    difference_table.to_csv(
        OUT_DIR / "paired_bootstrap_metric_differences.tsv",
        sep="\t",
        index=False,
    )

    np.savez_compressed(
        OUT_DIR / "bootstrap_metric_distributions.npz",
        **{
            f"{model_name}__{metric_name}": values
            for model_name, model_metrics in bootstrap.items()
            for metric_name, values in model_metrics.items()
        },
    )

    paper_table = ci_table[
        ci_table["metric"].isin(["auc", "auprc", "f1", "sensitivity", "specificity"])
    ].copy()
    paper_table["estimate_with_95ci"] = paper_table.apply(
        lambda row: (
            f"{row['point_estimate']:.3f} "
            f"({row['ci95_lower']:.3f}-{row['ci95_upper']:.3f})"
        ),
        axis=1,
    )
    paper_wide = paper_table.pivot(
        index="model",
        columns="metric",
        values="estimate_with_95ci",
    ).reset_index()
    paper_wide.to_csv(
        OUT_DIR / "paper_performance_table_with_95ci.tsv",
        sep="\t",
        index=False,
    )

    manifest = {
        "test_samples": int(len(y)),
        "test_positive": int((y == 1).sum()),
        "test_negative": int((y == 0).sum()),
        "bootstrap_method": (
            "10000 stratified paired bootstrap replicates; 40 positives and "
            "123 negatives resampled with replacement in every replicate"
        ),
        "bootstrap_random_state": RANDOM_STATE,
        "classification_thresholds": {
            name: item["threshold"] for name, item in models.items()
        },
        "prediction_sources": {
            name: item["path"] for name, item in models.items()
        },
        "delong_method": (
            "paired two-model DeLong variance computed from positive and "
            "negative structural components on the same test genes"
        ),
        "point_estimates": point_estimates,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
