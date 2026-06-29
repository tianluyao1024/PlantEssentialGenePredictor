from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from evaluate_ath_common6751_bootstrap_delong import paired_delong


ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/"
    "cross_species_ath_rice_common_features_models"
)
JOINT = ROOT / "joint_ath2601_rice_strict399_common6751"
OUT = ROOT / "paper_submission_experiments/bootstrap_delong_joint"
N_BOOTSTRAP = 10_000
SEED = 20260622

FILES = {
    ("rice", "single_species"): (
        JOINT
        / "rice_strict399_N4_common6751_baseline/fixed_test_predictions.tsv"
    ),
    ("rice", "joint_stacking"): (
        JOINT / "joint_common6751_stacking/rice_fixed_test_predictions.tsv"
    ),
    ("rice", "auxiliary_heads"): (
        JOINT
        / "joint_common6751_auxiliary_species_heads/"
        "rice_fixed_test_predictions.tsv"
    ),
    ("arabidopsis", "single_species"): (
        Path(
            "D:/\u62df\u5357\u82a5/\u6a21\u578b/"
            "ath_three_labelsets_common6751_fixed_core1623_80_10_10/"
            "strict2601_common6751/shared_test_predictions.tsv"
        )
    ),
    ("arabidopsis", "joint_stacking"): (
        JOINT / "joint_common6751_stacking/ath_fixed_test_predictions.tsv"
    ),
    ("arabidopsis", "auxiliary_heads"): (
        JOINT
        / "joint_common6751_auxiliary_species_heads/"
        "ath_fixed_test_predictions.tsv"
    ),
}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    probability_column = "probability"
    threshold_column = (
        "classification_threshold"
        if "classification_threshold" in frame
        else "threshold"
    )
    return pd.DataFrame(
        {
            "gene_id": frame["gene_id"].astype(str),
            "label": pd.to_numeric(frame["label"], errors="raise").astype(int),
            "probability": pd.to_numeric(
                frame[probability_column], errors="raise"
            ),
            "threshold": pd.to_numeric(
                frame[threshold_column], errors="raise"
            ),
        }
    ).sort_values("gene_id").reset_index(drop=True)


def score(y, probability, threshold):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, probability),
        "auprc": average_precision_score(y, probability),
        "f1": f1_score(y, prediction, zero_division=0),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
    }


def bootstrap_indices(y: np.ndarray, rng: np.random.Generator):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    return np.r_[
        rng.choice(positive, len(positive), replace=True),
        rng.choice(negative, len(negative), replace=True),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {key: load(path) for key, path in FILES.items()}
    ci_rows = []
    difference_rows = []
    delong_rows = []
    rng = np.random.default_rng(SEED)

    for species in ["rice", "arabidopsis"]:
        models = {
            model: frame
            for (current_species, model), frame in loaded.items()
            if current_species == species
        }
        reference = next(iter(models.values()))
        for model, frame in models.items():
            if not frame["gene_id"].equals(reference["gene_id"]):
                raise RuntimeError(f"{species}/{model}: test genes differ")
            if not frame["label"].equals(reference["label"]):
                raise RuntimeError(f"{species}/{model}: labels differ")

        y = reference["label"].to_numpy(int)
        bootstrap = {
            model: {metric: [] for metric in score(y, frame["probability"], frame["threshold"].iloc[0])}
            for model, frame in models.items()
        }
        for _ in range(N_BOOTSTRAP):
            index = bootstrap_indices(y, rng)
            for model, frame in models.items():
                values = score(
                    y[index],
                    frame["probability"].to_numpy(float)[index],
                    float(frame["threshold"].iloc[0]),
                )
                for metric, value in values.items():
                    bootstrap[model][metric].append(value)

        for model, frame in models.items():
            observed = score(
                y,
                frame["probability"].to_numpy(float),
                float(frame["threshold"].iloc[0]),
            )
            for metric, value in observed.items():
                distribution = np.asarray(bootstrap[model][metric])
                ci_rows.append(
                    {
                        "species": species,
                        "model": model,
                        "metric": metric,
                        "estimate": value,
                        "ci95_low": np.quantile(distribution, 0.025),
                        "ci95_high": np.quantile(distribution, 0.975),
                    }
                )

        for model_a, model_b in itertools.combinations(models, 2):
            a = models[model_a]
            b = models[model_b]
            for metric in bootstrap[model_a]:
                differences = np.asarray(bootstrap[model_b][metric]) - np.asarray(
                    bootstrap[model_a][metric]
                )
                p_value = 2 * min(
                    np.mean(differences <= 0), np.mean(differences >= 0)
                )
                difference_rows.append(
                    {
                        "species": species,
                        "reference_model": model_a,
                        "comparison_model": model_b,
                        "metric": metric,
                        "difference_comparison_minus_reference": np.mean(
                            differences
                        ),
                        "ci95_low": np.quantile(differences, 0.025),
                        "ci95_high": np.quantile(differences, 0.975),
                        "paired_bootstrap_p": min(1.0, p_value),
                    }
                )
            delong = paired_delong(
                y,
                a["probability"].to_numpy(float),
                b["probability"].to_numpy(float),
            )
            delong_rows.append(
                {
                    "species": species,
                    "reference_model": model_a,
                    "comparison_model": model_b,
                    **delong,
                }
            )

    pd.DataFrame(ci_rows).to_csv(
        OUT / "model_metric_bootstrap_95ci.tsv", sep="\t", index=False
    )
    pd.DataFrame(difference_rows).to_csv(
        OUT / "paired_bootstrap_metric_differences.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(delong_rows).to_csv(
        OUT / "paired_delong_auc_comparisons.tsv",
        sep="\t",
        index=False,
    )
    manifest = {
        "bootstrap_replicates": N_BOOTSTRAP,
        "random_seed": SEED,
        "design": "class-stratified paired bootstrap on fixed test genes",
        "files": {str(key): str(value) for key, value in FILES.items()},
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
