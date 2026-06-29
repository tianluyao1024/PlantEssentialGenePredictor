from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_joint_ath2601_rice_strict399_common6751 as joint_tools


def discover_root() -> Path:
    candidates = [
        path
        for path in Path("E:/CodexMoved/Desktop").rglob(
            "cross_species_ath_rice_common_features_models"
        )
        if (path / "paper_submission_experiments").exists()
    ]
    if not candidates:
        raise FileNotFoundError("Cross-species model root was not found.")
    return next(
        (path for path in candidates if "水稻" in str(path)),
        candidates[0],
    )


ROOT = discover_root()
EXPERIMENTS = ROOT / "paper_submission_experiments"
JOINT_ROOT = ROOT / "joint_ath2601_rice_strict399_common6751"
OUT = EXPERIMENTS / "paper_submission_complete"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def metric_row(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    return {
        "auc": roc_auc_score(y, probability),
        "auprc": average_precision_score(y, probability),
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "youden": sensitivity + specificity - 1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_sn80_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(
        np.r_[0.0, probability, np.nextafter(probability, -np.inf), 1.0]
    )
    rows = []
    for threshold in candidates:
        row = metric_row(y, probability, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table["sensitivity"] >= 0.80]
    if len(eligible):
        chosen = eligible.sort_values(
            ["specificity", "sensitivity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    else:
        chosen = table.sort_values(
            ["sensitivity", "specificity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    return float(chosen["threshold"])


def load_threshold_inputs():
    rice_x, names, rice_meta, rice_split = joint_tools.load_rice()
    ath_x, ath_names, ath_meta, ath_split = joint_tools.load_ath()
    single_npz = np.load(
        JOINT_ROOT
        / "rice_strict399_N4_common6751_baseline/target_predictions.npz"
    )
    joint_npz = np.load(
        JOINT_ROOT / "joint_common6751_stacking/target_predictions.npz"
    )
    ath_model_root = next(
        path
        for path in Path("D:/").rglob(
            "ath_three_labelsets_common6751_fixed_core1623_80_10_10"
        )
    ) / "strict2601_common6751"
    ath_val = pd.read_csv(
        ath_model_root / "shared_validation_predictions.tsv", sep="\t"
    )
    ath_test = pd.read_csv(
        ath_model_root / "shared_test_predictions.tsv", sep="\t"
    )
    data = {
        ("rice_single", "rice"): {
            "validation_y": rice_split["validation"]["label"].to_numpy(int),
            "validation_p": single_npz["rice_validation__logit_mean"],
            "test_y": rice_split["test"]["label"].to_numpy(int),
            "test_p": single_npz["rice_test__logit_mean"],
            "formal_threshold": float(
                pd.read_csv(
                    JOINT_ROOT
                    / "rice_strict399_N4_common6751_baseline/"
                    "fixed_test_predictions.tsv",
                    sep="\t",
                )["classification_threshold"].iloc[0]
            ),
        },
        ("arabidopsis_single", "arabidopsis"): {
            "validation_y": ath_val["label"].to_numpy(int),
            "validation_p": ath_val["probability"].to_numpy(float),
            "test_y": ath_test["label"].to_numpy(int),
            "test_p": ath_test["probability"].to_numpy(float),
            "formal_threshold": float(ath_test["classification_threshold"].iloc[0]),
        },
        ("joint", "rice"): {
            "validation_y": rice_split["validation"]["label"].to_numpy(int),
            "validation_p": joint_npz["rice_validation__meta"],
            "test_y": rice_split["test"]["label"].to_numpy(int),
            "test_p": joint_npz["rice_test__meta"],
            "formal_threshold": float(
                pd.read_csv(
                    JOINT_ROOT
                    / "joint_common6751_stacking/rice_fixed_test_predictions.tsv",
                    sep="\t",
                )["classification_threshold"].iloc[0]
            ),
        },
        ("joint", "arabidopsis"): {
            "validation_y": ath_split["validation"]["label"].to_numpy(int),
            "validation_p": joint_npz["ath_validation__meta"],
            "test_y": ath_split["test"]["label"].to_numpy(int),
            "test_p": joint_npz["ath_test__meta"],
            "formal_threshold": float(
                pd.read_csv(
                    JOINT_ROOT
                    / "joint_common6751_stacking/ath_fixed_test_predictions.tsv",
                    sep="\t",
                )["classification_threshold"].iloc[0]
            ),
        },
    }
    return data, (rice_x, names, rice_meta, rice_split), (
        ath_x,
        ath_names,
        ath_meta,
        ath_split,
    )


def threshold_analysis(data: dict) -> pd.DataFrame:
    joint_y = np.r_[
        data[("joint", "rice")]["validation_y"],
        data[("joint", "arabidopsis")]["validation_y"],
    ]
    joint_p = np.r_[
        data[("joint", "rice")]["validation_p"],
        data[("joint", "arabidopsis")]["validation_p"],
    ]
    common_sn80 = select_sn80_threshold(joint_y, joint_p)
    rows = []
    for (model, species), item in data.items():
        species_sn80 = select_sn80_threshold(
            item["validation_y"], item["validation_p"]
        )
        rules = [
            ("validation_formal", item["formal_threshold"]),
            ("fixed_0.5", 0.5),
            ("species_specific_SNge0.80_maxSP", species_sn80),
        ]
        if model == "joint":
            rules.append(("joint_common_SNge0.80_maxSP", common_sn80))
        for rule, threshold in rules:
            validation = metric_row(
                item["validation_y"], item["validation_p"], threshold
            )
            test = metric_row(item["test_y"], item["test_p"], threshold)
            rows.append(
                {
                    "model": model,
                    "species": species,
                    "threshold_rule": rule,
                    **{
                        f"validation_{key}": value
                        for key, value in validation.items()
                    },
                    **{f"test_{key}": value for key, value in test.items()},
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "threshold_sensitivity_analysis.tsv", sep="\t", index=False)
    return result


def adjusted_statistics():
    sources = {
        "delong": (
            EXPERIMENTS
            / "bootstrap_delong_joint/paired_delong_auc_comparisons.tsv",
            "p_value_two_sided",
        ),
        "paired_bootstrap": (
            EXPERIMENTS
            / "bootstrap_delong_joint/paired_bootstrap_metric_differences.tsv",
            "paired_bootstrap_p",
        ),
    }
    outputs = []
    for family, (path, p_column) in sources.items():
        table = pd.read_csv(path, sep="\t")
        table.insert(0, "test_family", family)
        table["p_raw"] = table[p_column]
        table["p_bh_global"] = bh_adjust(table["p_raw"])
        table["p_bonferroni_global"] = np.minimum(
            table["p_raw"] * len(table), 1.0
        )
        table["p_bh_within_species"] = table.groupby("species")["p_raw"].transform(
            lambda values: bh_adjust(values)
        )
        table["significant_fdr05"] = table["p_bh_global"] < 0.05
        outputs.append(table)
        table.to_csv(
            TABLES / f"{family}_with_multiple_testing_correction.tsv",
            sep="\t",
            index=False,
        )
    return outputs


def cliffs_delta(binary_group: np.ndarray, values: np.ndarray) -> float:
    first = values[binary_group == 1]
    second = values[binary_group == 0]
    if not len(first) or not len(second):
        return np.nan
    u = mannwhitneyu(first, second, alternative="two-sided").statistic
    return 2 * u / (len(first) * len(second)) - 1


def probability_analysis(data: dict) -> pd.DataFrame:
    rows = []
    for (model, species), item in data.items():
        y = item["test_y"]
        probability = item["test_p"]
        essential = probability[y == 1]
        nonessential = probability[y == 0]
        test = mannwhitneyu(essential, nonessential, alternative="two-sided")
        rows.append(
            {
                "model": model,
                "species": species,
                "n_essential": len(essential),
                "n_nonessential": len(nonessential),
                "essential_median_probability": np.median(essential),
                "nonessential_median_probability": np.median(nonessential),
                "mann_whitney_u": test.statistic,
                "p_raw": test.pvalue,
                "cliffs_delta": cliffs_delta(y, probability),
            }
        )
    result = pd.DataFrame(rows)
    result["p_bh"] = bh_adjust(result["p_raw"])
    result.to_csv(TABLES / "prediction_probability_distribution_tests.tsv", sep="\t", index=False)
    return result


def attach_features(
    matrix: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
    split: pd.DataFrame,
    prediction_path: Path,
) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path, sep="\t")
    gene_to_row = dict(zip(metadata["gene_id"], np.arange(len(metadata))))
    rows = np.array([gene_to_row[gene] for gene in prediction["gene_id"]])
    feature_table = pd.DataFrame(matrix[rows, :95], columns=feature_names[:95])
    return pd.concat([prediction.reset_index(drop=True), feature_table], axis=1)


def biological_error_analysis(rice_bundle, ath_bundle):
    rice_x, names, rice_meta, rice_split = rice_bundle
    ath_x, ath_names, ath_meta, ath_split = ath_bundle
    model_tables = {
        ("rice", "single"): attach_features(
            rice_x,
            names,
            rice_meta,
            rice_split["test"],
            JOINT_ROOT
            / "rice_strict399_N4_common6751_baseline/fixed_test_predictions.tsv",
        ),
        ("rice", "joint"): attach_features(
            rice_x,
            names,
            rice_meta,
            rice_split["test"],
            JOINT_ROOT
            / "joint_common6751_stacking/rice_fixed_test_predictions.tsv",
        ),
        ("arabidopsis", "single"): attach_features(
            ath_x,
            ath_names,
            ath_meta,
            ath_split["test"],
            next(
                path
                for path in Path("D:/").rglob(
                    "ath_three_labelsets_common6751_fixed_core1623_80_10_10"
                )
            )
            / "strict2601_common6751/shared_test_predictions.tsv",
        ),
        ("arabidopsis", "joint"): attach_features(
            ath_x,
            ath_names,
            ath_meta,
            ath_split["test"],
            JOINT_ROOT / "joint_common6751_stacking/ath_fixed_test_predictions.tsv",
        ),
    }
    go_columns = [name for name in names[:95] if name.startswith("go_")]
    enrichment_rows = []
    association_rows = []
    case_rows = []
    for (species, model), table in model_tables.items():
        table["error_type"] = np.select(
            [
                (table["label"] == 0) & (table["predicted_label"] == 1),
                (table["label"] == 1) & (table["predicted_label"] == 0),
            ],
            ["false_positive", "false_negative"],
            default="correct",
        )
        case_rows.append(
            table[
                [
                    "gene_id",
                    "label",
                    "probability",
                    "predicted_label",
                    "error_type",
                    "gene_family_size",
                    "singleton_status",
                    "string_network_connections_400",
                    "string_network_connections_700",
                    "go_embryo_development",
                    "go_cell_cycle",
                    "go_ribosome",
                ]
            ].assign(species=species, model=model)
        )
        for error_type in ["false_positive", "false_negative"]:
            in_error = table["error_type"].eq(error_type)
            for feature in go_columns:
                annotated = pd.to_numeric(table[feature], errors="coerce").fillna(0) > 0
                a = int((in_error & annotated).sum())
                b = int((in_error & ~annotated).sum())
                c = int((~in_error & annotated).sum())
                d = int((~in_error & ~annotated).sum())
                odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                enrichment_rows.append(
                    {
                        "species": species,
                        "model": model,
                        "error_type": error_type,
                        "go_feature": feature,
                        "error_annotated": a,
                        "error_total": a + b,
                        "background_annotated": c,
                        "background_total": c + d,
                        "odds_ratio": odds,
                        "p_raw": p,
                    }
                )
        for feature in [
            "gene_family_size",
            "singleton_status",
            "string_network_connections_400",
            "string_network_connections_700",
        ]:
            values = pd.to_numeric(table[feature], errors="coerce")
            valid = values.notna()
            rho, p_spearman = spearmanr(
                values[valid], table.loc[valid, "probability"]
            )
            essential = values[(table["label"] == 1) & valid]
            nonessential = values[(table["label"] == 0) & valid]
            if len(essential) and len(nonessential):
                mw = mannwhitneyu(essential, nonessential, alternative="two-sided")
                mw_u, mw_p = mw.statistic, mw.pvalue
            else:
                mw_u, mw_p = np.nan, np.nan
            association_rows.append(
                {
                    "species": species,
                    "model": model,
                    "feature": feature,
                    "n": int(valid.sum()),
                    "spearman_probability_rho": rho,
                    "spearman_probability_p": p_spearman,
                    "essential_median": essential.median(),
                    "nonessential_median": nonessential.median(),
                    "mann_whitney_u": mw_u,
                    "mann_whitney_p": mw_p,
                }
            )
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment["p_bh_within_species_model_error"] = enrichment.groupby(
        ["species", "model", "error_type"]
    )["p_raw"].transform(lambda values: bh_adjust(values))
    enrichment.to_csv(TABLES / "false_positive_false_negative_GO_enrichment.tsv", sep="\t", index=False)
    associations = pd.DataFrame(association_rows)
    associations["spearman_p_bh"] = bh_adjust(
        associations["spearman_probability_p"]
    )
    associations["mann_whitney_p_bh"] = bh_adjust(
        associations["mann_whitney_p"]
    )
    associations.to_csv(TABLES / "duplication_PPI_probability_associations.tsv", sep="\t", index=False)
    cases = pd.concat(case_rows, ignore_index=True)
    cases.to_csv(TABLES / "annotated_test_prediction_error_cases.tsv", sep="\t", index=False)
    return enrichment, associations, cases


def label_quality_summary() -> pd.DataFrame:
    rows = []
    n_dir = next(
        ROOT.rglob("rice_Oryzabase_strict399_Tos17_N1_to_N10_6755")
    )
    n_table = pd.read_csv(n_dir / "N1_to_N10_final_comparison.tsv", sep="\t")
    for n, group in n_table.groupby("minimum_Tos17_N"):
        best = group.sort_values(
            ["validation_auc", "validation_auprc"], ascending=False
        ).iloc[0]
        rows.append(
            {
                "experiment_family": "Tos17_evidence_threshold",
                "version": f"N>={int(n)}",
                "selection_basis": "best validation AUC within N tier",
                "training_total": best["training_total"],
                "training_essential": best["training_essential"],
                "training_nonessential": best["training_nonessential"],
                "validation_auc": best["validation_auc"],
                "test_auc": best["test_auc"],
                "test_auprc": best["test_auprc"],
                "test_sensitivity": best["test_sensitivity"],
                "test_specificity": best["test_specificity"],
                "test_f1": best["test_f1"],
            }
        )
    sr_dir = next(ROOT.rglob("rice_Oryzabase_strict_vs_relaxed_6755"))
    sr = pd.read_csv(sr_dir / "strict_relaxed_test_comparison.tsv", sep="\t")
    for _, item in sr.iterrows():
        rows.append(
            {
                "experiment_family": "Oryzabase_label_stringency",
                "version": item["label_version"],
                "selection_basis": "historical fixed comparison",
                "training_total": np.nan,
                "training_essential": np.nan,
                "training_nonessential": np.nan,
                "validation_auc": item["validation_auc"],
                "test_auc": item["test_auc"],
                "test_auprc": item["test_auprc"],
                "test_sensitivity": item["test_sensitivity"],
                "test_specificity": item["test_specificity"],
                "test_f1": item["test_f1"],
            }
        )
    pseudo_dir = next(
        ROOT.rglob("rice_strict399_all_sources_concordant_and_pseudo_ladder")
    )
    for path in pseudo_dir.rglob("result.tsv"):
        item = pd.read_csv(path, sep="\t").iloc[0]
        rows.append(
            {
                "experiment_family": "pseudo_label_extension",
                "version": item["version"],
                "selection_basis": "historical fixed split",
                "training_total": item["training_total"],
                "training_essential": np.nan,
                "training_nonessential": np.nan,
                "validation_auc": item["validation_auc"],
                "test_auc": item["test_auc"],
                "test_auprc": item["test_auprc"],
                "test_sensitivity": item["test_sensitivity"],
                "test_specificity": item["test_specificity"],
                "test_f1": item["test_f1"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "rice_label_quality_experiment_summary.tsv", sep="\t", index=False)
    selected_versions = [
        "N>=4",
        "N>=6",
        "N>=8",
        "strict",
        "relaxed",
        "route1_concordant_experimental",
        "route2_pseudo_pos0.95_neg0.05",
        "route2_pseudo_pos0.80_neg0.20",
    ]
    result[result["version"].isin(selected_versions)].to_csv(
        TABLES / "rice_label_quality_key_comparisons.tsv",
        sep="\t",
        index=False,
    )
    return result


def plot_workflow():
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axis("off")
    boxes = [
        (0.03, "Phenotype sources\nand label cleaning"),
        (0.23, "Longest transcript\nand 6,751 features"),
        (0.43, "Fixed train / validation /\ntest partition"),
        (0.63, "Single-species and\njoint stacking models"),
        (0.83, "Bootstrap, ablation,\nhomology-grouped tests"),
    ]
    for x, text in boxes:
        rect = plt.Rectangle(
            (x, 0.35), 0.15, 0.30, facecolor="#e8f1f8", edgecolor="#174a6e", lw=1.5
        )
        ax.add_patch(rect)
        ax.text(x + 0.075, 0.50, text, ha="center", va="center", fontsize=10)
    for x, _ in boxes[:-1]:
        ax.annotate(
            "",
            xy=(x + 0.195, 0.50),
            xytext=(x + 0.15, 0.50),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#444444"},
        )
    ax.text(
        0.5,
        0.15,
        "Validation selects models and thresholds; the fixed test set is evaluated once.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure1_modeling_workflow.png", dpi=300)
    plt.close(fig)


def plot_label_counts(rice_bundle, ath_bundle):
    rows = []
    for species, bundle in [("Rice", rice_bundle), ("Arabidopsis", ath_bundle)]:
        split = bundle[3]
        for part in ["train", "validation", "test"]:
            for label, name in [(0, "Non-essential"), (1, "Essential")]:
                rows.append(
                    {
                        "species": species,
                        "split": part.title(),
                        "class": name,
                        "count": int((split[part]["label"] == label).sum()),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "fixed_split_label_counts.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    colors = {"Non-essential": "#4c78a8", "Essential": "#e45756"}
    for ax, species in zip(axes, ["Rice", "Arabidopsis"]):
        subset = table[table["species"] == species]
        x = np.arange(3)
        bottom = np.zeros(3)
        for klass in ["Non-essential", "Essential"]:
            values = (
                subset[subset["class"] == klass]
                .set_index("split")
                .loc[["Train", "Validation", "Test"], "count"]
                .to_numpy()
            )
            ax.bar(x, values, bottom=bottom, color=colors[klass], label=klass)
            bottom += values
        ax.set_xticks(x, ["Train", "Validation", "Test"])
        ax.set_title(species)
        ax.set_ylabel("Genes")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure2_label_class_distribution.png", dpi=300)
    plt.close(fig)


def plot_roc_pr(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for column, species in enumerate(["rice", "arabidopsis"]):
        for model, label, color in [
            (f"{species}_single" if species == "rice" else "arabidopsis_single", "Single-species", "#1f77b4"),
            ("joint", "Joint", "#d62728"),
        ]:
            key = (model, species)
            item = data[key]
            fpr, tpr, _ = roc_curve(item["test_y"], item["test_p"])
            precision, recall, _ = precision_recall_curve(
                item["test_y"], item["test_p"]
            )
            auc = roc_auc_score(item["test_y"], item["test_p"])
            auprc = average_precision_score(item["test_y"], item["test_p"])
            axes[0, column].plot(fpr, tpr, label=f"{label} ({auc:.3f})", color=color)
            axes[1, column].plot(recall, precision, label=f"{label} ({auprc:.3f})", color=color)
        axes[0, column].plot([0, 1], [0, 1], "--", color="#999999")
        axes[0, column].set_title(f"{species.title()} ROC")
        axes[1, column].set_title(f"{species.title()} precision-recall")
        for row in range(2):
            axes[row, column].set_xlim(0, 1)
            axes[row, column].set_ylim(0, 1)
            axes[row, column].legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("Sensitivity")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 1].set_xlabel("Recall")
    axes[0, 0].set_xlabel("1 - specificity")
    axes[0, 1].set_xlabel("1 - specificity")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure3_ROC_PR_single_vs_joint.png", dpi=300)
    plt.close(fig)


def plot_ablation():
    table = pd.read_csv(
        EXPERIMENTS / "common6751_ablation/complete_ablation_results.tsv",
        sep="\t",
    )
    table["task"] = np.where(
        table["dataset"].eq("joint"),
        "Joint-" + table["evaluation_species"].fillna(""),
        table["dataset"].str.title(),
    )
    variants = [
        "bio95_only",
        "esm2_only",
        "protbert_only",
        "prott5_only",
        "all_plm_6656",
        "bio95_plus_esm2",
        "bio95_plus_protbert",
        "bio95_plus_prott5",
        "full_common6751",
        "full_without_GO",
        "full_without_PPI",
        "full_without_GO_PPI",
    ]
    labels = {
        "bio95_only": "Bio95",
        "esm2_only": "ESM2",
        "protbert_only": "ProtBERT",
        "prott5_only": "ProtT5",
        "all_plm_6656": "All PLM",
        "bio95_plus_esm2": "Bio+ESM2",
        "bio95_plus_protbert": "Bio+ProtBERT",
        "bio95_plus_prott5": "Bio+ProtT5",
        "full_common6751": "Full",
        "full_without_GO": "No GO",
        "full_without_PPI": "No PPI",
        "full_without_GO_PPI": "No GO/PPI",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    tasks = ["Rice", "Arabidopsis", "Joint-rice", "Joint-arabidopsis"]
    for ax, task in zip(axes.ravel(), tasks):
        subset = table[(table["task"].str.lower() == task.lower()) & table["variant"].isin(variants)]
        subset = subset.set_index("variant").reindex(variants)
        colors = ["#4c78a8" if v != "full_common6751" else "#e45756" for v in variants]
        ax.bar(np.arange(len(variants)), subset["test_auc"], color=colors)
        ax.set_ylim(0.65, 0.96)
        ax.set_title(task)
        ax.set_ylabel("Test AUC")
        ax.axhline(
            float(subset.loc["full_common6751", "test_auc"]),
            color="#e45756",
            ls="--",
            lw=1,
        )
    for ax in axes[1]:
        ax.set_xticks(np.arange(len(variants)), [labels[v] for v in variants], rotation=55, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure4_feature_ablation_AUC.png", dpi=300)
    plt.close(fig)


def plot_bootstrap():
    table = pd.read_csv(
        EXPERIMENTS / "bootstrap_delong_joint/model_metric_bootstrap_95ci.tsv",
        sep="\t",
    )
    subset = table[table["metric"].isin(["auc", "auprc"])].copy()
    subset["label"] = (
        subset["species"].str.title()
        + " - "
        + subset["model"].str.replace("_", " ").str.title()
        + " - "
        + subset["metric"].str.upper()
    )
    subset = subset.sort_values(["species", "metric", "model"])
    y = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        subset["estimate"],
        y,
        xerr=np.vstack(
            [
                subset["estimate"] - subset["ci95_low"],
                subset["ci95_high"] - subset["estimate"],
            ]
        ),
        fmt="o",
        color="#174a6e",
        ecolor="#6b8ba4",
        capsize=3,
    )
    ax.set_yticks(y, subset["label"])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Estimate with bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure5_bootstrap_confidence_intervals.png", dpi=300)
    plt.close(fig)


def plot_importance():
    table = pd.read_csv(
        EXPERIMENTS
        / "importance_and_errors/permutation_importance_bio95_and_plm_groups.tsv",
        sep="\t",
    )
    models = list(table["model"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, model in zip(axes.ravel(), models):
        subset = (
            table[table["model"] == model]
            .nlargest(12, "mean_auc_drop")
            .sort_values("mean_auc_drop")
        )
        labels = subset["feature_or_group"].str.replace("GROUP_", "", regex=False)
        ax.barh(labels, subset["mean_auc_drop"], color="#4c78a8")
        ax.set_title(model.replace("_", " "))
        ax.set_xlabel("Mean AUC decrease after permutation")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure6_permutation_importance.png", dpi=300)
    plt.close(fig)


def plot_errors(cases):
    counts = (
        cases.groupby(["species", "model", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    order = ["correct", "false_positive", "false_negative"]
    colors = ["#59a14f", "#e15759", "#f28e2b"]
    labels = counts["species"].str.title() + "\n" + counts["model"].str.title()
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(counts))
    for error, color in zip(order, colors):
        values = counts.get(error, pd.Series(np.zeros(len(counts)))).to_numpy()
        ax.bar(np.arange(len(counts)), values, bottom=bottom, label=error.replace("_", " "), color=color)
        bottom += values
    ax.set_xticks(np.arange(len(counts)), labels)
    ax.set_ylabel("Test genes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure7_error_type_counts.png", dpi=300)
    plt.close(fig)


def plot_probability(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    keys = [
        ("rice_single", "rice"),
        ("joint", "rice"),
        ("arabidopsis_single", "arabidopsis"),
        ("joint", "arabidopsis"),
    ]
    for ax, key in zip(axes.ravel(), keys):
        item = data[key]
        groups = [
            item["test_p"][item["test_y"] == 0],
            item["test_p"][item["test_y"] == 1],
        ]
        violin = ax.violinplot(groups, positions=[0, 1], showmedians=True)
        for body, color in zip(violin["bodies"], ["#4c78a8", "#e45756"]):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        ax.set_xticks([0, 1], ["Non-essential", "Essential"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted probability")
        ax.set_title(f"{key[1].title()} - {key[0].replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure8_prediction_probability_distributions.png", dpi=300)
    plt.close(fig)


def plot_label_quality(table):
    subset = table[table["experiment_family"] == "Tos17_evidence_threshold"].copy()
    subset["N"] = subset["version"].str.extract(r"(\d+)").astype(int)
    subset = subset.sort_values("N")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(subset["N"], subset["test_auc"], marker="o", label="Test AUC", color="#174a6e")
    ax1.plot(subset["N"], subset["test_auprc"], marker="s", label="Test AUPRC", color="#e45756")
    ax1.set_xlabel("Minimum Tos17 non-essential evidence count (N)")
    ax1.set_ylabel("Performance")
    ax1.set_ylim(0.75, 1.0)
    ax2 = ax1.twinx()
    ax2.plot(subset["N"], subset["training_total"], marker="^", color="#59a14f", label="Training genes")
    ax2.set_ylabel("Training genes")
    handles = ax1.get_lines() + ax2.get_lines()
    ax1.legend(handles, [line.get_label() for line in handles], frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "Supplementary_label_quality_Tos17_N_threshold.png", dpi=300)
    plt.close(fig)


def write_reports(thresholds, probability, enrichment, associations, label_quality):
    significant_go = enrichment[
        enrichment["p_bh_within_species_model_error"] < 0.05
    ]
    significant_associations = associations[
        (associations["spearman_p_bh"] < 0.05)
        | (associations["mann_whitney_p_bh"] < 0.05)
    ]
    report = f"""# Submission-ready analysis summary

## Completed analyses

- Complete 15-variant ablation for rice, Arabidopsis, and joint models.
- 10,000-replicate bootstrap confidence intervals.
- Paired DeLong AUC tests and paired bootstrap comparisons.
- BH-FDR and Bonferroni multiple-testing correction.
- Formal, fixed 0.5, SN>=0.80/max-SP, common-joint, and species-specific threshold analyses.
- Homology-cluster grouped validation using DIAMOND sequence clusters.
- Permutation importance for 95 biological features and three grouped PLM embeddings.
- Essential/non-essential probability distribution tests.
- False-positive and false-negative GO enrichment.
- Duplication/PPI association analysis.
- Consolidated rice label-quality comparisons.
- Publication figures and fixed-split data audit.

## Interpretation guardrails

1. The complete 6,751-feature model remains the prespecified primary model because its selection was supported by validation performance. Ablation variants with numerically higher test AUC must not replace it post hoc.
2. Joint training does not significantly improve AUC over the corresponding single-species model. It increases sensitivity while reducing specificity.
3. Homology-grouped AUC remains above random but is lower than the conventional fixed split, indicating that sequence similarity contributes to performance without fully explaining it.
4. GO embryo-development annotations are highly influential in Arabidopsis. The manuscript must discuss annotation/label-proxy bias and report the no-GO ablation.
5. Current fixed label tables do not encode a harmonized phenotype-stage field for embryo lethal, sterile, seedling lethal, and related subtypes. Therefore, subtype probability comparisons were not fabricated; they require a separately curated phenotype-stage table.

## New result counts

- Threshold result rows: {len(thresholds)}
- Probability tests: {len(probability)}
- Significant error GO enrichments after within-analysis FDR: {len(significant_go)}
- Significant duplication/PPI associations after FDR: {len(significant_associations)}
- Label-quality comparison rows: {len(label_quality)}
"""
    (REPORTS / "analysis_summary.md").write_text(report, encoding="utf-8")
    methods = """# Methods text for manuscript

## Threshold selection and evaluation

ROC AUC and area under the precision-recall curve were calculated from continuous prediction probabilities and were therefore independent of the classification threshold. Classification thresholds were selected using validation data only. We report the originally locked validation threshold, a fixed threshold of 0.5, and a sensitivity-prioritized threshold selected by retaining validation thresholds with sensitivity of at least 0.80 and maximizing specificity. Ties were resolved by sensitivity and then F1 score. For the joint model, both a common threshold pooled across the two validation sets and species-specific validation thresholds were evaluated. The test sets were not used for threshold selection.

## Statistical comparisons

Confidence intervals were estimated using 10,000 stratified bootstrap replicates. Models evaluated on the same test genes were compared using paired bootstrap resampling for AUPRC, sensitivity, specificity and F1 score, and the paired DeLong test for ROC AUC. Raw P values were adjusted using the Benjamini-Hochberg false-discovery-rate procedure and Bonferroni correction.

## Homology-aware evaluation

Protein sequences were aligned using DIAMOND. Edges required at least 40% sequence identity and at least 60% coverage of both query and subject. Connected components were treated as homology clusters, and clusters were assigned wholly to training, validation or test partitions. This prevented highly similar proteins from occurring in both training and evaluation sets.

## Model interpretation and error analysis

Permutation importance was calculated on the fixed test set with five repeats. The 95 biological features were permuted individually, whereas the ESM2, ProtBERT and ProtT5 embeddings were permuted as three feature groups. False-positive and false-negative genes were tested for GO-feature enrichment using one-sided Fisher exact tests, with FDR correction within each species, model and error class. Spearman correlation and Mann-Whitney tests were used to evaluate relationships between prediction probability, gene-family size, singleton status and STRING network connectivity.

## Leakage prevention

The longest transcript was selected before feature extraction using a single deterministic rule. Validation and test genes came exclusively from experimentally supported high-confidence labels. Pseudo-labelled genes, where used in historical extension experiments, were restricted to training. Missing-value imputation, dimensionality reduction, feature selection and model fitting were performed within the training folds. The fixed test sets were excluded from feature selection, model selection and threshold selection. The 6,751-dimensional common feature representation did not include the Arabidopsis-derived prior feature or the three rice-specific homology features.
"""
    (REPORTS / "manuscript_methods_ready.md").write_text(methods, encoding="utf-8")
    limitations = """# Remaining biological-data limitation

All computational and statistical analyses requested for the current fixed datasets are complete. One proposed analysis cannot be validly performed from the current model tables: comparison of embryo-lethal, sterile, seedling-lethal and other phenotype-stage probabilities. The fixed labels contain essential/non-essential status and source provenance, but no harmonized phenotype-stage category. A new phenotype-stage curation table is required before this comparison can be reported.
"""
    (REPORTS / "remaining_data_limitation.md").write_text(limitations, encoding="utf-8")


def main():
    for directory in [OUT, TABLES, FIGURES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    data, rice_bundle, ath_bundle = load_threshold_inputs()
    thresholds = threshold_analysis(data)
    adjusted_statistics()
    probability = probability_analysis(data)
    enrichment, associations, cases = biological_error_analysis(
        rice_bundle, ath_bundle
    )
    label_quality = label_quality_summary()
    plot_workflow()
    plot_label_counts(rice_bundle, ath_bundle)
    plot_roc_pr(data)
    plot_ablation()
    plot_bootstrap()
    plot_importance()
    plot_errors(cases)
    plot_probability(data)
    plot_label_quality(label_quality)
    write_reports(
        thresholds,
        probability,
        enrichment,
        associations,
        label_quality,
    )
    manifest = {
        "status": "complete",
        "output_root": str(OUT),
        "tables": sorted(path.name for path in TABLES.glob("*")),
        "figures": sorted(path.name for path in FIGURES.glob("*")),
        "reports": sorted(path.name for path in REPORTS.glob("*")),
        "explicit_limitation": (
            "Phenotype-stage comparison requires a harmonized subtype table "
            "that is absent from the fixed model labels."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
