"""Create compact, publication-ready figures from locked project outputs.

The script intentionally reads the fixed-test prediction tables for all ROC and
precision-recall curves. The small ablation and grouped-evaluation summary
tables below contain values reported in the locked manuscript and are retained
explicitly to make every plotted number auditable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


ROOT = Path("E:/PlantEssentialGenePredictor")
OUT = Path("C:/Users/tly/Downloads/plantessentialgene_nature_figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Rice single-species": "#2F6DAE",
    "Rice joint model": "#D17B28",
    "Arabidopsis single-species": "#3D8B5B",
    "Arabidopsis joint model": "#7B5AA6",
}
INK = "#1A1A1A"
GRID = "#D9D9D9"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })


def save(fig: plt.Figure, name: str, dpi: int = 600) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def minimal_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure1_architecture() -> None:
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    blocks = [
        (0.04, 0.14, 0.20, 0.72, "1  INPUT", "Longest protein-coding\ntranscript per gene\n\nCurated phenotype labels\n\nCDS and protein sequence", "#DDEAF7", "#2F6DAE"),
        (0.29, 0.14, 0.20, 0.72, "2  FEATURES", "95 biological features\nsequence, physicochemical,\nGO, PPI and homology\n\n6,656 PLM dimensions\nESM2 | ProtBERT | ProtT5\n\n6,751 features per gene", "#E5F1E6", "#3D8B5B"),
        (0.54, 0.14, 0.20, 0.72, "3  LEAKAGE-AWARE\n   TRAINING", "Fold-specific imputation\nstandardization, PCA and\nsupervised selection\n\nExtraTrees | RandomForest\nLightGBM | XGBoost\n\n5-fold OOF stacking", "#EEE7F5", "#7B5AA6"),
        (0.79, 0.14, 0.17, 0.72, "4  OUTPUT", "Essential-gene\nprobability\n\nValidation-derived\nclassification threshold\n\nRanked candidates", "#F9E9D9", "#D17B28"),
    ]
    for i, (x, y, w, h, heading, body, fill, stroke) in enumerate(blocks):
        box = plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=1.0)
        ax.add_patch(box)
        ax.add_patch(plt.Rectangle((x, y+h-0.105), w, 0.105, facecolor=stroke, edgecolor=stroke))
        ax.text(x+0.015, y+h-0.052, heading, color="white", fontsize=6.5, fontweight="bold", va="center", linespacing=0.85)
        ax.text(x+w/2, y+h/2-0.015, body, color=INK, fontsize=7.1, va="center", ha="center", linespacing=1.38)
        if i < len(blocks)-1:
            ax.annotate("", xy=(blocks[i+1][0]-0.012, y+h/2), xytext=(x+w+0.012, y+h/2),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9))
    ax.text(0.04, 0.92, "Plant essential-gene prediction framework", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.04, 0.075, "All preprocessing that learns from data is fitted within training folds. Single-species and joint Arabidopsis-rice models use the same common feature space.", fontsize=6.7, color="#4A4A4A")
    save(fig, "fig1_model_architecture_nature")


def load_predictions() -> list[tuple[str, pd.DataFrame]]:
    items = [
        ("Rice single-species", ROOT / "models/rice_single_strict399_Tos17N4_common6751/fixed_test_predictions.tsv"),
        ("Rice joint model", ROOT / "models/joint_arabidopsis_rice_common6751/rice_fixed_test_predictions.tsv"),
        ("Arabidopsis single-species", ROOT / "models/arabidopsis_single_strict2601_common6751/shared_test_predictions.tsv"),
        ("Arabidopsis joint model", ROOT / "models/joint_arabidopsis_rice_common6751/ath_fixed_test_predictions.tsv"),
    ]
    return [(name, pd.read_csv(path, sep="\t")) for name, path in items]


def figure2_performance() -> None:
    records = load_predictions()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.0, 1.08, 1.08]})
    metrics = []
    for name, df in records:
        y = df["label"].astype(int).to_numpy()
        p = df["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        auc_value = auc(fpr, tpr)
        ap = average_precision_score(y, p)
        axes[1].plot(fpr, tpr, lw=1.45, color=COLORS[name], label=f"{name} ({auc_value:.3f})")
        axes[2].plot(recall, precision, lw=1.45, color=COLORS[name], label=f"{name} ({ap:.3f})")
        metrics.append((name, auc_value, ap))
    names = ["Rice\nsingle", "Rice\njoint", "Arabidopsis\nsingle", "Arabidopsis\njoint"]
    x = np.arange(len(metrics))
    axes[0].bar(x-0.18, [x[1] for x in metrics], 0.35, color="#4C78A8", label="AUC")
    axes[0].bar(x+0.18, [x[2] for x in metrics], 0.35, color="#F28E2B", label="AUPRC")
    axes[0].set_ylim(0.70, 1.00)
    axes[0].set_xticks(x, names, fontsize=5.9)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    axes[1].plot([0, 1], [0, 1], color="#B0B0B0", lw=0.8, ls="--")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    axes[2].legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=1.4)
    for i, ax in enumerate(axes):
        minimal_axes(ax); panel_label(ax, chr(97+i)); ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.7)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.92, wspace=0.38)
    save(fig, "fig2_main_performance_nature")


def figure3_ablation() -> None:
    values = pd.DataFrame({
        "Feature setting": ["Full 6,751", "Biological only", "All PLMs", "No GO", "No PPI"],
        "Rice": [0.8812, 0.8198, 0.8234, 0.8795, 0.8880],
        "Arabidopsis": [0.9203, 0.9024, 0.8045, 0.8270, 0.9189],
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    for ax, species, color in zip(axes, ["Rice", "Arabidopsis"], ["#2F6DAE", "#3D8B5B"]):
        y = np.arange(len(values))[::-1]
        vals = values[species].to_numpy()
        ax.hlines(y, 0.75, vals, color="#CCCCCC", lw=0.9)
        ax.scatter(vals, y, s=28, color=color, zorder=3)
        for xi, yi in zip(vals, y): ax.text(xi+0.0015, yi, f"{xi:.3f}", va="center", fontsize=6.3)
        ax.set(xlim=(0.75, 0.94), yticks=y, yticklabels=values["Feature setting"], xlabel="Fixed-test AUC", title=species)
        minimal_axes(ax); ax.grid(axis="x", color=GRID, lw=0.45)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.20, top=0.88, wspace=0.18)
    save(fig, "fig3_feature_ablation_nature")


def figure4_grouped_and_stats() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.0, 1.18]})
    species = ["Rice", "Arabidopsis"]
    fixed = [0.8812, 0.9175]
    grouped = [0.8101, 0.8793]
    x = np.arange(2)
    axes[0].plot([0,1], [fixed[0], fixed[1]], "o-", color="#4C78A8", lw=1.4, label="Fixed split")
    axes[0].plot([0,1], [grouped[0], grouped[1]], "o-", color="#E17C05", lw=1.4, label="Homology-grouped")
    axes[0].set(xticks=x, xticklabels=species, ylim=(0.75,0.94), ylabel="AUC")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    comparisons = ["Rice\nsingle - joint", "Arabidopsis\nsingle - joint"]
    diff = [-0.0091, 0.0026]
    p_text = ["P = 0.6426\nFDR = 0.7711", "P = 0.7861\nFDR = 0.7861"]
    yy = np.arange(2)[::-1]
    axes[1].axvline(0, color="#777777", lw=0.8)
    axes[1].scatter(diff, yy, s=35, color=["#D17B28", "#7B5AA6"], zorder=3)
    for dx, yi, text in zip(diff, yy, p_text): axes[1].text(0.012, yi, text, va="center", fontsize=6.7)
    axes[1].set(xlim=(-0.03,0.045), yticks=yy, yticklabels=comparisons, xlabel="AUC difference (single - joint)")
    for i, ax in enumerate(axes):
        minimal_axes(ax); ax.grid(axis="y", color=GRID, lw=0.45); panel_label(ax, chr(97+i))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.91, wspace=0.45)
    save(fig, "fig4_homology_and_delong_nature")


if __name__ == "__main__":
    setup()
    figure1_architecture()
    figure2_performance()
    figure3_ablation()
    figure4_grouped_and_stats()
    print(f"Wrote figures to {OUT}")
