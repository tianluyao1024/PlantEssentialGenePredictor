"""Generate a publication figure for the locked external-evidence boundary.

Conclusion defended by the figure: a documented source screen identified
audited independent Arabidopsis LoF records and two evidence-rich core
candidates, but the pre-registered external-cohort minimum was not met, so no
external AUC/AUPRC is reported.  Rice is shown explicitly as qualitative-only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "results" / "tpc_candidate_resource" / "external_validation"
SCREEN = ROOT / "results" / "tpc_candidate_resource" / "external_source_screening"
CARDS = ROOT / "results" / "tpc_candidate_resource"
OUT = ROOT / "results" / "manuscript_figures" / "figure7_external_evidence_boundary"

COLORS = {
    "ink": "#172A46",
    "slate": "#60738A",
    "blue": "#4E79A7",
    "pale_blue": "#DCE8F4",
    "green": "#4C956C",
    "pale_green": "#DDEFE4",
    "orange": "#E38B37",
    "pale_orange": "#FBE8D5",
    "red": "#B44B4B",
    "pale_red": "#F7DEDE",
    "gray": "#D9DEE5",
    "pale_gray": "#F3F5F7",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=COLORS["ink"], va="top")


def box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, face: str, edge: str = "#B9C4D0", size: float = 7, bold: bool = False) -> None:
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.025", linewidth=0.7, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size, color=COLORS["ink"], fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": COLORS["slate"]})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    screen = json.loads((SCREEN / "europe_pmc_source_screening_summary.json").read_text(encoding="utf-8"))
    release = json.loads((EXTERNAL / "external_validation_release_summary.json").read_text(encoding="utf-8"))
    metric = json.loads((EXTERNAL / "locked_external_cohort_metrics.json").read_text(encoding="utf-8"))
    card_ath = pd.read_csv(CARDS / "arabidopsis_final_core10_evidence_card_summary.tsv", sep="\t", dtype=str)
    card_rice = pd.read_csv(CARDS / "rice_final_core10_evidence_card_summary.tsv", sep="\t", dtype=str)

    panel_a = pd.DataFrame([
        ["Europe PMC screen", "articles", screen["articles"]],
        ["Europe PMC screen", "gene-level source records", screen["records"]],
        ["Europe PMC screen", "automated true-unknown eligible records", screen["eligible_true_unknown_rows"]],
        ["manual curation", "direct LoF records", release["curated_records"]],
        ["prelocked cohort", "Arabidopsis direct LoF records", release["prelocked_records"]],
    ], columns=["panel", "measure", "value"])
    ath_metrics = metric["species"]["arabidopsis"]
    panel_b = pd.DataFrame([
        ["Arabidopsis", "essential", ath_metrics["essential"], 10],
        ["Arabidopsis", "non-essential", ath_metrics["nonessential"], 10],
        ["Arabidopsis", "total", ath_metrics["n"], 30],
        ["Rice", "curated direct LoF", 0, 30],
    ], columns=["species", "measure", "observed", "pre_registered_minimum"])
    selected = pd.concat([
        card_ath.loc[card_ath["main_text_evidence_card_eligible"].eq("yes")],
        card_rice.loc[card_rice["main_text_evidence_card_eligible"].eq("yes")],
    ], ignore_index=True)
    panel_c = selected[["species", "gene_id", "single_species_probability", "independent_evidence_categories", "direct_evidence_direction", "main_text_evidence_card_eligible"]].copy()
    panel_d = pd.DataFrame([
        ["Arabidopsis", "evidence-card eligible", int(card_ath["main_text_evidence_card_eligible"].eq("yes").sum())],
        ["Arabidopsis", "prediction only", int(card_ath["main_text_evidence_card_eligible"].eq("no").sum())],
        ["Rice", "evidence-card eligible", int(card_rice["main_text_evidence_card_eligible"].eq("yes").sum())],
        ["Rice", "prediction only", int(card_rice["main_text_evidence_card_eligible"].eq("no").sum())],
    ], columns=["species", "evidence_status", "n_core_candidates"])
    source = pd.concat([
        panel_a.assign(panel="a"),
        panel_b.assign(panel="b"),
        panel_c.assign(panel="c"),
        panel_d.assign(panel="d"),
    ], ignore_index=True, sort=False)
    source.to_csv(OUT / "Figure7_source_data.tsv", sep="\t", index=False)
    release_data = ROOT / "data" / "external_validation" / "release"
    release_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "Figure7_source_data.tsv", release_data / "Figure7_source_data.tsv")

    fig = plt.figure(figsize=(7.15, 4.85), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.45], height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a, source screen and exclusion path
    ax_a.axis("off")
    add_panel_label(ax_a, "a")
    ax_a.set_title("Source-screening and exclusion path", loc="left", fontsize=8, fontweight="bold", pad=8)
    box(ax_a, 0.05, 0.73, 0.42, 0.16, f"Europe PMC\n{screen['articles']} articles", COLORS["pale_blue"], edge=COLORS["blue"], bold=True)
    box(ax_a, 0.55, 0.73, 0.40, 0.16, f"{screen['records']} gene-level\nsource records", COLORS["pale_blue"], edge=COLORS["blue"])
    arrow(ax_a, 0.47, 0.81, 0.55, 0.81)
    box(ax_a, 0.12, 0.45, 0.76, 0.16, f"{screen['eligible_true_unknown_rows']} automated true-unknown records\n(after labels, pseudo-labels and raw phenotype archives were excluded)", COLORS["pale_green"], edge=COLORS["green"])
    arrow(ax_a, 0.50, 0.73, 0.50, 0.61)
    box(ax_a, 0.12, 0.18, 0.76, 0.16, f"{release['curated_records']} curator-checked direct LoF records\nall passed zero-overlap audit", COLORS["pale_orange"], edge=COLORS["orange"])
    arrow(ax_a, 0.50, 0.45, 0.50, 0.34)
    ax_a.text(0.50, 0.04, "Search hits are not phenotype labels;\nmanual allele-level adjudication is required.", ha="center", va="center", fontsize=6.2, color=COLORS["slate"])

    # b, gate, not performance
    add_panel_label(ax_b, "b")
    ax_b.set_title("Pre-registered external-cohort gate", loc="left", fontsize=8, fontweight="bold", pad=8)
    labels = ["Arabidopsis\nessential", "Arabidopsis\nnon-essential", "Arabidopsis\ntotal", "Rice\ncurated"]
    values = panel_b["observed"].tolist()
    required = panel_b["pre_registered_minimum"].tolist()
    positions = list(range(len(labels)))
    ax_b.bar(positions, required, color=COLORS["pale_gray"], edgecolor=COLORS["gray"], width=0.62, label="pre-registered minimum")
    ax_b.bar(positions, values, color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["slate"]], width=0.42, label="locked evidence")
    for x, value, target in zip(positions, values, required):
        ax_b.text(x, max(value, 0) + 1.0, f"{value}/{target}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_b.set_ylim(0, 36)
    ax_b.set_xticks(positions, labels, fontsize=6.4)
    ax_b.set_ylabel("genes", fontsize=7)
    ax_b.legend(loc="upper left", fontsize=6, handlelength=1.2)
    ax_b.text(0.5, 0.76, "External metrics withheld\npre-registered cohort minimum not met", transform=ax_b.transAxes, ha="center", va="center", fontsize=6.8, color=COLORS["red"], fontweight="bold", bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["pale_red"], "edgecolor": COLORS["red"], "linewidth": 0.7})

    # c, eligible qualitative evidence cards
    ax_c.axis("off")
    add_panel_label(ax_c, "c")
    ax_c.set_title("Independent core-candidate evidence cards", loc="left", fontsize=8, fontweight="bold", pad=8)
    cards = [
        ("AT1G01970 (DG409)", "Psingle = 0.514", "A: CRISPR embryo lethality\nB: dual organellar PPR context", COLORS["pale_green"], COLORS["green"], "supports prediction"),
        ("AT4G01400 (MISF74)", "Psingle = 0.534", "A: viable but severe mutant\nB: mitochondrial splicing\nC: GABI-Kat material", COLORS["pale_orange"], COLORS["orange"], "qualitative counterexample"),
    ]
    for index, (title, prob, description, face, edge, footer) in enumerate(cards):
        y = 0.55 - index * 0.47
        box(ax_c, 0.04, y, 0.92, 0.36, title + "\n" + prob + "\n" + description + "\n" + footer, face, edge=edge, size=6.7, bold=False)
    ax_c.text(0.50, 0.02, "Evidence categories are explicit; cards are not a selected-case accuracy estimate.", ha="center", va="bottom", fontsize=6.0, color=COLORS["slate"])

    # d, candidate evidence coverage
    add_panel_label(ax_d, "d")
    ax_d.set_title("Evidence coverage of frozen core candidates", loc="left", fontsize=8, fontweight="bold", pad=8)
    species = ["Arabidopsis", "Rice"]
    eligible = [int(card_ath["main_text_evidence_card_eligible"].eq("yes").sum()), int(card_rice["main_text_evidence_card_eligible"].eq("yes").sum())]
    prediction_only = [10 - eligible[0], 10 - eligible[1]]
    ax_d.barh(species, eligible, color=COLORS["green"], label=">=2 independent categories")
    ax_d.barh(species, prediction_only, left=eligible, color=COLORS["gray"], label="prediction-only")
    for y, e, p in zip(species, eligible, prediction_only):
        ax_d.text(e / 2 if e else 0.3, y, str(e), ha="center" if e else "left", va="center", fontsize=7, color="white" if e else COLORS["slate"], fontweight="bold")
        ax_d.text(e + p / 2, y, str(p), ha="center", va="center", fontsize=7, color=COLORS["ink"], fontweight="bold")
    ax_d.set_xlim(0, 10)
    ax_d.set_xlabel("frozen core candidates", fontsize=7)
    ax_d.set_xticks([0, 2, 4, 6, 8, 10])
    ax_d.tick_params(axis="y", labelsize=7)
    ax_d.legend(loc="lower right", fontsize=6, handlelength=1.2)
    ax_d.text(0.03, -0.30, "Rice records remain a prediction-only resource until\nindependent direct LoF evidence passes the same audit.", transform=ax_d.transAxes, fontsize=6.3, color=COLORS["slate"])

    basename = OUT / "Figure7_external_evidence_boundary"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (OUT / "Figure7_QA_notes.md").write_text(
        "# Figure 7 QA notes\n\n"
        "Core conclusion: independent source screening produced direct Arabidopsis LoF records and two evidence-rich core-candidate cards, but the pre-registered minimum external cohort was not met; no external discrimination metric is shown.\n\n"
        "Source data: `Figure7_source_data.tsv`; screen summary, curated-record audit, frozen evaluator status and evidence-card summaries.\n\n"
        "No microscopy, image manipulation, biological replicate or inferential p-value panel is included. Bars are deterministic record counts, not estimates.\n",
        encoding="utf-8",
    )
    print(f"Wrote Figure 7 to {OUT}")


if __name__ == "__main__":
    main()
