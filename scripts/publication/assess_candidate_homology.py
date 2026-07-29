"""Assess closest labelled homologs for the frozen candidate lists with DIAMOND."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tpc_candidate_resource"
LABELS = ROOT / "data" / "labels"
ATH_3359 = Path(
    r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\096style_aug3359_pseudo14731_ricedata_only_cross\arabidopsis_3359_train_validation_test_split.tsv"
)
DIAMOND = Path(r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\tools\diamond\diamond.exe")
ATH_FASTA = Path(r"C:\Users\tly\Desktop\植物\拟南芥\Protein.fasta")
RICE_FASTA = Path(r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\rice_rapdb_native_features_fresh_only\rice_rapdb_native_longest_protein.fasta")
ATH_RE = re.compile(r"AT[1-5MC]G\d{5}", re.I)
RICE_GENE_RE = re.compile(r"Os\d{2}[gt]\d+", re.I)


def canonical_gene(identifier: str, species: str) -> str | None:
    pattern = ATH_RE if species == "arabidopsis" else RICE_GENE_RE
    match = pattern.search(identifier)
    if not match:
        return None
    gene = match.group(0)
    if species == "rice":
        gene = gene[:4] + "g" + gene[5:]
    return gene.upper()


def read_longest_proteins(path: Path, species: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_header = ""
    chunks: list[str] = []

    def save() -> None:
        gene = canonical_gene(current_header, species)
        sequence = "".join(chunks).replace("*", "").upper()
        if gene and sequence and (gene not in sequences or len(sequence) > len(sequences[gene])):
            sequences[gene] = sequence

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(">"):
                if current_header:
                    save()
                current_header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if current_header:
        save()
    return sequences


def labelled_genes(species: str) -> dict[str, str]:
    if species == "arabidopsis":
        labels = pd.read_csv(ATH_3359, sep="\t", dtype=str)
    else:
        labels = pd.read_csv(LABELS / "rice_raw_strict399_Tos17N4_labels.tsv", sep="\t", dtype=str)
    labels["gene_id_key"] = labels["gene_id"].str.upper()
    return dict(zip(labels["gene_id_key"], labels["label"]))


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for gene, sequence in sorted(records.items()):
            handle.write(f">{gene}\n{sequence}\n")


def run_diamond(species: str, candidates: pd.DataFrame) -> pd.DataFrame:
    fasta_path = ATH_FASTA if species == "arabidopsis" else RICE_FASTA
    proteins = read_longest_proteins(fasta_path, species)
    labels = labelled_genes(species)
    query_genes = candidates["gene_id_key"].tolist()
    query = {gene: proteins[gene] for gene in query_genes if gene in proteins}
    subject = {gene: proteins[gene] for gene in labels if gene in proteins}
    missing = sorted(set(query_genes) - set(query))
    if missing:
        raise RuntimeError(f"Missing candidate protein sequences for {species}: {missing[:10]}")
    work = OUT / "homology" / species
    work.mkdir(parents=True, exist_ok=True)
    query_fasta = work / "candidates.fasta"
    subject_fasta = work / "study_labelled_proteins.fasta"
    db = work / "study_labelled_proteins"
    hits = work / "candidate_to_labelled_hits.tsv"
    write_fasta(query_fasta, query)
    write_fasta(subject_fasta, subject)
    subprocess.run([str(DIAMOND), "makedb", "--in", str(subject_fasta), "-d", str(db)], check=True)
    subprocess.run(
        [
            str(DIAMOND), "blastp", "-d", str(db), "-q", str(query_fasta), "-o", str(hits),
            "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "scovhsp", "evalue", "bitscore",
            "--id", "20", "--query-cover", "30", "--subject-cover", "30", "--max-target-seqs", "50", "--threads", "8",
        ],
        check=True,
    )
    columns = ["gene_id_key", "closest_labelled_gene", "identity_percent", "query_coverage_percent", "subject_coverage_percent", "evalue", "bitscore"]
    if not hits.exists() or hits.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(hits, sep="\t", names=columns)
    table = table.loc[table["gene_id_key"].ne(table["closest_labelled_gene"])].copy()
    table = table.sort_values(["gene_id_key", "bitscore", "identity_percent"], ascending=[True, False, False])
    table = table.groupby("gene_id_key", as_index=False).first()
    table["closest_labelled_label"] = table["closest_labelled_gene"].map(labels)
    return table


def select_balanced_core(candidates: pd.DataFrame) -> pd.DataFrame:
    # A candidate must exceed 0.50 in all three frozen probability outputs in
    # addition to satisfying each model's validation-derived class threshold.
    # This is a presentation criterion for the core panel, not a retraining or
    # a new classification threshold for any primary evaluation.
    candidates = candidates.loc[
        candidates["single_species_probability"].ge(0.50)
        & candidates["joint_probability"].ge(0.50)
        & candidates["annotation_light_probability"].ge(0.50)
    ].sort_values("stability_adjusted_rank").copy()
    if len(candidates) < 10:
        raise RuntimeError("Fewer than ten strongly concordant candidates are available for the core panel.")
    isolated = candidates.loc[candidates["homology_category"].eq("homology_isolated")]
    supported = candidates.loc[candidates["homology_category"].eq("homology_supported_by_essential")]
    isolated_selected = isolated.head(5).copy()
    isolated_selected["selection_group"] = "homology_isolated"
    supported_selected = supported.head(5).copy()
    supported_selected["selection_group"] = "homology_supported_by_essential"
    selected = pd.concat([isolated_selected, supported_selected]).drop_duplicates("gene_id_key")
    if len(selected) < 10:
        fill = candidates.loc[~candidates["gene_id_key"].isin(selected["gene_id_key"])].head(10 - len(selected)).copy()
        fill["selection_group"] = "stability_ranked_fill"
        selected = pd.concat([selected, fill])
    selected = selected.sort_values("stability_adjusted_rank").head(10).copy()
    selected["final_core_candidate_rank"] = range(1, len(selected) + 1)
    return selected


def main() -> None:
    manifest = {
        "identity_threshold": 40,
        "coverage_threshold": 60,
        "core_panel_probability_requirement": "single_species_probability >= 0.50 AND joint_probability >= 0.50 AND annotation_light_probability >= 0.50",
        "interpretation": {},
    }
    for species in ["arabidopsis", "rice"]:
        candidate_path = OUT / f"{species}_provisional_true_unknown_candidates.tsv"
        candidates = pd.read_csv(candidate_path, sep="\t")
        # The candidate table is intentionally overwritten at every run.  Strip
        # prior homology annotations before merging fresh DIAMOND results, so a
        # rerun cannot silently retain stale nearest-homolog values.
        derived_columns = [
            "closest_labelled_gene", "closest_labelled_label", "identity_percent",
            "query_coverage_percent", "subject_coverage_percent", "evalue",
            "bitscore", "homology_category", "selection_group",
            "final_core_candidate_rank",
        ]
        candidates = candidates.drop(columns=[c for c in derived_columns if c in candidates.columns])
        # Only strongly concordant candidates need a sequence-neighbourhood
        # screen for the core panel. This avoids a broad homology survey that
        # could be mistaken for a new label source.
        strong = candidates.loc[
            candidates["single_species_probability"].ge(0.50)
            & candidates["joint_probability"].ge(0.50)
            & candidates["annotation_light_probability"].ge(0.50)
        ].copy()
        hits = run_diamond(species, strong)
        result = candidates.merge(hits, on="gene_id_key", how="left")
        high_similarity = (
            result["identity_percent"].ge(40)
            & result["query_coverage_percent"].ge(60)
            & result["subject_coverage_percent"].ge(60)
        )
        result["homology_category"] = "homology_isolated"
        result.loc[high_similarity & result["closest_labelled_label"].eq("1"), "homology_category"] = "homology_supported_by_essential"
        result.loc[high_similarity & result["closest_labelled_label"].eq("0"), "homology_category"] = "homology_supported_by_nonessential"
        result["closest_labelled_gene"] = result["closest_labelled_gene"].fillna("")
        result.to_csv(candidate_path, sep="\t", index=False)
        core = select_balanced_core(result)
        core.to_csv(OUT / f"{species}_final_core10_candidates.tsv", sep="\t", index=False)
        manifest["interpretation"][species] = {
            "strongly_concordant_candidates": int(len(strong)),
            "candidate_proteins_with_sequence": int(result["gene_id_key"].isin(hits["gene_id_key"]).sum()),
            "homology_categories_among_strong_candidates": result.loc[
                result["gene_id_key"].isin(strong["gene_id_key"]), "homology_category"
            ].value_counts().to_dict(),
        }
    (OUT / "homology_candidate_screen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
