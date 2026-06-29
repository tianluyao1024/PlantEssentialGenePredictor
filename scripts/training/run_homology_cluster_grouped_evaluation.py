from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

import train_joint_ath2601_rice_strict399_common6751 as joint
import train_rice_E0_Nge6_common6751_fixed80_10_10 as trainer


ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/"
    "cross_species_ath_rice_common_features_models"
)
OUT = ROOT / "paper_submission_experiments/homology_cluster_grouped"
DIAMOND = ROOT / "tools/diamond/diamond.exe"
ATH_FASTA = ROOT / "ath/ath_longest_protein.fasta"
RICE_FASTA = (
    ROOT
    / "rice_rapdb_native_features_fresh_only/"
    "rice_rapdb_native_longest_protein.fasta"
)
SEED = 20260622


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def read_fasta(path: Path) -> dict[str, str]:
    result = {}
    current = None
    chunks = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current is not None:
                    result[current] = "".join(chunks)
                current = line[1:].split("|")[0].split()[0].split(".")[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if current is not None:
        result[current] = "".join(chunks)
    return result


def write_cluster_fasta(
    ath_genes: set[str], rice_genes: set[str]
) -> Path:
    ath = read_fasta(ATH_FASTA)
    rice = read_fasta(RICE_FASTA)
    missing_ath = ath_genes - set(ath)
    missing_rice = rice_genes - set(rice)
    if missing_ath or missing_rice:
        raise RuntimeError(
            f"Missing proteins: ath={len(missing_ath)}, rice={len(missing_rice)}"
        )
    path = OUT / "all_labeled_proteins.fasta"
    with path.open("w") as handle:
        for gene in sorted(ath_genes):
            handle.write(f">arabidopsis|{gene}\n{ath[gene]}\n")
        for gene in sorted(rice_genes):
            handle.write(f">rice|{gene}\n{rice[gene]}\n")
    return path


def make_clusters(fasta: Path) -> pd.DataFrame:
    db = OUT / "labeled_proteins"
    hits = OUT / "diamond_similarity_edges.tsv"
    if not hits.exists():
        subprocess.run(
            [str(DIAMOND), "makedb", "--in", str(fasta), "-d", str(db)],
            check=True,
        )
        subprocess.run(
            [
                str(DIAMOND),
                "blastp",
                "-d",
                str(db),
                "-q",
                str(fasta),
                "-o",
                str(hits),
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "pident",
                "qcovhsp",
                "scovhsp",
                "evalue",
                "bitscore",
                "--id",
                "40",
                "--query-cover",
                "60",
                "--subject-cover",
                "60",
                "--evalue",
                "1e-5",
                "--max-target-seqs",
                "200",
                "--threads",
                "8",
            ],
            check=True,
        )
    ids = []
    with fasta.open() as handle:
        ids = [line[1:].strip() for line in handle if line.startswith(">")]
    union = UnionFind(ids)
    edge_table = pd.read_csv(
        hits,
        sep="\t",
        names=[
            "query",
            "subject",
            "identity",
            "query_coverage",
            "subject_coverage",
            "evalue",
            "bitscore",
        ],
    )
    for query, subject in edge_table[["query", "subject"]].itertuples(
        index=False
    ):
        if query != subject:
            union.union(query, subject)
    roots = {}
    rows = []
    for identifier in ids:
        root = union.find(identifier)
        if root not in roots:
            roots[root] = f"cluster_{len(roots) + 1:05d}"
        species, gene = identifier.split("|", 1)
        rows.append(
            {
                "sequence_id": identifier,
                "species": species,
                "gene_id": gene,
                "homology_cluster": roots[root],
            }
        )
    clusters = pd.DataFrame(rows)
    clusters["cluster_size"] = clusters.groupby("homology_cluster")[
        "gene_id"
    ].transform("size")
    clusters.to_csv(OUT / "homology_clusters.tsv", sep="\t", index=False)
    return clusters


def grouped_split(
    labels: pd.DataFrame, cluster_map: dict[str, str], stratify: pd.Series
) -> pd.DataFrame:
    frame = labels.copy().reset_index(drop=True)
    frame["homology_cluster"] = frame["gene_id"].map(cluster_map)
    if frame["homology_cluster"].isna().any():
        raise RuntimeError("Missing cluster assignments")
    outer = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=SEED
    )
    train_val_idx, test_idx = next(
        outer.split(frame, stratify, frame["homology_cluster"])
    )
    train_val = frame.iloc[train_val_idx].reset_index()
    stratify_train_val = stratify.iloc[train_val_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(
        n_splits=9, shuffle=True, random_state=SEED + 1
    )
    train_local, validation_local = next(
        inner.split(
            train_val,
            stratify_train_val,
            train_val["homology_cluster"],
        )
    )
    frame["split"] = ""
    frame.loc[test_idx, "split"] = "test"
    frame.loc[
        train_val.loc[validation_local, "index"].to_numpy(int), "split"
    ] = "validation"
    frame.loc[
        train_val.loc[train_local, "index"].to_numpy(int), "split"
    ] = "train"
    return frame


def fit_dataset(
    name,
    matrix,
    meta,
    split,
    training_additions=None,
):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.tsv"
    if result_path.exists():
        return pd.read_csv(result_path, sep="\t").iloc[0].to_dict()
    training = split[split["split"].eq("train")].copy()
    validation = split[split["split"].eq("validation")].copy()
    test = split[split["split"].eq("test")].copy()
    if training_additions is not None:
        heldout_clusters = set(validation["homology_cluster"]) | set(
            test["homology_cluster"]
        )
        additions = training_additions[
            ~training_additions["homology_cluster"].isin(heldout_clusters)
        ].copy()
        training = pd.concat([training, additions], ignore_index=True)
    indices = {
        "train": joint.index_genes(meta, training),
        "validation": joint.index_genes(meta, validation),
        "test": joint.index_genes(meta, test),
    }
    if "training_component" in training:
        weights = np.where(
            training["training_component"].eq("strict_concordant_addition"),
            0.5,
            1.0,
        ).astype(np.float32)
    else:
        weights = np.ones(len(training), dtype=np.float32)
    _, predictions, folds, _, _, _ = trainer.fit_library(
        matrix[indices["train"]],
        training["label"].to_numpy(np.int8),
        weights,
        {
            "validation": matrix[indices["validation"]],
            "test": matrix[indices["test"]],
        },
        95,
    )
    rows = []
    thresholds = {}
    for method in ["meta", "mean", "logit_mean"]:
        probability = predictions["validation"][method]
        threshold = joint.select_single_species_threshold(
            validation["label"].to_numpy(np.int8), probability
        )
        thresholds[method] = threshold
        rows.append(
            {
                "method": method,
                "validation_auc": roc_auc_score(
                    validation["label"], probability
                ),
                "validation_auprc": average_precision_score(
                    validation["label"], probability
                ),
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["validation_auc", "validation_auprc"], ascending=False
    )
    method = str(candidates.iloc[0]["method"])
    threshold = thresholds[method]["threshold"]
    score = joint.metrics(
        test["label"].to_numpy(np.int8),
        predictions["test"][method],
        threshold,
    )
    result = {
        "model": name,
        "n_train": len(training),
        "n_validation": len(validation),
        "n_test": len(test),
        "train_clusters": training["homology_cluster"].nunique(),
        "validation_clusters": validation["homology_cluster"].nunique(),
        "test_clusters": test["homology_cluster"].nunique(),
        "selected_method": method,
        **score,
    }
    split.to_csv(out / "core_grouped_split.tsv", sep="\t", index=False)
    training.to_csv(out / "actual_training_labels.tsv", sep="\t", index=False)
    pd.DataFrame(folds).to_csv(
        out / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    pd.DataFrame([result]).to_csv(result_path, sep="\t", index=False)
    return result


def matrix_for_rows(rows, rice_x, rice_meta, ath_x, ath_meta):
    rice_lookup = dict(
        zip(rice_meta["gene_id"].astype(str), np.arange(len(rice_meta)))
    )
    ath_lookup = dict(
        zip(ath_meta["gene_id"].astype(str), np.arange(len(ath_meta)))
    )
    matrices = []
    for row in rows.itertuples(index=False):
        if row.species == "rice":
            matrices.append(rice_x[rice_lookup[row.gene_id]])
        else:
            matrices.append(ath_x[ath_lookup[row.gene_id]])
    return np.asarray(matrices, dtype=np.float32)


def fit_joint_grouped(
    rice_x,
    rice_meta,
    ath_x,
    ath_meta,
    split,
    ath_additions,
):
    name = "joint_grouped_common6751"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.tsv"
    if result_path.exists():
        return pd.read_csv(result_path, sep="\t").to_dict("records")
    training = split[split["split"].eq("train")].copy()
    validation = split[split["split"].eq("validation")].copy()
    test = split[split["split"].eq("test")].copy()
    heldout = set(validation["homology_cluster"]) | set(
        test["homology_cluster"]
    )
    additions = ath_additions[
        ~ath_additions["homology_cluster"].isin(heldout)
    ].copy()
    additions["species"] = "arabidopsis"
    training = pd.concat([training, additions], ignore_index=True)
    training["raw_weight"] = np.where(
        training.get("training_component", "").eq(
            "strict_concordant_addition"
        ),
        0.5,
        1.0,
    )
    for species in ["rice", "arabidopsis"]:
        mask = training["species"].eq(species)
        training.loc[mask, "sample_weight"] = (
            training.loc[mask, "raw_weight"]
            / training.loc[mask, "raw_weight"].sum()
        )
    training["sample_weight"] *= (
        len(training) / training["sample_weight"].sum()
    )
    _, predictions, folds, _, _, _ = trainer.fit_library(
        matrix_for_rows(training, rice_x, rice_meta, ath_x, ath_meta),
        training["label"].to_numpy(np.int8),
        training["sample_weight"].to_numpy(np.float32),
        {
            "rice_validation": matrix_for_rows(
                validation[validation["species"].eq("rice")],
                rice_x,
                rice_meta,
                ath_x,
                ath_meta,
            ),
            "rice_test": matrix_for_rows(
                test[test["species"].eq("rice")],
                rice_x,
                rice_meta,
                ath_x,
                ath_meta,
            ),
            "ath_validation": matrix_for_rows(
                validation[validation["species"].eq("arabidopsis")],
                rice_x,
                rice_meta,
                ath_x,
                ath_meta,
            ),
            "ath_test": matrix_for_rows(
                test[test["species"].eq("arabidopsis")],
                rice_x,
                rice_meta,
                ath_x,
                ath_meta,
            ),
        },
        95,
    )
    methods = []
    thresholds = {}
    rice_val_y = validation.loc[
        validation["species"].eq("rice"), "label"
    ].to_numpy(np.int8)
    ath_val_y = validation.loc[
        validation["species"].eq("arabidopsis"), "label"
    ].to_numpy(np.int8)
    for method in ["meta", "mean", "logit_mean"]:
        selected, _ = joint.select_joint_threshold(
            rice_val_y,
            predictions["rice_validation"][method],
            ath_val_y,
            predictions["ath_validation"][method],
        )
        thresholds[method] = selected
        rice_auc = roc_auc_score(
            rice_val_y, predictions["rice_validation"][method]
        )
        ath_auc = roc_auc_score(
            ath_val_y, predictions["ath_validation"][method]
        )
        methods.append(
            {
                "method": method,
                "min_validation_auc": min(rice_auc, ath_auc),
                "mean_validation_auc": np.mean([rice_auc, ath_auc]),
                "min_validation_sp": selected["min_sp"],
            }
        )
    method = str(
        pd.DataFrame(methods)
        .sort_values(
            ["min_validation_auc", "mean_validation_auc", "min_validation_sp"],
            ascending=False,
        )
        .iloc[0]["method"]
    )
    threshold = thresholds[method]["threshold"]
    rows = []
    for species, key, subset in [
        ("rice", "rice_test", test[test["species"].eq("rice")]),
        (
            "arabidopsis",
            "ath_test",
            test[test["species"].eq("arabidopsis")],
        ),
    ]:
        score = joint.metrics(
            subset["label"].to_numpy(np.int8),
            predictions[key][method],
            threshold,
        )
        rows.append(
            {
                "model": name,
                "evaluation_species": species,
                "n_train": len(training),
                "n_validation": len(validation),
                "n_test_species": len(subset),
                "selected_method": method,
                **score,
            }
        )
    split.to_csv(out / "grouped_core_split.tsv", sep="\t", index=False)
    training.to_csv(
        out / "actual_training_labels.tsv", sep="\t", index=False
    )
    pd.DataFrame(folds).to_csv(
        out / "inner_oof_fold_scores.tsv", sep="\t", index=False
    )
    pd.DataFrame(rows).to_csv(result_path, sep="\t", index=False)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rice_x, names, rice_meta, rice_fixed = joint.load_rice()
    ath_x, ath_names, ath_meta, ath_fixed = joint.load_ath()
    if names != ath_names:
        raise RuntimeError("Feature schemas differ")
    rice_labels = pd.concat(rice_fixed.values()).drop_duplicates("gene_id")
    ath_training = ath_fixed["train"].copy()
    ath_core = pd.read_csv(joint.ATH_CORE_SPLIT, sep="\t")
    ath_core["gene_id"] = ath_core["gene_id"].astype(str).str.upper()
    ath_core["label"] = ath_core["label"].astype(np.int8)
    ath_additions = ath_training[
        ~ath_training["training_component"].eq("core1623_train80")
    ].copy()

    clusters = make_clusters(
        write_cluster_fasta(
            set(ath_training["gene_id"]) | set(ath_core["gene_id"]),
            set(rice_labels["gene_id"]) & set(rice_meta["gene_id"]),
        )
    )
    cluster_map = dict(zip(clusters["gene_id"], clusters["homology_cluster"]))
    rice_labels = rice_labels[
        rice_labels["gene_id"].isin(rice_meta["gene_id"])
    ].copy()
    rice_split = grouped_split(
        rice_labels,
        cluster_map,
        rice_labels["label"].astype(str),
    )
    ath_split = grouped_split(
        ath_core,
        cluster_map,
        ath_core["label"].astype(str),
    )
    ath_additions["homology_cluster"] = ath_additions["gene_id"].map(
        cluster_map
    )

    results = []
    results.append(
        fit_dataset(
            "rice_grouped_common6751",
            rice_x,
            rice_meta,
            rice_split,
        )
    )
    combined_core = pd.concat(
        [
            rice_labels.assign(species="rice"),
            ath_core.assign(species="arabidopsis"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_split = grouped_split(
        combined_core,
        cluster_map,
        combined_core["species"].astype(str)
        + "_"
        + combined_core["label"].astype(str),
    )
    results.extend(
        fit_joint_grouped(
            rice_x,
            rice_meta,
            ath_x,
            ath_meta,
            combined_split,
            ath_additions,
        )
    )
    results.append(
        fit_dataset(
            "ath_strict2601_grouped_common6751",
            ath_x,
            ath_meta,
            ath_split,
            ath_additions,
        )
    )
    pd.DataFrame(results).to_csv(
        OUT / "homology_grouped_results.tsv", sep="\t", index=False
    )
    summary = {
        "clustering": (
            "DIAMOND edges >=40% identity, >=60% query and subject coverage; "
            "connected components define homology clusters"
        ),
        "clusters": int(clusters["homology_cluster"].nunique()),
        "multi_gene_clusters": int(
            clusters.loc[clusters["cluster_size"].gt(1), "homology_cluster"].nunique()
        ),
        "results": results,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
