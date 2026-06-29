from __future__ import annotations

import csv
import gzip
import math
import shutil
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


BASE = Path("D:/\u62df\u5357\u82a5/\u6587\u732e\u7279\u5f81\u590d\u73b0")
IN_DIR = BASE / "paper_style_features_araport11_latest"
DOWNLOADS = BASE / "downloads"
OUT_DIR = BASE / "paper_style_features_araport11_latest_modern_replacements"

EXPR_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/"
    "E-MTAB-7978/E-MTAB-7978-tpms.tsv"
)
BIOMART_URL = "https://plants.ensembl.org/biomart/martservice"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def parse_replicate_cell(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            pass
    return float(np.median(vals)) if vals else np.nan


def load_expression_matrix(expr_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    header = pd.read_csv(expr_path, sep="\t", nrows=0).columns.tolist()
    group_cols = [c for c in header if c not in {"GeneID", "Gene Name"}]
    gene_ids: list[str] = []
    rows: list[list[float]] = []

    for chunk in pd.read_csv(expr_path, sep="\t", chunksize=2000):
        for _, row in chunk.iterrows():
            gene = str(row["GeneID"]).upper()
            values = [parse_replicate_cell(row[c]) for c in group_cols]
            if not any(np.isfinite(values)):
                continue
            gene_ids.append(gene)
            rows.append(values)

    mat = np.asarray(rows, dtype=np.float32)
    return gene_ids, mat, group_cols


def compute_expression_features(
    genes: list[str],
    expr_gene_ids: list[str],
    expr_tpm: np.ndarray,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    expr_index = {g: i for i, g in enumerate(expr_gene_ids)}
    log_expr = np.log2(np.nan_to_num(expr_tpm, nan=0.0) + 1.0).astype(np.float32)

    # MiniBatchKMeans approximates the paper's k=2000 expression modules while
    # staying tractable on the current Expression Atlas RNA-seq matrix.
    k = min(2000, log_expr.shape[0])
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=1,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    clusters = km.fit_predict(log_expr)
    cluster_sizes = Counter(clusters)

    out = pd.DataFrame({"gene_id": genes})
    medians = []
    variations = []
    breadths = []
    modules = []
    atlas_present = []
    for gene in genes:
        idx = expr_index.get(gene)
        if idx is None:
            medians.append(np.nan)
            variations.append(np.nan)
            breadths.append(np.nan)
            modules.append(np.nan)
            atlas_present.append(0)
            continue
        tpm = np.nan_to_num(expr_tpm[idx], nan=0.0)
        log_vals = log_expr[idx]
        median_tpm = float(np.median(tpm))
        mad_tpm = float(np.median(np.abs(tpm - median_tpm)))
        medians.append(float(np.median(log_vals)))
        variations.append(float(mad_tpm / median_tpm) if median_tpm > 0 else np.nan)
        breadths.append(float(np.sum(tpm > 1.0)))
        modules.append(float(cluster_sizes[clusters[idx]]))
        atlas_present.append(1)

    out["median_expression"] = medians
    out["expression_variation"] = variations
    out["expression_breadth"] = breadths
    out["expression_module_size"] = modules
    out["expression_atlas_present"] = atlas_present

    corrs = []
    for _, row in feature_df.iterrows():
        gene = str(row["gene_id"]).upper()
        paralog = str(row.get("top_paralog_gene", "")).upper()
        i = expr_index.get(gene)
        j = expr_index.get(paralog)
        if i is None or j is None:
            corrs.append(np.nan)
            continue
        x = log_expr[i]
        y = log_expr[j]
        if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    out["expression_correlation"] = corrs
    return out


def biomart_query(attributes: list[str], dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    attrs = "\n".join(f'    <Attribute name="{a}" />' for a in attributes)
    query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
  <Dataset name="athaliana_eg_gene" interface="default">
{attrs}
  </Dataset>
</Query>"""
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(BIOMART_URL, data=data)
    with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def load_biomart_context(genes: list[str]) -> pd.DataFrame:
    biotype_path = DOWNLOADS / "ensembl_plants_gene_biotype.tsv"
    paralog_path = DOWNLOADS / "ensembl_plants_arabidopsis_paralogs.tsv"
    biomart_query(["ensembl_gene_id", "gene_biotype"], biotype_path)
    biomart_query(
        [
            "ensembl_gene_id",
            "athaliana_eg_paralog_ensembl_gene",
            "athaliana_eg_paralog_perc_id",
            "athaliana_eg_paralog_orthology_type",
            "athaliana_eg_paralog_subtype",
        ],
        paralog_path,
    )

    gene_set = set(genes)
    biotype = {}
    with biotype_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("Gene stable ID", row.get("ensembl_gene_id", "")).upper()
            val = row.get("Gene type", row.get("gene_biotype", ""))
            if gene in gene_set:
                biotype[gene] = val

    paralog_count = defaultdict(set)
    max_pid = defaultdict(lambda: np.nan)
    subtype_counts = defaultdict(Counter)
    with paralog_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("Gene stable ID", row.get("ensembl_gene_id", "")).upper()
            if gene not in gene_set:
                continue
            paralog = row.get("Arabidopsis thaliana paralogue gene stable ID", "")
            if paralog:
                paralog_count[gene].add(paralog.upper())
            pid_text = row.get(
                "Paralogue %id. target Arabidopsis thaliana gene identical to query gene",
                "",
            )
            try:
                pid = float(pid_text)
                if np.isnan(max_pid[gene]) or pid > max_pid[gene]:
                    max_pid[gene] = pid
            except ValueError:
                pass
            subtype = row.get("Paralogue last common ancestor with Arabidopsis thaliana", "")
            if subtype:
                subtype_counts[gene][subtype] += 1

    rows = []
    for gene in genes:
        bt = biotype.get(gene, "")
        rows.append(
            {
                "gene_id": gene,
                "ensembl_gene_biotype_is_pseudogene": float("pseudogene" in bt.lower()),
                "ensembl_compara_paralog_count": float(len(paralog_count.get(gene, set()))),
                "ensembl_compara_max_paralog_percent_identity": max_pid[gene],
                "ensembl_compara_paralog_lca_type_count": float(
                    len(subtype_counts.get(gene, {}))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr_path = DOWNLOADS / "E-MTAB-7978-tpms.tsv"
    download(EXPR_URL, expr_path)

    feature_path = IN_DIR / "paper_style_features_all.tsv"
    df = pd.read_csv(feature_path, sep="\t")
    df["gene_id"] = df["gene_id"].astype(str).str.upper()
    genes = df["gene_id"].tolist()

    expr_gene_ids, expr_tpm, group_cols = load_expression_matrix(expr_path)
    expr_features = compute_expression_features(genes, expr_gene_ids, expr_tpm, df)
    biomart_features = load_biomart_context(genes)

    updated = df.copy()
    replacement_cols = [
        "median_expression",
        "expression_variation",
        "expression_breadth",
        "expression_correlation",
        "expression_module_size",
    ]
    expr_by_gene = expr_features.set_index("gene_id")
    for col in replacement_cols:
        updated[col] = updated["gene_id"].map(expr_by_gene[col])
    updated["expression_correlation_ks_lt_2"] = np.nan
    updated["expression_atlas_present"] = updated["gene_id"].map(
        expr_by_gene["expression_atlas_present"]
    )

    bio_by_gene = biomart_features.set_index("gene_id")
    for col in biomart_features.columns:
        if col == "gene_id":
            continue
        updated[col] = updated["gene_id"].map(bio_by_gene[col])

    metadata_cols = ["seq_id", "gene_id", "label", "source_fasta", "top_paralog_gene"]
    numeric_cols = [c for c in updated.columns if c not in metadata_cols]
    matrix = updated[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)

    updated.to_csv(OUT_DIR / "paper_style_features_all_modern_replacements.tsv", sep="\t", index=False)
    np.save(OUT_DIR / "paper_style_features_all_modern_replacements.npy", matrix)
    np.save(OUT_DIR / "all_ids.npy", updated["gene_id"].to_numpy(dtype=object))
    np.save(OUT_DIR / "all_labels.npy", updated["label"].to_numpy(dtype=np.int64))
    pd.DataFrame({"feature_name": numeric_cols}).to_csv(
        OUT_DIR / "feature_names.tsv", sep="\t", index=False
    )

    old_status = pd.read_csv(IN_DIR / "feature_status.tsv", sep="\t")
    status_updates = {
        "median_expression": (
            "modern_equivalent",
            "Filled with Expression Atlas E-MTAB-7978 RNA-seq median log2(TPM+1) across 54 organism-part/stage groups.",
        ),
        "expression_variation": (
            "modern_equivalent",
            "Filled with Expression Atlas E-MTAB-7978 median absolute deviation divided by median TPM.",
        ),
        "expression_breadth": (
            "modern_equivalent",
            "Filled with number of Expression Atlas E-MTAB-7978 groups with TPM > 1.",
        ),
        "expression_correlation": (
            "modern_equivalent",
            "Filled with Pearson correlation of log2(TPM+1) between the gene and its top current DIAMOND paralog.",
        ),
        "expression_module_size": (
            "modern_equivalent",
            "Filled with MiniBatchKMeans k=2000 module size from Expression Atlas E-MTAB-7978 log2(TPM+1).",
        ),
        "expression_correlation_ks_lt_2": (
            "unavailable",
            "No current paralog Ks table was available, so the Ks<2-specific expression correlation remains NaN.",
        ),
    }
    old_status = old_status.copy()
    for feature, (status, note) in status_updates.items():
        mask = old_status["feature_name"] == feature
        old_status.loc[mask, "status"] = status
        old_status.loc[mask, "note"] = note

    extra_status = pd.DataFrame(
        [
            {
                "feature_name": "expression_atlas_present",
                "status": "modern_context",
                "note": "1 if the gene was present in Expression Atlas E-MTAB-7978.",
            },
            {
                "feature_name": "ensembl_gene_biotype_is_pseudogene",
                "status": "modern_context_not_paper_equivalent",
                "note": "Current Ensembl Plants gene biotype flag; this is not the paper's homologous pseudogene-present feature.",
            },
            {
                "feature_name": "ensembl_compara_paralog_count",
                "status": "modern_duplication_context",
                "note": "Current Ensembl Plants Compara Arabidopsis paralog count; not an alpha/beta/gamma WGD block replacement.",
            },
            {
                "feature_name": "ensembl_compara_max_paralog_percent_identity",
                "status": "modern_duplication_context",
                "note": "Maximum current Ensembl Plants Compara Arabidopsis paralog percent identity.",
            },
            {
                "feature_name": "ensembl_compara_paralog_lca_type_count",
                "status": "modern_duplication_context",
                "note": "Number of Compara paralog last-common-ancestor subtype labels observed for the gene.",
            },
        ]
    )
    status = pd.concat([old_status, extra_status], ignore_index=True)
    status.to_csv(OUT_DIR / "feature_status_modern_replacements.tsv", sep="\t", index=False)

    summary_lines = [
        f"rows\t{updated.shape[0]}",
        f"numeric_feature_columns\t{len(numeric_cols)}",
        f"matrix_shape\t{matrix.shape[0]}x{matrix.shape[1]}",
        f"expression_atlas_genes\t{len(expr_gene_ids)}",
        f"expression_atlas_groups\t{len(group_cols)}",
        f"expression_atlas_present_in_matrix\t{int(updated['expression_atlas_present'].sum())}",
        "filled_replacement_columns\t"
        + ",".join(replacement_cols + ["expression_atlas_present"]),
        "added_context_columns\tensembl_gene_biotype_is_pseudogene,"
        "ensembl_compara_paralog_count,ensembl_compara_max_paralog_percent_identity,"
        "ensembl_compara_paralog_lca_type_count",
        "still_unreplaced\talpha_wgd_duplicate_retained,beta_gamma_wgd_duplicate_retained,"
        "pseudogene_present,expression_correlation_ks_lt_2,gene_body_methylated,"
        "nucleotide_diversity,percentage_identity_in_metazoans,percentage_identity_in_fungi,"
        "core_eukaryotic_gene,KaKs_columns",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
