from __future__ import annotations

import gzip
import io
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
COMMON = ROOT / "cross_species_ath_rice_common_features_models"
FRESH = COMMON / "rice_rapdb_native_features_fresh_only"
EXTERNAL = COMMON / "external_raw_stable"
OUT_GO = FRESH / "rice_rapdb_native_go_features.tsv"
OUT_PPI = FRESH / "rice_rapdb_native_ppi_features.tsv"
OUT_BOTH = FRESH / "rice_rapdb_native_go_ppi_features.tsv"
MANIFEST = FRESH / "rice_rapdb_native_go_ppi_features_manifest.json"

RAP_GENE_RE = re.compile(r"\bOs\d{2}g\d{7}\b")
RAP_TX_RE = re.compile(r"\bOs(\d{2})t(\d{7})(?:-\d+)?\b")

BIOMART_GO_URL = "https://plants.ensembl.org/biomart/martservice"
STRING_ALIAS_URL = (
    "https://stringdb-downloads.org/download/protein.aliases.v12.0/"
    "39947.protein.aliases.v12.0.txt.gz"
)
STRING_LINKS = EXTERNAL / "39947.protein.links.v12.0.txt.gz"
STRING_ALIASES = EXTERNAL / "39947.protein.aliases.v12.0.txt.gz"
GO_RAW = EXTERNAL / "ensembl_plants_osativa_RAP2022-09-01_go.tsv"


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(path)


def load_gene_ids() -> list[str]:
    numeric = pd.read_csv(FRESH / "rice_rapdb_native_sequence_numeric_features.tsv", sep="\t", usecols=["gene_id"])
    genes = numeric["gene_id"].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate gene_id values in fresh RAP-DB numeric feature table.")
    return genes


def fetch_go_raw() -> None:
    if GO_RAW.exists() and GO_RAW.stat().st_size > 1000:
        return
    query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
  <Dataset name="osativa_eg_gene" interface="default">
    <Filter name="source" value="RAP2022-09-01" />
    <Filter name="with_go" excluded="0" />
    <Attribute name="ensembl_gene_id" />
    <Attribute name="external_gene_name" />
    <Attribute name="external_synonym" />
    <Attribute name="source" />
    <Attribute name="go_id" />
    <Attribute name="namespace_1003" />
  </Dataset>
</Query>"""
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(BIOMART_GO_URL, data=data)
    tmp = GO_RAW.with_suffix(GO_RAW.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=240) as response:
        tmp.write_bytes(response.read())
    tmp.replace(GO_RAW)


def build_go_features(genes: list[str], top_n_terms: int | None = None) -> tuple[pd.DataFrame, dict]:
    fetch_go_raw()
    raw = pd.read_csv(GO_RAW, sep="\t", dtype=str).fillna("")
    raw.columns = ["gene_id", "gene_name", "gene_synonym", "source", "go_id", "go_domain"]
    raw = raw[(raw["source"] == "RAP2022-09-01") & raw["gene_id"].str.match(r"^Os\d{2}g\d{7}$")]
    raw = raw[raw["go_id"].str.match(r"^GO:\d{7}$")]

    gene_set = set(genes)
    raw = raw[raw["gene_id"].isin(gene_set)]
    pairs = raw[["gene_id", "go_id", "go_domain"]].drop_duplicates()
    go_counter = Counter(pairs["go_id"].tolist())
    top_terms = [term for term, _ in go_counter.most_common(top_n_terms)]

    records: dict[str, dict[str, float]] = {
        gene: {
            "gene_id": gene,
            "go_count_total": 0.0,
            "go_count_bp": 0.0,
            "go_count_mf": 0.0,
            "go_count_cc": 0.0,
            "go_count_unknown_domain": 0.0,
            "go_has_any": 0.0,
        }
        for gene in genes
    }
    for gene, sub in pairs.groupby("gene_id", sort=False):
        domains = sub["go_domain"].astype(str).tolist()
        records[gene]["go_count_total"] = float(sub["go_id"].nunique())
        records[gene]["go_count_bp"] = float(sum(d == "biological_process" for d in domains))
        records[gene]["go_count_mf"] = float(sum(d == "molecular_function" for d in domains))
        records[gene]["go_count_cc"] = float(sum(d == "cellular_component" for d in domains))
        records[gene]["go_count_unknown_domain"] = float(sum(d == "" for d in domains))
        records[gene]["go_has_any"] = 1.0

    go_index = {term: i for i, term in enumerate(top_terms)}
    mat = np.zeros((len(genes), len(top_terms)), dtype=np.uint8)
    gene_to_i = {gene: i for i, gene in enumerate(genes)}
    for gene, term in pairs[["gene_id", "go_id"]].itertuples(index=False):
        j = go_index.get(term)
        if j is not None:
            mat[gene_to_i[gene], j] = 1
    base = pd.DataFrame.from_records([records[gene] for gene in genes])
    top = pd.DataFrame(mat, columns=[f"go_term_{term.replace(':', '_')}" for term in top_terms])
    out = pd.concat([base, top], axis=1)
    out.to_csv(OUT_GO, sep="\t", index=False)
    manifest = {
        "source": "Ensembl Plants BioMart osativa_eg_gene",
        "filter": "source=RAP2022-09-01 and with_go=true",
        "raw_path": str(GO_RAW),
        "genes_with_go": int((base["go_has_any"] > 0).sum()),
        "unique_go_terms_total": int(len(go_counter)),
        "top_go_terms_encoded": int(len(top_terms)),
        "raw_rows_after_rap_filter": int(len(raw)),
        "deduplicated_gene_go_pairs": int(len(pairs)),
        "id_policy": "Only direct RAP/IRGSP OsXXg... gene IDs returned by BioMart were used.",
    }
    return out, manifest


def alias_to_rap_gene(alias: str) -> str | None:
    m = RAP_GENE_RE.search(alias)
    if m:
        return m.group(0)
    m = RAP_TX_RE.search(alias)
    if m:
        return f"Os{m.group(1)}g{m.group(2)}"
    return None


def build_string_alias_map(gene_set: set[str]) -> tuple[dict[str, str], dict]:
    download_if_missing(STRING_ALIAS_URL, STRING_ALIASES)
    hits: dict[str, set[str]] = defaultdict(set)
    direct_alias_rows = 0
    with gzip.open(STRING_ALIASES, "rt", encoding="utf-8", errors="replace") as handle:
        header = next(handle, "")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            protein, alias = parts[0], parts[1]
            gene = alias_to_rap_gene(alias)
            if gene and gene in gene_set:
                hits[protein].add(gene)
                direct_alias_rows += 1
    protein_to_gene = {protein: next(iter(gs)) for protein, gs in hits.items() if len(gs) == 1}
    manifest = {
        "source": "STRING v12.0 Oryza sativa Japonica Group 39947 protein.aliases",
        "alias_path": str(STRING_ALIASES),
        "direct_rap_alias_rows": int(direct_alias_rows),
        "string_proteins_with_direct_rap_alias": int(len(hits)),
        "string_proteins_kept_unambiguous": int(len(protein_to_gene)),
        "ambiguous_string_proteins_skipped": int(len(hits) - len(protein_to_gene)),
        "id_policy": "Only STRING aliases directly matching RAP/IRGSP OsXXg... or OsXXt... IDs were used.",
    }
    return protein_to_gene, manifest


def build_ppi_features(genes: list[str]) -> tuple[pd.DataFrame, dict]:
    if not STRING_LINKS.exists():
        raise FileNotFoundError(f"Missing STRING links file: {STRING_LINKS}")
    gene_set = set(genes)
    protein_to_gene, alias_manifest = build_string_alias_map(gene_set)
    gene_to_i = {gene: i for i, gene in enumerate(genes)}
    n = len(genes)

    thresholds = [150, 400, 700, 900]
    deg = {thr: np.zeros(n, dtype=np.float32) for thr in thresholds}
    wdeg = {thr: np.zeros(n, dtype=np.float32) for thr in thresholds}
    max_score = np.zeros(n, dtype=np.float32)
    score_sum = np.zeros(n, dtype=np.float32)
    edge_count = np.zeros(n, dtype=np.float32)
    neighbor_sets = {thr: [set() for _ in range(n)] for thr in thresholds}

    mapped_edges = 0
    total_link_rows = 0
    with gzip.open(STRING_LINKS, "rt", encoding="utf-8", errors="replace") as handle:
        header = next(handle, "")
        for line in handle:
            total_link_rows += 1
            p1, p2, score_s = line.rstrip("\n").split()
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if g1 is None or g2 is None or g1 == g2:
                continue
            score = float(score_s)
            i, j = gene_to_i[g1], gene_to_i[g2]
            mapped_edges += 1
            for idx, other in ((i, j), (j, i)):
                edge_count[idx] += 1.0
                score_sum[idx] += score
                if score > max_score[idx]:
                    max_score[idx] = score
                for thr in thresholds:
                    if score >= thr:
                        neighbor_sets[thr][idx].add(other)

    records = []
    for i, gene in enumerate(genes):
        row = {
            "gene_id": gene,
            "ppi_string_edge_count": float(edge_count[i]),
            "ppi_string_score_sum": float(score_sum[i]),
            "ppi_string_score_mean": float(score_sum[i] / edge_count[i]) if edge_count[i] else 0.0,
            "ppi_string_score_max": float(max_score[i]),
            "ppi_string_has_any": 1.0 if edge_count[i] else 0.0,
        }
        for thr in thresholds:
            neighbors = neighbor_sets[thr][i]
            row[f"ppi_string_degree_ge{thr}"] = float(len(neighbors))
            if neighbors:
                vals = []
                # A compact local-neighborhood feature: how connected the direct neighbors are.
                for nb in neighbors:
                    vals.append(len(neighbor_sets[thr][nb]))
                row[f"ppi_string_neighbor_degree_mean_ge{thr}"] = float(np.mean(vals))
                row[f"ppi_string_neighbor_degree_max_ge{thr}"] = float(np.max(vals))
                row[f"ppi_string_degree_log1p_ge{thr}"] = float(math.log1p(len(neighbors)))
            else:
                row[f"ppi_string_neighbor_degree_mean_ge{thr}"] = 0.0
                row[f"ppi_string_neighbor_degree_max_ge{thr}"] = 0.0
                row[f"ppi_string_degree_log1p_ge{thr}"] = 0.0
        records.append(row)
    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT_PPI, sep="\t", index=False)
    manifest = {
        **alias_manifest,
        "links_path": str(STRING_LINKS),
        "total_string_link_rows": int(total_link_rows),
        "mapped_direct_rap_link_rows": int(mapped_edges),
        "genes_with_mapped_ppi": int(out["ppi_string_has_any"].sum()),
        "ppi_feature_count": int(out.shape[1] - 1),
        "thresholds": thresholds,
    }
    return out, manifest


def main() -> None:
    genes = load_gene_ids()
    go, go_manifest = build_go_features(genes)
    ppi, ppi_manifest = build_ppi_features(genes)
    merged = go.merge(ppi, on="gene_id", how="outer")
    merged = merged.set_index("gene_id").reindex(genes).fillna(0.0).reset_index()
    merged.to_csv(OUT_BOTH, sep="\t", index=False)
    manifest = {
        "gene_count": int(len(genes)),
        "go_feature_path": str(OUT_GO),
        "ppi_feature_path": str(OUT_PPI),
        "combined_feature_path": str(OUT_BOTH),
        "combined_feature_count": int(merged.shape[1] - 1),
        "go": go_manifest,
        "ppi": ppi_manifest,
        "critical_note": (
            "GO and PPI features are fresh RAP/IRGSP-native features. No MSU/RGAP mapping, "
            "no rekeyed features, and no old feature tables are used."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
