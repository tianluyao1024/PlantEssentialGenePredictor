from __future__ import annotations

import gzip
import io
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen, Request

import networkx as nx
import numpy as np
import pandas as pd
import requests


ROOT = Path.cwd()
DESKTOP = Path.home() / "Desktop"
OUT = DESKTOP / "\u6c34\u7a3b" / "cross_species_ath_rice_common_features_models"
RAW = OUT / "external_raw"
ANNOT = OUT / "annotation_features"
RAW.mkdir(parents=True, exist_ok=True)
ANNOT.mkdir(parents=True, exist_ok=True)

RAP_MSU = ROOT / "RAP-MSU_2025-03-19.txt"
SPECIES = {
    "ath": {
        "dataset": "athaliana_eg_gene",
        "string_taxid": "3702",
        "feature_table": OUT / "arabidopsis_cross_species_common_features_all_genes.tsv",
        "protein_fasta": OUT / "ath" / "ath_longest_protein.fasta",
    },
    "rice": {
        "dataset": "osativa_eg_gene",
        "string_taxid": "4530",
        "feature_table": OUT / "rice_cross_species_common_features_all_genes.tsv",
        "protein_fasta": OUT / "rice" / "rice_longest_protein.fasta",
    },
}

EXPERIMENTAL_GO = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}


def load_target_genes(species: str) -> list[str]:
    df = pd.read_csv(SPECIES[species]["feature_table"], sep="\t", usecols=["gene_id"])
    return df["gene_id"].astype(str).tolist()


def load_rap_to_msu() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with RAP_MSU.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            rap, msu_str = line.rstrip("\n").split("\t")[:2]
            genes = []
            if msu_str != "None":
                for token in msu_str.split(","):
                    m = re.search(r"LOC_Os\d{2}g\d{5}", token)
                    if m:
                        genes.append(m.group(0))
            mapping[rap] = sorted(set(genes))
    return mapping


def biomart_query(dataset: str, attrs: list[str], out_path: Path) -> pd.DataFrame:
    if out_path.exists() and out_path.stat().st_size > 0:
        return pd.read_csv(out_path, sep="\t", dtype=str)
    attr_xml = "".join(f'<Attribute name="{a}" />' for a in attrs)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
    {attr_xml}
  </Dataset>
</Query>"""
    url = "https://plants.ensembl.org/biomart/martservice"
    r = requests.post(url, data={"query": xml}, timeout=180)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")
    return pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str)


def map_ensembl_to_target(species: str, ensembl_id: str, rap_to_msu: dict[str, list[str]]) -> list[str]:
    if not isinstance(ensembl_id, str) or not ensembl_id:
        return []
    if species == "ath":
        m = re.search(r"AT[1-5CM]G\d{5}", ensembl_id, re.I)
        return [m.group(0).upper()] if m else []
    return rap_to_msu.get(ensembl_id, [])


def build_biomart_features(species: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    target_genes = load_target_genes(species)
    rap_to_msu = load_rap_to_msu()
    dataset = SPECIES[species]["dataset"]
    raw_path = RAW / f"{species}_biomart_annotation.tsv"
    attrs = [
        "ensembl_gene_id",
        "chromosome_name",
        "start_position",
        "end_position",
        "strand",
        "transcript_count",
        "percentage_gene_gc_content",
        "gene_biotype",
        "go_id",
        "go_linkage_type",
        "namespace_1003",
        "goslim_goa_accession",
        "pfam",
        "interpro",
        "cdd",
        "gene3d",
        "string",
    ]
    df = biomart_query(dataset, attrs, raw_path)
    df.columns = attrs

    gene_rows = defaultdict(list)
    string_map = defaultdict(set)
    pos_rows = {}
    for row in df.itertuples(index=False):
        ens = getattr(row, "ensembl_gene_id")
        genes = map_ensembl_to_target(species, ens, rap_to_msu)
        for gene in genes:
            gene_rows[gene].append(row)
            s = getattr(row, "string")
            if isinstance(s, str) and s.strip():
                string_map[gene].add(s.strip())
            if gene not in pos_rows:
                pos_rows[gene] = {
                    "chromosome": getattr(row, "chromosome_name"),
                    "start": pd.to_numeric(getattr(row, "start_position"), errors="coerce"),
                    "end": pd.to_numeric(getattr(row, "end_position"), errors="coerce"),
                    "strand": pd.to_numeric(getattr(row, "strand"), errors="coerce"),
                }

    rows = []
    for gene in target_genes:
        items = gene_rows.get(gene, [])
        def uniq(attr):
            return {str(getattr(x, attr)).strip() for x in items if isinstance(getattr(x, attr), str) and str(getattr(x, attr)).strip()}
        go_terms = uniq("go_id")
        exp_go = {
            str(getattr(x, "go_id")).strip()
            for x in items
            if isinstance(getattr(x, "go_id"), str)
            and isinstance(getattr(x, "go_linkage_type"), str)
            and getattr(x, "go_linkage_type").strip() in EXPERIMENTAL_GO
        }
        ns_terms = defaultdict(set)
        for x in items:
            go = getattr(x, "go_id")
            ns = getattr(x, "namespace_1003")
            if isinstance(go, str) and go.strip() and isinstance(ns, str) and ns.strip():
                ns_terms[ns.strip().lower()].add(go.strip())
        pos = pos_rows.get(gene, {})
        rows.append(
            {
                "gene_id": gene,
                "annot_transcript_count": float(pd.to_numeric(getattr(items[0], "transcript_count"), errors="coerce")) if items else np.nan,
                "annot_gene_gc_percent": float(pd.to_numeric(getattr(items[0], "percentage_gene_gc_content"), errors="coerce")) if items else np.nan,
                "annot_is_protein_coding": float(1 if items and str(getattr(items[0], "gene_biotype")) == "protein_coding" else 0),
                "annot_go_count": len(go_terms),
                "annot_exp_go_count": len(exp_go),
                "annot_go_bp_count": len(ns_terms.get("biological_process", set())),
                "annot_go_mf_count": len(ns_terms.get("molecular_function", set())),
                "annot_go_cc_count": len(ns_terms.get("cellular_component", set())),
                "annot_goslim_count": len(uniq("goslim_goa_accession")),
                "annot_interpro_count": len(uniq("interpro")),
                "annot_pfam_count": len(uniq("pfam")),
                "annot_cdd_count": len(uniq("cdd")),
                "annot_gene3d_count": len(uniq("gene3d")),
                "annot_domain_total_count": len(uniq("interpro")) + len(uniq("pfam")) + len(uniq("cdd")) + len(uniq("gene3d")),
                "genomic_chromosome_numeric": chrom_numeric(pos.get("chromosome", "")),
                "genomic_gene_start": float(pos.get("start")) if pos else np.nan,
                "genomic_gene_end": float(pos.get("end")) if pos else np.nan,
                "genomic_gene_span": float(pos.get("end") - pos.get("start") + 1) if pos and pd.notna(pos.get("start")) and pd.notna(pos.get("end")) else np.nan,
                "genomic_strand": float(pos.get("strand")) if pos else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANNOT / f"{species}_biomart_stat_features.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"gene_id": g, "string_ids": ";".join(sorted(v))} for g, v in string_map.items()]
    ).to_csv(ANNOT / f"{species}_gene_to_string.tsv", sep="\t", index=False)
    return out, {g: sorted(v) for g, v in string_map.items()}


def chrom_numeric(chrom) -> float:
    s = str(chrom)
    if s.lower().startswith("chr"):
        s = s[3:]
    if s.isdigit():
        return float(int(s))
    if s in {"Mt", "M", "mitochondria"}:
        return 90.0
    if s in {"Pt", "C", "chloroplast"}:
        return 91.0
    return np.nan


def download_url(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=300) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def build_string_features(species: str, string_map: dict[str, list[str]]) -> pd.DataFrame:
    target_genes = load_target_genes(species)
    taxid = SPECIES[species]["string_taxid"]
    links_gz = RAW / f"{taxid}.protein.links.v12.0.txt.gz"
    info_gz = RAW / f"{taxid}.protein.info.v12.0.txt.gz"
    download_url(f"https://stringdb-downloads.org/download/protein.links.v12.0/{taxid}.protein.links.v12.0.txt.gz", links_gz)
    download_url(f"https://stringdb-downloads.org/download/protein.info.v12.0/{taxid}.protein.info.v12.0.txt.gz", info_gz)

    protein_to_gene = {}
    for gene, proteins in string_map.items():
        for p in proteins:
            protein_to_gene[p] = gene
            protein_to_gene[p.split(".", 1)[-1]] = gene

    graphs = {400: nx.Graph(), 700: nx.Graph(), 900: nx.Graph()}
    for graph in graphs.values():
        graph.add_nodes_from(target_genes)
    with gzip.open(links_gz, "rt", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip().split()
        idx1, idx2, idxs = header.index("protein1"), header.index("protein2"), header.index("combined_score")
        for line in handle:
            parts = line.strip().split()
            if len(parts) <= idxs:
                continue
            g1 = protein_to_gene.get(parts[idx1])
            g2 = protein_to_gene.get(parts[idx2])
            if not g1 or not g2 or g1 == g2:
                continue
            score = float(parts[idxs])
            for cutoff, graph in graphs.items():
                if score >= cutoff:
                    old = graph.get_edge_data(g1, g2, default={}).get("weight", 0.0)
                    if score > old:
                        graph.add_edge(g1, g2, weight=score)

    rows = []
    clustering = {cut: nx.clustering(g) for cut, g in graphs.items()}
    for gene in target_genes:
        row = {"gene_id": gene}
        for cutoff, graph in graphs.items():
            nbrs = list(graph.neighbors(gene))
            weights = [graph[gene][n]["weight"] for n in nbrs]
            row[f"string{cutoff}_degree"] = len(nbrs)
            row[f"string{cutoff}_weighted_degree"] = float(sum(weights))
            row[f"string{cutoff}_mean_score"] = float(np.mean(weights)) if weights else 0.0
            row[f"string{cutoff}_max_score"] = float(np.max(weights)) if weights else 0.0
            row[f"string{cutoff}_clustering"] = float(clustering[cutoff].get(gene, 0.0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ANNOT / f"{species}_string_network_features.tsv", sep="\t", index=False)
    return out


def ensure_diamond() -> Path:
    found = shutil.which("diamond")
    if found:
        return Path(found)
    tool_dir = OUT / "tools" / "diamond"
    exe = tool_dir / "diamond.exe"
    if exe.exists():
        return exe
    tool_dir.mkdir(parents=True, exist_ok=True)
    api = requests.get("https://api.github.com/repos/bbuchfink/diamond/releases/latest", timeout=60).json()
    assets = api.get("assets", [])
    url = None
    for asset in assets:
        name = asset.get("name", "").lower()
        if "windows" in name and name.endswith(".zip"):
            url = asset["browser_download_url"]
            break
    if url is None:
        raise RuntimeError("No DIAMOND Windows zip asset found in latest GitHub release.")
    zip_path = tool_dir / "diamond_latest_windows.zip"
    download_url(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tool_dir)
    candidates = list(tool_dir.rglob("diamond.exe"))
    if not candidates:
        raise RuntimeError("DIAMOND executable not found after extracting release zip.")
    return candidates[0]


def fasta_to_gene_fasta(species: str) -> Path:
    src = SPECIES[species]["protein_fasta"]
    dst = ANNOT / f"{species}_diamond_gene_protein.fasta"
    if dst.exists():
        return dst
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if line.startswith(">"):
                gene = line[1:].split("|", 1)[0].strip()
                out.write(f">{gene}\n")
            else:
                out.write(line)
    return dst


def run_diamond_paralog(species: str) -> pd.DataFrame:
    target_genes = load_target_genes(species)
    diamond = ensure_diamond()
    fasta = fasta_to_gene_fasta(species)
    db = ANNOT / f"{species}_diamond_db"
    out_tsv = ANNOT / f"{species}_diamond_self.tsv"
    if not Path(str(db) + ".dmnd").exists():
        subprocess.run([str(diamond), "makedb", "--in", str(fasta), "-d", str(db)], check=True)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        subprocess.run(
            [
                str(diamond),
                "blastp",
                "-q",
                str(fasta),
                "-d",
                str(db),
                "-o",
                str(out_tsv),
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "evalue",
                "bitscore",
                "qlen",
                "slen",
                "--max-target-seqs",
                "200",
                "--evalue",
                "1e-5",
                "--threads",
                "8",
                "--very-sensitive",
            ],
            check=True,
        )

    best = {}
    family40 = defaultdict(set)
    family30 = defaultdict(set)
    with out_tsv.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            q, s, pid, length, evalue, bits, qlen, slen = line.rstrip("\n").split("\t")
            if q == s:
                continue
            pid_f = float(pid)
            bits_f = float(bits)
            cov = float(length) / max(1.0, min(float(qlen), float(slen)))
            if pid_f >= 40 and cov >= 0.4:
                family40[q].add(s)
            if pid_f >= 30 and cov >= 0.3:
                family30[q].add(s)
            if q not in best or bits_f > best[q]["bitscore"]:
                best[q] = {
                    "paralog_top_identity": pid_f,
                    "paralog_top_bitscore": bits_f,
                    "paralog_top_evalue_neglog10": -math.log10(max(float(evalue), 1e-300)),
                    "paralog_top_coverage": cov,
                }
    rows = []
    for gene in target_genes:
        row = {"gene_id": gene}
        row.update(best.get(gene, {
            "paralog_top_identity": 0.0,
            "paralog_top_bitscore": 0.0,
            "paralog_top_evalue_neglog10": 0.0,
            "paralog_top_coverage": 0.0,
        }))
        row["paralog_family_size_40cov40"] = len(family40.get(gene, set())) + 1
        row["paralog_family_size_30cov30"] = len(family30.get(gene, set())) + 1
        row["paralog_singleton_40cov40"] = float(1 if row["paralog_family_size_40cov40"] == 1 else 0)
        row["paralog_singleton_30cov30"] = float(1 if row["paralog_family_size_30cov30"] == 1 else 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ANNOT / f"{species}_diamond_paralog_features.tsv", sep="\t", index=False)
    return out


def build_species_annotation(species: str):
    print(f"Building annotation features: {species}", flush=True)
    bio, string_map = build_biomart_features(species)
    string_f = build_string_features(species, string_map)
    paralog = run_diamond_paralog(species)
    df = bio.merge(string_f, on="gene_id", how="left").merge(paralog, on="gene_id", how="left")
    df.to_csv(OUT / f"{species}_cross_species_annotation_features.tsv", sep="\t", index=False)
    return df


def main():
    summary = {}
    for species in ["rice", "ath"]:
        df = build_species_annotation(species)
        summary[species] = {"genes": int(len(df)), "feature_count": int(df.shape[1] - 1)}
    (ANNOT / "annotation_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
