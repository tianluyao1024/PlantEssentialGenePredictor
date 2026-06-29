from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np
import pandas as pd
import requests


ROOT = Path.cwd()
DESKTOP = Path.home() / "Desktop"
OUT = DESKTOP / "\u6c34\u7a3b" / "cross_species_ath_rice_common_features_models"
RAW = OUT / "external_raw_stable"
FEAT = OUT / "stable_external_features"
RAW.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)

RAP_MSU = ROOT / "RAP-MSU_2025-03-19.txt"
RELEASE = 63
SPECIES = {
    "ath": {
        "taxid": "3702",
        "gff_url": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-{RELEASE}/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.{RELEASE}.gff3.gz",
        "features": OUT / "arabidopsis_cross_species_common_features_all_genes.tsv",
        "protein": OUT / "ath" / "ath_longest_protein.fasta",
    },
    "rice": {
        "taxid": "39947",
        "gff_url": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-{RELEASE}/gff3/oryza_sativa/Oryza_sativa.IRGSP-1.0.{RELEASE}.gff3.gz",
        "features": OUT / "rice_cross_species_common_features_all_genes.tsv",
        "protein": OUT / "rice" / "rice_longest_protein.fasta",
    },
}


def download_url(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=300) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def target_genes(species: str) -> list[str]:
    return pd.read_csv(SPECIES[species]["features"], sep="\t", usecols=["gene_id"])["gene_id"].astype(str).tolist()


def load_rap_to_msu() -> dict[str, list[str]]:
    out = {}
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
            out[rap] = sorted(set(genes))
    return out


def parse_attrs(attr_text: str) -> dict[str, str]:
    d = {}
    for part in attr_text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def chrom_numeric(chrom: str) -> float:
    s = str(chrom).replace("Chr", "").replace("chr", "")
    if s.isdigit():
        return float(s)
    if s in {"Mt", "M", "mitochondria"}:
        return 90.0
    if s in {"Pt", "C", "chloroplast"}:
        return 91.0
    return np.nan


def map_gene_id(species: str, raw_id: str, rap_to_msu: dict[str, list[str]]) -> list[str]:
    if species == "ath":
        m = re.search(r"AT[1-5CM]G\d{5}", raw_id, re.I)
        return [m.group(0).upper()] if m else []
    return rap_to_msu.get(raw_id, [])


def gff_features(species: str) -> pd.DataFrame:
    genes = target_genes(species)
    rap_to_msu = load_rap_to_msu()
    gff = RAW / Path(SPECIES[species]["gff_url"]).name
    download_url(SPECIES[species]["gff_url"], gff)
    gene_info = {}
    transcript_counts = defaultdict(int)
    child_to_gene = {}
    with gzip.open(gff, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            seqid, source, feature, start, end, score, strand, phase, attrs = line.rstrip("\n").split("\t")
            ad = parse_attrs(attrs)
            if feature == "gene":
                raw = ad.get("ID", "").replace("gene:", "")
                for gid in map_gene_id(species, raw, rap_to_msu):
                    gene_info[gid] = {
                        "gff_chromosome_numeric": chrom_numeric(seqid),
                        "gff_gene_start": float(start),
                        "gff_gene_end": float(end),
                        "gff_gene_span": float(int(end) - int(start) + 1),
                        "gff_strand": 1.0 if strand == "+" else -1.0 if strand == "-" else 0.0,
                    }
                    child_to_gene[raw] = gid
            elif feature in {"mRNA", "transcript"}:
                parent = ad.get("Parent", "").replace("gene:", "").split(",")[0]
                mapped = map_gene_id(species, parent, rap_to_msu)
                for gid in mapped:
                    transcript_counts[gid] += 1
    rows = []
    for gid in genes:
        row = {"gene_id": gid}
        row.update(gene_info.get(gid, {}))
        row["gff_transcript_count"] = float(transcript_counts.get(gid, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def string_gene_from_name(species: str, protein_id: str, preferred_name: str) -> str | None:
    if species == "ath":
        for text in [preferred_name, protein_id]:
            m = re.search(r"AT[1-5CM]G\d{5}", str(text), re.I)
            if m:
                return m.group(0).upper()
    else:
        for text in [preferred_name, protein_id]:
            m = re.search(r"LOC_Os\d{2}g\d{5}", str(text))
            if m:
                return m.group(0)
    return None


def string_features(species: str) -> pd.DataFrame:
    genes = target_genes(species)
    taxid = SPECIES[species]["taxid"]
    info = RAW / f"{taxid}.protein.info.v12.0.txt.gz"
    links = RAW / f"{taxid}.protein.links.v12.0.txt.gz"
    download_url(f"https://stringdb-downloads.org/download/protein.info.v12.0/{taxid}.protein.info.v12.0.txt.gz", info)
    download_url(f"https://stringdb-downloads.org/download/protein.links.v12.0/{taxid}.protein.links.v12.0.txt.gz", links)
    p2g = {}
    with gzip.open(info, "rt", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        ip = header.index("#string_protein_id") if "#string_protein_id" in header else 0
        iname = header.index("preferred_name") if "preferred_name" in header else 1
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(ip, iname):
                continue
            gid = string_gene_from_name(species, parts[ip], parts[iname])
            if gid:
                p2g[parts[ip]] = gid
                p2g[parts[ip].split(".", 1)[-1]] = gid
    graphs = {400: nx.Graph(), 700: nx.Graph(), 900: nx.Graph()}
    for graph in graphs.values():
        graph.add_nodes_from(genes)
    with gzip.open(links, "rt", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().split()
        i1, i2, iscore = header.index("protein1"), header.index("protein2"), header.index("combined_score")
        for line in handle:
            parts = line.split()
            if len(parts) <= iscore:
                continue
            g1, g2 = p2g.get(parts[i1]), p2g.get(parts[i2])
            if not g1 or not g2 or g1 == g2:
                continue
            score = float(parts[iscore])
            for cutoff, graph in graphs.items():
                if score >= cutoff:
                    old = graph.get_edge_data(g1, g2, default={}).get("weight", 0.0)
                    if score > old:
                        graph.add_edge(g1, g2, weight=score)
    cluster = {cut: nx.clustering(graph) for cut, graph in graphs.items()}
    rows = []
    for gid in genes:
        row = {"gene_id": gid}
        for cutoff, graph in graphs.items():
            nbrs = list(graph.neighbors(gid))
            weights = [graph[gid][n]["weight"] for n in nbrs]
            row[f"string{cutoff}_degree"] = len(nbrs)
            row[f"string{cutoff}_weighted_degree"] = float(sum(weights))
            row[f"string{cutoff}_mean_score"] = float(np.mean(weights)) if weights else 0.0
            row[f"string{cutoff}_max_score"] = float(np.max(weights)) if weights else 0.0
            row[f"string{cutoff}_clustering"] = float(cluster[cutoff].get(gid, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_diamond() -> Path:
    found = shutil.which("diamond")
    if found:
        return Path(found)
    tool_dir = OUT / "tools" / "diamond"
    exe_candidates = list(tool_dir.rglob("diamond.exe")) if tool_dir.exists() else []
    if exe_candidates:
        return exe_candidates[0]
    tool_dir.mkdir(parents=True, exist_ok=True)
    api = requests.get("https://api.github.com/repos/bbuchfink/diamond/releases/latest", timeout=60).json()
    url = None
    for asset in api.get("assets", []):
        name = asset.get("name", "").lower()
        if "windows" in name and name.endswith(".zip"):
            url = asset["browser_download_url"]
            break
    if not url:
        raise RuntimeError("No DIAMOND Windows zip asset in latest release.")
    zpath = tool_dir / "diamond_windows.zip"
    download_url(url, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tool_dir)
    exe_candidates = list(tool_dir.rglob("diamond.exe"))
    if not exe_candidates:
        raise RuntimeError("diamond.exe not found")
    return exe_candidates[0]


def gene_fasta(species: str) -> Path:
    src = SPECIES[species]["protein"]
    dst = FEAT / f"{species}_diamond_gene_protein.fasta"
    if dst.exists():
        return dst
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if line.startswith(">"):
                out.write(">" + line[1:].split("|", 1)[0].strip() + "\n")
            else:
                out.write(line)
    return dst


def diamond_features(species: str) -> pd.DataFrame:
    genes = target_genes(species)
    diamond = ensure_diamond()
    fasta = gene_fasta(species)
    db = FEAT / f"{species}_diamond_db"
    dmnd = Path(str(db) + ".dmnd")
    out_tsv = FEAT / f"{species}_diamond_self.tsv"
    if not dmnd.exists():
        subprocess.run([str(diamond), "makedb", "--in", str(fasta), "-d", str(db)], check=True)
    if not out_tsv.exists():
        subprocess.run(
            [
                str(diamond), "blastp", "-q", str(fasta), "-d", str(db), "-o", str(out_tsv),
                "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qlen", "slen",
                "--max-target-seqs", "200", "--evalue", "1e-5", "--threads", "8", "--very-sensitive",
            ],
            check=True,
        )
    best = {}
    fam40 = defaultdict(set)
    fam30 = defaultdict(set)
    with out_tsv.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            q, s, pid, length, evalue, bits, qlen, slen = line.rstrip("\n").split("\t")
            if q == s:
                continue
            pid = float(pid); bits = float(bits)
            cov = float(length) / max(1.0, min(float(qlen), float(slen)))
            if pid >= 40 and cov >= 0.4:
                fam40[q].add(s)
            if pid >= 30 and cov >= 0.3:
                fam30[q].add(s)
            if q not in best or bits > best[q]["paralog_top_bitscore"]:
                best[q] = {
                    "paralog_top_identity": pid,
                    "paralog_top_bitscore": bits,
                    "paralog_top_evalue_neglog10": -math.log10(max(float(evalue), 1e-300)),
                    "paralog_top_coverage": cov,
                }
    rows = []
    for gid in genes:
        row = {"gene_id": gid}
        row.update(best.get(gid, {
            "paralog_top_identity": 0.0,
            "paralog_top_bitscore": 0.0,
            "paralog_top_evalue_neglog10": 0.0,
            "paralog_top_coverage": 0.0,
        }))
        row["paralog_family_size_40cov40"] = len(fam40.get(gid, set())) + 1
        row["paralog_family_size_30cov30"] = len(fam30.get(gid, set())) + 1
        row["paralog_singleton_40cov40"] = float(row["paralog_family_size_40cov40"] == 1)
        row["paralog_singleton_30cov30"] = float(row["paralog_family_size_30cov30"] == 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build(species: str) -> pd.DataFrame:
    print(f"{species}: GFF", flush=True)
    gff = gff_features(species)
    print(f"{species}: STRING", flush=True)
    st = string_features(species)
    print(f"{species}: DIAMOND", flush=True)
    dia = diamond_features(species)
    df = gff.merge(st, on="gene_id", how="left").merge(dia, on="gene_id", how="left")
    df.to_csv(OUT / f"{species}_stable_external_features.tsv", sep="\t", index=False)
    return df


def main():
    summary = {}
    for species in ["rice", "ath"]:
        df = build(species)
        summary[species] = {"genes": int(len(df)), "features": int(df.shape[1] - 1)}
    (FEAT / "stable_external_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
