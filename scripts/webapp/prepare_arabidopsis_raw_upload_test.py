from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "cds": Path(r"C:\Users\tly\Desktop\植物\拟南芥\Araport11_cds.fasta"),
    "protein": Path(r"C:\Users\tly\Desktop\植物\拟南芥\Araport11_pep.fasta"),
    "gff": Path(
        r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\external_raw_stable\Arabidopsis_thaliana.TAIR10.63.gff3.gz"
    ),
    "go": Path(r"C:\Users\tly\Desktop\植物\拟南芥\split_nonessential\ATH_GO_GOSLIM.txt\ATH_GO_GOSLIM.txt"),
    "string_links": Path(
        r"C:\Users\tly\Desktop\植物\拟南芥\split_nonessential\3702.protein.links.v12.0.txt\3702.protein.links.v12.0.txt"
    ),
    "tair_uniprot": Path(r"C:\Users\tly\Desktop\植物\拟南芥\split_nonessential\TAIR2UniprotMapping.txt"),
    "labels": ROOT / "data" / "labels" / "arabidopsis_validation_labels.tsv",
}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def gene_id(text: str) -> str | None:
    match = re.search(r"AT[1-5CM]G\d{5}", text, re.I)
    return match.group(0).upper() if match else None


def read_fasta_longest(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    current_id: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal current_id, chunks
        if current_id is None:
            return
        gid = gene_id(current_id)
        seq = "".join(chunks).replace("*", "").upper()
        if gid and (gid not in out or len(seq) > len(out[gid][1])):
            out[gid] = (current_id, seq)
        current_id = None
        chunks = []

    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].split()[0]
            else:
                chunks.append(line)
    flush()
    return out


def wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def write_fasta(records: dict[str, tuple[str, str]], genes: list[str], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for gid in genes:
            if gid not in records:
                continue
            _, seq = records[gid]
            handle.write(f">{gid}\n{wrap(seq)}\n")
            count += 1
    return count


def prepare_go(src: Path, genes: set[str], out: Path) -> int:
    rows = set()
    with open_text(src) as handle:
        for line in handle:
            if not line.strip() or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 5:
                gid = gene_id(parts[0])
                go = parts[5]
                if gid in genes and go.startswith("GO:"):
                    rows.add((gid, go))
    with out.open("w", encoding="utf-8") as handle:
        handle.write("gene_id\tgo_id\n")
        for gid, go in sorted(rows):
            handle.write(f"{gid}\t{go}\n")
    return len(rows)


def prepare_gff(src: Path, genes: set[str], out: Path) -> int:
    n = 0
    with open_text(src) as inp, out.open("w", encoding="utf-8") as handle:
        handle.write("##gff-version 3\n")
        for line in inp:
            if not line.strip() or line.startswith("#"):
                continue
            gid = gene_id(line)
            if gid in genes:
                handle.write(line)
                n += 1
    return n


def load_uniprot_mapping(path: Path) -> dict[str, str]:
    mapping = {}
    with open_text(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            accession = parts[0].split("-", 1)[0]
            gid = gene_id(parts[2])
            if accession and gid:
                mapping[accession] = gid
    return mapping


def string_to_gene(string_id: str, mapping: dict[str, str]) -> str | None:
    accession = string_id.split(".", 1)[-1].split("-", 1)[0]
    return mapping.get(accession)


def prepare_ppi(links: Path, mapping_path: Path, genes: set[str], out: Path) -> int:
    mapping = load_uniprot_mapping(mapping_path)
    rows = {}
    with open_text(links) as handle:
        header = handle.readline().split()
        i1, i2, iscore = header.index("protein1"), header.index("protein2"), header.index("combined_score")
        for line in handle:
            parts = line.split()
            if len(parts) <= iscore:
                continue
            g1 = string_to_gene(parts[i1], mapping)
            g2 = string_to_gene(parts[i2], mapping)
            if not g1 or not g2 or g1 == g2:
                continue
            if g1 not in genes and g2 not in genes:
                continue
            score = int(parts[iscore])
            key = tuple(sorted([g1, g2]))
            if score > rows.get(key, 0):
                rows[key] = score
    with out.open("w", encoding="utf-8") as handle:
        handle.write("gene_a\tgene_b\tscore\n")
        for (g1, g2), score in sorted(rows.items()):
            handle.write(f"{g1}\t{g2}\t{score}\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a realistic Arabidopsis raw-upload test package.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "webapp_data" / "jobs" / "arabidopsis_raw_upload_test")
    parser.add_argument("--n-genes", type=int, default=120)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(DEFAULTS["labels"], sep="\t")
    if "gene_id" not in labels.columns:
        raise RuntimeError("Label table must contain gene_id")
    candidates = [gid for gid in labels["gene_id"].astype(str).map(str.upper).drop_duplicates() if gene_id(gid)]
    cds = read_fasta_longest(DEFAULTS["cds"])
    prot = read_fasta_longest(DEFAULTS["protein"])
    plm_ids = np.load(
        Path(r"E:\CodexMoved\Desktop\水稻\cross_species_ath_rice_common_features_models\plm_embeddings\esm2\ath\all_ids.npy"),
        allow_pickle=True,
    ).astype(str)
    plm_genes = {gene_id(item) for item in plm_ids if gene_id(item)}
    genes = [gid for gid in candidates if gid in cds and gid in prot and gid in plm_genes][: args.n_genes]
    if not genes:
        raise RuntimeError("No genes matched CDS, protein and PLM IDs")

    counts = {
        "selected_genes": len(genes),
        "protein_records": write_fasta(prot, genes, args.out_dir / "protein.fasta"),
        "cds_records": write_fasta(cds, genes, args.out_dir / "cds.fasta"),
        "gff_rows": prepare_gff(DEFAULTS["gff"], set(genes), args.out_dir / "annotation.gff3"),
        "go_rows": prepare_go(DEFAULTS["go"], set(genes), args.out_dir / "go_annotation.tsv"),
        "ppi_edges": prepare_ppi(DEFAULTS["string_links"], DEFAULTS["tair_uniprot"], set(genes), args.out_dir / "ppi_edges.tsv"),
        "expression_matrix": "not_available_in_local_raw_sources",
        "domain_annotation": "not_available_in_local_raw_sources",
    }
    (args.out_dir / "upload_manifest.json").write_text(
        json.dumps({"genes": genes, "counts": counts, "source_files": {k: str(v) for k, v in DEFAULTS.items()}}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), **counts}, indent=2))


if __name__ == "__main__":
    main()
