from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


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
            if gene_id(line) in genes:
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


def prepare_ppi(links: Path, mapping_path: Path, genes: set[str], out: Path) -> int:
    mapping = load_uniprot_mapping(mapping_path)
    rows: dict[tuple[str, str], int] = {}
    with open_text(links) as handle:
        header = handle.readline().split()
        i1, i2, iscore = header.index("protein1"), header.index("protein2"), header.index("combined_score")
        for line in handle:
            parts = line.split()
            if len(parts) <= iscore:
                continue
            a = mapping.get(parts[i1].split(".", 1)[-1].split("-", 1)[0])
            b = mapping.get(parts[i2].split(".", 1)[-1].split("-", 1)[0])
            if not a or not b or a == b or (a not in genes and b not in genes):
                continue
            key = tuple(sorted((a, b)))
            rows[key] = max(rows.get(key, 0), int(parts[iscore]))
    with out.open("w", encoding="utf-8") as handle:
        handle.write("gene_a\tgene_b\tscore\n")
        for (a, b), score in sorted(rows.items()):
            handle.write(f"{a}\t{b}\t{score}\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create website-style Arabidopsis raw-upload files from user-supplied source paths.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "webapp_data" / "jobs" / "arabidopsis_raw_upload_test")
    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--cds", required=True, type=Path, help="Araport11 CDS FASTA.")
    parser.add_argument("--protein", required=True, type=Path, help="Araport11 protein FASTA.")
    parser.add_argument("--gff", required=True, type=Path, help="Arabidopsis GFF3 or GFF3.GZ.")
    parser.add_argument("--go", required=True, type=Path, help="TAIR GO annotation file.")
    parser.add_argument("--string-links", required=True, type=Path, help="STRING protein links file.")
    parser.add_argument("--tair-uniprot", required=True, type=Path, help="TAIR-to-UniProt mapping file.")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "arabidopsis_validation_labels.tsv")
    parser.add_argument("--plm-ids", type=Path, default=None, help="Optional all_ids.npy used to select genes represented in a cached PLM matrix.")
    args = parser.parse_args()

    sources = {
        "cds": args.cds,
        "protein": args.protein,
        "gff": args.gff,
        "go": args.go,
        "string_links": args.string_links,
        "tair_uniprot": args.tair_uniprot,
        "labels": args.labels,
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {', '.join(missing)}")
    if args.plm_ids is not None and not args.plm_ids.exists():
        raise FileNotFoundError(f"PLM ID file does not exist: {args.plm_ids}")

    labels = pd.read_csv(args.labels, sep="\t")
    if "gene_id" not in labels.columns:
        raise RuntimeError("Label table must contain gene_id")
    candidates = [gid for gid in labels["gene_id"].astype(str).str.upper().drop_duplicates() if gene_id(gid)]
    cds = read_fasta_longest(args.cds)
    prot = read_fasta_longest(args.protein)
    plm_genes: set[str | None] | None = None
    if args.plm_ids is not None:
        plm_genes = {gene_id(item) for item in np.load(args.plm_ids, allow_pickle=True).astype(str)}
    genes = [gid for gid in candidates if gid in cds and gid in prot and (plm_genes is None or gid in plm_genes)][: args.n_genes]
    if not genes:
        raise RuntimeError("No genes matched the supplied labels, CDS and protein FASTA files")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "selected_genes": len(genes),
        "protein_records": write_fasta(prot, genes, args.out_dir / "protein.fasta"),
        "cds_records": write_fasta(cds, genes, args.out_dir / "cds.fasta"),
        "gff_rows": prepare_gff(args.gff, set(genes), args.out_dir / "annotation.gff3"),
        "go_rows": prepare_go(args.go, set(genes), args.out_dir / "go_annotation.tsv"),
        "ppi_edges": prepare_ppi(args.string_links, args.tair_uniprot, set(genes), args.out_dir / "ppi_edges.tsv"),
    }
    (args.out_dir / "upload_manifest.json").write_text(
        json.dumps({"genes": genes, "counts": counts, "source_files": {k: str(v) for k, v in sources.items()}}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), **counts}, indent=2))


if __name__ == "__main__":
    main()
