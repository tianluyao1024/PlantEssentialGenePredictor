from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


BASE = Path(r"C:\Users\tly\Desktop\植物\拟南芥")
SEQ_DIR = BASE / "Araport11_综合必需非必需_longest_sequences"
FEATURE_ROOT = Path(r"D:\拟南芥\特征\Araport11_综合必需非必需_features")
OUTDIR = FEATURE_ROOT / "literature_sequence_features"

CDS_FASTA = BASE / "Araport11_cds.fasta"
PEP_FASTA = BASE / "Araport11_pep.fasta"

GENE_RE = re.compile(r"^(AT[1-5CM]G\d{5})", re.I)
TRANSCRIPT_RE = re.compile(r"^(AT[1-5CM]G\d{5}\.\d+)", re.I)
COORD_RE = re.compile(r"\b(chr[1-5CM]):(\d+)-(\d+)\s+(FORWARD|REVERSE)\b", re.I)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
NT_ORDER = list("ACGT")
AA_GROUPS = {
    "hydrophobic": set("AILMFWYV"),
    "polar": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "small": set("AGSTP"),
    "aromatic": set("FWY"),
    "sulfur": set("CM"),
}

LITERATURE_FEATURE_MAP = [
    ("WGD duplicate retained", "external_required", "Needs published WGD retained duplicate annotation."),
    ("bg WGD duplicate retained", "external_required", "Needs published background/WGD annotation."),
    ("Pseudogene present", "external_required", "Needs pseudogene annotation and paralog relationship."),
    ("Tandem duplicate", "approximated", "Approximated by nearest same-chromosome homolog proxy within 100 kb."),
    ("Paralog Ks", "external_required", "Needs coding-sequence alignment and synonymous substitution estimation."),
    ("Gene family size", "approximated", "Approximated from protein k-mer similarity to Araport11 longest proteins."),
    ("Median expression", "external_required", "Needs expression atlas matrix."),
    ("Expression variation", "external_required", "Needs expression atlas matrix."),
    ("Expression breadth", "external_required", "Needs expression atlas matrix."),
    ("Expression correlation", "external_required", "Needs expression atlas and paralog map."),
    ("Expression correlation (Ks < 2)", "external_required", "Needs expression atlas and Ks-filtered paralog map."),
    ("Core eukaryotic gene", "external_required", "Needs CEG/KOG-style conserved gene annotation."),
    ("Homolog not found in rice", "external_required", "Needs rice homology search."),
    ("Percentage identity in plants", "external_required", "Needs cross-plant homolog BLAST/DIAMOND search."),
    ("Percentage identity in metazoans", "external_required", "Needs metazoan homolog BLAST/DIAMOND search."),
    ("Percentage identity in fungi", "external_required", "Needs fungal homolog BLAST/DIAMOND search."),
    ("A. lyrata homolog Ka/Ks", "external_required", "Needs ortholog CDS alignments and Ka/Ks estimation."),
    ("P. trichocarpa homolog Ka/Ks", "external_required", "Needs ortholog CDS alignments and Ka/Ks estimation."),
    ("V. vinifera homolog Ka/Ks", "external_required", "Needs ortholog CDS alignments and Ka/Ks estimation."),
    ("Rice homolog Ka/Ks", "external_required", "Needs ortholog CDS alignments and Ka/Ks estimation."),
    ("P. patens homolog Ka/Ks", "external_required", "Needs ortholog CDS alignments and Ka/Ks estimation."),
    ("Nucleotide diversity", "external_required", "Needs population SNP/diversity data."),
    ("Paralog Ka/Ks", "external_required", "Needs paralog CDS alignments and Ka/Ks estimation."),
    ("Expression module size", "external_required", "Needs expression network/module annotation."),
    ("Gene network connections", "external_required", "Needs AraNet or equivalent functional network."),
    ("Protein-protein interactions", "external_required", "Needs PPI network."),
    ("Gene body methylated", "external_required", "Needs methylome annotation."),
    ("Paralog percentage identity", "approximated", "Approximated from protein k-mer similarity; not BLAST identity."),
    ("Protein length", "extracted_exact", "Computed from longest CDS-matched protein."),
    ("Domain number", "external_required", "Needs Pfam/InterPro/hmmscan domain annotation."),
]


def normalize_gene(value: str) -> str:
    match = GENE_RE.search(value.strip())
    return match.group(1).upper() if match else ""


def transcript_id(header: str) -> str:
    token = header[1:].split()[0]
    match = TRANSCRIPT_RE.match(token)
    return match.group(1).upper() if match else token.upper()


def parse_fasta(path: Path):
    header = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)


def parse_coord(header: str) -> tuple[str, int, int, str]:
    match = COORD_RE.search(header)
    if not match:
        return "", 0, 0, ""
    chrom, start, end, strand = match.groups()
    return chrom.upper(), int(start), int(end), strand.upper()


def all_longest_records() -> dict[str, dict[str, object]]:
    proteins = {transcript_id(h): (h, s) for h, s in parse_fasta(PEP_FASTA)}
    longest: dict[str, dict[str, object]] = {}
    for header, cds in parse_fasta(CDS_FASTA):
        tid = transcript_id(header)
        gid = tid.split(".", 1)[0]
        previous = longest.get(gid)
        if previous is None or len(cds) > len(previous["cds"]):
            pep_header, pep = proteins.get(tid, ("", ""))
            chrom, start, end, strand = parse_coord(header)
            longest[gid] = {
                "gene_id": gid,
                "transcript_id": tid,
                "cds_header": header,
                "pep_header": pep_header,
                "cds": cds.upper(),
                "pep": pep.upper().replace("*", ""),
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
            }
    return longest


def read_label_table() -> pd.DataFrame:
    label_table = FEATURE_ROOT / "id_check" / "all_label_table.tsv"
    if label_table.exists():
        df = pd.read_csv(label_table, sep="\t")
        df["gene_id"] = df["seq_id"].map(normalize_gene)
        return df[["seq_id", "gene_id", "label", "source_fasta"]]

    rows: list[dict[str, object]] = []
    for fasta_name, label in [("essential_longest_protein.fasta", 1), ("nonessential_longest_protein.fasta", 0)]:
        for header, _seq in parse_fasta(SEQ_DIR / fasta_name):
            sid = header[1:].split()[0]
            rows.append({"seq_id": sid, "gene_id": normalize_gene(sid), "label": label, "source_fasta": fasta_name})
    return pd.DataFrame(rows)


def kmer_set(seq: str, k: int = 4) -> set[str]:
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq.upper())
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def build_similarity_features(records: dict[str, dict[str, object]], target_genes: set[str]) -> dict[str, dict[str, float]]:
    ksets = {gene: kmer_set(str(rec["pep"])) for gene, rec in records.items() if rec.get("pep")}
    index: dict[str, set[str]] = defaultdict(set)
    for gene, kmers in ksets.items():
        for kmer in kmers:
            index[kmer].add(gene)

    out: dict[str, dict[str, float]] = {}
    for gene in sorted(target_genes):
        kmers = ksets.get(gene, set())
        candidate_counts: Counter[str] = Counter()
        for kmer in kmers:
            candidate_counts.update(index.get(kmer, ()))
        candidate_counts.pop(gene, None)

        best_gene = ""
        best_jaccard = 0.0
        family_size_jaccard_030 = 1
        family_size_jaccard_020 = 1
        top_candidates = sorted(candidate_counts.items(), key=lambda item: item[1], reverse=True)[:500]
        for cand, shared in top_candidates:
            denom = len(kmers) + len(ksets[cand]) - shared
            jaccard = shared / denom if denom else 0.0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_gene = cand
            if jaccard >= 0.30:
                family_size_jaccard_030 += 1
            if jaccard >= 0.20:
                family_size_jaccard_020 += 1

        out[gene] = {
            "paralog_kmer_identity_proxy": best_jaccard,
            "best_paralog_gene_proxy": best_gene,
            "gene_family_size_proxy_jaccard_030": float(family_size_jaccard_030),
            "gene_family_size_proxy_jaccard_020": float(family_size_jaccard_020),
            "has_paralog_proxy_jaccard_030": 1.0 if family_size_jaccard_030 > 1 else 0.0,
        }
    return out


def sequence_features(rec: dict[str, object]) -> dict[str, float]:
    cds = str(rec.get("cds", "")).upper()
    pep = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(rec.get("pep", "")).upper())
    features: dict[str, float] = {}

    cds_len = len(cds)
    pep_len = len(pep)
    features["protein_length"] = float(pep_len)
    features["cds_length"] = float(cds_len)
    features["gene_span_bp"] = float(abs(int(rec.get("end", 0)) - int(rec.get("start", 0))) + 1 if rec.get("start") else 0)
    features["gc_content"] = float((cds.count("G") + cds.count("C")) / cds_len) if cds_len else 0.0
    features["at_content"] = float((cds.count("A") + cds.count("T")) / cds_len) if cds_len else 0.0
    g_plus_c = cds.count("G") + cds.count("C")
    a_plus_t = cds.count("A") + cds.count("T")
    features["gc_skew"] = float((cds.count("G") - cds.count("C")) / g_plus_c) if g_plus_c else 0.0
    features["at_skew"] = float((cds.count("A") - cds.count("T")) / a_plus_t) if a_plus_t else 0.0
    third = cds[2::3]
    features["gc3_content"] = float((third.count("G") + third.count("C")) / len(third)) if third else 0.0

    for nt in NT_ORDER:
        features[f"nt_freq_{nt}"] = float(cds.count(nt) / cds_len) if cds_len else 0.0

    aa_total = len(pep)
    for aa in AA_ORDER:
        features[f"aa_freq_{aa}"] = float(pep.count(aa) / aa_total) if aa_total else 0.0
    for group, members in AA_GROUPS.items():
        features[f"aa_group_{group}"] = float(sum(pep.count(aa) for aa in members) / aa_total) if aa_total else 0.0

    if pep:
        try:
            analysis = ProteinAnalysis(pep)
            features["protein_molecular_weight"] = float(analysis.molecular_weight())
            features["protein_aromaticity"] = float(analysis.aromaticity())
            features["protein_instability_index"] = float(analysis.instability_index())
            features["protein_gravy"] = float(analysis.gravy())
            features["protein_isoelectric_point"] = float(analysis.isoelectric_point())
        except Exception:
            features.update(
                {
                    "protein_molecular_weight": 0.0,
                    "protein_aromaticity": 0.0,
                    "protein_instability_index": 0.0,
                    "protein_gravy": 0.0,
                    "protein_isoelectric_point": 0.0,
                }
            )
    else:
        features.update(
            {
                "protein_molecular_weight": 0.0,
                "protein_aromaticity": 0.0,
                "protein_instability_index": 0.0,
                "protein_gravy": 0.0,
                "protein_isoelectric_point": 0.0,
            }
        )

    return features


def tandem_proxy(records: dict[str, dict[str, object]], sim: dict[str, dict[str, float]], target_gene: str) -> float:
    rec = records.get(target_gene)
    best = sim.get(target_gene, {}).get("best_paralog_gene_proxy", "")
    best_rec = records.get(str(best))
    if not rec or not best_rec:
        return 0.0
    if rec.get("chrom") != best_rec.get("chrom") or not rec.get("start") or not best_rec.get("start"):
        return 0.0
    distance = min(
        abs(int(rec["start"]) - int(best_rec["end"])),
        abs(int(rec["end"]) - int(best_rec["start"])),
    )
    return 1.0 if distance <= 100_000 else 0.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    records = all_longest_records()
    labels = read_label_table()
    target_genes = set(labels["gene_id"])

    sim_features = build_similarity_features(records, target_genes)

    rows: list[dict[str, object]] = []
    for row in labels.itertuples(index=False):
        rec = records.get(row.gene_id)
        if not rec:
            continue
        feats = sequence_features(rec)
        feats.update(sim_features.get(row.gene_id, {}))
        feats["tandem_duplicate_proxy"] = tandem_proxy(records, sim_features, row.gene_id)
        rows.append(
            {
                "seq_id": row.seq_id,
                "gene_id": row.gene_id,
                "label": int(row.label),
                "transcript_id": rec["transcript_id"],
                "chrom": rec["chrom"],
                "start": rec["start"],
                "end": rec["end"],
                **feats,
            }
        )

    df = pd.DataFrame(rows)
    id_cols = ["seq_id", "gene_id", "label", "transcript_id", "chrom", "start", "end", "best_paralog_gene_proxy"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    df[id_cols + feature_cols].to_csv(OUTDIR / "literature_sequence_features_all.tsv", sep="\t", index=False)

    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["label"].astype(np.int64).to_numpy()
    ids = df["seq_id"].astype(str).to_numpy()
    np.save(OUTDIR / "literature_sequence_features_all.npy", X)
    np.save(OUTDIR / "all_labels.npy", y)
    np.save(OUTDIR / "all_ids.npy", ids)

    pd.DataFrame({"feature_name": feature_cols}).to_csv(OUTDIR / "feature_names.tsv", sep="\t", index=False)
    pd.DataFrame(LITERATURE_FEATURE_MAP, columns=["paper_feature", "status", "note"]).to_csv(
        OUTDIR / "paper_feature_reproducibility_status.tsv", sep="\t", index=False
    )

    summary = [
        f"rows\t{len(df)}",
        f"features\t{len(feature_cols)}",
        f"essential_label_1\t{int((df['label'] == 1).sum())}",
        f"nonessential_label_0\t{int((df['label'] == 0).sum())}",
        f"araport11_background_longest_genes\t{len(records)}",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
