from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


WORK = Path("C:/Users/tly/Desktop/\u690d\u7269/\u6c34\u7a3b\u65e5\u672c")
ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
RAW_CDS = WORK / "IRGSP-1.0_cds_2026-02-05" / "IRGSP-1.0_cds_2026-02-05.fasta"
RAW_PEP = WORK / "IRGSP-1.0_protein_2026-02-05" / "IRGSP-1.0_protein_2026-02-05.fasta"
OUT = ROOT / "cross_species_ath_rice_common_features_models" / "rice_rapdb_native_features_fresh_only"

NT_ORDER = "ACGT"
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
CODONS = [a + b + c for a in NT_ORDER for b in NT_ORDER for c in NT_ORDER]
STOP_CODONS = {"TAA", "TAG", "TGA"}
SENSE_CODONS = [c for c in CODONS if c not in STOP_CODONS]
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
AA_TO_CODONS: dict[str, list[str]] = defaultdict(list)
for codon, aa in GENETIC_CODE.items():
    if aa != "*":
        AA_TO_CODONS[aa].append(codon)
AA_MASS = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19, "G": 75.07, "H": 155.16,
    "I": 131.17, "K": 146.19, "L": 131.17, "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15,
    "R": 174.20, "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


def parse_fasta(path: Path):
    header = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks).upper()


def seq_id(header: str) -> str:
    return header.split()[0].split("|")[0]


def clean_cds(seq: str) -> str:
    return re.sub("[^ACGT]", "", str(seq).upper().replace("U", "T"))


def clean_pep(seq: str) -> str:
    return re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "", str(seq).upper().replace("*", ""))


def rap_gene_from_transcript(transcript_id: str) -> str:
    text = str(transcript_id).split()[0].split("|")[0].split("-")[0]
    if text.startswith("Os") and "t" in text[:5]:
        return text.replace("t", "g", 1)
    return ""


def sha1_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("ascii", errors="ignore")).hexdigest()


def shannon(counter: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    val = 0.0
    for count in counter.values():
        if count:
            p = count / total
            val -= p * math.log2(p)
    return val


def kmer_counts(seq: str, alphabet: str, k: int) -> dict[str, float]:
    names = [""]
    for _ in range(k):
        names = [p + c for p in names for c in alphabet]
    counts = dict.fromkeys(names, 0.0)
    total = max(0, len(seq) - k + 1)
    if total:
        allowed = set(alphabet)
        for i in range(total):
            word = seq[i : i + k]
            if set(word) <= allowed:
                counts[word] += 1.0
        for key in counts:
            counts[key] /= total
    return counts


def codon_features(cds: str) -> dict[str, float]:
    usable = cds[: len(cds) - len(cds) % 3]
    codons = [usable[i : i + 3] for i in range(0, len(usable), 3)]
    codons = [c for c in codons if len(c) == 3 and set(c) <= set(NT_ORDER)]
    sense = [c for c in codons if c not in STOP_CODONS]
    total = len(codons)
    sense_total = len(sense)
    counts = Counter(codons)
    sense_counts = Counter(sense)
    out = {
        "codon_count": float(total),
        "sense_codon_count": float(sense_total),
        "internal_stop_count": float(sum(1 for c in codons[:-1] if c in STOP_CODONS)),
        "terminal_stop": float(1 if codons and codons[-1] in STOP_CODONS else 0),
    }
    for c in CODONS:
        out[f"codon_freq_{c}"] = counts.get(c, 0) / max(1, total)
    for c in SENSE_CODONS:
        aa = GENETIC_CODE[c]
        aa_total = sum(sense_counts.get(x, 0) for x in AA_TO_CODONS[aa])
        out[f"rscu_{c}"] = (sense_counts.get(c, 0) * len(AA_TO_CODONS[aa]) / aa_total) if aa_total else 0.0
    return out


def sequence_features(cds: str, pep: str) -> dict[str, float]:
    cds = clean_cds(cds)
    pep = clean_pep(pep)
    out: dict[str, float] = {
        "cds_len": float(len(cds)),
        "protein_len": float(len(pep)),
        "cds_len_mod3": float(len(cds) % 3),
        "protein_to_cds_len_ratio": float(len(pep) / max(1, len(cds))),
    }
    nt_counts = Counter(cds)
    out["gc_content"] = (nt_counts["G"] + nt_counts["C"]) / max(1, len(cds))
    out["at_content"] = (nt_counts["A"] + nt_counts["T"]) / max(1, len(cds))
    for offset in range(3):
        pos = cds[offset::3]
        c = Counter(pos)
        out[f"gc{offset + 1}"] = (c["G"] + c["C"]) / max(1, len(pos))
    out["cds_shannon_nt"] = shannon(nt_counts, len(cds))
    out["protein_shannon_aa"] = shannon(Counter(pep), len(pep))
    for k in (1, 2, 3, 4):
        for word, val in kmer_counts(cds, NT_ORDER, k).items():
            out[f"cds_{k}mer_{word}"] = val
    for k in (1, 2):
        for word, val in kmer_counts(pep, AA_ORDER, k).items():
            out[f"aa_{k}mer_{word}"] = val
    aa_counts = Counter(pep)
    for aa in AA_ORDER:
        out[f"aa_comp_{aa}"] = aa_counts[aa] / max(1, len(pep))
    groups = {
        "hydrophobic": set("AILMFWV"),
        "polar": set("STNQCY"),
        "positive": set("KRH"),
        "negative": set("DE"),
        "aromatic": set("FWY"),
        "tiny": set("ACGST"),
    }
    for name, aas in groups.items():
        out[f"aa_group_{name}"] = sum(aa_counts[a] for a in aas) / max(1, len(pep))
    out["molecular_weight_mean_aa"] = sum(AA_MASS.get(a, 0.0) for a in pep) / max(1, len(pep))
    out["gravy_kd"] = sum(KD.get(a, 0.0) for a in pep) / max(1, len(pep))
    out["net_charge_approx_pH7"] = (
        aa_counts["K"] + aa_counts["R"] + 0.1 * aa_counts["H"] - aa_counts["D"] - aa_counts["E"]
    ) / max(1, len(pep))
    out.update(codon_features(cds))
    return out


def wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cds_records = {seq_id(h): clean_cds(s) for h, s in parse_fasta(RAW_CDS)}
    pep_records = {seq_id(h): clean_pep(s) for h, s in parse_fasta(RAW_PEP)}
    rows = []
    for tid, cds in cds_records.items():
        pep = pep_records.get(tid)
        gene = rap_gene_from_transcript(tid)
        if gene and pep:
            rows.append(
                {
                    "gene_id": gene,
                    "rap_gene_id": gene,
                    "transcript_id": tid,
                    "cds": cds,
                    "pep": pep,
                    "cds_len": len(cds),
                    "pep_len": len(pep),
                    "protein_sha1": sha1_seq(pep),
                }
            )
    records = pd.DataFrame(rows).sort_values(
        ["gene_id", "cds_len", "pep_len", "transcript_id"], ascending=[True, False, False, True]
    )
    records = records.drop_duplicates("gene_id", keep="first").reset_index(drop=True)

    with (OUT / "rice_rapdb_native_longest_cds.fasta").open("w", encoding="utf-8") as cds_out:
        with (OUT / "rice_rapdb_native_longest_protein.fasta").open("w", encoding="utf-8") as pep_out:
            for row in records.itertuples(index=False):
                header = f"{row.gene_id}|{row.transcript_id}|RAPDB_IRGSP1_2026-02-05"
                cds_out.write(f">{header}\n{wrap(row.cds)}\n")
                pep_out.write(f">{header}\n{wrap(row.pep)}\n")

    feature_rows = []
    for i, row in enumerate(records.itertuples(index=False), 1):
        feats = sequence_features(row.cds, row.pep)
        feats.update(
            {
                "gene_id": row.gene_id,
                "rap_gene_id": row.rap_gene_id,
                "transcript_id": row.transcript_id,
                "protein_sha1": row.protein_sha1,
            }
        )
        feature_rows.append(feats)
        if i % 5000 == 0:
            print(f"fresh RAPDB numeric features {i}/{len(records)}", flush=True)
    features = pd.DataFrame(feature_rows)
    front = ["gene_id", "rap_gene_id", "transcript_id", "protein_sha1"]
    feature_cols = [c for c in features.columns if c not in front]
    features[front + feature_cols].to_csv(OUT / "rice_rapdb_native_sequence_numeric_features.tsv", sep="\t", index=False)
    records.drop(columns=["cds", "pep"]).to_csv(OUT / "rice_rapdb_native_longest_transcript_map.tsv", sep="\t", index=False)
    manifest = {
        "raw_cds": str(RAW_CDS),
        "raw_protein": str(RAW_PEP),
        "id_rule": "fresh RAP-DB native only: OsXXt transcript to OsXXg gene; no MSU/RGAP mapping; no old feature table read",
        "longest_rule": "per RAP gene, sort by CDS length desc, protein length desc, transcript id asc",
        "rap_gene_count": int(len(records)),
        "numeric_feature_count": int(len(feature_cols)),
        "plm_rule": "PLM features must be generated separately from this fresh longest protein FASTA; no cached/rekeyed PLM accepted",
        "output_dir": str(OUT),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
