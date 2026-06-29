from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_cross_species_rice_ath_features_models import (
    clean_cds,
    clean_pep,
    parse_fasta,
    seq_id,
    sequence_features,
)


WORK = Path("C:/Users/tly/Desktop/\u690d\u7269/\u6c34\u7a3b\u65e5\u672c")
ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
RAW_CDS = WORK / "IRGSP-1.0_cds_2026-02-05" / "IRGSP-1.0_cds_2026-02-05.fasta"
RAW_PEP = WORK / "IRGSP-1.0_protein_2026-02-05" / "IRGSP-1.0_protein_2026-02-05.fasta"
COMMON = ROOT / "cross_species_ath_rice_common_features_models"
OLD_RICE_FASTA = COMMON / "rice" / "rice_longest_protein.fasta"
OLD_PLM_ROOT = COMMON / "plm_embeddings"
OUT = COMMON / "rice_rapdb_native_features"

PLM_MODELS = {
    "esm2": 2560,
    "protbert": 2048,
    "prott5": 2048,
}


def rap_gene_from_transcript(transcript_id: str) -> str:
    text = str(transcript_id).split()[0].split("-")[0]
    if text.startswith("Os") and "t" in text[:5]:
        return text.replace("t", "g", 1)
    return ""


def wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def sha1_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("ascii", errors="ignore")).hexdigest()


def load_rapdb_longest_records() -> pd.DataFrame:
    cds_records = {seq_id(h): clean_cds(s) for h, s in parse_fasta(RAW_CDS)}
    pep_records = {seq_id(h): clean_pep(s) for h, s in parse_fasta(RAW_PEP)}
    rows = []
    for tid, cds in cds_records.items():
        pep = pep_records.get(tid)
        gene = rap_gene_from_transcript(tid)
        if not gene or not pep:
            continue
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
    df = pd.DataFrame(rows)
    df = df.sort_values(["gene_id", "cds_len", "pep_len", "transcript_id"], ascending=[True, False, False, True])
    return df.drop_duplicates("gene_id", keep="first").reset_index(drop=True)


def write_native_fastas(records: pd.DataFrame) -> None:
    with (OUT / "rice_rapdb_native_longest_cds.fasta").open("w", encoding="utf-8") as cds_out:
        with (OUT / "rice_rapdb_native_longest_protein.fasta").open("w", encoding="utf-8") as pep_out:
            for row in records.itertuples(index=False):
                header = f"{row.gene_id}|{row.transcript_id}|RAPDB_IRGSP1_2026-02-05"
                cds_out.write(f">{header}\n{wrap(row.cds)}\n")
                pep_out.write(f">{header}\n{wrap(row.pep)}\n")


def build_numeric_features(records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
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
        rows.append(feats)
        if i % 5000 == 0:
            print(f"numeric sequence features {i}/{len(records)}", flush=True)
    df = pd.DataFrame(rows)
    front = ["gene_id", "rap_gene_id", "transcript_id", "protein_sha1"]
    feature_cols = [c for c in df.columns if c not in front]
    return df[front + feature_cols], feature_cols


def old_sequence_hash_to_gene() -> dict[str, str]:
    mapping = {}
    duplicate_hashes = 0
    for header, seq in parse_fasta(OLD_RICE_FASTA):
        old_gene = seq_id(header.split("|")[0])
        h = sha1_seq(clean_pep(seq))
        if h in mapping and mapping[h] != old_gene:
            duplicate_hashes += 1
            continue
        mapping[h] = old_gene
    print(f"old sequence hash map: {len(mapping)} unique sequences, duplicate_hashes={duplicate_hashes}", flush=True)
    return mapping


def rekey_plm_by_exact_sequence(records: pd.DataFrame) -> pd.DataFrame:
    hash_to_old_gene = old_sequence_hash_to_gene()
    coverage_rows = []
    for model_name, expected_dim in PLM_MODELS.items():
        old_dir = OLD_PLM_ROOT / model_name / "rice"
        out_dir = OUT / "plm_embeddings_rap_native" / model_name / "rice"
        out_dir.mkdir(parents=True, exist_ok=True)
        old_ids = np.load(old_dir / "all_ids.npy", allow_pickle=True).astype(str)
        old_emb = np.load(old_dir / "all_emb.npy", mmap_mode="r")
        old_lookup = {gene_id: idx for idx, gene_id in enumerate(old_ids)}
        arr = np.full((len(records), old_emb.shape[1]), np.nan, dtype=np.float32)
        hit = np.zeros(len(records), dtype=bool)
        old_gene_used = []
        for i, row in enumerate(records.itertuples(index=False)):
            old_gene = hash_to_old_gene.get(row.protein_sha1)
            old_gene_used.append(old_gene or "")
            idx = old_lookup.get(old_gene or "")
            if idx is not None:
                arr[i] = old_emb[idx]
                hit[i] = True
        np.save(out_dir / "all_ids.npy", records["gene_id"].astype(str).to_numpy())
        np.save(out_dir / "all_emb.npy", arr)
        coverage_rows.append(
            {
                "model": model_name,
                "expected_dim": expected_dim,
                "actual_dim": int(old_emb.shape[1]),
                "rap_gene_count": int(len(records)),
                "exact_sequence_embedding_hits": int(hit.sum()),
                "missing": int((~hit).sum()),
                "source": "rekeyed_by_exact_RAPDB_protein_sequence_sha1_not_by_MSU_id",
            }
        )
        print(f"{model_name}: exact sequence PLM hits {hit.sum()}/{len(records)}", flush=True)
    return pd.DataFrame(coverage_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_rapdb_longest_records()
    write_native_fastas(records)
    records.drop(columns=["cds", "pep"]).to_csv(OUT / "rice_rapdb_native_longest_transcript_map.tsv", sep="\t", index=False)
    numeric, numeric_cols = build_numeric_features(records)
    numeric.to_csv(OUT / "rice_rapdb_native_sequence_numeric_features.tsv", sep="\t", index=False)
    plm_cov = rekey_plm_by_exact_sequence(records)
    plm_cov.to_csv(OUT / "rice_rapdb_native_plm_exact_sequence_coverage.tsv", sep="\t", index=False)
    manifest = {
        "raw_cds": str(RAW_CDS),
        "raw_protein": str(RAW_PEP),
        "id_rule": "RAP native only: transcript OsXXt... converted to RAP gene OsXXg...; no MSU/RGAP mapping table used",
        "longest_rule": "per RAP gene, sort by CDS length desc, protein length desc, transcript id asc",
        "rap_gene_count": int(len(records)),
        "numeric_feature_count": int(len(numeric_cols)),
        "plm_reuse_rule": "Existing PLM vectors are reused only when the RAPDB-native longest protein sequence SHA1 exactly matches the cached sequence; no LOC/MSU/RGAP id mapping is used.",
        "plm_coverage": plm_cov.to_dict(orient="records"),
        "output_dir": str(OUT),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
