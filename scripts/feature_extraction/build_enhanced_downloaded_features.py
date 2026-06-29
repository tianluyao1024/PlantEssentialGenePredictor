from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("D:/拟南芥/增强特征数据")
DOWNLOADS = BASE / "downloads"
OUT_DIR = BASE / "processed"
PAPER_DIR = Path("D:/拟南芥/文献特征复现/paper_style_features_araport11_latest_modern_replacements")

GENE_RE = re.compile(r"(AT[1-5CM]G\d{5})", re.I)


def norm_gene(x: object) -> str | None:
    m = GENE_RE.search(str(x))
    return m.group(1).upper() if m else None


def read_target_genes() -> pd.DataFrame:
    df = pd.read_csv(PAPER_DIR / "paper_style_features_all_modern_replacements.tsv", sep="\t")
    return df[["seq_id", "gene_id", "label"]].assign(gene_id=lambda d: d["gene_id"].str.upper())


def clean_columns(cols) -> list[str]:
    out = []
    seen = {}
    for c in cols:
        s = str(c).strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_()+%.-]+", "_", s)
        s = s.strip("_").lower()
        if not s or s.startswith("unnamed"):
            s = "col"
        n = seen.get(s, 0)
        seen[s] = n + 1
        if n:
            s = f"{s}_{n+1}"
        out.append(s)
    return out


def read_supp_table(sheet: str, header: int = 1) -> pd.DataFrame:
    path = DOWNLOADS / "Shahzad_2025_NaturePlants_supplementary_tables.xlsx"
    df = pd.read_excel(path, sheet_name=sheet, header=header)
    df.columns = clean_columns(df.columns)
    return df


def methylation_features(target: pd.DataFrame) -> pd.DataFrame:
    rows = target[["gene_id"]].copy()

    t1 = read_supp_table("TableS1", header=1)
    gene_col = next(c for c in t1.columns if "gene" in c and "id" in c)
    t1["gene_id"] = t1[gene_col].map(norm_gene)
    keep = ["gene_id"]
    for c in t1.columns:
        if c == "gene_id":
            continue
        vals = pd.to_numeric(t1[c], errors="coerce")
        if vals.notna().sum() > 0:
            t1[f"meth2025_{c}"] = vals
            keep.append(f"meth2025_{c}")
    rows = rows.merge(t1[keep].dropna(subset=["gene_id"]).drop_duplicates("gene_id"), on="gene_id", how="left")

    # Association tables: count significant records and summarize p/effect/MAF where available.
    assoc_specs = {
        "TableS3": "meth_expr_assoc",
        "TableS4": "meth_expr_retained",
        "TableS8": "meth_fitness_assoc",
        "TableS12": "meth_flowering_assoc",
        "TableS13": "meth_ionome_assoc",
        "TableS19": "meth_environment_assoc",
    }
    for sheet, prefix in assoc_specs.items():
        try:
            df = read_supp_table(sheet, header=1)
        except Exception:
            continue
        gene_cols = [c for c in df.columns if "gene" in c and ("id" in c or "associated" in c or "candidate" in c)]
        if not gene_cols:
            continue
        gene_col = gene_cols[0]
        df["gene_id"] = df[gene_col].map(norm_gene)
        df = df.dropna(subset=["gene_id"])
        if df.empty:
            continue
        agg = df.groupby("gene_id").size().rename(f"{prefix}_record_count").to_frame()
        for pat, outname, reducer in [
            ("log10", "max_neglog10p", "max"),
            ("p_value", "min_pvalue", "min"),
            ("p_value", "max_pvalue", "max"),
            ("effect", "max_effect", "max"),
            ("maf", "max_maf", "max"),
        ]:
            cols = [c for c in df.columns if pat in c]
            if cols:
                vals = df[cols].apply(pd.to_numeric, errors="coerce")
                per_row = vals.max(axis=1) if reducer == "max" else vals.min(axis=1)
                if reducer == "max":
                    agg[f"{prefix}_{outname}"] = per_row.groupby(df["gene_id"]).max()
                else:
                    agg[f"{prefix}_{outname}"] = per_row.groupby(df["gene_id"]).min()
        rows = rows.merge(agg.reset_index(), on="gene_id", how="left")

    meth_cols = [c for c in rows.columns if c != "gene_id"]
    count_cols = [c for c in meth_cols if c.endswith("_record_count")]
    rows[count_cols] = rows[count_cols].fillna(0)
    return rows


def snpeff_gene_features(target: pd.DataFrame) -> pd.DataFrame:
    path = DOWNLOADS / "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.genes.txt"
    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = None
        for line in f:
            if line.startswith("#GeneId"):
                header = line[1:].rstrip("\n").split("\t")
                break
    if header is None:
        raise RuntimeError("Could not find snpEff genes header")
    df.columns = header
    df["gene_id"] = df["GeneId"].map(norm_gene)
    df = df.dropna(subset=["gene_id"])

    out = pd.DataFrame({"gene_id": target["gene_id"]})
    work = pd.DataFrame({"gene_id": df["gene_id"]})
    for col in df.columns:
        if col in {"GeneId", "GeneName", "BioType", "gene_id"}:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        cname = "snpeff_" + clean_columns([col])[0]
        work[cname] = vals
    work = work.groupby("gene_id").max(numeric_only=True).reset_index()

    count_cols = [c for c in work.columns if c.startswith("snpeff_count")]
    high_terms = ["frameshift", "stop_gained", "stop_lost", "start_lost", "splice_acceptor", "splice_donor"]
    moderate_terms = ["missense", "inframe", "initiator"]
    work["snpeff_high_impact_variant_count"] = work[
        [c for c in count_cols if any(t in c for t in high_terms)]
    ].sum(axis=1)
    work["snpeff_moderate_variant_count"] = work[
        [c for c in count_cols if any(t in c for t in moderate_terms)]
    ].sum(axis=1)
    if "snpeff_count_synonymous_variant" in work:
        work["snpeff_missense_synonymous_ratio"] = work.get("snpeff_count_missense_variant", 0) / (
            work["snpeff_count_synonymous_variant"] + 1.0
        )
    if "snpeff_length_gene" in work and "snpeff_bases_affected_gene" in work:
        work["snpeff_gene_variant_density"] = work["snpeff_bases_affected_gene"] / (work["snpeff_length_gene"] + 1.0)
    out = out.merge(work, on="gene_id", how="left")
    return out


def load_arath_bed() -> pd.DataFrame:
    bed = pd.read_csv(DOWNLOADS / "PGDD2_Arath.bed", sep="\t", header=None)
    bed = bed.iloc[:, :5]
    bed.columns = ["chr", "raw_id", "start", "end", "strand"]
    bed["gene_id"] = bed["raw_id"].map(norm_gene)
    bed["chr_num"] = bed["chr"].astype(str).str.extract(r"(\d+)").astype(float)
    bed = bed.dropna(subset=["gene_id", "chr_num"])
    bed["chr_num"] = bed["chr_num"].astype(int)
    return bed


def map_window_scores(target: pd.DataFrame, bed: pd.DataFrame, filename: str, prefix: str) -> pd.DataFrame:
    win = pd.read_csv(DOWNLOADS / filename, sep="\t")
    win["chr"] = pd.to_numeric(win["chr"], errors="coerce").astype("Int64")
    win["score"] = pd.to_numeric(win["score"], errors="coerce")
    features = []
    for _, g in bed.iterrows():
        sub = win[
            (win["chr"] == int(g["chr_num"]))
            & (win["win_end"] >= int(g["start"]))
            & (win["win_start"] <= int(g["end"]))
        ]
        if sub.empty:
            features.append({"gene_id": g["gene_id"], f"{prefix}_max": np.nan, f"{prefix}_mean": np.nan})
        else:
            features.append(
                {
                    "gene_id": g["gene_id"],
                    f"{prefix}_max": float(sub["score"].max()),
                    f"{prefix}_mean": float(sub["score"].mean()),
                }
            )
    feat = pd.DataFrame(features).groupby("gene_id").max(numeric_only=True).reset_index()
    return target[["gene_id"]].merge(feat, on="gene_id", how="left")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = read_target_genes()
    parts = [target[["gene_id"]]]

    meth = methylation_features(target)
    parts.append(meth.drop(columns=["gene_id"]))

    snp = snpeff_gene_features(target)
    parts.append(snp.drop(columns=["gene_id"]))

    bed = load_arath_bed()
    fst = map_window_scores(target, bed, "1001genomes_adm_global_fst_win10000.txt", "fst1001_window")
    omega = map_window_scores(target, bed, "1001genomes_global_omega_win10000.txt", "omega1001_window")
    parts.append(fst.drop(columns=["gene_id"]))
    parts.append(omega.drop(columns=["gene_id"]))

    feat = pd.concat(parts, axis=1)
    feat.insert(0, "seq_id", target["seq_id"].to_numpy())
    feat.insert(2, "label", target["label"].to_numpy())

    numeric_cols = [c for c in feat.columns if c not in {"seq_id", "gene_id", "label"}]
    for c in numeric_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    # Drop all-NaN and constant columns after merge.
    keep = []
    for c in numeric_cols:
        if feat[c].notna().sum() == 0:
            continue
        if feat[c].nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    feat = feat[["seq_id", "gene_id", "label"] + keep]
    X = feat[keep].to_numpy(np.float32)

    feat.to_csv(OUT_DIR / "enhanced_downloaded_gene_features.tsv", sep="\t", index=False)
    np.save(OUT_DIR / "enhanced_downloaded_gene_features.npy", X)
    pd.DataFrame({"feature_name": keep}).to_csv(OUT_DIR / "enhanced_downloaded_feature_names.tsv", sep="\t", index=False)

    summary = {
        "rows": int(feat.shape[0]),
        "features": int(len(keep)),
        "methylation_features": int(sum(c.startswith("meth") for c in keep)),
        "snpeff_features": int(sum(c.startswith("snpeff") for c in keep)),
        "fst_omega_features": int(sum(c.startswith("fst1001") or c.startswith("omega1001") for c in keep)),
        "methylation_tableS1_covered_genes": int(meth.drop(columns=["gene_id"]).notna().any(axis=1).sum()),
        "snpeff_covered_genes": int(snp.drop(columns=["gene_id"]).notna().any(axis=1).sum()),
        "fst_covered_genes": int(fst.drop(columns=["gene_id"]).notna().any(axis=1).sum()),
        "omega_covered_genes": int(omega.drop(columns=["gene_id"]).notna().any(axis=1).sum()),
    }
    (OUT_DIR / "enhanced_downloaded_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
