from __future__ import annotations

import csv
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from extract_literature_sequence_features_araport11 import (
    all_longest_records,
    read_label_table,
    sequence_features,
)


ROOT = Path(r"D:\拟南芥\文献特征复现")
DL = ROOT / "downloads"
BG = ROOT / "araport11_background"
OUT = ROOT / "paper_style_features_araport11_latest"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_ROOT = Path(r"D:\拟南芥\特征\Araport11_综合必需非必需_features")
ATH_GO_GOSLIM = Path(r"C:\Users\tly\Desktop\植物\拟南芥\split_nonessential\ATH_GO_GOSLIM.txt\ATH_GO_GOSLIM.txt")
STRING_LINKS = Path(r"C:\Users\tly\Desktop\植物\拟南芥\split_nonessential\3702.protein.links.v12.0.txt\3702.protein.links.v12.0.txt")

GENE_RE = re.compile(r"AT[1-5CM]G\d{5}", re.I)
EXPERIMENTAL_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
    "ISS",
    "ISO",
    "ISA",
    "ISM",
    "IGC",
    "IBA",
    "IBD",
    "IKR",
    "IRD",
    "RCA",
}


def norm_gene(x: object) -> str:
    if x is None:
        return ""
    m = GENE_RE.search(str(x))
    return m.group(0).upper() if m else ""


def load_diamond_features(records: dict[str, dict[str, object]]) -> pd.DataFrame:
    path = BG / "araport11_all_vs_all.tsv"
    rows = []
    all_genes = set(records)
    family_hits_40 = defaultdict(set)
    family_hits_1e10 = defaultdict(set)
    top_hit: dict[str, tuple[str, float, float, float]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            q, s, pident, alen, evalue, bitscore, qlen, slen = line.rstrip("\n").split("\t")
            qg = q.split("|", 1)[0].upper()
            sg = s.split("|", 1)[0].upper()
            if qg == sg:
                continue
            pident_f = float(pident)
            bits_f = float(bitscore)
            e_f = float(evalue)
            coverage = float(alen) / max(1.0, min(float(qlen), float(slen)))
            family_hits_1e10[qg].add(sg)
            if pident_f >= 40.0 and coverage >= 0.4:
                family_hits_40[qg].add(sg)
            old = top_hit.get(qg)
            if old is None or bits_f > old[3]:
                top_hit[qg] = (sg, pident_f, e_f, bits_f)

    # tandem duplicate: BLASTP E < 1e-10 and <=10 genes apart on same chromosome.
    chrom_order = defaultdict(list)
    for gene, rec in records.items():
        chrom = str(rec.get("chrom") or "")
        start = int(rec.get("start") or 0)
        if chrom and start:
            chrom_order[chrom].append((start, gene))
    rank = {}
    for chrom, items in chrom_order.items():
        for i, (_start, gene) in enumerate(sorted(items)):
            rank[gene] = (chrom, i)

    for gene in sorted(all_genes):
        best_gene, best_pid, best_e, best_bits = top_hit.get(gene, ("", np.nan, np.nan, np.nan))
        tandem = 0
        chrom_i = rank.get(gene)
        if chrom_i:
            for hit in family_hits_1e10.get(gene, set()):
                chrom_j = rank.get(hit)
                if chrom_j and chrom_i[0] == chrom_j[0] and abs(chrom_i[1] - chrom_j[1]) <= 10:
                    tandem = 1
                    break
        rows.append(
            {
                "gene_id": gene,
                "gene_family_size": len(family_hits_40.get(gene, set())) + 1,
                "singleton_status": 1 if len(family_hits_40.get(gene, set())) == 0 else 0,
                "paralog_percentage_identity": best_pid,
                "top_paralog_gene": best_gene,
                "top_paralog_bitscore": best_bits,
                "tandem_duplicate": tandem,
                "paralog_ks": np.nan,
                "paralog_kaks": np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_interpro_domains() -> pd.DataFrame:
    path = DL / "ensembl_plants_interpro.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str)
    gene_col = "Gene stable ID"
    ipr_col = "Interpro ID"
    pfam_col = "Pfam ID"
    counts = (
        df.assign(gene_id=df[gene_col].map(norm_gene))
        .query("gene_id != ''")
        .groupby("gene_id")
        .agg(domain_number=(ipr_col, lambda s: len(set(x for x in s.dropna() if str(x).strip()))),
             pfam_domain_number=(pfam_col, lambda s: len(set(x for x in s.dropna() if str(x).strip()))))
        .reset_index()
    )
    return counts


def load_go_features(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_to_terms = defaultdict(set)
    with ATH_GO_GOSLIM.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            gene = norm_gene(parts[0])
            slim_term = parts[8].strip()
            ev = parts[9].strip()
            if gene and slim_term and ev in EXPERIMENTAL_CODES:
                gene_to_terms[gene].add(slim_term)

    label_by_gene = dict(zip(labels["gene_id"], labels["label"]))
    genes = sorted(label_by_gene)
    essential = {g for g, y in label_by_gene.items() if int(y) == 1}
    nonessential = set(genes) - essential

    tests = []
    all_terms = sorted({t for terms in gene_to_terms.values() for t in terms})
    for term in all_terms:
        genes_with = {g for g in genes if term in gene_to_terms.get(g, set())}
        if len(genes_with) < max(1, int(0.01 * len(genes))):
            continue
        a = len(genes_with & essential)
        b = len(essential - genes_with)
        c = len(genes_with & nonessential)
        d = len(nonessential - genes_with)
        _odds, p = fisher_exact([[a, b], [c, d]])
        tests.append((term, p, a, c, len(genes_with)))

    if tests:
        rejected, qvals, _a, _b = multipletests([x[1] for x in tests], method="fdr_bh")
    else:
        rejected, qvals = [], []
    selected = []
    rows = []
    for (term, p, a, c, n), keep, q in zip(tests, rejected, qvals):
        rows.append({"go_slim_term": term, "p_value": p, "q_value": q, "essential_with_term": a, "nonessential_with_term": c, "genes_with_term": n, "selected": bool(keep)})
        if keep:
            selected.append(term)

    # If current labels are too different and no term passes, keep the paper-like
    # significant terms by using the top 27 slim terms by adjusted p-value.
    if len(selected) == 0 and rows:
        selected = [r["go_slim_term"] for r in sorted(rows, key=lambda r: r["q_value"])[:27]]
        for r in rows:
            r["selected"] = r["go_slim_term"] in selected
    elif len(selected) > 27:
        selected = [r["go_slim_term"] for r in sorted(rows, key=lambda r: r["q_value"])[:27]]
        for r in rows:
            r["selected"] = r["go_slim_term"] in selected

    feat_rows = []
    for gene in genes:
        row = {"gene_id": gene}
        terms = gene_to_terms.get(gene, set())
        for term in selected:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", term).strip("_").lower()
            row[f"go_{safe}"] = 1 if term in terms else 0
        feat_rows.append(row)

    return pd.DataFrame(feat_rows), pd.DataFrame(rows).sort_values(["selected", "q_value"], ascending=[False, True])


def load_homolog_features() -> pd.DataFrame:
    path = DL / "ensembl_plants_selected_homologs.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["gene_id"] = df["Gene stable ID"].map(norm_gene)
    species_cols = {
        "alyrata": ("Arabidopsis lyrata gene stable ID", "%id. query gene identical to target Arabidopsis lyrata gene"),
        "ptrichocarpa": ("Populus trichocarpa gene stable ID", "%id. query gene identical to target Populus trichocarpa gene"),
        "vvinifera": ("Vitis vinifera gene stable ID", "%id. query gene identical to target Vitis vinifera gene"),
        "rice": ("Oryza sativa Japonica Group gene stable ID", "%id. query gene identical to target Oryza sativa Japonica Group gene"),
        "ppatens": ("Physcomitrium patens gene stable ID", "%id. query gene identical to target Physcomitrium patens gene"),
    }
    rows = []
    for gene, sub in df.groupby("gene_id"):
        if not gene:
            continue
        row = {"gene_id": gene}
        plant_ids = []
        plant_pids = []
        for key, (id_col, pid_col) in species_cols.items():
            ids = [x for x in sub[id_col].dropna().astype(str) if x.strip()]
            pids = pd.to_numeric(sub[pid_col], errors="coerce").dropna().astype(float)
            row[f"{key}_homolog_found"] = 1 if ids else 0
            row[f"{key}_homolog_percent_identity"] = float(pids.max()) if len(pids) else np.nan
            if ids:
                plant_ids.extend(ids)
            if len(pids):
                plant_pids.append(float(pids.max()))
            row[f"{key}_homolog_kaks"] = np.nan
        row["homolog_not_found_in_rice"] = 0 if row.get("rice_homolog_found", 0) else 1
        row["percentage_identity_in_plants"] = float(np.nanmax(plant_pids)) if plant_pids else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_string_degree() -> pd.DataFrame:
    map_path = DL / "uniprot_arabidopsis_mapping.tsv"
    m = pd.read_csv(map_path, sep="\t", dtype=str)
    acc_to_gene = {}
    for r in m.itertuples(index=False):
        acc = str(r.Entry)
        gene = norm_gene(str(getattr(r, "TAIR")))
        if not gene:
            gene = norm_gene(str(getattr(r, "Araport")))
        if acc and gene:
            acc_to_gene[acc] = gene

    degree_700 = defaultdict(int)
    degree_400 = defaultdict(int)
    with STRING_LINKS.open("r", encoding="utf-8") as handle:
        header = next(handle)
        for line in handle:
            p1, p2, score = line.strip().split()
            a1 = p1.split(".", 1)[1]
            a2 = p2.split(".", 1)[1]
            g1, g2 = acc_to_gene.get(a1), acc_to_gene.get(a2)
            if not g1 or not g2 or g1 == g2:
                continue
            s = int(score)
            if s >= 400:
                degree_400[g1] += 1
                degree_400[g2] += 1
            if s >= 700:
                degree_700[g1] += 1
                degree_700[g2] += 1
    genes = set(degree_400) | set(degree_700)
    return pd.DataFrame(
        [{"gene_id": g, "string_network_connections_400": degree_400[g], "string_network_connections_700": degree_700[g]} for g in sorted(genes)]
    )


def main() -> None:
    labels = read_label_table().drop_duplicates("gene_id")
    records = all_longest_records()
    base = labels.copy()

    seq_rows = []
    for gene in base["gene_id"]:
        rec = records.get(gene)
        if rec:
            sf = sequence_features(rec)
            seq_rows.append({"gene_id": gene, **sf})
        else:
            seq_rows.append({"gene_id": gene})
    seq_df = pd.DataFrame(seq_rows)

    go_df, go_tests = load_go_features(base)
    dfs = [
        base,
        seq_df,
        load_diamond_features(records),
        load_interpro_domains(),
        load_homolog_features(),
        load_string_degree(),
        go_df,
    ]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="gene_id", how="left")

    # Paper columns that need historical/specialized data not obtained in this run.
    unavailable = {
        "alpha_wgd_duplicate_retained": "Bowers 2003 alpha WGD block annotations not available from current public API.",
        "beta_gamma_wgd_duplicate_retained": "Bowers 2003 beta/gamma WGD block annotations not available from current public API.",
        "pseudogene_present": "Zou 2009 pseudogene pipeline outputs not found/downloadable in this run.",
        "median_expression": "AtGenExpress CEL downloaded, but old ATH1 CDF package incompatible with R 4.6; pending RMA export.",
        "expression_variation": "Requires AtGenExpress gene-level expression matrix.",
        "expression_breadth": "Requires AtGenExpress gene-level expression matrix.",
        "expression_correlation": "Requires AtGenExpress expression plus paralog clusters.",
        "expression_correlation_ks_lt_2": "Requires AtGenExpress expression plus Ks-filtered paralogs.",
        "expression_module_size": "Requires K-means modules from AtGenExpress expression.",
        "aranet_gene_network_connections": "AraNet original download endpoints returned 404; STRING modern equivalent provided.",
        "protein_protein_interactions_aic": "AIC 2011 supplemental PPI not retrieved; STRING modern equivalent provided.",
        "gene_body_methylated": "Takuno and Gaut 2012 gene body methylation table not retrieved.",
        "nucleotide_diversity": "Moghe 2013 80-accession nucleotide diversity table not retrieved.",
        "percentage_identity_in_metazoans": "Requires metazoan proteome BLAST panel.",
        "percentage_identity_in_fungi": "Requires fungal proteome BLAST panel.",
        "core_eukaryotic_gene": "Requires COG/KOG clusters across seven eukaryotes.",
    }
    for col in unavailable:
        merged[col] = np.nan

    # Rename exact/modern columns toward paper terms where possible.
    if "protein_length" not in merged and "protein_length_x" in merged:
        merged["protein_length"] = merged["protein_length_x"]
    merged["domain_number"] = merged["domain_number"].fillna(0)
    merged["gene_family_size"] = merged["gene_family_size"].fillna(1)
    merged["tandem_duplicate"] = merged["tandem_duplicate"].fillna(0)

    id_cols = ["seq_id", "gene_id", "label", "source_fasta"]
    feature_cols = [c for c in merged.columns if c not in id_cols and c not in {"top_paralog_gene"}]
    # keep best paralog gene as metadata, not numeric model column
    ordered = id_cols + ["top_paralog_gene"] + feature_cols
    merged = merged[[c for c in ordered if c in merged.columns]]
    merged.to_csv(OUT / "paper_style_features_all.tsv", sep="\t", index=False)

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]
    X = merged[numeric_cols].astype(np.float32).to_numpy()
    np.save(OUT / "paper_style_features_all.npy", X)
    np.save(OUT / "all_labels.npy", merged["label"].astype(np.int64).to_numpy())
    np.save(OUT / "all_ids.npy", merged["seq_id"].astype(str).to_numpy())
    pd.DataFrame({"feature_name": numeric_cols}).to_csv(OUT / "feature_names.tsv", sep="\t", index=False)
    go_tests.to_csv(OUT / "go_slim_selection_tests.tsv", sep="\t", index=False)

    status_rows = []
    exact_cols = {
        "protein_length",
        "domain_number",
        "pfam_domain_number",
        "gene_family_size",
        "singleton_status",
        "paralog_percentage_identity",
        "tandem_duplicate",
        "homolog_not_found_in_rice",
        "percentage_identity_in_plants",
    }
    modern_cols = {"string_network_connections_400", "string_network_connections_700"}
    for col in numeric_cols:
        if col in unavailable:
            status = "unavailable"
            note = unavailable[col]
        elif col in exact_cols or col.startswith("go_") or col.endswith("_homolog_percent_identity"):
            status = "computed_from_downloaded_current_data"
            note = "Computed in this run from current Araport11/GO/BioMart/DIAMOND data."
        elif col in modern_cols:
            status = "modern_equivalent"
            note = "STRING v12 used as modern network/PPI equivalent, not original AraNet/AIC."
        elif col.endswith("_homolog_kaks") or col in {"paralog_ks", "paralog_kaks"}:
            status = "placeholder_nan"
            note = "Column retained for paper compatibility; Ka/Ks computation requires codon alignments."
        else:
            status = "derived_sequence_feature"
            note = "Additional sequence descriptor computed from longest CDS/protein."
        status_rows.append({"feature_name": col, "status": status, "note": note})
    pd.DataFrame(status_rows).to_csv(OUT / "feature_status.tsv", sep="\t", index=False)

    summary = [
        f"rows\t{len(merged)}",
        f"numeric_feature_columns\t{len(numeric_cols)}",
        f"essential_label_1\t{int((merged['label'] == 1).sum())}",
        f"nonessential_label_0\t{int((merged['label'] == 0).sum())}",
        f"go_features_selected\t{sum(c.startswith('go_') for c in numeric_cols)}",
        f"nan_placeholder_columns\t{sum(pd.DataFrame(status_rows)['status'].isin(['unavailable','placeholder_nan']))}",
    ]
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
