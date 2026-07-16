from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
NT_ORDER = list("ACGT")
AA_GROUPS = {
    "aa_group_hydrophobic": set("AILMFWYV"),
    "aa_group_polar": set("STNQCY"),
    "aa_group_positive": set("KRH"),
    "aa_group_negative": set("DE"),
    "aa_group_small": set("ACDGNPSTV"),
    "aa_group_aromatic": set("FWY"),
    "aa_group_sulfur": set("CM"),
}
AA_WEIGHTS = {
    "A": 89.09,
    "C": 121.15,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}
KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
INSTABILITY_DIWEIGHTS = {
    "A": 1.0,
    "C": 1.0,
    "D": 1.4,
    "E": 1.4,
    "F": 1.2,
    "G": 1.0,
    "H": 1.3,
    "I": 1.0,
    "K": 1.5,
    "L": 1.0,
    "M": 1.1,
    "N": 1.3,
    "P": 1.6,
    "Q": 1.2,
    "R": 1.5,
    "S": 1.2,
    "T": 1.1,
    "V": 1.0,
    "W": 1.2,
    "Y": 1.2,
}
GO_FEATURE_TO_TERM_NAME = {
    "go_cellular_component_organization": "cellular component organization",
    "go_rna_binding": "RNA binding",
    "go_cell_cycle": "cell cycle",
    "go_response_to_stress": "response to stress",
    "go_dna_binding_transcription_factor_activity": "DNA-binding transcription factor activity",
    "go_translation": "translation",
    "go_response_to_abiotic_stimulus": "response to abiotic stimulus",
    "go_nucleic_acid_binding": "nucleic acid binding",
    "go_pollination": "pollination",
    "go_dna_binding": "DNA binding",
    "go_response_to_biotic_stimulus": "response to biotic stimulus",
    "go_response_to_light_stimulus": "response to light stimulus",
    "go_ribosome": "ribosome",
    "go_nucleolus": "nucleolus",
    "go_structural_molecule_activity": "structural molecule activity",
    "go_signal_transduction": "signal transduction",
    "go_nucleobase_containing_compound_metabolic_process": "nucleobase-containing compound metabolic process",
    "go_chloroplast": "chloroplast",
    "go_extracellular_region": "extracellular region",
    "go_response_to_endogenous_stimulus": "response to endogenous stimulus",
    "go_embryo_development": "embryo development",
    "go_multicellular_organism_development": "multicellular organism development",
    "go_post_embryonic_development": "post-embryonic development",
    "go_anatomical_structure_development": "anatomical structure development",
    "go_response_to_external_stimulus": "response to external stimulus",
    "go_response_to_chemical": "response to chemical",
}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records.setdefault(current, [])
            elif current:
                records[current].append(line)
    return {key: "".join(chunks).upper().replace("*", "") for key, chunks in records.items()}


def first_gene_id(seq_id: str) -> str:
    match = re.search(r"(AT[1-5CM]G\d{5}|LOC_Os\d{2}g\d{5})", seq_id, re.I)
    if match:
        return match.group(1).upper() if match.group(1).upper().startswith("AT") else match.group(1)
    return seq_id.split(".")[0]


def collapse_to_longest_by_gene(records: dict[str, str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for seq_id, seq in records.items():
        gene_id = first_gene_id(seq_id)
        if gene_id not in out or len(seq) > len(out[gene_id][1]):
            out[gene_id] = (seq_id, seq)
    return out


def gc_fraction(seq: str, positions: list[int] | None = None) -> float:
    if positions is None:
        sub = [base for base in seq if base in "ACGT"]
    else:
        sub = [seq[idx] for idx in positions if idx < len(seq) and seq[idx] in "ACGT"]
    if not sub:
        return np.nan
    return sum(base in "GC" for base in sub) / len(sub)


def skew(pos: int, neg: int) -> float:
    total = pos + neg
    return (pos - neg) / total if total else np.nan


def pseudo_pi(seq: str) -> float:
    counts = {aa: seq.count(aa) for aa in AA_ORDER}
    acidic = counts["D"] + counts["E"]
    basic = counts["K"] + counts["R"] + counts["H"]
    return float(np.clip(7.0 + 0.08 * (basic - acidic), 3.0, 12.0))


def sequence_features(cds_by_gene: dict[str, tuple[str, str]], protein_by_gene: dict[str, tuple[str, str]]) -> pd.DataFrame:
    genes = sorted(set(cds_by_gene) | set(protein_by_gene))
    rows = []
    for gene in genes:
        cds = cds_by_gene.get(gene, ("", ""))[1].upper().replace("U", "T")
        prot = re.sub(r"[^A-Z]", "", protein_by_gene.get(gene, ("", ""))[1].upper())
        row: dict[str, float | str] = {"gene_id": gene}
        row["protein_length"] = float(len(prot)) if prot else np.nan
        row["cds_length"] = float(len(cds)) if cds else np.nan
        row["gc_content"] = gc_fraction(cds)
        row["at_content"] = 1 - row["gc_content"] if not pd.isna(row["gc_content"]) else np.nan
        counts_nt = {nt: cds.count(nt) for nt in NT_ORDER}
        row["gc_skew"] = skew(counts_nt["G"], counts_nt["C"])
        row["at_skew"] = skew(counts_nt["A"], counts_nt["T"])
        row["gc3_content"] = gc_fraction(cds, list(range(2, len(cds), 3)))
        valid_nt = sum(counts_nt.values())
        for nt in NT_ORDER:
            row[f"nt_freq_{nt}"] = counts_nt[nt] / valid_nt if valid_nt else np.nan
        counts_aa = {aa: prot.count(aa) for aa in AA_ORDER}
        valid_aa = sum(counts_aa.values())
        for aa in AA_ORDER:
            row[f"aa_freq_{aa}"] = counts_aa[aa] / valid_aa if valid_aa else np.nan
        for name, members in AA_GROUPS.items():
            row[name] = sum(counts_aa[aa] for aa in members) / valid_aa if valid_aa else np.nan
        row["protein_molecular_weight"] = (
            sum(counts_aa[aa] * AA_WEIGHTS[aa] for aa in AA_ORDER) - max(0, valid_aa - 1) * 18.015 if valid_aa else np.nan
        )
        row["protein_gravy"] = sum(counts_aa[aa] * KD[aa] for aa in AA_ORDER) / valid_aa if valid_aa else np.nan
        row["protein_isoelectric_point"] = pseudo_pi(prot) if valid_aa else np.nan
        row["protein_aromaticity"] = sum(counts_aa[aa] for aa in "FWY") / valid_aa if valid_aa else np.nan
        row["protein_instability_index"] = (
            10.0 * sum(counts_aa[aa] * INSTABILITY_DIWEIGHTS[aa] for aa in AA_ORDER) / valid_aa if valid_aa else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_attrs(text: str) -> dict[str, str]:
    out = {}
    for part in text.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def gff_features(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["gene_id", "gene_span_bp"])
    rows = []
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parse_attrs(parts[8])
            raw_id = attrs.get("ID", attrs.get("Name", ""))
            gene = first_gene_id(raw_id.replace("gene:", ""))
            if gene:
                rows.append({"gene_id": gene, "gene_span_bp": float(int(parts[4]) - int(parts[3]) + 1)})
    return pd.DataFrame(rows).drop_duplicates("gene_id") if rows else pd.DataFrame(columns=["gene_id", "gene_span_bp"])


def parse_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current: str | None = None
    in_term = False
    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                current = None
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                current = None
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id: GO:"):
                current = line.split("id: ", 1)[1]
            elif current and line.startswith("name: "):
                names[current] = line.split("name: ", 1)[1]
            elif current and line.startswith("is_a: GO:"):
                parents[current].add(line.split()[1])
            elif current and "relationship: part_of GO:" in line:
                parents[current].add(line.split("part_of ", 1)[1].split()[0])
    return names, parents


def ancestor_cache(parents: dict[str, set[str]]):
    cache: dict[str, set[str]] = {}

    def ancestors(term: str) -> set[str]:
        if term in cache:
            return cache[term]
        seen = {term}
        queue = deque(parents.get(term, set()))
        while queue:
            item = queue.popleft()
            if item in seen:
                continue
            seen.add(item)
            queue.extend(parents.get(item, set()))
        cache[term] = seen
        return seen

    return ancestors


def go_features(path: Path | None, obo: Path | None) -> pd.DataFrame:
    columns = ["gene_id"] + list(GO_FEATURE_TO_TERM_NAME)
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"gene_id", "go_id"}.issubset(df.columns):
        raise ValueError("GO table must contain gene_id and go_id columns")
    target_ids: dict[str, str] = {}
    ancestors = None
    if obo and obo.exists():
        names, parents = parse_obo(obo)
        norm_to_go = {value.lower(): key for key, value in names.items()}
        for feature, term_name in GO_FEATURE_TO_TERM_NAME.items():
            if term_name.lower() in norm_to_go:
                target_ids[feature] = norm_to_go[term_name.lower()]
        ancestors = ancestor_cache(parents)
    gene_to_go = defaultdict(set)
    for gene, go_id in zip(df["gene_id"].astype(str), df["go_id"].astype(str)):
        if go_id.startswith("GO:"):
            gene_to_go[first_gene_id(gene)].add(go_id)
    rows = []
    for gene, terms in gene_to_go.items():
        row = {"gene_id": gene}
        expanded = set(terms)
        if ancestors:
            for term in list(terms):
                expanded.update(ancestors(term))
        for feature in GO_FEATURE_TO_TERM_NAME:
            target = target_ids.get(feature)
            row[feature] = float(target in expanded) if target else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def ppi_features(path: Path | None) -> pd.DataFrame:
    columns = ["gene_id", "string_network_connections_400", "string_network_connections_700"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, sep="\t", dtype={"gene_a": str, "gene_b": str})
    if not {"gene_a", "gene_b", "score"}.issubset(df.columns):
        raise ValueError("PPI table must contain gene_a, gene_b and score columns")
    degrees = {400: defaultdict(set), 700: defaultdict(set)}
    for _, row in df.iterrows():
        try:
            score = float(row["score"])
        except Exception:
            continue
        a = first_gene_id(str(row["gene_a"]))
        b = first_gene_id(str(row["gene_b"]))
        if not a or not b or a == b:
            continue
        for cutoff in [400, 700]:
            if score >= cutoff:
                degrees[cutoff][a].add(b)
                degrees[cutoff][b].add(a)
    genes = sorted(set(degrees[400]) | set(degrees[700]))
    rows = []
    for gene in genes:
        rows.append(
            {
                "gene_id": gene,
                "string_network_connections_400": float(len(degrees[400].get(gene, set()))),
                "string_network_connections_700": float(len(degrees[700].get(gene, set()))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def expression_features(path: Path | None) -> pd.DataFrame:
    columns = ["gene_id", "median_expression", "expression_variation", "expression_breadth", "expression_module_size"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, sep="\t")
    if "gene_id" not in df.columns:
        raise ValueError("Expression matrix must contain a gene_id column")
    values = df.drop(columns=["gene_id"]).apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"gene_id": df["gene_id"].astype(str).map(first_gene_id)})
    out["median_expression"] = values.median(axis=1)
    out["expression_variation"] = values.std(axis=1) / values.mean(axis=1).replace(0, np.nan)
    out["expression_breadth"] = (values > 0).sum(axis=1) / values.shape[1]
    out["expression_module_size"] = np.nan
    return out[columns]


def domain_features(path: Path | None) -> pd.DataFrame:
    columns = ["gene_id", "domain_number", "pfam_domain_number"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"gene_id", "domain_id", "source"}.issubset(df.columns):
        raise ValueError("Domain table must contain gene_id, domain_id and source columns")
    df["gene_id"] = df["gene_id"].map(first_gene_id)
    total = df.groupby("gene_id")["domain_id"].nunique().rename("domain_number")
    pfam = (
        df[df["source"].str.contains("pfam", case=False, na=False) | df["domain_id"].str.startswith("PF", na=False)]
        .groupby("gene_id")["domain_id"]
        .nunique()
        .rename("pfam_domain_number")
    )
    return pd.concat([total, pfam], axis=1).fillna(0).reset_index()[columns]


def load_plm(prefix: Path, genes: list[str], feature_names: list[str]) -> pd.DataFrame:
    plm_blocks = []
    for model_name in ["esm2", "protbert", "prott5"]:
        emb_path = prefix / model_name / "ath" / "all_emb.npy"
        ids_path = prefix / model_name / "ath" / "all_ids.npy"
        if not emb_path.exists() or not ids_path.exists():
            emb_path = prefix / model_name / "all_emb.npy"
            ids_path = prefix / model_name / "all_ids.npy"
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing precomputed PLM files for {model_name} under {prefix}")
        ids = np.load(ids_path, allow_pickle=True).astype(str)
        arr = np.load(emb_path, mmap_mode="r")
        lookup = {first_gene_id(seq_id): idx for idx, seq_id in enumerate(ids)}
        needed = [name for name in feature_names if name.startswith(f"{model_name}_")]
        block = np.full((len(genes), len(needed)), np.nan, dtype=np.float32)
        for row_idx, gene in enumerate(genes):
            src_idx = lookup.get(gene)
            if src_idx is not None:
                block[row_idx, :] = arr[src_idx, : len(needed)]
        plm_blocks.append(pd.DataFrame(block, columns=needed))
    return pd.concat([pd.DataFrame({"gene_id": genes})] + plm_blocks, axis=1)


def merge_feature_tables(gene_ids: list[str], tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame({"gene_id": gene_ids})
    for table in tables:
        if table is None or table.empty:
            continue
        table = table.copy()
        table["gene_id"] = table["gene_id"].astype(str).map(first_gene_id)
        out = out.merge(table.drop_duplicates("gene_id"), on="gene_id", how="left")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deployable profile features from website-style raw uploads.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--plm-dir", type=Path, default=None, help="Precomputed PLM root containing esm2/protbert/prott5.")
    parser.add_argument("--go-obo", type=Path, default=None)
    args = parser.parse_args()

    feature_names = pd.read_csv(args.profile_dir / "feature_names.tsv", sep="\t")["feature_name"].astype(str).tolist()
    input_dir = args.input_dir
    cds_path = input_dir / "cds.fasta"
    protein_path = input_dir / "protein.fasta"
    if not protein_path.exists():
        raise FileNotFoundError("protein.fasta is required")
    cds_records = collapse_to_longest_by_gene(read_fasta(cds_path)) if cds_path.exists() else {}
    protein_records = collapse_to_longest_by_gene(read_fasta(protein_path))
    seq = sequence_features(cds_records, protein_records)
    gene_ids = seq["gene_id"].astype(str).tolist()

    tables = [
        seq,
        gff_features(input_dir / "annotation.gff3"),
        go_features(input_dir / "go_annotation.tsv", args.go_obo),
        ppi_features(input_dir / "ppi_edges.tsv"),
        expression_features(input_dir / "expression_matrix.tsv"),
        domain_features(input_dir / "domain_annotation.tsv"),
    ]
    merged = merge_feature_tables(gene_ids, tables)
    if args.plm_dir is not None:
        plm = load_plm(args.plm_dir, gene_ids, feature_names)
        merged = merged.merge(plm, on="gene_id", how="left")

    for feature in feature_names:
        if feature not in merged.columns:
            merged[feature] = np.nan
    final = merged[["gene_id"] + feature_names].copy()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out_prefix.with_suffix(".features.tsv"), sep="\t", index=False)
    x = final[feature_names].to_numpy(np.float32)
    np.savez_compressed(
        args.out_prefix.with_suffix(".features.npz"),
        X=x,
        gene_id=final["gene_id"].astype(str).to_numpy(),
        feature_names=np.array(feature_names),
    )
    report = {
        "genes": int(len(final)),
        "features": int(len(feature_names)),
        "missing_fraction": float(pd.isna(final[feature_names]).to_numpy().mean()),
        "profile_dir": str(args.profile_dir),
    }
    args.out_prefix.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
