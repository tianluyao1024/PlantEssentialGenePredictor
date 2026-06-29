from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).parent))
import train_ath_no_conflict_optimized as opt
import train_rice_species_specific_strict_highconf_repeated as base
from train_pseudo06_predict_all_longest_unknown import transform_with


ROOT = Path("E:/CodexMoved/Desktop/\u6c34\u7a3b")
COMMON = ROOT / "cross_species_ath_rice_common_features_models"
FRESH = COMMON / "rice_rapdb_native_features_fresh_only"
ATH_SCHEMA = COMMON / "ath_feature_schema_from_final_model" / "ath_final_feature_names.tsv"
LABELS = ROOT / "rice_mutant_sources" / "source_pair_noise_diagnostic" / "Tos17_RAP_ES_PLUS_Oryzabase_trait_gene_labels_no_conflict.tsv"
OUT = COMMON / "rice_internal_enhanced_replacement_v1"

SEQ = FRESH / "rice_rapdb_native_sequence_numeric_features.tsv"
EXTRA = FRESH / "rice_rapdb_native_extra_features.tsv"
GO_PPI = FRESH / "rice_rapdb_native_go_ppi_features.tsv"
PLM = FRESH / "plm_embeddings_rap_native_fresh"
COMMON_LOC = COMMON / "rice_cross_species_common_features_all_genes.tsv"
STABLE_LOC = COMMON / "rice_stable_external_features.tsv"
PAPER_LOC = ROOT / "paper_2015_lloyd_essential_gene" / "paper_style_features" / "lloyd2015_rice_paper_style_features.tsv"

PRED_COLS = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3", "logistic", "mean_all", "mean_tree"]
CONFIGS = [
    {"name": "seed20260610_pca512_k700", "seed": 20260610, "pca": 512, "k": 700},
    {"name": "seed20260611_pca512_k700", "seed": 20260611, "pca": 512, "k": 700},
    {"name": "seed20260612_pca768_k900", "seed": 20260612, "pca": 768, "k": 900},
    {"name": "seed20260613_pca1024_k1100", "seed": 20260613, "pca": 1024, "k": 1100},
    {"name": "seed20260614_pca384_k550", "seed": 20260614, "pca": 384, "k": 550},
]
SEEDS = list(range(20260720, 20260730))

AA = "ACDEFGHIKLMNPQRSTVWY"
AROMATIC_AA = set("FWY")
INSTABILITY_WEIGHT = {
    # Kyte-Doolittle-based fallback proxy, not the Guruprasad dipeptide table.
    # Used only to preserve an ordered protein-stability signal without external packages.
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
GO_NAME_TO_SOURCE = {
    "go_cellular_component_organization": ("paper", "GOslim P cellular component organization"),
    "go_rna_binding": ("go_term", "GO:0003723"),
    "go_cell_cycle": ("paper", "GOslim P cell cycle"),
    "go_response_to_stress": ("go_term", "GO:0006950"),
    "go_dna_binding_transcription_factor_activity": ("paper", "GOslim F sequence-specific DNA binding transcription factor activity"),
    "go_translation": ("paper", "GOslim P translation"),
    "go_response_to_abiotic_stimulus": ("paper", "GOslim P response to abiotic stimulus"),
    "go_pollination": ("paper", "GOslim P pollination"),
    "go_dna_binding": ("go_term", "GO:0003677"),
    "go_response_to_biotic_stimulus": ("paper", "GOslim P response to biotic stimulus"),
    "go_ribosome": ("go_term", "GO:0005840"),
    "go_nucleolus": ("paper", "GOslim C nucleolus"),
    "go_structural_molecule_activity": ("go_term", "GO:0005198"),
    "go_signal_transduction": ("paper", "GOslim P signal transduction"),
    "go_nucleobase_containing_compound_metabolic_process": ("paper", "GOslim P nucleobase-containing compound metabolic process"),
    "go_chloroplast": ("paper", "GOslim C plastid"),
    "go_extracellular_region": ("paper", "GOslim C extracellular region"),
    "go_response_to_endogenous_stimulus": ("paper", "GOslim P response to endogenous stimulus"),
    "go_response_to_chemical": ("go_term", "GO:0042221"),
    "go_response_to_light_stimulus": ("go_term", "GO:0009416"),
    "go_nucleic_acid_binding": ("go_term", "GO:0003676"),
}

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Features .* are constant.*")
warnings.filterwarnings("ignore", message="invalid value encountered in divide.*")


def rap_from_transcript(value: object) -> str:
    text = str(value).split()[0].split("-")[0]
    if text.startswith("Os") and "t" in text[:5]:
        return text.replace("t", "g", 1)
    return ""


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce")


def parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    current = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0].split("-")[0]
                seqs[current] = []
            elif current is not None:
                seqs[current].append(line.strip().upper().replace("*", ""))
    return {k: "".join(v) for k, v in seqs.items()}


def fallback_instability(seq: str) -> float:
    if not seq:
        return np.nan
    vals = [INSTABILITY_WEIGHT.get(aa, 0.0) for aa in seq if aa in INSTABILITY_WEIGHT]
    return float(np.std(vals) * 10.0) if vals else np.nan


def fallback_pi(row: pd.Series) -> float:
    pos = float(row.get("aa_1mer_K", 0) or 0) + float(row.get("aa_1mer_R", 0) or 0) + 0.1 * float(row.get("aa_1mer_H", 0) or 0)
    neg = float(row.get("aa_1mer_D", 0) or 0) + float(row.get("aa_1mer_E", 0) or 0)
    return float(7.0 + 3.0 * math.tanh((pos - neg) * 8.0))


def load_loc_tables() -> pd.DataFrame:
    common = pd.read_csv(COMMON_LOC, sep="\t", usecols=["gene_id", "transcript_id"], dtype=str)
    common["rap_gene_id"] = common["transcript_id"].map(rap_from_transcript)
    common = common[common["rap_gene_id"].astype(str).str.len().gt(0)]
    common = common.sort_values(["rap_gene_id", "gene_id"]).drop_duplicates("rap_gene_id", keep="first")
    common = common.rename(columns={"gene_id": "loc_gene_id"})

    stable = pd.read_csv(STABLE_LOC, sep="\t")
    stable["gene_id"] = stable["gene_id"].astype(str)
    paper = pd.read_csv(PAPER_LOC, sep="\t")
    paper["gene_id"] = paper["gene_id"].astype(str)
    merged = common.merge(stable, left_on="loc_gene_id", right_on="gene_id", how="left", suffixes=("", "_stable"))
    merged = merged.merge(paper, left_on="loc_gene_id", right_on="gene_id", how="left", suffixes=("", "_paper"))
    return merged.drop(columns=[c for c in ["gene_id", "gene_id_paper"] if c in merged.columns], errors="ignore")


def load_plm(gene_order: pd.Series) -> tuple[np.ndarray, list[str], list[dict]]:
    order = gene_order.astype(str).to_numpy()
    blocks = []
    names = []
    coverage = []
    for model, dim in [("esm2", 2560), ("protbert", 2048), ("prott5", 2048)]:
        root = PLM / model / "rice"
        ids = np.load(root / "all_ids.npy", allow_pickle=True).astype(str)
        emb = np.load(root / "all_emb.npy", mmap_mode="r")
        lookup = {gene_id: i for i, gene_id in enumerate(ids)}
        arr = np.full((len(order), dim), np.nan, dtype=np.float32)
        hit = np.zeros(len(order), dtype=bool)
        for row, gene_id in enumerate(order):
            idx = lookup.get(gene_id)
            if idx is not None:
                arr[row] = emb[idx]
                hit[row] = True
        blocks.append(arr)
        names.extend([f"{model}_{i}" for i in range(dim)])
        coverage.append({"feature_group": model, "available_genes": int(hit.sum()), "total_genes": int(len(order)), "feature_count": dim})
    return np.hstack(blocks).astype(np.float32), names, coverage


def build_bio_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ath = pd.read_csv(ATH_SCHEMA, sep="\t")
    bio_names = ath.loc[ath["block"].eq("bio"), "feature_name"].tolist()

    seq = pd.read_csv(SEQ, sep="\t", low_memory=False)
    seq["gene_id"] = seq["gene_id"].astype(str)
    extra = pd.read_csv(EXTRA, sep="\t", low_memory=False)
    extra["gene_id"] = extra["gene_id"].astype(str)
    go_ppi_cols = ["gene_id"] + [f"go_term_{go.replace(':', '_')}" for _, go in GO_NAME_TO_SOURCE.values() if _ == "go_term"]
    go_ppi_cols += ["ppi_string_degree_ge400", "ppi_string_degree_ge700"]
    go_ppi_cols = list(dict.fromkeys([c for c in go_ppi_cols if c == "gene_id" or c in pd.read_csv(GO_PPI, sep="\t", nrows=0).columns]))
    go_ppi = pd.read_csv(GO_PPI, sep="\t", usecols=go_ppi_cols, low_memory=False)
    go_ppi["gene_id"] = go_ppi["gene_id"].astype(str)
    loc = load_loc_tables()
    loc["rap_gene_id"] = loc["rap_gene_id"].astype(str)

    df = seq.merge(extra.drop(columns=["rap_gene_id", "transcript_id"], errors="ignore"), on="gene_id", how="left", suffixes=("", "_extra"))
    df = df.merge(go_ppi, on="gene_id", how="left")
    df = df.merge(loc, left_on="gene_id", right_on="rap_gene_id", how="left")
    df = df.sort_values("gene_id").drop_duplicates("gene_id", keep="first").reset_index(drop=True)

    proteins = parse_fasta(FRESH / "rice_rapdb_native_longest_protein.fasta")
    transcript_col = "transcript_id"
    if transcript_col not in df.columns:
        transcript_col = "transcript_id_x" if "transcript_id_x" in df.columns else "transcript_id_y"
    out = pd.DataFrame({"gene_id": df["gene_id"].astype(str), "transcript_id": df[transcript_col].astype(str)})
    status_rows = []

    def assign(name: str, values, source: str, note: str = ""):
        out[name] = pd.to_numeric(values, errors="coerce")
        status_rows.append(
            {
                "feature_name": name,
                "status": "available" if out[name].notna().sum() > 0 else "unavailable",
                "source": source,
                "non_missing": int(out[name].notna().sum()),
                "note": note,
            }
        )

    direct = {
        "protein_length": "protein_len",
        "cds_length": "cds_len",
        "gene_span_bp": "gene_len_gff",
        "gc_content": "gc_content",
        "at_content": "at_content",
        "gc3_content": "gc3",
        "protein_molecular_weight": "molecular_weight_mean_aa",
        "protein_gravy": "gravy_kd",
        "aa_group_hydrophobic": "aa_group_hydrophobic",
        "aa_group_polar": "aa_group_polar",
        "aa_group_positive": "aa_group_positive",
        "aa_group_negative": "aa_group_negative",
        "aa_group_aromatic": "aa_group_aromatic",
        "aa_group_small": "aa_group_tiny",
        "gene_family_size": "paralog_family_size_40cov40",
        "singleton_status": "paralog_singleton_40cov40",
        "paralog_percentage_identity": "paralog_top_identity",
        "top_paralog_bitscore": "paralog_top_bitscore",
        "tandem_duplicate": "Tandem duplicate",
        "domain_number": "No. of protein domains",
        "pfam_domain_number": "No. of protein domains",
        "median_expression": "Median expression",
        "expression_variation": "Expression variation",
        "expression_breadth": "Expression breadth",
        "expression_module_size": "Co-expression module size",
        "string_network_connections_400": "ppi_string_degree_ge400",
        "string_network_connections_700": "ppi_string_degree_ge700",
        "ensembl_compara_paralog_count": "paralog_family_size_30cov30",
        "ensembl_compara_max_paralog_percent_identity": "paralog_top_identity",
    }
    for feat, col in direct.items():
        assign(feat, safe_num(df, col), col)
    for nt in "ACGT":
        assign(f"nt_freq_{nt}", safe_num(df, f"cds_1mer_{nt}"), f"cds_1mer_{nt}")
    for aa in AA:
        assign(f"aa_freq_{aa}", safe_num(df, f"aa_1mer_{aa}"), f"aa_1mer_{aa}")

    assign("gc_skew", (safe_num(df, "cds_1mer_G") - safe_num(df, "cds_1mer_C")) / (safe_num(df, "cds_1mer_G") + safe_num(df, "cds_1mer_C")).replace(0, np.nan), "computed")
    assign("at_skew", (safe_num(df, "cds_1mer_A") - safe_num(df, "cds_1mer_T")) / (safe_num(df, "cds_1mer_A") + safe_num(df, "cds_1mer_T")).replace(0, np.nan), "computed")
    assign("aa_group_sulfur", safe_num(df, "aa_1mer_C") + safe_num(df, "aa_1mer_M"), "computed")
    assign("protein_aromaticity", df["gene_id"].map(lambda g: sum(1 for x in proteins.get(g, "") if x in AROMATIC_AA) / max(1, len(proteins.get(g, "")))), "computed_from_longest_protein")
    assign("protein_instability_index", df["gene_id"].map(lambda g: fallback_instability(proteins.get(g, ""))), "computed_proxy_from_longest_protein")
    assign("protein_isoelectric_point", df.apply(fallback_pi, axis=1), "computed_proxy_from_aa_frequency")

    for feat in bio_names:
        if feat in out.columns:
            continue
        if feat in GO_NAME_TO_SOURCE:
            kind, source_name = GO_NAME_TO_SOURCE[feat]
            if kind == "paper":
                assign(feat, safe_num(df, source_name), source_name)
            else:
                assign(feat, safe_num(df, f"go_term_{source_name.replace(':', '_')}"), source_name)
        elif feat == "rice_homolog_found":
            assign(feat, pd.Series(np.nan, index=df.index), "not_used", "self-homolog feature; omitted in rice-internal model")
        elif feat == "rice_homolog_percent_identity":
            assign(feat, pd.Series(np.nan, index=df.index), "not_used", "self-homolog feature; omitted in rice-internal model")
        elif feat == "homolog_not_found_in_rice":
            assign(feat, pd.Series(np.nan, index=df.index), "not_used", "self-homolog feature; omitted in rice-internal model")
        elif feat.startswith(("alyrata_", "ptrichocarpa_", "vvinifera_", "ppatens_", "meth", "snpeff", "fst1001", "omega1001")) or feat in {
            "percentage_identity_in_plants",
            "expression_correlation",
            "ensembl_compara_paralog_lca_type_count",
            "go_embryo_development",
            "go_multicellular_organism_development",
            "go_post_embryonic_development",
            "go_anatomical_structure_development",
            "go_other_intracellular_components",
            "go_response_to_external_stimulus",
        }:
            assign(feat, pd.Series(np.nan, index=df.index), "unavailable_v1", "requires new external download/orthology/variant/methylome processing")
        else:
            assign(feat, pd.Series(np.nan, index=df.index), "unmapped_v1")

    status = pd.DataFrame(status_rows).drop_duplicates("feature_name", keep="last")
    available = [f for f in bio_names if f in out.columns and out[f].notna().sum() > 0 and out[f].nunique(dropna=True) > 1]
    bio = out[["gene_id", "transcript_id"] + available].copy()
    return bio, status, pd.DataFrame({"ath_bio_feature_name": bio_names, "used_in_v1": [f in available for f in bio_names]})


def load_feature_matrix() -> tuple[np.ndarray, pd.DataFrame, list[str], int, pd.DataFrame, pd.DataFrame]:
    bio, status, schema = build_bio_matrix()
    X_bio = bio.drop(columns=["gene_id", "transcript_id"]).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    bio_names = [c for c in bio.columns if c not in {"gene_id", "transcript_id"}]
    X_plm, plm_names, plm_cov = load_plm(bio["gene_id"])
    X = np.hstack([X_bio, X_plm]).astype(np.float32)
    feature_names = bio_names + plm_names
    meta = bio[["gene_id", "transcript_id"]].copy()
    coverage = pd.DataFrame(
        [{"feature_group": "bio_replacement_v1", "available_genes": int(len(bio)), "total_genes": int(len(bio)), "feature_count": int(len(bio_names))}]
        + plm_cov
    )
    return X, meta, feature_names, len(bio_names), coverage, status.merge(schema, left_on="feature_name", right_on="ath_bio_feature_name", how="outer")


def logit_mean(x: np.ndarray) -> np.ndarray:
    p = np.clip(x, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def fit_predict_library(X_train: np.ndarray, y_train: np.ndarray, targets: dict[str, np.ndarray], n_bio: int, tag: str):
    target_cols = {name: [] for name in targets}
    train_cols = []
    meta_names = []
    final_models = []
    fold_rows = []
    pos_weight = float((y_train == 0).sum() / max(1, y_train.sum()))
    for config in CONFIGS:
        seed = int(config["seed"])
        model_defs = opt.make_models(pos_weight, seed)
        Xtr, _, transforms = opt.make_fold_features(
            X_train, X_train, n_bio, y_train, k=int(config["k"]), n_pca_limit=int(config["pca"])
        )
        model_preds = {}
        for model_name, model_def in model_defs.items():
            model = base.limit_model_jobs(clone(model_def))
            base.fit_with_optional_weight(model, Xtr, y_train, np.ones(len(y_train), dtype=np.float32))
            model_preds[model_name] = model.predict_proba(Xtr)[:, 1]
            final_models.append({"config": config, "model_name": model_name, "model": model, "transforms": transforms})
        tree_names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
        model_preds["mean_all"] = np.mean([model_preds[n] for n in model_defs], axis=0)
        model_preds["mean_tree"] = np.mean([model_preds[n] for n in tree_names], axis=0)
        for model_name in PRED_COLS:
            train_cols.append(model_preds[model_name])
            meta_names.append(f"{config['name']}__{model_name}")
        for target_name, X_target in targets.items():
            Xt = transform_with(transforms, X_target, n_bio)
            target_model_preds = {}
            for item in final_models[-len(model_defs):]:
                target_model_preds[item["model_name"]] = item["model"].predict_proba(Xt)[:, 1]
            target_model_preds["mean_all"] = np.mean([target_model_preds[n] for n in model_defs], axis=0)
            target_model_preds["mean_tree"] = np.mean([target_model_preds[n] for n in tree_names], axis=0)
            for model_name in PRED_COLS:
                target_cols[target_name].append(target_model_preds[model_name])
        train_mean_auc = roc_auc_score(y_train, np.mean([model_preds[n] for n in PRED_COLS if n not in {"mean_all", "mean_tree"}], axis=0))
        fold_rows.append({"tag": tag, "config": config["name"], "train_resub_mean_base_auc": float(train_mean_auc)})
        print(f"{tag} trained {config['name']} train_resub_mean_base_auc={train_mean_auc:.4f}", flush=True)
    return np.vstack(train_cols).T.astype(np.float32), {k: np.vstack(v).T.astype(np.float32) for k, v in target_cols.items()}, meta_names, final_models, fold_rows


def fit_cv_stack_predict(X_train, y_train, targets: dict[str, np.ndarray], n_bio: int, tag: str):
    oof_cols = []
    target_cols = {name: [] for name in targets}
    fold_rows = []
    for config in CONFIGS:
        seed = int(config["seed"])
        folds = opt.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pos_weight = float((y_train == 0).sum() / max(1, y_train.sum()))
        model_defs = opt.make_models(pos_weight, seed)
        oof = {name: np.zeros(len(y_train), dtype=np.float32) for name in model_defs}
        oof["mean_all"] = np.zeros(len(y_train), dtype=np.float32)
        oof["mean_tree"] = np.zeros(len(y_train), dtype=np.float32)
        target_fold = {name: {model_name: [] for model_name in PRED_COLS} for name in targets}
        for fold, (tr, va) in enumerate(folds.split(X_train, y_train), 1):
            Xtr, Xva, transforms = opt.make_fold_features(
                X_train[tr], X_train[va], n_bio, y_train[tr], k=int(config["k"]), n_pca_limit=int(config["pca"])
            )
            transformed_targets = {name: transform_with(transforms, X_target, n_bio) for name, X_target in targets.items()}
            val_preds = {}
            target_preds = {name: {} for name in targets}
            for model_name, model_def in model_defs.items():
                model = base.limit_model_jobs(clone(model_def))
                model.fit(Xtr, y_train[tr])
                val_prob = model.predict_proba(Xva)[:, 1]
                oof[model_name][va] = val_prob
                val_preds[model_name] = val_prob
                for target_name, Xt in transformed_targets.items():
                    target_preds[target_name][model_name] = model.predict_proba(Xt)[:, 1]
            tree_names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
            oof["mean_all"][va] = np.mean([val_preds[n] for n in model_defs], axis=0)
            oof["mean_tree"][va] = np.mean([val_preds[n] for n in tree_names], axis=0)
            for target_name in targets:
                target_fold[target_name]["mean_all"].append(np.mean([target_preds[target_name][n] for n in model_defs], axis=0))
                target_fold[target_name]["mean_tree"].append(np.mean([target_preds[target_name][n] for n in tree_names], axis=0))
                for model_name in model_defs:
                    target_fold[target_name][model_name].append(target_preds[target_name][model_name])
            auc = roc_auc_score(y_train[va], oof["mean_all"][va])
            fold_rows.append({"tag": tag, "config": config["name"], "fold": fold, "mean_all_auc": float(auc)})
            print(f"{tag} {config['name']} fold {fold}: mean_all_auc={auc:.4f}", flush=True)
        for model_name in PRED_COLS:
            oof_cols.append(oof[model_name])
            for target_name in targets:
                target_cols[target_name].append(np.mean(target_fold[target_name][model_name], axis=0))
    X_oof = np.vstack(oof_cols).T.astype(np.float32)
    X_targets = {name: np.vstack(cols).T.astype(np.float32) for name, cols in target_cols.items()}
    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(512, len(y_train)), output_distribution="normal", random_state=1)),
            ("clf", LogisticRegression(C=0.02, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=7)),
        ]
    )
    meta.fit(X_oof, y_train)
    preds = {"meta": meta.predict_proba(X_oof)[:, 1], "mean": X_oof.mean(axis=1), "logit_mean": logit_mean(X_oof)}
    target_preds = {
        name: {"meta": meta.predict_proba(Xm)[:, 1], "mean": Xm.mean(axis=1), "logit_mean": logit_mean(Xm)}
        for name, Xm in X_targets.items()
    }
    return preds, target_preds, fold_rows


def train_eval_and_predict(args):
    OUT.mkdir(parents=True, exist_ok=True)
    X_all, meta_all, feature_names, n_bio, coverage, feature_status = load_feature_matrix()
    labels = pd.read_csv(LABELS, sep="\t")
    if "final_label" in labels.columns:
        labels["label"] = pd.to_numeric(labels["final_label"], errors="coerce").astype(int)
    labels["rap_gene_id"] = labels["rap_gene_id"].astype(str)
    meta_all["matrix_row"] = np.arange(len(meta_all), dtype=int)
    labeled = meta_all.merge(labels, left_on="gene_id", right_on="rap_gene_id", how="inner").reset_index(drop=True)
    X = X_all[labeled["matrix_row"].to_numpy(int)].astype(np.float32)
    y = labeled["label"].astype(int).to_numpy(np.int8)
    labeled = labeled.drop(columns=["matrix_row"])
    unknown_meta = meta_all[~meta_all["gene_id"].isin(set(labeled["gene_id"]))].copy().reset_index(drop=True)
    X_unknown = X_all[unknown_meta["matrix_row"].to_numpy(int)].astype(np.float32)
    unknown_meta = unknown_meta.drop(columns=["matrix_row"])

    pd.DataFrame({"feature_name": feature_names, "block": ["bio"] * n_bio + ["plm"] * (len(feature_names) - n_bio)}).to_csv(OUT / "feature_names.tsv", sep="\t", index=False)
    feature_status.to_csv(OUT / "ath_schema_replacement_status.tsv", sep="\t", index=False)
    coverage.to_csv(OUT / "feature_coverage.tsv", sep="\t", index=False)
    labeled.to_csv(OUT / "labeled_feature_intersection.tsv", sep="\t", index=False)

    score_path = OUT / "holdout_scores.tsv"
    fold_path = OUT / "fold_scores.tsv"
    pred_path = OUT / "test_predictions.tsv"
    all_scores = []
    all_folds = []
    all_test_predictions = []
    completed_seeds: set[int] = set()
    if score_path.exists():
        prev = pd.read_csv(score_path, sep="\t")
        all_scores = prev.to_dict("records")
        done = prev.groupby("seed")["model"].nunique()
        completed_seeds = set(done[done >= 3].index.astype(int))
    if fold_path.exists():
        all_folds = pd.read_csv(fold_path, sep="\t").to_dict("records")
    if pred_path.exists():
        all_test_predictions = [pd.read_csv(pred_path, sep="\t")]
    for seed in SEEDS[: args.max_seeds if args.max_seeds else None]:
        if seed in completed_seeds:
            print(f"skip completed seed {seed}", flush=True)
            continue
        train_idx, val_idx, test_idx = base.split_80_10_10(y, seed)
        source_preds, target_preds, fold_rows = fit_cv_stack_predict(
            X[train_idx], y[train_idx], {"validation": X[val_idx], "test": X[test_idx]}, n_bio, f"seed_{seed}"
        )
        all_folds.extend(fold_rows)
        for model_name in ["meta", "mean", "logit_mean"]:
            val_prob = target_preds["validation"][model_name]
            test_prob = target_preds["test"][model_name]
            threshold = base.best_threshold(y[val_idx], val_prob)
            metrics = base.binary_metrics(y[test_idx], test_prob, threshold["threshold"])
            all_scores.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_n": int(len(train_idx)),
                    "validation_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                    "train_positive": int(y[train_idx].sum()),
                    "validation_positive": int(y[val_idx].sum()),
                    "test_positive": int(y[test_idx].sum()),
                    "validation_threshold": float(threshold["threshold"]),
                    "train_oof_auc": float(roc_auc_score(y[train_idx], source_preds[model_name])),
                    "train_oof_auprc": float(average_precision_score(y[train_idx], source_preds[model_name])),
                    "validation_auc": float(roc_auc_score(y[val_idx], val_prob)),
                    "validation_auprc": float(average_precision_score(y[val_idx], val_prob)),
                    **{f"test_{k}": v for k, v in metrics.items()},
                }
            )
            all_test_predictions.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "model": model_name,
                        "rap_gene_id": labeled.iloc[test_idx]["gene_id"].to_numpy(),
                        "label": y[test_idx],
                        "probability": test_prob,
                        "threshold": threshold["threshold"],
                    }
                )
            )
        score_df = pd.DataFrame(all_scores)
        score_df.to_csv(score_path, sep="\t", index=False)
        base.summarize_scores(score_df).to_csv(OUT / "holdout_summary.tsv", sep="\t", index=False)
        pd.DataFrame(all_folds).to_csv(fold_path, sep="\t", index=False)
        pd.concat(all_test_predictions, ignore_index=True).to_csv(pred_path, sep="\t", index=False)
        print(base.summarize_scores(score_df).to_string(index=False), flush=True)

    if args.predict_unknown:
        X_meta_train, X_meta_targets, meta_names, final_models, train_rows = fit_predict_library(
            X, y, {"unknown": X_unknown, "all_genes": X_all}, n_bio, "final_all_labeled"
        )
        meta_model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("qt", QuantileTransformer(n_quantiles=min(512, len(y)), output_distribution="normal", random_state=1)),
                ("clf", LogisticRegression(C=0.02, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=7)),
            ]
        )
        base_cols = [i for i, name in enumerate(meta_names) if not name.endswith(("mean_all", "mean_tree"))]
        meta_model.fit(X_meta_train[:, base_cols], y)
        unk_prob = meta_model.predict_proba(X_meta_targets["unknown"][:, base_cols])[:, 1]
        all_prob = meta_model.predict_proba(X_meta_targets["all_genes"][:, base_cols])[:, 1]
        unknown_pred = unknown_meta.copy()
        unknown_pred["essential_probability"] = unk_prob
        unknown_pred["prediction_0.5"] = np.where(unknown_pred["essential_probability"].ge(0.5), "essential", "nonessential")
        unknown_pred.sort_values("essential_probability", ascending=False).to_csv(OUT / "unknown_gene_essentiality_predictions.tsv", sep="\t", index=False)
        all_pred = meta_all.drop(columns=["matrix_row"], errors="ignore").copy()
        all_pred["essential_probability"] = all_prob
        all_pred["is_labeled_training_gene"] = all_pred["gene_id"].isin(set(labeled["gene_id"]))
        all_pred.sort_values("essential_probability", ascending=False).to_csv(OUT / "all_gene_essentiality_predictions.tsv", sep="\t", index=False)
        pd.DataFrame({"meta_feature_name": meta_names}).to_csv(OUT / "final_meta_feature_names.tsv", sep="\t", index=False)
        pd.DataFrame(train_rows).to_csv(OUT / "final_training_rows.tsv", sep="\t", index=False)
        joblib.dump({"meta_model": meta_model, "final_models": final_models, "meta_feature_names": meta_names, "feature_names": feature_names, "n_bio": n_bio}, OUT / "final_model_package.joblib")

    manifest = {
        "task": "rice internal enhanced-replacement v1 model and unknown-gene prediction",
        "label_file": str(LABELS),
        "label_rule": "Tos17_RAP_ES + Oryzabase_trait_gene no-conflict binary labels; conditional Tos17 genes excluded upstream",
        "feature_rule": "RAP-native longest transcript features aligned to Arabidopsis final schema where available; unavailable 3K/SnpEff/methylation/FST/omega features are omitted and logged",
        "n_all_feature_genes": int(len(meta_all)),
        "n_labeled_with_features": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "n_unknown_predicted": int(len(unknown_meta)),
        "n_features": int(len(feature_names)),
        "n_bio_features_used": int(n_bio),
        "n_plm_features": int(len(feature_names) - n_bio),
        "configs": CONFIGS,
        "seeds": SEEDS[: args.max_seeds if args.max_seeds else None],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--predict-unknown", action="store_true")
    args = parser.parse_args()
    train_eval_and_predict(args)


if __name__ == "__main__":
    main()
