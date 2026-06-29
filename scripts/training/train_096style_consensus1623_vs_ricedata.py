from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).parent))
import train_ath_high_confidence_models as ath_base
import train_ath_no_conflict_optimized as opt


DESKTOP = Path.home() / "Desktop"
CS_ROOT = DESKTOP / "水稻" / "cross_species_ath_rice_common_features_models"
PAPER_ROOT = DESKTOP / "水稻" / "paper_2015_lloyd_essential_gene"
RICE_LABELS = DESKTOP / "水稻" / "rice_list_output0" / "essentiality_processed_strict_lof" / "rice_documented_gene_labels_strict_lof.csv"
ATH_CONSENSUS = Path("D:/拟南芥/模型/essential_gene_prediction_consensus_2plus/consensus_2plus_gene_labels.tsv")
OUT_DIR = DESKTOP / "水稻" / "cross_species_ath_rice_common_features_models" / "096style_consensus1623_vs_ricedata_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"name": "seed20260610_pca512_k700", "seed": 20260610, "pca": 512, "k": 700},
    {"name": "seed20260611_pca512_k700", "seed": 20260611, "pca": 512, "k": 700},
    {"name": "seed20260612_pca768_k900", "seed": 20260612, "pca": 768, "k": 900},
    {"name": "seed20260613_pca1024_k1100", "seed": 20260613, "pca": 1024, "k": 1100},
    {"name": "seed20260614_pca384_k550", "seed": 20260614, "pca": 384, "k": 550},
]
PRED_COLS = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3", "logistic", "mean_all", "mean_tree"]


def load_ath_matrix_and_labels():
    X, ids, genes, feature_names, n_bio = ath_base.load_matrix()
    genes = np.array([str(g).upper() for g in genes])
    labels = pd.read_csv(ATH_CONSENSUS, sep="\t")
    labels["gene_id"] = labels["gene_id"].astype(str).str.upper()
    label_map = labels.set_index("gene_id")["label"].astype(int)
    idx = np.array([i for i, g in enumerate(genes) if g in label_map.index], dtype=int)
    y = np.array([label_map[g] for g in genes[idx]], dtype=np.int8)
    meta = pd.DataFrame({"seq_id": ids[idx], "gene_id": genes[idx], "label": y})
    return X[idx].astype(np.float32), meta, feature_names, n_bio


def load_global_plm(species: str, model: str, gene_order: list[str]) -> np.ndarray:
    root = CS_ROOT / "plm_embeddings" / model / species
    ids = np.load(root / "all_ids.npy", allow_pickle=True).astype(str)
    emb = np.load(root / "all_emb.npy").astype(np.float32)
    lookup = {g: i for i, g in enumerate(ids)}
    return emb[[lookup[g] for g in gene_order]]


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce")


def load_rice_bio_source() -> pd.DataFrame:
    seq = pd.read_csv(CS_ROOT / "rice_cross_species_common_features_all_genes.tsv", sep="\t")
    ext = pd.read_csv(CS_ROOT / "rice_stable_external_features.tsv", sep="\t")
    paper = pd.read_csv(PAPER_ROOT / "paper_style_features" / "lloyd2015_rice_paper_style_features.tsv", sep="\t")
    df = seq.merge(ext, on="gene_id", how="left").merge(paper.drop(columns=["representative_transcript"], errors="ignore"), on="gene_id", how="left")
    df["gene_id"] = df["gene_id"].astype(str)
    return df


def fill_rice_bio_features(df: pd.DataFrame, bio_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"gene_id": df["gene_id"].astype(str), "seq_id": df["transcript_id"].astype(str)})
    for name in bio_names:
        out[name] = np.nan

    direct = {
        "protein_length": "protein_len",
        "cds_length": "cds_len",
        "gene_span_bp": "gff_gene_span",
        "gc_content": "gc_content",
        "at_content": "at_content",
        "gc3_content": "gc3",
        "nt_freq_A": "cds_1mer_A",
        "nt_freq_C": "cds_1mer_C",
        "nt_freq_G": "cds_1mer_G",
        "nt_freq_T": "cds_1mer_T",
        "aa_group_hydrophobic": "aa_group_hydrophobic",
        "aa_group_polar": "aa_group_polar",
        "aa_group_positive": "aa_group_positive",
        "aa_group_negative": "aa_group_negative",
        "aa_group_aromatic": "aa_group_aromatic",
        "protein_molecular_weight": "molecular_weight_mean_aa",
        "protein_gravy": "gravy_kd",
        "gene_family_size": "paralog_family_size_40cov40",
        "singleton_status": "paralog_singleton_40cov40",
        "paralog_percentage_identity": "paralog_top_identity",
        "top_paralog_bitscore": "paralog_top_bitscore",
        "string_network_connections_400": "string400_degree",
        "string_network_connections_700": "string700_degree",
        "median_expression": "Median expression",
        "expression_variation": "Expression variation",
        "expression_breadth": "Expression breadth",
        "expression_module_size": "Co-expression module size",
        "ensembl_compara_paralog_count": "paralog_family_size_30cov30",
        "ensembl_compara_max_paralog_percent_identity": "paralog_top_identity",
    }
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        direct[f"aa_freq_{aa}"] = f"aa_comp_{aa}"
    for name, src in direct.items():
        if name in out.columns:
            out[name] = safe_numeric(df, src)

    if "gc_skew" in out:
        g = safe_numeric(df, "cds_1mer_G")
        c = safe_numeric(df, "cds_1mer_C")
        out["gc_skew"] = (g - c) / (g + c).replace(0, np.nan)
    if "at_skew" in out:
        a = safe_numeric(df, "cds_1mer_A")
        t = safe_numeric(df, "cds_1mer_T")
        out["at_skew"] = (a - t) / (a + t).replace(0, np.nan)
    if "aa_group_small" in out:
        out["aa_group_small"] = safe_numeric(df, "aa_group_tiny")
    if "aa_group_sulfur" in out:
        out["aa_group_sulfur"] = safe_numeric(df, "aa_comp_C") + safe_numeric(df, "aa_comp_M")
    if "tandem_duplicate" in out:
        out["tandem_duplicate"] = safe_numeric(df, "Tandem duplicate")
    if "domain_number" in out:
        out["domain_number"] = safe_numeric(df, "No. of protein domains")
    if "pfam_domain_number" in out:
        out["pfam_domain_number"] = safe_numeric(df, "No. of protein domains")

    go_map = {
        "go_cellular_component_organization": "GOslim P cellular component organization",
        "go_cell_cycle": "GOslim P cell cycle",
        "go_response_to_abiotic_stimulus": "GOslim P response to abiotic stimulus",
        "go_pollination": "GOslim P pollination",
        "go_response_to_biotic_stimulus": "GOslim P response to biotic stimulus",
        "go_nucleolus": "GOslim C nucleolus",
        "go_signal_transduction": "GOslim P signal transduction",
        "go_nucleobase_containing_compound_metabolic_process": "GOslim P nucleobase-containing compound metabolic process",
        "go_extracellular_region": "GOslim C extracellular region",
        "go_response_to_endogenous_stimulus": "GOslim P response to endogenous stimulus",
        "go_translation": "GOslim P translation",
        "go_chloroplast": "GOslim C plastid",
        "go_rna_binding": "GOslim F nucleotide binding",
        "go_dna_binding_transcription_factor_activity": "GOslim F sequence-specific DNA binding transcription factor activity",
    }
    for name, src in go_map.items():
        if name in out.columns:
            out[name] = safe_numeric(df, src)
    return out


def load_rice_matrix(feature_names: list[str], n_bio: int):
    bio_names = feature_names[:n_bio]
    df = load_rice_bio_source()
    bio = fill_rice_bio_features(df, bio_names)
    gene_order = bio["gene_id"].astype(str).tolist()
    plm_blocks = [
        load_global_plm("rice", "esm2", gene_order),
        load_global_plm("rice", "protbert", gene_order),
        load_global_plm("rice", "prott5", gene_order),
    ]
    X_bio = bio[bio_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    X = np.hstack([X_bio] + plm_blocks).astype(np.float32)
    meta = bio[["seq_id", "gene_id"]].copy()
    meta["seq_id"] = meta["seq_id"].astype(str)
    meta["gene_id"] = meta["gene_id"].astype(str)
    return X, meta, bio_names


def load_rice_ricedata_labels() -> pd.DataFrame:
    labels = pd.read_csv(RICE_LABELS)
    labels = labels.rename(columns={"final_label": "label"})
    labels["gene_id"] = labels["gene_id"].astype(str)
    labels["label"] = labels["label"].astype(int)
    return labels[["gene_id", "label", "final_classification"]]


def logit_mean(X: np.ndarray) -> np.ndarray:
    p = np.clip(X, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def transform_with(transforms, X, n_bio: int):
    imp, scaler, pca, selector = transforms
    X_imp = imp.transform(X)
    X_emb = scaler.transform(X_imp[:, n_bio:])
    X_pca = pca.transform(X_emb)
    X_fold = np.hstack([X_imp[:, :n_bio], X_pca]).astype(np.float32)
    return selector.transform(X_fold)


def fit_oof_predict_target(X_source, y_source, X_target, n_bio: int, tag: str):
    oof_cols, target_cols, names, fold_rows, fold_models = [], [], [], [], []
    for config in CONFIGS:
        seed = int(config["seed"])
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pos_weight = float((y_source == 0).sum() / max(1, y_source.sum()))
        model_defs = opt.make_models(pos_weight, seed)
        oof = {name: np.zeros(len(y_source), dtype=np.float32) for name in model_defs}
        target_fold = {name: [] for name in model_defs}
        oof["mean_all"] = np.zeros(len(y_source), dtype=np.float32)
        oof["mean_tree"] = np.zeros(len(y_source), dtype=np.float32)
        target_fold["mean_all"] = []
        target_fold["mean_tree"] = []
        for fold, (tr, va) in enumerate(folds.split(X_source, y_source), 1):
            Xtr, Xva, transforms = opt.make_fold_features(
                X_source[tr],
                X_source[va],
                n_bio,
                y_source[tr],
                k=int(config["k"]),
                n_pca_limit=int(config["pca"]),
            )
            Xt = transform_with(transforms, X_target, n_bio)
            val_preds, tar_preds = {}, {}
            for model_name, model_def in model_defs.items():
                model = clone(model_def)
                model.fit(Xtr, y_source[tr])
                vp = model.predict_proba(Xva)[:, 1]
                tp = model.predict_proba(Xt)[:, 1]
                oof[model_name][va] = vp
                val_preds[model_name] = vp
                tar_preds[model_name] = tp
                target_fold[model_name].append(tp)
                fold_models.append({"tag": tag, "config": config, "fold": fold, "model_name": model_name, "model": model, "transforms": transforms})
            tree_names = ["extra_sqrt", "extra_log2", "rf_sqrt", "lgbm_gbdt", "xgb_depth3"]
            oof["mean_all"][va] = np.mean([val_preds[n] for n in model_defs], axis=0)
            oof["mean_tree"][va] = np.mean([val_preds[n] for n in tree_names], axis=0)
            target_fold["mean_all"].append(np.mean([tar_preds[n] for n in model_defs], axis=0))
            target_fold["mean_tree"].append(np.mean([tar_preds[n] for n in tree_names], axis=0))
            fold_rows.append(
                {
                    "tag": tag,
                    "config": config["name"],
                    "fold": fold,
                    "mean_all_auc": float(roc_auc_score(y_source[va], oof["mean_all"][va])),
                    "mean_tree_auc": float(roc_auc_score(y_source[va], oof["mean_tree"][va])),
                }
            )
            print(f"{tag} {config['name']} fold{fold}: mean_all_auc={fold_rows[-1]['mean_all_auc']:.4f}", flush=True)
        for model_name in PRED_COLS:
            oof_cols.append(oof[model_name])
            target_cols.append(np.mean(target_fold[model_name], axis=0))
            names.append(f"{config['name']}__{model_name}")
    return (
        np.vstack(oof_cols).T.astype(np.float32),
        np.vstack(target_cols).T.astype(np.float32),
        names,
        fold_rows,
        fold_models,
    )


def run_direction(name: str, X_source, source_meta, X_target, target_meta, n_bio: int):
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    y_source = source_meta["label"].astype(int).to_numpy()
    y_target = target_meta["label"].astype(int).to_numpy()
    X_oof, X_target_meta, meta_names, fold_rows, fold_models = fit_oof_predict_target(X_source, y_source, X_target, n_bio, name)
    meta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(512, len(y_source)), output_distribution="normal", random_state=1)),
            ("clf", LogisticRegression(C=0.02, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=7)),
        ]
    )
    meta.fit(X_oof, y_source)
    source_prob = meta.predict_proba(X_oof)[:, 1]
    target_prob = meta.predict_proba(X_target_meta)[:, 1]
    source_mean = X_oof.mean(axis=1)
    target_mean = X_target_meta.mean(axis=1)
    source_logit = logit_mean(X_oof)
    target_logit = logit_mean(X_target_meta)
    scores = []
    for model_name, sp, tp in [
        ("meta_quantile_logistic_C0.02", source_prob, target_prob),
        ("mean_probability_40", source_mean, target_mean),
        ("logit_mean_40", source_logit, target_logit),
    ]:
        scores.append(
            {
                "direction": name,
                "model": model_name,
                "source_n": int(len(y_source)),
                "source_positive": int(y_source.sum()),
                "source_oof_auc": float(roc_auc_score(y_source, sp)),
                "source_oof_auprc": float(average_precision_score(y_source, sp)),
                "target_n": int(len(y_target)),
                "target_positive": int(y_target.sum()),
                "target_auc": float(roc_auc_score(y_target, tp)),
                "target_auprc": float(average_precision_score(y_target, tp)),
            }
        )
    pd.DataFrame(scores).to_csv(out / "scores.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "source_oof_fold_scores.tsv", sep="\t", index=False)
    pd.DataFrame({"meta_feature_name": meta_names}).to_csv(out / "meta_feature_names.tsv", sep="\t", index=False)
    src_pred = source_meta[["seq_id", "gene_id", "label"]].copy()
    src_pred["meta_oof_probability"] = source_prob
    src_pred["mean_oof_probability"] = source_mean
    src_pred["logit_mean_oof_probability"] = source_logit
    src_pred.to_csv(out / "source_oof_predictions.tsv", sep="\t", index=False)
    tar_pred = target_meta[["seq_id", "gene_id", "label"]].copy()
    tar_pred["meta_probability"] = target_prob
    tar_pred["mean_probability"] = target_mean
    tar_pred["logit_mean_probability"] = target_logit
    tar_pred.to_csv(out / "target_predictions.tsv", sep="\t", index=False)
    joblib.dump({"meta_model": meta, "fold_models": fold_models, "meta_feature_names": meta_names, "configs": CONFIGS}, out / "model_package.joblib")
    return scores


def main():
    X_ath, ath_meta_all, feature_names, n_bio = load_ath_matrix_and_labels()
    X_rice_all, rice_meta_all, bio_names = load_rice_matrix(feature_names, n_bio)
    rice_labels = load_rice_ricedata_labels()
    rice_meta_all = rice_meta_all.copy()
    rice_meta_all["matrix_row"] = np.arange(len(rice_meta_all), dtype=int)
    rice_meta = rice_meta_all.merge(rice_labels, on="gene_id", how="inner")
    rice_idx = rice_meta["matrix_row"].to_numpy(dtype=int)
    X_rice = X_rice_all[rice_idx]
    rice_meta = rice_meta.drop(columns=["matrix_row"])

    ath_meta = ath_meta_all.copy()
    ath_meta["final_classification"] = np.where(ath_meta["label"].eq(1), "essential", "nonessential")

    coverage = []
    rice_bio = pd.DataFrame(X_rice_all[:, :n_bio], columns=feature_names[:n_bio])
    for col in feature_names[:n_bio]:
        coverage.append({"feature": col, "rice_non_missing": int(pd.Series(rice_bio[col]).notna().sum()), "rice_missing": int(pd.Series(rice_bio[col]).isna().sum())})
    pd.DataFrame(coverage).to_csv(OUT_DIR / "rice_096style_bio_feature_coverage.tsv", sep="\t", index=False)

    all_scores = []
    all_scores.extend(run_direction("ath1623_train_predict_ricedata", X_ath, ath_meta, X_rice, rice_meta, n_bio))
    all_scores.extend(run_direction("ricedata_train_predict_ath1623", X_rice, rice_meta, X_ath, ath_meta, n_bio))
    score_df = pd.DataFrame(all_scores)
    score_df.to_csv(OUT_DIR / "all_direction_scores.tsv", sep="\t", index=False)

    manifest = {
        "feature_protocol": "A. thaliana 0.96-style 6839-column schema: 183 bio features + 6656 ESM2/ProtBERT/ProtT5 global embeddings. Rice uses mappable homologous bio columns; A. thaliana-specific bio columns are NaN and imputed.",
        "n_bio_features": int(n_bio),
        "n_total_features": int(len(feature_names)),
        "ath_label_set": "consensus_2plus 1623 high-confidence labels",
        "ath_n": int(len(ath_meta)),
        "ath_positive": int(ath_meta["label"].sum()),
        "rice_label_set": str(RICE_LABELS),
        "rice_n": int(len(rice_meta)),
        "rice_positive": int(rice_meta["label"].sum()),
        "configs": CONFIGS,
        "scores": all_scores,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
