from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import train_rice_internal_enhanced_replacement_v1 as v1


ROOT = Path("E:/CodexMoved/Desktop/水稻")
COMMON = ROOT / "cross_species_ath_rice_common_features_models"
SUPPLEMENT = COMMON / "rice_internal_enhanced_replacement_v2_supplement" / "rice_enhanced_replacement_v2_supplement_features.tsv"


def build_bio_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bio, status, schema = v1.build_bio_matrix()
    supplement = pd.read_csv(SUPPLEMENT, sep="\t")
    supplement["gene_id"] = supplement["gene_id"].astype(str)
    supplement_cols = [c for c in supplement.columns if c != "gene_id"]

    merged = bio.merge(supplement, on="gene_id", how="left", suffixes=("", "__v2"))
    status_rows = status.to_dict("records")
    used_from_supplement = []

    for col in supplement_cols:
        supp_col = f"{col}__v2" if col in bio.columns else col
        vals = pd.to_numeric(merged[supp_col], errors="coerce")
        non_missing = int(vals.notna().sum())
        non_unique = int(vals.nunique(dropna=True))
        if non_missing == 0 or non_unique <= 1:
            status_rows.append(
                {
                    "feature_name": col,
                    "status": "unavailable",
                    "source": "v2_supplement",
                    "non_missing": non_missing,
                    "note": "supplement column is empty or constant after RAP longest-gene merge",
                }
            )
            continue
        if col in merged.columns and supp_col != col:
            merged[col] = vals
        elif col not in merged.columns:
            merged[col] = vals
        used_from_supplement.append(col)
        status_rows.append(
            {
                "feature_name": col,
                "status": "available",
                "source": "v2_supplement",
                "non_missing": non_missing,
                "note": "recomputed from RAP-native longest protein, Ensembl Plants GO DAG, or BioMart orthologs",
            }
        )

    drop_cols = [c for c in merged.columns if c.endswith("__v2")]
    merged = merged.drop(columns=drop_cols)
    feature_cols = [c for c in merged.columns if c not in {"gene_id", "transcript_id"}]
    feature_cols = [
        c
        for c in feature_cols
        if pd.to_numeric(merged[c], errors="coerce").notna().sum() > 0
        and pd.to_numeric(merged[c], errors="coerce").nunique(dropna=True) > 1
    ]
    out = merged[["gene_id", "transcript_id"] + feature_cols].copy()
    status_out = pd.DataFrame(status_rows).drop_duplicates("feature_name", keep="last")
    schema_out = schema.rename(columns={"used_in_v1": "used_in_v2"}).copy()
    schema_out["used_in_v2"] = schema_out["ath_bio_feature_name"].isin(feature_cols)
    schema_out["v2_supplemented"] = schema_out["ath_bio_feature_name"].isin(used_from_supplement)
    return out, status_out, schema_out


def load_feature_matrix() -> tuple[np.ndarray, pd.DataFrame, list[str], int, pd.DataFrame, pd.DataFrame]:
    bio, status, schema = build_bio_matrix()
    x_bio = bio.drop(columns=["gene_id", "transcript_id"]).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    bio_names = [c for c in bio.columns if c not in {"gene_id", "transcript_id"}]
    x_plm, plm_names, plm_cov = v1.load_plm(bio["gene_id"])
    x = np.hstack([x_bio, x_plm]).astype(np.float32)
    feature_names = bio_names + plm_names
    meta = bio[["gene_id", "transcript_id"]].copy()
    coverage = pd.DataFrame(
        [{"feature_group": "bio_replacement_v2", "available_genes": int(len(bio)), "total_genes": int(len(bio)), "feature_count": int(len(bio_names))}]
        + plm_cov
    )
    return x, meta, feature_names, len(bio_names), coverage, status.merge(schema, left_on="feature_name", right_on="ath_bio_feature_name", how="outer")
