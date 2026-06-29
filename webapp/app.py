from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "prediction"))

from predict_from_processed_features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    load_npz_features,
    predict,
)


st.set_page_config(page_title="Plant Essential Gene Predictor", layout="wide")
st.title("Plant Essential Gene Predictor")
st.caption(
    "Predict essential-gene probability using the released 6751-dimensional "
    "Arabidopsis/rice feature space and trained stacking models."
)

with st.sidebar:
    st.header("Input")
    model = st.selectbox(
        "Model",
        ["arabidopsis_single", "rice_single", "joint"],
        format_func=lambda x: {
            "arabidopsis_single": "Arabidopsis single-species model",
            "rice_single": "Rice single-species model",
            "joint": "Joint Arabidopsis-rice model",
        }[x],
    )
    threshold = st.number_input(
        "Classification threshold",
        value=float(DEFAULT_THRESHOLDS[model]),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    example = st.selectbox(
        "Use bundled feature matrix",
        ["None", "Rice all genes", "Arabidopsis unknown20460"],
    )
    uploaded = st.file_uploader("Or upload processed .npz features", type=["npz"])


def bundled_path(choice: str) -> Path | None:
    if choice == "Rice all genes":
        return ROOT / "data" / "processed_features" / "rice_common6751_all_genes.npz"
    if choice == "Arabidopsis unknown20460":
        return (
            ROOT
            / "data"
            / "processed_features"
            / "arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz"
        )
    return None


input_path = None
if uploaded is not None:
    temp_path = ROOT / "webapp" / "_uploaded_features.npz"
    temp_path.write_bytes(uploaded.getbuffer())
    input_path = temp_path
elif bundled_path(example) is not None:
    input_path = bundled_path(example)

st.markdown(
    """
### Required input format

Upload a compressed NumPy `.npz` file containing:

- `X`: numeric feature matrix, shape `n_genes x 6751`;
- `gene_id`: gene identifiers;
- optional `transcript_id` or `sequence_id`;
- optional `feature_names` and `n_bio`.

This web app does not download GO, PPI, expression or PLM models. It expects
already processed features, matching the released `common6751` schema.
"""
)

if input_path is None:
    st.info("Choose a bundled matrix or upload a processed feature .npz file.")
    st.stop()

if st.button("Run prediction", type="primary"):
    with st.spinner("Loading features and deploying model..."):
        x, meta = load_npz_features(input_path)
        prob = predict(model, x)
        out = meta.copy()
        out["essential_probability"] = prob
        out["classification_threshold"] = float(threshold)
        out["predicted_label"] = (out["essential_probability"] >= float(threshold)).astype(int)
        out["predicted_class"] = out["predicted_label"].map({1: "essential", 0: "nonessential"})
        out["model_name"] = model
        out = out.sort_values(["essential_probability", "gene_id"], ascending=[False, True])

    c1, c2, c3 = st.columns(3)
    c1.metric("Genes", f"{len(out):,}")
    c2.metric("Predicted essential", f"{int(out['predicted_label'].sum()):,}")
    c3.metric("Predicted non-essential", f"{int((out['predicted_label'] == 0).sum()):,}")

    st.dataframe(out.head(200), use_container_width=True)
    st.download_button(
        "Download predictions",
        out.to_csv(sep="\t", index=False).encode("utf-8"),
        file_name=f"{model}_predictions.tsv",
        mime="text/tab-separated-values",
    )
