from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "prediction"))

from predict_from_processed_features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    load_npz_features,
    predict,
)


APP_DATA = ROOT / "webapp_data"
JOBS_DIR = APP_DATA / "jobs"
PUBLIC_DIR = APP_DATA / "public_predictions"
for directory in [JOBS_DIR, PUBLIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "arabidopsis_single": "Arabidopsis single-species full model",
    "rice_single": "Rice single-species full model",
    "joint": "Joint Arabidopsis-rice full model",
}
MODEL_SLUGS = {
    "arabidopsis_single": "arabidopsis_single_common6751",
    "rice_single": "rice_single_common6751",
    "joint": "joint_common6751",
}
BUNDLED_MATRICES = {
    "Rice all genes": ROOT / "data" / "processed_features" / "rice_common6751_all_genes.npz",
    "Arabidopsis unknown20460": (
        ROOT
        / "data"
        / "processed_features"
        / "arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz"
    ),
}
BUNDLED_PREDICTIONS = {
    "Arabidopsis unknown genes, single model": ROOT
    / "predictions"
    / "arabidopsis_unknown20460_single_model_predictions.tsv",
    "Arabidopsis unknown genes, joint model": ROOT
    / "predictions"
    / "arabidopsis_unknown20460_joint_model_predictions.tsv",
    "Rice all genes, single model": ROOT / "predictions" / "rice_unknown_all_single_model_predictions.tsv",
    "Rice all genes, joint model": ROOT / "predictions" / "rice_unknown_all_joint_model_predictions.tsv",
}
LABEL_TABLES = {
    "Arabidopsis strict2601 fixed split labels": ROOT
    / "data"
    / "labels"
    / "arabidopsis_strict2601_fixed_split_labels.tsv",
    "Arabidopsis training labels": ROOT / "data" / "labels" / "arabidopsis_strict2601_training_labels.tsv",
    "Arabidopsis validation labels": ROOT / "data" / "labels" / "arabidopsis_validation_labels.tsv",
    "Arabidopsis test labels": ROOT / "data" / "labels" / "arabidopsis_test_labels.tsv",
    "Rice strict399 + Tos17 N4 fixed split labels": ROOT
    / "data"
    / "labels"
    / "rice_strict399_Tos17N4_fixed_split_labels.tsv",
    "Rice raw strict399 + Tos17 N4 labels": ROOT / "data" / "labels" / "rice_raw_strict399_Tos17N4_labels.tsv",
}
TEMPLATE_FILES = {
    "CDS FASTA example": ROOT / "docs" / "input_templates" / "cds_example.fasta",
    "Protein FASTA example": ROOT / "docs" / "input_templates" / "protein_example.fasta",
    "GO annotation template": ROOT / "docs" / "input_templates" / "go_annotation_template.tsv",
    "PPI edge-list template": ROOT / "docs" / "input_templates" / "ppi_edges_template.tsv",
    "PPI degree template": ROOT / "docs" / "input_templates" / "ppi_degree_template.tsv",
    "Expression matrix template": ROOT / "docs" / "input_templates" / "expression_matrix_template.tsv",
    "Expression summary template": ROOT / "docs" / "input_templates" / "expression_summary_template.tsv",
    "Domain annotation template": ROOT / "docs" / "input_templates" / "domain_annotation_template.tsv",
    "Minimal GFF3 template": ROOT / "docs" / "input_templates" / "gff3_minimal_template.gff3",
}
PROFILE_MODEL_DIR = ROOT / "models" / "deployable_feature_profiles"
PROFILE_COMPARISON = PROFILE_MODEL_DIR / "profile_model_comparison.tsv"
PROFILE_LABELS = {
    "sequence_plm": "Sequence + PLM",
    "sequence_plm_go": "Sequence + PLM + GO",
    "sequence_plm_ppi": "Sequence + PLM + PPI",
    "sequence_plm_expression": "Sequence + PLM + expression",
    "sequence_plm_go_ppi": "Sequence + PLM + GO + PPI",
    "sequence_plm_go_expression": "Sequence + PLM + GO + expression",
    "sequence_plm_ppi_expression": "Sequence + PLM + PPI + expression",
    "sequence_plm_go_ppi_expression": "Sequence + PLM + GO + PPI + expression",
    "full_uploadable_without_cross_species_homologs": "Advanced full uploaded-feature profile",
}


@dataclass
class FastaStats:
    records: int
    duplicate_ids: int
    invalid_records: int
    min_length: int
    median_length: float
    max_length: int
    checksum: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "unnamed"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_uploaded_file(uploaded, path: Path) -> str:
    data = uploaded.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def parse_fasta_bytes(data: bytes, protein: bool) -> FastaStats:
    text = data.decode("utf-8", errors="replace")
    ids: list[str] = []
    lengths: list[int] = []
    invalid = 0
    current_id: str | None = None
    chunks: list[str] = []
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ*-.") if protein else set("ACGTUNacgtun-.")

    def flush() -> None:
        nonlocal invalid, current_id, chunks
        if current_id is None:
            return
        seq = "".join(chunks).replace(" ", "").replace("\t", "")
        ids.append(current_id)
        lengths.append(len(seq))
        if not seq or any(char not in allowed for char in seq):
            invalid += 1
        current_id = None
        chunks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_id = line[1:].split()[0]
            chunks = []
        elif current_id is None:
            invalid += 1
        else:
            chunks.append(line)
    flush()

    duplicated = len(ids) - len(set(ids))
    arr = np.array(lengths, dtype=float) if lengths else np.array([0.0])
    return FastaStats(
        records=len(ids),
        duplicate_ids=max(0, duplicated),
        invalid_records=invalid,
        min_length=int(arr.min()),
        median_length=float(np.median(arr)),
        max_length=int(arr.max()),
        checksum=sha256_bytes(data),
    )


def metadata_for_public_cache(
    species: str,
    assembly: str,
    annotation: str,
    model: str,
    input_checksum: str,
) -> dict[str, str]:
    return {
        "species": species,
        "assembly_version": assembly,
        "annotation_version": annotation,
        "model": model,
        "model_slug": MODEL_SLUGS[model],
        "input_sha256": input_checksum,
        "created_utc": now_iso(),
    }


def cache_dir_for(metadata: dict[str, str]) -> Path:
    species = safe_slug(metadata["species"])
    assembly = safe_slug(metadata["assembly_version"])
    annotation = safe_slug(metadata["annotation_version"])
    checksum = metadata["input_sha256"][:16]
    return PUBLIC_DIR / species / assembly / annotation / metadata["model_slug"] / checksum


def find_public_cache(metadata: dict[str, str]) -> Path | None:
    candidate = cache_dir_for(metadata) / "prediction_result.tsv.gz"
    return candidate if candidate.exists() else None


def save_public_cache(predictions: pd.DataFrame, metadata: dict[str, str]) -> Path:
    out_dir = cache_dir_for(metadata)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prediction_result.tsv.gz"
    with gzip.open(out_file, "wt", encoding="utf-8", newline="") as handle:
        predictions.to_csv(handle, sep="\t", index=False)
    summary = {
        **metadata,
        "genes": int(len(predictions)),
        "predicted_essential": int(predictions["predicted_label"].sum()),
        "predicted_nonessential": int((predictions["predicted_label"] == 0).sum()),
        "mean_probability": float(predictions["essential_probability"].mean()),
        "median_probability": float(predictions["essential_probability"].median()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_file


def iter_public_prediction_summaries() -> Iterable[dict[str, object]]:
    for summary_path in PUBLIC_DIR.glob("*/*/*/*/*/summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["relative_path"] = str(summary_path.parent.relative_to(PUBLIC_DIR))
            data["prediction_path"] = str(summary_path.parent / "prediction_result.tsv.gz")
            yield data
        except Exception:
            continue


def run_full_prediction(input_path: Path, model: str, threshold: float) -> pd.DataFrame:
    x, meta = load_npz_features(input_path)
    if x.ndim != 2 or x.shape[1] != 6751:
        raise ValueError(f"Expected an n_genes x 6751 matrix; observed shape {x.shape}.")
    probability = predict(model, x)
    out = meta.copy()
    out["essential_probability"] = probability
    out["classification_threshold"] = float(threshold)
    out["predicted_label"] = (out["essential_probability"] >= float(threshold)).astype(int)
    out["predicted_class"] = np.where(out["predicted_label"].eq(1), "essential", "nonessential")
    out["model_name"] = model
    return out.sort_values(["essential_probability", "gene_id"], ascending=[False, True])


def show_prediction_metrics(out: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genes", f"{len(out):,}")
    c2.metric("Predicted essential", f"{int(out['predicted_label'].sum()):,}")
    c3.metric("Predicted non-essential", f"{int((out['predicted_label'] == 0).sum()):,}")
    c4.metric("Median probability", f"{out['essential_probability'].median():.3f}")


st.set_page_config(page_title="Plant Essential Gene Predictor", layout="wide")
st.title("Plant Essential Gene Predictor")
st.caption(
    "A local/server Streamlit interface for plant essential-gene prioritization. "
    "The current released models use the 6,751-dimensional common Arabidopsis-rice feature space."
)

with st.sidebar:
    st.header("Server policy")
    st.write("Temporary uploads are stored under `webapp_data/jobs/` and can be removed with the cleanup script.")
    st.write("Public species-level results are saved only when explicit sharing consent is selected.")
    st.divider()
    st.header("Model status")
    st.success("Full 6751-feature model: available")
    st.success("Deployable profile models: available")

tabs = st.tabs(
    [
        "Full-model prediction",
        "Basic FASTA mode",
        "Input formats",
        "Released predictions",
        "Known labels",
        "Public species cache",
        "Server notes",
    ]
)

with tabs[0]:
    st.subheader("Full-model prediction from processed 6751-dimensional features")
    st.write(
        "Use this mode when the input `.npz` already matches the released `common6751` schema. "
        "This is the same feature space used by the manuscript models."
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        model = st.selectbox(
            "Model",
            list(MODEL_LABELS),
            format_func=lambda x: MODEL_LABELS[x],
            key="full_model",
        )
        threshold = st.number_input(
            "Classification threshold",
            value=float(DEFAULT_THRESHOLDS[model]),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="full_threshold",
        )
        bundled_choice = st.selectbox(
            "Use bundled feature matrix",
            ["None", *BUNDLED_MATRICES.keys()],
            key="bundled_matrix",
        )
        uploaded_npz = st.file_uploader("Or upload processed `.npz` features", type=["npz"], key="npz_upload")

    with col_right:
        species = st.text_input("Species name for cache metadata", value="custom_species")
        assembly = st.text_input("Assembly version", value="unknown_assembly")
        annotation = st.text_input("Annotation version", value="unknown_annotation")
        consent = st.checkbox(
            "Save final predictions to the public downloadable cache for this species/assembly/model.",
            value=False,
        )
        st.caption("Only final prediction tables and metadata are cached. Raw uploads are not saved publicly.")

    input_path: Path | None = None
    input_checksum: str | None = None
    job_dir: Path | None = None
    if uploaded_npz is not None:
        job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        job_dir = JOBS_DIR / job_id
        input_path = job_dir / "input_features.npz"
        input_checksum = write_uploaded_file(uploaded_npz, input_path)
    elif bundled_choice != "None":
        input_path = BUNDLED_MATRICES[bundled_choice]
        input_checksum = sha256_bytes(input_path.read_bytes())

    if input_path is None:
        st.info("Choose a bundled matrix or upload a processed `.npz` file.")
    else:
        cache_metadata = metadata_for_public_cache(species, assembly, annotation, model, input_checksum or "unknown")
        cached = find_public_cache(cache_metadata)
        if cached is not None:
            st.success("A matching public cached prediction already exists.")
            st.download_button(
                "Download cached prediction table",
                cached.read_bytes(),
                file_name="prediction_result.tsv.gz",
                mime="application/gzip",
            )

        if st.button("Run full-model prediction", type="primary"):
            try:
                with st.spinner("Loading features, applying trained preprocessors and predicting..."):
                    out = run_full_prediction(input_path, model, float(threshold))
                show_prediction_metrics(out)
                st.dataframe(out.head(300), use_container_width=True)
                tsv = out.to_csv(sep="\t", index=False).encode("utf-8")
                st.download_button(
                    "Download predictions",
                    tsv,
                    file_name=f"{MODEL_SLUGS[model]}_predictions.tsv",
                    mime="text/tab-separated-values",
                )
                if consent:
                    cached_path = save_public_cache(out, cache_metadata)
                    st.success(f"Saved public cached result: {cached_path}")
                if job_dir is not None:
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tabs[1]:
    st.subheader("Basic FASTA mode for sequence and PLM prediction")
    st.write(
        "This mode validates FASTA uploads for the deployable sequence + PLM model. "
        "Online PLM extraction can be enabled on the local server; large proteomes should be processed in batches."
    )
    cds_file = st.file_uploader("CDS FASTA", type=["fa", "fasta", "fna", "txt"], key="cds_fasta")
    protein_file = st.file_uploader("Protein FASTA", type=["fa", "fasta", "faa", "txt"], key="protein_fasta")

    if cds_file is None and protein_file is None:
        st.info("Upload CDS and/or protein FASTA to validate file structure.")
    else:
        rows = []
        if cds_file is not None:
            cds_data = cds_file.getvalue()
            cds_stats = parse_fasta_bytes(cds_data, protein=False)
            rows.append({"file": "CDS", **cds_stats.__dict__})
        if protein_file is not None:
            protein_data = protein_file.getvalue()
            protein_stats = parse_fasta_bytes(protein_data, protein=True)
            rows.append({"file": "Protein", **protein_stats.__dict__})
        stats_df = pd.DataFrame(rows)
        st.dataframe(stats_df, use_container_width=True)
        st.info(
            "FASTA validation is complete. Probability prediction requires extracting ESM2, ProtBERT and ProtT5 "
            "embeddings, then applying the released `sequence_plm` profile model."
        )

with tabs[2]:
    st.subheader("Input formats and feature preparation")
    st.write(
        "All uploaded files must use the same `gene_id`. For FASTA files, the first token after `>` is treated as "
        "the gene ID. For tables, the first column should be `gene_id` unless the template says otherwise."
    )

    with st.expander("Feature processing methods", expanded=True):
        st.markdown(
            """
**Sequence and protein features**

- `cds.fasta` and `protein.fasta` are matched by `gene_id`.
- If multiple transcripts are present, use the longest protein-coding transcript per gene.
- CDS length, protein length, GC, AT, GC skew, AT skew, GC3, nucleotide frequency, amino-acid frequency, amino-acid group frequency and protein physicochemical features are computed from sequence.

**GO features**

- Upload `go_annotation.tsv` with `gene_id` and `go_id`.
- GO terms are collapsed into the curated GO summary groups used by the manuscript model.
- Missing GO means unknown annotation, not true absence.

**PPI features**

- Upload an edge list with `gene_a`, `gene_b`, `score`, or a degree table with `string_network_connections_400` and `string_network_connections_700`.
- Edge lists are collapsed as undirected interactions; self-loops are ignored.

**Expression features**

- Upload a sample matrix or a summary table.
- The pipeline summarizes median expression, expression variation, expression breadth and optionally co-expression module size.

**Gene structure, paralogs, homologs and domains**

- GFF3 is used for gene span and genomic location.
- Whole-proteome FASTA is required for paralog summaries.
- Domain features require Pfam/InterProScan-style domain annotations.
- Cross-species homolog features require reference proteomes and a documented homology search.

**PLM embeddings**

- Protein sequences are embedded with ESM2, ProtBERT and ProtT5.
- Special tokens and padding are excluded from pooling.
- The released full model expects 2,560 ESM2 + 2,048 ProtBERT + 2,048 ProtT5 dimensions.
"""
        )

    with st.expander("Recommended model choice for partial annotations", expanded=True):
        st.markdown(
            """
The safest deployment design is to train separate models for common input profiles:

- sequence + PLM only;
- sequence + PLM + GO;
- sequence + PLM + PPI;
- sequence + PLM + expression;
- sequence + PLM + GO + PPI;
- sequence + PLM + GO + expression;
- sequence + PLM + PPI + expression;
- sequence + PLM + GO + PPI + expression;
- full 6751-feature model.

This avoids using zeros for missing GO/PPI/expression fields. The current release includes all profile models listed above.
"""
        )

    if PROFILE_COMPARISON.exists():
        st.markdown("### Released deployable profile models")
        profile_df = pd.read_csv(PROFILE_COMPARISON, sep="\t")
        profile_df["profile_label"] = profile_df["profile"].map(PROFILE_LABELS).fillna(profile_df["profile"])
        test_df = profile_df.loc[profile_df["split"].eq("test")].copy()
        display_cols = [
            "profile_label",
            "evaluation_species",
            "selected_method",
            "threshold",
            "feature_count",
            "bio_feature_count",
            "auc",
            "auprc",
            "sensitivity",
            "specificity",
            "f1",
        ]
        st.dataframe(
            test_df[display_cols].sort_values(["profile_label", "evaluation_species"]),
            use_container_width=True,
        )
        st.download_button(
            "Download profile-model comparison table",
            PROFILE_COMPARISON.read_bytes(),
            file_name=PROFILE_COMPARISON.name,
            mime="text/tab-separated-values",
        )

        profile_names = [name for name in PROFILE_LABELS if (PROFILE_MODEL_DIR / name).exists()]
        selected_profile = st.selectbox(
            "Download a deployable profile model",
            profile_names,
            format_func=lambda value: PROFILE_LABELS.get(value, value),
        )
        profile_dir = PROFILE_MODEL_DIR / selected_profile
        c1, c2, c3 = st.columns(3)
        with c1:
            model_path = profile_dir / "model.joblib"
            if model_path.exists():
                st.download_button(
                    "Model package",
                    model_path.read_bytes(),
                    file_name=f"{selected_profile}_model.joblib",
                    mime="application/octet-stream",
                )
        with c2:
            manifest_path = profile_dir / "manifest.json"
            if manifest_path.exists():
                st.download_button(
                    "Manifest",
                    manifest_path.read_bytes(),
                    file_name=f"{selected_profile}_manifest.json",
                    mime="application/json",
                )
        with c3:
            feature_path = profile_dir / "feature_names.tsv"
            if feature_path.exists():
                st.download_button(
                    "Feature names",
                    feature_path.read_bytes(),
                    file_name=f"{selected_profile}_feature_names.tsv",
                    mime="text/tab-separated-values",
                )

    guide = ROOT / "docs" / "input_feature_preparation.md"
    if guide.exists():
        st.download_button(
            "Download complete feature-preparation guide",
            guide.read_bytes(),
            file_name=guide.name,
            mime="text/markdown",
        )

    st.markdown("### Download input templates")
    template_cols = st.columns(3)
    for idx, (label, path) in enumerate(TEMPLATE_FILES.items()):
        if path.exists():
            with template_cols[idx % 3]:
                st.download_button(
                    label,
                    path.read_bytes(),
                    file_name=path.name,
                    mime="text/plain",
                    key=f"template_{idx}",
                )

with tabs[3]:
    st.subheader("Released manuscript prediction tables")
    choice = st.selectbox("Prediction table", list(BUNDLED_PREDICTIONS), key="released_pred")
    path = BUNDLED_PREDICTIONS[choice]
    if path.exists():
        df = pd.read_csv(path, sep="\t")
        show_prediction_metrics(df)
        st.dataframe(df.head(300), use_container_width=True)
        st.download_button(
            "Download full table",
            path.read_bytes(),
            file_name=path.name,
            mime="text/tab-separated-values",
        )
    else:
        st.error(f"Missing table: {path}")

with tabs[4]:
    st.subheader("Known experimental and modeling-label tables")
    st.write(
        "These tables contain the known Arabidopsis and rice labels used for fixed train/validation/test evaluation. "
        "They are separate from the unknown-gene prediction tables."
    )
    label_choice = st.selectbox("Known label table", list(LABEL_TABLES), key="known_labels")
    label_path = LABEL_TABLES[label_choice]
    if label_path.exists():
        label_df = pd.read_csv(label_path, sep="\t")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(label_df):,}")
        if "label" in label_df.columns:
            c2.metric("Essential", f"{int(pd.to_numeric(label_df['label'], errors='coerce').fillna(0).sum()):,}")
            c3.metric(
                "Non-essential",
                f"{int((pd.to_numeric(label_df['label'], errors='coerce').fillna(-1) == 0).sum()):,}",
            )
        st.dataframe(label_df.head(300), use_container_width=True)
        st.download_button(
            "Download selected known-label table",
            label_path.read_bytes(),
            file_name=label_path.name,
            mime="text/tab-separated-values",
        )
    else:
        st.error(f"Missing label table: {label_path}")

with tabs[5]:
    st.subheader("Public species-level cached predictions")
    summaries = list(iter_public_prediction_summaries())
    if not summaries:
        st.info("No public species-level predictions have been cached yet.")
    else:
        summary_df = pd.DataFrame(summaries)
        st.dataframe(summary_df, use_container_width=True)
        selected_idx = st.number_input(
            "Row number to download",
            min_value=0,
            max_value=max(0, len(summary_df) - 1),
            value=0,
            step=1,
        )
        selected_path = Path(summary_df.iloc[int(selected_idx)]["prediction_path"])
        st.download_button(
            "Download selected cached prediction",
            selected_path.read_bytes(),
            file_name=selected_path.name,
            mime="application/gzip",
        )

with tabs[6]:
    st.subheader("Deployment notes")
    st.markdown(
        """
**Recommended local-server workflow**

1. Run the app on a GPU or CPU server with enough disk space for uploaded jobs.
2. Process large proteomes in batches when the sequence-only model is added.
3. Delete private uploads and intermediate embeddings after each job or with the cleanup script.
4. Keep public cached predictions only when the user explicitly agrees to share final species-level results.

**Current release**

- Full-model prediction from processed 6,751-dimensional features is enabled.
- Deployable feature-profile models are available for sequence + PLM, GO, PPI and expression combinations.
- Bundled Arabidopsis and rice prediction tables can be browsed and downloaded.
- FASTA upload validation is enabled.
- Online PLM extraction and raw FASTA-to-probability jobs are designed for local-server batch execution.
"""
    )
