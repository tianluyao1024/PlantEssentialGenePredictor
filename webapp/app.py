from __future__ import annotations

import gzip
import hashlib
from io import StringIO
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
    "Expression matrix template": ROOT / "docs" / "input_templates" / "expression_matrix_template.tsv",
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
    "full_uploadable_without_cross_species_homologs": "Processed 6751 full profile",
}
RAW_TABLE_SCHEMAS = {
    "GO annotation": ["gene_id", "go_id"],
    "PPI edge list": ["gene_a", "gene_b", "score"],
    "Expression matrix": ["gene_id"],
    "Domain annotation": ["gene_id", "domain_id", "source"],
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


def parse_tsv_preview(data: bytes, required_columns: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    text = data.decode("utf-8", errors="replace")
    frame = pd.read_csv(StringIO(text), sep="\t")
    missing = [column for column in required_columns if column not in frame.columns]
    id_column = "gene_id" if "gene_id" in frame.columns else None
    if id_column is None and {"gene_a", "gene_b"}.issubset(frame.columns):
        ids = pd.concat([frame["gene_a"].astype(str), frame["gene_b"].astype(str)], ignore_index=True)
        unique_ids = int(ids.nunique())
    elif id_column is not None:
        unique_ids = int(frame[id_column].astype(str).nunique())
    else:
        unique_ids = 0
    report = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "required_columns_missing": ", ".join(missing) if missing else "none",
        "unique_gene_ids": unique_ids,
    }
    return frame.head(8), report


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
        "Raw data upload demo",
        "Input formats",
        "Released predictions",
        "Known labels",
        "Public species cache",
        "Server notes",
    ]
)

with tabs[0]:
    st.subheader("Model selection and prediction")
    st.write(
        "Choose one of the three released model families. Arabidopsis and rice use their single-species "
        "full models. The joint Arabidopsis-rice model can also be used with deployable feature-profile "
        "models when users provide only part of the optional annotations."
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        model = st.selectbox(
            "Primary model family",
            list(MODEL_LABELS),
            format_func=lambda x: MODEL_LABELS[x],
            key="full_model",
        )
        if model == "joint":
            joint_profile = st.selectbox(
                "Joint-model feature profile",
                list(PROFILE_LABELS),
                format_func=lambda value: PROFILE_LABELS[value],
                key="joint_profile",
            )
            st.caption(
                "The processed `.npz` prediction button below runs the released 6751-dimensional joint model. "
                "The profile selector documents which raw-data feature combination should be extracted for "
                "the corresponding deployable joint profile."
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
    st.subheader("Raw data upload and feature-extraction demo")
    st.write(
        "Users upload raw biological files; the server validates IDs and formats, then the backend extracts "
        "model features. The core files are `protein.fasta`, `cds.fasta` and `annotation.gff3`. "
        "GO, PPI, expression and domain files are optional enhancement inputs."
    )
    raw_model_family = st.selectbox(
        "Model to use after feature extraction",
        ["arabidopsis_single", "rice_single", "joint"],
        format_func=lambda value: MODEL_LABELS[value],
        key="raw_model_family",
    )
    if raw_model_family == "joint":
        st.selectbox(
            "Available joint-model feature combination",
            list(PROFILE_LABELS),
            format_func=lambda value: PROFILE_LABELS[value],
            key="raw_joint_profile",
        )
    st.markdown(
        """
**Required / recommended core files**

- `protein.fasta`: required for protein sequence features and PLM embeddings.
- `cds.fasta`: strongly recommended for CDS length, GC, GC3 and nucleotide composition.
- `annotation.gff3`: strongly recommended for gene span and transcript structure.

**Optional annotation files**

- `go_annotation.tsv`: `gene_id`, `go_id`.
- `ppi_edges.tsv`: `gene_a`, `gene_b`, `score`; tab-separated edge list only.
- `expression_matrix.tsv`: `gene_id` followed by sample columns; TPM/FPKM/normalized values recommended.
- `domain_annotation.tsv`: `gene_id`, `domain_id`, `source`.
"""
    )
    cds_file = st.file_uploader("CDS FASTA", type=["fa", "fasta", "fna", "txt"], key="cds_fasta")
    protein_file = st.file_uploader("Protein FASTA", type=["fa", "fasta", "faa", "txt"], key="protein_fasta")
    gff3_file = st.file_uploader("GFF3 annotation", type=["gff", "gff3", "txt"], key="gff3_upload")
    go_file = st.file_uploader("GO annotation TSV", type=["tsv", "txt"], key="go_upload")
    ppi_file = st.file_uploader("PPI edge-list TSV", type=["tsv", "txt"], key="ppi_upload")
    expr_file = st.file_uploader("Expression matrix TSV", type=["tsv", "txt"], key="expr_upload")
    domain_file = st.file_uploader("Domain annotation TSV", type=["tsv", "txt"], key="domain_upload")

    if all(file is None for file in [cds_file, protein_file, gff3_file, go_file, ppi_file, expr_file, domain_file]):
        st.info("Upload one or more raw input files to validate file structure.")
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
        if rows:
            st.markdown("### FASTA validation")
            stats_df = pd.DataFrame(rows)
            st.dataframe(stats_df, use_container_width=True)
        if gff3_file is not None:
            gff3_text = gff3_file.getvalue().decode("utf-8", errors="replace")
            gff3_rows = [line for line in gff3_text.splitlines() if line and not line.startswith("#")]
            st.markdown("### GFF3 validation")
            st.write(
                {
                    "non_comment_rows": len(gff3_rows),
                    "has_nine_tab_separated_columns_in_preview": all(
                        len(row.split("\t")) >= 9 for row in gff3_rows[:20]
                    )
                    if gff3_rows
                    else False,
                }
            )
            st.code("\n".join(gff3_rows[:3]), language="text")
        table_uploads = [
            ("GO annotation", go_file),
            ("PPI edge list", ppi_file),
            ("Expression matrix", expr_file),
            ("Domain annotation", domain_file),
        ]
        for label, uploaded in table_uploads:
            if uploaded is None:
                continue
            st.markdown(f"### {label} validation")
            try:
                preview, report = parse_tsv_preview(uploaded.getvalue(), RAW_TABLE_SCHEMAS[label])
                st.write(report)
                st.dataframe(preview, use_container_width=True)
            except Exception as exc:
                st.error(f"{label} could not be parsed as tab-separated text: {exc}")
        st.info(
            "This page currently validates raw inputs and demonstrates how feature extraction starts. "
            "The next backend step is to run the feature-extraction pipeline, concatenate features in the "
            "released schema, and call the selected model."
        )

    st.markdown("### Example extracted feature row")
    demo_feature_row = pd.DataFrame(
        [
            {
                "gene_id": "Gene001",
                "cds_length": 1230,
                "protein_length": 409,
                "gc3_content": 0.58,
                "go_embryo_development": 1,
                "string_network_connections_700": 8,
                "median_expression": 8.9,
                "domain_number": 2,
                "esm2_dims": 2560,
                "protbert_dims": 2048,
                "prott5_dims": 2048,
            }
        ]
    )
    st.dataframe(demo_feature_row, use_container_width=True)

with tabs[2]:
    st.subheader("Input formats and feature preparation")
    st.write(
        "The web server is designed for raw-data upload. Users do not need to manually build the model feature "
        "matrix. All files must use the same stable gene ID. For FASTA files, the first token after `>` is treated "
        "as the sequence ID. For TSV files, columns and separators must match the templates below."
    )

    with st.expander("Feature processing methods", expanded=True):
        st.markdown(
            """
**Core input files**

- `protein.fasta`: required. Used for protein length, amino-acid composition, physicochemical features and PLM embeddings.
- `cds.fasta`: strongly recommended. Used for CDS length, GC, GC skew, GC3 and nucleotide composition.
- `annotation.gff3`: strongly recommended. Used for gene span and transcript structure.
- If multiple transcripts are present, the backend selects one longest protein-coding transcript per gene.

**Sequence and protein features**

- `cds.fasta` and `protein.fasta` are matched by `gene_id`.
- If FASTA headers contain transcript IDs, the GFF3 `ID`/`Parent` relationships are used to map transcripts to genes.
- CDS length, protein length, GC, AT, GC skew, AT skew, GC3, nucleotide frequency, amino-acid frequency, amino-acid group frequency and protein physicochemical features are computed from sequence.

**GO features**

- Upload a tab-separated `go_annotation.tsv` with columns `gene_id` and `go_id`.
- GO terms are collapsed into the curated GO summary groups used by the manuscript model.
- Missing GO means unknown annotation, not true absence.

**PPI features**

- Upload only a tab-separated edge list with columns `gene_a`, `gene_b`, `score`.
- The network is treated as undirected; self-loops and duplicate edges are removed.
- `string_network_connections_400` is the number of partners with `score >= 400`.
- `string_network_connections_700` is the number of partners with `score >= 700`.
- If no confidence score is available, users may set all scores to `1000`.

**Expression features**

- Upload only a tab-separated expression matrix: first column `gene_id`, remaining columns are samples.
- TPM, FPKM or normalized counts are recommended; raw read counts are not recommended.
- The backend computes median expression, coefficient-of-variation-style expression variation and expression breadth.
- `expression_module_size` is not computed in the basic web workflow unless a future co-expression module workflow is enabled.

**Gene structure, within-species paralogs and domains**

- GFF3 is used for gene span and genomic location.
- Whole-proteome protein FASTA is required for within-species paralog and gene-family summaries.
- Domain features require Pfam/InterProScan-style domain annotations.
- The web workflow does not request an external homolog table.

**PLM embeddings**

- Protein sequences are embedded with ESM2, ProtBERT and ProtT5.
- Special tokens and padding are excluded from pooling.
- The released full model expects 2,560 ESM2 + 2,048 ProtBERT + 2,048 ProtT5 dimensions.
"""
        )

    with st.expander("Exact raw-data templates", expanded=True):
        st.markdown(
            """
All tabular files must be tab-separated (`.tsv`), not comma-separated.

**GO annotation**

```text
gene_id    go_id
Gene001    GO:0009790
Gene001    GO:0006950
Gene002    GO:0006412
```

**PPI edge list**

```text
gene_a    gene_b    score
Gene001   Gene002   850
Gene001   Gene003   420
Gene002   Gene004   760
```

**Expression matrix**

```text
gene_id    sample_1    sample_2    sample_3
Gene001    12.4        8.9         0.0
Gene002    0.0         0.2         4.1
```

**Domain annotation**

```text
gene_id    domain_id    source
Gene001    PF00069      Pfam
Gene001    PF07714      Pfam
Gene002    PF00001      Pfam
```

**Minimal GFF3**

```text
Chr1    source    gene    1000    3500    .    +    .    ID=Gene001
Chr1    source    mRNA    1000    3500    .    +    .    ID=Gene001.1;Parent=Gene001
Chr1    source    CDS     1100    1400    .    +    0    Parent=Gene001.1
```
"""
        )

    with st.expander("Recommended model choice for partial annotations", expanded=True):
        st.markdown(
            """
The website exposes three main model choices:

- Arabidopsis single-species model;
- rice single-species model;
- joint Arabidopsis-rice model.

For the joint model, users can choose among feature-profile models according to the raw files they provide:

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
