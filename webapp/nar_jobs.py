"""Private, process-backed jobs for the PlantEGP NAR preview.

Only server-owned example matrices are loaded with pickle support. User input
is text; uploads never enter the legacy object-NPZ loader.
"""
from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
import zipfile

# Apply only to PlantEGP and its child processes, never to other services.
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "webapp_data" / "jobs" / "nar"
EXAMPLES = ROOT / "supplementary_data/web_worked_examples"
PROCESSED = ROOT / "NAR_webserver_20260919/examples"
RAW = EXAMPLES / "Supplementary_Data_WebExample_RawInputs_100Genes"
FILES = {"protein": "protein.fasta", "cds": "cds.fasta", "gff3": "annotation.gff3",
         "go": "go_annotation.tsv", "ppi": "ppi_edges.tsv",
         "expression": "expression_matrix.tsv", "domain": "domain_annotation.tsv"}
RAW_INPUT_FILES = set(FILES.values())
RAW_INTERMEDIATE_FILES = {"embeddings", "features.features.tsv", "features.features.npz", "prediction.tsv",
                          "homology_query.fasta", "homology_hits.tsv", "worker.log"}
INPUT_RETENTION_SECONDS = 24 * 3600
JOB_RETENTION_SECONDS = 7 * 24 * 3600
sys.path.insert(0, str(ROOT / "scripts/prediction"))


def job_path(token: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        raise ValueError("Invalid private job link")
    return JOBS / token


def save_state(path: Path, state: dict) -> None:
    temporary = path / "state.tmp"
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path / "state.json")


def read_state(token: str) -> dict:
    return json.loads((job_path(token) / "state.json").read_text(encoding="utf-8"))


def cleanup_expired_jobs(now: float | None = None) -> dict[str, int]:
    """Remove expired user inputs first; remove terminal job directories after seven days.

    Never touches queued/running jobs or any directory outside the private NAR job root.
    The caller may safely run this repeatedly.
    """
    import shutil
    now = time.time() if now is None else now
    removed_inputs = removed_jobs = 0
    if not JOBS.exists():
        return {"inputs": 0, "jobs": 0}
    for path in JOBS.iterdir():
        if not path.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{43}", path.name):
            continue
        try:
            state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("state") not in {"complete", "failed"}:
            continue
        finished = state.get("completed_unix", state.get("submitted_unix", now))
        age = now - finished
        if age >= JOB_RETENTION_SECONDS:
            shutil.rmtree(path)
            removed_jobs += 1
            continue
        if age >= INPUT_RETENTION_SECONDS and not state.get("raw_inputs_deleted_unix"):
            for name in RAW_INPUT_FILES | RAW_INTERMEDIATE_FILES:
                target = path / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.is_file():
                    target.unlink()
            state["raw_inputs_deleted_unix"] = now
            save_state(path, state)
            removed_inputs += 1
    return {"inputs": removed_inputs, "jobs": removed_jobs}


def validate_protein(data: bytes) -> int:
    from Bio import SeqIO
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Protein FASTA must be nonempty and at most 10 MB.")
    text = data.decode("utf-8-sig")
    if not text.lstrip().startswith(">"):
        raise ValueError("Protein input must start with a FASTA header.")
    records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    if not 1 <= len(records) <= 100:
        raise ValueError("This preview accepts 1-100 protein sequences per job.")
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Use unique FASTA identifiers.")
    # Match the legacy extractor's gene-ID collapse, to avoid silently losing rows.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("raw_features", ROOT / "scripts/feature_extraction/raw_upload_to_profile_features.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    normalized = [module.first_gene_id(x) for x in ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Multiple sequence IDs resolve to one gene. Submit one protein per gene.")
    for record in records:
        seq = str(record.seq).upper()
        if not seq or len(seq) > 10000 or set(seq) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
            raise ValueError("Invalid protein sequence or sequence longer than 10,000 residues.")
    return len(records)


def _submit(kind: str, species: str, uploads: dict[str, bytes] | None = None,
           launch: bool = True) -> str:
    if kind not in {"demo", "raw"} or species not in {"arabidopsis", "rice"}:
        raise ValueError("Unsupported job type or species")
    uploads = uploads or {}
    validation = {}
    if set(uploads) - set(FILES):
        raise ValueError("Unsupported file")
    if "domain" in uploads:
        raise ValueError("Domain annotation is not used by a deployed model profile and is not accepted for prediction.")
    if kind == "raw":
        validate_protein(uploads.get("protein", b""))
        if sum(map(len, uploads.values())) > 50 * 1024 * 1024:
            raise ValueError("Combined input exceeds the 50 MB preview limit.")
        for data in uploads.values():
            data.decode("utf-8-sig")
        from nar_inputs import validate_annotations
        validation = validate_annotations(uploads)
    token = secrets.token_urlsafe(32)
    path = job_path(token)
    path.mkdir(parents=True, mode=0o700)
    for name, data in uploads.items():
        (path / FILES[name]).write_bytes(data)
    profile = "sequence_plm" + "".join("_" + n for n in ("go", "ppi", "expression") if n in uploads)
    input_hashes = {FILES[k]: hashlib.sha256(v).hexdigest() for k, v in uploads.items()}
    if kind == "demo":
        with (PROCESSED / f"{species}_100genes_common6751.npz").open("rb") as handle:
            input_hashes["bundled_feature_matrix"] = hashlib.file_digest(handle, "sha256").hexdigest()
    save_state(path, {"state": "queued", "kind": kind, "species": species,
                      "profile": profile, "submitted_unix": time.time(),
                      "input_sha256": input_hashes, "input_validation": validation})
    if launch:
        try:
            with (path / "worker.log").open("ab") as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), token],
                                 cwd=ROOT, stdout=log, stderr=log,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                 start_new_session=os.name != "nt")
        except Exception:
            state = read_state(token)
            state.update(state="failed", error="The server could not start this job.")
            save_state(path, state)
            raise
    return token


def submit(kind: str, species: str, uploads: dict[str, bytes] | None = None,
           launch: bool = True) -> str:
    import shutil
    with worker_slot('admission.lock'):
        cleanup_expired_jobs()
        active = 0
        for state_file in JOBS.glob('*/state.json'):
            try:
                active += json.loads(state_file.read_text(encoding='utf-8')).get('state') in {'queued', 'running'}
            except (OSError, ValueError):
                continue
        if active >= 3:
            raise ValueError('The server queue is full. Please try again after a current job finishes.')
        if shutil.disk_usage(JOBS).free < 10 * 1024**3:
            raise ValueError('The server has insufficient free storage. Please contact the maintainer.')
        return _submit(kind, species, uploads, launch)


@contextlib.contextmanager
def worker_slot(lock_name='worker.lock'):
    """OS-owned lock serializes heavy jobs and releases even on worker exit."""
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS / lock_name).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def reference(species: str) -> pd.DataFrame:
    name = "Arabidopsis" if species == "arabidopsis" else "Rice"
    return pd.read_csv(ROOT / f"predictions/publication_release/{name}_master_prediction_table.tsv.gz", sep="\t", low_memory=False)


def demo_prediction(species: str) -> pd.DataFrame:
    from predict_from_processed_features import predict
    matrix = np.load(PROCESSED / f"{species}_100genes_common6751.npz", allow_pickle=True)
    out = pd.DataFrame({"gene_id": matrix["gene_id"].astype(str)})
    out["single_species_score"] = predict(species + "_single", matrix["X"].astype(np.float32))
    out["joint_model_score"] = predict("joint", matrix["X"].astype(np.float32))
    ref = reference(species)
    out = out.merge(ref.rename(columns={c: "reference_" + c for c in ref if c != "gene_id"}),
                    on="gene_id", how="left", validate="one_to_one")
    out["score"] = out["single_species_score"]
    out["evidence_context"] = "Fresh aligned-matrix scores; reference_* fields from the master release, checked to 1e-5"
    if not np.allclose(out["single_species_score"], out["reference_single_species_score"], atol=1e-5, rtol=0) or not np.allclose(out["joint_model_score"], out["reference_joint_model_score"], atol=1e-5, rtol=0):
        raise ValueError("Example scores do not match the reference snapshot; resource version audit required")
    out["score_spread"] = (out["single_species_score"] - out["joint_model_score"]).abs()
    out["reference_release_percentile"] = 100 * (1 - (out["reference_single_species_rank"] - 1) / max(len(ref) - 1, 1))
    out["difference_from_reference_species_score"] = out["single_species_score"] - out["reference_single_species_score"]
    return out


def run_command(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True, timeout=6 * 3600)


def raw_prediction(path: Path, state: dict) -> pd.DataFrame:
    candidates = [ROOT.parent / "plm_model_weights", ROOT / "webapp_data/plm_model_weights",
                  ROOT.parent / "PlantEssentialGenePredictor_Portable/plm_model_weights"]
    weights = Path(os.environ["PLANT_EG_PLM_WEIGHTS"]) if os.getenv("PLANT_EG_PLM_WEIGHTS") else next((p for p in candidates if p.exists()), candidates[0])
    if not weights.exists():
        raise FileNotFoundError("Server PLM weights are unavailable")
    plm = path / "embeddings"
    run_command([str(ROOT / "scripts/feature_extraction/extract_plm_embeddings_from_fasta.py"),
                 "--protein-fasta", str(path / "protein.fasta"), "--out-dir", str(plm),
                 "--weights-root", str(weights), "--batch-size", "1", "--device", "auto"])
    profile = ROOT / "models/deployable_feature_profiles" / state["profile"]
    prefix = path / "features"
    cmd = [str(ROOT / "scripts/feature_extraction/raw_upload_to_profile_features.py"),
           "--input-dir", str(path), "--profile-dir", str(profile), "--out-prefix", str(prefix), "--plm-dir", str(plm)]
    obo_candidates = [ROOT.parent / "raw_data/go-basic.obo", ROOT / "webapp_data/go-basic.obo", RAW / "go-basic.obo"]
    obo = Path(os.environ["PLANT_EG_GO_OBO"]) if os.getenv("PLANT_EG_GO_OBO") else next((p for p in obo_candidates if p.exists()), obo_candidates[0])
    if "go" in state["profile"]:
        if not obo.exists():
            raise FileNotFoundError("Server GO ontology is unavailable")
        cmd += ["--go-obo", str(obo)]
    run_command(cmd)
    features = pd.read_csv(prefix.with_suffix(".features.tsv"), sep="\t")
    plm_columns = [c for c in features if c.startswith(("esm2_", "protbert_", "prott5_"))]
    if not plm_columns or features[plm_columns].isna().any().any():
        raise ValueError("Incomplete sequence embeddings; refusing an imputation-only sequence prediction")
    run_command([str(ROOT / "scripts/prediction/predict_from_profile_features.py"),
                 "--features", str(prefix.with_suffix(".features.npz")),
                 "--model", str(profile / "model.joblib"), "--out", str(path / "prediction.tsv")])
    out = pd.read_csv(path / "prediction.tsv", sep="\t").rename(columns={"essential_probability": "score"})
    out = out.drop(columns=["profile_model_path"], errors="ignore")
    out["feature_profile"] = state["profile"]
    if "predicted_class" in out:
        out["predicted_class"] = out["predicted_class"].replace({"essential": "severe_phenotype_prioritized", "nonessential": "below_prioritization_threshold"})
    coverage = pd.DataFrame({"gene_id": features.gene_id,
                             "feature_nonmissing_fraction": features.drop(columns="gene_id").notna().mean(axis=1)})
    biological = [c for c in features if c != "gene_id" and c not in plm_columns]
    coverage["biological_feature_nonmissing_fraction"] = features[biological].notna().mean(axis=1)
    out = out.merge(coverage, on="gene_id", validate="one_to_one")
    out["evidence_context"] = "User sequence; prior query labels not established; sequence-similarity evidence reported separately"
    # Do not attach reference evidence just because a user supplied a familiar ID.
    out["label_status"] = "not_audited_for_user_sequence"
    from nar_evidence import attach_homology
    out = attach_homology(out, path, state["species"])
    return out


def finish_outputs(path: Path, out: pd.DataFrame, state: dict) -> None:
    from nar_reports import score_distribution_svg
    out = out.sort_values(["score", "gene_id"], ascending=[False, True])
    out["query_rank"] = out.score.rank(method="min", ascending=False).astype(int)
    out["query_rank_percentile"] = 100 * (1 - (out["query_rank"] - 1) / max(len(out) - 1, 1))
    out["prioritization_band"] = pd.cut(out["query_rank_percentile"],
                                        bins=[-1, 90, 99, 100],
                                        labels=["standard priority", "high-priority", "top-priority"]).astype(str)
    out.to_csv(path / "results.tsv", sep="\t", index=False)
    out.to_csv(path / "results.csv", index=False)
    message = ("Scores prioritize experimental follow-up; they are not calibrated probabilities of lethality. "
               "Query ranks refer only to this submitted batch. Reference percentiles, when present, use the complete frozen species release. "
               "Model score spread is descriptive disagreement, not a confidence interval. "
               "Missing homology is not evidence of absence. Examples are not independent validation.")
    report = "<!doctype html><meta charset='utf-8'><title>PlantEGP report</title><h1>PlantEGP result report</h1>"
    report += "<p>" + html.escape(message) + "</p><p>Species: " + html.escape(state["species"]) + "</p>"
    report += '<details><summary>Input coverage and execution provenance</summary><pre>' + html.escape(json.dumps({'input_validation':state.get('input_validation',{}),'execution_manifest':state.get('execution_manifest',{})},indent=2)) + '</pre></details>'
    graphic = score_distribution_svg(out.score)
    (path / "score_distribution.svg").write_text(graphic, encoding="utf-8")
    report += '<div style="max-width:850px">' + graphic + '</div>'
    report += out.to_html(index=False, escape=True)
    (path / "report.html").write_text(report, encoding="utf-8")
    with zipfile.ZipFile(path / "report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ["results.tsv", "results.csv", "report.html", "score_distribution.svg"]:
            archive.write(path / name, name)
        public_state={k:v for k,v in state.items() if k not in {'recovery_of','worker_pid'}}
        archive.writestr("provenance.json", json.dumps(public_state, indent=2))


def run_job(token: str) -> None:
    import psutil
    process = psutil.Process()
    if os.name == 'nt':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        available = process.cpu_affinity()
        process.cpu_affinity(available[:min(2, len(available))])
    path = job_path(token)
    with worker_slot():
        state = read_state(token)
        if state["state"] != "queued":
            return
        state.update(state="running", started_unix=time.time(), worker_pid=os.getpid())
        save_state(path, state)
        try:
            from nar_provenance import execution_manifest
            state['execution_manifest'] = execution_manifest(state)
            out = demo_prediction(state["species"]) if state["kind"] == "demo" else raw_prediction(path, state)
            state.update(state="complete", completed_unix=time.time(), genes=len(out))
            finish_outputs(path, out, state)
        except Exception:
            import traceback
            traceback.print_exc()
            state.update(state="failed", error="Processing failed. Check the input formats or contact the server maintainer with your private job link.")
        save_state(path, state)


if __name__ == "__main__":
    run_job(sys.argv[1])
