"""Package audited candidate-release assets for a Zenodo version update.

The base v1.0 archive already contains models and processed matrices. This
small supplement is meant to be added to a Zenodo *new version* together with
the retained base artifact, avoiding a second copy of pretrained weights or raw
source databases.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = Path(r"E:\PlantEssentialGenePredictor_Zenodo\PlantEssentialGenePredictor_audited_candidate_resource_v1_1")
ZIP = DEST.with_suffix(".zip")

RESOURCE_FILES = [
    ROOT / "predictions" / "publication_release" / "arabidopsis_all_feature_covered_genes_reclassified.tsv",
    ROOT / "predictions" / "publication_release" / "rice_all_feature_covered_genes_reclassified.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "study_label_and_phenotype_registry.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "frozen_submission_inputs.json",
    ROOT / "results" / "tpc_candidate_resource" / "candidate_release_summary.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "candidate_exclusion_audit.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "arabidopsis_provisional_true_unknown_candidates.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "rice_provisional_true_unknown_candidates.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "arabidopsis_final_core10_candidates.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "rice_final_core10_candidates.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "arabidopsis_final_core10_evidence_cards.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "rice_final_core10_evidence_cards.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "arabidopsis_candidate_ensemble_stability.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "rice_candidate_ensemble_stability.tsv",
    ROOT / "results" / "tpc_candidate_resource" / "ensemble_stability_manifest.json",
    ROOT / "results" / "tpc_candidate_resource" / "homology_candidate_screen_manifest.json",
    ROOT / "data" / "external_validation" / "independent_phenotype_cohort_template.tsv",
    ROOT / "results" / "manuscript_figures" / "figure6_audited_candidate_resource" / "Figure6_source_data.tsv",
]
DOC_FILES = [
    ROOT / "docs" / "independent_validation_protocol.md",
    ROOT / "docs" / "candidate_evidence_card_guidance.md",
    ROOT / "docs" / "candidate_release_notes_v1_1.md",
    ROOT / "docs" / "the_plant_cell_submission_checklist.md",
    ROOT / "scripts" / "publication" / "run_tpc_candidate_audit_pipeline.ps1",
    ROOT / "scripts" / "publication" / "build_tpc_candidate_registry.py",
    ROOT / "scripts" / "publication" / "compute_candidate_ensemble_stability.py",
    ROOT / "scripts" / "publication" / "assess_candidate_homology.py",
    ROOT / "scripts" / "publication" / "prepare_core_candidate_evidence_cards.py",
    ROOT / "scripts" / "publication" / "audit_candidate_exclusions.py",
    ROOT / "scripts" / "publication" / "evaluate_external_phenotype_cohort.py",
    ROOT / "scripts" / "manuscript" / "generate_audited_candidate_resource_figure.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination_for(path: Path) -> Path:
    return DEST / path.relative_to(ROOT)


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    copied: list[Path] = []
    for source in RESOURCE_FILES + DOC_FILES:
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination_for(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)

    metadata = {
        "title": "PlantEssentialGenePredictor: audited candidate-resource supplement",
        "version": "v1.1-candidate-audit",
        "base_archive_doi": "10.5281/zenodo.21387076",
        "purpose": "Audited true-unknown candidate tables and independent-validation protocol for a versioned Zenodo update.",
        "contains_raw_database_dumps": False,
        "contains_pretrained_plm_weights": False,
        "external_validation_status": "Protocol and template only; no independent phenotype cohort results are claimed.",
    }
    (DEST / "zenodo_candidate_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    copied.append(DEST / "zenodo_candidate_audit_metadata.json")
    manifest = DEST / "zenodo_candidate_audit_manifest.tsv"
    manifest.write_text(
        "path\tbytes\tsha256\n"
        + "\n".join(
            f"{path.relative_to(DEST).as_posix()}\t{path.stat().st_size}\t{sha256(path)}"
            for path in sorted(copied)
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DEST.parent))
    print(f"Wrote {DEST}")
    print(f"Wrote {ZIP}")
    print(f"ZIP SHA256 {sha256(ZIP)}")


if __name__ == "__main__":
    main()
