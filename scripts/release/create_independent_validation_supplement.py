"""Package lightweight independent-validation assets for a Zenodo version update.

The package supplements, but does not duplicate, the existing large model and
processed-feature archive at DOI 10.5281/zenodo.21387076.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(r"E:\PlantEssentialGenePredictor_Zenodo")
PACKAGE = DESTINATION / "PlantEssentialGenePredictor_independent_validation_v1_2_3.zip"
RELEASE_VERSION = "v1.2.3-independent-validation"

FILES = [
    "README.md",
    "docs/independent_validation_protocol.md",
    "docs/candidate_evidence_card_guidance.md",
    "docs/external_validation_release_notes.md",
    "docs/the_plant_cell_submission_checklist.md",
    "data/external_validation/independent_phenotype_cohort_template.tsv",
    "data/external_validation/curated_external_phenotype_records.tsv",
    "data/external_validation/candidate_independent_evidence.tsv",
    "data/external_validation/release/arabidopsis_final_core10_evidence_cards_with_independent_evidence.tsv",
    "data/external_validation/release/arabidopsis_final_core10_evidence_card_summary.tsv",
    "data/external_validation/release/rice_final_core10_evidence_cards_with_independent_evidence.tsv",
    "data/external_validation/release/rice_final_core10_evidence_card_summary.tsv",
    "data/external_validation/release/curated_external_phenotype_records_audited.tsv",
    "data/external_validation/release/curated_external_source_screening_ledger.tsv",
    "data/external_validation/release/europe_pmc_source_screening_ledger.tsv",
    "data/external_validation/release/prelocked_external_phenotype_cohort.tsv",
    "data/external_validation/release/locked_external_cohort_scored.tsv",
    "data/external_validation/release/locked_external_cohort_metrics.json",
    "data/external_validation/release/external_validation_release_summary.json",
    "data/external_validation/release/Figure7_source_data.tsv",
    "scripts/publication/screen_external_phenotype_sources.py",
    "scripts/publication/build_external_validation_release.py",
    "scripts/publication/evaluate_external_phenotype_cohort.py",
    "scripts/publication/run_external_validation_release_pipeline.ps1",
    "scripts/manuscript/generate_external_validation_figure.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    """Return immutable repository provenance when Git is available."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    missing = [item for item in FILES if not (ROOT / item).exists()]
    if missing:
        raise FileNotFoundError("Missing release assets: " + ", ".join(missing))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in FILES:
        path = ROOT / name
        manifest.append({"path": name.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    metadata = {
        "title": "PlantEssentialGenePredictor independent-evidence and external-validation supplement",
        "version": RELEASE_VERSION,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_describe": git_value("describe", "--always", "--tags", "--dirty"),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_zenodo_doi": "10.5281/zenodo.21387076",
        "description": (
            "Lightweight supplement containing source-screening ledgers, curated direct loss-of-function records, "
            "zero-overlap audits, prelocked external-cohort status, candidate evidence cards, figure source data and "
            "reproduction scripts. The current external cohort is underpowered by the preregistered rule, so no external "
            "AUC/AUPRC is included. Large model binaries and processed feature matrices remain in the base archive."
        ),
        "license": "CC-BY-4.0",
        "keywords": ["plant essential genes", "Arabidopsis", "rice", "external validation", "machine learning"],
        "files": manifest,
    }
    readme = (
        "# PlantEssentialGenePredictor v1.2.3 independent-evidence supplement\n\n"
        "This package is an incremental supplement to Zenodo DOI 10.5281/zenodo.21387076. "
        "It does not include raw phenotype databases, protein-language-model weights, processed matrices or model binaries.\n\n"
        "The pre-registered external-cohort gate is intentionally enforced: the included Arabidopsis cohort has 16 records "
        "(9 essential and 7 viable/non-essential) and the rice cohort has no curator-locked record. Therefore no external "
        "AUC, AUPRC or threshold metric is reported. See docs/external_validation_release_notes.md.\n"
    )
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in FILES:
            archive.write(ROOT / item, arcname=item.replace("\\", "/"))
        archive.writestr("release_metadata.json", json.dumps(metadata, indent=2))
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("README_v1_2.md", readme)
    print(json.dumps({"package": str(PACKAGE), "bytes": PACKAGE.stat().st_size, "sha256": sha256(PACKAGE), "files": len(FILES)}, indent=2))


if __name__ == "__main__":
    main()
