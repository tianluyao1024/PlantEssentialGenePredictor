"""Run the PlantEGP NAR private-job retention policy once.

This script is designed for a scheduled task. It only acts on terminal,
token-validated directories under webapp_data/jobs/nar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp"))

from nar_jobs import cleanup_expired_jobs


if __name__ == "__main__":
    print(json.dumps(cleanup_expired_jobs(), sort_keys=True))
