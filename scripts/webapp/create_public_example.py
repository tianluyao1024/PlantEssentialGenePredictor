"""Create the stable, bundled Arabidopsis example result for the public UI.

The output contains only server-provided worked-example data. It is explicitly
marked as a public example and is excluded from user-job retention cleanup.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp"))

import nar_jobs as jobs

TOKEN = "PlantEGP_demo_Arabidopsis_00000000000000000"


def main() -> None:
    target = jobs.job_path(TOKEN)
    if target.exists():
        print("Public example already exists: ?job=" + TOKEN)
        return
    temporary = jobs.submit("demo", "arabidopsis", launch=False)
    jobs.run_job(temporary)
    source = jobs.job_path(temporary)
    state = jobs.read_state(temporary)
    if state.get("state") != "complete":
        raise RuntimeError("Bundled example did not complete; refusing to publish it.")
    state["public_example"] = True
    jobs.save_state(source, state)
    source.replace(target)
    print("Created public example: ?job=" + TOKEN)


if __name__ == "__main__":
    main()
