from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = ROOT / "webapp_data" / "jobs"


def cleanup_jobs(jobs_dir: Path, max_age_hours: float, dry_run: bool) -> list[Path]:
    if not jobs_dir.exists():
        return []
    cutoff = time.time() - max_age_hours * 3600
    removed: list[Path] = []
    for path in jobs_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime >= cutoff:
            continue
        removed.append(path)
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete old private Streamlit job files.")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    removed = cleanup_jobs(args.jobs_dir, args.max_age_hours, args.dry_run)
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(removed)} private job paths older than {args.max_age_hours:g} hours.")
    for path in removed[:50]:
        print(path)
    if len(removed) > 50:
        print(f"... {len(removed) - 50} more")


if __name__ == "__main__":
    main()
