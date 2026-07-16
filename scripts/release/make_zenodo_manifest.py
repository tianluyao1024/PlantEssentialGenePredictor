from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Zenodo file manifest and SHA256 checksums.")
    parser.add_argument("--root", required=True, type=Path, help="Zenodo artifact directory")
    parser.add_argument("--manifest", default="zenodo_file_manifest.tsv", help="TSV manifest output name")
    parser.add_argument("--checksums", default="zenodo_sha256sums.txt", help="Checksum output name")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / args.manifest
    checksum_path = root / args.checksums

    rows = ["relative_path\tsize_bytes\tsha256"]
    sums = []
    for path in iter_files(root):
        if path in {manifest_path, checksum_path}:
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append(f"{rel}\t{size}\t{digest}")
        sums.append(f"{digest}  {rel}")

    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    checksum_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {checksum_path}")


if __name__ == "__main__":
    main()
