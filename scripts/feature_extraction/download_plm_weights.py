from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ESM2_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
HF_MODELS = {"protbert": "Rostlab/prot_bert", "prott5": "Rostlab/prot_t5_xl_uniref50"}


def download_esm2(weights_root: Path) -> None:
    target = weights_root / "esm2" / "esm2_t33_650M_UR50D.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 100_000_000:
        print(f"ESM2 checkpoint already present: {target}")
        return
    partial = target.with_suffix(".pt.part")
    print(f"Downloading ESM2 checkpoint to {target}")
    with urllib.request.urlopen(ESM2_URL) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(target)


def download_huggingface(model: str, weights_root: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    repo_id = HF_MODELS[model]
    print(f"Downloading {repo_id} into {weights_root / 'huggingface_hub'}")
    snapshot_download(repo_id=repo_id, cache_dir=str(weights_root / "huggingface_hub"), token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PLM checkpoints for raw FASTA prediction; this requires more than 10 GB.")
    parser.add_argument("--weights-root", type=Path, default=ROOT.parent / "plm_model_weights")
    parser.add_argument("--models", nargs="+", choices=["esm2", *HF_MODELS], default=["esm2", *HF_MODELS])
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for restricted networks.")
    args = parser.parse_args()
    args.weights_root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        if model == "esm2":
            download_esm2(args.weights_root)
        else:
            download_huggingface(model, args.weights_root, args.hf_token)
    print(f"PLM weights are ready under: {args.weights_root}")


if __name__ == "__main__":
    main()
