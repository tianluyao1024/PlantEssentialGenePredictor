from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DIMS = {"esm2": 2560, "protbert": 2048, "prott5": 2048}


def open_text(path: Path):
    return path.open("r", encoding="utf-8", errors="replace")


def first_gene_id(seq_id: str) -> str:
    match = re.search(r"(AT[1-5CM]G\d{5}|LOC_Os\d{2}g\d{5})", seq_id, re.I)
    if match:
        return match.group(1).upper() if match.group(1).upper().startswith("AT") else match.group(1)
    return seq_id.split(".")[0]


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records.setdefault(current, [])
            elif current:
                records[current].append(line)
    return {key: clean_protein("".join(chunks)) for key, chunks in records.items()}


def clean_protein(seq: str) -> str:
    seq = seq.upper().replace("*", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    return re.sub(r"[UZOB]", "X", seq)


def collapse_to_longest_by_gene(records: dict[str, str]) -> list[tuple[str, str]]:
    by_gene: dict[str, tuple[str, str]] = {}
    for seq_id, seq in records.items():
        gene = first_gene_id(seq_id)
        if gene not in by_gene or len(seq) > len(by_gene[gene][1]):
            by_gene[gene] = (gene, seq)
    return [(gene, seq) for gene, seq in sorted(by_gene.values()) if seq]


def chunks(seq: str, max_aa: int) -> list[str]:
    if len(seq) <= max_aa:
        return [seq]
    return [seq[start : start + max_aa] for start in range(0, len(seq), max_aa)]


def resolve_hf_snapshot(cache_root: Path, model_dir_name: str) -> Path:
    model_root = cache_root / "huggingface_hub" / model_dir_name
    snapshots = model_root / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"Missing Hugging Face snapshots directory: {snapshots}")
    candidates = [path for path in snapshots.iterdir() if path.is_dir() and (path / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No usable local snapshot with config.json under {snapshots}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def save_block(out_root: Path, model_name: str, ids: list[str], arr: np.ndarray) -> None:
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "all_ids.npy", np.array(ids, dtype=object))
    np.save(out_dir / "all_emb.npy", arr.astype(np.float32))


def extract_esm2(
    records: list[tuple[str, str]],
    weights_root: Path,
    batch_size: int,
    max_aa_per_chunk: int,
    device: torch.device,
) -> np.ndarray:
    import esm

    model_path = weights_root / "esm2" / "esm2_t33_650M_UR50D.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ESM2 checkpoint: {model_path}")
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    model.eval().to(device)
    converter = alphabet.get_batch_converter()

    per_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    work: list[tuple[str, str]] = []
    for gene, seq in records:
        for part in chunks(seq, max_aa_per_chunk):
            work.append((gene, part))

    with torch.no_grad():
        for start in range(0, len(work), batch_size):
            batch = work[start : start + batch_size]
            _, _, tokens = converter(batch)
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            for idx, (gene, part) in enumerate(batch):
                length = len(part)
                aa_reps = reps[idx, 1 : length + 1, :]
                pooled = torch.cat([aa_reps.mean(dim=0), aa_reps.max(dim=0).values], dim=0)
                per_gene[gene].append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.vstack([np.mean(per_gene[gene], axis=0) for gene, _ in records]).astype(np.float32)


def load_transformer(model_name: str, weights_root: Path, device: torch.device):
    from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

    if model_name == "protbert":
        snapshot = resolve_hf_snapshot(weights_root, "models--Rostlab--prot_bert")
        tokenizer = BertTokenizer.from_pretrained(str(snapshot), local_files_only=True, do_lower_case=False)
        model = BertModel.from_pretrained(str(snapshot), local_files_only=True)
    elif model_name == "prott5":
        snapshot = resolve_hf_snapshot(weights_root, "models--Rostlab--prot_t5_xl_uniref50")
        tokenizer = T5Tokenizer.from_pretrained(str(snapshot), local_files_only=True, legacy=True)
        model = T5EncoderModel.from_pretrained(str(snapshot), local_files_only=True)
    else:
        raise ValueError(model_name)
    model.eval().to(device)
    return tokenizer, model


def spaced(seq: str) -> str:
    return " ".join(list(seq))


def extract_transformer(
    model_name: str,
    records: list[tuple[str, str]],
    weights_root: Path,
    batch_size: int,
    max_aa_per_chunk: int,
    device: torch.device,
) -> np.ndarray:
    tokenizer, model = load_transformer(model_name, weights_root, device)
    per_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    work: list[tuple[str, str]] = []
    for gene, seq in records:
        for part in chunks(seq, max_aa_per_chunk):
            work.append((gene, part))

    with torch.no_grad():
        for start in range(0, len(work), batch_size):
            batch = work[start : start + batch_size]
            encoded = tokenizer(
                [spaced(seq) for _, seq in batch],
                add_special_tokens=True,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            special = encoded.pop("special_tokens_mask")
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            keep = (encoded["attention_mask"].bool() & ~special.bool()).unsqueeze(-1)
            lengths = keep.sum(dim=1).clamp(min=1)
            mean_vec = (hidden * keep).sum(dim=1) / lengths
            masked = hidden.masked_fill(~keep, torch.finfo(hidden.dtype).min)
            max_vec = masked.max(dim=1).values
            pooled = torch.cat([mean_vec, max_vec], dim=1)
            for idx, (gene, _) in enumerate(batch):
                per_gene[gene].append(pooled[idx].detach().cpu().numpy().astype(np.float32))
    return np.vstack([np.mean(per_gene[gene], axis=0) for gene, _ in records]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ESM2, ProtBERT and ProtT5 embeddings from protein FASTA.")
    parser.add_argument("--protein-fasta", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--weights-root", type=Path, default=ROOT.parent / "plm_model_weights")
    parser.add_argument("--models", nargs="+", choices=list(DIMS), default=list(DIMS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-aa-per-chunk", type=int, default=900)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))

    records = collapse_to_longest_by_gene(read_fasta(args.protein_fasta))
    if not records:
        raise ValueError(f"No protein records found in {args.protein_fasta}")
    ids = [gene for gene, _ in records]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protein_fasta": str(args.protein_fasta),
        "weights_root": str(args.weights_root),
        "genes": len(records),
        "models": args.models,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_aa_per_chunk": args.max_aa_per_chunk,
        "outputs": {},
    }

    for model_name in args.models:
        if model_name == "esm2":
            arr = extract_esm2(records, args.weights_root, args.batch_size, args.max_aa_per_chunk, device)
        else:
            arr = extract_transformer(model_name, records, args.weights_root, args.batch_size, args.max_aa_per_chunk, device)
        expected = DIMS[model_name]
        if arr.shape != (len(records), expected):
            raise ValueError(f"{model_name} produced {arr.shape}; expected {(len(records), expected)}")
        save_block(args.out_dir, model_name, ids, arr)
        manifest["outputs"][model_name] = {"shape": list(arr.shape), "dir": str(args.out_dir / model_name)}
        del arr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.out_dir / "plm_extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
