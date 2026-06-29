from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DESKTOP = Path.home() / "Desktop"
OUT = DESKTOP / "\u6c34\u7a3b" / "cross_species_ath_rice_common_features_models"
DIMS = {"esm2": 2560, "protbert": 2048, "prott5": 2048}


def safe_id(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def ids_for_species(species: str):
    table = OUT / ("rice_cross_species_common_features_all_genes.tsv" if species == "rice" else "arabidopsis_cross_species_common_features_all_genes.tsv")
    return pd.read_csv(table, sep="\t", usecols=["gene_id"])["gene_id"].astype(str).tolist()


def aggregate(model: str, species: str):
    root = OUT / "plm_embeddings" / model / species
    per = root / "per_gene"
    ids = ids_for_species(species)
    rows = []
    missing = []
    dim = DIMS[model]
    for gid in ids:
        path = per / f"{safe_id(gid)}.npy"
        if path.exists():
            rows.append(np.load(path).astype(np.float32))
        else:
            missing.append(gid)
            rows.append(np.zeros(dim, dtype=np.float32))
    emb = np.vstack(rows).astype(np.float32)
    np.save(root / "all_ids.npy", np.array(ids, dtype=object))
    np.save(root / "all_emb.npy", emb)
    if missing:
        (root / "missing_in_aggregate.txt").write_text("\n".join(missing), encoding="utf-8")
    print(f"{model} {species}: {emb.shape}, missing={len(missing)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(DIMS), required=True)
    parser.add_argument("--species", choices=["rice", "ath", "both"], default="both")
    args = parser.parse_args()
    for sp in (["rice", "ath"] if args.species == "both" else [args.species]):
        aggregate(args.model, sp)


if __name__ == "__main__":
    main()
