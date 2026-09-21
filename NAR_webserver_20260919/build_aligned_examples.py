"""Create NEW examples aligned to current master-table matrices; preserve old packages."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "examples"
OLD = ROOT / "supplementary_data/web_worked_examples/Supplementary_Data_WebExample_Processed6751_100Genes"
sys.path.insert(0, str(ROOT / "scripts/prediction"))
from predict_from_processed_features import predict

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

OUT.mkdir(exist_ok=True)
audit = []
for species, matrix_name, master_name in [
    ("arabidopsis", "arabidopsis_all27561_common6751.npz", "Arabidopsis"),
    ("rice", "rice_common6751_all_genes.npz", "Rice")]:
    old = np.load(OLD / f"{species}_100genes_common6751.npz", allow_pickle=True)
    source = ROOT / "data/processed_features" / matrix_name
    whole = np.load(source, allow_pickle=True)
    old_genes = old["gene_id"].astype(str)
    master_all = pd.read_csv(ROOT / f"predictions/publication_release/{master_name}_master_prediction_table.tsv.gz", sep="\t", low_memory=False).set_index("gene_id")
    old_statuses = master_all.loc[old_genes].label_status.value_counts().to_dict()
    eligible = set(master_all.index[master_all.label_status.eq("true_unknown_candidate")]) & set(whole["gene_id"].astype(str))
    retained = [g for g in old_genes if g in eligible]
    genes = np.asarray((retained + sorted(eligible - set(retained)))[:100])
    lookup = {str(g): i for i, g in enumerate(whole["gene_id"])}
    x = whole["X"][[lookup[g] for g in genes]].astype(np.float32)
    assert np.array_equal(old["feature_names"].astype(str), whole["feature_names"].astype(str))
    old_aligned_x = whole["X"][[lookup[g] for g in old_genes]].astype(np.float32)
    changed = ~np.isclose(old["X"], old_aligned_x, equal_nan=True)
    master = master_all.loc[genes]
    assert master.label_status.eq("true_unknown_candidate").all()
    target = OUT / f"{species}_100genes_common6751.npz"
    np.savez_compressed(target, X=x, gene_id=genes, feature_names=whole["feature_names"].astype(str))
    observed = pd.DataFrame({"gene_id": genes})
    differences = {}
    for model, column in [(species + "_single", "single_species_score"), ("joint", "joint_model_score")]:
        observed[column] = predict(model, x)
        differences[column] = float(np.abs(observed[column].to_numpy() - master[column].to_numpy()).max())
        assert differences[column] < 1e-5, differences
    observed.to_csv(OUT / f"{species}_expected_predictions.tsv", sep="\t", index=False)
    audit.append({"species": species, "genes": len(genes), "source_matrix": str(source.relative_to(ROOT)),
                  "source_sha256": sha(source), "example_sha256": sha(target),
                  "old_example_current_label_status_counts": old_statuses,
                  "retained_old_example_genes": len(retained),
                  "old_vs_current_different_cells": int(changed.sum()),
                  "old_vs_current_different_rows": int(changed.any(axis=1).sum()),
                  "maximum_prediction_difference_from_master": differences})
(OUT / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
