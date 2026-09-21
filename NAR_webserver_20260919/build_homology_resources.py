"""Build labelled-protein search databases from the original study resources."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/publication"))
import add_nearest_labelled_homologs_to_master_tables as original

OUT = ROOT / "webapp_data/nar_reference"
OUT.mkdir(parents=True, exist_ok=True)
manifest = {"diamond_executable": str(original.DIAMOND), "species": {}}
def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

for species, fasta, label_path in [("arabidopsis", original.ATH_FASTA, original.ATH_LABELS),
                                   ("rice", original.RICE_FASTA, original.RICE_LABELS)]:
    proteins = original.read_longest(fasta, species)
    labels = pd.read_csv(label_path, sep="\t")[["gene_id", "label"]]
    if labels.gene_id.duplicated().any() or not labels.label.isin([0,1]).all():
        raise ValueError("Invalid labelled reference")
    available = labels[labels.gene_id.isin(proteins)]
    subject = OUT / f"{species}.fasta"
    original.write_fasta(subject, {g: proteins[g] for g in available.gene_id})
    available.to_csv(OUT / f"{species}.labels.tsv", sep="\t", index=False)
    subprocess.run([str(original.DIAMOND), "makedb", "--in", str(subject), "-d", str(OUT / species)], check=True, capture_output=True)
    manifest["species"][species] = {"labelled_genes":len(labels),"searchable_proteins":len(available),
        "missing_proteins":sorted(set(labels.gene_id)-set(available.gene_id)),
        "source_fasta_sha256":sha(fasta),"source_labels_sha256":sha(label_path),
        "subject_sha256":sha(subject),"database_sha256":sha(OUT / f"{species}.dmnd")}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest["species"], indent=2))
