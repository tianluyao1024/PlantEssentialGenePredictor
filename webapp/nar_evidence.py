"""Actual sequence alignment; reference labels never become query labels."""
import io
import json
import os
from pathlib import Path
import subprocess
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]


def attach_homology(out, job_dir, species):
    resources = Path(os.environ.get("PLANT_EG_REFERENCE", str(ROOT / "webapp_data/nar_reference")))
    manifest_path = resources / "manifest.json"
    out = out.copy()
    out["homology_search_status"] = "unavailable_on_this_server"
    if not manifest_path.exists():
        return out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diamond = os.environ.get("PLANT_EG_DIAMOND", manifest["diamond_executable"])
    # Match IDs through the same normalizer as feature extraction.
    import sys
    sys.path.insert(0, str(ROOT / "scripts/feature_extraction"))
    from raw_upload_to_profile_features import first_gene_id
    sequences = {first_gene_id(r.id): str(r.seq).upper().replace("*", "") for r in SeqIO.parse(job_dir / "protein.fasta", "fasta")}
    query = job_dir / "homology_query.fasta"
    # Internal query keys prevent FASTA metadata from becoming executable/format control.
    id_map = {f"q{i}":g for i,g in enumerate(sequences)}
    query.write_text("".join(f">{q}\n{sequences[g]}\n" for q,g in id_map.items()), encoding="ascii")
    hits_path = job_dir / "homology_hits.tsv"
    command=[diamond,"blastp","-d",str(resources/species),"-q",str(query),"-o",str(hits_path),
        "--outfmt","6","qseqid","sseqid","pident","qcovhsp","scovhsp","evalue","bitscore",
        "--max-target-seqs","10","--evalue","1e-5","--id","20","--query-cover","20","--subject-cover","20","--threads","2"]
    try:
        subprocess.run(command,check=True,capture_output=True,timeout=300)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        out['homology_search_status']='search_failed_prediction_preserved'
        return out
    out["homology_search_status"] = "no_hit_at_documented_thresholds"
    if hits_path.stat().st_size == 0:
        return out
    cols=["query_key","nearest_labelled_homologue","nearest_labelled_identity_percent",
          "nearest_labelled_query_coverage_percent","nearest_labelled_subject_coverage_percent",
          "nearest_labelled_evalue","nearest_labelled_bitscore"]
    hits=pd.read_csv(hits_path,sep="\t",names=cols)
    hits["gene_id"]=hits.query_key.map(id_map)
    hits=hits.sort_values(["nearest_labelled_bitscore","nearest_labelled_identity_percent","nearest_labelled_homologue"],ascending=[False,False,True]).drop_duplicates("gene_id")
    labels=pd.read_csv(resources/f"{species}.labels.tsv",sep="\t").set_index("gene_id").label
    hits["nearest_labelled_homologue_label"]=hits.nearest_labelled_homologue.map(labels)
    subject={r.id:str(r.seq).upper() for r in SeqIO.parse(resources/f"{species}.fasta","fasta")}
    hits["exact_reference_protein_sequence_match"]=[sequences[g]==subject[s] for g,s in zip(hits.gene_id,hits.nearest_labelled_homologue)]
    out=out.merge(hits.drop(columns="query_key"),on="gene_id",how="left",validate="one_to_one")
    out.loc[out.nearest_labelled_homologue.notna(),"homology_search_status"]="hit_found_reference_label_only"
    out["homology_interpretation"]="Protein similarity does not establish the query phenotype; reference labels follow the original severe-phenotype definition."
    return out
