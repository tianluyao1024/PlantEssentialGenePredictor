"""Retrieve non-phenotype UniProt context for core candidates.

The output is deliberately labelled as contextual evidence only. It must not be
used to claim independent phenotype validation until a curator verifies that
the exact information was not directly encoded in the feature set and is not
derived from a study-label source.
"""

from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tpc_candidate_resource"
SPECIES = {"arabidopsis": "3702", "rice": "39947"}
FIELDS = "accession,gene_names,protein_name,cc_subcellular_location,cc_function,cc_pathway,cc_interaction"


def query_uniprot(gene_id: str, organism_id: str, session: requests.Session) -> dict[str, str]:
    response = session.get(
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query": f"gene:{gene_id} AND organism_id:{organism_id}",
            "format": "tsv",
            "fields": FIELDS,
            "size": 5,
        },
        timeout=60,
    )
    response.raise_for_status()
    rows = pd.read_csv(StringIO(response.text), sep="\t", dtype=str).fillna("")
    if rows.empty:
        return {}
    record = rows.iloc[0].to_dict()
    return {
        "uniprot_accession": record.get("Entry", ""),
        "uniprot_entry_name": record.get("Entry Name", ""),
        "uniprot_gene_names": record.get("Gene Names", ""),
        "uniprot_protein_name": record.get("Protein names", ""),
        "uniprot_subcellular_location": record.get("Subcellular location [CC]", ""),
        "uniprot_function_context": record.get("Function [CC]", ""),
        "uniprot_pathway_context": record.get("Pathway", ""),
        "uniprot_interaction_context": record.get("Interacts with", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between UniProt API requests")
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "PlantEssentialGenePredictor/1.1 (candidate context retrieval)"})
    for species, taxonomy_id in SPECIES.items():
        cards_path = OUT / f"{species}_final_core10_evidence_cards.tsv"
        cards = pd.read_csv(cards_path, sep="\t", dtype=str, keep_default_na=False)
        contexts: list[dict[str, str]] = []
        for gene_id in cards["gene_id"].tolist():
            try:
                context = query_uniprot(gene_id, taxonomy_id, session)
                status = "retrieved_context_requires_independence_review" if context else "no_uniprot_match"
            except requests.RequestException as exc:
                context = {}
                status = f"retrieval_error:{type(exc).__name__}"
            contexts.append({"gene_id": gene_id, "uniprot_context_status": status, **context})
            time.sleep(args.delay)
        context_frame = pd.DataFrame(contexts)
        context_frame.to_csv(OUT / f"{species}_final_core10_uniprot_context.tsv", sep="\t", index=False)
        merged = cards.merge(context_frame, on="gene_id", how="left", validate="one_to_one")
        merged["evidence_status"] = merged["evidence_status"].where(
            merged["evidence_status"].ne("pending_manual_independent_curation"),
            "model_nomination_complete; external_context_requires_manual_independence_review",
        )
        merged.to_csv(cards_path, sep="\t", index=False)
        print(f"Wrote {species} UniProt context for {len(cards)} candidates")


if __name__ == "__main__":
    main()
