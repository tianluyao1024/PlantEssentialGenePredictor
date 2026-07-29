"""Create a machine-readable external-phenotype source-screening ledger.

The script searches Europe PMC for pre-defined Arabidopsis and rice LoF
phenotype queries, extracts stable locus identifiers from the title, abstract,
and non-reference full-text passages when open full text is available, and
joins each hit to the frozen candidate registry and released prediction tables.

It intentionally *does not* adjudicate a phenotype or add a gene to the locked
external cohort.  A curator must inspect the source and complete the cohort
schema after the automated exclusion audit.  This separation prevents a search
result or an annotation from being mistaken for independent phenotype evidence.

Run from the repository root:
    python scripts/publication/screen_external_phenotype_sources.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "predictions" / "publication_release"
REGISTRY = ROOT / "results" / "tpc_candidate_resource" / "study_label_and_phenotype_registry.tsv"
OUT = ROOT / "results" / "tpc_candidate_resource" / "external_source_screening"

ATH_RE = re.compile(r"\bAT[1-5MC]G\d{5}\b", re.I)
RICE_RE = re.compile(r"\b(?:LOC_)?OS\d{2}G\d{5,8}\b", re.I)
EXCLUDE_SECTIONS = {"ref", "reference", "references", "ack", "acknowledgments"}

QUERIES = {
    "arabidopsis": {
        "direct_lethal": (
            'TITLE_ABS:(Arabidopsis AND (CRISPR OR knockout OR "T-DNA" OR mutant) '
            'AND ("embryo lethal" OR "embryo lethality" OR "seed abortion" OR sterility))'
        ),
        "direct_viable": (
            'TITLE_ABS:(Arabidopsis AND (CRISPR OR knockout OR "T-DNA" OR mutant) '
            'AND (viable OR "no abnormal phenotype" OR "normal growth" OR "no developmental abnormality"))'
        ),
    },
    "rice": {
        "direct_lethal": (
            'TITLE_ABS:((rice OR "Oryza sativa") AND (CRISPR OR knockout OR "T-DNA" OR mutant) '
            'AND ("embryo lethal" OR "embryo lethality" OR "seed abortion" OR sterile OR lethality))'
        ),
        "direct_viable": (
            'TITLE_ABS:((rice OR "Oryza sativa") AND (CRISPR OR knockout OR "T-DNA" OR mutant) '
            'AND (viable OR "normal growth" OR "no obvious phenotype" OR "no developmental abnormality"))'
        ),
    },
}


def canonical(value: object) -> str:
    value = str(value).strip().upper()
    return value[4:] if value.startswith("LOC_") else value


def fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "PlantEssentialGenePredictor/1.2 (scientific screening)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def epmc_search(query: str, page_size: int) -> list[dict[str, object]]:
    params = urllib.parse.urlencode({"query": query, "format": "json", "pageSize": page_size, "resultType": "core"})
    payload = fetch_json(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}")
    return list(payload.get("resultList", {}).get("result", []))


def section_text_from_bioc(pmcid: str) -> str:
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    try:
        payload = fetch_json(url)
    except Exception:
        return ""
    documents = payload if isinstance(payload, list) else [payload]
    pieces: list[str] = []
    for document in documents:
        for item in document.get("documents", [document]):
            for passage in item.get("passages", []):
                infons = passage.get("infons", {}) or {}
                section = " ".join(str(value).lower() for key, value in infons.items() if "section" in key.lower())
                if any(token in section for token in EXCLUDE_SECTIONS):
                    continue
                text = passage.get("text", "")
                if text:
                    pieces.append(str(text))
    return "\n".join(pieces)


def ids_and_context(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    normalized = re.sub(r"\s+", " ", text)
    for match in pattern.finditer(normalized):
        gene_id = canonical(match.group(0))
        output.setdefault(gene_id, normalized[max(0, match.start() - 180): match.end() + 240])
    return output


def load_statuses() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    prediction_status: dict[tuple[str, str], str] = {}
    for species in ("arabidopsis", "rice"):
        table = pd.read_csv(PUBLISHED / f"{species}_all_feature_covered_genes_reclassified.tsv", sep="\t", dtype=str, keep_default_na=False)
        for row in table[["gene_id_key", "candidate_status"]].itertuples(index=False):
            prediction_status[(species, canonical(row.gene_id_key))] = row.candidate_status
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    registry_status = {
        (row.species.lower(), canonical(row.gene_id_key)): row.candidate_status
        for row in registry[["species", "gene_id_key", "candidate_status"]].itertuples(index=False)
    }
    return prediction_status, registry_status


def screen_status(species: str, gene_id: str, prediction: dict[tuple[str, str], str], registry: dict[tuple[str, str], str]) -> tuple[str, str, str]:
    key = (species, gene_id)
    candidate_status = prediction.get(key, "not_feature_covered_or_identifier_unresolved")
    registry_status = registry.get(key, "")
    if registry_status in {"known_label_used_in_study", "pseudo_label_used_in_study"}:
        return candidate_status, registry_status, "excluded_study_label_or_pseudo_label"
    if registry_status == "phenotype_recorded_but_excluded":
        return candidate_status, registry_status, "excluded_raw_phenotype_archive_record"
    if candidate_status != "true_unknown_candidate":
        return candidate_status, registry_status, "not_feature_covered_or_identifier_unresolved"
    return candidate_status, registry_status, "automated_eligible_for_manual_phenotype_curation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=100, help="Europe PMC results per query")
    parser.add_argument("--sleep", type=float, default=0.08, help="Polite delay between full-text requests")
    args = parser.parse_args()

    prediction, registry = load_statuses()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger: list[dict[str, object]] = []
    seen_articles: set[tuple[str, str, str]] = set()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for species, searches in QUERIES.items():
        pattern = ATH_RE if species == "arabidopsis" else RICE_RE
        for query_name, query in searches.items():
            for article in epmc_search(query, args.max_results):
                source = str(article.get("source", "MED"))
                article_id = str(article.get("id", ""))
                article_key = (species, source, article_id)
                if not article_id or article_key in seen_articles:
                    continue
                seen_articles.add(article_key)
                pmcid = str(article.get("pmcid", ""))
                doi = str(article.get("doi", ""))
                base = " ".join(str(article.get(field, "")) for field in ("title", "abstractText"))
                full_text = section_text_from_bioc(pmcid) if pmcid else ""
                if pmcid:
                    time.sleep(args.sleep)
                hits = ids_and_context(base + "\n" + full_text, pattern)
                source_url = f"https://europepmc.org/article/{source}/{article_id}"
                if pmcid:
                    source_url = f"https://europepmc.org/article/MED/{article.get('pmid', article_id)}"
                common = {
                    "species_target": species,
                    "query_name": query_name,
                    "screened_source": "Europe PMC",
                    "source_title": str(article.get("title", "")),
                    "author_string": str(article.get("authorString", "")),
                    "journal": str(article.get("journalTitle", "")),
                    "publication_date": str(article.get("firstPublicationDate", article.get("pubYear", ""))),
                    "doi": doi,
                    "pmid": str(article.get("pmid", "")),
                    "pmcid": pmcid,
                    "source_url": source_url,
                    "open_full_text_screened": "yes" if full_text else "no",
                    "screened_at_utc": now,
                }
                if not hits:
                    ledger.append({
                        **common,
                        "detected_gene_id": "",
                        "candidate_status": "identifier_not_detected",
                        "registry_status": "",
                        "automated_screening_outcome": "identifier_not_detected_requires_manual_title_abstract_review",
                        "matched_text_context": "",
                    })
                    continue
                for gene_id, context in sorted(hits.items()):
                    candidate_status, registry_status, outcome = screen_status(species, gene_id, prediction, registry)
                    ledger.append({
                        **common,
                        "detected_gene_id": gene_id,
                        "candidate_status": candidate_status,
                        "registry_status": registry_status,
                        "automated_screening_outcome": outcome,
                        "matched_text_context": context,
                    })

    columns = [
        "species_target", "query_name", "screened_source", "source_title", "author_string", "journal",
        "publication_date", "doi", "pmid", "pmcid", "source_url", "open_full_text_screened",
        "detected_gene_id", "candidate_status", "registry_status", "automated_screening_outcome",
        "matched_text_context", "screened_at_utc",
    ]
    table = pd.DataFrame(ledger, columns=columns).sort_values(
        ["species_target", "query_name", "publication_date", "source_title", "detected_gene_id"],
        kind="stable",
    )
    table.to_csv(OUT / "europe_pmc_source_screening_ledger.tsv", sep="\t", index=False)
    summary = {
        "screened_at_utc": now,
        "queries": QUERIES,
        "max_results_per_query": args.max_results,
        "records": int(len(table)),
        "articles": int(len(seen_articles)),
        "outcomes": table["automated_screening_outcome"].value_counts().to_dict(),
        "eligible_true_unknown_rows": int(table["automated_screening_outcome"].eq("automated_eligible_for_manual_phenotype_curation").sum()),
        "note": "Automated screening does not establish phenotype direction, assay quality, or cohort eligibility. Curator review remains mandatory.",
    }
    (OUT / "europe_pmc_source_screening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
