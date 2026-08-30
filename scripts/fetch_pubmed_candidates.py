import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TOOL_NAME = "biomedical_literature_search_hf_sql"

NCBI_EMAIL = os.environ.get("NCBI_EMAIL")

if not NCBI_EMAIL:
    raise RuntimeError(
        "Please set the NCBI_EMAIL environment variable before running."
    )

N_PER_QUERY = 30

OUTPUT_PATH = Path("data/cache/pubmed_candidates.csv")


SEARCHES = {
    "white_matter_connectivity": (
    	'('
    	'"white matter"[tiab] '
    	'OR myelin*[tiab] '
    	'OR tractography[tiab] '
    	'OR "diffusion MRI"[tiab] '
    	'OR "diffusion magnetic resonance imaging"[tiab]'
    	') '
    	'AND '
    	'('
    	'"structural connectivity"[tiab] '
    	'OR connectome*[tiab] '
    	'OR "brain network"[tiab] '
    	'OR "brain networks"[tiab] '
    	'OR tractography[tiab] '
    	'OR "white matter microstructure"[tiab]'
    	') '
    	'AND hasabstract '
    	'AND english[la] '
    	'AND 2018:2026[pdat]'
    ),

    "eeg_memory": (
        '(EEG[tiab] OR electroencephalograph*[tiab] '
        'OR electrophysiolog*[tiab] '
        'OR "event-related potential"[tiab] '
        'OR "event-related potentials"[tiab]) '
        'AND '
        '(memory[tiab] OR "episodic memory"[tiab] '
        'OR "working memory"[tiab] OR recall[tiab]) '
        'AND hasabstract '
        'AND english[la] '
        'AND 2018:2026[pdat]'
    ),

    "neurodevelopment": (
        '(neurodevelopment*[tiab] OR "brain development"[tiab] '
        'OR "developmental trajectory"[tiab] '
        'OR "developmental trajectories"[tiab] '
        'OR "brain maturation"[tiab]) '
        'AND '
        '("brain network"[tiab] OR "brain networks"[tiab] '
        'OR connectivity[tiab] OR connectome*[tiab] '
        'OR "functional connectivity"[tiab] '
        'OR "structural connectivity"[tiab]) '
        'AND hasabstract '
        'AND english[la] '
        'AND 2018:2026[pdat]'
    ),

    "pediatric_complex_care": (
        '('
        '"children with medical complexity"[tiab] '
        'OR "child with medical complexity"[tiab] '
        'OR "pediatric complex care"[tiab] '
        'OR '
        '('
        '("complex chronic condition"[tiab] '
        'OR "complex chronic conditions"[tiab]) '
        'AND '
        '(child*[tiab] OR pediatric*[tiab])'
        ')'
        ') '
        'AND '
        '("care coordination"[tiab] OR outcomes[tiab] '
        'OR "health services"[tiab] OR "home care"[tiab] '
        'OR telehealth[tiab] OR hospitalization[tiab]) '
        'AND hasabstract '
        'AND english[la] '
        'AND 2018:2026[pdat]'
    ),
}

# ---------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------

def ncbi_params(extra):
    """Add standard NCBI identification parameters."""
    params = {
        "tool": TOOL_NAME,
        "email": NCBI_EMAIL,
    }
    params.update(extra)
    return params


def search_pubmed(query, retmax=N_PER_QUERY):
    """Run PubMed ESearch and return PMIDs."""
    url = f"{BASE_URL}/esearch.fcgi"

    params = ncbi_params({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    })

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


def fetch_pubmed_records(pmids):
    """Retrieve PubMed records in one batched EFetch request."""
    url = f"{BASE_URL}/efetch.fcgi"

    params = ncbi_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    return response.text


# ---------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------

def element_text(element):
    """Return all text contained inside an XML element."""
    if element is None:
        return ""

    return "".join(element.itertext()).strip()


def extract_year(article):
    """Try several PubMed date fields and return publication year."""
    pub_date = article.find(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
    )

    if pub_date is not None:
        year = pub_date.findtext("Year")

        if year:
            return int(year)

        medline_date = pub_date.findtext("MedlineDate")

        if medline_date:
            match = re.search(r"\b(19|20)\d{2}\b", medline_date)

            if match:
                return int(match.group())

    article_year = article.findtext(
        "./MedlineCitation/Article/ArticleDate/Year"
    )

    if article_year:
        return int(article_year)

    return None


def parse_pubmed_xml(xml_text):
    """Convert PubMed XML into one dictionary per article."""
    root = ET.fromstring(xml_text)

    records = []

    for article in root.findall(".//PubmedArticle"):

        pmid = article.findtext("./MedlineCitation/PMID")

        title_element = article.find(
            "./MedlineCitation/Article/ArticleTitle"
        )
        title = element_text(title_element)

        abstract_elements = article.findall(
            "./MedlineCitation/Article/Abstract/AbstractText"
        )

        abstract_parts = []

        for element in abstract_elements:
            text = element_text(element)

            if not text:
                continue

            label = element.attrib.get("Label")

            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        journal = article.findtext(
            "./MedlineCitation/Article/Journal/Title"
        )

        year = extract_year(article)

        doi = ""

        for article_id in article.findall(
            "./PubmedData/ArticleIdList/ArticleId"
        ):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text or ""
                break

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "doi": doi,
        })

    return records


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pmid_sources = {}

    # Search each topic separately.
    for topic, query in SEARCHES.items():

        print(f"\nSearching: {topic}")

        pmids = search_pubmed(query)

        print(f"Retrieved {len(pmids)} PMIDs")

        for pmid in pmids:
            pmid_sources.setdefault(pmid, []).append(topic)

        # Keep comfortably below NCBI's unauthenticated rate limit.
        time.sleep(0.4)

    unique_pmids = list(pmid_sources)

    print(f"\nUnique PMIDs after deduplication: {len(unique_pmids)}")

    # Fetch all records in manageable batches.
    all_records = []

    batch_size = 100

    for start in range(0, len(unique_pmids), batch_size):

        batch = unique_pmids[start:start + batch_size]

        print(
            f"Fetching records "
            f"{start + 1}-{start + len(batch)}..."
        )

        xml_text = fetch_pubmed_records(batch)

        records = parse_pubmed_xml(xml_text)

        all_records.extend(records)

        time.sleep(0.4)

    df = pd.DataFrame(all_records)

    # Record which candidate search(es) produced each PMID.
    df["candidate_source"] = df["pmid"].map(
        lambda x: "|".join(pmid_sources.get(x, []))
    )

    # Remove anything without usable text.
    df = df[
        df["title"].str.len().gt(0)
        & df["abstract"].str.len().gt(0)
    ].copy()

    # Columns for manual review.
    df["include"] = ""
    df["gold_topic"] = ""
    df["review_notes"] = ""

    df = df.sort_values(
        ["candidate_source", "year"],
        ascending=[True, False],
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} candidates to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
