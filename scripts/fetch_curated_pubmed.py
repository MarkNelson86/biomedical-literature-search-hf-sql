from pathlib import Path

import pandas as pd

from fetch_pubmed_candidates import (
    fetch_pubmed_records,
    parse_pubmed_xml,
)


LABEL_PATH = Path("data/pubmed_labels.csv")
OUTPUT_PATH = Path("data/cache/pubmed_curated.csv")


def main():

    labels = pd.read_csv(
        LABEL_PATH,
        dtype={"pmid": str},
    )

    pmids = labels["pmid"].tolist()

    print(f"Curated PMIDs: {len(pmids)}")

    all_records = []

    batch_size = 100

    for start in range(0, len(pmids), batch_size):

        batch = pmids[start:start + batch_size]

        print(
            f"Fetching records "
            f"{start + 1}-{start + len(batch)}..."
        )

        xml_text = fetch_pubmed_records(batch)
        records = parse_pubmed_xml(xml_text)

        all_records.extend(records)

    corpus = pd.DataFrame(all_records)

    corpus["pmid"] = corpus["pmid"].astype(str)

    # Add the manually curated gold labels.
    corpus = corpus.merge(
        labels,
        on=["pmid", "doi"],
        how="inner",
        validate="one_to_one",
    )

    # Stable paper identifier for downstream use.
    corpus["paper_id"] = "PMID_" + corpus["pmid"]

    # Combine title + abstract for ML models.
    corpus["text"] = (
        corpus["title"].fillna("")
        + ". "
        + corpus["abstract"].fillna("")
    )

    # Keep a predictable column order.
    corpus = corpus[
        [
            "paper_id",
            "pmid",
            "doi",
            "title",
            "abstract",
            "text",
            "year",
            "journal",
            "gold_topic",
        ]
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nCorpus shape:")
    print(corpus.shape)

    print("\nGold-topic counts:")
    print(corpus["gold_topic"].value_counts())

    print(f"\nSaved corpus to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
