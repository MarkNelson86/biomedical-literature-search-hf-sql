from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import sqlite3

from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


DB_PATH = Path("data/cache/pubmed_literature.db")

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------

class SearchState(TypedDict):
    natural_query: str
    fts_query: str
    candidate_k: int
    final_k: int
    candidates: list[dict]
    results: list[dict]


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedding_model


# ---------------------------------------------------------
# Node 1: lexical candidate retrieval
# ---------------------------------------------------------

def lexical_retrieve(
    state: SearchState,
) -> dict:

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. "
            "Run the PubMed corpus setup first."
        )

    sql = """
    SELECT
        p.paper_id,
        p.title,
        p.abstract,
        p.year,
        p.gold_topic,
        bm25(papers_fts) AS bm25_score
    FROM papers_fts
    JOIN papers AS p
        ON p.paper_id = papers_fts.paper_id
    WHERE papers_fts MATCH ?
    ORDER BY bm25_score
    LIMIT ?;
    """

    with sqlite3.connect(DB_PATH) as conn:
        candidates = pd.read_sql_query(
            sql,
            conn,
            params=(
                state["fts_query"],
                state["candidate_k"],
            ),
        )

    return {
        "candidates":
            candidates.to_dict(orient="records")
    }


# ---------------------------------------------------------
# Node 2: semantic reranking
# ---------------------------------------------------------

def semantic_rerank(
    state: SearchState,
) -> dict:

    candidates = pd.DataFrame(
        state["candidates"]
    )

    if candidates.empty:
        return {"results": []}

    model = get_embedding_model()

    documents = (
        candidates["title"].fillna("")
        + ". "
        + candidates["abstract"].fillna("")
    ).tolist()

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = model.encode(
        [state["natural_query"]],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Because both vectors are normalized,
    # dot product equals cosine similarity.
    similarities = (
        document_embeddings
        @ query_embedding
    )

    candidates["semantic_score"] = (
        similarities
    )

    candidates = (
        candidates
        .sort_values(
            "semantic_score",
            ascending=False,
        )
        .head(state["final_k"])
        .reset_index(drop=True)
    )

    candidates["hybrid_rank"] = (
        np.arange(len(candidates)) + 1
    )

    return {
        "results":
            candidates.to_dict(orient="records")
    }


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

def build_graph():

    workflow = StateGraph(SearchState)

    workflow.add_node(
        "lexical_retrieve",
        lexical_retrieve,
    )

    workflow.add_node(
        "semantic_rerank",
        semantic_rerank,
    )

    workflow.add_edge(
        START,
        "lexical_retrieve",
    )

    workflow.add_edge(
        "lexical_retrieve",
        "semantic_rerank",
    )

    workflow.add_edge(
        "semantic_rerank",
        END,
    )

    return workflow.compile()


hybrid_search_graph = build_graph()


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

DEMO_QUERIES = {
    "white_matter_connectivity": {
        "natural_query":
            "white matter structural connectivity "
            "and brain networks",

        "fts_query":
            '"white matter" AND '
            '(connectivity OR connectome '
            'OR tractography)',
    },

    "eeg_memory": {
        "natural_query":
            "EEG signals and electrophysiological "
            "mechanisms of memory",

        "fts_query":
            '(EEG OR electroencephalography) '
            'AND memory',
    },

    "neurodevelopment": {
        "natural_query":
            "development and maturation "
            "of brain connectivity",

        "fts_query":
            '(development OR maturation) '
            'AND '
            '(connectivity OR connectome OR network)',
    },

    "pediatric_complex_care": {
        "natural_query":
            "care and outcomes for children "
            "with medical complexity",

        "fts_query":
            '"medical complexity" '
            'OR "complex chronic conditions"',
    },
}


def precision_at_k(
    retrieved_topics,
    expected_topic,
    k=5,
):
    retrieved_topics = list(
        retrieved_topics[:k]
    )

    relevant = sum(
        topic == expected_topic
        for topic in retrieved_topics
    )

    return relevant / k


def reciprocal_rank(
    retrieved_topics,
    expected_topic,
):
    for rank, topic in enumerate(
        retrieved_topics,
        start=1,
    ):
        if topic == expected_topic:
            return 1.0 / rank

    return 0.0


def evaluate_hybrid(
    candidate_k=15,
    final_k=5,
):
    rows = []

    for expected_topic, query in DEMO_QUERIES.items():

        output = hybrid_search_graph.invoke({
            "natural_query":
                query["natural_query"],

            "fts_query":
                query["fts_query"],

            "candidate_k":
                candidate_k,

            "final_k":
                final_k,

            "candidates": [],
            "results": [],
        })

        results = pd.DataFrame(
            output["results"]
        )

        topics = (
            results["gold_topic"]
            .tolist()
        )

        rows.append({
            "expected_topic":
                expected_topic,

            "precision@5":
                precision_at_k(
                    topics,
                    expected_topic,
                    k=5,
                ),

            "reciprocal_rank":
                reciprocal_rank(
                    topics,
                    expected_topic,
                ),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------

if __name__ == "__main__":

    evaluation = evaluate_hybrid(
        candidate_k=15,
        final_k=5,
    )

    print("\nHybrid retrieval evaluation:\n")

    print(
        evaluation.to_string(
            index=False
        )
    )

    print(
        "\nMean Precision@5:",
        round(
            evaluation["precision@5"].mean(),
            3,
        ),
    )

    print(
        "Mean Reciprocal Rank:",
        round(
            evaluation[
                "reciprocal_rank"
            ].mean(),
            3,
        ),
    )
