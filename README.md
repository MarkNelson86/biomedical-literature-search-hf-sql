# Biomedical Literature Search with Hugging Face + SQL

A compact portfolio project demonstrating an end-to-end data-science workflow around pretrained transformer models.

## What the project does

- stores biomedical literature metadata in **SQLite**
- queries data with **SQL**, including joins, CTEs, aggregation, and window functions
- generates dense embeddings with a Hugging Face **Sentence Transformer**
- performs **semantic search**
- evaluates retrieval with **Precision@K** and **Mean Reciprocal Rank**
- applies Hugging Face **zero-shot classification**
- evaluates classification with **accuracy**, **macro-F1**, a confusion matrix, confidence analysis, and failure-case review
- persists model outputs back to SQL

The notebook includes a deterministic synthetic corpus for inspection and an optional mode for retrieving current abstracts from **PubMed**.

## Why this is useful

The project is intentionally designed around **model evaluation**, not just model invocation. It shows how a pretrained/foundation-model component fits into a reproducible analytical workflow with database operations, metrics, and error analysis.

## Stack

- Python
- SQL / SQLite
- pandas / NumPy
- scikit-learn
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- Matplotlib
- PubMed / NCBI E-utilities (optional live-data mode)

## Models

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings and semantic retrieval
- `MoritzLaurer/DeBERTa-v3-base-mnli` for zero-shot classification

An optional notebook section shows how an instruction-tuned generative model such as `google/flan-t5-small` could be added for evidence-grounded synthesis.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open:

`biomedical_literature_search_hf_sql.ipynb`

The first run downloads model weights from Hugging Face.

## Live PubMed mode

Inside the notebook:

```python
USE_LIVE_PUBMED = True
PUBMED_PER_TOPIC = 15
```

The topic attached to each PubMed record is a **weak label based on the retrieval query**, not expert ground truth.

## Evaluation philosophy

The project reports multiple complementary metrics:

- Precision@K
- Mean Reciprocal Rank
- accuracy
- macro-F1
- confusion matrix
- confidence/error analysis

This is deliberate: model quality is usually a spectrum, not simply a pass/fail test.

## Suggested interview description

> I built an end-to-end biomedical literature retrieval and classification workflow using Hugging Face pretrained transformers and SQL. I used sentence embeddings for semantic search, evaluated retrieval with Precision@K and MRR, applied a zero-shot classifier, and evaluated it with accuracy, macro-F1, confusion matrices, and failure analysis. I also persisted inputs and model outputs in SQLite and used joins, CTEs, aggregation, and window functions to inspect results.

## Limitations / next steps

A production version would add:

- expert relevance annotations
- comparison of multiple embedding and classification models
- vector indexing for larger corpora
- calibrated confidence / abstention rules
- experiment tracking
- model and data drift monitoring
- privacy/security review for clinical data
- latency and compute-cost benchmarking
