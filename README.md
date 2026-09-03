# Biomedical Literature Search with Hugging Face + SQL

An end-to-end biomedical literature retrieval and classification project that moves from a controlled prototype to a curated PubMed benchmark and a reusable hybrid retrieval workflow.

The core idea is simple: **evaluate model complexity against a credible baseline rather than assuming a transformer is automatically better.**

## Project summary

I built a reproducible workflow for biomedical literature retrieval and topic classification using SQL and pretrained Hugging Face models. The project includes PubMed acquisition, manual corpus curation, SQLite data modeling, lexical retrieval, semantic retrieval, zero-shot classification, quantitative evaluation, paper-level error analysis, and a LangGraph-orchestrated hybrid retrieval workflow.

The real-data benchmark contains **80 manually reviewed PubMed papers** across four topics:

- `white_matter_connectivity`
- `eeg_memory`
- `neurodevelopment`
- `pediatric_complex_care`

Main result: on this small, terminology-rich benchmark, the conventional lexical baseline slightly outperformed MiniLM semantic retrieval. A subsequent LangGraph hybrid workflow — BM25 candidate generation followed by MiniLM semantic reranking — matched full-corpus MiniLM performance rather than improving on the lexical baseline. Together, these results demonstrate model and architecture selection based on evidence rather than novelty.

## Key results

<p align="center">
  <img src="assets/retrieval_comparison.png"
       alt="Retrieval benchmark comparing SQLite FTS5/BM25 and MiniLM semantic search"
       width="560">
</p>

**Zero-shot classification:** DeBERTa achieved **87.5% accuracy** and **macro-F1 = 0.875** on the 80-paper PubMed corpus without task-specific fine-tuning.

The lexical baseline slightly outperformed MiniLM on the four retrieval questions, although MiniLM's sole cross-topic result was still scientifically relevant to the query, illustrating an important limitation of strict single-topic relevance labels.

For zero-shot classification, correct predictions had higher mean confidence (**0.614**) than errors (**0.368**), with most mistakes occurring at interpretable boundaries between overlapping biomedical topics.

**Hybrid retrieval (LangGraph: BM25 → MiniLM):** the two-node LangGraph workflow achieved **mean Precision@5 = 0.95** and **MRR = 1.00**, matching full-corpus MiniLM retrieval. At this 80-paper scale the hybrid does not improve accuracy; its value is architectural, demonstrating how the evaluated components can be composed into a reusable multi-stage search pipeline.

## Why this project matters

Biomedical literature search is a useful test case for a broader engineering problem: **how should simple retrieval methods and pretrained language models be combined in document-heavy workflows?**

A transformer can provide semantic flexibility, but it also adds complexity, compute requirements, dependencies, and latency. This project therefore treats a strong lexical method as a baseline rather than a straw man.

The results suggest a practical design principle:

> Start with the simplest method that satisfies the task, then add semantic modeling or workflow complexity where it provides measurable value.

The LangGraph extension applies that principle directly. Rather than replacing the strong lexical baseline, it uses FTS5/BM25 as a fast candidate-generation stage and MiniLM as a semantic reranker. On this small corpus the hybrid architecture does not improve benchmark accuracy, but on larger corpora the same pattern could reduce the number of documents requiring semantic processing while preserving flexible ranking.

That principle extends beyond PubMed and applies to many search, triage, and knowledge-management systems.

## Potential applications

Although the benchmark uses biomedical literature, the same architecture can be adapted to other document-heavy workflows, including:

- scientific literature surveillance
- internal knowledge-base search
- evidence-review and document-triage pipelines
- regulatory or policy document retrieval
- patent and technical-document search
- research-monitoring systems
- domain-specific document classification

The project also includes an experimental **hybrid retrieval pipeline** implemented with LangGraph: fast lexical search generates candidate documents, then MiniLM semantically reranks that reduced set. The same pattern can be reused wherever large document collections benefit from a cheap first-pass filter followed by more flexible semantic processing.

## Workflow

The project develops in five stages:

1. **Deterministic prototype**  
   A small synthetic biomedical corpus validates the SQL, embedding, retrieval, classification, and evaluation code before real data are introduced.

2. **Curated PubMed retrieval benchmark**  
   PubMed candidates are generated programmatically, manually reviewed, and reduced to a balanced 80-paper corpus. SQLite FTS5/BM25 and MiniLM semantic retrieval are evaluated on the same information needs and labels.

3. **Retrieval disagreement analysis**  
   Ranked results are retained at the paper level so the aggregate metrics can be traced back to specific documents.

4. **Real-data zero-shot classification**  
   A DeBERTa MNLI model assigns one of four natural-language topic labels without task-specific fine-tuning. Performance is evaluated with accuracy, macro-F1, a confusion matrix, confidence analysis, and manual inspection of errors.

5. **LangGraph hybrid retrieval workflow**  
   A standalone two-node workflow composes the evaluated retrieval methods: SQLite FTS5/BM25 first retrieves a candidate set, then MiniLM reranks those candidates by semantic similarity. The hybrid achieved mean Precision@5 = **0.95** and MRR = **1.00**, matching full-corpus MiniLM on the current benchmark.

## Hybrid retrieval architecture

```mermaid
flowchart LR
    A[User query] --> B[SQLite FTS5 / BM25]
    B --> C[Top-N candidate papers]
    C --> D[MiniLM semantic reranking]
    D --> E[Final ranked results]
```

The graph maintains a shared search state containing the query, candidate-set size, candidate papers, and final ranked results. The first node performs lexical candidate retrieval; the second embeds only those candidates and reranks them by cosine similarity.

At the present scale of 80 papers, this orchestration is not required for efficiency and does not improve benchmark accuracy. Its purpose is to demonstrate a production-oriented pattern that can scale more naturally to much larger document collections.

## Technical stack

- Python 3.12
- SQL / SQLite
- SQLite FTS5 / BM25
- pandas / NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- LangGraph
- Matplotlib
- PubMed / NCBI E-utilities

## Models

- `sentence-transformers/all-MiniLM-L6-v2` — document/query embeddings and semantic retrieval
- `MoritzLaurer/DeBERTa-v3-base-mnli` — zero-shot topic classification

No model is fine-tuned on the project corpus.

## Data and reproducibility

The repository commits the curated identifier/label manifest rather than a static copy of PubMed abstracts:

```text
data/pubmed_labels.csv
```

The final corpus contains **80 manually reviewed papers, 20 per topic**. Candidate-search provenance is not used as the gold label; papers are assigned according to their primary scientific or clinical focus.

Full titles and abstracts are fetched locally from PubMed and written to the ignored cache directory. This keeps acquisition reproducible while avoiding unnecessary redistribution of abstract text.

## Run locally

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fetch the curated PubMed corpus:

```bash
export NCBI_EMAIL="your-email@example.com"
python scripts/fetch_curated_pubmed.py
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
biomedical_literature_search_hf_sql.ipynb
```

The first run downloads pretrained model weights from Hugging Face.

To run the hybrid LangGraph retrieval workflow:

```bash
python scripts/hybrid_search_graph.py
```

This evaluates the two-node BM25 → MiniLM workflow across the same four information needs used in the retrieval benchmark.

## Evaluation

### Retrieval

The lexical, semantic, and hybrid retrieval approaches are evaluated with:

- **Precision@5** — proportion of the first five results belonging to the expected topic
- **Reciprocal rank** — inverse rank of the first relevant result
- **MRR** — mean reciprocal rank across information needs

FTS5 receives Boolean/phrase queries appropriate for lexical search, while MiniLM receives natural-language versions of the same underlying information needs. The LangGraph hybrid uses both forms: FTS5/BM25 retrieves the candidate set, then MiniLM reranks those candidates semantically.

| Retrieval method | Mean Precision@5 | MRR |
| --- | ---: | ---: |
| SQLite FTS5/BM25 | **1.00** | **1.00** |
| MiniLM semantic | **0.95** | **1.00** |
| LangGraph hybrid (BM25 → MiniLM) | **0.95** | **1.00** |

### Classification

Zero-shot classification is evaluated with:

- accuracy
- macro-F1
- confusion matrix
- confidence summaries
- paper-level error inspection

The classifier receives only candidate label descriptions at inference time; no labeled training examples from the benchmark are provided.

## Why the baseline matters

On this small, balanced, terminology-rich corpus, **SQLite FTS5/BM25 performs extremely well** and slightly exceeds MiniLM's Precision@5. That is a useful outcome rather than a failed experiment: the project demonstrates model selection based on evidence rather than novelty.

The disagreement analysis also shows why aggregate metrics need context. MiniLM's only cross-topic top-five result was a neurodevelopment paper about the development of structure-function coupling and white-matter architecture — semantically relevant to the white-matter/connectivity query even though its single gold label belonged to another class.

The hybrid LangGraph workflow produced the same aggregate performance as full-corpus MiniLM. That result reinforces the same lesson at the architecture level: composing more components does not automatically improve a small benchmark. The hybrid's value is instead in separating candidate generation from semantic reranking, a design that becomes more meaningful as corpus size and compute cost increase.

## Future directions

The most useful next improvements follow directly from the limitations observed in the benchmark:

- **Evaluate hybrid retrieval at scale:** test whether BM25 candidate generation reduces semantic-search cost while preserving retrieval quality on substantially larger corpora.
- **Query-specific relevance judgments:** evaluate whether a paper actually answers a given information need rather than treating broad topic membership as relevance.
- **Multi-label classification:** allow papers to belong to overlapping scientific domains instead of forcing one primary label.
- **Larger-scale indexing:** expand from tens of papers to thousands or millions and introduce a scalable vector or hybrid index.
- **Conditional human review:** extend the LangGraph workflow with confidence- or disagreement-based branches that route ambiguous results for manual inspection.
- **Broader query testing:** add paraphrased, ambiguous, and terminology-mismatched queries where semantic retrieval may offer more value.
- **Query transformation and routing:** test whether a lightweight preprocessing node can translate natural-language questions into retrieval strategies without adding unnecessary model complexity.

These extensions would move the project from a compact benchmark toward a more realistic literature-triage or knowledge-retrieval system.

## AI-assisted development

This project was developed iteratively with AI coding assistance. The workflow included environment debugging, dependency pinning, query design, code review, and refactoring with AI support, while the PubMed corpus was manually reviewed and the final benchmark labels were human-curated.

The Git history is intentionally preserved to show that progression from prototype through debugging, real-data acquisition, evaluation, refinement, and workflow orchestration.

## Limitations

This is a compact portfolio benchmark rather than a production search engine or definitive scientific benchmark:

- only 80 papers and four balanced topic classes;
- one primary-topic label per paper rather than multi-label/query-specific relevance;
- only four retrieval information needs;
- targeted PubMed candidate generation rather than a representative PubMed sample;
- one embedding model and one zero-shot classification model;
- classifier confidence is not calibrated;
- the hybrid workflow is demonstrated on a corpus too small to test its scalability advantage directly.

## 30-second project summary

> I built an end-to-end biomedical literature retrieval and classification workflow using SQL and Hugging Face pretrained transformers. I validated the pipeline on synthetic data, generated and manually curated an 80-paper PubMed benchmark, compared SQLite FTS5/BM25 against MiniLM semantic retrieval, and evaluated DeBERTa zero-shot classification at 87.5% accuracy and macro-F1 of 0.875. I then used LangGraph to compose the evaluated retrieval components into a two-stage BM25 → MiniLM hybrid workflow, demonstrating how the benchmarked methods could be reused in a scalable multi-stage search architecture.
