# Text-to-Network Engine

**From Text to Networks: Building a Conceptual Network Engine for Data Science and Social Network Analysis**

A Python pipeline that transforms unstructured text into conceptual networks for social network analysis (SNA). Built for SNA Toolbox as part of COS70008 Technology Innovation Project.

## Overview

The engine ingests raw text (policy documents, Yelp reviews, or narrative text), extracts key concepts and their relationships using NLP techniques, constructs a network graph, and applies SNA methods to produce actionable insights.

## Pipeline Stages

1. **Data Ingestion** — Read PDF policy documents or Yelp JSON reviews
2. **Preprocessing** — Clean, tokenise, lemmatise, and filter text
3. **Concept & Relationship Extraction** — NER, noun-phrase chunking, TF-IDF; co-occurrence windows, dependency parsing, and causal pattern matching
4. **Network Construction** — Build NetworkX graphs with typed, weighted edges
5. **Network Analysis** — Centrality, community detection, brokerage, path analysis
6. **Visualisation** — Static matplotlib plots and interactive pyvis HTML graphs
7. **Temporal Comparison** *(optional)* — Build separate networks per time slice / document section and compare overlap and drift
8. **Conversational Exploration** *(optional)* — LLM/RAG chatbot grounded in the conceptual network

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the full pipeline on the policy document
python pipeline.py --source policy --output results/

# Run on Yelp reviews (sampled)
python pipeline.py --source yelp --yelp-category Restaurants --yelp-sample 500 --output results/

# Add temporal comparison (3 slices) and sentiment-weighted edges
python pipeline.py --source combined --temporal 3 --sentiment --output results/

# Launch the interactive Streamlit dashboard (5 analysis tabs + Temporal + Chat)
streamlit run src/extensions/dashboard.py

# Stage 8: conversational Q&A over the knowledge graph
python chat.py --source policy              # standalone
python pipeline.py --source policy --chat   # appended to the main pipeline
```

### LLM / chatbot configuration

The chatbot (`chat.py`, `--chat`, and the dashboard **Chat** tab) is
provider-agnostic. Select a backend via `LLM_PROVIDER`:

| Provider             | Env vars                        | Install                       |
|----------------------|----------------------------------|-------------------------------|
| `echo` *(default)*   | none                             | none — no-LLM fallback        |
| `openai`             | `OPENAI_API_KEY`, `LLM_MODEL`    | `pip install openai`          |
| `dashscope` *(Qwen)* | `DASHSCOPE_API_KEY`, `LLM_MODEL` | `pip install openai`          |
| `huggingface_api`    | `HUGGINGFACE_API_TOKEN`, `LLM_MODEL` | `pip install huggingface_hub` |
| `huggingface_local`  | `LLM_MODEL` (local model id)    | `pip install transformers`    |

Optional semantic retrieval (improves routing for paraphrased questions)
requires `sentence-transformers` + `faiss-cpu` (already in `requirements.txt`).

```bash
# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python chat.py --source combined --vector-store

# Aliyun DashScope (Qwen) — OpenAI-compatible, same client, just a different endpoint
export LLM_PROVIDER=dashscope
export LLM_MODEL=qwen-turbo            # or qwen-plus / qwen-max / qwen2.5-72b-instruct
export DASHSCOPE_API_KEY=sk-...
python chat.py --source combined --vector-store
```

> **Note on embeddings.** The vector store (FAISS / Neo4j) keeps using the
> local `sentence-transformers` model regardless of which LLM you pick.
> Switching to a remote embedding model (e.g. DashScope `text-embedding-v2`,
> 1536-dim) would require setting `NEO4J_EMBEDDING_DIM=1536` and re-running
> the pipeline with `--neo4j-reset` to rebuild every node and edge vector.

## Project Structure

```
src/
  config.py               # Global settings and parameters (including LLM/embedding)
  ingest/                 # Data readers (PDF, Yelp JSON)
  preprocessing/          # Text cleaning and tokenisation
  extraction/             # Concept and relationship extraction
  network/                # Graph construction and SNA analysis
  visualisation/          # Static and interactive visualisations
  extensions/             # Sentiment, concept dictionary, temporal,
                          # Streamlit dashboard, chatbot (Stage 8)
notebooks/                # Jupyter notebooks for exploration and demos
config/                   # User-editable concept_dictionary.yaml
tests/                    # Unit tests
pipeline.py               # CLI entry point (stages 1-8)
chat.py                   # Standalone Stage-8 chatbot entry point
```

## Optional Extensions

- **Sentiment-weighted edges** — Attach VADER/TextBlob sentiment scores to relationships (`--sentiment`)
- **User-defined concept dictionary** — Analyst-controlled vocabularies (`--concept-dictionary default`, see `config/concept_dictionary.yaml`)
- **Temporal comparison** — Compare networks across document sections or time periods (`--temporal N`, or the Temporal tab in the dashboard)
- **Interactive dashboard** — Streamlit app for exploring networks in the browser, including a Chat tab wired to the Stage-8 chatbot
- **LLM/RAG chatbot (Stage 8)** — Conversational Q&A grounded in the graph, with pluggable OpenAI / Hugging Face / local providers (`python chat.py`, `--chat`, or the dashboard Chat tab)
- **Neo4j backend** — Persist the conceptual network in a local Neo4j instance; route chatbot retrieval through Cypher + the native vector index (see below)

## Optional: Neo4j backend

NetworkX + CSV remains the default. Neo4j is an additive backend that gives
you a persistent, Cypher-queryable graph and a server-side vector index for
the Stage-8 chatbot.

### 1. Start Neo4j locally

A `docker-compose.yml` is included. It launches Neo4j 5 with APOC + Graph
Data Science plugins and mounts data/logs under `./.neo4j/` (gitignored).

```bash
cp .env.example .env                  # contains NEO4J_PASSWORD
set -a && source .env && set +a       # or use direnv / your preferred tool
docker compose up -d
# Browser UI at http://localhost:7474 (user: neo4j, pwd: from .env)
```

### 2. Populate it from the pipeline

```bash
# Build the graph and push nodes + edges + embeddings into Neo4j
python pipeline.py --source combined --neo4j

# Re-run cleanly (wipes prior Concept nodes with this source label)
python pipeline.py --source combined --neo4j --neo4j-reset
```

What gets written:

- `(:Concept {id, label, concept_type, source_type, frequency, community, pagerank, betweenness, source_label})`
- `-[:RELATED {weight, types, sentiment, source_label, top_verb, verb_count, verb_list}]->`
- `c.embedding` (384-d cosine vector, same `all-MiniLM-L6-v2` model as FAISS)
- `r.embedding` (384-d cosine vector, curated subset of edges — see below)
- Schema: uniqueness constraint on `:Concept(id)`, node vector index
  `concept_embedding`, and relationship vector index `related_embedding`

### 2b. Edge-level semantic search

Verb lemmas are preserved end-to-end from the spaCy dependency parse through
to Neo4j, so the chatbot can answer relational questions like _"who
recommended the food?"_ rather than only "which concepts are important?".

To keep storage and runtime reasonable we embed a curated subset of edges:

- **all** edges produced by dependency parsing (ACTION + CAUSATION — i.e.
  every edge with a non-empty `r.verb_list`), and
- the **top-N** ASSOCIATION edges by weight (default N = 2000, configurable
  via `--edge-embed-top-n`).

```bash
# Embed the default top-2000 association edges
python pipeline.py --source combined --neo4j

# Richer edge recall (slower, more storage)
python pipeline.py --source combined --neo4j --edge-embed-top-n 5000

# Verb-only embedding (skip ASSOCIATION co-occurrence edges entirely)
python pipeline.py --source combined --neo4j --edge-embed-top-n 0
```

Inside Neo4j the curated subset is queried via the native relationship
vector index:

```cypher
CALL db.index.vector.queryRelationships('related_embedding', 5, $query_vec)
YIELD relationship, score
RETURN startNode(relationship).label AS src,
       endNode(relationship).label   AS tgt,
       relationship.top_verb         AS verb,
       relationship.weight           AS weight,
       score
ORDER BY score DESC;
```

The chatbot's `QueryRouter` automatically switches on edge-level search
when the question contains a relational verb (recommend, hate, order,
cause, prefer, ...) or a `who/what ... verb ...` pattern. Example
questions that now work well:

- "Who recommended the food?"
- "What causes resilience?"
- "Which customers complain about service?"

Backward-compatible: questions without relational verbs behave exactly
as before, and running without `--neo4j` (in-memory FAISS backend) also
builds a `GraphEdgeVectorStore` with the same verb-aware descriptions.

### 3. Query it from the chatbot

```bash
# Standalone chat against Neo4j
python chat.py --source combined --neo4j

# Or let the pipeline chain it after a push
python pipeline.py --source combined --neo4j --chat

# In the Streamlit dashboard, toggle "Retrieval backend → Neo4j" in the Chat tab
streamlit run src/extensions/dashboard.py
```

All chatbot methods (`describe_concept`, `top_concepts`, `shortest_path`,
`compare_sources`, `graph_summary`) execute as Cypher, and semantic search
runs via `db.index.vector.queryNodes` instead of FAISS.

### 4. Environment variables

Set in `.env` (see `.env.example` for defaults):

| Variable | Default | Notes |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Use `neo4j+s://...` for AuraDB |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | *(required)* | Also consumed by `docker compose` |
| `NEO4J_DATABASE` | `neo4j` | |
| `NEO4J_VECTOR_INDEX` | `concept_embedding` | |
| `NEO4J_EDGE_VECTOR_INDEX` | `related_embedding` | Used by edge-level semantic search |
| `NEO4J_EMBEDDING_DIM` | `384` | Must match `EMBEDDING_MODEL` output |

If Neo4j is unreachable, `pipeline.py --neo4j`, `chat.py --neo4j`, and the
dashboard toggle all fall back gracefully to the in-memory NetworkX + FAISS
path and print a warning.

## Data Sources

- Australian National Climate Resilience and Adaptation Strategy (2021–2025)
- Yelp Open Dataset (academic review subset)

**Local setup:** Large files are gitignored. After clone, place the policy PDF and/or extracted Yelp JSON under `data/` as described in [`data/README.md`](data/README.md). Configured paths are defined in `src/config.py` (`POLICY_PDF_PATH`, `YELP_EXTRACTED_DIR`).

**GitHub hygiene:** Do not commit `venv/` (reinstall from `requirements.txt`). The `papers/`, `project_brief/`, and `Research_Ethics/` folders are excluded via `.gitignore` for this upload.
