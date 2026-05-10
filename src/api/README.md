# Conceptual Network HTTP API

A read-only FastAPI layer over `Neo4jGraphReader`. Exposes the Neo4j graph
as plain HTTP endpoints with auto-generated Swagger UI, so callers don't
need to write Cypher or even speak Python.

## Architecture

```
HTTP client (browser / curl / frontend / notebook)
        │
        ▼
FastAPI  (src/api/server.py)
        │
        ▼
Neo4jGraphReader  (src/extensions/neo4j_reader.py)  ← typed Cypher queries
        │
        ▼
Neo4j 5  (bolt://localhost:7687)  ← populated by `pipeline.py --neo4j`
```

The server holds **one** `Neo4jStore` (driver) for its lifetime and
constructs **one `Neo4jGraphReader` per `source` value** on first use.
Semantic endpoints lazily load a `sentence-transformers` model the
first time they're called, then keep it warm.

## Run

```bash
pip install -r requirements.txt          # installs fastapi + uvicorn
docker compose up -d                     # start Neo4j
python pipeline.py --source combined --neo4j --neo4j-reset   # populate
uvicorn src.api.server:app --reload      # start the API
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

All endpoints are `GET` and accept `?source=combined` (default), with
`policy` and `yelp` as alternatives.

## Endpoints — purpose and what they do in Neo4j

### Meta

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/health` | Is the server up? Is Neo4j reachable? | Driver `verify_connectivity()` ping |
| `/stats` | One-shot summary: nodes, edges, communities, density | `count(c)`, `count(r)`, `collect(DISTINCT c.community)` |
| `/sources` | Node count broken down by `source_type` | `RETURN source_type, count(*)` |

### Concepts

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/concepts` | Browse the full concept table (paginated) | `MATCH (c:Concept) ORDER BY c.pagerank DESC` |
| `/concepts/search?q=...` | Find concepts by substring on label | `WHERE toLower(c.label) CONTAINS toLower($q)` |
| `/concepts/top?metric=pagerank` | Ranked list by `pagerank`, `betweenness`, or `frequency` | `ORDER BY c.{metric} DESC LIMIT N` |
| `/concepts/{ident}` | Inspect a single concept (id or fuzzy label) | `MATCH (c:Concept {id: $id}) RETURN c.*` |
| `/concepts/{ident}/neighbours` | Direct neighbours (in / out / both), ranked by edge weight | `MATCH (c)-[r:RELATED]->(n)` and the reverse, ordered by `r.weight` |
| `/concepts/{ident}/subgraph?depth=2` | k-hop neighbourhood as node-link JSON | `MATCH (c)-[:RELATED*1..d]-(n)` then induced edges between collected ids |

### Communities & brokers

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/communities` | List communities with size and top-5 members | `WITH c.community AS community, collect(c) AS members` |
| `/communities/{id}/subgraph` | All concepts and inner edges of one community | Members where `c.community = $id`, then their `:RELATED` edges |
| `/brokers` | Bridging nodes ranked by `betweenness × cross_community_edges` | `count(DISTINCT n WHERE n.community <> c.community)`, multiplied by stored `c.betweenness` |

### Paths and cross-source

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/path?a=...&b=...` | Shortest path between two concepts as a list of edges | `MATCH p = shortestPath((s)-[:RELATED*..N]-(t))` |
| `/cross-source/edges` | Edges whose endpoints have different `source_type` | `WHERE a.source_type <> b.source_type` |
| `/cross-source/bridging-concepts` | Concepts that link policy ↔ yelp, by `bridge_score` | `betweenness * (1 + cross_ratio) + 0.1 * is_shared` |

### Whole graph

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/graph` | Full network reconstructed as node-link JSON (or top-N by PageRank) | `MATCH` all `:Concept` nodes and `:RELATED` edges; reassembled as `nx.DiGraph` then serialised |

### Semantic search

| Endpoint | Purpose | Neo4j operation |
|---|---|---|
| `/semantic/concepts?q=...` | Top-k concepts by cosine similarity to your text | Server embeds `q` with `sentence-transformers` → `CALL db.index.vector.queryNodes('concept_embedding', k, $vec)` |
| `/semantic/edges?q=...` | Top-k edges by cosine similarity (verb-aware) | Same flow over `CALL db.index.vector.queryRelationships('related_embedding', k, $vec)` |

## Quick-try

```bash
# 1. Network at a glance
curl 'http://localhost:8000/stats?source=combined'

# 2. Five most central concepts
curl 'http://localhost:8000/concepts/top?metric=pagerank&top_n=5'

# 3. What is "climate change" connected to?
curl 'http://localhost:8000/concepts/climate%20change/neighbours?top_n=5'

# 4. Free-text semantic search (server embeds the query)
curl 'http://localhost:8000/semantic/concepts?q=adaptation%20strategy&k=3'
```

## Common gotchas

- **503 from `/health`**: Neo4j wasn't reachable when the server started.
  Run `docker compose up -d` and restart `uvicorn`.
- **404 from `/concepts/{ident}`**: The reader does fuzzy match (cutoff
  0.7) — if your label is too far off, you'll get 404. Try
  `/concepts/search?q=...` first to find an exact label.
- **Empty results after re-running the pipeline**: each `Neo4jGraphReader`
  caches its label → id index. Restart `uvicorn` (or hit a different
  `source` value) to force a refresh.
- **First semantic call is slow**: it loads the embedding model on
  demand. Subsequent calls reuse the warm model.
