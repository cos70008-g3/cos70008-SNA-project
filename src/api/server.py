"""FastAPI HTTP layer over :class:`Neo4jGraphReader`.

Exposes read-only endpoints so the conceptual network can be browsed
without writing Cypher. Designed to be self-documenting via Swagger:

    uvicorn src.api.server:app --reload

* Swagger UI:  http://localhost:8000/docs
* ReDoc UI:    http://localhost:8000/redoc

The server holds a single :class:`Neo4jStore` for its lifetime (opened
on startup, closed on shutdown) and constructs one
:class:`Neo4jGraphReader` per ``source_label`` on first use. Semantic
endpoints lazily load a ``sentence-transformers`` model the first time
they are called.
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.extensions.neo4j_reader import Neo4jGraphReader
from src.extensions.neo4j_store import Neo4jStore, Neo4jUnavailableError

logger = logging.getLogger(__name__)


# ── Shared app state ───────────────────────────────────────────────
_state: dict[str, Any] = {
    "store": None,
    "readers": {},   # source_label -> Neo4jGraphReader
    "embedder": None,
}


def _get_store() -> Neo4jStore:
    store = _state["store"]
    if store is None:
        raise HTTPException(503, "Neo4j is not connected")
    return store


def _get_reader(source: str) -> Neo4jGraphReader:
    readers = _state["readers"]
    if source not in readers:
        readers[source] = Neo4jGraphReader(_get_store(), source_label=source)
    return readers[source]


def _get_embedder():
    if _state["embedder"] is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise HTTPException(
                503,
                "sentence-transformers is not installed; semantic endpoints "
                "are disabled. Run `pip install sentence-transformers`.",
            ) from e
        from src import config as _cfg
        _state["embedder"] = SentenceTransformer(_cfg.EMBEDDING_MODEL)
    return _state["embedder"]


# ── Lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        store = Neo4jStore.from_config()
        store.connect()
        _state["store"] = store
        logger.info("Neo4j connected at %s", store.uri)
    except Neo4jUnavailableError as e:
        logger.warning(
            "Neo4j unavailable at startup (%s). API will return 503 until "
            "the database is reachable; restart the server after fixing.",
            e,
        )
    yield
    if _state["store"] is not None:
        _state["store"].close()
    _state["store"] = None
    _state["readers"].clear()


app = FastAPI(
    title="Conceptual Network API",
    description=(
        "HTTP read-only interface over the Neo4j-backed conceptual network. "
        "Wraps `Neo4jGraphReader`, scoped by a `source` query parameter "
        "(`policy`, `yelp`, `combined`, default `combined`). "
        "Visit `/docs` for the interactive Swagger UI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS so notebooks and local dashboards can hit the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Serialization helpers ──────────────────────────────────────────
def _jsonable(value: Any) -> Any:
    """Recursively coerce values into JSON-friendly forms.

    Handles Python sets, NaN/Infinity floats, and numpy scalars (which
    leak in via ``pandas`` columns).
    """
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.where(df.notna(), None)
    return [
        {k: _jsonable(v) for k, v in row.items()}
        for row in safe.to_dict(orient="records")
    ]


def _graph_to_dict(G: nx.DiGraph) -> dict:
    """Return a JSON-friendly node-link representation of a DiGraph."""
    try:
        data = nx.node_link_data(G, edges="links")
    except TypeError:  # older networkx without `edges` kwarg
        data = nx.node_link_data(G)
    return _jsonable(data)


# ── Common dependency ──────────────────────────────────────────────
def reader_dep(
    source: str = Query(
        "combined",
        description="source_label scoping the queries (e.g. policy / yelp / combined)",
    ),
) -> Neo4jGraphReader:
    return _get_reader(source)


# ── Endpoints — meta ───────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check + Neo4j connectivity probe."""
    if _state["store"] is None:
        return {"status": "degraded", "neo4j": False}
    return {"status": "ok", "neo4j": Neo4jStore.ping()}


@app.get("/stats", tags=["meta"])
def stats(reader: Neo4jGraphReader = Depends(reader_dep)) -> dict:
    """Top-level node/edge/community counts and density."""
    return _jsonable(reader.graph_stats())


@app.get("/sources", tags=["meta"])
def sources(reader: Neo4jGraphReader = Depends(reader_dep)) -> list[dict]:
    """Node count broken down by `source_type`."""
    return _df_to_records(reader.source_distribution())


# ── Endpoints — concepts ───────────────────────────────────────────
@app.get("/concepts", tags=["concepts"])
def concepts_all(
    limit: int | None = Query(None, ge=1, le=10000),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """All concepts ordered by PageRank."""
    return _df_to_records(reader.all_concepts(limit=limit))


@app.get("/concepts/search", tags=["concepts"])
def concepts_search(
    q: str = Query(..., description="Substring matched (case-insensitive) against concept labels"),
    limit: int = Query(20, ge=1, le=200),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Substring search over concept labels, ranked by PageRank."""
    return _df_to_records(reader.search_concepts(q, limit=limit))


@app.get("/concepts/top", tags=["concepts"])
def concepts_top(
    metric: str = Query("pagerank", description="One of: pagerank, betweenness, frequency"),
    top_n: int = Query(20, ge=1, le=500),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Top-N concepts by a persisted centrality metric."""
    try:
        df = reader.top_concepts(metric=metric, top_n=top_n)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _df_to_records(df)


@app.get("/concepts/{ident}", tags=["concepts"])
def concepts_get(
    ident: str,
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> dict:
    """Single concept lookup by id or (fuzzy) label."""
    row = reader.get_concept(ident)
    if row is None:
        raise HTTPException(404, f"Concept '{ident}' not found")
    return _jsonable(row)


@app.get("/concepts/{ident}/neighbours", tags=["concepts"])
def concepts_neighbours(
    ident: str,
    top_n: int = Query(10, ge=1, le=200),
    direction: str = Query("both", pattern="^(in|out|both)$"),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Direct neighbours of a concept, ranked by edge weight."""
    nid = reader.resolve(ident)
    if nid is None:
        raise HTTPException(404, f"Concept '{ident}' not found")
    return _df_to_records(reader.neighbours(nid, top_n=top_n, direction=direction))


@app.get("/concepts/{ident}/subgraph", tags=["concepts"])
def concepts_subgraph(
    ident: str,
    depth: int = Query(2, ge=1, le=5),
    min_weight: int = Query(1, ge=0),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> dict:
    """Neighbourhood subgraph as node-link JSON (the format `nx.node_link_data` produces)."""
    nid = reader.resolve(ident)
    if nid is None:
        raise HTTPException(404, f"Concept '{ident}' not found")
    G = reader.neighbourhood_subgraph(nid, depth=depth, min_weight=min_weight)
    return _graph_to_dict(G)


# ── Endpoints — communities & brokers ──────────────────────────────
@app.get("/communities", tags=["communities"])
def communities_all(
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """All communities with size and top-5 members."""
    return _df_to_records(reader.communities())


@app.get("/communities/{community_id}/subgraph", tags=["communities"])
def communities_subgraph(
    community_id: int,
    min_weight: int = Query(1, ge=0),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> dict:
    """Induced subgraph on the members of a single community."""
    G = reader.community_subgraph(community_id, min_weight=min_weight)
    if G.number_of_nodes() == 0:
        raise HTTPException(404, f"Community {community_id} has no members")
    return _graph_to_dict(G)


@app.get("/brokers", tags=["communities"])
def brokers(
    top_n: int = Query(10, ge=1, le=200),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Top brokers by `betweenness * cross_community_edges`."""
    return _df_to_records(reader.brokers(top_n=top_n))


# ── Endpoints — paths & cross-source ───────────────────────────────
@app.get("/path", tags=["paths"])
def shortest_path(
    a: str = Query(..., description="Source concept (id or label)"),
    b: str = Query(..., description="Target concept (id or label)"),
    max_hops: int = Query(10, ge=1, le=20),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Shortest path between two concepts as a list of edge dicts (empty if none)."""
    return _jsonable(reader.shortest_path(a, b, max_hops=max_hops))


@app.get("/cross-source/edges", tags=["cross-source"])
def cross_source_edges(
    min_weight: int = Query(1, ge=0),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Edges that bridge concepts of different `source_type`."""
    return _df_to_records(reader.cross_source_edges(min_weight=min_weight))


@app.get("/cross-source/bridging-concepts", tags=["cross-source"])
def cross_source_bridging(
    top_n: int = Query(20, ge=1, le=200),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Concepts that connect different sources, ranked by `bridge_score`."""
    return _df_to_records(reader.bridging_concepts(top_n=top_n))


# ── Endpoints — whole graph ────────────────────────────────────────
@app.get("/graph", tags=["graph"])
def graph_full(
    top_n_by_pagerank: int | None = Query(
        None,
        ge=1,
        le=2000,
        description="If set, return only the top-N nodes by PageRank (and the edges between them).",
    ),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> dict:
    """Full graph as node-link JSON. Use `top_n_by_pagerank` to subset large graphs."""
    G = reader.load_graph(top_n_by_pagerank=top_n_by_pagerank)
    return _graph_to_dict(G)


# ── Endpoints — semantic ───────────────────────────────────────────
@app.get("/semantic/concepts", tags=["semantic"])
def semantic_concepts(
    q: str = Query(..., description="Free text — embedded server-side via sentence-transformers"),
    k: int = Query(5, ge=1, le=50),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Top-k concepts by cosine similarity to the embedded query."""
    model = _get_embedder()
    vec = model.encode([q], convert_to_numpy=True)[0].tolist()
    return _df_to_records(reader.semantic_search_concepts(vec, k=k))


@app.get("/semantic/edges", tags=["semantic"])
def semantic_edges(
    q: str = Query(..., description="Free text — embedded server-side via sentence-transformers"),
    k: int = Query(5, ge=1, le=50),
    reader: Neo4jGraphReader = Depends(reader_dep),
) -> list[dict]:
    """Top-k edges by cosine similarity to the embedded query."""
    model = _get_embedder()
    vec = model.encode([q], convert_to_numpy=True)[0].tolist()
    return _df_to_records(reader.semantic_search_edges(vec, k=k))
