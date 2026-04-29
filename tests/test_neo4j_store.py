"""Smoke tests for the optional Neo4j backend.

These tests only run when:
  * the ``neo4j`` driver is installed, and
  * ``NEO4J_URI`` + ``NEO4J_PASSWORD`` are set in the environment, and
  * the server at that URI is reachable.

Otherwise the whole module is skipped so CI and contributors without a
local Neo4j instance aren't penalised.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

pytest.importorskip("neo4j")

import networkx as nx
import pandas as pd

from src.extensions.neo4j_store import Neo4jStore, Neo4jUnavailableError


_SOURCE_LABEL = "pytest_neo4j_smoke"


@pytest.fixture(scope="module")
def store():
    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set; skipping Neo4j integration test.")
    try:
        s = Neo4jStore.from_config()
        s.connect()
    except Neo4jUnavailableError as e:
        pytest.skip(f"Neo4j not reachable: {e}")
    yield s
    try:
        s.wipe_source(_SOURCE_LABEL)
    finally:
        s.close()


@pytest.fixture
def sample_graph() -> tuple[nx.DiGraph, dict[str, int], pd.DataFrame]:
    G = nx.DiGraph()
    G.add_node("climate_change", label="climate change",
               type="noun_phrase", source_type="policy", frequency=10)
    G.add_node("adaptation", label="adaptation",
               type="keyword", source_type="both", frequency=8)
    G.add_node("resilience", label="resilience",
               type="keyword", source_type="policy", frequency=6)
    G.add_edge(
        "climate_change", "adaptation",
        weight=5, types={"CAUSATION"},
        verbs=Counter({"cause": 4, "drive": 1}),
    )
    G.add_edge(
        "adaptation", "resilience",
        weight=3, types={"ASSOCIATION"},
        verbs=Counter(),
    )

    partition = {"climate_change": 0, "adaptation": 0, "resilience": 1}
    centrality_df = pd.DataFrame(
        [
            {"node_id": "climate_change", "label": "climate change",
             "pagerank": 0.5, "betweenness": 0.3},
            {"node_id": "adaptation", "label": "adaptation",
             "pagerank": 0.3, "betweenness": 0.6},
            {"node_id": "resilience", "label": "resilience",
             "pagerank": 0.2, "betweenness": 0.1},
        ]
    )
    return G, partition, centrality_df


class TestNeo4jRoundTrip:
    def test_push_and_query_roundtrip(self, store, sample_graph):
        G, partition, centrality_df = sample_graph
        counts = store.push_graph(
            G, partition=partition, centrality_df=centrality_df,
            source_label=_SOURCE_LABEL, reset=True,
        )
        assert counts["nodes"] == 3
        assert counts["edges"] == 2

        with store.session() as s:
            rec = s.run(
                "MATCH (c:Concept {source_label: $sl}) RETURN count(c) AS n",
                sl=_SOURCE_LABEL,
            ).single()
            assert rec["n"] == 3
            rec = s.run(
                "MATCH (:Concept {source_label: $sl})-"
                "[r:RELATED {source_label: $sl}]->(:Concept {source_label: $sl}) "
                "RETURN count(r) AS n",
                sl=_SOURCE_LABEL,
            ).single()
            assert rec["n"] == 2

    def test_graph_context_cypher(self, store, sample_graph):
        from src.extensions.neo4j_graph_context import Neo4jGraphContext

        G, partition, centrality_df = sample_graph
        store.push_graph(
            G, partition=partition, centrality_df=centrality_df,
            source_label=_SOURCE_LABEL, reset=True,
        )
        gc = Neo4jGraphContext(store, source_label=_SOURCE_LABEL)
        assert set(gc.all_labels()) == {"climate change", "adaptation", "resilience"}
        assert gc.resolve("climate change") == "climate_change"

        summary = gc.graph_summary()
        assert "3 concepts" in summary

        path = gc.shortest_path("climate change", "resilience")
        assert "hops" in path.lower() or "path" in path.lower()

        top = gc.top_concepts(metric="pagerank", n=3)
        assert "climate change" in top

    def test_edge_embeddings_roundtrip(self, store, sample_graph):
        pytest.importorskip("sentence_transformers")

        G, partition, centrality_df = sample_graph
        store.push_graph(
            G, partition=partition, centrality_df=centrality_df,
            source_label=_SOURCE_LABEL, reset=True,
        )

        # Verb preservation on the edge itself.
        with store.session() as s:
            rec = s.run(
                "MATCH (:Concept {source_label: $sl, id: 'climate_change'})"
                "-[r:RELATED {source_label: $sl}]->"
                "(:Concept {source_label: $sl, id: 'adaptation'}) "
                "RETURN r.top_verb AS top_verb, r.verb_count AS verb_count, "
                "       r.verb_list AS verb_list",
                sl=_SOURCE_LABEL,
            ).single()
            assert rec is not None
            assert rec["top_verb"] == "cause"
            assert int(rec["verb_count"]) == 4
            assert "cause" in (rec["verb_list"] or [])

        # Embed all verb edges + top-N=1 ASSOCIATION (so both edges get vectors).
        embedded = store.embed_edges(
            G, source_label=_SOURCE_LABEL, top_n_association=1
        )
        assert embedded == 2

        # Relationship vector index exists and returns results.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode(
            ["what causes adaptation?"],
            convert_to_numpy=True,
        ).astype("float32")[0].tolist()
        with store.session() as s:
            rows = list(
                s.run(
                    "CALL db.index.vector.queryRelationships("
                    "$idx, 5, $vec) "
                    "YIELD relationship, score "
                    "WHERE relationship.source_label = $sl "
                    "RETURN startNode(relationship).id AS src, "
                    "       endNode(relationship).id AS tgt, "
                    "       relationship.top_verb AS verb, "
                    "       score",
                    idx="related_embedding",
                    vec=vec,
                    sl=_SOURCE_LABEL,
                )
            )
        assert rows, "edge vector query returned no results"
        top = rows[0]
        assert top["src"] == "climate_change"
        assert top["tgt"] == "adaptation"
        assert top["verb"] == "cause"
