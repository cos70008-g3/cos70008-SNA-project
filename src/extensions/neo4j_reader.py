"""Structured read API for the conceptual network stored in Neo4j.

``Neo4jGraphReader`` is the layer other components program against when
they need graph data — visualisation, the Streamlit dashboard, the Stage-8
chatbot, future HTTP endpoints. It complements :class:`Neo4jStore` (write
path) by providing typed Cypher queries that return ``pandas.DataFrame``,
``dict``, or ``networkx.DiGraph`` objects rather than formatted strings.

The reader does **not** manage its own driver lifetime — pass an already
connected ``Neo4jStore`` so the caller controls when the connection is
closed. All queries are scoped by ``source_label`` so multiple corpora can
coexist in the same database.
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)

_VALID_METRICS = {"pagerank", "betweenness", "frequency"}


class Neo4jGraphReader:
    """Read-only Cypher-backed view over a Concept subgraph in Neo4j.

    Parameters
    ----------
    store:
        A connected :class:`~src.extensions.neo4j_store.Neo4jStore`. The
        reader uses ``store.session()`` for every query and never opens or
        closes the underlying driver.
    source_label:
        Tag used to scope queries (matches the ``source_label`` written by
        ``Neo4jStore.push_graph``). Defaults to ``"combined"``.
    """

    def __init__(self, store: "Neo4jStore", source_label: str = "combined"):
        self.store = store
        self.source_label = source_label
        self._label_index: dict[str, str] | None = None

    # ── Cypher helpers ─────────────────────────────────────────────
    def _query_one(self, cypher: str, **params: Any) -> dict | None:
        with self.store.session() as s:
            rec = s.run(cypher, **params).single()
            return dict(rec) if rec else None

    def _query_all(self, cypher: str, **params: Any) -> list[dict]:
        with self.store.session() as s:
            return [dict(r) for r in s.run(cypher, **params)]

    # ── Label resolution ───────────────────────────────────────────
    def _ensure_label_index(self) -> dict[str, str]:
        if self._label_index is None:
            rows = self._query_all(
                "MATCH (c:Concept {source_label: $sl}) "
                "RETURN c.id AS id, c.label AS label",
                sl=self.source_label,
            )
            self._label_index = {(r["label"] or r["id"]): r["id"] for r in rows}
        return self._label_index

    def resolve(self, query: str) -> str | None:
        """Resolve a label or id to a concept id, with fuzzy fallback."""
        idx = self._ensure_label_index()
        if query in idx.values():
            return query
        if query in idx:
            return idx[query]
        lower = {k.lower(): v for k, v in idx.items()}
        if query.lower() in lower:
            return lower[query.lower()]
        match = get_close_matches(
            query.lower(), list(lower.keys()), n=1, cutoff=0.7
        )
        return lower[match[0]] if match else None

    def all_labels(self) -> list[str]:
        """Return every concept label under the configured source_label."""
        return list(self._ensure_label_index().keys())

    def invalidate_cache(self) -> None:
        """Drop the cached label index. Call after the store is repopulated."""
        self._label_index = None

    # ── Stats ──────────────────────────────────────────────────────
    def graph_stats(self) -> dict:
        """Top-level counts: nodes, edges, communities, density."""
        row = self._query_one(
            "MATCH (c:Concept {source_label: $sl}) "
            "OPTIONAL MATCH (c)-[r:RELATED {source_label: $sl}]->() "
            "WITH count(DISTINCT c) AS n, count(r) AS e, "
            "     size(collect(DISTINCT c.community)) AS comms "
            "RETURN n, e, comms",
            sl=self.source_label,
        )
        if not row:
            return {"nodes": 0, "edges": 0, "communities": 0, "density": 0.0}
        n = int(row["n"] or 0)
        e = int(row["e"] or 0)
        comms = int(row["comms"] or 0)
        density = (e / (n * (n - 1))) if n > 1 else 0.0
        return {"nodes": n, "edges": e, "communities": comms, "density": density}

    def source_distribution(self) -> pd.DataFrame:
        """Node count grouped by ``source_type`` (policy / yelp / both / unknown)."""
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "RETURN coalesce(c.source_type, 'unknown') AS source_type, "
            "       count(*) AS count "
            "ORDER BY source_type",
            sl=self.source_label,
        )
        return pd.DataFrame(rows, columns=["source_type", "count"])

    # ── Concept lookup ─────────────────────────────────────────────
    def get_concept(self, query: str) -> dict | None:
        """Fetch a single concept's properties by id or (fuzzy) label."""
        nid = self.resolve(query) or query
        return self._query_one(
            "MATCH (c:Concept {id: $id, source_label: $sl}) "
            "RETURN c.id AS id, c.label AS label, c.concept_type AS type, "
            "       c.source_type AS source_type, c.frequency AS frequency, "
            "       c.community AS community, c.pagerank AS pagerank, "
            "       c.betweenness AS betweenness",
            id=nid,
            sl=self.source_label,
        )

    def search_concepts(self, keyword: str, limit: int = 20) -> pd.DataFrame:
        """Case-insensitive substring search over concept labels."""
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "WHERE toLower(c.label) CONTAINS toLower($q) "
            "RETURN c.id AS node_id, c.label AS label, c.concept_type AS type, "
            "       c.source_type AS source_type, c.frequency AS frequency, "
            "       coalesce(c.pagerank, 0.0) AS pagerank "
            "ORDER BY pagerank DESC LIMIT $n",
            sl=self.source_label,
            q=keyword,
            n=int(limit),
        )
        return pd.DataFrame(
            rows,
            columns=["node_id", "label", "type", "source_type",
                     "frequency", "pagerank"],
        )

    def all_concepts(self, limit: int | None = None) -> pd.DataFrame:
        """All concepts as a DataFrame — drop-in replacement for ``centrality_df``."""
        cypher = (
            "MATCH (c:Concept {source_label: $sl}) "
            "RETURN c.id AS node_id, c.label AS label, "
            "       c.concept_type AS type, c.source_type AS source_type, "
            "       c.frequency AS frequency, c.community AS community, "
            "       coalesce(c.pagerank, 0.0) AS pagerank, "
            "       coalesce(c.betweenness, 0.0) AS betweenness "
            "ORDER BY pagerank DESC"
        )
        if limit is not None:
            cypher += f" LIMIT {int(limit)}"
        rows = self._query_all(cypher, sl=self.source_label)
        return pd.DataFrame(
            rows,
            columns=["node_id", "label", "type", "source_type", "frequency",
                     "community", "pagerank", "betweenness"],
        )

    # ── Top concepts ───────────────────────────────────────────────
    def top_concepts(
        self, metric: str = "pagerank", top_n: int = 20
    ) -> pd.DataFrame:
        """Top-N concepts by a persisted centrality metric.

        ``metric`` must be one of ``"pagerank"``, ``"betweenness"``, or
        ``"frequency"`` — the only ranking attributes ``Neo4jStore`` writes.
        """
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"metric must be one of {sorted(_VALID_METRICS)}; got {metric!r}"
            )
        rows = self._query_all(
            f"MATCH (c:Concept {{source_label: $sl}}) "
            f"WHERE c.{metric} IS NOT NULL "
            f"RETURN c.id AS node_id, c.label AS label, "
            f"       c.{metric} AS value, c.community AS community, "
            f"       c.source_type AS source_type "
            f"ORDER BY value DESC LIMIT $n",
            sl=self.source_label,
            n=int(top_n),
        )
        if not rows:
            return pd.DataFrame(
                columns=["node_id", "label", metric, "community", "source_type"]
            )
        df = pd.DataFrame(rows).rename(columns={"value": metric})
        return df[["node_id", "label", metric, "community", "source_type"]]

    # ── Neighbourhood ──────────────────────────────────────────────
    def neighbours(
        self,
        concept: str,
        top_n: int = 10,
        direction: str = "both",
    ) -> pd.DataFrame:
        """Direct neighbours of one concept, ranked by edge weight."""
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be 'in', 'out', or 'both'")
        nid = self.resolve(concept) or concept

        out_rows: list[dict] = []
        in_rows: list[dict] = []
        if direction in {"out", "both"}:
            out_rows = self._query_all(
                "MATCH (c:Concept {id: $id, source_label: $sl})"
                "-[r:RELATED]->(n:Concept) "
                "RETURN n.id AS node_id, n.label AS label, "
                "       r.weight AS weight, r.types AS types, "
                "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
                "       'out' AS direction "
                "ORDER BY r.weight DESC LIMIT $k",
                id=nid, sl=self.source_label, k=int(top_n),
            )
        if direction in {"in", "both"}:
            in_rows = self._query_all(
                "MATCH (n:Concept)-[r:RELATED]->"
                "(c:Concept {id: $id, source_label: $sl}) "
                "RETURN n.id AS node_id, n.label AS label, "
                "       r.weight AS weight, r.types AS types, "
                "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
                "       'in' AS direction "
                "ORDER BY r.weight DESC LIMIT $k",
                id=nid, sl=self.source_label, k=int(top_n),
            )

        merged = sorted(
            out_rows + in_rows,
            key=lambda r: r.get("weight") or 0,
            reverse=True,
        )[:top_n]
        return pd.DataFrame(
            merged,
            columns=["node_id", "label", "weight", "types",
                     "sentiment", "top_verb", "direction"],
        )

    def neighbourhood_subgraph(
        self,
        concept: str,
        depth: int = 2,
        min_weight: int = 1,
    ) -> nx.DiGraph:
        """Subgraph of all concepts within ``depth`` hops of the seed.

        Edges below ``min_weight`` are excluded after the node set is fixed,
        so the seed node is always present even if isolated.
        """
        nid = self.resolve(concept) or concept
        depth = max(1, int(depth))
        rows = self._query_all(
            f"MATCH (c:Concept {{id: $id, source_label: $sl}}) "
            f"OPTIONAL MATCH (c)-[:RELATED*1..{depth}]-"
            f"(n:Concept {{source_label: $sl}}) "
            f"WITH collect(DISTINCT c) + collect(DISTINCT n) AS xs "
            f"UNWIND xs AS x WITH DISTINCT x WHERE x IS NOT NULL "
            f"RETURN x.id AS id",
            id=nid, sl=self.source_label,
        )
        ids = [r["id"] for r in rows]
        if not ids:
            return nx.DiGraph()
        return self._build_subgraph_from_ids(ids, min_weight=min_weight)

    # ── Communities ────────────────────────────────────────────────
    def communities(self) -> pd.DataFrame:
        """One row per community: size and top-5 members by frequency."""
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "WITH c.community AS community, collect(c) AS members "
            "RETURN community, size(members) AS size, "
            "       [m IN members | "
            "         {label: m.label, freq: coalesce(m.frequency, 0)}] AS info "
            "ORDER BY community",
            sl=self.source_label,
        )
        out = []
        for r in rows:
            info = r.get("info") or []
            info.sort(key=lambda x: x.get("freq", 0), reverse=True)
            top = ", ".join(str(i["label"]) for i in info[:5])
            out.append({
                "community": (
                    int(r["community"]) if r["community"] is not None else -1
                ),
                "size": int(r["size"] or 0),
                "top_concepts": top,
            })
        return pd.DataFrame(out, columns=["community", "size", "top_concepts"])

    def community_subgraph(
        self, community_id: int, min_weight: int = 1
    ) -> nx.DiGraph:
        """Induced subgraph on the members of a single community."""
        rows = self._query_all(
            "MATCH (c:Concept {community: $cid, source_label: $sl}) "
            "RETURN c.id AS id",
            cid=int(community_id), sl=self.source_label,
        )
        ids = [r["id"] for r in rows]
        if not ids:
            return nx.DiGraph()
        return self._build_subgraph_from_ids(ids, min_weight=min_weight)

    # ── Path ───────────────────────────────────────────────────────
    def shortest_path(
        self, a: str, b: str, max_hops: int = 10
    ) -> list[dict]:
        """Shortest path between two concepts as a list of edge dicts.

        Returns ``[]`` if either endpoint is unknown or no path exists.
        Each edge dict has ``source_id``, ``source_label``, ``target_id``,
        ``target_label``, ``weight``, ``types`` (set), ``top_verb``,
        ``sentiment``. Direction reflects path order, not the underlying
        edge orientation.
        """
        n1, n2 = self.resolve(a), self.resolve(b)
        if n1 is None or n2 is None:
            return []
        max_hops = max(1, int(max_hops))
        row = self._query_one(
            f"MATCH (s:Concept {{id: $a, source_label: $sl}}) "
            f"MATCH (t:Concept {{id: $b, source_label: $sl}}) "
            f"MATCH p = shortestPath((s)-[:RELATED*..{max_hops}]-(t)) "
            f"RETURN [n IN nodes(p) | {{id: n.id, label: n.label}}] AS nodes, "
            f"       [r IN relationships(p) | {{weight: r.weight, "
            f"        types: r.types, top_verb: r.top_verb, "
            f"        sentiment: r.sentiment}}] AS rels",
            a=n1, b=n2, sl=self.source_label,
        )
        if not row:
            return []
        nodes = row.get("nodes") or []
        rels = row.get("rels") or []
        edges: list[dict] = []
        for i, r in enumerate(rels):
            edges.append({
                "source_id": nodes[i]["id"],
                "source_label": nodes[i]["label"],
                "target_id": nodes[i + 1]["id"],
                "target_label": nodes[i + 1]["label"],
                "weight": float(r.get("weight") or 0),
                "types": _parse_types(r.get("types")),
                "top_verb": r.get("top_verb"),
                "sentiment": r.get("sentiment"),
            })
        return edges

    # ── Cross-source ───────────────────────────────────────────────
    def cross_source_edges(self, min_weight: int = 1) -> pd.DataFrame:
        """Edges that bridge concepts of different ``source_type``."""
        rows = self._query_all(
            "MATCH (a:Concept {source_label: $sl})-[r:RELATED]->"
            "(b:Concept {source_label: $sl}) "
            "WHERE coalesce(a.source_type,'unknown') "
            "      <> coalesce(b.source_type,'unknown') "
            "  AND coalesce(r.weight, 1) >= $w "
            "RETURN a.label AS source_node, b.label AS target_node, "
            "       coalesce(a.source_type,'unknown') AS source_type_from, "
            "       coalesce(b.source_type,'unknown') AS source_type_to, "
            "       r.weight AS weight, r.types AS types, "
            "       r.top_verb AS top_verb "
            "ORDER BY weight DESC",
            sl=self.source_label, w=int(min_weight),
        )
        return pd.DataFrame(
            rows,
            columns=["source_node", "target_node", "source_type_from",
                     "source_type_to", "weight", "types", "top_verb"],
        )

    def bridging_concepts(self, top_n: int = 20) -> pd.DataFrame:
        """Concepts that connect different sources, ranked by ``bridge_score``.

        Mirrors ``GraphAnalyser.bridging_concepts``:
        ``bridge_score = betweenness * (1 + cross_ratio) + 0.1 * is_shared``.
        """
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "OPTIONAL MATCH (c)-[:RELATED]-(n:Concept {source_label: $sl}) "
            "WITH c, count(DISTINCT n) AS total, "
            "     count(DISTINCT CASE "
            "       WHEN coalesce(n.source_type,'unknown') "
            "            <> coalesce(c.source_type,'unknown') THEN n END) "
            "       AS cross_src "
            "WITH c, total, cross_src, "
            "     CASE WHEN total = 0 THEN 0.0 "
            "          ELSE toFloat(cross_src) / total END AS cross_ratio, "
            "     CASE WHEN c.source_type = 'both' THEN 1.0 ELSE 0.0 END AS shared "
            "RETURN c.id AS node_id, c.label AS label, "
            "       coalesce(c.source_type,'unknown') AS source_type, "
            "       coalesce(c.betweenness, 0.0) AS betweenness, "
            "       cross_src AS cross_source_neighbours, "
            "       total AS total_neighbours, cross_ratio, "
            "       coalesce(c.betweenness, 0.0) * (1 + cross_ratio) "
            "         + 0.1 * shared AS bridge_score "
            "ORDER BY bridge_score DESC LIMIT $n",
            sl=self.source_label, n=int(top_n),
        )
        return pd.DataFrame(
            rows,
            columns=["node_id", "label", "source_type", "betweenness",
                     "cross_source_neighbours", "total_neighbours",
                     "cross_ratio", "bridge_score"],
        )

    # ── Brokers ────────────────────────────────────────────────────
    def brokers(self, top_n: int = 10) -> pd.DataFrame:
        """Top brokers by ``broker_score = betweenness * cross_community_edges``.

        Uses the persisted ``betweenness`` and ``community`` values, so the
        ranking is computed in Cypher rather than client-side. Mirrors
        :meth:`src.network.graph_analysis.GraphAnalyser.find_brokers`.
        """
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "OPTIONAL MATCH (c)-[:RELATED]-(n:Concept {source_label: $sl}) "
            "WITH c, count(DISTINCT n) AS total, "
            "     count(DISTINCT CASE "
            "       WHEN coalesce(n.community, -1) "
            "            <> coalesce(c.community, -1) THEN n END) "
            "       AS cross_comm "
            "RETURN c.id AS node_id, c.label AS label, "
            "       coalesce(c.community, -1) AS community, "
            "       coalesce(c.betweenness, 0.0) AS betweenness, "
            "       cross_comm AS cross_community_edges, "
            "       total AS total_edges, "
            "       coalesce(c.betweenness, 0.0) * cross_comm AS broker_score "
            "ORDER BY broker_score DESC LIMIT $n",
            sl=self.source_label, n=int(top_n),
        )
        return pd.DataFrame(
            rows,
            columns=["node_id", "label", "community", "betweenness",
                     "cross_community_edges", "total_edges", "broker_score"],
        )

    # ── Whole-graph load ───────────────────────────────────────────
    def load_graph(self, top_n_by_pagerank: int | None = None) -> nx.DiGraph:
        """Reconstruct an in-memory ``nx.DiGraph`` from Neo4j.

        Node and edge attributes match what ``GraphBuilder`` produces, so
        ``InteractiveVisualiser``, ``StaticVisualiser``, and
        ``GraphAnalyser`` can run against the result unchanged.

        If ``top_n_by_pagerank`` is set, only those top-N nodes (and the
        edges between them) are loaded — useful for previewing very large
        graphs.
        """
        if top_n_by_pagerank is not None:
            ids_rows = self._query_all(
                "MATCH (c:Concept {source_label: $sl}) "
                "WHERE c.pagerank IS NOT NULL "
                "RETURN c.id AS id ORDER BY c.pagerank DESC LIMIT $n",
                sl=self.source_label, n=int(top_n_by_pagerank),
            )
            ids = [r["id"] for r in ids_rows]
            if not ids:
                return nx.DiGraph()
            return self._build_subgraph_from_ids(ids, min_weight=1)
        return self._build_full_graph()

    def _build_full_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for r in self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "RETURN c.id AS id, c.label AS label, c.concept_type AS type, "
            "       c.source_type AS source_type, c.frequency AS frequency, "
            "       c.community AS community, c.pagerank AS pagerank, "
            "       c.betweenness AS betweenness",
            sl=self.source_label,
        ):
            _add_node(G, r)
        for r in self._query_all(
            "MATCH (a:Concept {source_label: $sl})"
            "-[r:RELATED]->(b:Concept {source_label: $sl}) "
            "RETURN a.id AS source, b.id AS target, "
            "       r.weight AS weight, r.types AS types, "
            "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
            "       r.verb_count AS verb_count, r.verb_list AS verb_list",
            sl=self.source_label,
        ):
            _add_edge(G, r)
        return G

    def _build_subgraph_from_ids(
        self, ids: list[str], min_weight: int = 1
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        for r in self._query_all(
            "MATCH (c:Concept {source_label: $sl}) "
            "WHERE c.id IN $ids "
            "RETURN c.id AS id, c.label AS label, c.concept_type AS type, "
            "       c.source_type AS source_type, c.frequency AS frequency, "
            "       c.community AS community, c.pagerank AS pagerank, "
            "       c.betweenness AS betweenness",
            sl=self.source_label, ids=ids,
        ):
            _add_node(G, r)
        for r in self._query_all(
            "MATCH (a:Concept {source_label: $sl})"
            "-[r:RELATED]->(b:Concept {source_label: $sl}) "
            "WHERE a.id IN $ids AND b.id IN $ids "
            "  AND coalesce(r.weight, 1) >= $w "
            "RETURN a.id AS source, b.id AS target, "
            "       r.weight AS weight, r.types AS types, "
            "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
            "       r.verb_count AS verb_count, r.verb_list AS verb_list",
            sl=self.source_label, ids=ids, w=int(min_weight),
        ):
            _add_edge(G, r)
        return G

    # ── Semantic search ────────────────────────────────────────────
    def semantic_search_concepts(
        self,
        query_vec: list[float],
        k: int = 5,
        index_name: str | None = None,
    ) -> pd.DataFrame:
        """Top-k concepts by cosine similarity to a pre-computed embedding.

        The reader is embedding-model-agnostic — callers provide the vector
        themselves (typically from the same ``sentence-transformers`` model
        used to populate the index).
        """
        from src import config as _cfg

        idx = index_name or _cfg.NEO4J_VECTOR_INDEX
        try:
            rows = self._query_all(
                "CALL db.index.vector.queryNodes($idx, $k, $vec) "
                "YIELD node, score "
                "WHERE node.source_label = $sl "
                "RETURN node.id AS node_id, node.label AS label, "
                "       node.concept_type AS type, "
                "       node.source_type AS source_type, score",
                idx=idx, k=int(k), vec=list(query_vec), sl=self.source_label,
            )
        except Exception as e:
            logger.debug("semantic_search_concepts failed: %s", e)
            return pd.DataFrame(
                columns=["node_id", "label", "type", "source_type", "score"]
            )
        return pd.DataFrame(
            rows, columns=["node_id", "label", "type", "source_type", "score"]
        )

    def semantic_search_edges(
        self,
        query_vec: list[float],
        k: int = 5,
        index_name: str | None = None,
    ) -> pd.DataFrame:
        """Top-k edges by cosine similarity to a pre-computed embedding."""
        from src import config as _cfg

        idx = index_name or _cfg.NEO4J_EDGE_VECTOR_INDEX
        try:
            rows = self._query_all(
                "CALL db.index.vector.queryRelationships($idx, $k, $vec) "
                "YIELD relationship, score "
                "WHERE relationship.source_label = $sl "
                "RETURN startNode(relationship).label AS source_label, "
                "       endNode(relationship).label   AS target_label, "
                "       relationship.top_verb         AS verb, "
                "       relationship.weight           AS weight, "
                "       relationship.sentiment        AS sentiment, "
                "       score",
                idx=idx, k=int(k), vec=list(query_vec), sl=self.source_label,
            )
        except Exception as e:
            logger.debug("semantic_search_edges failed: %s", e)
            return pd.DataFrame(
                columns=["source_label", "target_label", "verb",
                         "weight", "sentiment", "score"],
            )
        return pd.DataFrame(
            rows,
            columns=["source_label", "target_label", "verb",
                     "weight", "sentiment", "score"],
        )


# ── Helpers ────────────────────────────────────────────────────────
def _add_node(G: nx.DiGraph, r: dict) -> None:
    nid = r["id"]
    ctype = r.get("type") or "concept"
    G.add_node(
        nid,
        label=r.get("label") or nid,
        # GraphBuilder uses ``type``; Neo4jStore writes ``concept_type``.
        # Populate both so every existing consumer reads the same value.
        type=ctype,
        concept_type=ctype,
        source_type=r.get("source_type") or "unknown",
        frequency=int(r.get("frequency") or 0),
        community=(
            int(r["community"]) if r.get("community") is not None else -1
        ),
        pagerank=float(r.get("pagerank") or 0.0),
        betweenness=float(r.get("betweenness") or 0.0),
    )


def _add_edge(G: nx.DiGraph, r: dict) -> None:
    u, v = r["source"], r["target"]
    if u not in G or v not in G:
        return
    attrs: dict[str, Any] = {
        "weight": float(r.get("weight") or 1.0),
        "types": _parse_types(r.get("types")),
        "top_verb": r.get("top_verb"),
        "verb_count": int(r.get("verb_count") or 0),
        "verb_list": list(r.get("verb_list") or []),
    }
    if r.get("sentiment") is not None:
        try:
            attrs["sentiment"] = float(r["sentiment"])
        except (TypeError, ValueError):
            pass
    G.add_edge(u, v, **attrs)


def _parse_types(value: Any) -> set[str]:
    """Reverse the CSV serialisation that ``Neo4jStore`` applies to ``r.types``."""
    if value is None:
        return set()
    if isinstance(value, set):
        return {str(x) for x in value if x}
    if isinstance(value, (list, tuple)):
        return {str(x) for x in value if x}
    if isinstance(value, str):
        return {p.strip() for p in value.split(",") if p.strip()}
    return {str(value)}
