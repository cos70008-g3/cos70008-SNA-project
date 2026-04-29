"""Cypher-backed GraphContext implementation.

Mirrors the public surface of :class:`src.extensions.graph_context.GraphContext`
so the Stage-8 chatbot and Streamlit Chat tab can swap backends without
touching their code. Expensive graph operations (path finding, neighbour
ranking, aggregations) are executed server-side as Cypher queries against
Neo4j; a tiny in-memory label index is kept locally to satisfy the router's
``gc.G.nodes[nid]["label"]`` access pattern.
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


class Neo4jGraphContext:
    """Read-only, prompt-friendly view over a Concept subgraph stored in Neo4j."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str = "combined",
    ):
        self.store = store
        self.source_label = source_label
        # Thin NX shim so chatbot.QueryRouter can still read
        # ``gc.G.nodes[nid].get("label", nid)`` without special-casing.
        self.G: nx.DiGraph = nx.DiGraph()
        self._label_to_id: dict[str, str] = {}
        self.centrality_df: pd.DataFrame = pd.DataFrame(
            columns=["node_id", "label", "pagerank"]
        )
        self.partition: dict[str, int] = {}
        self._refresh_indexes()

    # ── Internal cache ─────────────────────────────────────────────
    def _refresh_indexes(self) -> None:
        """Pull minimal node metadata so the router can look up labels."""
        rows: list[dict] = []
        with self.store.session() as s:
            result = s.run(
                "MATCH (c:Concept {source_label: $s}) "
                "RETURN c.id AS id, c.label AS label, c.concept_type AS type, "
                "       c.source_type AS source_type, c.community AS community, "
                "       c.frequency AS frequency, c.pagerank AS pagerank, "
                "       c.betweenness AS betweenness",
                s=self.source_label,
            )
            for rec in result:
                rows.append(dict(rec))

        self.G.clear()
        self._label_to_id.clear()
        for r in rows:
            nid = r["id"]
            self.G.add_node(
                nid,
                label=r.get("label") or nid,
                concept_type=r.get("type"),
                source_type=r.get("source_type"),
                community=r.get("community"),
                frequency=r.get("frequency"),
                pagerank=r.get("pagerank"),
                betweenness=r.get("betweenness"),
            )
            lbl = r.get("label") or nid
            self._label_to_id[lbl] = nid
            self.partition[nid] = int(r.get("community") or -1)

        if rows:
            self.centrality_df = pd.DataFrame(
                [
                    {
                        "node_id": r["id"],
                        "label": r.get("label") or r["id"],
                        "pagerank": float(r.get("pagerank") or 0.0),
                        "betweenness": float(r.get("betweenness") or 0.0),
                    }
                    for r in rows
                ]
            )

    # ── Label resolution ───────────────────────────────────────────
    def resolve(self, label: str) -> str | None:
        if label in self.G.nodes():
            return label
        if label in self._label_to_id:
            return self._label_to_id[label]
        lower = {k.lower(): v for k, v in self._label_to_id.items()}
        if label.lower() in lower:
            return lower[label.lower()]
        match = get_close_matches(label.lower(), list(lower.keys()), n=1, cutoff=0.7)
        return lower[match[0]] if match else None

    def all_labels(self) -> list[str]:
        return list(self._label_to_id.keys())

    # ── Cypher helpers ─────────────────────────────────────────────
    def _query_one(self, query: str, **params):
        with self.store.session() as s:
            rec = s.run(query, **params).single()
            return dict(rec) if rec else None

    def _query_all(self, query: str, **params) -> list[dict]:
        with self.store.session() as s:
            return [dict(r) for r in s.run(query, **params)]

    # ── Single-concept description ─────────────────────────────────
    def describe_concept(self, label: str, top_neighbours: int = 8) -> str:
        nid = self.resolve(label)
        if nid is None:
            return f"No concept matching '{label}' was found in the network."

        node = self._query_one(
            "MATCH (c:Concept {id: $id, source_label: $s}) "
            "RETURN c.label AS label, c.concept_type AS type, "
            "       c.source_type AS source_type, c.community AS community, "
            "       c.frequency AS frequency, c.pagerank AS pagerank, "
            "       c.betweenness AS betweenness",
            id=nid,
            s=self.source_label,
        )
        if not node:
            return f"Concept '{label}' is not stored under source_label '{self.source_label}'."

        out_nbs = self._query_all(
            "MATCH (c:Concept {id: $id, source_label: $s})-[r:RELATED]->(n:Concept) "
            "RETURN n.label AS label, r.weight AS weight, r.types AS types, "
            "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
            "       'out' AS direction "
            "ORDER BY r.weight DESC LIMIT $k",
            id=nid,
            s=self.source_label,
            k=top_neighbours,
        )
        in_nbs = self._query_all(
            "MATCH (n:Concept)-[r:RELATED]->(c:Concept {id: $id, source_label: $s}) "
            "RETURN n.label AS label, r.weight AS weight, r.types AS types, "
            "       r.sentiment AS sentiment, r.top_verb AS top_verb, "
            "       'in' AS direction "
            "ORDER BY r.weight DESC LIMIT $k",
            id=nid,
            s=self.source_label,
            k=top_neighbours,
        )
        neighbours = sorted(
            out_nbs + in_nbs,
            key=lambda r: r.get("weight") or 0,
            reverse=True,
        )[:top_neighbours]

        display = node.get("label") or nid
        lines = [
            f"Concept: {display}",
            f"  type: {node.get('type') or 'concept'}; "
            f"source: {node.get('source_type') or 'n/a'}; "
            f"community: {node.get('community', -1)}; "
            f"frequency: {node.get('frequency', 0)}",
        ]
        pr = node.get("pagerank")
        if pr is not None:
            btwn = node.get("betweenness")
            lines.append(
                f"  centrality: pagerank={pr:.4f}"
                + (f", betweenness={btwn:.4f}" if btwn is not None else "")
            )

        if neighbours:
            lines.append(f"  top neighbours (up to {top_neighbours}):")
            for n in neighbours:
                verb = n.get("top_verb")
                if n["direction"] == "out":
                    if verb:
                        arrow = f"-- {verb} -->"
                    else:
                        arrow = "→"
                else:
                    if verb:
                        arrow = f"<-- {verb} --"
                    else:
                        arrow = "←"
                extras = ""
                if n.get("types"):
                    extras += f", types={n['types']}"
                if n.get("sentiment") is not None:
                    extras += f", sentiment={float(n['sentiment']):+.2f}"
                lines.append(
                    f"    {display} {arrow} {n['label']} "
                    f"(weight={n.get('weight', 1)}{extras})"
                )
        else:
            lines.append("  no neighbours in the network.")
        return "\n".join(lines)

    # ── Relationship vector query (edge search) ────────────────────
    def relations_matching(
        self,
        query_vec: list[float],
        k: int = 10,
        index_name: str | None = None,
    ) -> list[dict]:
        """Return the top ``k`` :RELATED edges matching a pre-computed vector.

        Convenience wrapper around ``db.index.vector.queryRelationships`` for
        callers that already hold an embedding (e.g. the chatbot router).
        """
        from src import config as _cfg

        idx = index_name or _cfg.NEO4J_EDGE_VECTOR_INDEX
        try:
            rows = self._query_all(
                "CALL db.index.vector.queryRelationships($idx, $k, $vec) "
                "YIELD relationship, score "
                "WHERE relationship.source_label = $sl "
                "RETURN startNode(relationship).label AS src, "
                "       endNode(relationship).label   AS tgt, "
                "       relationship.top_verb         AS verb, "
                "       relationship.weight           AS weight, "
                "       relationship.sentiment        AS sentiment, "
                "       score",
                idx=idx,
                k=int(k),
                vec=list(query_vec),
                sl=self.source_label,
            )
        except Exception as e:
            logger.debug("relations_matching failed: %s", e)
            return []
        out: list[dict] = []
        for r in rows:
            verb = r.get("verb")
            src = r.get("src")
            tgt = r.get("tgt")
            out.append(
                {
                    "source_label": src,
                    "target_label": tgt,
                    "verb": verb,
                    "weight": float(r.get("weight") or 0),
                    "sentiment": r.get("sentiment"),
                    "score": float(r.get("score") or 0),
                    "description": (
                        f"{src} {verb} {tgt}." if verb
                        else f"{src} co-occurs with {tgt}."
                    ),
                }
            )
        return out

    # ── Community ──────────────────────────────────────────────────
    def describe_community(self, community_id: int, top_n: int = 10) -> str:
        rows = self._query_all(
            "MATCH (c:Concept {community: $cid, source_label: $s}) "
            "RETURN c.id AS id, c.label AS label, "
            "       coalesce(c.pagerank, 0.0) AS pagerank "
            "ORDER BY pagerank DESC",
            cid=int(community_id),
            s=self.source_label,
        )
        if not rows:
            return f"Community {community_id} has no members in the partition."

        density_row = self._query_one(
            "MATCH (c:Concept {community: $cid, source_label: $s}) "
            "WITH collect(c) AS members "
            "UNWIND members AS a "
            "UNWIND members AS b "
            "WITH members, a, b WHERE a <> b "
            "OPTIONAL MATCH (a)-[r:RELATED]->(b) "
            "WITH size(members) AS n, count(r) AS e "
            "RETURN n, e, CASE WHEN n < 2 THEN 0.0 "
            "             ELSE toFloat(e) / (n * (n - 1)) END AS density",
            cid=int(community_id),
            s=self.source_label,
        )
        n = density_row["n"] if density_row else len(rows)
        density = float(density_row["density"]) if density_row else 0.0
        top = [r["label"] for r in rows[:top_n]]

        degs = self._query_all(
            "MATCH (c:Concept {community: $cid, source_label: $s}) "
            "OPTIONAL MATCH (c)-[r:RELATED]-(:Concept {community: $cid, source_label: $s}) "
            "WITH c, count(r) AS deg "
            "RETURN c.label AS label, deg "
            "ORDER BY deg DESC LIMIT 3",
            cid=int(community_id),
            s=self.source_label,
        )

        lines = [
            f"Community {community_id}: {n} concepts, internal density={density:.4f}",
            f"  top concepts by PageRank: {', '.join(top)}",
        ]
        if degs:
            lines.append(
                "  hubs inside community: "
                + ", ".join(f"{d['label']} (deg={d['deg']})" for d in degs)
            )
        return "\n".join(lines)

    # ── Shortest path ──────────────────────────────────────────────
    def shortest_path(self, a: str, b: str) -> str:
        n1, n2 = self.resolve(a), self.resolve(b)
        if n1 is None or n2 is None:
            missing = a if n1 is None else b
            return f"Concept '{missing}' was not found in the network."
        row = self._query_one(
            "MATCH (s:Concept {id: $a, source_label: $sl}) "
            "MATCH (t:Concept {id: $b, source_label: $sl}) "
            "MATCH p = shortestPath((s)-[:RELATED*..10]-(t)) "
            "RETURN [n IN nodes(p) | n.label] AS labels, "
            "       [r IN relationships(p) | r.types] AS edge_types, "
            "       length(p) AS hops",
            a=n1,
            b=n2,
            sl=self.source_label,
        )
        if not row:
            return f"No path between '{a}' and '{b}' in the network."
        labels: list[str] = row["labels"] or []
        edge_types: list[str] = row["edge_types"] or []
        hops = row["hops"]
        steps: list[str] = []
        for i in range(len(labels) - 1):
            t = edge_types[i] if i < len(edge_types) else ""
            steps.append(f"{labels[i]} --[{t or 'related'}]--> {labels[i + 1]}")
        return f"Shortest path ({hops} hops): " + " ; ".join(steps)

    # ── Top concepts ───────────────────────────────────────────────
    def top_concepts(self, metric: str = "pagerank", n: int = 10) -> str:
        if metric not in {"pagerank", "betweenness", "frequency"}:
            return f"No centrality data available for metric '{metric}'."
        rows = self._query_all(
            f"MATCH (c:Concept {{source_label: $s}}) "
            f"WHERE c.{metric} IS NOT NULL "
            f"RETURN c.label AS label, c.{metric} AS value "
            f"ORDER BY value DESC LIMIT $n",
            s=self.source_label,
            n=int(n),
        )
        if not rows:
            return f"No centrality data available for metric '{metric}'."
        lines = [f"Top {n} concepts by {metric}:"]
        for i, r in enumerate(rows, 1):
            lines.append(f"  {i}. {r['label']} ({metric}={float(r['value']):.4f})")
        return "\n".join(lines)

    # ── Cross-source comparison ────────────────────────────────────
    def compare_sources(self) -> str:
        rows = self._query_all(
            "MATCH (c:Concept {source_label: $s}) "
            "RETURN coalesce(c.source_type, 'unknown') AS source_type, "
            "       count(*) AS cnt",
            s=self.source_label,
        )
        total = sum(int(r["cnt"]) for r in rows) or 1
        lines = [f"Source distribution across {total} concepts:"]
        for r in sorted(rows, key=lambda x: x["source_type"]):
            lines.append(
                f"  {r['source_type']}: {int(r['cnt'])} ({int(r['cnt']) / total:.1%})"
            )

        both = self._query_all(
            "MATCH (c:Concept {source_label: $s, source_type: 'both'}) "
            "RETURN c.label AS label LIMIT 10",
            s=self.source_label,
        )
        if both:
            lines.append(
                "  shared concepts: " + ", ".join(r["label"] for r in both)
            )
        return "\n".join(lines)

    # ── Graph summary ──────────────────────────────────────────────
    def graph_summary(self) -> str:
        row = self._query_one(
            "MATCH (c:Concept {source_label: $s}) "
            "OPTIONAL MATCH (c)-[r:RELATED {source_label: $s}]->() "
            "WITH count(DISTINCT c) AS n, count(r) AS e, "
            "     size(collect(DISTINCT c.community)) AS comms "
            "RETURN n, e, comms",
            s=self.source_label,
        )
        if not row:
            return "Conceptual network (empty)."
        n = int(row["n"] or 0)
        e = int(row["e"] or 0)
        comms = int(row["comms"] or 0)
        density = (e / (n * (n - 1))) if n > 1 else 0.0

        lines = [
            f"Conceptual network ({self.source_label}):",
            f"  {n} concepts, {e} relationships, "
            f"{comms} communities, density={density:.4f}",
        ]
        top = self._query_all(
            "MATCH (c:Concept {source_label: $s}) "
            "WHERE c.pagerank IS NOT NULL "
            "RETURN c.label AS label ORDER BY c.pagerank DESC LIMIT 5",
            s=self.source_label,
        )
        if top:
            lines.append(
                "  most central concepts: "
                + ", ".join(r["label"] for r in top)
            )
        return "\n".join(lines)
