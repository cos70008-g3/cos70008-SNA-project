"""Build NetworkX graphs from extracted concepts and relationships."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import MIN_EDGE_WEIGHT
from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import Relationship


class GraphBuilder:
    """Construct a conceptual network from concepts and relationships."""

    def __init__(self, min_edge_weight: int = MIN_EDGE_WEIGHT):
        self.min_edge_weight = min_edge_weight

    def build(
        self,
        concepts: list[Concept],
        relationships: list[Relationship],
    ) -> nx.DiGraph:
        """Build a directed graph with typed, weighted edges.

        Uses DiGraph rather than MultiDiGraph so that parallel edges of
        different types are aggregated (summed weights) for simpler analysis.
        The edge attribute 'types' stores a set of all relationship types.
        """
        G = nx.DiGraph()

        for c in concepts:
            G.add_node(c.id, label=c.label, type=c.type,
                        frequency=c.frequency,
                        source_docs=list(c.source_docs),
                        source_type=c.source_type,
                        source_origins=list(c.source_origins))

        for r in relationships:
            if r.weight < self.min_edge_weight:
                continue
            if r.source_id not in G or r.target_id not in G:
                continue

            if G.has_edge(r.source_id, r.target_id):
                G[r.source_id][r.target_id]["weight"] += r.weight
                G[r.source_id][r.target_id]["types"].add(r.type)
                G[r.source_id][r.target_id].setdefault("verbs", Counter()).update(r.verbs)
            else:
                G.add_edge(
                    r.source_id, r.target_id,
                    weight=r.weight,
                    types={r.type},
                    directed=r.directed,
                    source_docs=list(r.source_docs),
                    verbs=Counter(r.verbs),
                )

            if not r.directed and not G.has_edge(r.target_id, r.source_id):
                G.add_edge(
                    r.target_id, r.source_id,
                    weight=r.weight,
                    types={r.type},
                    directed=False,
                    source_docs=list(r.source_docs),
                    verbs=Counter(r.verbs),
                )

        return G

    @staticmethod
    def to_undirected(G: nx.DiGraph) -> nx.Graph:
        """Convert to undirected graph, summing edge weights."""
        return G.to_undirected()

    @staticmethod
    def export_graphml(G: nx.DiGraph, path: str | Path) -> None:
        """Export graph to GraphML format."""
        H = G.copy()
        for _, _, data in H.edges(data=True):
            if "types" in data:
                data["types"] = ",".join(data["types"])
            if "source_docs" in data:
                data["source_docs"] = ",".join(str(d) for d in data["source_docs"])
            if "verbs" in data:
                verbs_counter = data["verbs"] or Counter()
                top = verbs_counter.most_common(1)
                data["top_verb"] = top[0][0] if top else ""
                data["verb_count"] = int(top[0][1]) if top else 0
                data["verbs"] = ",".join(v for v, _ in verbs_counter.most_common())
        for _, data in H.nodes(data=True):
            if "source_docs" in data:
                data["source_docs"] = ",".join(str(d) for d in data["source_docs"])
            if "source_origins" in data:
                data["source_origins"] = ",".join(str(o) for o in data["source_origins"])
        nx.write_graphml(H, str(path))

    @staticmethod
    def export_edge_csv(G: nx.DiGraph, path: str | Path) -> None:
        """Export edge list as CSV."""
        rows = []
        for u, v, data in G.edges(data=True):
            verbs_counter: Counter = data.get("verbs") or Counter()
            top = verbs_counter.most_common(1)
            row = {
                "source": G.nodes[u].get("label", u),
                "target": G.nodes[v].get("label", v),
                "weight": data.get("weight", 1),
                "types": ",".join(data.get("types", set())),
                "source_type_from": G.nodes[u].get("source_type", "unknown"),
                "source_type_to": G.nodes[v].get("source_type", "unknown"),
                "top_verb": top[0][0] if top else "",
                "verb_count": int(top[0][1]) if top else 0,
            }
            if "sentiment" in data:
                row["sentiment"] = data["sentiment"]
                row["sentiment_label"] = data.get("sentiment_label", "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(str(path), index=False)

    @staticmethod
    def export_node_csv(G: nx.DiGraph, path: str | Path) -> None:
        """Export node list as CSV with source_type."""
        rows = []
        for node, data in G.nodes(data=True):
            rows.append({
                "node_id": node,
                "label": data.get("label", node),
                "type": data.get("type", "unknown"),
                "frequency": data.get("frequency", 0),
                "source_type": data.get("source_type", "unknown"),
            })
        pd.DataFrame(rows).to_csv(str(path), index=False)

    @staticmethod
    def summary(G: nx.DiGraph) -> dict:
        """Return basic graph statistics."""
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
            "components": nx.number_weakly_connected_components(G),
        }
