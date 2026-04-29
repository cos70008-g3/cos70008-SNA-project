"""Graph context retrieval for the Stage-8 chatbot.

Wraps the annotated ``networkx.DiGraph`` + partition + centrality DataFrame
that the pipeline produces and exposes read-only helpers that return
plain-text paragraphs ready to inject into an LLM prompt.
"""

from __future__ import annotations

from collections import Counter
from difflib import get_close_matches

import networkx as nx
import pandas as pd


class GraphContext:
    """Read-only, prompt-friendly view over a conceptual network."""

    def __init__(
        self,
        G: nx.DiGraph,
        partition: dict[str, int] | None = None,
        centrality_df: pd.DataFrame | None = None,
        source_label: str = "combined",
    ):
        self.G = G
        self.partition = partition or {}
        self.centrality_df = (
            centrality_df
            if centrality_df is not None
            else pd.DataFrame(columns=["node_id", "label", "pagerank"])
        )
        self.source_label = source_label
        self._label_to_id = {
            self.G.nodes[n].get("label", n): n for n in self.G.nodes()
        }

    # ── Label resolution ───────────────────────────────────────────
    def resolve(self, label: str) -> str | None:
        """Best-effort map from user-typed label to a node id."""
        if label in self.G.nodes():
            return label
        if label in self._label_to_id:
            return self._label_to_id[label]
        lower_map = {k.lower(): v for k, v in self._label_to_id.items()}
        if label.lower() in lower_map:
            return lower_map[label.lower()]
        match = get_close_matches(label.lower(), list(lower_map.keys()), n=1, cutoff=0.7)
        return lower_map[match[0]] if match else None

    def all_labels(self) -> list[str]:
        return list(self._label_to_id.keys())

    # ── Single-concept description ─────────────────────────────────
    def describe_concept(self, label: str, top_neighbours: int = 8) -> str:
        nid = self.resolve(label)
        if nid is None:
            return f"No concept matching '{label}' was found in the network."

        data = self.G.nodes[nid]
        display = data.get("label", nid)
        freq = data.get("frequency", "?")
        ctype = data.get("concept_type") or data.get("type") or "concept"
        src = data.get("source_type", "n/a")
        community = self.partition.get(nid, data.get("community", -1))
        pr = data.get("pagerank")
        btwn = data.get("betweenness")

        lines = [
            f"Concept: {display}",
            f"  type: {ctype}; source: {src}; community: {community}; frequency: {freq}",
        ]
        if pr is not None:
            lines.append(
                f"  centrality: pagerank={pr:.4f}"
                + (f", betweenness={btwn:.4f}" if btwn is not None else "")
            )

        neighbours = []
        for _, v, edata in self.G.out_edges(nid, data=True):
            neighbours.append((v, edata, "out"))
        for u, _, edata in self.G.in_edges(nid, data=True):
            neighbours.append((u, edata, "in"))
        neighbours.sort(key=lambda x: x[1].get("weight", 1), reverse=True)

        if neighbours:
            lines.append(f"  top neighbours (up to {top_neighbours}):")
            for n_id, edata, direction in neighbours[:top_neighbours]:
                lbl = self.G.nodes[n_id].get("label", n_id)
                w = edata.get("weight", 1)
                arrow = "→" if direction == "out" else "←"
                types = edata.get("types") or edata.get("type") or ""
                if isinstance(types, set):
                    types = ",".join(sorted(types))
                sent = edata.get("sentiment")
                extras = f", types={types}" if types else ""
                if sent is not None:
                    extras += f", sentiment={sent:+.2f}"
                lines.append(f"    {display} {arrow} {lbl} (weight={w}{extras})")
        else:
            lines.append("  no neighbours in the network.")
        return "\n".join(lines)

    # ── Community description ──────────────────────────────────────
    def describe_community(self, community_id: int, top_n: int = 10) -> str:
        members = [n for n, cid in self.partition.items() if cid == community_id]
        if not members:
            return f"Community {community_id} has no members in the partition."

        ranked = sorted(
            members,
            key=lambda n: self.G.nodes[n].get("pagerank", 0),
            reverse=True,
        )
        top = [self.G.nodes[n].get("label", n) for n in ranked[:top_n]]
        sub = self.G.subgraph(members)
        density = nx.density(sub)
        lines = [
            f"Community {community_id}: {len(members)} concepts, internal density={density:.4f}",
            f"  top concepts by PageRank: {', '.join(top)}",
        ]

        degs = {n: sub.degree(n) for n in sub.nodes()}
        brokers = sorted(degs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if brokers:
            lines.append(
                "  hubs inside community: "
                + ", ".join(
                    f"{self.G.nodes[n].get('label', n)} (deg={d})"
                    for n, d in brokers
                )
            )
        return "\n".join(lines)

    # ── Path ───────────────────────────────────────────────────────
    def shortest_path(self, a: str, b: str) -> str:
        n1, n2 = self.resolve(a), self.resolve(b)
        if n1 is None or n2 is None:
            missing = a if n1 is None else b
            return f"Concept '{missing}' was not found in the network."
        try:
            path = nx.shortest_path(self.G.to_undirected(), n1, n2)
        except nx.NetworkXNoPath:
            return f"No path between '{a}' and '{b}' in the network."
        except nx.NodeNotFound:
            return f"Either '{a}' or '{b}' is not in the network."

        steps: list[str] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edata = self.G.edges[u, v]
            elif self.G.has_edge(v, u):
                edata = self.G.edges[v, u]
            else:
                edata = {}
            types = edata.get("types") or edata.get("type") or ""
            if isinstance(types, set):
                types = ",".join(sorted(types))
            steps.append(
                f"{self.G.nodes[u].get('label', u)} --[{types or 'related'}]--> "
                f"{self.G.nodes[v].get('label', v)}"
            )
        return f"Shortest path ({len(path) - 1} hops): " + " ; ".join(steps)

    # ── Top concepts ───────────────────────────────────────────────
    def top_concepts(self, metric: str = "pagerank", n: int = 10) -> str:
        df = self.centrality_df
        if df.empty or metric not in df.columns:
            return f"No centrality data available for metric '{metric}'."
        head = df.sort_values(metric, ascending=False).head(n)
        lines = [f"Top {n} concepts by {metric}:"]
        for i, (_, row) in enumerate(head.iterrows(), 1):
            lines.append(f"  {i}. {row['label']} ({metric}={row[metric]:.4f})")
        return "\n".join(lines)

    # ── Cross-source comparison ────────────────────────────────────
    def compare_sources(self) -> str:
        counts = Counter(
            self.G.nodes[n].get("source_type", "unknown") for n in self.G.nodes()
        )
        total = sum(counts.values()) or 1
        lines = [
            f"Source distribution across {total} concepts:",
        ]
        for src, c in sorted(counts.items()):
            lines.append(f"  {src}: {c} ({c / total:.1%})")

        both = [
            self.G.nodes[n].get("label", n)
            for n in self.G.nodes()
            if self.G.nodes[n].get("source_type") == "both"
        ]
        if both:
            lines.append("  shared concepts: " + ", ".join(both[:10]))
        return "\n".join(lines)

    # ── Overall summary ────────────────────────────────────────────
    def graph_summary(self) -> str:
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        n_comms = len(set(self.partition.values())) if self.partition else 0
        density = nx.density(self.G) if n_nodes > 1 else 0.0

        lines = [
            f"Conceptual network ({self.source_label}):",
            f"  {n_nodes} concepts, {n_edges} relationships, "
            f"{n_comms} communities, density={density:.4f}",
        ]
        if not self.centrality_df.empty:
            top = self.centrality_df.sort_values("pagerank", ascending=False).head(5)
            lines.append(
                "  most central concepts: "
                + ", ".join(top["label"].astype(str).tolist())
            )
        return "\n".join(lines)
