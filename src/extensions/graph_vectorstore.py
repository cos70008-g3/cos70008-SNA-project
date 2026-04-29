"""FAISS-backed semantic index over the graph's concepts and edges.

Optional dependency. If ``sentence-transformers`` or ``faiss-cpu`` are not
installed, :class:`GraphVectorStore` degrades gracefully: ``available``
is ``False`` and :meth:`search` returns an empty list, letting the
chatbot fall back to structured queries / fuzzy label matching.

:class:`GraphEdgeVectorStore` provides the edge-level equivalent and is
used by the Stage-8 chatbot to answer relational questions ("who
recommended X?", "what causes Y?") via semantic similarity on verb-aware
edge descriptions.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.extensions.graph_context import GraphContext


class GraphVectorStore:
    """Semantic search over node descriptions.

    Parameters
    ----------
    graph_context : GraphContext
        Provides concept descriptions for each node.
    model_name : str
        ``sentence-transformers`` model to use.
    """

    def __init__(
        self,
        graph_context: "GraphContext",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.graph_context = graph_context
        self.model_name = model_name
        self.available = False
        self._model = None
        self._index = None
        self._node_ids: list[str] = []
        self._descriptions: list[str] = []

        try:
            self._build()
        except Exception as e:
            # Any failure (missing deps, network, etc.) -> disable cleanly
            self._last_error = str(e)
            self.available = False

    # ── Build index ────────────────────────────────────────────────
    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to enable semantic search."
            ) from e
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Run `pip install faiss-cpu` to enable semantic search."
            ) from e

        self._model = SentenceTransformer(self.model_name)

        G = self.graph_context.G
        node_ids: list[str] = list(G.nodes())
        descriptions: list[str] = []
        for nid in node_ids:
            label = G.nodes[nid].get("label", nid)
            descriptions.append(self._short_description(label))
        if not descriptions:
            raise ValueError("No nodes in graph to index.")

        vectors = self._model.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        # normalise for cosine/IP equivalence
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._index = index
        self._node_ids = node_ids
        self._descriptions = descriptions
        self.available = True

    def _short_description(self, label: str) -> str:
        """Compact one-line node blurb good for embedding."""
        G = self.graph_context.G
        nid = self.graph_context.resolve(label) or label
        if nid not in G.nodes():
            return label
        data = G.nodes[nid]
        ctype = data.get("concept_type") or data.get("type") or "concept"
        src = data.get("source_type", "n/a")
        neighbours = []
        for _, v, edata in G.out_edges(nid, data=True):
            neighbours.append((G.nodes[v].get("label", v), edata.get("weight", 1)))
        for u, _, edata in G.in_edges(nid, data=True):
            neighbours.append((G.nodes[u].get("label", u), edata.get("weight", 1)))
        neighbours.sort(key=lambda x: x[1], reverse=True)
        top_nbs = ", ".join(lbl for lbl, _ in neighbours[:5])
        return (
            f"{label}. A {ctype} concept from {src} data."
            + (f" Related to: {top_nbs}." if top_nbs else "")
        )

    # ── Query ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (node_id, similarity) pairs, or [] if unavailable."""
        if not self.available or self._model is None or self._index is None:
            return []
        try:
            import faiss  # type: ignore

            vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            k = min(k, len(self._node_ids))
            scores, idx = self._index.search(vec, k)
            out: list[tuple[str, float]] = []
            for score, i in zip(scores[0], idx[0]):
                if i < 0:
                    continue
                out.append((self._node_ids[i], float(score)))
            return out
        except Exception:
            return []

    def __len__(self) -> int:
        return len(self._node_ids)


# ══════════════════════════════════════════════════════════════════
# Edge-level vector store
# ══════════════════════════════════════════════════════════════════


class GraphEdgeVectorStore:
    """Semantic search over edge descriptions (FAISS backend).

    Parity with :class:`src.extensions.neo4j_edge_vectorstore.Neo4jEdgeVectorStore`
    so the chatbot can use either backend transparently.

    Embeds the curated subset specified by the Edge Embedding plan:

    * all edges with non-empty ``verbs`` attribute (ACTION + CAUSATION)
    * plus the top ``top_n_association`` ASSOCIATION-only edges by weight.
    """

    def __init__(
        self,
        graph_context: "GraphContext",
        model_name: str = "all-MiniLM-L6-v2",
        top_n_association: int = 2000,
    ):
        self.graph_context = graph_context
        self.model_name = model_name
        self.top_n_association = top_n_association
        self.available = False
        self._model = None
        self._index = None
        self._edges: list[tuple[str, str]] = []
        self._descriptions: list[str] = []

        try:
            self._build()
        except Exception as e:
            self._last_error = str(e)
            self.available = False

    # ── Build index ────────────────────────────────────────────────
    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to enable edge semantic search."
            ) from e
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Run `pip install faiss-cpu` to enable edge semantic search."
            ) from e

        G = self.graph_context.G
        verb_edges: list[tuple[str, str, dict[str, Any]]] = []
        assoc_edges: list[tuple[str, str, dict[str, Any]]] = []
        for u, v, data in G.edges(data=True):
            verbs = data.get("verbs")
            has_verb = False
            if verbs:
                try:
                    has_verb = sum(verbs.values()) > 0
                except AttributeError:
                    has_verb = bool(verbs)
            if has_verb:
                verb_edges.append((u, v, data))
            else:
                assoc_edges.append((u, v, data))
        assoc_edges.sort(
            key=lambda tup: float(tup[2].get("weight", 0) or 0),
            reverse=True,
        )
        selected = verb_edges + assoc_edges[: max(0, int(self.top_n_association))]
        if not selected:
            raise ValueError("No edges in graph to index.")

        self._model = SentenceTransformer(self.model_name)

        descriptions = [_edge_description(G, u, v, d) for (u, v, d) in selected]

        vectors = self._model.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._index = index
        self._edges = [(str(u), str(v)) for (u, v, _d) in selected]
        self._descriptions = descriptions
        self.available = True

    # ── Query ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-k edge matches as dicts, or [] if unavailable.

        Each dict has keys: ``source_label``, ``target_label``, ``verb``,
        ``weight``, ``sentiment``, ``score``, ``description``.
        """
        if not self.available or self._model is None or self._index is None:
            return []
        try:
            import faiss  # type: ignore

            vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            k = min(k, len(self._edges))
            if k <= 0:
                return []
            scores, idx = self._index.search(vec, k)
            G = self.graph_context.G
            out: list[dict[str, Any]] = []
            for score, i in zip(scores[0], idx[0]):
                if i < 0:
                    continue
                u, v = self._edges[i]
                data = G.get_edge_data(u, v) or {}
                verbs = data.get("verbs") or Counter()
                try:
                    top_v = verbs.most_common(1)
                    top_verb = top_v[0][0] if top_v else None
                except AttributeError:
                    top_verb = None
                out.append(
                    {
                        "source_label": G.nodes[u].get("label", u),
                        "target_label": G.nodes[v].get("label", v),
                        "verb": top_verb,
                        "weight": float(data.get("weight", 0) or 0),
                        "sentiment": data.get("sentiment"),
                        "score": float(score),
                        "description": self._descriptions[i],
                    }
                )
            return out
        except Exception:
            return []

    def __len__(self) -> int:
        return len(self._edges)


# ── Shared helper ──────────────────────────────────────────────────
def _edge_description(G, u, v, data: dict[str, Any]) -> str:
    """Verb-aware edge description, identical to Neo4jStore._edge_description."""
    u_lbl = G.nodes[u].get("label", u)
    v_lbl = G.nodes[v].get("label", v)
    verbs = data.get("verbs")
    top_verb: str | None = None
    if verbs:
        try:
            ordered = verbs.most_common(1)
        except AttributeError:
            ordered = []
        if ordered:
            top_verb = str(ordered[0][0])
    if top_verb:
        sent = data.get("sentiment")
        tail = ""
        if sent is not None:
            try:
                sentf = float(sent)
                if sentf > 0.15:
                    tail = " (positive tone)"
                elif sentf < -0.15:
                    tail = " (negative tone)"
            except (TypeError, ValueError):
                tail = ""
        return f"{u_lbl} {top_verb} {v_lbl}{tail}."
    return f"{u_lbl} co-occurs with {v_lbl}."
