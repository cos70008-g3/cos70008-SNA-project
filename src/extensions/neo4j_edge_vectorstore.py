"""Neo4j-backed edge vector store for the Stage-8 chatbot.

Mirrors :class:`src.extensions.graph_vectorstore.GraphEdgeVectorStore` but
delegates similarity search to Neo4j's native relationship vector index
(``db.index.vector.queryRelationships``, Neo4j 5.18+).

The chatbot router treats either backend interchangeably so the project can
be demoed with or without a running Neo4j instance. When the driver, model,
or server are unavailable, ``available`` is set to ``False`` and
:meth:`search` returns ``[]`` — the chatbot falls back to node-level vector
search and fuzzy label matching without error.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


class Neo4jEdgeVectorStore:
    """Semantic search over ``[:RELATED {embedding}]`` edges via Cypher."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str = "combined",
        model_name: str | None = None,
        index_name: str | None = None,
    ):
        from src import config as _cfg

        self.store = store
        self.source_label = source_label
        self.model_name = model_name or _cfg.EMBEDDING_MODEL
        self.index_name = index_name or _cfg.NEO4J_EDGE_VECTOR_INDEX
        self.available = False
        self._model = None
        self._count = 0

        try:
            self._build()
        except Exception as e:
            logger.warning("Neo4jEdgeVectorStore disabled: %s", e)
            self.available = False

    # ── Initialisation ─────────────────────────────────────────────
    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to enable edge semantic search."
            ) from e

        # Check that there are embedded edges for this source_label.
        with self.store.session() as s:
            rec = s.run(
                "MATCH ()-[r:RELATED {source_label: $sl}]->() "
                "WHERE r.embedding IS NOT NULL "
                "RETURN count(r) AS n",
                sl=self.source_label,
            ).single()
            n_with_vec = int(rec["n"]) if rec else 0
        if n_with_vec == 0:
            raise RuntimeError(
                f"No edges with embeddings for source_label='{self.source_label}'. "
                "Run the pipeline with --neo4j (and ensure sentence-transformers "
                "is installed) so edge embeddings are populated."
            )

        self._model = SentenceTransformer(self.model_name)
        self._count = n_with_vec
        self.available = True

    # ── Query ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-k edge matches.

        Each dict has keys: ``source_label``, ``target_label``, ``verb``,
        ``weight``, ``sentiment``, ``score``, ``description``.
        """
        if not self.available or self._model is None:
            return []
        try:
            vec = (
                self._model.encode([query], convert_to_numpy=True)
                .astype("float32")[0]
                .tolist()
            )
            with self.store.session() as s:
                rows = s.run(
                    "CALL db.index.vector.queryRelationships($idx, $k, $vec) "
                    "YIELD relationship, score "
                    "WHERE relationship.source_label = $sl "
                    "RETURN startNode(relationship).label AS src, "
                    "       endNode(relationship).label   AS tgt, "
                    "       relationship.top_verb         AS verb, "
                    "       relationship.weight           AS weight, "
                    "       relationship.sentiment        AS sentiment, "
                    "       score",
                    idx=self.index_name,
                    k=int(k),
                    vec=vec,
                    sl=self.source_label,
                )
                out: list[dict[str, Any]] = []
                for r in rows:
                    verb = r.get("verb")
                    src = r.get("src")
                    tgt = r.get("tgt")
                    if verb:
                        description = f"{src} {verb} {tgt}."
                    else:
                        description = f"{src} co-occurs with {tgt}."
                    out.append(
                        {
                            "source_label": src,
                            "target_label": tgt,
                            "verb": verb,
                            "weight": float(r.get("weight") or 0),
                            "sentiment": r.get("sentiment"),
                            "score": float(r.get("score") or 0),
                            "description": description,
                        }
                    )
                return out
        except Exception as e:
            logger.debug("Neo4jEdgeVectorStore.search failed: %s", e)
            return []

    def __len__(self) -> int:
        return int(self._count or 0)
