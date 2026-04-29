"""Neo4j-backed vector store for the Stage-8 chatbot.

Mirrors the contract of :class:`src.extensions.graph_vectorstore.GraphVectorStore`
so ``chatbot.QueryRouter`` can treat either backend interchangeably. Uses the
native ``db.index.vector.queryNodes`` procedure (Neo4j 5.11+).

The embedding model remains ``sentence-transformers`` — only the index
changes. If the driver / model / server are unavailable, ``available`` is
``False`` and :meth:`search` returns an empty list, letting the router fall
back to fuzzy label matching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


class Neo4jVectorStore:
    """Semantic search over ``:Concept(embedding)`` via Cypher."""

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
        self.index_name = index_name or _cfg.NEO4J_VECTOR_INDEX
        self.available = False
        self._model = None

        try:
            self._build()
        except Exception as e:
            logger.warning("Neo4jVectorStore disabled: %s", e)
            self.available = False

    # ── Initialisation ─────────────────────────────────────────────
    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to enable semantic search."
            ) from e

        # Ensure the vector index exists and has entries for this source_label.
        with self.store.session() as s:
            rec = s.run(
                "MATCH (c:Concept {source_label: $sl}) "
                "WHERE c.embedding IS NOT NULL RETURN count(c) AS n",
                sl=self.source_label,
            ).single()
            n_with_vec = int(rec["n"]) if rec else 0
        if n_with_vec == 0:
            raise RuntimeError(
                f"No nodes with embeddings for source_label='{self.source_label}'. "
                f"Run the pipeline with --neo4j first."
            )

        self._model = SentenceTransformer(self.model_name)
        self._count = n_with_vec
        self.available = True

    # ── Query ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
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
                    "CALL db.index.vector.queryNodes($idx, $k, $vec) "
                    "YIELD node, score "
                    "WHERE node.source_label = $sl "
                    "RETURN node.id AS id, score",
                    idx=self.index_name,
                    k=int(k),
                    vec=vec,
                    sl=self.source_label,
                )
                return [(r["id"], float(r["score"])) for r in rows]
        except Exception as e:
            logger.debug("Neo4jVectorStore.search failed: %s", e)
            return []

    def __len__(self) -> int:
        return getattr(self, "_count", 0)
