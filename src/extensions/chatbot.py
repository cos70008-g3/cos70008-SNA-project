"""Stage 8: LLM/RAG chatbot grounded in the conceptual network.

Components:
    - ``LLMProvider``          abstract text-completion interface
    - ``OpenAIProvider``       uses the official ``openai`` package
    - ``HuggingFaceAPIProvider``  uses ``huggingface_hub.InferenceClient``
    - ``HuggingFaceLocalProvider``  uses ``transformers`` pipeline
    - ``EchoProvider``          zero-dependency fallback: returns context
    - ``QueryRouter``           routes the user question to ``GraphContext``
                                methods / vector search
    - ``GraphChatbot``          orchestrator: route → retrieve → prompt → LLM

Only the chosen provider's package needs to be installed. Missing deps are
raised as clear ``ImportError`` messages so the caller can pick a different
provider or install the optional dependency.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src import config

if TYPE_CHECKING:
    from src.extensions.graph_context import GraphContext
    from src.extensions.graph_vectorstore import (
        GraphEdgeVectorStore,
        GraphVectorStore,
    )
    from src.extensions.neo4j_edge_vectorstore import Neo4jEdgeVectorStore


# ══════════════════════════════════════════════════════════════════
# LLM providers
# ══════════════════════════════════════════════════════════════════


class LLMProvider(ABC):
    """Text-completion interface the chatbot expects."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        ...


class EchoProvider(LLMProvider):
    """Zero-dependency fallback: returns retrieved context verbatim.

    Useful for unit tests and for environments where no LLM is configured.
    The chatbot remains functional (structured retrieval still works) but
    no free-form generation occurs.
    """

    name = "echo"

    def complete(self, system_prompt: str, user_message: str) -> str:
        return (
            "No LLM is configured — returning the retrieved graph context "
            "verbatim. Set `LLM_PROVIDER` in src/config.py or via environment "
            "variable to enable natural-language answers.\n\n"
            f"{user_message.strip()}"
        )


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible Chat Completions (``openai>=1.0``).

    Works with any provider that exposes the OpenAI Chat Completions API,
    including official OpenAI, Azure, and OpenAI-compatible gateways. Pass
    ``base_url`` to point at a different endpoint (used by
    :class:`DashScopeProvider` to talk to Aliyun's ``compatible-mode`` API).
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The OpenAI provider requires the `openai` package. "
                "Install it with `pip install openai>=1.0`."
            ) from e
        self._OpenAI = OpenAI
        self.model = model or config.LLM_MODEL
        self.temperature = (
            temperature if temperature is not None else config.LLM_TEMPERATURE
        )
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell or set "
                "config.OPENAI_API_KEY before constructing OpenAIProvider."
            )
        client_kwargs: dict = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""


class DashScopeProvider(LLMProvider):
    """Aliyun DashScope (Qwen) via the OpenAI-compatible endpoint.

    DashScope exposes a drop-in OpenAI Chat Completions API at
    ``https://dashscope.aliyuncs.com/compatible-mode/v1`` so we can reuse the
    same ``openai`` Python client — no new dependency required.

    Configure via:

    * ``DASHSCOPE_API_KEY``   — required
    * ``LLM_MODEL``           — defaults to ``qwen-turbo``; also supports
      ``qwen-plus``, ``qwen-max``, ``qwen2.5-72b-instruct``, etc.
    * ``DASHSCOPE_BASE_URL``  — override only if you have a private endpoint.
    """

    name = "dashscope"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The DashScope provider reuses the `openai` package "
                "(OpenAI-compatible mode). Install it with "
                "`pip install openai>=1.0`."
            ) from e
        key = api_key or config.DASHSCOPE_API_KEY
        if not key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Get a key from "
                "https://dashscope.console.aliyun.com/ and export it, "
                "or add it to your .env file."
            )
        # Prefer the explicit DashScope default if the user kept the OpenAI
        # default model name in LLM_MODEL.
        chosen_model = model or config.LLM_MODEL
        if chosen_model.startswith("gpt-"):
            chosen_model = "qwen-turbo"
        self.model = chosen_model
        self.temperature = (
            temperature if temperature is not None else config.LLM_TEMPERATURE
        )
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self._client = OpenAI(
            api_key=key,
            base_url=base_url or config.DASHSCOPE_BASE_URL,
        )

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""


class HuggingFaceAPIProvider(LLMProvider):
    """Uses ``huggingface_hub.InferenceClient`` (hosted inference endpoints)."""

    name = "huggingface_api"

    def __init__(
        self,
        model: str | None = None,
        api_token: str | None = None,
        max_tokens: int | None = None,
    ):
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The Hugging Face API provider requires `huggingface_hub`. "
                "Install it with `pip install huggingface_hub>=0.20`."
            ) from e
        self.model = model or config.LLM_MODEL
        token = api_token or config.HUGGINGFACE_API_TOKEN or None
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self._client = InferenceClient(model=self.model, token=token)

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            resp = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=config.LLM_TEMPERATURE,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Fallback for older HF endpoints without chat_completion
            prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
            return self._client.text_generation(
                prompt, max_new_tokens=self.max_tokens
            )


class HuggingFaceLocalProvider(LLMProvider):
    """Runs a local ``transformers`` text-generation pipeline."""

    name = "huggingface_local"

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The local Hugging Face provider requires `transformers`. "
                "Install it with `pip install transformers>=4.30`."
            ) from e
        self.model = model or config.LLM_MODEL
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self._pipe = pipeline("text-generation", model=self.model)

    def complete(self, system_prompt: str, user_message: str) -> str:
        prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        outputs = self._pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()


def build_default_provider() -> LLMProvider:
    """Construct the provider named in ``config.LLM_PROVIDER``.

    Falls back to :class:`EchoProvider` on any construction error so the
    chatbot remains usable without external dependencies or API keys.

    Recognised provider names (case-insensitive):

    * ``openai``                              — official OpenAI
    * ``dashscope`` / ``qwen`` / ``aliyun``   — Aliyun DashScope (Qwen)
    * ``huggingface_api`` / ``hf_api``        — HF hosted inference
    * ``huggingface_local`` / ``transformers``— local transformers pipeline
    * ``echo`` (default)                      — return retrieved context only
    """
    provider = (config.LLM_PROVIDER or "echo").lower()
    try:
        if provider == "openai":
            return OpenAIProvider()
        if provider in {"dashscope", "qwen", "aliyun", "modelscope"}:
            return DashScopeProvider()
        if provider in {"huggingface_api", "hf_api", "huggingface"}:
            return HuggingFaceAPIProvider()
        if provider in {"huggingface_local", "hf_local", "transformers"}:
            return HuggingFaceLocalProvider()
    except Exception as e:
        print(f"[chatbot] Falling back to EchoProvider ({provider} init failed: {e})")
    return EchoProvider()


# ══════════════════════════════════════════════════════════════════
# Query routing
# ══════════════════════════════════════════════════════════════════


_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")


@dataclass
class RoutedQuery:
    question: str
    snippets: list[str] = field(default_factory=list)
    matched_labels: list[str] = field(default_factory=list)


class QueryRouter:
    """Heuristic router that turns a user question into prompt context."""

    def __init__(
        self,
        graph_context: "GraphContext",
        vector_store: Optional["GraphVectorStore"] = None,
        edge_vector_store: Optional[
            "GraphEdgeVectorStore | Neo4jEdgeVectorStore"
        ] = None,
    ):
        self.gc = graph_context
        self.vs = vector_store
        self.evs = edge_vector_store

    # ── Intent regexes ────────────────────────────────────────────
    _RE_TOP = re.compile(
        r"(?:top|most\s+(?:central|important|influential)|rank)", re.I
    )
    _RE_PATH = re.compile(r"(?:path|connect(?:ion)?|bridge)\s+.+\s+(?:and|to|with)\s+", re.I)
    _RE_COMMUNITY = re.compile(r"\bcommunit", re.I)
    _RE_SUMMARY = re.compile(r"(?:summary|overview|size|how many|describe\s+(?:the\s+)?(?:graph|network))", re.I)
    _RE_COMPARE = re.compile(r"(?:compare|overlap|shared)", re.I)
    # Relational-verb hint: fires for questions like "who recommended X?",
    # "what did customers order?", "who loves Y?". Triggers edge-level vector
    # search when an edge vector store is available.
    _RE_RELATION = re.compile(
        r"\b(recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|hate(?:d|s)?|"
        r"love(?:d|s)?|order(?:ed|s|ing)?|buy|bought|give|gave|gives|"
        r"serve(?:d|s)?|cause(?:d|s)?|bring|brought|complain(?:ed|s)?|"
        r"prefer(?:red|s)?|tip|try|tried|tries|enjoy(?:ed|s)?|"
        r"dislike(?:d|s)?)\b",
        re.I,
    )
    _RE_RELATION_WH = re.compile(
        r"\b(who|what|which)\b.+\b(recommend|suggest|hate|love|order|buy|"
        r"give|serve|cause|bring|complain|prefer|tip|try|enjoy|dislike)\b",
        re.I,
    )

    # ── Route ──────────────────────────────────────────────────────
    def route(self, question: str, top_k: int = 5) -> RoutedQuery:
        q = question.strip()
        routed = RoutedQuery(question=q)
        ql = q.lower()

        # 0. High-level summary / source comparison (cheap, always useful)
        if self._RE_SUMMARY.search(ql):
            routed.snippets.append(self.gc.graph_summary())
        if self._RE_COMPARE.search(ql):
            routed.snippets.append(self.gc.compare_sources())

        # 1. Top concepts
        if self._RE_TOP.search(ql):
            metric = "pagerank"
            for m in ("betweenness", "degree", "closeness", "eigenvector", "pagerank"):
                if m in ql:
                    metric = m
                    break
            routed.snippets.append(self.gc.top_concepts(metric=metric, n=10))

        # 2. Community
        if self._RE_COMMUNITY.search(ql):
            # Try to extract a community id if user typed one
            m = re.search(r"communit[yi]\s*#?\s*(\d+)", ql)
            if m:
                cid = int(m.group(1))
                routed.snippets.append(self.gc.describe_community(cid))
            else:
                routed.snippets.append(
                    "Communities are groups of tightly related concepts. "
                    "Pass an integer id (e.g. 'community 0') for details."
                )

        # 3. Quoted labels → concept descriptions + path
        quoted = _QUOTED.findall(q)
        resolved = [lbl for lbl in quoted if self.gc.resolve(lbl)]
        for lbl in resolved:
            routed.snippets.append(self.gc.describe_concept(lbl))
            routed.matched_labels.append(lbl)
        if self._RE_PATH.search(ql) and len(resolved) >= 2:
            routed.snippets.append(self.gc.shortest_path(resolved[0], resolved[1]))

        # 4.5. Edge-level semantic search for relational questions
        #      (e.g. "who recommended X?", "what causes Y?"). Only runs when
        #      an edge vector store is plugged in; stays silent otherwise so
        #      backward-compatible questions behave identically.
        if self.evs and getattr(self.evs, "available", False):
            if self._RE_RELATION.search(ql) or self._RE_RELATION_WH.search(ql):
                try:
                    edge_hits: list[dict[str, Any]] = self.evs.search(q, k=top_k)
                except Exception:
                    edge_hits = []
                for hit in edge_hits:
                    src = hit.get("source_label") or "?"
                    tgt = hit.get("target_label") or "?"
                    verb = hit.get("verb") or "co-occurs with"
                    score = float(hit.get("score") or 0.0)
                    weight = hit.get("weight")
                    sentiment = hit.get("sentiment")
                    tail = ""
                    if weight is not None:
                        tail += f" (weight={weight}"
                        if sentiment is not None:
                            try:
                                tail += f", sentiment={float(sentiment):+.2f}"
                            except (TypeError, ValueError):
                                pass
                        tail += ")"
                    routed.snippets.append(
                        f"[edge match score={score:.3f}]\n"
                        f"\"{src} {verb} {tgt}\"{tail}"
                    )

        # 4. Vector search for any remaining semantic intent
        if self.vs and self.vs.available:
            hits = self.vs.search(q, k=top_k)
            for nid, score in hits:
                label = self.gc.G.nodes[nid].get("label", nid)
                if label in routed.matched_labels:
                    continue
                routed.snippets.append(
                    f"[vector match score={score:.3f}]\n"
                    + self.gc.describe_concept(label, top_neighbours=5)
                )
                routed.matched_labels.append(label)
        else:
            # Fallback: fuzzy label matching against the question tokens
            candidates = {
                lbl
                for lbl in self.gc.all_labels()
                if lbl.lower() in ql and lbl not in routed.matched_labels
            }
            for lbl in list(candidates)[:3]:
                routed.snippets.append(self.gc.describe_concept(lbl))
                routed.matched_labels.append(lbl)

        # 5. Always include a short graph summary if nothing matched
        if not routed.snippets:
            routed.snippets.append(self.gc.graph_summary())

        return routed


# ══════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════


SYSTEM_PROMPT = """You are a research assistant that answers questions about a
conceptual knowledge graph extracted from documents. You have access ONLY to
the graph context provided below. Cite specific concepts by their label in
single quotes and mention communities by id. If the context is insufficient,
say so clearly rather than guessing."""


class GraphChatbot:
    """End-to-end: route → retrieve → build prompt → call LLM → answer."""

    def __init__(
        self,
        graph_context: "GraphContext",
        llm_provider: Optional[LLMProvider] = None,
        vector_store: Optional["GraphVectorStore"] = None,
        edge_vector_store: Optional[
            "GraphEdgeVectorStore | Neo4jEdgeVectorStore"
        ] = None,
    ):
        self.gc = graph_context
        self.llm = llm_provider or build_default_provider()
        self.vs = vector_store
        self.evs = edge_vector_store
        self.router = QueryRouter(
            graph_context,
            vector_store,
            edge_vector_store=edge_vector_store,
        )

    # ── Public entry point ────────────────────────────────────────
    def ask(self, question: str, history: list[dict] | None = None) -> str:
        routed = self.router.route(question)
        prompt = self._build_user_message(routed, history=history)
        return self.llm.complete(SYSTEM_PROMPT, prompt)

    def ask_with_context(self, question: str) -> tuple[str, RoutedQuery]:
        """Return both the LLM answer and the raw retrieval, for dashboards."""
        routed = self.router.route(question)
        prompt = self._build_user_message(routed)
        answer = self.llm.complete(SYSTEM_PROMPT, prompt)
        return answer, routed

    # ── Prompt construction ───────────────────────────────────────
    def _build_user_message(
        self, routed: RoutedQuery, history: list[dict] | None = None
    ) -> str:
        parts: list[str] = []
        if history:
            parts.append("Conversation so far:")
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"  {role}: {content}")
            parts.append("")

        parts.append("Graph context:")
        for i, snippet in enumerate(routed.snippets, 1):
            parts.append(f"[{i}] {snippet}")
        parts.append("")
        parts.append(f"Question: {routed.question}")
        parts.append(
            "Answer based only on the graph context above. "
            "Be concrete and cite concept labels in single quotes."
        )
        return "\n".join(parts)
