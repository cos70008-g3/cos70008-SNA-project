"""Streamlit interactive dashboard for exploring conceptual networks.

Run with:
    streamlit run src/extensions/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Text-to-Network Explorer", layout="wide")

# #region agent log
import json as _dbg_json_mod, os as _dbg_os_mod, sys as _dbg_sys_mod, time as _dbg_time_mod
_DBG_LOG_PATH = "/Users/spring.dai/Documents/DS/DS/cos70008/.cursor/debug-a90223.log"
_DBG_LOG_FALLBACK = "/tmp/debug-a90223.log"


def _agent_log(msg, data=None, hid="H0", location="dashboard.py:module"):
    payload = _dbg_json_mod.dumps({
        "sessionId": "a90223", "runId": "post-fix", "hypothesisId": hid,
        "location": location, "message": msg, "data": data or {},
        "timestamp": int(_dbg_time_mod.time() * 1000),
    })
    # stderr is captured by Streamlit's terminal, so we always get SOME trace
    print(f"[agent-log] {payload}", file=_dbg_sys_mod.stderr, flush=True)
    for _path in (_DBG_LOG_PATH, _DBG_LOG_FALLBACK):
        try:
            _dbg_os_mod.makedirs(_dbg_os_mod.path.dirname(_path), exist_ok=True)
            with open(_path, "a") as _fh:
                _fh.write(payload + "\n")
            break
        except Exception as _e:
            print(f"[agent-log] write-fail {_path}: {_e}", file=_dbg_sys_mod.stderr, flush=True)


_agent_log("dashboard_module_loaded", hid="H0")
# #endregion


@st.cache_resource
def load_pipeline(
    source_key: str,
    methods_tuple: tuple[str, ...],
    min_weight: int,
    yelp_sample: int,
    apply_sentiment: bool,
):
    """Run ingest → extract → graph → optional sentiment; match CLI pipeline behaviour."""
    from src.extraction.concept_extractor import ConceptExtractor
    from src.extraction.relationship_extractor import RelationshipExtractor
    from src.network.graph_analysis import GraphAnalyser
    from src.network.graph_builder import GraphBuilder
    from src.preprocessing.cleaner import clean_text
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    documents = []

    if source_key in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf
        pages = read_policy_pdf()
        raw_policy = [(f"policy_page_{p}", text) for p, text in pages]
        cleaned_policy = [(did, clean_text(t, source="policy")) for did, t in raw_policy]
        cleaned_policy = [(did, t) for did, t in cleaned_policy if t.strip()]
        full_clean = " ".join(t for _, t in cleaned_policy)
        documents.append(
            tokeniser.process(full_clean, doc_id="policy_doc", source="policy")
        )

    if source_key in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews
        yelp_n = yelp_sample if yelp_sample > 0 else None
        df = load_yelp_reviews(
            category_filter="Restaurants",
            sample_n=yelp_n,
        )
        raw_yelp = [(row["review_id"], row["text"]) for _, row in df.iterrows()]
        cleaned_yelp = [(did, clean_text(t, source="yelp")) for did, t in raw_yelp]
        cleaned_yelp = [(did, t) for did, t in cleaned_yelp if t.strip()]
        documents.extend(tokeniser.process_batch(cleaned_yelp, source="yelp"))

    use_ner = "NER" in methods_tuple
    use_np = "Noun Phrases" in methods_tuple
    use_tfidf = "TF-IDF" in methods_tuple
    if not methods_tuple:
        use_ner = use_np = use_tfidf = True

    extractor = ConceptExtractor(
        use_ner=use_ner,
        use_noun_phrases=use_np,
        use_tfidf=use_tfidf,
    )
    concepts = extractor.extract(documents)
    rels = RelationshipExtractor().extract(documents, concepts)
    G = GraphBuilder(min_edge_weight=min_weight).build(concepts, rels)

    if apply_sentiment and documents:
        from src.extensions.sentiment import SentimentAnnotator
        try:
            ann = SentimentAnnotator(method="vader")
        except ImportError:
            ann = SentimentAnnotator(method="textblob")
        concept_labels = {G.nodes[n].get("label", n) for n in G.nodes()}
        G = ann.annotate_graph(G, documents, concept_labels)

    analyser = GraphAnalyser(G)
    centrality_df = analyser.all_centralities()

    try:
        partition = analyser.detect_communities_louvain()
    except ImportError:
        partition = analyser.detect_communities_label_propagation()

    comm_df = analyser.community_summary(partition)
    brokers_df = analyser.find_brokers(partition)
    G = analyser.annotate_graph(partition)

    return G, centrality_df, partition, comm_df, brokers_df, documents


@st.cache_resource
def _run_temporal(
    source_key: str,
    n_chunks: int,
    yelp_sample: int,
    min_weight: int,
):
    """Build temporal slices for the dashboard's Temporal tab."""
    from src.extensions.temporal import TemporalAnalyser
    from src.extensions.temporal_slicing import (
        policy_pages_to_temporal_slices,
        yelp_reviews_to_year_slices,
    )
    from src.extraction.concept_extractor import ConceptExtractor
    from src.network.graph_builder import GraphBuilder
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    slices: list = []

    if source_key in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf
        pages = read_policy_pdf()
        slices.extend(
            policy_pages_to_temporal_slices(pages, tokeniser, n_chunks=n_chunks)
        )

    if source_key in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews
        yelp_n = yelp_sample if yelp_sample > 0 else None
        df = load_yelp_reviews(category_filter="Restaurants", sample_n=yelp_n)
        slices.extend(
            yelp_reviews_to_year_slices(
                df, tokeniser, sample_per_year=min(150, yelp_n or 150)
            )
        )

    if len(slices) < 2:
        return None, None

    ta = TemporalAnalyser(
        concept_extractor=ConceptExtractor(min_freq=1),
        graph_builder=GraphBuilder(min_edge_weight=min_weight),
    )
    tslices = ta.build_slices(slices)
    return ta.slice_summary(tslices), ta.comparison_table(tslices)


def _graph_fingerprint(G) -> str:
    """Cheap hash to let Streamlit cache recognise graph identity."""
    return f"{G.number_of_nodes()}:{G.number_of_edges()}"


@st.cache_resource
def _build_chatbot(
    graph_fp: str,
    provider: str,
    enable_vector: bool,
    backend: str,
    *,
    _G,
    _partition,
    _centrality_df,
    source_key: str,
):
    """Construct a GraphChatbot for the current graph + provider.

    Leading underscores on _G, _partition, _centrality_df tell Streamlit to
    skip hashing those complex objects — the cache key is built from
    graph_fp + provider + enable_vector + backend + source_key instead.

    Parameters
    ----------
    backend : {"memory", "neo4j"}
        ``memory`` uses the in-process NetworkX graph + FAISS vector store.
        ``neo4j`` routes retrieval through a running Neo4j instance (requires
        the pipeline to have been run once with ``--neo4j`` to populate it).
    """
    from src import config
    from src.extensions.chatbot import GraphChatbot, build_default_provider

    prior_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = provider
    llm = build_default_provider()
    config.LLM_PROVIDER = prior_provider

    gc = None
    vs = None
    evs = None

    if backend == "neo4j":
        try:
            from src.extensions.neo4j_store import Neo4jStore
            from src.extensions.neo4j_graph_context import Neo4jGraphContext
            from src.extensions.neo4j_vectorstore import Neo4jVectorStore
            from src.extensions.neo4j_edge_vectorstore import (
                Neo4jEdgeVectorStore,
            )

            store = Neo4jStore.from_config()
            store.connect()
            gc = Neo4jGraphContext(store, source_label=source_key)
            if enable_vector:
                vs = Neo4jVectorStore(store, source_label=source_key)
                if not vs.available:
                    vs = None
                evs = Neo4jEdgeVectorStore(store, source_label=source_key)
                if not evs.available:
                    evs = None
        except Exception as e:
            _agent_log(
                "neo4j_backend_failed",
                data={"error": str(e)},
                hid="H2",
                location="dashboard.py:_build_chatbot",
            )
            gc = None

    if gc is None:
        from src.extensions.graph_context import GraphContext

        gc = GraphContext(
            G=_G,
            partition=_partition,
            centrality_df=_centrality_df,
            source_label=source_key,
        )
        if enable_vector:
            from src.extensions.graph_vectorstore import (
                GraphEdgeVectorStore,
                GraphVectorStore,
            )
            vs = GraphVectorStore(gc)
            if not vs.available:
                vs = None
            evs = GraphEdgeVectorStore(gc)
            if not evs.available:
                evs = None

    return GraphChatbot(
        graph_context=gc,
        llm_provider=llm,
        vector_store=vs,
        edge_vector_store=evs,
    )


def main():
    st.title("Text-to-Network Explorer")
    st.markdown("Transform unstructured text into conceptual networks for SNA analysis.")

    with st.sidebar:
        st.header("Configuration")
        source = st.selectbox(
            "Data Source",
            [
                "Policy Document",
                "Yelp Reviews",
                "Combined (Policy + Yelp)",
            ],
        )
        source_key = {
            "Policy Document": "policy",
            "Yelp Reviews": "yelp",
            "Combined (Policy + Yelp)": "combined",
        }[source]

        methods = st.multiselect(
            "Extraction Methods",
            ["NER", "Noun Phrases", "TF-IDF"],
            default=["NER", "Noun Phrases", "TF-IDF"],
        )
        min_weight = st.slider("Min Edge Weight", 1, 10, 1)
        top_n = st.slider("Top N Concepts", 10, 100, 30)
        yelp_sample = 500
        if source_key in ("yelp", "combined"):
            yelp_sample = st.slider("Yelp review sample size", 100, 2000, 500, step=50)
        apply_sentiment = st.checkbox(
            "Sentiment-weighted edges (VADER / TextBlob)",
            value=False,
            help="Annotate edges using sentence sentiment where endpoint concepts co-occur.",
        )

    methods_tuple = tuple(sorted(methods))

    try:
        G, centrality_df, partition, comm_df, brokers_df, documents = load_pipeline(
            source_key,
            methods_tuple,
            min_weight,
            yelp_sample,
            apply_sentiment,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("For Yelp sources, extract the Yelp academic dataset JSONL files into `data/Yelp JSON/yelp_dataset/`.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Network",
            "Centrality",
            "Communities",
            "Brokers",
            "Temporal",
            "Chat",
            "Extensions",
        ]
    )

    with tab1:
        st.subheader("Interactive Network")
        from src.visualisation.interactive_viz import InteractiveVisualiser
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            iv = InteractiveVisualiser(tmpdir)
            path = iv.create_interactive_network(
                G,
                partition=partition,
                top_n=top_n,
                colour_by_source=(source_key == "combined"),
                colour_edges_by_sentiment=apply_sentiment,
            )
            html_content = path.read_text()
            st.components.v1.html(html_content, height=700, scrolling=True)

    with tab2:
        st.subheader(f"Top {top_n} Concepts by Centrality")
        st.dataframe(centrality_df.head(top_n), use_container_width=True)

        metric = st.selectbox(
            "Plot metric", ["pagerank", "betweenness", "degree", "closeness"]
        )
        chart_data = centrality_df.nlargest(top_n, metric).set_index("label")[metric]
        st.bar_chart(chart_data)

    with tab3:
        st.subheader("Community Structure")
        st.dataframe(comm_df, use_container_width=True)

        from collections import Counter

        sizes = Counter(partition.values())
        st.bar_chart(pd.Series(sizes).sort_index().rename("Size"))

    with tab4:
        st.subheader("Broker Concepts")
        st.markdown("Nodes that bridge different thematic communities.")
        st.dataframe(brokers_df, use_container_width=True)

    with tab5:
        st.subheader("Temporal comparison")
        st.markdown(
            "Split the corpus into slices (policy = page chunks, Yelp = by year) "
            "and compare how the conceptual network changes across slices."
        )
        n_chunks = st.slider("Number of policy page chunks", 2, 8, 3)
        run_temporal = st.button("Build temporal slices", type="primary")

        if run_temporal:
            with st.spinner("Building temporal networks..."):
                summary_df, pairwise_df = _run_temporal(
                    source_key=source_key,
                    n_chunks=n_chunks,
                    yelp_sample=yelp_sample,
                    min_weight=min_weight,
                )
            if summary_df is None or summary_df.empty:
                st.warning(
                    "Need at least two slices to compare. "
                    "Increase chunks or use a source that yields multiple periods."
                )
            else:
                st.markdown("**Slice summary**")
                st.dataframe(summary_df, use_container_width=True)
                st.markdown("**Consecutive slice comparisons**")
                st.dataframe(pairwise_df, use_container_width=True)
                if "jaccard_nodes" in pairwise_df.columns and not pairwise_df.empty:
                    st.bar_chart(
                        pairwise_df.set_index(
                            pairwise_df["slice_a"] + " → " + pairwise_df["slice_b"]
                        )[["jaccard_nodes", "jaccard_edges"]]
                    )
        else:
            st.info(
                "Press **Build temporal slices** to construct separate networks per slice "
                "and compare them. This may take a minute for Yelp/combined sources."
            )

    with tab6:
        st.subheader("Chat with the knowledge graph")
        st.caption(
            "Conversational Q&A grounded in the graph. Default provider is "
            "loaded from `LLM_PROVIDER` in `.env` (or `src/config.py`); "
            "`echo` is a no-LLM fallback that returns retrieved graph "
            "context verbatim."
        )

        with st.expander("Chatbot settings", expanded=False):
            _provider_options = [
                "echo",
                "openai",
                "dashscope",
                "huggingface_api",
                "huggingface_local",
            ]
            _default_provider = (config.LLM_PROVIDER or "echo").lower()
            _alias = {"qwen": "dashscope", "aliyun": "dashscope", "modelscope": "dashscope"}
            _default_provider = _alias.get(_default_provider, _default_provider)
            if _default_provider not in _provider_options:
                _default_provider = "echo"
            provider_choice = st.selectbox(
                "LLM provider",
                _provider_options,
                index=_provider_options.index(_default_provider),
                help=(
                    "Default comes from `LLM_PROVIDER` in `.env`. "
                    "`echo` requires no API key. `dashscope` uses Qwen via "
                    "`DASHSCOPE_API_KEY`. `openai` uses `OPENAI_API_KEY`. "
                    "Hugging Face options need `HUGGINGFACE_API_TOKEN` (api) "
                    "or a local install (local). See README."
                ),
            )
            use_vec = st.checkbox(
                "Enable semantic vector search",
                value=False,
                help=(
                    "Uses `sentence-transformers` to embed queries. "
                    "Backed by FAISS in-memory, or Neo4j's vector index "
                    "when the Neo4j backend is selected."
                ),
            )

            from src.extensions.neo4j_store import Neo4jStore as _NStore
            neo4j_available = _NStore.ping()
            backend_choice = st.radio(
                "Retrieval backend",
                options=["memory", "neo4j"],
                format_func=lambda v: {
                    "memory": "In-memory NetworkX (FAISS)",
                    "neo4j": "Neo4j (Cypher + native vector index)",
                }[v],
                index=0,
                horizontal=True,
                help=(
                    "Neo4j requires `docker compose up -d` plus `python pipeline.py "
                    "--source <src> --neo4j` to have populated the instance."
                ),
            )
            if backend_choice == "neo4j" and not neo4j_available:
                st.warning(
                    "Neo4j is not reachable at the configured NEO4J_URI — "
                    "falling back to the in-memory backend. Check `.env` and "
                    "that `docker compose up -d` is running."
                )

        # #region agent log
        _agent_log("chat_tab_entered", hid="H1", location="dashboard.py:tab6")
        # #endregion

        effective_backend = (
            "neo4j" if backend_choice == "neo4j" and neo4j_available else "memory"
        )
        bot = _build_chatbot(
            _graph_fingerprint(G),
            provider_choice,
            use_vec,
            effective_backend,
            _G=G,
            _partition=partition,
            _centrality_df=centrality_df,
            source_key=source_key,
        )
        st.caption(
            f"Backend in use: **{effective_backend}**"
            + (" (Cypher + vector index)" if effective_backend == "neo4j" else " (NetworkX + FAISS)")
        )

        # #region agent log
        _agent_log(
            "chatbot_ready",
            data={"bot_type": type(bot).__name__, "provider": provider_choice},
            hid="H1",
            location="dashboard.py:tab6",
        )
        # #endregion

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        user_q = st.chat_input("Ask about concepts, communities, or overall structure…")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer, routed = bot.ask_with_context(user_q)
                    except Exception as e:
                        answer = f"(error: {e})"
                        routed = None
                st.markdown(answer)
                if routed and routed.snippets:
                    with st.expander("Graph context used"):
                        for i, s in enumerate(routed.snippets, 1):
                            st.code(s, language="text")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        if st.button("Clear conversation"):
            st.session_state.chat_history = []

    with tab7:
        st.subheader("Optional extensions")
        st.markdown(
            """
**Temporal comparison** — Use the *Temporal* tab, or run
`python pipeline.py --temporal 3` to export `temporal_slice_summary.csv` and
`temporal_pairwise.csv`. The notebook `notebooks/06_optional_extensions_temporal.ipynb`
also walks through the same logic interactively.

**User-defined concepts** — Edit `config/concept_dictionary.yaml` and run the CLI with
`--concept-dictionary default` (or a path to your YAML).

**Sentiment-weighted edges** — `python pipeline.py --source combined --sentiment`
produces sentiment-coloured edges in `interactive_network.html` (same logic as the
checkbox in this app).

**Conversational exploration (Stage 8)** — `python pipeline.py --chat` or
`python chat.py --source policy` opens an interactive chatbot grounded in the graph.
            """
        )


if __name__ == "__main__":
    main()
