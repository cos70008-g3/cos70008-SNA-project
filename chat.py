"""Standalone chat entry point — Stage 8 of the pipeline.

Runs the full pipeline (or re-uses cached results) to build the conceptual
network, then drops the user into an interactive chatbot grounded in that
network.

Usage::

    python chat.py --source policy
    python chat.py --source combined --vector-store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import POLICY_PDF_PATH, RESULTS_DIR


def build_graph(args: argparse.Namespace):
    """Run the minimum pipeline needed to produce G, partition, centrality_df."""
    from src.extraction.concept_extractor import ConceptExtractor
    from src.extraction.relationship_extractor import RelationshipExtractor
    from src.network.graph_analysis import GraphAnalyser
    from src.network.graph_builder import GraphBuilder
    from src.preprocessing.cleaner import clean_text
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    documents = []

    if args.source in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf
        pages = read_policy_pdf(args.pdf_path)
        cleaned = [(f"policy_page_{p}", clean_text(t, source="policy")) for p, t in pages]
        cleaned = [(did, t) for did, t in cleaned if t.strip()]
        full = " ".join(t for _, t in cleaned)
        documents.append(tokeniser.process(full, doc_id="policy_doc", source="policy"))

    if args.source in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews
        yelp_n = args.yelp_sample if args.yelp_sample > 0 else None
        df = load_yelp_reviews(category_filter="Restaurants", sample_n=yelp_n)
        raw = [(row["review_id"], clean_text(row["text"], source="yelp")) for _, row in df.iterrows()]
        raw = [(rid, t) for rid, t in raw if t.strip()]
        documents.extend(tokeniser.process_batch(raw, source="yelp"))

    if not documents:
        sys.exit("No documents loaded. Check --source and paths.")

    concepts = ConceptExtractor().extract(documents)
    rels = RelationshipExtractor().extract(documents, concepts)
    G = GraphBuilder(min_edge_weight=args.min_weight).build(concepts, rels)
    analyser = GraphAnalyser(G)
    centrality_df = analyser.all_centralities()
    try:
        partition = analyser.detect_communities_louvain()
    except ImportError:
        partition = analyser.detect_communities_label_propagation()
    G = analyser.annotate_graph(partition)
    return G, partition, centrality_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 8: conversational exploration of the conceptual network."
    )
    parser.add_argument(
        "--source", choices=["policy", "yelp", "combined"], default="policy"
    )
    parser.add_argument("--pdf-path", type=str, default=str(POLICY_PDF_PATH))
    parser.add_argument("--yelp-sample", type=int, default=300)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument(
        "--vector-store",
        action="store_true",
        help="Enable FAISS semantic retrieval (requires sentence-transformers + faiss-cpu).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override LLM provider for this session (openai, huggingface_api, huggingface_local, echo).",
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR),
        help="Output directory (unused at the moment, reserved for future caching).",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Route retrieval through a local Neo4j instance (requires "
             "`python pipeline.py --neo4j` to have populated it first).",
    )
    args = parser.parse_args()

    if args.provider:
        from src import config
        config.LLM_PROVIDER = args.provider

    from src.extensions.chatbot import GraphChatbot, build_default_provider

    gc = None
    vs = None

    if args.neo4j:
        try:
            from src.extensions.neo4j_store import Neo4jStore
            from src.extensions.neo4j_graph_context import Neo4jGraphContext
            from src.extensions.neo4j_vectorstore import Neo4jVectorStore

            store = Neo4jStore.from_config()
            store.connect()
            gc = Neo4jGraphContext(store, source_label=args.source)
            vs = Neo4jVectorStore(store, source_label=args.source)
            if not vs.available:
                vs = None
            print(
                f"  Backend: Neo4j (source_label='{args.source}', "
                f"{len(gc.all_labels())} concepts loaded)"
            )
        except Exception as e:
            print(f"  Neo4j backend unavailable ({e}); falling back to in-memory build.")
            gc = None

    if gc is None:
        print(f"Building conceptual network from source: {args.source} ...")
        G, partition, centrality_df = build_graph(args)
        print(
            f"  Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{len(set(partition.values()))} communities."
        )
        from src.extensions.graph_context import GraphContext

        gc = GraphContext(
            G=G, partition=partition, centrality_df=centrality_df,
            source_label=args.source,
        )
        if args.vector_store:
            from src.extensions.graph_vectorstore import GraphVectorStore
            vs = GraphVectorStore(gc)
            if not vs.available:
                print("  (vector store unavailable; falling back to structured retrieval)")
                vs = None

    bot = GraphChatbot(
        graph_context=gc, llm_provider=build_default_provider(), vector_store=vs
    )

    print("\n" + "=" * 60)
    print("  Graph Chatbot — ask questions about the conceptual network")
    print("  Type 'exit' or press Ctrl+D to quit.")
    print("=" * 60)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            print("Goodbye.")
            break
        try:
            answer = bot.ask(q)
        except Exception as e:
            answer = f"(error: {e})"
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()
