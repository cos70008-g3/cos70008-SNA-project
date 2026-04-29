"""CLI entry point: run the full Text-to-Network Engine pipeline end-to-end."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import CONCEPT_DICTIONARY_PATH, POLICY_PDF_PATH, RESULTS_DIR


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_cd = (args.concept_dictionary or "").strip()
    if raw_cd == "default":
        concept_dict_path: Path | None = CONCEPT_DICTIONARY_PATH
    elif raw_cd:
        concept_dict_path = Path(raw_cd)
    else:
        concept_dict_path = None

    print(f"{'='*60}")
    print("  Text-to-Network Engine")
    print(f"{'='*60}")
    print(f"  Source : {args.source}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Stage 1: Ingestion ─────────────────────────────────────────
    print("[1/8] Ingesting data...")
    from src.preprocessing.cleaner import clean_text
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    documents = []
    pages: list = []
    df = None

    if args.source in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf, pages_to_document
        pages = read_policy_pdf(args.pdf_path)
        raw_policy = [(f"policy_page_{p}", text) for p, text in pages]
        cleaned_policy = [(did, clean_text(t, source="policy")) for did, t in raw_policy]
        cleaned_policy = [(did, t) for did, t in cleaned_policy if t.strip()]
        full_clean = " ".join(t for _, t in cleaned_policy)
        documents.append(tokeniser.process(full_clean, doc_id="policy_doc", source="policy"))
        print(f"       Loaded {len(pages)} pages from policy PDF")

    if args.source in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews
        yelp_n = args.yelp_sample if args.yelp_sample > 0 else None
        df = load_yelp_reviews(
            category_filter=args.yelp_category,
            sample_n=yelp_n,
        )
        raw_yelp = [(row["review_id"], row["text"]) for _, row in df.iterrows()]
        cleaned_yelp = [(did, clean_text(t, source="yelp")) for did, t in raw_yelp]
        cleaned_yelp = [(did, t) for did, t in cleaned_yelp if t.strip()]
        documents.extend(tokeniser.process_batch(cleaned_yelp, source="yelp"))
        df.to_csv(output_dir / "yelp_sample.csv", index=False)
        print(f"       Loaded {len(df)} Yelp reviews")

    if not documents:
        sys.exit("No documents were loaded. Check source and paths.")

    # ── Stage 2: Preprocessing summary ─────────────────────────────
    total_sents = sum(len(d.sentences) for d in documents)
    print(f"\n[2/8] Preprocessing complete")
    print(f"       {len(documents)} document(s), {total_sents} sentences")

    # ── Stage 3: Extraction ────────────────────────────────────────
    print("[3/8] Extracting concepts and relationships...")
    from src.extraction.concept_extractor import ConceptExtractor
    from src.extraction.relationship_extractor import RelationshipExtractor

    concept_extractor = ConceptExtractor(
        use_ner=("ner" in args.methods or "all" in args.methods),
        use_noun_phrases=("np" in args.methods or "all" in args.methods),
        use_tfidf=("tfidf" in args.methods or "all" in args.methods),
    )
    concepts = concept_extractor.extract(documents)
    if concept_dict_path:
        from src.extensions.concept_dictionary import apply_concept_dictionary
        concepts = apply_concept_dictionary(concepts, concept_dict_path)
    print(f"       {len(concepts)} concepts extracted")

    rel_extractor = RelationshipExtractor(
        use_cooccurrence=True,
        use_dependency=True,
    )
    relationships = rel_extractor.extract(documents, concepts)
    print(f"       {len(relationships)} relationships extracted")

    # ── Stage 4: Network Construction ──────────────────────────────
    print("[4/8] Building network...")
    from src.network.graph_builder import GraphBuilder

    builder = GraphBuilder(min_edge_weight=args.min_weight)
    G = builder.build(concepts, relationships)
    print(f"       Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.sentiment:
        print("       Annotating edges with sentiment...")
        from src.extensions.sentiment import SentimentAnnotator
        try:
            ann = SentimentAnnotator(method="vader")
        except ImportError:
            ann = SentimentAnnotator(method="textblob")
        concept_labels = {G.nodes[n].get("label", n) for n in G.nodes()}
        G = ann.annotate_graph(G, documents, concept_labels)

    builder.export_graphml(G, output_dir / "network.graphml")
    builder.export_edge_csv(G, output_dir / "edges.csv")
    builder.export_node_csv(G, output_dir / "nodes.csv")

    # ── Stage 5: Analysis ──────────────────────────────────────────
    print("[5/8] Analysing network...")
    from src.network.graph_analysis import GraphAnalyser

    analyser = GraphAnalyser(G)
    centrality_df = analyser.all_centralities()
    centrality_df.to_csv(output_dir / "centralities.csv", index=False)
    print(f"       Top concepts by PageRank:")
    for _, row in centrality_df.head(10).iterrows():
        print(f"         - {row['label']} ({row['pagerank']:.4f})")

    try:
        partition = analyser.detect_communities_louvain()
    except ImportError:
        partition = analyser.detect_communities_label_propagation()

    comm_df = analyser.community_summary(partition)
    comm_df.to_csv(output_dir / "communities.csv", index=False)
    print(f"       {len(comm_df)} communities detected")

    brokers_df = analyser.find_brokers(partition)
    brokers_df.to_csv(output_dir / "brokers.csv", index=False)

    G = analyser.annotate_graph(partition)

    stats = analyser.summary_stats()
    print(f"       Network density: {stats['density']:.4f}")
    print(f"       Clustering coefficient: {stats['avg_clustering']:.4f}")

    if args.source == "combined":
        print("\n       ── Cross-Source Analysis ──")
        src_dist = analyser.source_distribution()
        src_dist.to_csv(output_dir / "source_distribution.csv", index=False)
        for _, row in src_dist.iterrows():
            print(f"         {row['source_type']}: {row['node_count']} concepts")

        comparison = analyser.source_comparison_summary()
        print(f"         Policy-only: {comparison['policy_only_concepts']}")
        print(f"         Yelp-only:   {comparison['yelp_only_concepts']}")
        print(f"         Shared:      {comparison['shared_concepts']} ({comparison['overlap_pct']:.1f}%)")

        cross_edges = analyser.cross_source_edges()
        cross_edges.to_csv(output_dir / "cross_source_edges.csv", index=False)
        print(f"         Cross-source edges: {len(cross_edges)}")

        bridges = analyser.bridging_concepts(top_n=20)
        bridges.to_csv(output_dir / "bridging_concepts.csv", index=False)
        if not bridges.empty:
            print(f"         Top bridging concepts:")
            for _, row in bridges.head(5).iterrows():
                print(f"           - {row['label']} (score: {row['bridge_score']:.4f}, src: {row['source_type']})")

        import json
        with open(output_dir / "cross_source_summary.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # ── Optional: push to Neo4j ────────────────────────────────────
    if args.neo4j:
        print("\n       Pushing graph to Neo4j...")
        try:
            from src.extensions.neo4j_store import (
                Neo4jStore,
                Neo4jUnavailableError,
            )
            with Neo4jStore.from_config() as _store:
                counts = _store.push_graph(
                    G,
                    partition=partition,
                    centrality_df=centrality_df,
                    source_label=args.source,
                    reset=args.neo4j_reset,
                )
                embedded = _store.embed_and_store(G, source_label=args.source)
                edge_embedded = _store.embed_edges(
                    G,
                    source_label=args.source,
                    top_n_association=args.edge_embed_top_n,
                )
            print(
                f"         Neo4j: {counts['nodes']} nodes, {counts['edges']} edges"
                + (f", {embedded} node embeddings" if embedded else "")
                + (f", {edge_embedded} edge embeddings" if edge_embedded else "")
                + f" under source_label='{args.source}'"
            )
        except Neo4jUnavailableError as e:
            print(f"         Neo4j unavailable: {e}")
        except Exception as e:
            print(f"         Neo4j push failed: {e}")

    # ── Stage 6: Visualisation ─────────────────────────────────────
    print("\n[6/8] Generating visualisations...")
    from src.visualisation.static_viz import StaticVisualiser
    from src.visualisation.interactive_viz import InteractiveVisualiser

    static_viz = StaticVisualiser(output_dir)

    if args.source == "combined":
        static_viz.plot_network_by_source(G, title="Combined Network — Coloured by Source")
        static_viz.plot_source_overlap(G)
    static_viz.plot_network(G, partition=partition,
                            title=f"Conceptual Network ({args.source.title()} Data)")
    static_viz.plot_top_concepts(centrality_df, top_n=20)
    static_viz.plot_community_distribution(partition)
    static_viz.plot_cooccurrence_heatmap(G, top_n=20)

    interactive_viz = InteractiveVisualiser(output_dir)
    html_path = interactive_viz.create_interactive_network(
        G, partition=partition,
        title=f"Conceptual Network Explorer - {args.source.title()} Data",
        colour_by_source=(args.source == "combined"),
        colour_edges_by_sentiment=args.sentiment,
    )

    # ── Stage 7: Temporal comparison (optional) ────────────────────
    if args.temporal and args.temporal > 1:
        print(f"\n[7/8] Temporal comparison ({args.temporal} slices)...")
        from src.extensions.temporal import TemporalAnalyser
        from src.extensions.temporal_slicing import (
            policy_pages_to_temporal_slices,
            yelp_reviews_to_year_slices,
        )
        from src.extraction.concept_extractor import ConceptExtractor as _CE
        from src.network.graph_builder import GraphBuilder as _GB

        slices: list = []
        if args.source in ("policy", "combined") and pages:
            slices.extend(
                policy_pages_to_temporal_slices(
                    pages, tokeniser, n_chunks=args.temporal
                )
            )
        if args.source in ("yelp", "combined") and df is not None:
            slices.extend(
                yelp_reviews_to_year_slices(
                    df, tokeniser, sample_per_year=args.yelp_sample or 150
                )
            )

        if len(slices) >= 2:
            ta = TemporalAnalyser(
                concept_extractor=_CE(min_freq=1),
                graph_builder=_GB(min_edge_weight=1),
            )
            tslices = ta.build_slices(slices)
            summary_df = ta.slice_summary(tslices)
            pairwise_df = ta.comparison_table(tslices)
            summary_df.to_csv(output_dir / "temporal_slice_summary.csv", index=False)
            pairwise_df.to_csv(output_dir / "temporal_pairwise.csv", index=False)
            print(f"       {len(tslices)} slices built; summary saved")
            for _, row in summary_df.iterrows():
                print(
                    f"         {row['slice']}: {row['nodes']} nodes, "
                    f"{row['edges']} edges (top: {row['top_concept']})"
                )
        else:
            print("       Not enough slices (need >= 2); skipping.")
    else:
        print("\n[7/8] Temporal comparison skipped (use --temporal N to enable)")

    # ── Stage 8: Conversational exploration (chatbot) ──────────────
    if args.chat:
        print("\n[8/8] Starting chatbot (Ctrl+D or 'exit' to quit)...")
        try:
            from src.extensions.chatbot import GraphChatbot, build_default_provider

            gc = None
            vector_store = None
            edge_vector_store = None

            use_neo4j = args.neo4j
            if use_neo4j:
                try:
                    from src.extensions.neo4j_store import Neo4jStore
                    from src.extensions.neo4j_graph_context import Neo4jGraphContext
                    from src.extensions.neo4j_vectorstore import Neo4jVectorStore
                    from src.extensions.neo4j_edge_vectorstore import (
                        Neo4jEdgeVectorStore,
                    )

                    store = Neo4jStore.from_config()
                    store.connect()
                    gc = Neo4jGraphContext(store, source_label=args.source)
                    vector_store = Neo4jVectorStore(store, source_label=args.source)
                    edge_vector_store = Neo4jEdgeVectorStore(
                        store, source_label=args.source
                    )
                    if not getattr(edge_vector_store, "available", False):
                        edge_vector_store = None
                    print("       Backend: Neo4j (Cypher + native vector index)")
                except Exception as e:
                    print(f"       Neo4j backend unavailable ({e}); falling back.")
                    gc = None

            if gc is None:
                from src.extensions.graph_context import GraphContext
                from src.extensions.graph_vectorstore import (
                    GraphEdgeVectorStore,
                    GraphVectorStore,
                )

                gc = GraphContext(
                    G=G,
                    partition=partition,
                    centrality_df=centrality_df,
                    source_label=args.source,
                )
                vector_store = GraphVectorStore(gc)
                edge_vector_store = GraphEdgeVectorStore(
                    gc, top_n_association=args.edge_embed_top_n
                )
                if not getattr(edge_vector_store, "available", False):
                    edge_vector_store = None
                print("       Backend: in-memory NetworkX + FAISS")

            provider = build_default_provider()
            bot = GraphChatbot(
                graph_context=gc,
                llm_provider=provider,
                vector_store=vector_store,
                edge_vector_store=edge_vector_store,
            )
            _run_chat_loop(bot)
        except Exception as e:
            print(f"       Chatbot unavailable: {e}")
    else:
        print("\n[8/8] Chatbot skipped (use --chat to start an interactive session)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {output_dir}")
    print(f"  Interactive network: {html_path}")
    print(f"{'='*60}")


def _run_chat_loop(bot) -> None:
    """Simple REPL for Stage 8 (chatbot)."""
    print("\n" + "=" * 60)
    print("  Graph Chatbot — ask questions about the conceptual network")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-Network Engine: transform text into conceptual networks",
    )
    parser.add_argument(
        "--source", choices=["policy", "yelp", "combined"], default="policy",
        help="Data source to process: policy, yelp, or combined (default: policy)",
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        choices=["all", "ner", "np", "tfidf"],
        help="Concept extraction methods to use",
    )
    parser.add_argument(
        "--min-weight", type=int, default=1,
        help="Minimum edge weight to include in network",
    )
    parser.add_argument(
        "--pdf-path", type=str, default=str(POLICY_PDF_PATH),
        help="Path to policy PDF (for --source policy)",
    )
    parser.add_argument(
        "--yelp-category", type=str, default="Restaurants",
        help="Yelp business category filter",
    )
    parser.add_argument(
        "--yelp-sample", type=int, default=500,
        help="Number of Yelp reviews to sample (0 = use ALL matching reviews)",
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Annotate edges with sentiment from co-occurring sentences (VADER or TextBlob)",
    )
    parser.add_argument(
        "--concept-dictionary",
        type=str,
        default="",
        help="YAML path for user-defined concepts, or the keyword 'default' for bundled config; omit to disable",
    )
    parser.add_argument(
        "--temporal",
        type=int,
        default=0,
        metavar="N",
        help="Run temporal comparison with N slices (policy=page chunks, yelp=by year). "
             "Writes temporal_slice_summary.csv and temporal_pairwise.csv. 0 = disabled.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="After Stage 7, enter an interactive chatbot (Stage 8) over the knowledge graph.",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Push the built network to a local Neo4j instance (configure via NEO4J_* env vars) "
             "and route the --chat backend through Cypher + the native vector index.",
    )
    parser.add_argument(
        "--neo4j-reset",
        action="store_true",
        help="Delete existing Concept nodes with the current --source label before pushing. "
             "Useful when iterating to keep the graph deterministic.",
    )
    parser.add_argument(
        "--edge-embed-top-n",
        type=int,
        default=2000,
        metavar="N",
        help="Edge-level semantic search: embed all verb-bearing edges plus the "
             "top-N ASSOCIATION edges by weight (default 2000). Applies to both "
             "the Neo4j push and the in-memory chatbot backend.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
