"""Neo4j Data view — sidebar-toggleable section in the Streamlit dashboard.

Owns the cached Neo4j reader, the streamlit-agraph rendering helper, and
the page entry point invoked from ``dashboard.main`` when the View radio
is set to ``"Neo4j Data"``.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_neo4j_reader(source_key: str):
    """Cached (Neo4jStore, Neo4jGraphReader); ``None`` if Neo4j is unreachable."""
    from src.extensions.neo4j_reader import Neo4jGraphReader
    from src.extensions.neo4j_store import Neo4jStore, Neo4jUnavailableError

    try:
        store = Neo4jStore.from_config()
        store.connect()
    except Neo4jUnavailableError as e:
        logger.debug("Neo4j unreachable for source_key=%r: %s", source_key, e)
        return None
    return store, Neo4jGraphReader(store, source_label=source_key)


def _ego_subgraph(reader, seed_label: str, max_neighbours: int):
    """Bounded ego subgraph: seed + top-N *unique* neighbours by edge weight."""
    n = int(max_neighbours)
    nb_df = reader.neighbours(seed_label, top_n=max(n * 5, 100), direction="both")
    seed_id = reader.resolve(seed_label) or seed_label
    if nb_df.empty:
        keep_ids = [seed_id]
    else:
        seen: list = []
        for nid in nb_df["node_id"]:
            if nid not in seen:
                seen.append(nid)
                if len(seen) >= n:
                    break
        keep_ids = list({*seen, seed_id})
    return reader._build_subgraph_from_ids(keep_ids, min_weight=1)


def _star_filter(G, anchor_id: str):
    """Keep only edges touching ``anchor_id`` (spoke layout for hub ego graphs)."""
    if anchor_id not in G:
        return G
    keep_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if u == anchor_id or v == anchor_id
    ]
    keep_nodes = {anchor_id} | {n for u, v, _ in keep_edges for n in (u, v)}
    H = G.__class__()
    for n in keep_nodes:
        H.add_node(n, **G.nodes[n])
    for u, v, d in keep_edges:
        H.add_edge(u, v, **d)
    return H


def _prune_subgraph_to_top_pagerank(G, max_nodes: int, anchor_id: str | None = None):
    """Keep at most ``max_nodes`` nodes ranked by stored PageRank, plus anchor."""
    if G.number_of_nodes() <= max_nodes:
        return G
    by_pr = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n].get("pagerank", 0.0),
        reverse=True,
    )
    keep = set(by_pr[: int(max_nodes)])
    if anchor_id is not None and anchor_id in G:
        keep.add(anchor_id)
        if len(keep) > max_nodes:
            for n in reversed(by_pr):
                if n in keep and n != anchor_id:
                    keep.discard(n)
                    break
    return G.subgraph(keep).copy()


def _render_agraph_subgraph(
    G_sub,
    *,
    source_key: str,
    apply_sentiment: bool,
    key: str,
    height: int = 750,
):
    """Render an ``nx.DiGraph`` subgraph via streamlit-agraph (vis.js)."""
    if G_sub.number_of_nodes() == 0:
        st.info("No nodes to display.")
        return None

    import networkx as nx
    from streamlit_agraph import Config, Edge, Node, agraph

    from src.visualisation.interactive_viz import (
        COMMUNITY_COLORS,
        InteractiveVisualiser,
    )
    src_colors = InteractiveVisualiser.SOURCE_COLORS
    HIGHLIGHT = "#ff6347"

    node_ids_str = {str(n) for n in G_sub.nodes()}
    selected = st.session_state.get(f"{key}_last_click")
    if selected not in node_ids_str:
        selected = None

    if selected:
        cap_col, btn_col = st.columns([6, 1])
        cap_col.caption(
            f"Selected: `{selected}` — connected edges shown in red."
        )
        if btn_col.button("Clear", key=f"{key}_clear", type="secondary"):
            st.session_state[f"{key}_last_click"] = None
            st.rerun()

    pr = nx.pagerank(G_sub, weight="weight") if G_sub.number_of_edges() else {
        n: 0.0 for n in G_sub.nodes()
    }
    max_pr = max(pr.values()) if pr else 1.0
    if max_pr <= 0:
        max_pr = 1.0

    nodes: list[Node] = []
    for n, attrs in G_sub.nodes(data=True):
        label = str(attrs.get("label", n))
        size = 8 + 18 * (pr.get(n, 0.0) / max_pr)
        community = int(attrs.get("community", 0) or 0)
        source_type = attrs.get("source_type", "unknown") or "unknown"
        if source_key == "combined":
            color = src_colors.get(source_type, "#808080")
        else:
            color = COMMUNITY_COLORS[community % len(COMMUNITY_COLORS)]
        if selected is not None and str(n) == selected:
            color = HIGHLIGHT
        tooltip = (
            f"<b>{label}</b><br>"
            f"Type: {attrs.get('type', 'n/a')}<br>"
            f"Source: {source_type}<br>"
            f"Frequency: {attrs.get('frequency', 0)}<br>"
            f"PageRank: {pr.get(n, 0.0):.4f}<br>"
            f"Community: {community}"
        )
        nodes.append(
            Node(
                id=str(n),
                label=label[:20],
                size=size,
                color=color,
                title=tooltip,
                shape="circle",
                font={
                    "size": 13,
                    "color": "#ffffff",
                    "strokeWidth": 3,
                    "strokeColor": "#000000",
                    "face": "Inter, system-ui, sans-serif",
                    "bold": True,
                },
            )
        )

    # Spring-length tuning: clamp(BASE + SLOPE*avg_degree, FLOOR, CAP).
    # Raise to spread, lower to tighten.
    BASE, SLOPE, FLOOR, CAP = 180, 10, 180, 400
    n_nodes = max(G_sub.number_of_nodes(), 1)
    n_edges = G_sub.number_of_edges()
    avg_degree = n_edges / n_nodes
    spring_length = max(FLOOR, min(BASE + int(SLOPE * avg_degree), CAP))
    # Suppress per-edge labels on dense graphs; type still in tooltip.
    hide_edge_labels = avg_degree > 5

    edges: list[Edge] = []
    for u, v, data in G_sub.edges(data=True):
        weight = float(data.get("weight", 1) or 1)
        types = data.get("types", set())
        if isinstance(types, (set, list, tuple)):
            type_str = ", ".join(sorted(types))
        else:
            type_str = str(types or "")
        edge_label = type_str if type_str else str(data.get("top_verb") or "")
        edge_tooltip = f"Weight: {weight:g}<br>Types: {type_str}"
        ewidth = max(0.5, min(0.5 + weight * 0.15, 2.5))
        ecolor = "#a8a8a8"
        if apply_sentiment and data.get("sentiment") is not None:
            try:
                s = float(data["sentiment"])
                sl = data.get("sentiment_label", "neutral")
                edge_tooltip += f"<br>Sentiment: {s:.3f} ({sl})"
                if s > 0.05:
                    ecolor = "#2ca02c"
                elif s < -0.05:
                    ecolor = "#d62728"
                else:
                    ecolor = "#a8a8a8"
                ewidth = max(0.5, min(0.5 + 1.5 * abs(s), 2.5))
            except (TypeError, ValueError):
                pass

        if selected is not None and (str(u) == selected or str(v) == selected):
            ecolor = HIGHLIGHT
            ewidth = max(ewidth, 3.0)

        edges.append(
            Edge(
                source=str(u),
                target=str(v),
                label="" if hide_edge_labels else edge_label,
                color=ecolor,
                width=ewidth,
                title=edge_tooltip,
                arrows={"to": {"enabled": True, "scaleFactor": 0.4}},
                length=spring_length,
                font={
                    "size": 8,
                    "color": "#d0d0d0",
                    "strokeWidth": 2,
                    "strokeColor": "#000000",
                    "face": "Inter, system-ui, sans-serif",
                    "align": "horizontal",
                },
            )
        )

    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#ff6347",
        collapsible=False,
    )

    with st.container(border=True):
        clicked = agraph(nodes=nodes, edges=edges, config=config)

    # Rerun only on a *new* node click. Reject ``None`` (programmatic
    # rerun resets agraph's value) and non-node ids (future edge events).
    prev = st.session_state.get(f"{key}_last_click")
    if clicked and clicked != prev and clicked in node_ids_str:
        st.session_state[f"{key}_last_click"] = clicked
        st.rerun()
    return clicked


def _render_search(reader, source_key: str, apply_sentiment: bool) -> None:
    st.markdown(
        "Substring + fuzzy search over concept labels. "
        "Pick a hit to render its **top-N strongest 1-hop connections** "
        "(capped to keep rendering snappy on hub concepts)."
    )
    col_q, col_m = st.columns([4, 1])
    keyword = col_q.text_input(
        "Keyword",
        value="",
        placeholder="e.g. food, policy, restaurant…",
        key="qe_search_kw",
    )
    max_n = col_m.slider("Max neighbours", 5, 50, 20, key="qe_search_max_n")
    if not keyword.strip():
        st.caption("Enter a keyword to search.")
        return

    df = reader.search_concepts(keyword.strip(), limit=50)
    if df.empty:
        st.warning("No matches.")
        return

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{len(df)} match(es)**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        pick = st.selectbox(
            "Render ego graph for…",
            options=df["label"].tolist(),
            key="qe_search_pick",
        )
    with right:
        G_full = _ego_subgraph(reader, pick, max_neighbours=max_n)
        seed_id = reader.resolve(pick) or pick
        G_sub = _star_filter(G_full, seed_id)
        dropped = G_full.number_of_edges() - G_sub.number_of_edges()
        dropped_note = (
            f" (dropped {dropped} inter-neighbour edges for clarity)"
            if dropped > 0 else ""
        )
        st.caption(
            f"Subgraph: {G_sub.number_of_nodes()} nodes, "
            f"{G_sub.number_of_edges()} edges (top-{max_n} by edge weight)"
            f"{dropped_note}"
        )
        _render_agraph_subgraph(
            G_sub,
            source_key=source_key,
            apply_sentiment=apply_sentiment,
            key="qe_search_graph",
        )


def _render_neighbourhood(reader, source_key: str, apply_sentiment: bool) -> None:
    st.markdown(
        "Pick a concept and explore its k-hop neighbourhood. "
        "If the result exceeds the node cap, it's pruned to the top by PageRank "
        "(seed always kept)."
    )
    labels = reader.all_labels()
    if not labels:
        st.warning(
            "No concepts found in Neo4j for this source. "
            "Run the pipeline with `--neo4j` first."
        )
        return

    col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
    seed = col_a.selectbox("Concept", options=sorted(labels), key="qe_nb_seed")
    depth = col_b.slider("Depth", 1, 3, 1, key="qe_nb_depth")
    min_w = col_c.slider("Min weight", 1, 10, 1, key="qe_nb_minw")
    max_nodes = col_d.slider("Max nodes", 10, 100, 30, key="qe_nb_max_nodes")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Direct neighbours**")
        nb_df = reader.neighbours(seed, top_n=20, direction="both")
        if nb_df.empty:
            st.info("No direct neighbours for this concept.")
        else:
            st.dataframe(nb_df, use_container_width=True, hide_index=True)
    with right:
        G_full = reader.neighbourhood_subgraph(seed, depth=depth, min_weight=min_w)
        seed_id = reader.resolve(seed) or seed
        G_sub = _prune_subgraph_to_top_pagerank(
            G_full, max_nodes=max_nodes, anchor_id=seed_id,
        )
        pruned_note = (
            f" (pruned from {G_full.number_of_nodes()} by PageRank)"
            if G_sub.number_of_nodes() < G_full.number_of_nodes() else ""
        )
        st.caption(
            f"Subgraph: {G_sub.number_of_nodes()} nodes, "
            f"{G_sub.number_of_edges()} edges{pruned_note}"
        )
        _render_agraph_subgraph(
            G_sub,
            source_key=source_key,
            apply_sentiment=apply_sentiment,
            key="qe_nb_graph",
        )


def _render_top(reader) -> None:
    st.markdown("Top-N concepts by a persisted centrality metric.")
    col_m, col_n = st.columns([1, 1])
    metric = col_m.selectbox(
        "Metric",
        options=["pagerank", "betweenness", "frequency"],
        key="qe_top_metric",
    )
    top_n = col_n.slider("Top N", 5, 100, 30, key="qe_top_n")
    df = reader.top_concepts(metric=metric, top_n=top_n)
    if df.empty:
        st.warning(
            "No values for this metric. "
            "Make sure the pipeline computed and persisted it."
        )
        return
    st.bar_chart(df.set_index("label")[metric])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_neo4j_data_page() -> None:
    """Render the entire Neo4j Data view (sidebar config + main canvas)."""
    with st.sidebar:
        st.header("Neo4j Data config")
        source = st.selectbox(
            "Data source",
            ["combined", "policy", "yelp"],
            index=0,
            key="n4j_source",
        )
        max_nodes = st.slider(
            "Max nodes to display", 5, 100, 20, key="n4j_max_nodes"
        )
        min_weight = st.slider(
            "Min edge weight", 1, 10, 1, key="n4j_min_weight"
        )

        st.divider()
        st.radio(
            "View",
            ["Main Dashboard", "Neo4j Data"],
            key="view_mode",
        )

    st.title("Neo4j Data")
    st.caption(
        "Pulled directly from Neo4j — no in-memory rebuild. "
        "Skips the ingest → extract → graph pipeline that the main "
        "dashboard runs on every reload."
    )

    reader_pair = _get_neo4j_reader(source)
    if reader_pair is None:
        st.warning(
            "Neo4j is not reachable. Start it with `docker compose up -d` "
            f"and populate it via `python pipeline.py --source {source} "
            "--neo4j`, then reload."
        )
        return
    _store, reader = reader_pair

    stats = reader.graph_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total nodes", f"{stats['nodes']:,}")
    c2.metric("Total edges", f"{stats['edges']:,}")
    c3.metric("Communities", f"{stats['communities']:,}")
    c4.metric("Density", f"{stats['density']:.4f}")

    if stats["nodes"] == 0:
        st.warning(
            f"No concepts found for source_label='{source}'. "
            f"Run `python pipeline.py --source {source} --neo4j` first."
        )
        return

    st.header("Network")

    G = reader.load_graph(top_n_by_pagerank=max_nodes)
    if min_weight > 1:
        edges_to_drop = [
            (u, v) for u, v, d in G.edges(data=True)
            if float(d.get("weight", 1) or 1) < min_weight
        ]
        G.remove_edges_from(edges_to_drop)

    st.caption(
        f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} "
        f"edges (top-{max_nodes} by stored PageRank, "
        f"min edge weight {min_weight})"
    )

    _render_agraph_subgraph(
        G,
        source_key=source,
        apply_sentiment=False,
        key="n4j_data_graph",
        height=700,
    )

    st.divider()
    st.header("Query & Explore")

    mode = st.radio(
        "Mode",
        options=["Search concepts", "Concept neighbourhood", "Top concepts"],
        horizontal=True,
        key="qe_mode",
    )
    st.divider()
    if mode == "Search concepts":
        _render_search(reader, source, apply_sentiment=False)
    elif mode == "Concept neighbourhood":
        _render_neighbourhood(reader, source, apply_sentiment=False)
    else:
        _render_top(reader)
