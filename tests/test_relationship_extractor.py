"""Tests for the relationship extractor, focused on verb preservation.

The :class:`RelationshipExtractor` must keep the verb lemma that produced
each ACTION / CAUSATION edge on the resulting :class:`Relationship` so that
the graph builder can aggregate it and the Neo4j store can both (a) persist
``r.top_verb`` on the edge and (b) embed a verb-aware edge description.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import (
    Relationship,
    RelationshipExtractor,
)
from src.preprocessing.tokeniser import SpacyTokeniser


@pytest.fixture(scope="module")
def tokeniser() -> SpacyTokeniser:
    return SpacyTokeniser()


def _concepts(*labels: str) -> list[Concept]:
    return [
        Concept(
            id=lbl.replace(" ", "_"),
            label=lbl,
            type="noun_phrase",
            frequency=1,
            source_docs={"doc0"},
        )
        for lbl in labels
    ]


class TestRelationshipVerbPreservation:
    def test_recommend_verb_captured(self, tokeniser):
        # Both subject and object must match concept labels for the
        # dependency-based extractor to produce an edge. Use plain single
        # tokens so _match_concept's lowercased subtree lookup succeeds.
        doc = tokeniser.process(
            "Customers recommend pizza.",
            doc_id="doc0",
            source="yelp",
        )
        concepts = _concepts("customers", "pizza")
        extractor = RelationshipExtractor(
            use_cooccurrence=False,
            use_dependency=True,
        )
        rels = extractor.extract([doc], concepts)
        action_rels = [r for r in rels if r.type == "ACTION"]
        assert action_rels, "expected at least one ACTION relationship"
        verbs = Counter()
        for r in action_rels:
            verbs.update(r.verbs)
        assert "recommend" in verbs, f"got verbs={verbs!r}"

    def test_cooccurrence_only_has_empty_verbs(self, tokeniser):
        doc = tokeniser.process(
            "Pizza and salad were served on the same plate.",
            doc_id="doc0",
            source="yelp",
        )
        concepts = _concepts("pizza", "salad")
        extractor = RelationshipExtractor(
            use_cooccurrence=True,
            use_dependency=False,
        )
        rels = extractor.extract([doc], concepts)
        assoc_rels = [r for r in rels if r.type == "ASSOCIATION"]
        if assoc_rels:
            for r in assoc_rels:
                assert isinstance(r.verbs, Counter)
                assert sum(r.verbs.values()) == 0

    def test_relationship_dataclass_default_verbs(self):
        """Default-constructed Relationship has an empty Counter."""
        r = Relationship(
            source_id="a", target_id="b", type="ASSOCIATION",
        )
        assert isinstance(r.verbs, Counter)
        assert len(r.verbs) == 0


class TestGraphBuilderVerbAggregation:
    def test_verbs_merge_on_edge(self):
        """When two Relationship rows share (src, tgt, type), verbs combine."""
        from src.network.graph_builder import GraphBuilder

        builder = GraphBuilder(min_edge_weight=1)
        concepts = _concepts("pizza", "customer")
        rels = [
            Relationship(
                source_id="customer", target_id="pizza", type="ACTION",
                weight=2, directed=True,
                verbs=Counter({"recommend": 2}),
            ),
            Relationship(
                source_id="customer", target_id="pizza", type="ACTION",
                weight=1, directed=True,
                verbs=Counter({"recommend": 1, "order": 3}),
            ),
        ]
        G = builder.build(concepts, rels)
        assert G.has_edge("customer", "pizza")
        data = G.get_edge_data("customer", "pizza")
        merged: Counter = data["verbs"]
        assert merged["recommend"] == 3
        assert merged["order"] == 3
