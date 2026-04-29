"""Extract and classify relationships between concepts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from src.config import CO_OCCURRENCE_WINDOW, CAUSAL_VERBS
from src.extraction.concept_extractor import Concept
from src.preprocessing.tokeniser import ProcessedDocument


@dataclass
class Relationship:
    """A typed, weighted relationship between two concepts.

    ``verbs`` is a Counter of verb lemmas observed when this relationship was
    extracted via dependency parsing. Co-occurrence-only relationships (type
    = ``ASSOCIATION``) leave this empty. Downstream code uses it to pick a
    ``top_verb`` for display and edge embeddings.
    """
    source_id: str
    target_id: str
    type: str          # ASSOCIATION, ACTION, CAUSATION
    weight: float = 1.0
    directed: bool = False
    source_docs: set[str] = field(default_factory=set)
    verbs: Counter = field(default_factory=Counter)


class RelationshipExtractor:
    """Extract relationships using co-occurrence and dependency parsing."""

    def __init__(
        self,
        use_cooccurrence: bool = True,
        use_dependency: bool = True,
        window_size: int = CO_OCCURRENCE_WINDOW,
        causal_verbs: set[str] | None = None,
    ):
        self.use_cooccurrence = use_cooccurrence
        self.use_dependency = use_dependency
        self.window_size = window_size
        self.causal_verbs = causal_verbs or CAUSAL_VERBS

    def extract(
        self,
        documents: list[ProcessedDocument],
        concepts: list[Concept],
    ) -> list[Relationship]:
        """Extract relationships between the given concepts across all documents."""
        concept_labels = {c.label for c in concepts}
        concept_id_map = {c.label: c.id for c in concepts}

        edge_counter: Counter[tuple[str, str, str]] = Counter()
        edge_docs: dict[tuple[str, str, str], set[str]] = {}
        edge_verbs: dict[tuple[str, str, str], Counter] = {}

        for doc in documents:
            if self.use_cooccurrence:
                self._extract_cooccurrence(
                    doc, concept_labels, concept_id_map, edge_counter, edge_docs
                )
            if self.use_dependency:
                self._extract_dependency(
                    doc, concept_labels, concept_id_map,
                    edge_counter, edge_docs, edge_verbs,
                )

        relationships = []
        for (src, tgt, rtype), weight in edge_counter.items():
            relationships.append(Relationship(
                source_id=src,
                target_id=tgt,
                type=rtype,
                weight=weight,
                directed=(rtype != "ASSOCIATION"),
                source_docs=edge_docs.get((src, tgt, rtype), set()),
                verbs=edge_verbs.get((src, tgt, rtype), Counter()),
            ))

        relationships.sort(key=lambda r: r.weight, reverse=True)
        return relationships

    def _extract_cooccurrence(
        self,
        doc: ProcessedDocument,
        concept_labels: set[str],
        id_map: dict[str, str],
        counter: Counter,
        docs_map: dict,
    ) -> None:
        """Link concepts co-occurring within a sliding sentence window."""
        n_sents = len(doc.sentences)
        for i in range(n_sents):
            window_sents = doc.sentences[i: i + self.window_size]
            window_text = " ".join(s.text.lower() for s in window_sents)

            present = [c for c in concept_labels if c in window_text]
            for a, b in combinations(sorted(present), 2):
                key = (id_map[a], id_map[b], "ASSOCIATION")
                counter[key] += 1
                docs_map.setdefault(key, set()).add(doc.doc_id)

    def _extract_dependency(
        self,
        doc: ProcessedDocument,
        concept_labels: set[str],
        id_map: dict[str, str],
        counter: Counter,
        docs_map: dict,
        verbs_map: dict,
    ) -> None:
        """Extract SVO triples from dependency parses and classify by verb type."""
        spacy_doc = doc.spacy_doc
        if spacy_doc is None:
            return

        for sent in spacy_doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects = [
                    child for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                ]
                objects = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "pobj", "attr", "oprd")
                ]

                for subj in subjects:
                    subj_text = self._match_concept(subj, concept_labels)
                    if not subj_text:
                        continue
                    for obj in objects:
                        obj_text = self._match_concept(obj, concept_labels)
                        if not obj_text or subj_text == obj_text:
                            continue

                        verb_lemma = token.lemma_.lower()
                        if verb_lemma in self.causal_verbs:
                            rtype = "CAUSATION"
                        else:
                            rtype = "ACTION"

                        key = (id_map[subj_text], id_map[obj_text], rtype)
                        counter[key] += 1
                        docs_map.setdefault(key, set()).add(doc.doc_id)
                        verbs_map.setdefault(key, Counter())[verb_lemma] += 1

    @staticmethod
    def _match_concept(token, concept_labels: set[str]) -> str | None:
        """Check if a token (or its subtree chunk) matches a known concept."""
        chunk_text = " ".join(t.text for t in token.subtree).lower().strip()
        if chunk_text in concept_labels:
            return chunk_text
        token_text = token.text.lower().strip()
        if token_text in concept_labels:
            return token_text
        return None
