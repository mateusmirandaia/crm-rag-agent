import re
from dataclasses import dataclass

from crm_rag.prompts import NO_ANSWER
from crm_rag.providers.base import Embedder, Generator
from crm_rag.store import SearchResult, VectorStore

_CITATION = re.compile(r"\[([A-Z]{2,4}-\d{3,5})\]")


@dataclass(frozen=True)
class Source:
    id: str
    type: str
    title: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    grounded: bool


class RagAgent:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore,
        min_score: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.store = store
        self.min_score = embedder.default_min_score if min_score is None else min_score

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        """Best chunk per record, `top_k` distinct records.

        Over-fetching then deduplicating stops one long article from filling every
        slot with its own chunks and crowding out the case that actually answers.
        """
        query = self.embedder.embed([question])[0]
        best: dict[str, SearchResult] = {}
        for result in self.store.search(query, top_k * 3):
            doc_id = result.chunk.doc_id
            if doc_id not in best or result.score > best[doc_id].score:
                best[doc_id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]

    def ask(self, question: str, top_k: int = 5) -> Answer:
        context = [r for r in self.retrieve(question, top_k) if r.score >= self.min_score]
        # Refusing here, before the LLM call, is the cheapest and most reliable guard
        # against answers built on vaguely related records.
        if not context:
            return Answer(NO_ANSWER, [], grounded=False)

        text = self.generator.generate(question, context)
        if not text or text.startswith("I don't know"):
            return Answer(text or NO_ANSWER, [], grounded=False)

        by_id = {r.chunk.doc_id: r for r in context}
        cited = list(dict.fromkeys(m for m in _CITATION.findall(text) if m in by_id))
        # Citations that point outside the context are hallucinated; drop them from
        # the text rather than return a source the user cannot verify.
        text = _CITATION.sub(lambda m: m.group(0) if m.group(1) in by_id else "", text)
        used = [by_id[i] for i in cited] if cited else context
        sources = [
            Source(r.chunk.doc_id, r.chunk.doc_type, r.chunk.title, round(r.score, 4)) for r in used
        ]
        return Answer(" ".join(text.split()), sources, grounded=bool(cited))
