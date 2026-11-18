"""Paragraph-aware chunking.

CRM text is short and structured: a case is a header, a problem and a resolution;
a KB article is a few paragraphs. Splitting on paragraph and sentence boundaries
keeps each chunk readable as a citation, which matters more here than squeezing
chunks to an exact token count.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from crm_rag.loaders import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9'\"])")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_type: str
    title: str
    header: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """What gets embedded and shown to the LLM: labelled, self-contained."""
        return f"{self.header}\n{self.body}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pieces(text: str, chunk_size: int) -> Iterator[str]:
    """Yield paragraphs, falling back to sentences (and finally hard cuts) when too long."""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            yield paragraph
            continue
        for sentence in split_sentences(paragraph):
            while len(sentence) > chunk_size:
                yield sentence[:chunk_size]
                sentence = sentence[chunk_size:]
            if sentence:
                yield sentence


def _tail(text: str, overlap: int) -> str:
    """Trailing whole sentences of `text` that fit in `overlap` characters."""
    if overlap <= 0:
        return ""
    kept: list[str] = []
    size = 0
    for sentence in reversed(split_sentences(text)):
        if size + len(sentence) > overlap:
            break
        kept.insert(0, sentence)
        size += len(sentence) + 1
    return " ".join(kept)


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    # Each chunk is prefixed with the record title and header so a chunk taken from
    # the middle of an article still says what it is about, both to the embedder and
    # to the LLM. The prefix counts against the size budget.
    header = f"{doc.title}. {doc.header}".strip() if doc.header else doc.title
    budget = max(chunk_size - len(header) - 1, 1)

    bodies: list[str] = []
    current = ""
    for piece in _pieces(doc.text, budget):
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= budget:
            current = candidate
            continue
        bodies.append(current)
        carry = _tail(current, overlap)
        current = f"{carry} {piece}".strip() if carry else piece
        if len(current) > budget:
            current = piece
    if current:
        bodies.append(current)

    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            doc_type=doc.type,
            title=doc.title,
            header=header,
            body=body,
            metadata=doc.metadata,
        )
        for i, body in enumerate(bodies)
    ]


def chunk_documents(
    docs: Iterable[Document], chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, chunk_size, overlap)]
