"""Offline provider: no network, no keys, deterministic.

It exists so the pipeline, the API and the eval can run anywhere (CI included).
Answer quality is deliberately modest; retrieval quality is what it lets us measure.
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from crm_rag.chunking import split_sentences
from crm_rag.store import SearchResult


class HashingEmbedder:
    """Word and character n-gram hashing, concatenated.

    Hashing is stateless, so queries embed the same way as documents without
    persisting a fitted vocabulary. Character n-grams make "sync", "syncing" and
    "synced" overlap, which plain word features would miss.
    """

    name = "local-hashing-v1"
    default_min_score = 0.15

    def __init__(self, n_features: int = 2**15) -> None:
        common = {"n_features": n_features, "alternate_sign": False, "norm": "l2"}
        self._words = HashingVectorizer(
            analyzer="word", ngram_range=(1, 2), stop_words="english", **common
        )
        self._chars = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)

    def embed(self, texts: list[str]) -> np.ndarray:
        words = self._words.transform(texts).toarray()
        chars = self._chars.transform(texts).toarray()
        # Equal weights keep the concatenated vector unit length.
        return np.hstack([words, chars]).astype(np.float32) * np.float32(np.sqrt(0.5))


class ExtractiveGenerator:
    """Quotes the sentences from the top records that are closest to the question.

    No paraphrasing means no hallucination, at the cost of fluency. Sentence
    similarity reuses the hashing embedder so it agrees with retrieval.
    """

    def __init__(
        self,
        embedder: HashingEmbedder | None = None,
        max_records: int = 2,
        sentences_per_record: int = 2,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.max_records = max_records
        self.sentences_per_record = sentences_per_record

    def generate(self, question: str, context: list[SearchResult]) -> str:
        query = self.embedder.embed([question])[0]
        parts: list[str] = []
        for result in context[: self.max_records]:
            sentences = split_sentences(result.chunk.body)
            if not sentences:
                continue
            scores = self.embedder.embed(sentences) @ query
            best = sorted(np.argsort(-scores)[: self.sentences_per_record])
            quoted = " ".join(sentences[i] for i in best)
            parts.append(f"{result.chunk.title} [{result.chunk.doc_id}]: {quoted}")
        return "\n".join(parts)
