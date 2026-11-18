"""In-memory cosine vector store persisted as .npy + JSON.

A few thousand CRM chunks fit comfortably in memory and a brute-force dot product
over them takes well under a millisecond, so an ANN index or a vector database
would add operational weight without a measurable benefit at this size.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from crm_rag.chunking import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class IndexMismatchError(RuntimeError):
    """The index on disk was built with a different embedder than the one in use."""


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class VectorStore:
    def __init__(self, embedder_name: str) -> None:
        self.embedder_name = embedder_name
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        normalized = _normalize(vectors)
        if self._vectors is not None and normalized.shape[1] != self._vectors.shape[1]:
            raise ValueError("vector dimension does not match the existing index")
        self._vectors = (
            normalized if self._vectors is None else np.vstack([self._vectors, normalized])
        )
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self._vectors is None or top_k <= 0:
            return []
        query = _normalize(query_vector)[0]
        scores = self._vectors @ query
        k = min(top_k, len(scores))
        # argpartition avoids sorting the whole array; only the k winners get sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), np.float32)
        np.save(directory / VECTORS_FILE, vectors)
        (directory / CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "embedder": self.embedder_name,
            "chunks": len(self._chunks),
            "dimensions": int(vectors.shape[1]) if vectors.size else 0,
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_embedder: str | None = None) -> "VectorStore":
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        # Querying an index with vectors from another model returns confident-looking
        # garbage rather than an error, so refuse it loudly instead.
        if expected_embedder and manifest["embedder"] != expected_embedder:
            raise IndexMismatchError(
                f"index at {directory} was built with '{manifest['embedder']}', "
                f"current embedder is '{expected_embedder}'; re-run ingest"
            )
        store = cls(manifest["embedder"])
        chunks = [Chunk(**c) for c in json.loads((directory / CHUNKS_FILE).read_text("utf-8"))]
        if chunks:
            store.add(chunks, np.load(directory / VECTORS_FILE))
        return store

    @staticmethod
    def exists(directory: Path) -> bool:
        return (Path(directory) / MANIFEST_FILE).exists()
