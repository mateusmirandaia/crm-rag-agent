import numpy as np
import pytest

from crm_rag.chunking import Chunk
from crm_rag.store import IndexMismatchError, VectorStore


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"D-{i}#0",
        doc_id=f"D-{i}",
        doc_type="case",
        title=f"Doc {i}",
        header=f"Doc {i}",
        body="body",
    )


@pytest.fixture
def store() -> VectorStore:
    s = VectorStore("test-embedder")
    vectors = np.array([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]])
    s.add([_chunk(i) for i in range(3)], vectors)
    return s


def test_search_ranks_by_cosine(store: VectorStore) -> None:
    results = store.search(np.array([1.0, 0.1, 0.0]), top_k=3)
    assert [r.chunk.doc_id for r in results] == ["D-0", "D-1", "D-2"]
    assert results[0].score == pytest.approx(0.995, abs=1e-3)
    assert results[2].score == pytest.approx(0.0, abs=1e-6)


def test_scores_are_scale_invariant(store: VectorStore) -> None:
    a = store.search(np.array([0.0, 2.0, 0.0]), top_k=1)[0]
    b = store.search(np.array([0.0, 200.0, 0.0]), top_k=1)[0]
    assert a.chunk.id == b.chunk.id
    assert a.score == pytest.approx(b.score)


def test_top_k_larger_than_store(store: VectorStore) -> None:
    assert len(store.search(np.array([1.0, 0.0, 0.0]), top_k=10)) == 3


def test_empty_store_returns_nothing() -> None:
    assert VectorStore("x").search(np.array([1.0, 0.0]), top_k=3) == []


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        VectorStore("x").add([_chunk(0)], np.zeros((2, 3)))


def test_roundtrip(store: VectorStore, tmp_path) -> None:
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, expected_embedder="test-embedder")
    query = np.array([0.2, 0.9, 0.1])
    assert [r.chunk for r in loaded.search(query, 3)] == [r.chunk for r in store.search(query, 3)]


def test_load_refuses_other_embedder(store: VectorStore, tmp_path) -> None:
    store.save(tmp_path)
    with pytest.raises(IndexMismatchError):
        VectorStore.load(tmp_path, expected_embedder="another-model")
