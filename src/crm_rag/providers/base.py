from typing import Protocol

import numpy as np

from crm_rag.store import SearchResult


class Embedder(Protocol):
    # Stored in the index manifest; changing it forces a re-ingest.
    name: str
    # Cosine similarity below which a chunk is treated as unrelated. It lives on the
    # embedder because score distributions differ a lot between models.
    default_min_score: float

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, question: str, context: list[SearchResult]) -> str: ...
