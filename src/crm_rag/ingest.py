"""Build the vector index: `python -m crm_rag.ingest [--data-dir data] [--index-dir .index]`."""

import argparse
import logging
import time
from pathlib import Path

from crm_rag.chunking import chunk_documents
from crm_rag.config import Settings, get_settings
from crm_rag.loaders import JsonDirectoryLoader, Loader
from crm_rag.providers import build_providers
from crm_rag.providers.base import Embedder
from crm_rag.store import VectorStore

log = logging.getLogger(__name__)


def build_index(
    settings: Settings, embedder: Embedder, loader: Loader | None = None
) -> VectorStore:
    loader = loader or JsonDirectoryLoader(settings.data_dir)
    docs = list(loader.load())
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store = VectorStore(embedder.name)
    if chunks:
        store.add(chunks, embedder.embed([c.text for c in chunks]))
    # Write the full index in one go at the end, so a failed embedding call never
    # leaves a half-built index on disk next to a valid manifest.
    store.save(settings.index_dir)
    log.info(
        "indexed %d documents as %d chunks into %s", len(docs), len(chunks), settings.index_dir
    )
    return store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index CRM data for retrieval.")
    parser.add_argument("--data-dir", type=Path, help="directory with the JSON export")
    parser.add_argument("--index-dir", type=Path, help="where to write the index")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = {
        k: v for k, v in {"data_dir": args.data_dir, "index_dir": args.index_dir}.items() if v
    }
    settings = get_settings().model_copy(update=overrides)
    embedder, _ = build_providers(settings)

    started = time.perf_counter()
    store = build_index(settings, embedder)
    print(
        f"Indexed {len(store)} chunks with {embedder.name} "
        f"in {time.perf_counter() - started:.2f}s -> {settings.index_dir}"
    )


if __name__ == "__main__":
    main()
