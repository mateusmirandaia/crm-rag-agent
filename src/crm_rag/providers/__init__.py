from crm_rag.config import Settings
from crm_rag.providers.base import Embedder, Generator


def build_providers(settings: Settings) -> tuple[Embedder, Generator]:
    if settings.llm_provider == "azure":
        # Imported lazily so local mode never touches the openai package.
        from crm_rag.providers.azure_openai import (
            AzureOpenAIEmbedder,
            AzureOpenAIGenerator,
            make_client,
        )

        client = make_client(settings)
        return (
            AzureOpenAIEmbedder(client, settings.azure_openai_embedding_deployment),
            AzureOpenAIGenerator(client, settings.azure_openai_chat_deployment),
        )

    from crm_rag.providers.local import ExtractiveGenerator, HashingEmbedder

    embedder = HashingEmbedder()
    return embedder, ExtractiveGenerator(embedder)


__all__ = ["Embedder", "Generator", "build_providers"]
