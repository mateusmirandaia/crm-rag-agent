import numpy as np
from openai import AzureOpenAI

from crm_rag.config import Settings
from crm_rag.prompts import SYSTEM_PROMPT, build_user_prompt
from crm_rag.store import SearchResult


def make_client(settings: Settings) -> AzureOpenAI:
    if not settings.azure_openai_endpoint or settings.azure_openai_api_key is None:
        raise ValueError("Azure OpenAI endpoint and key must be configured")
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        max_retries=3,
    )


class AzureOpenAIEmbedder:
    # text-embedding-3-* cosine scores sit higher than the hashing embedder's;
    # 0.30 is a starting point to recalibrate with eval/run_eval.py on real data.
    default_min_score = 0.30

    def __init__(self, client: AzureOpenAI, deployment: str, batch_size: int = 64) -> None:
        self._client = client
        self._deployment = deployment
        self._batch_size = batch_size
        self.name = f"azure-openai:{deployment}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._deployment, input=batch)
            # The API documents index-ordered output, but sorting is cheap insurance.
            vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
        return np.asarray(vectors, dtype=np.float32)


class AzureOpenAIGenerator:
    def __init__(self, client: AzureOpenAI, deployment: str, max_tokens: int = 500) -> None:
        self._client = client
        self._deployment = deployment
        self._max_tokens = max_tokens

    def generate(self, question: str, context: list[SearchResult]) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
        )
        return (response.choices[0].message.content or "").strip()
