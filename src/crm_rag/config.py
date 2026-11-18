from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["local", "azure"] = "local"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    data_dir: Path = Path("data")
    index_dir: Path = Path(".index")
    top_k: int = Field(default=5, ge=1, le=50)
    # None means "use the embedder's own default". Cosine scores are not comparable
    # across embedding models, so a single global number would be wrong for one of them.
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    @field_validator("min_score", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        # An empty MIN_SCORE= line in .env should mean "unset", not a parse error.
        return None if value == "" else value

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.llm_provider == "azure" and not (
            self.azure_openai_endpoint and self.azure_openai_api_key
        ):
            raise ValueError(
                "LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
