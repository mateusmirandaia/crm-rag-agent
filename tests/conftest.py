from pathlib import Path

import pytest

from crm_rag.config import Settings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # _env_file=None keeps a developer's local .env from leaking into tests.
    return Settings(
        _env_file=None, llm_provider="local", data_dir=DATA_DIR, index_dir=tmp_path / "index"
    )
