from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from crm_rag.api import create_app


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_before_ingest(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "local"
    assert body["index_loaded"] is False


def test_ask_without_index_is_503(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "anything at all"})
    assert response.status_code == 503


def test_ingest_then_ask(client: TestClient) -> None:
    ingest = client.post("/ingest").json()
    assert ingest["chunks"] >= 45
    assert ingest["embedder"] == "local-hashing-v1"
    assert client.get("/health").json()["index_loaded"] is True

    response = client.post(
        "/ask",
        json={
            "question": "Which customer had gateways go silent after the firmware 4.2.0 update?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"], body
    first = body["sources"][0]
    assert set(first) == {"id", "type", "title", "score"}
    assert any(s["id"] in {"CAS-1001", "CAS-1011", "KB-0001"} for s in body["sources"])


def test_index_survives_restart(settings) -> None:
    with TestClient(create_app(settings)) as c:
        c.post("/ingest")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["index_loaded"] is True


def test_validation_rejects_bad_top_k(client: TestClient) -> None:
    client.post("/ingest")
    response = client.post("/ask", json={"question": "sync issues", "top_k": 0})
    assert response.status_code == 422
