import pytest

from crm_rag.agent import RagAgent
from crm_rag.ingest import build_index
from crm_rag.prompts import NO_ANSWER
from crm_rag.providers import build_providers
from crm_rag.store import SearchResult


@pytest.fixture
def agent(settings) -> RagAgent:
    embedder, generator = build_providers(settings)
    return RagAgent(embedder, generator, build_index(settings, embedder))


class StubGenerator:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, question: str, context: list[SearchResult]) -> str:
        self.calls += 1
        return self.reply


def test_answers_with_cited_sources(agent: RagAgent) -> None:
    result = agent.ask("How do I fix 'Invalid SAML response signature' errors?")
    assert result.grounded
    ids = [s.id for s in result.sources]
    assert "KB-0002" in ids
    assert "[KB-0002]" in result.answer


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "Give me a recipe for chocolate cake",
    ],
)
def test_off_topic_question_gets_dont_know(agent: RagAgent, question: str) -> None:
    result = agent.ask(question)
    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert not result.grounded


def test_threshold_skips_generator_entirely(agent: RagAgent) -> None:
    stub = StubGenerator("should not be called")
    strict = RagAgent(agent.embedder, stub, agent.store, min_score=0.99)
    assert strict.ask("Why did gateways stop reporting?").answer == NO_ANSWER
    assert stub.calls == 0


def test_hallucinated_citations_are_removed(agent: RagAgent) -> None:
    stub = StubGenerator("Roll back the firmware [KB-0001]. Also see [KB-4242].")
    result = RagAgent(agent.embedder, stub, agent.store).ask(
        "How do I roll back Gateway G2 firmware?"
    )
    assert "[KB-4242]" not in result.answer
    assert [s.id for s in result.sources] == ["KB-0001"]


def test_retrieve_returns_distinct_records(agent: RagAgent) -> None:
    results = agent.retrieve("gateway firmware rollback", top_k=5)
    doc_ids = [r.chunk.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids)) == 5
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
