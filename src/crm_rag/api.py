import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from crm_rag import __version__
from crm_rag.agent import RagAgent
from crm_rag.config import Settings, get_settings
from crm_rag.ingest import build_index
from crm_rag.providers import build_providers
from crm_rag.store import IndexMismatchError, VectorStore

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    id: str
    type: str
    title: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    grounded: bool


class IngestResponse(BaseModel):
    chunks: int
    embedder: str


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    index_loaded: bool
    chunks: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        embedder, generator = build_providers(settings)
        app.state.embedder = embedder
        app.state.generator = generator
        app.state.agent = None
        if VectorStore.exists(settings.index_dir):
            try:
                store = VectorStore.load(settings.index_dir, expected_embedder=embedder.name)
                app.state.agent = RagAgent(embedder, generator, store, settings.min_score)
            except IndexMismatchError as exc:
                # Keep serving /health and /ingest so the index can be rebuilt in place.
                log.warning("%s", exc)
        yield

    app = FastAPI(title="CRM RAG Agent", version=__version__, lifespan=lifespan)

    def current_agent(request: Request) -> RagAgent:
        agent: RagAgent | None = request.app.state.agent
        if agent is None:
            raise HTTPException(status_code=503, detail="Index not built. Call POST /ingest.")
        return agent

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        agent: RagAgent | None = request.app.state.agent
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=settings.llm_provider,
            index_loaded=agent is not None,
            chunks=len(agent.store) if agent else 0,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: Request) -> IngestResponse:
        state = request.app.state
        store = build_index(settings, state.embedder)
        # Swap the agent only after the new index is complete; in-flight requests
        # keep using the old one.
        state.agent = RagAgent(state.embedder, state.generator, store, settings.min_score)
        return IngestResponse(chunks=len(store), embedder=state.embedder.name)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, agent: RagAgent = Depends(current_agent)) -> AskResponse:
        result = agent.ask(body.question, body.top_k or settings.top_k)
        return AskResponse(
            answer=result.answer,
            sources=[SourceOut(**asdict(s)) for s in result.sources],
            grounded=result.grounded,
        )

    return app


app = create_app()
