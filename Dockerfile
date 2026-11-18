FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Build the sample index at image build time only in local mode; with Azure the
# index should be built against the real embedding deployment at runtime.
ARG LLM_PROVIDER=local
ENV LLM_PROVIDER=${LLM_PROVIDER}
RUN if [ "$LLM_PROVIDER" = "local" ]; then python -m crm_rag.ingest; fi

EXPOSE 8000
CMD ["uvicorn", "crm_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
