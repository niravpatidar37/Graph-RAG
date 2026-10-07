from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import tracing
from .brand import FAVICON_SVG
from .pipeline import CloudGraphRAG
from .ui import PAGE, SECURITY_HEADERS

logger = logging.getLogger("graph_rag.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Spans are exported in background batches; flush them before the process exits.
    tracing.shutdown()


app = FastAPI(title="Graph RAG API", version="0.2.0", lifespan=lifespan)

MAX_QUESTION_CHARS = 2000


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> Response:
    # Static, build-time constant (see docs/brand/tools/build_brand.py, which rejects scripts,
    # event handlers and external references); nosniff keeps browsers from reinterpreting it.
    return Response(
        FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(PAGE, headers=SECURITY_HEADERS)


class QueryRequest(BaseModel):
    # Capped so a single request can't push megabytes through NER, embedding and the prompt.
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    limit: int = Field(default=8, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    graph_facts: list[str]
    retrieved_entities: list[str]
    graph: dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str = ""


@lru_cache(maxsize=1)
def get_pipeline() -> CloudGraphRAG:
    return CloudGraphRAG.from_env()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return get_pipeline().metrics.snapshot()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse(**get_pipeline().query_result(request.question, request.limit))
    except RuntimeError as error:
        # Details go to the server log, not the client: they can name hosts, models and config keys.
        logger.exception("query failed")
        raise HTTPException(status_code=503, detail="The Graph RAG backend is unavailable.") from error


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    def events():
        try:
            for event in get_pipeline().query_stream(request.question, request.limit):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:  # a broken stream must still close with an error event, not a silent truncation
            logger.exception("streaming query failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'The answer stream failed.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def main() -> None:
    import os

    import uvicorn

    # Loopback by default: the API has no authentication, so exposing it on every interface
    # should be a deliberate choice (GRAPH_RAG_HOST=0.0.0.0 inside a container, behind a proxy).
    host = os.environ.get("GRAPH_RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("GRAPH_RAG_PORT", "8000"))
    uvicorn.run("graph_rag.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
