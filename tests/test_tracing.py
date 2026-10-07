"""Tracing tests: what actually leaves the process, captured by an in-memory OTel exporter.

The canaries are words that occur only in the question, the retrieved passages, the graph
facts and the answer. In metadata mode none of them may reach any exported span attribute;
in full mode they must, minus anything credential-shaped.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langfuse import get_client
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from graph_rag import api, tracing
from graph_rag.huggingface import HuggingFaceModels
from tests.test_retrieval import _pipeline

QUESTION = "Where is Alice Johnson's company based?"
CANARIES = ("Alice Johnson", "Acme Labs", "Seattle")
FAKE_HF_TOKEN = "hf_" + "Q" * 34           # built at runtime so secret scanners don't flag the test
FAKE_EMAIL = "alice.johnson@example.org"


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRACE_CONTENT", raising=False)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    LangfuseResourceManager._instances.clear()
    tracing.reset_for_tests()
    spans = InMemorySpanExporter()
    tracing.init_tracing(public_key="pk-lf-test", secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
                         span_exporter=spans, flush_at=1)
    yield spans
    LangfuseResourceManager._instances.clear()
    tracing.reset_for_tests()


def _exported(spans: InMemorySpanExporter) -> list[Any]:
    get_client().flush()
    return list(spans.get_finished_spans())


def _attributes_text(finished: list[Any]) -> str:
    return json.dumps([dict(span.attributes or {}) for span in finished], default=str)


def _by_name(finished: list[Any]) -> dict[str, Any]:
    return {span.name: span for span in finished}


# ------------------------------------------------------------------ content gate
@pytest.mark.parametrize("value, expected", [
    (None, "metadata"), ("", "metadata"), ("metadata", "metadata"), ("full", "full"),
    (" FULL ", "full"), ("ful", "metadata"), ("true", "metadata"), ("all", "metadata"),
])
def test_content_mode_fails_closed(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str) -> None:
    if value is None:
        monkeypatch.delenv("TRACE_CONTENT", raising=False)
    else:
        monkeypatch.setenv("TRACE_CONTENT", value)
    assert tracing.content_mode() == expected


def test_metadata_mode_exports_no_question_passage_fact_or_answer_text(exporter) -> None:
    result = _pipeline().query_result(QUESTION, limit=4)
    finished = _exported(exporter)
    text = _attributes_text(finished)

    for canary in CANARIES:
        assert canary not in text, canary
    names = _by_name(finished)
    assert {"graph-rag.query", "graph-rag.retrieve", "retrieve.embedding", "retrieve.entity_extraction",
            "retrieve.entity_linking", "retrieve.vector_search", "retrieve.graph_retrieval",
            "retrieve.reranking", "retrieve.graph_expansion"} <= set(names)
    assert "sample.md" in text                       # IDs and scores are metadata, so they are exported
    root = dict(names["graph-rag.query"].attributes or {})
    assert "langfuse.observation.input" not in root and "langfuse.observation.output" not in root
    assert root["langfuse.observation.metadata.trace_content"] == "metadata"
    assert result["answer"] == "Seattle [sample.md]"   # the gate changes what is traced, not what is answered


def test_all_spans_share_one_trace_and_stage_spans_nest_under_retrieval(exporter) -> None:
    result = _pipeline().query_result(QUESTION, limit=4)
    finished = _exported(exporter)
    names = _by_name(finished)

    assert len({span.context.trace_id for span in finished}) == 1
    assert result["trace_id"] == format(names["graph-rag.query"].context.trace_id, "032x")
    retrieve_id = names["graph-rag.retrieve"].context.span_id
    for stage in ("embedding", "entity_extraction", "entity_linking", "vector_search", "graph_expansion"):
        # the first three run in worker threads; contextvars must be copied for them to nest
        assert names[f"retrieve.{stage}"].parent.span_id == retrieve_id, stage
    assert names["graph-rag.retrieve"].parent.span_id == names["graph-rag.query"].context.span_id


def test_streaming_query_is_traced_with_its_answer_gated(exporter) -> None:
    events = list(_pipeline().query_stream(QUESTION, limit=4))
    finished = _exported(exporter)
    text = _attributes_text(finished)

    assert events[-1]["type"] == "done"
    root = _by_name(finished)["graph-rag.query"]
    assert events[-1]["trace_id"] == format(root.context.trace_id, "032x")
    assert len({span.context.trace_id for span in finished}) == 1
    for canary in CANARIES:
        assert canary not in text, canary
    assert dict(root.attributes or {})["langfuse.observation.metadata.mode"] == "stream"


def test_full_mode_exports_content_but_redacts_credentials_and_emails(exporter, monkeypatch) -> None:
    monkeypatch.setenv("TRACE_CONTENT", "full")
    question = f"{QUESTION} token {FAKE_HF_TOKEN} mail {FAKE_EMAIL}"
    _pipeline().query_result(question, limit=4)
    text = _attributes_text(_exported(exporter))

    for canary in CANARIES:
        assert canary in text, canary
    assert FAKE_HF_TOKEN not in text and FAKE_EMAIL not in text
    assert "[REDACTED:hf-token]" in text and "[REDACTED:email]" in text


def test_mask_truncates_long_strings_and_walks_nested_data() -> None:
    masked = tracing.mask(data={"a": ["x" * (tracing.MAX_TRACE_STRING_CHARS + 50)], "b": f"key sk-lf-{'0' * 32}"})

    assert masked["a"][0].endswith("[truncated 50 chars]")
    assert masked["b"] == "key [REDACTED:langfuse-key]"


# ------------------------------------------------------------------ generation spans
def _models(chat_client: Any) -> HuggingFaceModels:
    models = HuggingFaceModels(token="unused", embedding_model="e", ner_model="n", llm_model="llama3.1")
    models.chat_client = chat_client
    return models


def _usage() -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=120, completion_tokens=9, total_tokens=129)


class _ChatClient:
    def chat_completion(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            assert kwargs["stream_options"] == {"include_usage": True}
            delta = lambda text: SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)
            return iter([delta("Seattle "), delta("[sample.md]"), SimpleNamespace(choices=[], usage=_usage())])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Seattle [sample.md]"))], usage=_usage())


@pytest.mark.parametrize("streamed", [False, True])
def test_generation_span_records_model_parameters_and_usage_without_prompt(exporter, streamed: bool) -> None:
    models = _models(_ChatClient())
    answer = "".join(models.answer_stream(QUESTION, "Acme Labs is based in Seattle.")) if streamed \
        else models.answer(QUESTION, "Acme Labs is based in Seattle.")
    finished = _exported(exporter)
    generation = _by_name(finished)["llm.answer"]
    attributes = dict(generation.attributes or {})

    assert answer == "Seattle [sample.md]"
    assert attributes["langfuse.observation.type"] == "generation"
    assert attributes["langfuse.observation.model.name"] == "llama3.1"
    assert json.loads(attributes["langfuse.observation.usage_details"]) == {"input": 120, "output": 9, "total": 129}
    if streamed:
        assert "langfuse.observation.completion_start_time" in attributes
    text = _attributes_text(finished)
    for canary in CANARIES:
        assert canary not in text, canary
    assert "You answer questions" not in text        # the system prompt is content too


def test_generation_span_carries_the_prompt_in_full_mode(exporter, monkeypatch) -> None:
    monkeypatch.setenv("TRACE_CONTENT", "full")
    _models(_ChatClient()).answer(QUESTION, "Acme Labs is based in Seattle.")
    attributes = dict(_by_name(_exported(exporter))["llm.answer"].attributes or {})

    assert "Acme Labs is based in Seattle." in attributes["langfuse.observation.input"]
    assert "Seattle [sample.md]" in attributes["langfuse.observation.output"]


# ------------------------------------------------------------------ API error hygiene
class _BrokenPipeline:
    def query_result(self, question: str, limit: int) -> dict[str, Any]:
        raise RuntimeError("bolt://neo4j-internal.corp:7687 refused; NEO4J_PASSWORD wrong")

    def query_stream(self, question: str, limit: int):
        yield {"type": "evidence"}
        raise ValueError("qdrant at http://10.0.0.7:6333 returned 500")


def test_api_errors_do_not_leak_internal_details(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_pipeline", lambda: _BrokenPipeline())
    client = TestClient(api.app)

    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 503
    assert "neo4j-internal" not in response.text and "NEO4J_PASSWORD" not in response.text

    stream = client.post("/query/stream", json={"question": "q"})
    assert '"type": "error"' in stream.text
    assert "10.0.0.7" not in stream.text and "qdrant" not in stream.text


def test_query_response_includes_trace_id(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_pipeline", lambda: _pipeline())
    body = TestClient(api.app).post("/query", json={"question": QUESTION, "limit": 4}).json()

    assert "trace_id" in body
