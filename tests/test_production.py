import json

from graph_rag.huggingface import HuggingFaceModels
from graph_rag.production import IngestionCheckpoint, IngestionService, InMemoryJobQueue, rerank_evidence, should_refuse_answer
from graph_rag.stores import Chunk, QdrantStore


def test_rerank_prefers_exact_entity_matches() -> None:
    hits = [
        {"document": "b.md", "text": "Company updates for the quarter", "score": 0.91, "entities": []},
        {"document": "a.md", "text": "Alice Johnson works at Acme Labs in Seattle.", "score": 0.62, "entities": ["Alice Johnson", "Acme Labs"]},
    ]

    ranked = rerank_evidence("Where is Alice Johnson's company based?", hits, ["Alice Johnson", "Acme Labs"])

    assert ranked[0]["document"] == "a.md"
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_queue_retries_until_success() -> None:
    queue = InMemoryJobQueue()
    attempts = {"count": 0}

    def job() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("temporary")
        return "ok"

    queue.enqueue("demo", job, retries=2)
    result = queue.run_next()

    assert result == "ok"
    assert attempts["count"] == 2


def test_checkpoint_tracks_progress_and_status() -> None:
    checkpoint = IngestionCheckpoint(path="/tmp/demo.json")
    checkpoint.mark_processed("doc-1")
    checkpoint.mark_processed("doc-2")

    assert checkpoint.processed == {"doc-1", "doc-2"}
    assert checkpoint.status == "in_progress"


def test_should_refuse_answer_when_evidence_is_missing() -> None:
    assert should_refuse_answer("", "I cannot answer") is True
    assert should_refuse_answer("Alice works at Acme Labs", "I don't know") is True
    assert should_refuse_answer("Alice works at Acme Labs", "Acme Labs is in Seattle") is False


def test_ingestion_service_marks_processed_after_success(tmp_path) -> None:
    checkpoint = IngestionCheckpoint(path=tmp_path / "checkpoint.json")
    service = IngestionService(checkpoint)
    attempts = {"count": 0}

    def job() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("temporary")
        return "complete"

    service.enqueue("doc-7", job, retries=2)
    results = service.run_pending()

    assert results == ["complete"]
    assert checkpoint.processed == {"doc-7"}
    assert attempts["count"] == 2


def test_redis_queue_produces_serialized_tasks(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRedis:
        def __init__(self) -> None:
            self.items: list[str] = []

        def rpush(self, name: str, value: str) -> None:
            calls.append({"name": name, "value": value})
            self.items.append(value)

        def lpop(self, name: str) -> str | None:
            if not self.items:
                return None
            return self.items.pop(0)

    monkeypatch.setattr("redis.Redis.from_url", lambda *args, **kwargs: FakeRedis())

    from graph_rag.production import RedisJobQueue

    queue = RedisJobQueue(url="redis://localhost:6379/0")
    queue.enqueue("job-1", "demo_task", payload={"a": 1})

    assert calls
    assert json.loads(calls[0]["value"])["task"] == "demo_task"


def test_huggingface_models_reuse_cached_embeddings_and_entities(monkeypatch) -> None:
    calls = {"embed": 0, "entities": 0}

    class FakeClient:
        def feature_extraction(self, text: str, model: str):
            calls["embed"] += 1
            return [0.1, 0.2, 0.3]

        def token_classification(self, text: str, model: str):
            calls["entities"] += 1
            return [{"word": "Alice"}, {"word": "Acme"}]

    monkeypatch.setattr("graph_rag.huggingface.InferenceClient", lambda token=None: FakeClient())
    models = HuggingFaceModels("token", "embed-model", "ner-model", "llm-model")

    assert models.embed("Alice works at Acme") == [0.1, 0.2, 0.3]
    assert models.embed("Alice works at Acme") == [0.1, 0.2, 0.3]
    assert models.entities("Alice works at Acme") == ["Acme", "Alice"]
    assert models.entities("Alice works at Acme") == ["Acme", "Alice"]
    assert calls == {"embed": 1, "entities": 1}


def test_qdrant_store_batches_points_for_bulk_upserts(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.collections: set[str] = set()

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name in self.collections

        def create_collection(self, *args, **kwargs) -> None:
            self.collections.add(kwargs.get("collection_name"))

        def upsert(self, **kwargs) -> None:
            calls.append({"points": kwargs.get("points", []), "collection": kwargs.get("collection_name")})

    monkeypatch.setattr("graph_rag.stores.QdrantClient", lambda *args, **kwargs: FakeClient())
    store = QdrantStore("http://localhost", "key", "demo")
    first = Chunk("doc-1", "hello", ["hello"])
    second = Chunk("doc-2", "world", ["world"])

    store.upsert_batch([(first, [0.1, 0.2]), (second, [0.3, 0.4])])

    assert len(calls) == 1
    assert len(calls[0]["points"]) == 2
