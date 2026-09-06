from graph_rag.huggingface import HuggingFaceModels
from graph_rag.production import IngestionCheckpoint, IngestionService, InMemoryJobQueue, rerank_evidence, should_refuse_answer
from graph_rag.stores import Chunk, Neo4jStore, QdrantStore


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


def test_checkpoint_round_trips_processed_ids(tmp_path) -> None:
    path = tmp_path / "demo.json"
    checkpoint = IngestionCheckpoint(path=path)
    checkpoint.mark_processed("doc-1")
    checkpoint.mark_processed("doc-2")

    assert IngestionCheckpoint(path=path).processed == {"doc-1", "doc-2"}
    assert not path.with_suffix(".json.tmp").exists()


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


def test_embed_batch_sends_one_request_for_many_texts(monkeypatch) -> None:
    calls: list[object] = []

    class FakeClient:
        def feature_extraction(self, text, model: str):
            calls.append(text)
            return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr("graph_rag.huggingface.InferenceClient", lambda token=None: FakeClient())
    models = HuggingFaceModels("token", "embed-model", "ner-model", "llm-model")

    assert models.embed_batch(["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert calls == [["one", "two"]]
    assert models.embed_batch([]) == []


def test_upsert_chunk_unwinds_entities_and_drops_empty_edges(monkeypatch) -> None:
    queries: list[tuple[str, dict]] = []

    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def run(self, query: str, **params) -> None:
            queries.append((query, params))

    monkeypatch.setattr(
        "graph_rag.stores.GraphDatabase",
        type("G", (), {"driver": staticmethod(lambda *a, **k: type("D", (), {"session": lambda self: FakeSession()})())}),
    )
    store = Neo4jStore("bolt://x", "u", "p")
    chunk = Chunk("doc-1", "Alice works at Acme", ["Alice", "Acme"])

    store.upsert_chunk(chunk, [
        {"source": "Alice", "predicate": "WORKS_AT", "target": "Acme", "confidence": 0.9},
        {"source": "", "predicate": "WORKS_AT", "target": "Acme"},
    ])

    assert len(queries) == 2
    assert queries[0][1]["entities"] == ["Alice", "Acme"]
    assert [edge["source"] for edge in queries[1][1]["edges"]] == ["Alice"]


def test_neo4j_facts_returns_structured_rows(monkeypatch) -> None:
    class FakeRecord(dict):
        def __getitem__(self, key): return dict.__getitem__(self, key)

    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def run(self, query: str, **params):
            return [FakeRecord(source="Alice", predicate="WORKS_AT", target="Acme")]

    monkeypatch.setattr(
        "graph_rag.stores.GraphDatabase",
        type("G", (), {"driver": staticmethod(lambda *a, **k: type("D", (), {"session": lambda self: FakeSession()})())}),
    )
    store = Neo4jStore("bolt://x", "u", "p")

    assert store.facts(["Alice"]) == [{"source": "Alice", "predicate": "WORKS_AT", "target": "Acme"}]
    assert store.facts([]) == []


def test_chat_client_defaults_to_hf_client_without_llm_base_url() -> None:
    models = HuggingFaceModels("token", "embed-model", "ner-model", "llm-model")
    assert models.chat_client is models.client


def test_chat_client_points_elsewhere_when_llm_base_url_is_set() -> None:
    models = HuggingFaceModels("token", "embed-model", "ner-model", "llm-model", "http://localhost:11434/v1")
    assert models.chat_client is not models.client


def test_local_models_flag_routes_embed_and_entities_off_hf(monkeypatch) -> None:
    calls = {"hf_embed": 0, "hf_entities": 0}

    class FakeClient:
        def feature_extraction(self, text, model: str):
            calls["hf_embed"] += 1
            return [0.0]

        def token_classification(self, text, model: str):
            calls["hf_entities"] += 1
            return []

    class FakeEmbedder:
        def __init__(self, model, device): pass
        def encode(self, texts):
            return [9.0, 9.0] if isinstance(texts, str) else [[9.0, 9.0] for _ in texts]

    monkeypatch.setattr("graph_rag.huggingface.InferenceClient", lambda token=None, base_url=None, api_key=None: FakeClient())
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda model, device: FakeEmbedder(model, device))
    monkeypatch.setattr("transformers.pipeline", lambda *a, **k: (lambda text: [{"word": "Alice"}]))

    models = HuggingFaceModels("token", "embed-model", "ner-model", "llm-model", local_models=True)

    assert models.embed("hi") == [9.0, 9.0]
    assert models.entities("hi") == ["Alice"]
    assert calls == {"hf_embed": 0, "hf_entities": 0}
