"""Retrieval behaviour: entity linking, seed ordering, graph expansion, reranking, context."""
from __future__ import annotations

from typing import Any

from graph_rag.dataset_ingest import batches
from graph_rag.pipeline import CloudGraphRAG, build_context, build_graph, merge_hits, order_seeds
from graph_rag.production import IngestionCheckpoint, MetricsCollector, rerank_evidence, should_refuse_answer
from graph_rag.stores import Neo4jStore, fulltext_query, name_coverage


# ---------------------------------------------------------------- entity linking
def test_fulltext_query_keeps_only_word_tokens() -> None:
    hostile = 'title:"x" OR *:* AND foo~2 \\ /re.*gex/ (a) [b TO c] {d} ^3 && || ! name:' + "?" * 5
    query = fulltext_query(hostile)

    assert query == "title foo re gex name"
    for char in ':*~"\\/()[]{}^&|!?':
        assert char not in query
    assert query == query.lower()               # Lucene's AND/OR/NOT/TO operators are uppercase-only


def test_fulltext_query_keeps_unicode_names_and_drops_stop_words() -> None:
    assert fulltext_query("Which paper did Lluís García-Pueyo write?") == "paper did lluís garcía pueyo write"


def test_name_coverage_rejects_titles_that_only_share_one_word() -> None:
    terms = set(fulltext_query("Which papers in this dataset were published by NASA?").split())

    assert name_coverage("Papeos: Augmenting Research Papers with Talk Videos", terms) < 0.75
    assert name_coverage("NASA", terms) == 1.0


# ---------------------------------------------------------------- seeds and graph payload
def test_order_seeds_prefers_question_entities_over_alphabetical_noise() -> None:
    hits = [{"entities": ['"Quoted" title', "Acme Labs"]}, {"entities": ["Zeta"]}]

    seeds = order_seeds([], ["Alice Johnson", "A"], hits)

    assert seeds[0] == "Alice Johnson"          # question entity first, not alphabetical
    assert "A" not in seeds                     # NER fragment dropped
    assert seeds[1:] == ['"Quoted" title', "Acme Labs", "Zeta"]


def test_order_seeds_skips_hit_entities_when_question_names_a_graph_entity() -> None:
    hits = [{"entities": ["New York University"]}]

    assert order_seeds(["A Mathematical Framework"], ["Anthropic"], hits) == ["A Mathematical Framework", "Anthropic"]


def test_build_graph_assigns_roles_and_keeps_edge_direction() -> None:
    rows = [
        {"source": "Paper", "predicate": "AFFILIATED_WITH", "target": "Anthropic", "hop": 1},
        {"source": "Author", "predicate": "AUTHORED", "target": "Paper", "hop": 2},
    ]
    hits = [{"document": "doc-1", "entities": ["Paper"]}]

    graph = build_graph(["Paper", "Unused seed"], ["Paper"], rows, hits)
    roles = {node["id"]: node["role"] for node in graph["nodes"]}

    assert roles == {"Paper": "linked", "Anthropic": "hop1", "Author": "hop2"}
    assert graph["nodes"][0]["sources"] == ["doc-1"]
    assert all(node["label"] == node["id"] for node in graph["nodes"])
    assert graph["edges"][1] == {"source": "Author", "target": "Paper", "predicate": "AUTHORED", "hop": 2}


def test_build_graph_labels_paper_id_nodes_with_their_title() -> None:
    rows = [{"source": "https://openalex.org/W1", "predicate": "HAS_TITLE", "target": "Reliable AI", "hop": 1}]

    graph = build_graph([], [], rows + [{"source": "https://openalex.org/W1", "predicate": "AFFILIATED_WITH", "target": "Org", "hop": 1}], [])
    labels = {node["id"]: node["label"] for node in graph["nodes"]}

    assert labels == {"https://openalex.org/W1": "Reliable AI", "Org": "Org"}   # title folded into the label
    assert [edge["predicate"] for edge in graph["edges"]] == ["AFFILIATED_WITH"]


# ---------------------------------------------------------------- ranking
def test_rerank_bonuses_are_bounded_so_common_words_cannot_bury_the_answer() -> None:
    hits = [
        {"document": "noise", "score": 0.55, "text": "large language models language models evaluation large models"},
        {"document": "answer", "score": 0.80, "text": "Acme Labs is based in Seattle."},
    ]

    ranked = rerank_evidence("Which large language models evaluation is Acme Labs based in?", hits, ["Acme Labs"])

    assert ranked[0]["document"] == "answer"
    assert ranked[0]["similarity"] == 0.80


def test_merge_hits_dedupes_and_graph_reached_chunks_get_a_bonus() -> None:
    vector = [{"id": "1", "document": "a", "score": 0.7, "text": "x"}, {"id": "2", "document": "b", "score": 0.6, "text": "y"}]
    graph = [{"id": "2", "document": "b", "score": 0.6, "text": "y"}, {"id": "3", "document": "c", "score": 0.5, "text": "z"}]

    merged = {hit["id"]: hit for hit in merge_hits(vector, graph)}
    ranked = rerank_evidence("unrelated", list(merged.values()), [])

    assert {key: hit["retrieval"] for key, hit in merged.items()} == {"1": "vector", "2": "vector+graph", "3": "graph"}
    assert [hit["id"] for hit in ranked] == ["2", "3", "1"]


# ---------------------------------------------------------------- context and refusal
def test_context_puts_graph_facts_before_long_sources() -> None:
    hits = [{"document": "d1", "text": "x" * 5000}, {"document": "d2", "text": "short"}]

    context = build_context(["Paper"], ["Paper -[AFFILIATED_WITH]-> Anthropic"], hits, max_chars=2200, per_source_chars=500)

    assert context.index("Graph facts") < context.index("[d1]")
    assert "AFFILIATED_WITH" in context and "[d2] short" in context
    assert build_context(["Paper"], [], []) == ""


def test_refusal_detects_common_phrasings() -> None:
    assert should_refuse_answer("ctx", "I do not know.")
    assert should_refuse_answer("ctx", "I don\u2019t know.")
    assert not should_refuse_answer("ctx", "Anthropic [doc-1]")


# ---------------------------------------------------------------- graph expansion with a fake Neo4j
class _Session:
    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self.results, self.queries = results, []

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((query, params))
        return self.results.pop(0) if self.results else []


def _store(monkeypatch, session: _Session) -> Neo4jStore:
    driver = type("D", (), {"session": lambda self: session})()
    monkeypatch.setattr("graph_rag.stores.GraphDatabase", type("G", (), {"driver": staticmethod(lambda *a, **k: driver)}))
    return Neo4jStore("bolt://x", "u", "p")


def test_neighborhood_expands_two_hops_but_not_through_hubs(monkeypatch) -> None:
    hop1 = [
        {"source": "Paper", "predicate": "AFFILIATED_WITH", "target": "Anthropic", "other": "Anthropic", "degree": 900},
        {"source": "Author", "predicate": "AUTHORED", "target": "Paper", "other": "Author", "degree": 2},
    ]
    hop2 = [{"source": "Author", "predicate": "AUTHORED", "target": "Other paper"}]
    session = _Session([hop1, hop2])

    rows = _store(monkeypatch, session).neighborhood(["Paper"], hops=2, limit=10, hub_degree=40)

    assert [(row["source"], row["target"], row["hop"]) for row in rows] == [
        ("Paper", "Anthropic", 1), ("Author", "Paper", 1), ("Author", "Other paper", 2),
    ]
    assert session.queries[1][1]["frontier"] == ["Author"]   # the 900-degree hub isn't expanded


def test_neighborhood_respects_the_fact_budget(monkeypatch) -> None:
    hop1 = [{"source": f"S{i}", "predicate": "P", "target": "T", "other": f"S{i}", "degree": 1} for i in range(10)]
    session = _Session([hop1])

    rows = _store(monkeypatch, session).neighborhood(["T"], hops=2, limit=4)

    assert len(rows) == 4
    assert len(session.queries) == 1                          # budget spent; no second hop


# ---------------------------------------------------------------- ingestion batching
def test_batches_skip_checkpointed_records_and_honour_limit(tmp_path) -> None:
    checkpoint = IngestionCheckpoint(tmp_path / "c.json")
    checkpoint.mark_many(["p1"])
    records = [{"paper_id": f"p{i}"} for i in range(1, 8)]

    stats: dict[str, int] = {}
    out = list(batches(records, checkpoint, limit=5, size=2, stats=stats))

    assert [[r["paper_id"] for r in batch] for batch in out] == [["p2", "p3"], ["p4", "p5"], ["p6"]]
    assert stats["skipped"] == 1
    assert IngestionCheckpoint(tmp_path / "c.json").processed == {"p1"}


def test_batches_count_skips_even_when_everything_is_checkpointed(tmp_path) -> None:
    checkpoint = IngestionCheckpoint(tmp_path / "c.json")
    checkpoint.mark_many([f"p{i}" for i in range(4)])
    stats: dict[str, int] = {}

    assert list(batches([{"paper_id": f"p{i}"} for i in range(4)], checkpoint, limit=0, size=2, stats=stats)) == []
    assert stats["skipped"] == 4


# ---------------------------------------------------------------- pipeline end to end with fakes
class _Models:
    def embed(self, text: str) -> list[float]: return [1.0, 0.0]
    def entities(self, text: str) -> list[str]: return ["Alice Johnson"]
    def answer(self, question: str, context: str) -> str: return "Seattle [sample.md]"
    def answer_stream(self, question: str, context: str):
        yield "Seattle "
        yield "[sample.md]"


class _Graph:
    def ensure_schema(self) -> None: pass
    def link_entities(self, text: str) -> list[dict[str, Any]]: return [{"name": "Acme Labs", "score": 3.0}]
    def chunks_mentioning(self, entities: list[str], limit: int = 8) -> list[str]: return ["c2"]
    def neighborhood(self, seeds: list[str], hops: int = 2, limit: int = 24) -> list[dict[str, Any]]:
        assert seeds == ["Acme Labs", "Alice Johnson"]
        return [
            {"source": "Alice Johnson", "predicate": "works at", "target": "Acme Labs", "hop": 1},
            {"source": "Acme Labs", "predicate": "is based in", "target": "Seattle", "hop": 1},
        ]


class _Vectors:
    def search(self, vector: list[float], limit: int = 10, ids: list[str] | None = None) -> list[dict[str, Any]]:
        if ids:
            return [{"id": "c2", "score": 0.4, "document": "sample.md", "text": "Acme Labs is based in Seattle.", "entities": ["Acme Labs", "Seattle"]}]
        return [{"id": "c1", "score": 0.6, "document": "sample.md", "text": "Alice Johnson works at Acme Labs.", "entities": ["Alice Johnson", "Acme Labs"]}]


def _pipeline() -> CloudGraphRAG:
    return CloudGraphRAG(models=_Models(), graph=_Graph(), vectors=_Vectors(), metrics=MetricsCollector())  # type: ignore[arg-type]


def test_query_result_returns_graph_sources_and_stage_timings() -> None:
    result = _pipeline().query_result("Where is Alice Johnson's company based?", limit=4)

    assert result["answer"] == "Seattle [sample.md]"
    assert {s["retrieval"] for s in result["sources"]} == {"vector", "graph"}
    assert "Acme Labs -[is based in]-> Seattle" in result["graph_facts"]
    roles = {node["id"]: node["role"] for node in result["graph"]["nodes"]}
    assert roles["Acme Labs"] == "linked" and roles["Alice Johnson"] == "seed" and roles["Seattle"] == "hop1"
    for stage in ("embedding_ms", "entity_linking_ms", "vector_search_ms", "graph_retrieval_ms", "graph_expansion_ms",
                  "retrieval_ms", "generation_ms", "total_ms"):
        assert stage in result["timings"], stage


def test_query_stream_sends_evidence_then_tokens_then_done_with_timings() -> None:
    events = list(_pipeline().query_stream("Where is Alice Johnson's company based?", limit=4))

    assert [event["type"] for event in events] == ["evidence", "token", "token", "citations", "done"]
    assert events[-2]["verdict"] == "supported" and events[-2]["answer"] == "Seattle [sample.md]"
    assert events[0]["graph"]["edges"]
    assert "first_token_ms" in events[-1]["timings"]
