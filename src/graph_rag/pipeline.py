from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, TypedDict, TypeVar

from langfuse import get_client, observe

from .core import GraphRAG
from .huggingface import HuggingFaceModels
from .production import MetricsCollector, rerank_evidence, should_refuse_answer
from .settings import Settings
from .stores import Chunk, Neo4jStore, QdrantStore

T = TypeVar("T")

REFUSAL = "I do not have enough reliable evidence to answer this question."
MAX_SEEDS = 12
MAX_FACTS = 24
CONTEXT_CHARS = 16000
SOURCE_CHARS = 1500


class EvidenceSource(TypedDict):
    document_id: str
    score: float
    similarity: float
    retrieval: str       # "vector", "graph", or "vector+graph"
    text: str


class GraphNode(TypedDict):
    id: str
    label: str           # display name: the title for paper-ID nodes, else the id
    role: str            # "linked" (named in the question), "seed" (from evidence), "hop1", "hop2"
    sources: list[str]   # document IDs of retrieved chunks that mention this entity


class GraphEdge(TypedDict):
    source: str
    target: str
    predicate: str
    hop: int


class EvidenceGraph(TypedDict):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class RetrievalEvidence(TypedDict):
    context: str
    sources: list[EvidenceSource]
    graph_facts: list[str]
    retrieved_entities: list[str]
    graph: EvidenceGraph
    timings: dict[str, float]
    trace_id: str


def _usable_entity(name: str) -> bool:
    """NER on short questions emits fragments like "A"; they would only add noise seeds."""
    stripped = name.strip()
    return len(stripped) >= 2 and any(ch.isalnum() for ch in stripped)


def order_seeds(linked: list[str], question_entities: list[str], hits: list[dict[str, Any]],
                top_hits: int = 3, limit: int = MAX_SEEDS) -> list[str]:
    """Seed entities in priority order: named in the question, then NER, then top evidence.

    Order matters because graph expansion is budgeted per seed: earlier seeds get their
    facts first. (Sorting alphabetically here once let unrelated titles starting with
    quotes or digits take the whole budget.) When the question names a graph entity
    outright, that anchor is precise, so entities of loosely related top hits are left out:
    their facts only crowd the context, and small models then answer from the wrong ones.
    """
    ordered: dict[str, None] = {}
    for name in [*linked, *question_entities]:
        if _usable_entity(name):
            ordered.setdefault(name, None)
    for hit in ([] if linked else hits[:top_hits]):
        for name in hit.get("entities", []) or []:
            if _usable_entity(str(name)):
                ordered.setdefault(str(name), None)
    return list(ordered)[:limit]


def merge_hits(vector_hits: list[dict[str, Any]], graph_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One candidate pool, deduplicated by chunk ID, remembering how each chunk was found."""
    merged: dict[str, dict[str, Any]] = {}
    for hit in vector_hits:
        key = str(hit.get("id") or f"{hit.get('document')}:{hit.get('text')}")
        merged[key] = {**hit, "retrieval": "vector"}
    for hit in graph_hits:
        key = str(hit.get("id") or f"{hit.get('document')}:{hit.get('text')}")
        if key in merged:
            merged[key].update(via_graph=True, retrieval="vector+graph")
        else:
            merged[key] = {**hit, "via_graph": True, "retrieval": "graph"}
    return list(merged.values())


def build_graph(seeds: list[str], linked: list[str], rows: list[dict[str, Any]],
                hits: list[dict[str, Any]]) -> EvidenceGraph:
    mentions: dict[str, list[str]] = {}
    for hit in hits:
        for name in hit.get("entities", []) or []:
            docs = mentions.setdefault(str(name), [])
            document = str(hit.get("document", ""))
            if document and document not in docs:
                docs.append(document)
    linked_set, seed_set = set(linked), set(seeds)
    hop_of: dict[str, int] = {}
    for row in rows:
        for name in (row["source"], row["target"]):
            hop_of[name] = min(hop_of.get(name, 9), int(row.get("hop", 1)))

    def role(name: str) -> str:
        if name in linked_set:
            return "linked"
        if name in seed_set:
            return "seed"
        return "hop2" if hop_of.get(name, 1) >= 2 else "hop1"

    names = [name for name in seeds if name in hop_of]          # seeds that actually have facts
    names += [name for name in hop_of if name not in set(names)]
    # HAS_TITLE is folded into the paper node's label: drawing the title as its own node only
    # repeats the same text twice. The fact itself stays in graph_facts and the context.
    titles = {row["source"]: row["target"] for row in rows if row["predicate"] == "HAS_TITLE"}
    title_only = {
        title for title in titles.values()
        if all(row["predicate"] == "HAS_TITLE" for row in rows if title in (row["source"], row["target"]))
    } - linked_set
    names = [name for name in names if name not in title_only]
    rows = [row for row in rows if not (row["predicate"] == "HAS_TITLE" and row["target"] in title_only)]
    nodes: list[GraphNode] = [
        {"id": name, "label": titles.get(name, name), "role": role(name), "sources": mentions.get(name, [])} for name in names
    ]
    edges: list[GraphEdge] = [
        {"source": row["source"], "target": row["target"], "predicate": row["predicate"], "hop": int(row.get("hop", 1))}
        for row in rows
    ]
    return {"nodes": nodes, "edges": edges}


def build_context(question_entities: list[str], facts: list[str], hits: list[dict[str, Any]],
                  max_chars: int = CONTEXT_CHARS, per_source_chars: int = SOURCE_CHARS) -> str:
    """Compact graph facts first, then sources, each source capped.

    Facts used to be appended after the full source texts and silently cut off by the overall
    cap whenever abstracts were long, which hid exactly the multi-hop evidence graph
    retrieval exists to provide. Empty when there is no evidence at all, so callers refuse.
    """
    if not facts and not hits:
        return ""
    parts: list[str] = []
    named = [name for name in dict.fromkeys(question_entities) if name]
    if named:
        parts.append("Entities named in the question: " + "; ".join(named))
    if facts:
        parts.append("Graph facts: " + "; ".join(facts))
    for hit in hits:
        text = str(hit.get("text", ""))
        if len(text) > per_source_chars:
            text = text[:per_source_chars].rstrip() + " ..."
        parts.append(f"[{hit.get('document', 'unknown')}] {text}")
    return "\n".join(parts)[:max_chars]


@dataclass
class CloudGraphRAG:
    models: HuggingFaceModels
    graph: Neo4jStore
    vectors: QdrantStore
    metrics: MetricsCollector = field(default_factory=MetricsCollector, repr=False)

    @classmethod
    def from_env(cls) -> "CloudGraphRAG":
        settings = Settings.from_env()
        settings.validate_cloud()
        return cls(
            models=HuggingFaceModels(settings.hf_token, settings.hf_embedding_model, settings.hf_ner_model, settings.hf_llm_model, settings.llm_base_url, settings.local_models),
            graph=Neo4jStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password),
            vectors=QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection),
            metrics=MetricsCollector(),
        )

    # ------------------------------------------------------------------ ingestion
    def ingest_file(self, path: str | Path) -> int:
        file_path = Path(path)
        texts = GraphRAG._chunk(file_path.read_text(encoding="utf-8"))
        if not texts:
            return 0
        items = [(Chunk(file_path.name, text, self.models.entities(text)), self.models.relations(text)) for text in texts]
        self.ingest_batch(items)
        return len(items)

    def ingest_chunk(self, chunk: Chunk, relations: list[dict[str, object]]) -> None:
        self.ingest_batch([(chunk, relations)])

    def ingest_batch(self, items: list[tuple[Chunk, list[dict[str, object]]]]) -> None:
        """Embed in one call and write each store once. Both writes are idempotent upserts
        keyed by the stable chunk ID, so retrying a failed batch is safe."""
        if not items:
            return
        vectors = self.models.embed_batch([chunk.text for chunk, _ in items])
        self.vectors.ensure_collection(len(vectors[0]))
        self.graph.ensure_schema()
        self.graph.upsert_batch(items)  # type: ignore[arg-type]
        self.vectors.upsert_batch([(chunk, vector) for (chunk, _), vector in zip(items, vectors, strict=True)])

    # ------------------------------------------------------------------ query
    @observe(name="graph-rag.query")
    def query_result(self, question: str, limit: int = 8) -> dict[str, object]:
        self.metrics.increment("queries")
        start = perf_counter()
        try:
            evidence = self.retrieve_evidence(question, limit)
            generation_start = perf_counter()
            answer = self.models.answer(question, evidence["context"]) if evidence["context"].strip() else ""
            timings = {**evidence["timings"], "generation_ms": _ms(perf_counter() - generation_start)}
            if should_refuse_answer(evidence["context"], answer):
                answer = REFUSAL
            timings["total_ms"] = _ms(perf_counter() - start)
            return {
                "answer": answer,
                "sources": evidence["sources"],
                "graph_facts": evidence["graph_facts"],
                "retrieved_entities": evidence["retrieved_entities"],
                "graph": evidence["graph"],
                "timings": timings,
            }
        finally:
            self.metrics.timing("query_result_seconds", perf_counter() - start)

    def query_stream(self, question: str, limit: int = 8) -> Iterator[dict[str, object]]:
        """Stream the answer as it generates: one evidence event, token events, then done.

        No post-hoc refusal rewrite here: once tokens are sent to a client they can't be
        un-sent. Empty context is caught before generating; a model that hedges mid-answer
        streams its hedge as-is instead of being swapped for the canned refusal message.
        """
        self.metrics.increment("queries")
        start = perf_counter()
        try:
            evidence = self.retrieve_evidence(question, limit)
            yield {
                "type": "evidence",
                "sources": evidence["sources"],
                "graph_facts": evidence["graph_facts"],
                "retrieved_entities": evidence["retrieved_entities"],
                "graph": evidence["graph"],
                "timings": evidence["timings"],
            }
            generation_start = perf_counter()
            first_token: float | None = None
            if not evidence["context"].strip():
                yield {"type": "token", "text": REFUSAL}
            else:
                for delta in self.models.answer_stream(question, evidence["context"]):
                    if first_token is None:
                        first_token = perf_counter() - generation_start
                    yield {"type": "token", "text": delta}
            timings = {**evidence["timings"], "generation_ms": _ms(perf_counter() - generation_start)}
            if first_token is not None:
                timings["first_token_ms"] = _ms(first_token)
            timings["total_ms"] = _ms(perf_counter() - start)
            yield {"type": "done", "timings": timings}
        finally:
            self.metrics.timing("query_result_seconds", perf_counter() - start)

    @observe(name="graph-rag.retrieve")
    def retrieve_evidence(self, question: str, limit: int = 8) -> RetrievalEvidence:
        timings: dict[str, float] = {}
        start = perf_counter()

        def timed(name: str, fn: Callable[[], T]) -> T:
            began = perf_counter()
            try:
                return fn()
            finally:
                elapsed = perf_counter() - began
                self.metrics.timing(f"{name}_seconds", elapsed)
                timings[f"{name}_ms"] = _ms(elapsed)

        self.graph.ensure_schema()
        with ThreadPoolExecutor(max_workers=3) as executor:
            embedding_future = executor.submit(timed, "embedding", lambda: self.models.embed(question))
            entities_future = executor.submit(timed, "entity_extraction", lambda: self.models.entities(question))
            linking_future = executor.submit(timed, "entity_linking", lambda: self.graph.link_entities(question))
            question_vector = embedding_future.result()
            question_entities = [name for name in entities_future.result() if _usable_entity(name)]
            linked = [row["name"] for row in linking_future.result()]
        hits = timed("vector_search", lambda: self.vectors.search(question_vector, limit=limit * 3))
        anchors = [*linked, *question_entities]
        graph_chunk_ids = timed("graph_retrieval", lambda: self.graph.chunks_mentioning(anchors, limit=limit))
        graph_hits = self.vectors.search(question_vector, ids=graph_chunk_ids) if graph_chunk_ids else []
        candidates = merge_hits(hits, graph_hits)
        ranked_hits = timed("reranking", lambda: rerank_evidence(question, candidates, anchors, limit=limit))
        seeds = order_seeds(linked, question_entities, ranked_hits)
        rows = timed("graph_expansion", lambda: self.graph.neighborhood(seeds, hops=2, limit=MAX_FACTS))
        facts = [f"{row['source']} -[{row['predicate']}]-> {row['target']}" for row in rows]
        context = build_context([*linked, *question_entities], facts, ranked_hits)
        sources: list[EvidenceSource] = [
            {
                "document_id": str(hit.get("document", "")),
                "score": float(hit.get("score", 0)),
                "similarity": float(hit.get("similarity", hit.get("score", 0))),
                "retrieval": str(hit.get("retrieval", "vector")),
                "text": str(hit.get("text", "")),
            }
            for hit in ranked_hits
        ]
        timings["retrieval_ms"] = _ms(perf_counter() - start)
        return {
            "context": context,
            "sources": sources,
            "graph_facts": facts,
            "retrieved_entities": seeds,
            "graph": build_graph(seeds, linked, rows, ranked_hits),
            "timings": timings,
            "trace_id": get_client().get_current_trace_id() or "",
        }


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
