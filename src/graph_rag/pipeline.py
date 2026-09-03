from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from langfuse import get_client, observe

from .core import GraphRAG
from .huggingface import HuggingFaceModels
from .production import MetricsCollector, rerank_evidence, should_refuse_answer
from .settings import Settings
from .stores import Chunk, Neo4jStore, QdrantStore


class EvidenceSource(TypedDict):
    document_id: str
    score: float
    text: str


class RetrievalEvidence(TypedDict):
    context: str
    sources: list[EvidenceSource]
    graph_facts: list[str]
    retrieved_entities: list[str]
    trace_id: str


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
            models=HuggingFaceModels(settings.hf_token, settings.hf_embedding_model, settings.hf_ner_model, settings.hf_llm_model),
            graph=Neo4jStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password),
            vectors=QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection),
            metrics=MetricsCollector(),
        )

    def ingest_file(self, path: str | Path) -> int:
        file_path = Path(path)
        chunks = GraphRAG._chunk(file_path.read_text(encoding="utf-8"))
        indexed = 0
        for text in chunks:
            chunk = Chunk(file_path.name, text, self.models.entities(text))
            relations = self.models.relations(text)
            self.ingest_chunk(chunk, relations)
            indexed += 1
        return indexed

    def ingest_chunk(self, chunk: Chunk, relations: list[dict[str, object]]) -> None:
        vector = self.models.embed(chunk.text)
        self.vectors.ensure_collection(len(vector))
        self.graph.upsert_chunk(chunk, relations)
        self.vectors.upsert(chunk, vector)

    def query(self, question: str, limit: int = 8) -> str:
        return self.query_result(question, limit)["answer"]

    @observe(name="graph-rag.query")
    def query_result(self, question: str, limit: int = 8) -> dict[str, object]:
        self.metrics.increment("queries")
        start = perf_counter()
        try:
            evidence = self.retrieve_evidence(question, limit)
            answer = self.models.answer(question, evidence["context"])
            if should_refuse_answer(evidence["context"], answer):
                answer = "I do not have enough reliable evidence to answer this question."
            return {
                "answer": answer,
                "sources": evidence["sources"],
                "graph_facts": evidence["graph_facts"],
                "retrieved_entities": evidence["retrieved_entities"],
            }
        finally:
            self.metrics.timing("query_result_seconds", perf_counter() - start)

    def retrieve(self, question: str, limit: int = 8) -> str:
        return str(self.retrieve_evidence(question, limit)["context"])

    @observe(name="graph-rag.retrieve")
    def retrieve_evidence(self, question: str, limit: int = 8) -> RetrievalEvidence:
        with ThreadPoolExecutor(max_workers=2) as executor:
            embedding_future = executor.submit(self.metrics.measure, "embedding_seconds", lambda: self.models.embed(question))
            entities_future = executor.submit(self.metrics.measure, "entity_extraction_seconds", lambda: self.models.entities(question))
            question_vector = embedding_future.result()
            entities = entities_future.result()
        hits = self.metrics.measure("qdrant_search_seconds", lambda: self.vectors.search(question_vector, limit=max(limit * 2, limit)))
        ranked_hits = self.metrics.measure("reranking_seconds", lambda: rerank_evidence(question, hits, entities, limit=limit))
        retrieved_entities = {entity for hit in ranked_hits for entity in hit.get("entities", [])}
        graph_entities = sorted(set(entities) | retrieved_entities)[:12]
        with ThreadPoolExecutor(max_workers=2) as executor:
            neighbors_future = executor.submit(self.metrics.measure, "neo4j_neighbors_seconds", lambda: self.graph.neighbors(graph_entities))
            facts_future = executor.submit(self.metrics.measure, "neo4j_facts_seconds", lambda: self.graph.facts(graph_entities, limit=12))
            neighbors = neighbors_future.result()
            facts = facts_future.result()
        context_parts = [f"[{hit.get('document', 'unknown')}] {hit.get('text', '')}" for hit in ranked_hits]
        if neighbors:
            context_parts.append(f"Graph neighbors of {', '.join(graph_entities)}: {', '.join(neighbors)}")
        if facts:
            context_parts.append("Graph facts: " + "; ".join(facts))
        sources: list[EvidenceSource] = []
        for hit in ranked_hits:
            document_id = hit.get("document", "")
            score = hit.get("score", 0)
            sources.append(
                {
                    "document_id": str(document_id),
                    "score": float(score),
                    "text": str(hit.get("text", "")),
                }
            )
        return {
            "context": "\n".join(context_parts)[:16000],
            "sources": sources,
            "graph_facts": facts,
            "retrieved_entities": graph_entities,
            "trace_id": get_client().get_current_trace_id() or "",
        }