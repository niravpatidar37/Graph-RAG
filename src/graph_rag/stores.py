from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class Chunk:
    document: str
    text: str
    entities: list[str]

    @property
    def id(self) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.document}:{self.text}"))


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def upsert_chunk(self, chunk: Chunk, relations: list[dict[str, Any]]) -> None:
        edges = [
            {
                "source": str(relation.get("source", "")).strip(),
                "target": str(relation.get("target", "")).strip(),
                "predicate": str(relation.get("predicate", "") or "RELATED"),
                "confidence": float(relation.get("confidence", 0)),
            }
            for relation in relations
        ]
        edges = [edge for edge in edges if edge["source"] and edge["target"]]
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $document}) "
                "MERGE (c:Chunk {id: $chunk}) SET c.text = $text "
                "MERGE (c)-[:FROM_DOCUMENT]->(d) "
                "WITH c UNWIND $entities AS name "
                "MERGE (e:Entity {name: name}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                document=chunk.document, chunk=chunk.id, text=chunk.text, entities=chunk.entities,
            )
            if edges:
                session.run(
                    "UNWIND $edges AS edge "
                    "MERGE (source:Entity {name: edge.source}) "
                    "MERGE (target:Entity {name: edge.target}) "
                    "MERGE (source)-[r:RELATED {predicate: edge.predicate}]->(target) "
                    "SET r.confidence = edge.confidence, r.chunk_id = $chunk",
                    edges=edges, chunk=chunk.id,
                )

    def facts(self, entities: list[str], limit: int = 50) -> list[dict[str, str]]:
        if not entities:
            return []
        with self.driver.session() as session:
            records = session.run(
                "MATCH (source:Entity)-[r:RELATED]-(target:Entity) "
                "WHERE source.name IN $entities OR target.name IN $entities "
                "RETURN DISTINCT source.name AS source, r.predicate AS predicate, "
                "target.name AS target LIMIT $limit",
                entities=entities, limit=limit,
            )
            return [{"source": record["source"], "predicate": record["predicate"], "target": record["target"]} for record in records]


class QdrantStore:
    def __init__(self, url: str, api_key: str, collection: str) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self._ready = False

    def ensure_collection(self, vector_size: int) -> None:
        if self._ready:
            return
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        self._ready = True

    def upsert_batch(self, items: list[tuple[Chunk, list[float]]]) -> None:
        if not items:
            return
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={"document": chunk.document, "text": chunk.text, "entities": chunk.entities},
            )
            for chunk, vector in items
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
        response = self.client.query_points(collection_name=self.collection, query=vector, limit=limit)
        return [{"score": hit.score, **(hit.payload or {})} for hit in response.points]