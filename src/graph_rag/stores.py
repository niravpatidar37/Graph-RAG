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
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $document}) "
                "MERGE (c:Chunk {id: $chunk}) SET c.text = $text "
                "MERGE (c)-[:FROM_DOCUMENT]->(d)",
                document=chunk.document, chunk=chunk.id, text=chunk.text,
            )
            for entity in chunk.entities:
                session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "WITH e MATCH (c:Chunk {id: $chunk}) MERGE (c)-[:MENTIONS]->(e)",
                    name=entity, chunk=chunk.id,
                )
            for relation in relations:
                session.run(
                    "MERGE (source:Entity {name: $source}) "
                    "MERGE (target:Entity {name: $target}) "
                    "MERGE (source)-[r:RELATED {predicate: $predicate}]->(target) "
                    "SET r.confidence = $confidence, r.chunk_id = $chunk",
                    source=relation.get("source", ""), target=relation.get("target", ""),
                    predicate=relation.get("predicate", "RELATED"),
                    confidence=float(relation.get("confidence", 0)), chunk=chunk.id,
                )

    def neighbors(self, entities: list[str], limit: int = 20) -> list[str]:
        with self.driver.session() as session:
            records = session.run(
                "MATCH (e:Entity)-[:RELATED]-(neighbor:Entity) "
                "WHERE e.name IN $entities RETURN DISTINCT neighbor.name AS name LIMIT $limit",
                entities=entities, limit=limit,
            )
            return [record["name"] for record in records]

    def facts(self, entities: list[str], limit: int = 50) -> list[str]:
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
            return [
                f"{record['source']} -[{record['predicate']}]-> {record['target']}"
                for record in records
            ]


class QdrantStore:
    def __init__(self, url: str, api_key: str, collection: str) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        self.upsert_batch([(chunk, vector)])

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