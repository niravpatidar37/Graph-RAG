from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

ENTITY_INDEX = "entity_name_fulltext"
_TOKEN = re.compile(r"[^\W_]+")  # Unicode letters/digits; every Lucene operator is ASCII punctuation
_LINK_STOP_WORDS = frozenset(
    "a an and are as at be by does for from how in is it of on or that the this to was what when where which who whom why with".split()
)


@dataclass(frozen=True)
class Chunk:
    document: str
    text: str
    entities: list[str]

    @property
    def id(self) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.document}:{self.text}"))


def fulltext_query(text: str, max_terms: int = 24) -> str:
    """Build a Lucene query from plain word tokens only.

    Questions are untrusted input. Keeping nothing but letter/digit tokens means no Lucene
    operator, wildcard, fuzzy/regex syntax, or field selector can reach the index, so a
    question can't widen, slow down, or redirect the search. The tokens become an OR query.
    """
    terms: list[str] = []
    for token in _TOKEN.findall(text):
        lowered = token.lower()
        if lowered in _LINK_STOP_WORDS or len(token) < 2 or lowered in terms:
            continue
        terms.append(lowered)
    return " ".join(terms[:max_terms])


def name_coverage(name: str, terms: set[str]) -> float:
    """Fraction of an entity name's meaningful words that appear in `terms`."""
    words = fulltext_query(name, max_terms=64).split()
    return sum(1 for word in words if word in terms) / len(words) if words else 0.0


def _edge_rows(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "source": str(edge.get("source", "")).strip(),
            "target": str(edge.get("target", "")).strip(),
            "predicate": str(edge.get("predicate", "") or "RELATED"),
            "confidence": float(edge.get("confidence", 0) or 0),
        }
        for edge in edges
        if isinstance(edge, dict)
    ]
    return [row for row in rows if row["source"] and row["target"] and row["source"] != row["target"]]


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._schema_ready = False

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        """Uniqueness constraints make MERGE an index seek instead of a label scan; the
        full-text index powers entity linking. All statements are idempotent."""
        if self._schema_ready:
            return
        with self.driver.session() as session:
            for statement in (
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                f"CREATE FULLTEXT INDEX {ENTITY_INDEX} IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
            ):
                session.run(statement)
        self._schema_ready = True

    def upsert_chunk(self, chunk: Chunk, relations: list[dict[str, Any]]) -> None:
        self.upsert_batch([(chunk, relations)])

    def upsert_batch(self, items: list[tuple[Chunk, list[dict[str, Any]]]]) -> None:
        """Write many chunks and their relations in two round trips."""
        if not items:
            return
        rows = [
            {"document": chunk.document, "chunk": chunk.id, "text": chunk.text, "entities": [e for e in chunk.entities if e]}
            for chunk, _ in items
        ]
        edges = [{**edge, "chunk": chunk.id} for chunk, relations in items for edge in _edge_rows(relations)]
        with self.driver.session() as session:
            session.run(
                "UNWIND $rows AS row "
                "MERGE (d:Document {id: row.document}) "
                "MERGE (c:Chunk {id: row.chunk}) SET c.text = row.text "
                "MERGE (c)-[:FROM_DOCUMENT]->(d) "
                "WITH c, row UNWIND row.entities AS name "
                "MERGE (e:Entity {name: name}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                rows=rows,
            )
            if edges:
                session.run(
                    "UNWIND $edges AS edge "
                    "MERGE (source:Entity {name: edge.source}) "
                    "MERGE (target:Entity {name: edge.target}) "
                    "MERGE (source)-[r:RELATED {predicate: edge.predicate}]->(target) "
                    "SET r.confidence = edge.confidence, r.chunk_id = edge.chunk",
                    edges=edges,
                )

    def link_entities(self, text: str, limit: int = 10, min_relative_score: float = 0.5,
                      min_name_coverage: float = 0.75) -> list[dict[str, Any]]:
        """Find graph entities named in free text (e.g. a paper title inside a question).

        NER only sees person/org/location spans; full-text linking also catches titles,
        classifications, and anything else that exists as an Entity node. Full-text scoring
        alone links any title that shares a word like "papers", so a candidate must also have
        most of its own words present in the text (`min_name_coverage`).
        """
        query = fulltext_query(text)
        if not query:
            return []
        with self.driver.session() as session:
            records = session.run(
                f"CALL db.index.fulltext.queryNodes('{ENTITY_INDEX}', $terms) YIELD node, score "
                "RETURN node.name AS name, score ORDER BY score DESC LIMIT $limit",
                terms=query, limit=limit,  # not `query=`: that name is Session.run's own first parameter
            )
            rows = [{"name": record["name"], "score": float(record["score"])} for record in records]
        question_terms = set(query.split())
        rows = [row for row in rows if name_coverage(row["name"], question_terms) >= min_name_coverage]
        if not rows:
            return []
        best = rows[0]["score"]
        return [row for row in rows if row["score"] >= best * min_relative_score]

    def facts(self, entities: list[str], limit: int = 50) -> list[dict[str, Any]]:
        """Directed one-hop facts around `entities`, earlier entities first."""
        return self.neighborhood(entities, hops=1, limit=limit)

    def neighborhood(self, entities: list[str], hops: int = 2, limit: int = 24, per_node: int = 6,
                     hub_degree: int = 40) -> list[dict[str, Any]]:
        """Bounded, directed graph expansion.

        Hop 1 takes up to `per_node` edges per seed, seeds in priority order and specific
        neighbours (low degree) before hubs. Hop 2 expands only from non-hub hop-1
        neighbours, so one popular institution or classification can't flood the context.
        Every row keeps its true direction (startNode -> endNode).
        """
        seeds = [name for name in dict.fromkeys(entities) if name]
        if not seeds:
            return []
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def add(record: Any, hop: int) -> None:
            key = (record["source"], record["predicate"], record["target"])
            if key not in seen and len(rows) < limit:
                seen.add(key)
                rows.append({"source": key[0], "predicate": key[1], "target": key[2], "hop": hop})

        with self.driver.session() as session:
            first = session.run(
                "UNWIND range(0, size($seeds) - 1) AS rank "
                "MATCH (seed:Entity {name: $seeds[rank]}) "
                "CALL (seed) { "
                "  MATCH (seed)-[r:RELATED]-(other:Entity) "
                "  WITH r, other, COUNT { (other)-[:RELATED]-() } AS degree "
                "  ORDER BY degree ASC LIMIT $per_node "
                "  RETURN r, other, degree } "
                "RETURN startNode(r).name AS source, r.predicate AS predicate, endNode(r).name AS target, "
                "other.name AS other, degree, rank "
                "ORDER BY rank, degree LIMIT $limit",
                seeds=seeds, per_node=per_node, limit=limit,
            )
            frontier: list[str] = []
            for record in first:
                add(record, 1)
                if record["degree"] <= hub_degree and record["other"] not in seeds and record["other"] not in frontier:
                    frontier.append(record["other"])
            if hops >= 2 and frontier and len(rows) < limit:
                second = session.run(
                    "UNWIND range(0, size($frontier) - 1) AS rank "
                    "MATCH (node:Entity {name: $frontier[rank]}) "
                    "CALL (node) { "
                    "  MATCH (node)-[r:RELATED]-(other:Entity) WHERE NOT other.name IN $seeds "
                    "  WITH r, other, COUNT { (other)-[:RELATED]-() } AS degree "
                    "  ORDER BY degree ASC LIMIT $per_node "
                    "  RETURN r, degree } "
                    "RETURN startNode(r).name AS source, r.predicate AS predicate, endNode(r).name AS target, rank, degree "
                    "ORDER BY rank, degree LIMIT $limit",
                    frontier=frontier, seeds=seeds, per_node=per_node, limit=limit,
                )
                for record in second:
                    add(record, 2)
        return rows

    def chunks_mentioning(self, entities: list[str], limit: int = 8, max_mentions: int = 25) -> list[str]:
        """IDs of chunks that mention the given entities, earlier entities first.

        This is how the graph retrieves documents, not just facts: a question naming an author
        reaches that author's paper even though the paper's text never contains the name.
        Entities mentioned by more than `max_mentions` chunks (an institution, a class) are
        too unspecific to pull in documents and are skipped.
        """
        names = [name for name in dict.fromkeys(entities) if name]
        if not names:
            return []
        with self.driver.session() as session:
            records = session.run(
                "UNWIND range(0, size($names) - 1) AS rank "
                "MATCH (e:Entity {name: $names[rank]}) "
                "WITH e, rank, COUNT { (e)<-[:MENTIONS]-(:Chunk) } AS mentions WHERE mentions <= $max_mentions "
                "MATCH (e)<-[:MENTIONS]-(c:Chunk) "
                "RETURN c.id AS id, min(rank) AS rank ORDER BY rank LIMIT $limit",
                names=names, max_mentions=max_mentions, limit=limit,
            )
            return [str(record["id"]) for record in records]

    def clear(self) -> None:
        """Delete every node in batches. Only used by explicit reset tooling."""
        with self.driver.session() as session:
            session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")


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

    def search(self, vector: list[float], limit: int = 10, ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Nearest chunks; with `ids`, score only those points (graph-reached chunks)."""
        if ids is not None and not ids:
            return []
        query_filter = models.Filter(must=[models.HasIdCondition(has_id=list(ids))]) if ids else None
        response = self.client.query_points(
            collection_name=self.collection, query=vector, limit=len(ids) if ids else limit, query_filter=query_filter,
        )
        return [{"score": hit.score, **(hit.payload or {}), "id": str(hit.id)} for hit in response.points]
