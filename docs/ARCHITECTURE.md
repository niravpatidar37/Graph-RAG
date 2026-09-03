# Graph RAG System Design

This document is the authoritative top-level design for Graph RAG. It separates the system into a reliable ingestion plane and a low-latency query plane.

## Goals

- Answer questions from retrieved evidence, not model memory.
- Combine semantic text retrieval with bounded graph traversal.
- Support repeatable, idempotent ingestion.
- Keep API workers stateless so they can scale horizontally.
- Preserve source identity and provenance for every answer.
- Keep model providers and storage providers replaceable.

## Non-Goals for the First Production Milestone

- Building a general-purpose knowledge graph for every domain.
- Extracting every possible relationship from every document.
- Unbounded graph traversal.
- Training or hosting foundation models ourselves.
- Supporting multi-region active-active deployment before measurements require it.

## System Context

```text
                         +----------------------+
                         |  Source documents    |
                         |  AI Safety dataset   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Ingestion plane      |
                         | parse, map, embed     |
                         +----+------------+-----+
                              |            |
                              v            v
                     +-------------+  +-------------+
                     | Neo4j       |  | Qdrant      |
                     | graph facts |  | chunk vector|
                     +------+------+  +------+------+
                            ^                ^
                            |                |
+----------+       +--------+----------------+--------+       +---------+
| Client   +------>| FastAPI query orchestrator       +------>| HF API  |
+----------+       | retrieve, expand, ground, answer |       +---------+
                    +----------------------------------+
```

## Planes and Responsibilities

### Ingestion plane

The ingestion plane runs asynchronously and can be scaled independently from query traffic.

1. Discover a source record or document.
2. Compute a stable document ID and content hash.
3. Normalize the source into a canonical text representation.
4. Split text into chunks with stable chunk IDs.
5. Extract entities and structured relationships.
6. Generate embeddings in batches.
7. Upsert graph facts into Neo4j.
8. Upsert chunks and vectors into Qdrant.
9. Record the completed model and schema versions.

The AI Safety importer uses the dataset's existing OpenAlex-style fields as high-confidence facts. Qwen relationship extraction is optional enrichment, not a required step for every record.

### Query plane

The query plane is synchronous, stateless, and bounded.

1. Validate the request and tenant permissions.
2. Embed the question.
3. Search Qdrant for semantic candidates.
4. Extract query entities.
5. Expand those entities by one or two Neo4j hops.
6. Merge and rerank text and graph evidence.
7. Build a context with source IDs.
8. Ask the answer model to answer only from that context.
9. Return the answer and citations.

The answer model is never used as the retrieval source of truth.

## Data Ownership

| Data | System of record | Purpose |
|---|---|---|
| Original source record | Object storage or source dataset | Reprocessing and audit |
| Document metadata | PostgreSQL, future phase | Status, version, tenant, permissions |
| Entity and relationship facts | Neo4j | Traversal and graph reasoning |
| Chunk text and embeddings | Qdrant | Semantic retrieval and filters |
| Model outputs | Metadata store, future phase | Reproducibility and review |
| Query logs and metrics | Observability system, future phase | Quality and operations |

Neo4j should not become the primary store for large full-text documents. Qdrant should not become the source of truth for graph relationships.

## Canonical Data Model

### Document and chunk

```text
Document
  id: stable source identifier
  source_uri: original URL or dataset ID
  content_hash: hash of normalized content
  version: source version or dump date

Chunk
  id: stable hash(document_id + chunk_index + content_hash)
  document_id: parent document
  text: normalized text
  embedding_model: model identifier
  extraction_version: pipeline version
  source_offsets: optional character offsets
```

### Graph

```text
(:Document {id, source_uri})
(:Chunk {id})
(:Entity {name, type, canonical_id})

(:Chunk)-[:FROM_DOCUMENT]->(:Document)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED {predicate, confidence, chunk_id}]->(:Entity)
```

For the AI Safety dataset, structured relations include `AUTHORED`, `HAS_TITLE`, `AFFILIATED_WITH`, and `CLASSIFIED_AS`.

## Reliability Rules

- Every write must be idempotent.
- Every model call must have a timeout and retry policy.
- Every relationship must carry source evidence and confidence.
- Every query must have a graph-hop and context-token limit.
- Every answer must be able to return no answer when evidence is insufficient.
- Embedding model and vector dimension are immutable for a Qdrant collection.
- Credentials are loaded from `.env` locally and secret management in deployment; never from tracked files.

## Scaling Plan

### First milestone

- Use the current FastAPI service, Neo4j AuraDB, Qdrant Cloud, and Hugging Face Inference API.
- Ingest 100 AI Safety records.
- Verify graph facts, vector hits, citations, and repeatable upserts.
- Add a small evaluation set of expected questions and source IDs.

### Second milestone

- Add PostgreSQL for ingestion jobs, document metadata, and permissions.
- Add batch embedding and bounded concurrency.
- Add checkpoints, retries, and dead-letter handling.
- Add hybrid keyword plus vector ranking and a reranker.

### Production milestone

- Put ingestion behind a queue or workflow engine.
- Run stateless API replicas behind a load balancer.
- Add Redis for rate limits and carefully keyed caches.
- Add metrics for retrieval recall, citation accuracy, latency, token usage, and cost.
- Load test before increasing graph size or query concurrency.

## Operational SLO Starting Points

```text
Query p95 latency:        less than 3 seconds
Retrieval p95 latency:    less than 500 milliseconds
Citation correctness:     greater than 95 percent
Unsupported answers:      less than 2 percent
Ingestion retries:        safe and idempotent
```

These are initial targets, not guarantees. Adjust them using production measurements.

## Security and Compliance

- Rotate any credential that has been exposed.
- Use least-privilege API keys and database users.
- Apply tenant and document filters before retrieval and graph expansion.
- Do not log prompts, documents, or credentials without an explicit data policy.
- Keep source licenses, attribution, and dataset versions with ingested records.
- Propagate deletion requests to object storage, Neo4j, Qdrant, caches, and logs.

## Repository Mapping

```text
src/graph_rag/core.py             local deterministic prototype
src/graph_rag/dataset.py          AI Safety record mapping
src/graph_rag/dataset_ingest.py   streaming dataset importer
src/graph_rag/stores.py            Neo4j and Qdrant adapters
src/graph_rag/huggingface.py      Hugging Face model adapter
src/graph_rag/pipeline.py         cloud ingestion and query orchestration
src/graph_rag/api.py              FastAPI application
```

The next code change should add ingestion checkpoints and PostgreSQL metadata without changing the public query contract.
