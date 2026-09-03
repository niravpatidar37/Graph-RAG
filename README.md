# Graph RAG

This project is a starting point for a production-oriented Graph RAG system. It combines semantic document retrieval with entity and relationship traversal so answers can use both unstructured evidence and structured connections.

Read the complete top-level design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It defines the ingestion plane, query plane, data ownership, reliability rules, security boundaries, and scaling milestones.

## Current Status

The first vertical slice is local and deterministic:

- Reads Markdown and text files
- Splits documents into small chunks
- Extracts simple entities and relationships
- Performs lexical retrieval with one-hop graph expansion
- Returns supporting text and source names

The implementation is intentionally dependency-light. The in-memory prototype in `src/graph_rag/core.py` is a learning and test surface, not the final storage layer for a large corpus.

## Run with uv

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run pytest
uv run graph-rag data "Where is Alice Johnson's company based?"
```

Add your own `.md` or `.txt` files to `data/` and query the directory again.

## Top-Level Design

The system has two independently scalable planes:

- **Ingestion plane:** parse, normalize, chunk, extract, embed, and upsert source records asynchronously.
- **Query plane:** embed questions, retrieve Qdrant candidates, expand Neo4j neighbors, rerank evidence, and generate cited answers through Hugging Face.

The current cloud slice implements the core Neo4j, Qdrant, Hugging Face, and FastAPI boundaries. PostgreSQL metadata, queue-backed workers, checkpoints, and observability are the next production milestones.

## Scale Principles

1. Keep ingestion offline and asynchronous. Never parse a large document or extract a graph during a user request.
2. Use deterministic document and chunk IDs so retries are idempotent.
3. Track document checksums, parser versions, extractor versions, embedding models, and source offsets.
4. Resolve duplicate entities before creating relationships. Store aliases and extraction confidence.
5. Combine vector search, keyword search, and bounded graph traversal. Start with 20 to 100 retrieval candidates and one or two graph hops.
6. Enforce tenant and document permissions at every retrieval layer.
7. Measure retrieval recall, citation correctness, latency, token usage, and unsupported-answer rate.

## Implementation Roadmap

### Phase 1: Stable local foundation

- The local prototype provides a fast, deterministic test surface.
- Repository boundaries are now defined by the cloud stores and pipeline.
- Document checksums, ingestion status, and richer provenance remain next hardening steps.

### Phase 2: Durable retrieval

- Neo4j and Qdrant integrations are now available in cloud mode.
- PostgreSQL metadata and ingestion state remain to be added.
- Hybrid retrieval and reranking are the next retrieval improvements.

### Phase 3: Production ingestion

- Add batch workers with retries and dead-letter handling.
- Process only new or changed documents.
- Add entity resolution, confidence thresholds, and human review for low-confidence facts.
- Add observability with logs, metrics, traces, and per-stage latency.

### Phase 4: Grounded generation

- Connect a language model only after retrieval is complete.
- Pass selected evidence, source IDs, and graph facts into the prompt.
- Require citations and return an explicit no-answer when evidence is insufficient.
- Add model, prompt, and retrieval regression tests.

## Suggested First Production Stack

```text
FastAPI       query and ingestion API
PostgreSQL    metadata, jobs, permissions
Neo4j         entities and relationships
Qdrant        chunk vectors and metadata filters
Redis         cache and rate limiting
Celery/Temporal  asynchronous ingestion workflows
S3-compatible storage  original documents
```

## Why This Stack

Graph RAG has two fundamentally different retrieval problems: semantic similarity over text and relationship traversal across entities. We chose specialized services so each workload uses the access pattern it was designed for instead of forcing one database to do everything.

- **Neo4j AuraDB** is used for entities and relationships because Cypher makes multi-hop questions natural to express. AuraDB provides a managed public deployment, so we do not need to operate a graph cluster while developing.
- **Qdrant Cloud** is used for embeddings because vector similarity search, collection management, and metadata filters are its primary purpose. It can retrieve relevant chunks quickly without scanning the whole corpus.
- **Hugging Face** provides replaceable open models through one inference interface. `BAAI/bge-base-en-v1.5` handles embeddings, `dslim/bert-base-NER` provides an inexpensive first entity pass, and Qwen handles structured relationship extraction and final answers.
- **FastAPI** gives us a small typed HTTP boundary for health checks and queries. The API can remain stateless and scale horizontally as the databases and model service scale independently.
- **uv** makes Python versions, dependencies, scripts, and the lockfile reproducible. Every developer and deployment can use the same `uv sync` and `uv run` workflow.
- **The local implementation** stays in the repository because it is cheap for unit tests and offline development. Cloud services are used only when we need durable storage, shared access, or large-scale retrieval.

We deliberately chose two retrieval databases rather than one all-in-one database. A single PostgreSQL deployment with vector and graph extensions can reduce operational overhead for a small system, but Neo4j and Qdrant give clearer scaling boundaries and better specialized query behavior for this project. PostgreSQL can still be added for users, permissions, document metadata, and ingestion state without replacing either retrieval store.

The models are also separated by job. A small NER model is cheaper for entity candidates, an instruction model is more suitable for schema-constrained relationship extraction, and the answer model receives only retrieved evidence. This reduces cost and makes it possible to replace one model without redesigning the storage layer.

Dependencies should be added through `uv`, for example:

```powershell
uv add fastapi neo4j qdrant-client pydantic
uv add --dev ruff mypy
uv sync
```

The local prototype remains available for fast tests, while cloud mode provides the first durable storage and hosted-model boundary.

## Cloud Mode

Copy `.env.example` to `.env` and fill in your Hugging Face, Neo4j AuraDB, and Qdrant Cloud credentials. The application loads this file automatically, then run:

```powershell
uv sync
uv run graph-rag-ingest data
uv run graph-rag-api
```

The API exposes `GET /health` and `POST /query`:

```powershell
Invoke-RestMethod http://localhost:8000/query -Method Post -ContentType "application/json" -Body '{"question":"Where is Alice Johnson\u0027s company based?"}'
```

`POST /query` returns the generated answer together with `sources`, `graph_facts`, and `retrieved_entities`. This makes citations and graph reasoning inspectable by clients instead of hiding them inside the answer prompt.

Cloud mode uses `BAAI/bge-base-en-v1.5` for embeddings, `dslim/bert-base-NER` for entities, and `Qwen/Qwen2.5-7B-Instruct` for relationship extraction and grounded answers through the Hugging Face Inference API. The service currently assumes one Qdrant collection uses one embedding model and dimension.

## AI Safety Dataset

The first real corpus is `Disclosures-SSRC/AI-Safety_Reliability_Reseach`. It contains approximately 9,439 OpenAlex-style research records with titles, abstracts, authors, institutions, publication metadata, and safety classifications. It is a structured research dataset rather than a collection of long documents, so the importer preserves its existing author, paper, institution, and classification relationships instead of asking an LLM to rediscover every fact.

Run a small cloud ingestion first:

```powershell
uv run graph-rag-dataset --limit 100
```

Run the complete dataset after the small run succeeds:

```powershell
uv run graph-rag-dataset --limit 0
```

Dataset ingestion stores an atomic checkpoint at `.graph-rag/ai-safety-checkpoint.json`. If a run is interrupted, repeat the same command; records already checkpointed by stable paper ID are skipped. Use a different checkpoint path when creating a fresh collection:

```powershell
uv run graph-rag-dataset --limit 0 --checkpoint .graph-rag/ai-safety-v2.json
```

Add `--llm-relations` only when you want Qwen to extract additional relationships from abstracts and safety summaries. This increases Hugging Face inference cost and latency, so the default importer uses the dataset's structured fields as high-confidence facts. Source IDs are retained as document identifiers for citations.

## Golden Evaluation Set

The reviewed questions in `evaluation/golden_questions.json` are the quality baseline for this corpus. They cover direct lookup, entity retrieval, graph relationships, multi-hop-style reasoning, and an explicit no-answer case. Run the retrieval evaluation after ingestion:

```powershell
uv run python evaluation/evaluate.py
```

The evaluator measures whether expected entities and relationship predicates appear in retrieved evidence. Generated answer quality should be reviewed separately because LLM wording is nondeterministic.

The evaluator now also checks expected source IDs and answer phrases where provided:

```powershell
uv run python evaluation/evaluate.py --limit 8
```

The report includes document-level `precision_at_k`, `recall_at_k`, and `mrr`. These metrics use only reviewed expected source IDs, so no LLM judge is involved:

```powershell
uv run python evaluation/evaluate.py --limit 8 --report .graph-rag/evaluation-report.json
```

Use RAGAS for model-judged faithfulness, answer relevance, context precision, and context recall. Configure an OpenAI-compatible evaluator endpoint, such as OmniRoute, then run:

```powershell
$env:OPENAI_API_KEY = "your-omniroute-or-provider-key"
$env:OPENAI_BASE_URL = "http://localhost:20128/v1"
$env:RAGAS_LLM_MODEL = "your-evaluator-model"
$env:RAGAS_EMBEDDING_MODEL = "your-embedding-model"
uv run python evaluation/evaluate.py --limit 8 --ragas
```

RAGAS requires an evaluator-capable chat model and embedding model available through the configured endpoint. Review its scores alongside the deterministic source metrics; RAGAS grades answer quality but does not replace a reviewed ground-truth dataset.

## Langfuse Tracing

The query and retrieval pipeline is instrumented with Langfuse. Traces are disabled unless valid Langfuse credentials are configured, so local development continues to work without remote observability.

Add these values to `.env`:

```text
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Then run the API normally:

```powershell
uv run graph-rag-api
```

Each query appears in Langfuse with nested spans for the complete query and retrieval stages. Inspect the question, retrieved evidence, source IDs, graph facts, answer, errors, and latency. Do not enable full prompt or source-text capture in environments containing confidential data unless the retention and access policy allows it.
