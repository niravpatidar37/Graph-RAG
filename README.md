# Graph RAG

A Graph RAG system: semantic document retrieval combined with entity and relationship traversal, so answers can draw on both unstructured evidence and structured connections between things.

The full design — ingestion plane, query plane, data ownership, reliability rules, security boundaries, scaling milestones — lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This file is the practical guide to running it.

## Two ways to run it

**Local prototype** — zero setup, in-memory, regex-based entity extraction. Good for tests and offline development, not for a real corpus.

**Full pipeline** — Neo4j for entities/relationships, Qdrant for chunk vectors, a chat model for relation extraction and answers, and an embeddings+NER model for indexing. Every one of those four pieces is swappable between a hosted service and something running on your own machine — pick per-piece, not all-or-nothing.

## Local prototype

```powershell
uv sync
uv run pytest
uv run graph-rag data "Where is Alice Johnson's company based?"
```

Add your own `.md`/`.txt` files to `data/` and query the directory again. This path never leaves your machine and needs no credentials — it's [src/graph_rag/core.py](src/graph_rag/core.py), a self-contained ~90 lines.

## Full pipeline

### The four pieces, and your options for each

| Piece | Hosted | Self-hosted |
|---|---|---|
| Graph store | Neo4j AuraDB (free tier) | Neo4j in Docker |
| Vector store | Qdrant Cloud (free tier) | Qdrant in Docker |
| Chat (relations + answers) | Hugging Face Inference API — billed per token | [Ollama](https://ollama.com), any OpenAI-compatible local server |
| Embeddings + NER | Hugging Face Inference API — free | Local GPU/CPU via `sentence-transformers`/`transformers` |

Free-tier cloud instances get suspended after inactivity, and HF's chat inference has a monthly credit cap — the self-hosted column has no such limits and, with a GPU, is faster. Mix and match freely; nothing else in the code changes based on which you pick.

### Fastest path: everything local

```powershell
# Graph + vector stores
docker run -d --name graph-rag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:5-community
docker run -d --name graph-rag-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Chat model
winget install Ollama.Ollama   # or https://ollama.com
ollama pull llama3.2
```

Then `.env`:

```bash
HF_TOKEN=your_hf_token              # still needed unless LOCAL_MODELS=1 (embeddings/NER)
HF_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
HF_NER_MODEL=dslim/bert-base-NER
HF_LLM_MODEL=llama3.2               # the model name your chat backend serves
LLM_BASE_URL=http://localhost:11434/v1   # blank = use HF Inference for chat instead
LOCAL_MODELS=1                      # 1 = embeddings/NER run locally; blank = use HF Inference

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=yourpassword

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=anything-nonempty    # unauthenticated local Qdrant still needs a non-empty value here
QDRANT_COLLECTION=graph_rag_chunks
```

`LOCAL_MODELS=1` loads `BAAI/bge-base-en-v1.5` and `dslim/bert-base-NER` into memory on first use and puts them on your GPU if `torch.cuda.is_available()`. This repo pins a CUDA-enabled `torch` build via `[tool.uv.sources]` in `pyproject.toml` — on Windows, `uv add torch` alone gets you a CPU-only wheel, so that pin matters. Verify it landed correctly:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Cloud path instead

Sign up for [Neo4j AuraDB](https://neo4j.com/cloud/aura/) and [Qdrant Cloud](https://cloud.qdrant.io) free tiers, leave `LLM_BASE_URL` and `LOCAL_MODELS` blank, and fill in `NEO4J_URI`/`QDRANT_URL`/`QDRANT_API_KEY` with the values from each dashboard instead of localhost. Everything else is identical.

### Run it

```powershell
uv sync
uv run graph-rag-ingest data
uv run graph-rag-api
```

The API exposes:

| Route | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /metrics` | per-stage timing/counters snapshot |
| `POST /query` | ask a question, get the full answer back at once |
| `POST /query/stream` | same, but as Server-Sent Events — an `evidence` event first, then `token` events as the answer generates, then `done` |
| `GET /` | a minimal built-in page that calls `/query/stream` |

```powershell
Invoke-RestMethod http://localhost:8000/query -Method Post -ContentType "application/json" -Body '{"question":"Where is Alice Johnson's company based?"}'
```

Every response returns `sources`, `graph_facts`, and `retrieved_entities` alongside the answer — citations and graph reasoning are inspectable by the client, not buried inside the generated text. When context doesn't support an answer, `/query` returns the canned refusal after generating; `/query/stream` catches the empty-context case before generating and streams the refusal directly, so a model that hedges mid-stream is shown as-is rather than retroactively rewritten.

## Why two retrieval databases, and swappable models

Semantic similarity and relationship traversal are different query shapes, so they get different specialized stores rather than one database doing both. Neo4j makes multi-hop Cypher queries natural; Qdrant is built for vector search at scale. A single Postgres-with-extensions deployment is a legitimate simpler alternative for a smaller system — this project chose clearer scaling boundaries instead.

Models are separated the same way, by job: a small NER model handles entity candidates cheaply, an instruction-tuned chat model handles schema-constrained relation extraction and grounded answers, and neither is hardwired to a specific provider. `chat_client` in `HuggingFaceModels` points at HF's router by default and at `LLM_BASE_URL` when one is set — any OpenAI-compatible `/v1/chat/completions` server works, tested against both HF's hosted router and a local Ollama. Embeddings/NER take the same shape: HF Inference by default, local `sentence-transformers`/`transformers` when `LOCAL_MODELS=1`. Swapping either doesn't touch the storage layer.

Add dependencies through `uv`:

```powershell
uv add <package>
uv add --dev <package>
uv sync
```

## AI Safety Dataset

The first real corpus is `Disclosures-SSRC/AI-Safety_Reliability_Reseach` — roughly 9,439 OpenAlex-style research records with titles, abstracts, authors, institutions, and safety classifications. It's structured data rather than long documents, so the importer preserves its existing author/paper/institution/classification relationships instead of asking an LLM to rediscover every fact.

```powershell
uv run graph-rag-dataset --limit 100   # small run first
uv run graph-rag-dataset --limit 0     # then the full dataset
```

Ingestion checkpoints atomically at `.graph-rag/ai-safety-checkpoint.json`; an interrupted run resumes by paper ID when repeated. Use a different `--checkpoint` path to start a fresh collection.

Add `--llm-relations` only when you want the chat model to extract additional relationships from abstracts and safety summaries — this adds one chat call per record, so it costs more and runs slower than the default structured-field import.

## Golden Evaluation Set

`evaluation/golden_questions.json` is the quality baseline: direct lookup, entity retrieval, graph relationships, multi-hop reasoning, and an explicit no-answer case.

```powershell
uv run python evaluation/evaluate.py --limit 8 --report .graph-rag/evaluation-report.json
```

Reports `precision_at_k`, `recall_at_k`, and `mrr` from reviewed expected source IDs — no LLM judge involved. Add `--answers` to also check generated-answer content, or `--ragas` for model-judged faithfulness/relevance/context scores through an OpenAI-compatible evaluator endpoint (e.g. a local router, or OpenAI itself):

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "http://localhost:20128/v1"
$env:RAGAS_LLM_MODEL = "your-evaluator-model"
$env:RAGAS_EMBEDDING_MODEL = "your-embedding-model"
uv run python evaluation/evaluate.py --limit 8 --ragas
```

RAGAS needs its own evaluator-capable chat and embedding models available at that endpoint — review its scores alongside the deterministic source metrics, it grades answer quality but doesn't replace a reviewed ground-truth set.

## Langfuse Tracing

The query and retrieval pipeline is instrumented with Langfuse `@observe` spans. Tracing is inert until credentials are present, so local development works without it.

```bash
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Each query then appears in Langfuse with nested spans for the query and retrieval stages — question, retrieved evidence, source IDs, graph facts, answer, errors, latency. Don't enable full prompt/source-text capture on confidential data unless your retention and access policy allows it.
