from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

try:
    import redis
except ModuleNotFoundError:  # pragma: no cover - dependency is optional until runtime setup
    redis = None


@dataclass
class IngestionCheckpoint:
    path: str | Path
    processed: set[str] = field(default_factory=set)
    status: str = "in_progress"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.processed = set(payload.get("processed", []))
            self.status = str(payload.get("status", "in_progress"))

    def mark_processed(self, record_id: str) -> None:
        self.processed.add(record_id)
        self.status = "in_progress"
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self.status,
            "processed": sorted(self.processed),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    fn: Callable[[], Any]
    retries: int = 0


class InMemoryJobQueue:
    def __init__(self) -> None:
        self._jobs: deque[QueueJob] = deque()

    def enqueue(self, job_id: str, fn: Callable[[], Any], retries: int = 0) -> QueueJob:
        job = QueueJob(job_id=job_id, fn=fn, retries=retries)
        self._jobs.append(job)
        return job

    def run_next(self) -> Any:
        if not self._jobs:
            raise RuntimeError("No queued jobs to run")

        job = self._jobs.popleft()
        for attempt in range(job.retries + 1):
            try:
                return job.fn()
            except Exception:
                if attempt >= job.retries:
                    raise
                self._jobs.append(job)
                return self.run_next()
        raise RuntimeError(f"Job {job.job_id} failed after retries")


JOB_REGISTRY: dict[str, Callable[..., Any]] = {}


class RedisJobQueue:
    def __init__(self, url: str = "redis://localhost:6379/0", queue_name: str = "graph-rag-jobs") -> None:
        if redis is None:
            raise ModuleNotFoundError("redis package is required to use RedisJobQueue. Install it with: pip install redis")
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.queue_name = queue_name

    def enqueue(self, job_id: str, task_name: str, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "job_id": job_id,
            "task": task_name,
            "args": kwargs,
            "retries": 0,
        }
        self.client.rpush(self.queue_name, json.dumps(payload))
        return payload

    def run_next(self) -> Any:
        item = self.client.lpop(self.queue_name)
        if item is None:
            raise RuntimeError("No queued jobs to run")
        payload = json.loads(item)
        task_name = payload["task"]
        handler = JOB_REGISTRY.get(task_name)
        if handler is None:
            raise KeyError(f"No handler registered for task '{task_name}'")
        return handler(**payload.get("args", {}))


def rerank_evidence(question: str, hits: list[dict[str, Any]], entities: list[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    stop_words = {"a", "an", "and", "are", "as", "at", "based", "by", "for", "how", "is", "of", "the", "which", "with"}
    query_terms = {
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", question)
        if token.lower() not in stop_words and len(token) > 2
    }
    entity_names = [entity.lower() for entity in (entities or []) if entity]
    scored: list[tuple[float, dict[str, Any]]] = []
    seen_documents: set[str] = set()
    for hit in hits:
        document = str(hit.get("document", ""))
        if document in seen_documents:
            continue
        seen_documents.add(document)
        text = str(hit.get("text", ""))
        text_lower = text.lower()
        score = float(hit.get("score", 0.0))
        entity_overlap = sum(1 for entity in entity_names if entity in text_lower)
        keyword_overlap = sum(1 for term in query_terms if term and term in text_lower)
        score += entity_overlap * 2.0
        score += keyword_overlap * 1.5
        scored.append((score, {**hit, "score": score}))
    ranked = [hit for _, hit in sorted(scored, key=lambda item: item[0], reverse=True)]
    return ranked[:limit]


def should_refuse_answer(context: str, answer: str) -> bool:
    cleaned_context = (context or "").strip()
    cleaned_answer = (answer or "").strip()
    if not cleaned_context or not cleaned_answer:
        return True
    answer_lower = cleaned_answer.lower()
    if any(marker in answer_lower for marker in (
        "i don't know",
        "cannot determine",
        "not enough information",
        "insufficient evidence",
        "unable to determine",
    )):
        return True
    return False


@dataclass
class MetricsCollector:
    counters: dict[str, int] = field(default_factory=dict)
    timings: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def timing(self, name: str, duration_seconds: float) -> None:
        self.timings.setdefault(name, []).append(duration_seconds)

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        start = perf_counter()
        try:
            return fn()
        finally:
            self.timing(name, perf_counter() - start)

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "timings": {name: sorted(values) for name, values in self.timings.items()},
        }


@dataclass(frozen=True)
class TenantPolicy:
    allowed_document_ids: set[str] | None = None
    allowed_tenants: set[str] | None = None

    def can_access(self, tenant: str | None, document_id: str) -> bool:
        if self.allowed_tenants is not None and tenant not in self.allowed_tenants:
            return False
        if self.allowed_document_ids is not None and document_id not in self.allowed_document_ids:
            return False
        return True

    def filter_documents(self, tenant: str | None, document_ids: list[str]) -> list[str]:
        return [doc_id for doc_id in document_ids if self.can_access(tenant, doc_id)]


@dataclass
class IngestionService:
    checkpoint: IngestionCheckpoint
    queue: InMemoryJobQueue = field(default_factory=InMemoryJobQueue)

    def enqueue(self, job_id: str, fn: Callable[[], Any], retries: int = 0) -> QueueJob:
        if job_id in self.checkpoint.processed:
            return QueueJob(job_id=job_id, fn=lambda: None, retries=retries)
        return self.queue.enqueue(job_id, fn, retries=retries)

    def run_pending(self) -> list[Any]:
        results: list[Any] = []
        while self.queue._jobs:
            job = self.queue._jobs[0]
            try:
                result = self.queue.run_next()
                self.checkpoint.mark_processed(job.job_id)
                results.append(result)
            except Exception:
                if job.job_id in self.checkpoint.processed:
                    raise
                raise
        return results
