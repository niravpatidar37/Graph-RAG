from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable


@dataclass
class IngestionCheckpoint:
    path: str | Path
    processed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.processed = set(payload.get("processed", []))

    def mark_processed(self, record_id: str) -> None:
        self.processed.add(record_id)
        self.save()

    def mark_many(self, record_ids: Iterable[str]) -> None:
        """Mark a whole batch with one atomic write."""
        self.processed.update(record_ids)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"processed": sorted(self.processed)}, indent=2), encoding="utf-8")
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


_RERANK_STOP_WORDS = frozenset(
    "a an and are as at based be by did does for from has have how in is it its of on or that the this to was what when where which who why with".split()
)


def rerank_evidence(question: str, hits: list[dict[str, Any]], entities: list[str] | None = None, limit: int = 10,
                    entity_weight: float = 0.35, term_weight: float = 0.25, graph_weight: float = 0.3) -> list[dict[str, Any]]:
    """Blend vector similarity with how much of the question's entities and terms a chunk covers.

    Both bonuses are fractions in [0, 1], so they nudge the cosine score instead of drowning
    it: the earlier per-match bonus (+2.0 per entity, +1.5 per word) let any chunk sharing
    common words like "language" or "models" outrank the chunk that actually answers.
    The raw cosine score is kept as `similarity` so clients can show both. Chunks reached
    through the graph (`via_graph`: they mention an entity the question names) get a fixed
    bonus, because their text often lacks the very words that connect them to the question.
    """
    query_terms = {
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", question)
        if token.lower() not in _RERANK_STOP_WORDS and len(token) > 2
    }
    entity_names = list(dict.fromkeys(entity.lower() for entity in (entities or []) if entity and len(entity) > 1))
    scored: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        text_lower = str(hit.get("text", "")).lower()
        similarity = float(hit.get("score", 0.0))
        entity_cover = sum(1 for entity in entity_names if entity in text_lower) / len(entity_names) if entity_names else 0.0
        term_cover = sum(1 for term in query_terms if term in text_lower) / len(query_terms) if query_terms else 0.0
        score = similarity + entity_weight * entity_cover + term_weight * term_cover + (graph_weight if hit.get("via_graph") else 0.0)
        scored.append((score, {**hit, "score": round(score, 6), "similarity": similarity}))
    return [hit for _, hit in sorted(scored, key=lambda item: item[0], reverse=True)][:limit]


def should_refuse_answer(context: str, answer: str) -> bool:
    cleaned_answer = (answer or "").strip().lower().replace("\u2019", "'")
    if not (context or "").strip() or not cleaned_answer:
        return True
    return any(marker in cleaned_answer for marker in (
        "i don't know",
        "i do not know",
        "cannot determine",
        "not enough information",
        "don't have enough",
        "do not have enough",
        "insufficient evidence",
        "unable to determine",
    ))


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
            results.append(self.queue.run_next())
            self.checkpoint.mark_processed(job.job_id)
        return results
