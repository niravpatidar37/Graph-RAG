from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .dataset import record_relations, record_to_chunk
from .pipeline import CloudGraphRAG
from .production import IngestionCheckpoint
from .stores import Chunk

DATASET_NAME = "Disclosures-SSRC/AI-Safety_Reliability_Reseach"


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("paper_id") or record.get("id") or record.get("url") or record.get("title"))


def batches(records: Iterable[dict[str, Any]], checkpoint: IngestionCheckpoint, limit: int, size: int,
            stats: dict[str, int] | None = None) -> Iterator[list[dict[str, Any]]]:
    """Yield unprocessed records in batches. `stats["skipped"]` counts checkpointed records,
    including those after the last batch (a fully checkpointed run yields nothing)."""
    stats = stats if stats is not None else {}
    stats.setdefault("skipped", 0)
    batch: list[dict[str, Any]] = []
    taken = 0
    for record in records:
        if record_id(record) in checkpoint.processed:
            stats["skipped"] += 1
            continue
        batch.append(record)
        taken += 1
        if len(batch) >= size:
            yield batch
            batch = []
        if limit and taken >= limit:
            break
    if batch:
        yield batch


def ingest_with_retry(pipeline: CloudGraphRAG, items: list[tuple[Chunk, list[dict[str, Any]]]], attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            pipeline.ingest_batch(items)  # type: ignore[arg-type]
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream the AI Safety dataset into Graph RAG stores.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to ingest; use 0 for all rows")
    parser.add_argument("--batch-size", type=int, default=64, help="Records embedded and written per batch")
    parser.add_argument("--llm-relations", action="store_true", help="Also ask the chat model to extract relationships from each record")
    parser.add_argument("--checkpoint", type=Path, default=Path(".graph-rag/ai-safety-checkpoint.json"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    from datasets import load_dataset

    pipeline = CloudGraphRAG.from_env()
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    checkpoint = IngestionCheckpoint(args.checkpoint)
    indexed = 0
    stats: dict[str, int] = {"skipped": 0}
    started = time.perf_counter()
    for batch in batches(dataset, checkpoint, args.limit, args.batch_size, stats):
        items: list[tuple[Chunk, list[dict[str, Any]]]] = []
        for record in batch:
            chunk = record_to_chunk(record)
            relations = record_relations(record)
            if args.llm_relations:
                relations.extend(pipeline.models.relations(chunk.text))
            items.append((chunk, relations))
        ingest_with_retry(pipeline, items)
        checkpoint.mark_many(record_id(record) for record in batch)  # only after both stores accepted the batch
        indexed += len(batch)
        rate = indexed / max(time.perf_counter() - started, 1e-9)
        print(f"indexed {indexed} records ({rate:.1f}/s)", flush=True)
    print(f"Indexed {indexed} AI Safety records; skipped {stats['skipped']} checkpointed records.")


if __name__ == "__main__":
    main()
