from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from .dataset import record_relations, record_to_chunk
from .pipeline import CloudGraphRAG
from .production import IngestionCheckpoint, IngestionService


DATASET_NAME = "Disclosures-SSRC/AI-Safety_Reliability_Reseach"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream the AI Safety dataset into Graph RAG cloud stores.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to ingest; use 0 for all rows")
    parser.add_argument("--llm-relations", action="store_true", help="Also ask Qwen to extract relationships from each record")
    parser.add_argument("--checkpoint", type=Path, default=Path(".graph-rag/ai-safety-checkpoint.json"))
    args = parser.parse_args()

    pipeline = CloudGraphRAG.from_env()
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    checkpoint = IngestionCheckpoint(args.checkpoint)
    service = IngestionService(checkpoint)
    indexed = 0
    skipped = 0
    for record in dataset:
        record_id = str(record.get("paper_id") or record.get("id") or record.get("url") or record.get("title"))
        if record_id in checkpoint.processed:
            skipped += 1
            continue
        service.enqueue(
            record_id,
            lambda record=record, record_id=record_id: _ingest_record(pipeline, record, args.llm_relations, record_id, checkpoint),
            retries=2,
        )
        indexed += 1
        if args.limit and indexed >= args.limit:
            break

    service.run_pending()
    print(f"Indexed {indexed} AI Safety records; skipped {skipped} checkpointed records.")


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def _save_checkpoint(path: Path, processed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")
    temporary.replace(path)


def _ingest_record(pipeline: CloudGraphRAG, record: dict[str, object], include_llm_relations: bool, record_id: str, checkpoint: IngestionCheckpoint) -> str:
    chunk = record_to_chunk(record)
    relations = record_relations(record)
    if include_llm_relations:
        relations.extend(pipeline.models.relations(chunk.text))
    pipeline.ingest_chunk(chunk, relations)
    checkpoint.mark_processed(record_id)
    return chunk.document
