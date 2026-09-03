from __future__ import annotations

from collections.abc import Sequence


def retrieval_metrics(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> dict[str, float | None]:
    """Calculate document-level retrieval metrics for one answerable question."""
    expected = set(expected_ids)
    if not expected:
        return {"precision": None, "recall": None, "mrr": None}

    relevant_positions = [index for index, source_id in enumerate(retrieved_ids, start=1) if source_id in expected]
    relevant_retrieved = len(relevant_positions)
    return {
        "precision": relevant_retrieved / max(len(retrieved_ids), 1),
        "recall": relevant_retrieved / len(expected),
        "mrr": 1 / relevant_positions[0] if relevant_positions else 0.0,
    }


def mean_metric(rows: Sequence[dict[str, float | None]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row[name] is not None]
    return sum(values) / len(values) if values else None