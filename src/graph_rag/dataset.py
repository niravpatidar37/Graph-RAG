from __future__ import annotations

from typing import Any

from .stores import Chunk


def record_to_chunk(record: dict[str, Any]) -> Chunk:
    paper_id = str(record.get("paper_id") or record.get("id") or record.get("url") or record.get("title"))
    title = str(record.get("title") or record.get("title_clean") or "Untitled paper")
    abstract = str(record.get("ab") or "").strip()
    summary = str(record.get("reasoning_summary") or "").strip()
    classification = str(record.get("safety_classification") or "").strip()
    institution = str(record.get("institution_display_name") or record.get("institution_name") or "").strip()
    text_parts = [f"Title: {title}"]
    if abstract:
        text_parts.append(f"Abstract: {abstract}")
    if summary:
        text_parts.append(f"Safety reasoning: {summary}")
    if classification:
        text_parts.append(f"Safety classification: {classification}")
    if institution:
        text_parts.append(f"Institution: {institution}")
    return Chunk(paper_id, "\n".join(text_parts), record_entities(record))


def record_entities(record: dict[str, Any]) -> list[str]:
    values = [
        record.get("title"),
        record.get("institution_display_name"),
        record.get("institution_name"),
        record.get("institution_group"),
        record.get("au_display_name"),
        record.get("safety_classification"),
    ]
    return sorted({str(value).strip() for value in values if value and str(value).strip()})


def record_relations(record: dict[str, Any]) -> list[dict[str, Any]]:
    paper = str(record.get("paper_id") or record.get("id") or record.get("title") or "").strip()
    title = str(record.get("title") or record.get("title_clean") or "").strip()
    author = str(record.get("au_display_name") or "").strip()
    institution = str(record.get("institution_display_name") or record.get("institution_name") or "").strip()
    classification = str(record.get("safety_classification") or "").strip()
    relations = []
    for source, predicate, target in [
        (author, "AUTHORED", paper),
        (paper, "HAS_TITLE", title),
        (paper, "AFFILIATED_WITH", institution),
        (paper, "CLASSIFIED_AS", classification),
    ]:
        if source and target:
            relations.append({"source": source, "predicate": predicate, "target": target, "confidence": 1.0})
    return relations