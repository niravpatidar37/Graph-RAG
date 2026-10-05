from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langfuse import get_client

from graph_rag.pipeline import CloudGraphRAG
from graph_rag.production import should_refuse_answer


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


def check_golden_sources(pipeline: CloudGraphRAG, cases: Sequence[dict[str, Any]]) -> list[str]:
    """Catch mislabelled ground truth before it turns into fake passes or fake failures.

    Every expected source must exist in the graph, and when its chunk starts with a
    "Title:" line, that title should appear in the question (the golden questions name the
    paper; multi-hop cases that deliberately don't set `"question_names_title": false`).
    A wrong source ID once matched an unrelated paper that merely shared a word.
    """
    problems: list[str] = []
    with pipeline.graph.driver.session() as session:
        for case in cases:
            for source in case.get("expected_sources", []):
                record = session.run(
                    "MATCH (d:Document {id: $id})<-[:FROM_DOCUMENT]-(c:Chunk) RETURN c.text AS text LIMIT 1", id=source,
                ).single()
                if record is None:
                    problems.append(f"{case['id']}: expected source {source} is not indexed")
                    continue
                first_line = str(record["text"]).split("\n", 1)[0]
                if first_line.startswith("Title: ") and case.get("question_names_title", True):
                    title = first_line.removeprefix("Title: ").strip()
                    if title and title.lower() not in str(case["question"]).lower():
                        problems.append(f"{case['id']}: expected source {source} is titled {title!r}, which the question doesn't name")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Graph RAG retrieval against a reviewed golden set.")
    parser.add_argument("--file", type=Path, default=Path("evaluation/golden_questions.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path(".graph-rag/evaluation-report.json"))
    parser.add_argument("--answers", action="store_true", help="Evaluate generated answers in addition to retrieval metrics")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS faithfulness, answer relevance, and context metrics after retrieval evaluation")
    parser.add_argument("--skip-golden-check", action="store_true", help="Don't verify that expected sources exist and match the question")
    args = parser.parse_args()

    cases = json.loads(args.file.read_text(encoding="utf-8"))
    pipeline = CloudGraphRAG.from_env()
    if not args.skip_golden_check:
        problems = check_golden_sources(pipeline, cases)
        for problem in problems:
            print(f"GOLDEN {problem}")
        if problems:
            raise SystemExit("Golden set check failed; fix the expected sources or pass --skip-golden-check.")
    passed = 0
    report: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, float | None]] = []
    ragas_rows: list[dict[str, object]] = []
    evaluate_answers = args.answers or args.ragas
    for case in cases:
        evidence = pipeline.retrieve_evidence(case["question"], args.limit)
        context = str(evidence["context"])
        ranked_source_ids = [str(source["document_id"]) for source in evidence["sources"]]
        source_ids = set(ranked_source_ids)
        found_entities = [entity for entity in case.get("expected_entities", []) if entity.lower() in context.lower()]
        found_relationships = [relation for relation in case.get("expected_relationships", []) if relation.lower() in context.lower()]
        found_sources = [source for source in case.get("expected_sources", []) if source in source_ids]
        entity_ok = len(found_entities) == len(case.get("expected_entities", []))
        relation_ok = len(found_relationships) == len(case.get("expected_relationships", []))
        source_ok = len(found_sources) == len(case.get("expected_sources", []))
        metrics = retrieval_metrics(ranked_source_ids, case.get("expected_sources", []))
        retrieval_rows.append(metrics)
        case_passed = entity_ok and relation_ok
        if case.get("expected_sources"):
            case_passed = case_passed and source_ok
        expected_answer = [value.lower() for value in case.get("expected_answer_contains", [])]
        answer = ""
        answer_ok: bool | None = None
        answer_error = ""
        if evaluate_answers and case.get("evaluate_answer", True):
            try:
                answer = pipeline.models.answer(case["question"], context)
                answer_ok = all(value in answer.lower() for value in expected_answer)
            except Exception as error:
                answer_error = str(error)
                print(f"WARN {case['id']}: answer generation unavailable: {error}")
        if expected_answer and answer_ok is not None:
            case_passed = case_passed and answer_ok
        if case.get("no_answer") and answer_ok is not None:
            case_passed = case_passed and not found_relationships and should_refuse_answer(context, answer)
        _submit_scores(
            trace_id=str(evidence.get("trace_id", "")),
            scores={
                "answer_correct": float(answer_ok) if answer_ok is not None else float(case_passed),
                "source_recall": float(source_ok) if case.get("expected_sources") else float(case_passed),
                "relationship_recall": float(relation_ok),
                "case_passed": float(case_passed),
            },
        )
        passed += int(case_passed)
        report.append({
            "id": case["id"],
            "passed": case_passed,
            "answer_correct": answer_ok,
            "answer_error": answer_error or None,
            "source_recall": source_ok,
            "relationship_recall": relation_ok,
            "trace_id": str(evidence.get("trace_id", "")),
            "retrieval": metrics,
        })
        if answer_ok is not None and expected_answer and not case.get("no_answer"):
            ragas_rows.append({
                "question": case["question"],
                "answer": answer,
                "ground_truth": " ".join(case["expected_answer_contains"]),
                "contexts": [str(source["text"]) for source in evidence["sources"]],
            })
        precision = metrics["precision"]
        recall = metrics["recall"]
        mrr = metrics["mrr"]
        answer_status = "not evaluated" if answer_ok is None else "ok" if answer_ok else "miss"
        print(f"{'PASS' if case_passed else 'FAIL'} {case['id']}: precision={precision:.2f} recall={recall:.2f} mrr={mrr:.2f} answer={answer_status}" if precision is not None and recall is not None and mrr is not None else f"{'PASS' if case_passed else 'FAIL'} {case['id']}: no answerable source metric answer={answer_status}")
    summary: dict[str, Any] = {
        "case_pass_rate": passed / max(len(cases), 1),
        "precision_at_k": mean_metric(retrieval_rows, "precision"),
        "recall_at_k": mean_metric(retrieval_rows, "recall"),
        "mrr": mean_metric(retrieval_rows, "mrr"),
        "evaluated_cases": len(cases),
    }
    print("Evaluation summary: " + json.dumps(summary))
    ragas_result: dict[str, float] | None = _run_ragas(ragas_rows) if args.ragas else None
    if ragas_result is not None:
        summary["ragas"] = ragas_result
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "cases": report}, indent=2), encoding="utf-8")


def _submit_scores(trace_id: str, scores: dict[str, float]) -> None:
    if not trace_id:
        return
    client = get_client()
    for name, value in scores.items():
        client.create_score(name=name, value=value, trace_id=trace_id, data_type="NUMERIC")
    client.flush()


def _run_ragas(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("--ragas requires OPENAI_API_KEY and an OpenAI-compatible evaluator endpoint.")

    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # pyright: ignore[reportMissingTypeStubs]
    from ragas.llms import LangchainLLMWrapper  # pyright: ignore[reportMissingTypeStubs]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # pyright: ignore[reportMissingTypeStubs]

    model = os.environ.get("RAGAS_LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.environ.get("OPENAI_BASE_URL")
    llm = LangchainLLMWrapper(ChatOpenAI(model=model, base_url=base_url, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model, base_url=base_url))
    result = evaluate(
        Dataset.from_list(rows),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    aggregates: dict[str, list[float]] = {}
    for row in result.scores:
        for name, value in row.items():
            if isinstance(value, int | float) and math.isfinite(float(value)):
                aggregates.setdefault(name, []).append(float(value))
    return {name: sum(values) / len(values) for name, values in aggregates.items() if values}


if __name__ == "__main__":
    main()