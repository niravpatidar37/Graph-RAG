from graph_rag.dataset import record_entities, record_relations, record_to_chunk


def test_ai_safety_record_becomes_searchable_graph_evidence() -> None:
    record = {
        "id": "https://openalex.org/W1",
        "title": "Reliable AI Systems",
        "ab": "A study of model robustness.",
        "au_display_name": "Ada Lovelace",
        "institution_display_name": "Analytical Engine Institute",
        "safety_classification": "reliability",
    }

    chunk = record_to_chunk(record)
    relations = record_relations(record)

    assert chunk.document == "https://openalex.org/W1"
    assert "model robustness" in chunk.text
    assert "Ada Lovelace" in record_entities(record)
    assert {relation["predicate"] for relation in relations} == {"AUTHORED", "HAS_TITLE", "AFFILIATED_WITH", "CLASSIFIED_AS"}