from graph_rag import GraphRAG


def test_graph_expansion_finds_related_context() -> None:
    rag = GraphRAG()
    rag.add_documents([
        ("notes.md", "Alice Johnson works at Acme Labs. Acme Labs is based in Seattle."),
        ("team.md", "Alice Johnson collaborates with Bob Smith on retrieval systems."),
    ])

    results = rag.search("Where is Alice Johnson's company based?")

    assert results
    assert results[0].document == "notes.md"
    assert "Seattle" in results[0].text
    assert "Acme Labs" in rag.nodes
    assert "Acme Labs" in rag.edges["Seattle"]
