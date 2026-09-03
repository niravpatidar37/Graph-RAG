from graph_rag.evaluation import mean_metric, retrieval_metrics


def test_retrieval_metrics_calculates_precision_recall_and_mrr() -> None:
    metrics = retrieval_metrics(["wrong", "right", "other"], ["right", "also-right"])

    assert metrics == {"precision": 1 / 3, "recall": 0.5, "mrr": 0.5}


def test_retrieval_metrics_excludes_unanswerable_questions() -> None:
    metrics = retrieval_metrics(["source"], [])

    assert metrics == {"precision": None, "recall": None, "mrr": None}
    assert mean_metric([metrics], "precision") is None