from graph_rag.dataset_ingest import _load_checkpoint, _save_checkpoint


def test_checkpoint_round_trip_is_atomic(tmp_path) -> None:
    checkpoint = tmp_path / "state.json"
    _save_checkpoint(checkpoint, {"paper-2", "paper-1"})

    assert _load_checkpoint(checkpoint) == {"paper-1", "paper-2"}
    assert not checkpoint.with_suffix(".json.tmp").exists()