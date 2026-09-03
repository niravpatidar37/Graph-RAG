import pytest

from graph_rag.settings import Settings


def test_missing_cloud_configuration_is_explicit(monkeypatch) -> None:
    for name in ("HF_TOKEN", "NEO4J_URI", "NEO4J_PASSWORD", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.setenv(name, "")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        Settings.from_env().validate_cloud()