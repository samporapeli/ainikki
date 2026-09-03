import pytest


@pytest.fixture(autouse=True)
def configure_test_llm_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://test.invalid/v1")
