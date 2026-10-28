from a2a_system.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.port == 8000
    assert settings.remote_agent_urls == []
    assert settings.tavily_api_key == ""


def test_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "my-model")
    monkeypatch.setenv("REMOTE_AGENT_URLS", "http://a:8000, http://b:8000,")
    settings = Settings()
    assert settings.llm_model == "my-model"
    assert settings.remote_agent_urls == ["http://a:8000", "http://b:8000"]
