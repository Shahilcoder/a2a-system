"""Settings read from environment variables."""

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode


class Settings(BaseSettings):
    # LLM (any OpenAI-compatible API)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "not-set"
    llm_model: str = "gpt-4o-mini"

    redis_url: str = "redis://localhost:6379/0"
    port: int = 8000

    # Per service
    agent_public_url: str = "http://localhost:8000" # URL written into this agent's card
    remote_agent_urls: Annotated[list[str], NoDecode] = [] # sprout: comma-separated agent base URLs
    mcp_server_url: str = "http://localhost:8004/mcp" # information agent
    sprout_url: str = "http://localhost:8001" # gateway

    # External APIs (optional; tools say "not configured" without them)
    tavily_api_key: str = ""
    rapidapi_key: str = ""
    aviationstack_api_key: str = ""

    @field_validator("remote_agent_urls", mode="before")
    @classmethod
    def _split_urls(cls, value: object) -> object:
        if isinstance(value, str):
            return [url.strip() for url in value.split(",") if url.strip()]
        return value
