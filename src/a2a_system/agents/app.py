from fastapi import FastAPI
from openai import AsyncOpenAI
from redis.asyncio import Redis

from a2a_system.agents.base import LLMAgent
from a2a_system.agents.coding import CodingAgent
from a2a_system.agents.information import InformationAgent
from a2a_system.agents.sprout import SproutAgent
from a2a_system.config import Settings
from a2a_system.protocol.client import A2AClient
from a2a_system.llm.client import LLMClient
from a2a_system.protocol.request_handler import RequestHandler
from a2a_system.protocol.server import build_app
from a2a_system.storage import ConversationStore, TaskStore


def build_agent(service: str, settings: Settings) -> FastAPI:
    redis = Redis.from_url(settings.redis_url)
    llm = LLMClient(AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key), settings.llm_model)
    conversations = ConversationStore(redis, service)
    agent: LLMAgent

    if service == "sprout":
        agent = SproutAgent(llm, conversations, A2AClient(), settings.remote_agent_urls)
    elif service == "information":
        agent = InformationAgent(llm, conversations, settings.mcp_server_url)
    else:
        agent = CodingAgent(llm, conversations)

    return build_app(agent.card(settings.agent_public_url), RequestHandler(agent, TaskStore(redis, service)))
