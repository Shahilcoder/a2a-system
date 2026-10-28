"""Information agent: answers questions with live data from the MCP server's tools."""

import logging

from a2a_system.agents.base import LLMAgent
from a2a_system.llm.client import LLMClient
from a2a_system.protocol.models import AgentSkill
from a2a_system.storage import ConversationStore
from a2a_system.tools.base import Tool
from a2a_system.tools.mcp_tool import McpTarget, load_mcp_tools

log = logging.getLogger(__name__)


class InformationAgent(LLMAgent):
    name = "Information Agent"
    description = "Looks up live information: weather, news, web search, Indian trains and flights."
    skills = [
        AgentSkill(id="weather", name="Weather", description="Current weather and forecast for a city.",
                   tags=["weather"], examples=["Weather in Mumbai this weekend?"]),
        AgentSkill(id="news", name="News", description="Latest news headlines, optionally on a topic.",
                   tags=["news"]),
        AgentSkill(id="web-search", name="Web search", description="Search the web for anything else.",
                   tags=["search"]),
        AgentSkill(id="travel", name="Trains and flights",
                   description="Indian Railways trains between stations, and flight status between airports.",
                   tags=["travel", "trains", "flights"], examples=["Trains from NDLS to BCT on 2026-11-02"]),
    ]
    system_prompt = (
        "You are an information agent. Use your tools to answer with current, factual data and cite "
        "sources (links) when the tool gives them. If a tool fails or is not configured, say so plainly. "
        "Be concise."
    )

    def __init__(self, llm: LLMClient, conversations: ConversationStore, mcp_server: McpTarget):
        super().__init__(llm, conversations)
        self.mcp_server = mcp_server
        self._tools: list[Tool] | None = None

    async def tools(self) -> list[Tool]:
        if self._tools is None:
            try:
                self._tools = await load_mcp_tools(self.mcp_server)
            except Exception as exc:  # not cached: retried on the next request
                log.warning("MCP server unavailable: %s", exc)
                return []
        return self._tools
