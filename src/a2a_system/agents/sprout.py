"""Sprout: the friendly host agent. It talks to the user and delegates to remote agents."""

import logging

from a2a_system.agents.base import LLMAgent
from a2a_system.llm.client import LLMClient
from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.models import AgentSkill
from a2a_system.storage import ConversationStore
from a2a_system.tools.base import Tool
from a2a_system.tools.remote_agent_tool import RemoteAgentTool

log = logging.getLogger(__name__)


class SproutAgent(LLMAgent):
    name = "Sprout"
    description = "A friendly assistant that chats with you and asks specialist agents for help."
    skills = [AgentSkill(id="chat", name="Friendly chat",
                         description="Conversation, plus answers from the information and coding agents.",
                         tags=["chat", "assistant"], examples=["What's the weather in Delhi?", "Write a Python fizzbuzz"])]
    system_prompt = (
        "You are Sprout, a warm, upbeat and concise assistant. "
        "You have specialist agents available as tools. Use them for live facts (weather, news, web search, "
        "trains, flights) and for writing or explaining code. Never invent live data. "
        "The agents do not see this conversation, so every query you send them must be complete and "
        "self-contained (resolve 'there', 'it', 'tomorrow' etc. into concrete names and dates). "
        "Answer simple chit-chat yourself."
    )

    def __init__(self, llm: LLMClient, conversations: ConversationStore, a2a: A2AClient, remote_urls: list[str]):
        super().__init__(llm, conversations)
        self.a2a, self.remote_urls = a2a, remote_urls
        self._tools: dict[str, Tool] = {}  # url -> tool, filled lazily

    async def tools(self) -> list[Tool]:
        for url in self.remote_urls:
            if url in self._tools:
                continue
            try:
                self._tools[url] = RemoteAgentTool(await self.a2a.get_agent_card(url), url, self.a2a)
            except Exception as exc:  # not cached: retried on the next request
                log.warning("remote agent %s unavailable: %s", url, exc)
        return list(self._tools.values())
