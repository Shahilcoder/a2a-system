"""LLMAgent: memory, a tool-calling loop, and the answer streamed back as artifact chunks."""

import json
import logging
from collections.abc import AsyncIterator
from datetime import date
from typing import Any
from uuid import uuid4

from a2a_system.llm.client import LLMClient, TextDelta, ToolCall
from a2a_system.protocol.executor import AgentExecutor, RequestContext
from a2a_system.protocol.models import AgentCard, AgentInterface, AgentSkill, Event, TaskState
from a2a_system.storage import ConversationStore
from a2a_system.tools.base import Tool

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


class LLMAgent(AgentExecutor):
    """Subclasses set the class attributes and, if they have tools, override `tools()`."""

    name: str
    description: str
    skills: list[AgentSkill]
    system_prompt: str

    def __init__(self, llm: LLMClient, conversations: ConversationStore):
        self.llm, self.conversations = llm, conversations

    def card(self, url: str) -> AgentCard:
        return AgentCard(name=self.name, description=self.description, version="1.0.0",
                         supported_interfaces=[AgentInterface(url=url)], skills=self.skills)

    async def tools(self) -> list[Tool]:
        return []

    async def execute(self, ctx: RequestContext) -> AsyncIterator[Event]:
        yield ctx.status(TaskState.WORKING)
        tools = {tool.name: tool for tool in await self.tools()}
        history = await self.conversations.history(ctx.context_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{self.system_prompt}\n\nToday's date: {date.today().isoformat()}"},
            *history,
            {"role": "user", "content": ctx.user_text},
        ]

        artifact_id = uuid4().hex
        answer = ""
        for round_ in range(MAX_TOOL_ROUNDS + 1):
            offered = list(tools.values()) if round_ < MAX_TOOL_ROUNDS else []  # last round: answer, no more tools
            round_text, calls = "", []
            async for item in self.llm.stream(messages, [tool.schema() for tool in offered]):
                if isinstance(item, TextDelta):
                    yield ctx.artifact_chunk(artifact_id, item.text, append=bool(answer))
                    round_text += item.text
                    answer += item.text
                else:
                    calls.append(item)
            if not calls:
                break
            # The assistant message with tool_calls must come before the tool results that answer it.
            messages.append({
                "role": "assistant",
                "content": round_text or None,
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                    for call in calls
                ]
            })
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": await self._run_tool(tools, call)})

        yield ctx.artifact_chunk(artifact_id, "", append=bool(answer), last_chunk=True)
        # Only the user's text and the final answer are remembered, not the tool traffic.
        await self.conversations.append(ctx.context_id, {"role": "user", "content": ctx.user_text},
                                        {"role": "assistant", "content": answer})
        yield ctx.status(TaskState.COMPLETED)

    async def _run_tool(self, tools: dict[str, Tool], call: ToolCall) -> str:
        tool = tools.get(call.name)
        if tool is None:
            return f"Unknown tool {call.name!r}."
        try:
            return await tool.call(**json.loads(call.arguments or "{}"))
        except Exception as exc:  # the model sees the failure and can recover or apologise
            log.exception("tool %s failed", call.name)
            return f"Tool {call.name} failed: {exc}"
