"""Streaming chat completions with tool calls, over any OpenAI-compatible API."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str # JSON text, exactly as the model produce it


class LLMClient:
    def __init__(self, openai: AsyncOpenAI, model: str):
        self.openai, self.model = openai, model

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AsyncIterator[TextDelta | ToolCall]:
        """Yield text as it arrives, then any complete tool calls once the response ends."""
        extra = {"tools": tools} if tools else {}
        # **extra in a function call argument, unpacks the dictionary items into keyword arguments
        response = await self.openai.chat.completions.create(model=self.model, messages=messages, stream=True, **extra)
        calls: dict[int, ToolCall] = {}
        async for chunk in response:
            if not chunk.choices:  # some providers send usage-only chunks
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield TextDelta(delta.content)

            # A tool call arrives in pieces: id and name first, then the arguments a few characters at a time.
            for piece in delta.tool_calls or []:
                tool_call = calls.setdefault(piece.index or 0, ToolCall(id="", name="", arguments=""))
                tool_call.id = piece.id or tool_call.id

                if piece.function:
                    tool_call.name += piece.function.name or ""
                    tool_call.arguments += piece.function.arguments or ""

        for index in sorted(calls):
            yield calls[index]

