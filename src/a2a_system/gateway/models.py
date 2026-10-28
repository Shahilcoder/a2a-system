"""The OpenAI chat request we accept. Responses use the openai package's own types."""

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, list):  # [{"type": "text", "text": "..."}, ...]
            return "".join(part.get("text", "") for part in self.content)
        return self.content or ""


class ChatCompletionRequest(BaseModel):
    model: str = "sprout"
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None  # used as the A2A contextId, so Sprout remembers the conversation
