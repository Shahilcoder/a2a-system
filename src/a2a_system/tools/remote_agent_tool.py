"""A remote A2A agent wrapped as a Tool, so Sprout's LLM can delegate by function calling."""

from __future__ import annotations
import re
from uuid import uuid4

from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.models import AgentCard, Message, Part, Role, TaskState, task_text
from a2a_system.tools.base import Tool


class RemoteAgentTool(Tool):
    def __init__(self, card: AgentCard, url: str, client: A2AClient):
        self.card, self.url, self.client = card, url, client
        self.name = "ask_" + re.sub(r"[^a-z0-9]+", "_", card.name.lower()).strip("_")
        skills = "; ".join(f"{skill.name}: {skill.description}" for skill in card.skills)
        self.description = f"{card.description} Skills: {skills}"
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A complete, self-contained request for this agent."
                }
            },
            "required": ["query"],
        }

    async def call(self, query: str) -> str:
        # No contextId: the remote agent sees each call fresh, so the query must stand on its own.
        message = Message(message_id=uuid4().hex, role=Role.USER, parts=[Part(text=query)])
        # ponytail: non-streaming delegation, stream the remote answer through if latency matters
        task = await self.client.send_message(self.url, message)
        text = task_text(task)
        if task.status.state is not TaskState.COMPLETED:
            return f"{self.card.name} ended with {task.status.state}: {text}"
        return text
