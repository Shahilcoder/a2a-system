"""The agent-side contract: an executor turns one incoming message into a stream of task events."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from a2a_system.protocol.models import (
    Artifact,
    Event,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    get_text,
)


@dataclass
class RequestContext:
    task_id: str
    context_id: str
    message: Message

    @property
    def user_text(self) -> str:
        return get_text(self.message.parts)

    def status(self, state: TaskState, text: str | None = None) -> TaskStatusUpdateEvent:
        message = None
        if text:
            message = Message(
                message_id=uuid4().hex,
                role=Role.AGENT,
                parts=[Part(text=text)],
                context_id=self.context_id,
                task_id=self.task_id,
            )
        return TaskStatusUpdateEvent(
            task_id=self.task_id,
            context_id=self.context_id,
            status=TaskStatus(state=state, message=message),
        )

    def artifact_chunk(self, artifact_id: str, text: str, *, append: bool, last_chunk: bool = False) -> TaskArtifactUpdateEvent:
        return TaskArtifactUpdateEvent(
            task_id=self.task_id,
            context_id=self.context_id,

            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=append,
            last_chunk=last_chunk,
        )


class AgentExecutor(ABC):
    @abstractmethod
    def execute(self, ctx: RequestContext) -> AsyncIterator[Event]:
        """Implement as `async def execute(...)` that yields events and ends with a terminal status."""
