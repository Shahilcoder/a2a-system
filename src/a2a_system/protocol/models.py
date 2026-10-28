"""A2A v1.0 data model. JSON uses camelCase, Python uses snake_case."""

from __future__ import annotations
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class A2AModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True, serialize_by_alias=True)

    def dump(self) -> dict[str, Any]:
        """JSON-ready dict with camelCase keys and unset optional fields left out."""
        return self.model_dump(mode="json", exclude_none=True)


class TaskState(StrEnum):
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    REJECTED = "TASK_STATE_REJECTED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED)


class Role(StrEnum):
    USER = "ROLE_USER"
    AGENT = "ROLE_AGENT"


class Part(A2AModel):
    """One piece of content. Exactly one of text, raw (base64), url or data is set."""

    text: str | None = None
    raw: str | None = None
    url: str | None = None
    data: Any = None
    media_type: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _exactly_one_content(self) -> Part:
        if sum(value is not None for value in (self.text, self.raw, self.url, self.data)) != 1:
            raise ValueError("a Part needs exactly one of text, raw, url or data")
        return self


class Message(A2AModel):
    message_id: str
    role: Role
    parts: list[Part]
    context_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] | None = None


class TaskStatus(A2AModel):
    state: TaskState
    message: Message | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Artifact(A2AModel):
    artifact_id: str
    parts: list[Part]
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


class Task(A2AModel):
    id: str
    context_id: str
    status: TaskStatus
    artifacts: list[Artifact] = []
    history: list[Message] = []
    metadata: dict[str, Any] | None = None


class TaskStatusUpdateEvent(A2AModel):
    task_id: str
    context_id: str
    status: TaskStatus
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(A2AModel):
    task_id: str
    context_id: str
    artifact: Artifact
    append: bool = False
    last_chunk: bool = False
    metadata: dict[str, Any] | None = None


Event = TaskStatusUpdateEvent | TaskArtifactUpdateEvent


class StreamResponse(A2AModel):
    """One SSE event. Exactly one field is set."""

    task: Task | None = None
    message: Message | None = None
    status_update: TaskStatusUpdateEvent | None = None
    artifact_update: TaskArtifactUpdateEvent | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> StreamResponse:
        payloads = (self.task, self.message, self.status_update, self.artifact_update)

        if sum(value is not None for value in payloads) != 1:
            raise ValueError("a StreamResponse needs exactly one of task, message, statusUpdate, artifactUpdate")

        return self


# JSON-RPC params
class SendMessageParams(A2AModel):
    message: Message
    configuration: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TaskIdParams(A2AModel):
    id: str


# Agent Card (served at /.well-known/agent-card.json)
class AgentInterface(A2AModel):
    url: str
    protocol_binding: str = "JSONRPC"
    protocol_version: str = "1.0"


class AgentCapabilities(A2AModel):
    streaming: bool = True
    push_notifications: bool = False


class AgentSkill(A2AModel):
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str] | None = None


class AgentCard(A2AModel):
    name: str
    description: str
    supported_interfaces: list[AgentInterface]
    version: str
    capabilities: AgentCapabilities = AgentCapabilities()
    default_input_modes: list[str] = ["text/plain"]
    default_output_modes: list[str] = ["text/plain"]
    skills: list[AgentSkill]


def get_text(parts: list[Part]) -> str:
    """All text parts joined together (non-text parts are skipped)."""
    return "".join(part.text for part in parts if part.text)


def task_text(task: Task) -> str:
    """The task's answer: artifact text, or the status message (e.g. the error) when there is none."""
    text = "\n".join(get_text(artifact.parts) for artifact in task.artifacts)
    if not text and task.status.message:
        text = get_text(task.status.message.parts)
    return text
