# Implementation guide

You type the source code from this guide, one file at a time. Each file already exists in `src/` as a
one-line stub (just its docstring). Replace the stub with the full file shown here.

The sections go bottom-up. After each one there is a **checkpoint**: run the tests for that layer and they
should pass before you move on. Each test folder only imports its own layer and the layers below it, so
the checkpoints work even while later files are still stubs.

Run a checkpoint either locally or in Docker:

```bash
uv run pytest tests/protocol                                     # local
docker compose --profile test run --rm tests tests/protocol      # in Docker (src/ and tests/ are mounted)
```

Running plain `pytest` before you finish shows collection errors for the layers you haven't typed yet. That's expected.

The `__init__.py` files only hold a docstring and are already complete.

```
user ──A2A──▶ sprout:8001 ──A2A──▶ information-agent:8002 ──MCP──▶ mcp-server:8004 ──▶ weather/news/search/trains/flights
OpenAI SDK ─▶ gateway:8080 ─A2A─▶ sprout        └──A2A──▶ coding-agent:8003
```

Inside one agent, a request goes `server.py` (HTTP) → `RequestHandler` (JSON-RPC, task lifecycle, Redis) →
`AgentExecutor.execute()` (yields events) → `LLMAgent` tool loop → `Tool`s.

---

## 1. Config

Every setting comes from an environment variable with the same name in upper case (`LLM_MODEL` → `llm_model`).
`pydantic-settings` does the reading. `REMOTE_AGENT_URLS` is a comma-separated string. `NoDecode` stops
pydantic-settings from trying to parse it as JSON, and the validator splits it instead.

### `src/a2a_system/config.py`

```python
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
    agent_public_url: str = "http://localhost:8000"  # URL written into this agent's card
    remote_agent_urls: Annotated[list[str], NoDecode] = []  # sprout: comma-separated agent base URLs
    mcp_server_url: str = "http://localhost:8004/mcp"  # information agent
    sprout_url: str = "http://localhost:8001"  # gateway

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
```

**Checkpoint:** `pytest tests/test_config.py`

---

## 2. Protocol models and errors

These are the A2A v1.0 data types. Some things to notice:

- **camelCase on the wire, snake_case in Python.** `alias_generator=to_camel` gives every field a camelCase
  alias. `validate_by_name` also accepts the Python names, and `serialize_by_alias` makes dumps camelCase.
  `dump()` is the one call used everywhere to get JSON-ready dicts.
- **No `kind` field.** v1.0 dropped the discriminator. A `Part` is "whichever field is set", so a validator
  enforces exactly one of `text | raw | url | data`. `StreamResponse` works the same way.
- **Enums are SCREAMING_CASE strings** (`TASK_STATE_WORKING`, `ROLE_USER`). `StrEnum` values compare equal
  to plain strings.
- `task_text()` is shared by the remote-agent tool and the gateway. It returns the answer, or the error
  message when the task failed.

### `src/a2a_system/protocol/models.py`

```python
"""A2A v1.0 data model. JSON uses camelCase, Python uses snake_case."""

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
    def _exactly_one_content(self) -> "Part":
        if sum(value is not None for value in (self.text, self.raw, self.url, self.data)) != 1:
            raise ValueError("a Part needs exactly one of text, raw, url, data")
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
    def _exactly_one_payload(self) -> "StreamResponse":
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
```

Errors are exceptions that know their JSON-RPC code. The server turns them into `{"error": {...}}`.
The client turns them back into the same exception class with `from_dict`.

### `src/a2a_system/protocol/errors.py`

```python
"""JSON-RPC errors: the standard codes plus the A2A ones this project uses."""

from typing import Any


class JsonRpcError(Exception):
    code = -32603
    message = "Internal error"

    def __init__(self, message: str | None = None, data: Any = None):
        self.message = message or self.message
        self.data = data
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        error = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error

    @classmethod
    def from_dict(cls, error: dict[str, Any]) -> "JsonRpcError":
        """Rebuild the matching subclass from an error object received over the wire."""
        by_code = {sub.code: sub for sub in JsonRpcError.__subclasses__()}
        return by_code.get(error.get("code"), JsonRpcError)(error.get("message"), error.get("data"))


class ParseError(JsonRpcError):
    code, message = -32700, "Parse error"


class InvalidRequestError(JsonRpcError):
    code, message = -32600, "Invalid request"


class MethodNotFoundError(JsonRpcError):
    code, message = -32601, "Method not found"


class InvalidParamsError(JsonRpcError):
    code, message = -32602, "Invalid params"


class InternalError(JsonRpcError):
    code, message = -32603, "Internal error"


class TaskNotFoundError(JsonRpcError):
    code, message = -32001, "Task not found"


class TaskNotCancelableError(JsonRpcError):
    code, message = -32002, "Task cannot be canceled"


class UnsupportedOperationError(JsonRpcError):
    code, message = -32004, "This operation is not supported"


class InvalidAgentResponseError(JsonRpcError):
    code, message = -32006, "Invalid agent response"


class VersionNotSupportedError(JsonRpcError):
    code, message = -32009, "A2A version not supported (send header A2A-Version: 1.0)"
```

**Checkpoint:** `pytest tests/protocol/test_models.py tests/protocol/test_errors.py`

---

## 3. Storage

Two small Redis wrappers. Every key is prefixed with the agent's name, so all agents can share one Redis.

- `TaskStore` keeps the whole `Task` as JSON at `"{agent}:task:{id}"`.
- `ConversationStore` keeps OpenAI-style chat messages in a Redis list at `"{agent}:conversation:{contextId}"`.
  `ltrim` keeps only the latest `max_messages`, so the LLM context can't grow without limit.

### `src/a2a_system/storage.py`

```python
"""Redis persistence: A2A tasks and per-context conversation memory."""

import json

from redis.asyncio import Redis

from a2a_system.protocol.models import Task

WEEK = 7 * 24 * 3600


class TaskStore:
    def __init__(self, redis: Redis, agent: str, ttl: int = WEEK):
        self.redis, self.agent, self.ttl = redis, agent, ttl

    def _key(self, task_id: str) -> str:
        return f"{self.agent}:task:{task_id}"

    async def get(self, task_id: str) -> Task | None:
        raw = await self.redis.get(self._key(task_id))
        return Task.model_validate_json(raw) if raw else None

    async def save(self, task: Task) -> None:
        await self.redis.set(self._key(task.id), task.model_dump_json(exclude_none=True), ex=self.ttl)


class ConversationStore:
    """Chat history as OpenAI-style messages ({"role": ..., "content": ...}), one list per contextId."""

    def __init__(self, redis: Redis, agent: str, ttl: int = WEEK, max_messages: int = 40):
        self.redis, self.agent, self.ttl, self.max_messages = redis, agent, ttl, max_messages

    def _key(self, context_id: str) -> str:
        return f"{self.agent}:conversation:{context_id}"

    async def history(self, context_id: str) -> list[dict]:
        return [json.loads(item) for item in await self.redis.lrange(self._key(context_id), 0, -1)]

    async def append(self, context_id: str, *messages: dict) -> None:
        key = self._key(context_id)
        await self.redis.rpush(key, *(json.dumps(message) for message in messages))
        await self.redis.ltrim(key, -self.max_messages, -1)
        await self.redis.expire(key, self.ttl)
```

**Checkpoint:** `pytest tests/storage`

---

## 4. Protocol runtime: executor, handler, server, client

This mirrors the layering of the official SDK, without its event-queue machinery:

- **`AgentExecutor`** is the only thing an agent implements. `execute(ctx)` is an *async generator* that
  yields status and artifact events. `RequestContext` has helpers to build those events.
- **`RequestHandler`** owns the task lifecycle. `_run()` creates the task, saves it, runs the executor and
  saves every event before passing it on. `SendMessage` drains `_run()` and returns the final task.
  `SendStreamingMessage` turns each item into a `StreamResponse`.
  - Before every save it **re-reads** the task. If someone called `CancelTask` meanwhile, the run stops and
    emits a `CANCELED` event. `aclosing()` then closes the executor's generator.
  - If the executor raises, the task becomes `FAILED` with the error text, so every stream ends with a
    terminal event.
- **`build_app`** is the HTTP side: the Agent Card route, and one JSON-RPC `POST /`. It checks the
  `A2A-Version: 1.0` header (a missing header means v0.3, so the server rejects it with `-32009`). Streaming
  responses are Server-Sent Events, one `data: <json-rpc response>` per event.
- **`A2AClient`** is the other end of the same protocol. It takes an `httpx2.AsyncClient`, so tests can plug
  in an in-memory ASGI transport.

> `httpx2` is the HTTP client that `openai` 3.x and `mcp` 2.x are built on. This project uses it too, so
> there's only one HTTP library. Its API is the same as `httpx`.

### `src/a2a_system/protocol/executor.py`

```python
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
            message = Message(message_id=uuid4().hex, role=Role.AGENT, parts=[Part(text=text)],
                              context_id=self.context_id, task_id=self.task_id)
        return TaskStatusUpdateEvent(task_id=self.task_id, context_id=self.context_id,
                                     status=TaskStatus(state=state, message=message))

    def artifact_chunk(self, artifact_id: str, text: str, *, append: bool,
                       last_chunk: bool = False) -> TaskArtifactUpdateEvent:
        return TaskArtifactUpdateEvent(task_id=self.task_id, context_id=self.context_id,
                                       artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
                                       append=append, last_chunk=last_chunk)


class AgentExecutor(ABC):
    @abstractmethod
    def execute(self, ctx: RequestContext) -> AsyncIterator[Event]:
        """Implement as `async def execute(...)` that yields events and ends with a terminal status."""
```

### `src/a2a_system/protocol/request_handler.py`

```python
"""JSON-RPC method dispatch and the task lifecycle: create, run the executor, persist every event."""

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from a2a_system.protocol.errors import (
    InvalidParamsError,
    MethodNotFoundError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from a2a_system.protocol.executor import AgentExecutor, RequestContext
from a2a_system.protocol.models import (
    A2AModel,
    Event,
    Message,
    SendMessageParams,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a_system.storage import TaskStore

log = logging.getLogger(__name__)

UNSUPPORTED_METHODS = {
    "ListTasks",
    "SubscribeToTask",
    "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig",
    "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig",
    "GetExtendedAgentCard",
}

def _parse[M: A2AModel](model: type[M], params: Any) -> M:
    try:
        return model.model_validate(params)
    except ValidationError as exc:
        raise InvalidParamsError(str(exc)) from exc


class RequestHandler:
    def __init__(self, executor: AgentExecutor, tasks: TaskStore):
        self.executor, self.tasks = executor, tasks

    async def call(self, method: str, params: Any) -> dict[str, Any]:
        """Run a unary JSON-RPC method and return its result object."""
        if method == "SendMessage":
            return {"task": (await self.send_message(_parse(SendMessageParams, params).message)).dump()}
        if method == "GetTask":
            return (await self.get_task(_parse(TaskIdParams, params).id)).dump()
        if method == "CancelTask":
            return (await self.cancel_task(_parse(TaskIdParams, params).id)).dump()
        if method in UNSUPPORTED_METHODS:
            raise UnsupportedOperationError(f"{method} is not supported by this agent")
        raise MethodNotFoundError(f"Unknown method {method!r}")

    def stream(self, params: Any) -> AsyncIterator[dict[str, Any]]:
        """SendStreamingMessage. Params are checked now, so a bad request fails before the stream starts."""
        message = _parse(SendMessageParams, params).message
        return self._stream(message)

    async def _stream(self, message: Message) -> AsyncIterator[dict[str, Any]]:
        async with aclosing(self._run(message)) as items:
            async for item in items:
                if isinstance(item, Task):
                    yield StreamResponse(task=item).dump()
                elif isinstance(item, TaskStatusUpdateEvent):
                    yield StreamResponse(status_update=item).dump()
                else:
                    yield StreamResponse(artifact_update=item).dump()

    async def send_message(self, message: Message) -> Task:
        async with aclosing(self._run(message)) as items:
            task = await anext(items)
            async for _ in items:  # drain: the executor runs to the end
                pass
        return await self.get_task(task.id)

    async def get_task(self, task_id: str) -> Task:
        task = await self.tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"No task {task_id!r}")
        return task

    async def cancel_task(self, task_id: str) -> Task:
        task = await self.get_task(task_id)
        if task.status.state.is_terminal:
            raise TaskNotCancelableError(f"Task is already {task.status.state}")
        task.status = TaskStatus(state=TaskState.CANCELED)
        await self.tasks.save(task)
        return task

    async def _run(self, message: Message) -> AsyncIterator[Task | Event]:
        """Yield the new Task, then each executor event after it has been saved."""
        # Every message starts a new task; an incoming taskId is ignored (no multi-turn tasks).
        task = Task(id=uuid4().hex, context_id=message.context_id or uuid4().hex,
                    status=TaskStatus(state=TaskState.SUBMITTED))
        message = message.model_copy(update={"task_id": task.id, "context_id": task.context_id})
        task.history = [message]
        await self.tasks.save(task)
        yield task

        ctx = RequestContext(task_id=task.id, context_id=task.context_id, message=message)
        try:
            # ponytail: cancel takes effect at the executor's next event, an in-flight LLM/tool call finishes first
            async with aclosing(self.executor.execute(ctx)) as events:
                async for event in events:
                    if not await self._apply(event):
                        yield ctx.status(TaskState.CANCELED)
                        return
                    yield event
        except Exception as exc:
            log.exception("executor failed for task %s", task.id)
            failed = ctx.status(TaskState.FAILED, f"Agent error: {exc}")
            if await self._apply(failed):
                yield failed

    async def _apply(self, event: Event) -> bool:
        """Apply an event to the stored task. Returns False if the task was canceled meanwhile."""
        # Re-read every time: CancelTask may have changed the task since our last save.
        # ponytail: read-modify-write without a lock, a cancel landing mid-save can be lost; use WATCH/MULTI if it matters
        task = await self.tasks.get(event.task_id)
        if task is None or task.status.state is TaskState.CANCELED:
            return False
        if isinstance(event, TaskStatusUpdateEvent):
            task.status = event.status
        else:
            _merge_artifact(task, event)
        await self.tasks.save(task)
        return True


def _merge_artifact(task: Task, event: TaskArtifactUpdateEvent) -> None:
    for i, existing in enumerate(task.artifacts):
        if existing.artifact_id == event.artifact.artifact_id:
            if event.append:
                existing.parts.extend(event.artifact.parts)
            else:
                task.artifacts[i] = event.artifact
            return
    task.artifacts.append(event.artifact)
```

### `src/a2a_system/protocol/server.py`

```python
"""FastAPI app for one A2A agent: the Agent Card route and the JSON-RPC endpoint (with SSE streaming)."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from a2a_system.protocol.errors import (
    InternalError,
    InvalidRequestError,
    JsonRpcError,
    ParseError,
    VersionNotSupportedError,
)
from a2a_system.protocol.models import AgentCard
from a2a_system.protocol.request_handler import RequestHandler

log = logging.getLogger(__name__)

SUPPORTED_VERSIONS = {"1.0"}


def _result(rpc_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _error(rpc_id: Any, error: JsonRpcError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error.to_dict()}


async def _sse(rpc_id: Any, events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield f"data: {json.dumps(_result(rpc_id, event))}\n\n"
    except Exception:
        log.exception("stream failed")
        yield f"data: {json.dumps(_error(rpc_id, InternalError()))}\n\n"


def build_app(card: AgentCard, handler: RequestHandler) -> FastAPI:
    app = FastAPI(title=card.name, description=card.description, version=card.version)

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict[str, Any]:
        return card.dump()

    @app.post("/")
    async def jsonrpc(request: Request):
        rpc_id = None
        try:
            try:
                body = await request.json()
            except ValueError:
                raise ParseError() from None
            if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
                raise InvalidRequestError()
            rpc_id = body.get("id")
            # A missing header means the client speaks 0.3, which this server does not.
            if request.headers.get("A2A-Version") not in SUPPORTED_VERSIONS:
                raise VersionNotSupportedError()
            params = body.get("params") or {}
            if body["method"] == "SendStreamingMessage":
                events = handler.stream(params)
                return StreamingResponse(_sse(rpc_id, events), media_type="text/event-stream")
            return _result(rpc_id, await handler.call(body["method"], params))
        except JsonRpcError as exc:
            return JSONResponse(_error(rpc_id, exc))
        except Exception:
            log.exception("request failed")
            return JSONResponse(_error(rpc_id, InternalError()))

    return app
```

### `src/a2a_system/protocol/client.py`

```python
"""A2A client: fetch an Agent Card, send a message, or stream one."""

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx2

from a2a_system.protocol.errors import InvalidAgentResponseError, JsonRpcError
from a2a_system.protocol.models import AgentCard, Message, StreamResponse, Task

HEADERS = {"A2A-Version": "1.0"}


def _request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": uuid4().hex, "method": method, "params": params}


def _result(body: Any) -> Any:
    if not isinstance(body, dict):
        raise InvalidAgentResponseError("Response is not a JSON-RPC object")
    if "error" in body:
        raise JsonRpcError.from_dict(body["error"])
    if "result" not in body:
        raise InvalidAgentResponseError("Response has neither result nor error")
    return body["result"]


class A2AClient:
    """`url` is an agent's base URL, e.g. http://information-agent:8000."""

    def __init__(self, http: httpx2.AsyncClient | None = None):
        self.http = http or httpx2.AsyncClient(timeout=120)

    async def get_agent_card(self, url: str) -> AgentCard:
        response = await self.http.get(f"{url.rstrip('/')}/.well-known/agent-card.json")
        response.raise_for_status()
        return AgentCard.model_validate(response.json())

    async def send_message(self, url: str, message: Message) -> Task:
        response = await self.http.post(url, json=_request("SendMessage", {"message": message.dump()}), headers=HEADERS)
        response.raise_for_status()
        result = _result(response.json())
        if "task" not in result:
            raise InvalidAgentResponseError("Expected a task in the SendMessage result")
        return Task.model_validate(result["task"])

    async def stream_message(self, url: str, message: Message) -> AsyncIterator[StreamResponse]:
        request = _request("SendStreamingMessage", {"message": message.dump()})
        async with self.http.stream("POST", url, json=request, headers=HEADERS) as response:
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                _result(json.loads(await response.aread()))  # a JSON-RPC error instead of a stream
                raise InvalidAgentResponseError("Expected an event stream")
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    yield StreamResponse.model_validate(_result(json.loads(line[5:])))
```

**Checkpoint:** `pytest tests/protocol`. The tests run an "echo" agent (see `tests/fakes.py`) through your
handler and server, then talk to it with your client.

---

## 5. LLM client

A thin wrapper over the `openai` SDK pointed at any OpenAI-compatible `base_url`. It always streams.
Text is yielded as `TextDelta`s as soon as it arrives. Tool calls arrive in fragments: the first fragment
has the `id` and `name`, and later ones add pieces of the `arguments` JSON. Fragments are joined by `index`,
and the complete `ToolCall`s are yielded at the end. Chunks with no `choices` (usage-only chunks from some
providers) are skipped.

### `src/a2a_system/llm/client.py`

```python
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
    arguments: str  # JSON text, exactly as the model produced it


class LLMClient:
    def __init__(self, openai: AsyncOpenAI, model: str):
        self.openai, self.model = openai, model

    async def stream(self, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None) -> AsyncIterator[TextDelta | ToolCall]:
        """Yield text as it arrives, then any complete tool calls once the response ends."""
        extra = {"tools": tools} if tools else {}
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
                call = calls.setdefault(piece.index or 0, ToolCall(id="", name="", arguments=""))
                call.id = piece.id or call.id
                if piece.function:
                    call.name += piece.function.name or ""
                    call.arguments += piece.function.arguments or ""
        for index in sorted(calls):
            yield calls[index]
```

**Checkpoint:** `pytest tests/llm`

---

## 6. Tools

`Tool` is the shape the LLM sees: a name, a description and a JSON schema for the arguments. `schema()`
renders it as an OpenAI function definition. It has two implementations:

- **`McpTool`** calls one tool on the MCP server. `load_mcp_tools()` lists the server's tools and wraps each
  one. `mcp.Client` accepts a URL (Streamable HTTP) or a server object (in-memory, used by the tests).
- **`RemoteAgentTool`** is how Sprout delegates. Each remote Agent Card becomes a tool (`ask_information_agent`,
  `ask_coding_agent`) with a single `query` parameter. The description comes from the card's skills. Calling
  it sends an A2A message and returns the answer text. No `contextId` is sent, so the remote agent is
  stateless and the query has to stand on its own.

### `src/a2a_system/tools/base.py`

```python
"""A Tool is anything the LLM can call: a name, a description and a JSON-schema for its arguments."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object

    @abstractmethod
    async def call(self, **arguments: Any) -> str: ...

    def schema(self) -> dict[str, Any]:
        """The OpenAI function-calling definition of this tool."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
```

### `src/a2a_system/tools/mcp_tool.py`

```python
"""MCP server tools wrapped as Tools."""

from typing import Any

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from a2a_system.tools.base import Tool

McpTarget = str | MCPServer  # a Streamable HTTP URL, or a server object (in-memory, used by tests)


class McpTool(Tool):
    def __init__(self, server: McpTarget, name: str, description: str, parameters: dict[str, Any]):
        self.server, self.name, self.description, self.parameters = server, name, description, parameters

    async def call(self, **arguments: Any) -> str:
        # ponytail: one MCP session per call, keep a session open if tool latency matters
        async with Client(self.server) as client:
            result = await client.call_tool(self.name, arguments)
        text = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
        return f"Tool error: {text}" if result.is_error else text


async def load_mcp_tools(server: McpTarget) -> list[McpTool]:
    async with Client(server) as client:
        listed = await client.list_tools()
    return [McpTool(server, tool.name, tool.description or "", tool.input_schema) for tool in listed.tools]
```

### `src/a2a_system/tools/remote_agent_tool.py`

```python
"""A remote A2A agent wrapped as a Tool, so Sprout's LLM can delegate by function calling."""

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
            "properties": {"query": {"type": "string",
                                     "description": "A complete, self-contained request for this agent."}},
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
```

**Checkpoint:** `pytest tests/tools`

---

## 7. Agents

`LLMAgent` is the shared `AgentExecutor`:

1. It loads history for the `contextId` and builds `system + history + user` messages.
2. It runs the **tool loop**: stream the LLM response, and if it asked for tools, append the assistant
   message *with its `tool_calls`*, then one `tool` message per result, and go again. After 5 rounds no tools
   are offered, which forces a text answer.
3. It streams every text delta as an artifact chunk. All chunks share one `artifactId`, the first has
   `append=false` and the rest `append=true`, and a final empty chunk carries `lastChunk=true`.
4. It saves only the user text and the final answer to memory, never the tool traffic.

A tool failure doesn't fail the task. The error text goes back to the LLM so it can recover or apologise.

The three agents are mostly data: a name, skills (these become the Agent Card), a system prompt and their tools.
Sprout and the Information agent load their tools lazily on the first request and cache what succeeded.
A remote agent or MCP server that is down is skipped and retried next time, so there's no startup ordering.

### `src/a2a_system/agents/base.py`

```python
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
            messages.append({"role": "assistant", "content": round_text or None, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in calls]})
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
```

### `src/a2a_system/agents/sprout.py`

```python
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
```

### `src/a2a_system/agents/information.py`

```python
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
```

### `src/a2a_system/agents/coding.py`

```python
"""Coding agent: reasons about and writes code. It never runs anything."""

from a2a_system.agents.base import LLMAgent
from a2a_system.protocol.models import AgentSkill


class CodingAgent(LLMAgent):
    name = "Coding Agent"
    description = "Writes, explains, reviews and debugs code. It reasons only and cannot run code."
    skills = [AgentSkill(id="coding", name="Coding",
                         description="Write, explain, review or debug code in any language.",
                         tags=["code", "programming"], examples=["Write a Python function that reverses a list"])]
    system_prompt = (
        "You are an expert software engineer. Think through the problem step by step, then give correct, "
        "idiomatic code in fenced code blocks with a short explanation. You cannot execute code, so say "
        "which parts are untested assumptions."
    )
```

**Checkpoint:** `pytest tests/agents`

---

## 8. MCP server

Each external API is one small provider class that takes an `httpx2.AsyncClient` (tests pass a
`MockTransport`). Providers return plain text the LLM can read. When an API key is missing, they say so
instead of failing.

- Weather: Open-Meteo (no key). Geocode the city, then fetch the forecast.
- News: Google News RSS (no key), parsed with the stdlib `xml.etree`.
- Search: Tavily if `TAVILY_API_KEY` is set, otherwise DuckDuckGo via `ddgs`. `ddgs` is synchronous, so it
  runs in a thread.
- Trains: RapidAPI `irctc1`. **Check the endpoint path, params and field names in your RapidAPI dashboard.**
  They were written from memory, and everything that would change is in this one file.
- Flights: AviationStack. The free tier only allows plain HTTP.

### `src/a2a_system/mcp_server/providers/weather.py`

```python
"""Weather from Open-Meteo (free, no API key)."""

import httpx2

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes, grouped: (highest code in the group, description)
WMO_GROUPS = [(0, "clear sky"), (3, "partly cloudy"), (48, "fog"), (57, "drizzle"), (67, "rain"),
              (77, "snow"), (82, "rain showers"), (86, "snow showers"), (99, "thunderstorm")]


def describe(code: int) -> str:
    return next((text for limit, text in WMO_GROUPS if code <= limit), "unknown")


class WeatherProvider:
    def __init__(self, http: httpx2.AsyncClient):
        self.http = http

    async def forecast(self, city: str, days: int = 3) -> str:
        response = await self.http.get(GEOCODE_URL, params={"name": city, "count": 1})
        response.raise_for_status()
        places = response.json().get("results")
        if not places:
            return f"No place called {city!r} was found."
        place = places[0]

        response = await self.http.get(FORECAST_URL, params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 7)),
        })
        response.raise_for_status()
        data = response.json()
        now, daily = data["current"], data["daily"]
        current = (f"{place['name']}, {place.get('country', '')}: now {now['temperature_2m']}°C, "
                   f"{describe(now['weather_code'])}, wind {now['wind_speed_10m']} km/h")
        lines = [current]
        for day, code, high, low, rain in zip(daily["time"], daily["weather_code"], daily["temperature_2m_max"],
                                              daily["temperature_2m_min"], daily["precipitation_probability_max"]):
            lines.append(f"{day}: {low}–{high}°C, {describe(code)}, rain chance {rain}%")
        return "\n".join(lines)
```

### `src/a2a_system/mcp_server/providers/news.py`

```python
"""Headlines from Google News RSS (free, no API key)."""

import xml.etree.ElementTree as ET

import httpx2

TOP_URL = "https://news.google.com/rss"
SEARCH_URL = "https://news.google.com/rss/search"
LOCALE = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}


class NewsProvider:
    def __init__(self, http: httpx2.AsyncClient):
        self.http = http

    async def headlines(self, topic: str = "", limit: int = 5) -> str:
        if topic:
            response = await self.http.get(SEARCH_URL, params={"q": topic, **LOCALE})
        else:
            response = await self.http.get(TOP_URL, params=LOCALE)
        response.raise_for_status()
        items = ET.fromstring(response.text).findall("./channel/item")[:limit]
        if not items:
            return f"No news found for {topic!r}."
        return "\n".join(f"- {item.findtext('title')} ({item.findtext('pubDate')})\n  {item.findtext('link')}"
                         for item in items)
```

### `src/a2a_system/mcp_server/providers/search.py`

```python
"""Web search: Tavily when TAVILY_API_KEY is set, otherwise DuckDuckGo (no key needed)."""

import asyncio

import httpx2
from ddgs import DDGS

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchProvider:
    def __init__(self, http: httpx2.AsyncClient, tavily_api_key: str = ""):
        self.http, self.tavily_api_key = http, tavily_api_key

    @property
    def backend(self) -> str:
        return "tavily" if self.tavily_api_key else "duckduckgo"

    async def search(self, query: str, max_results: int = 5) -> str:
        if self.tavily_api_key:
            results = await self._tavily(query, max_results)
        else:
            results = await self._duckduckgo(query, max_results)
        if not results:
            return f"No results for {query!r}."
        return "\n".join(f"- {title}\n  {url}\n  {snippet}" for title, url, snippet in results)

    async def _tavily(self, query: str, max_results: int) -> list[tuple[str, str, str]]:
        response = await self.http.post(TAVILY_URL, json={"query": query, "max_results": max_results},
                                        headers={"Authorization": f"Bearer {self.tavily_api_key}"})
        response.raise_for_status()
        return [(r["title"], r["url"], r.get("content", "")) for r in response.json().get("results", [])]

    async def _duckduckgo(self, query: str, max_results: int) -> list[tuple[str, str, str]]:
        # ddgs is synchronous, so run it in a worker thread to keep the event loop free.
        results = await asyncio.to_thread(lambda: DDGS().text(query, max_results=max_results))
        return [(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in results]
```

### `src/a2a_system/mcp_server/providers/trains.py`

```python
"""Indian Railways trains between two stations, via the RapidAPI "IRCTC" API (irctc1).

Endpoint path, params and response fields are from memory: check them in the RapidAPI dashboard.
"""

import httpx2

HOST = "irctc1.p.rapidapi.com"
URL = f"https://{HOST}/api/v3/trainBetweenStations"


class TrainProvider:
    def __init__(self, http: httpx2.AsyncClient, rapidapi_key: str = ""):
        self.http, self.rapidapi_key = http, rapidapi_key

    async def between_stations(self, from_station: str, to_station: str, date: str) -> str:
        if not self.rapidapi_key:
            return "Train search is not configured (RAPIDAPI_KEY is missing)."
        response = await self.http.get(
            URL,
            params={"fromStationCode": from_station.upper(), "toStationCode": to_station.upper(),
                    "dateOfJourney": date},
            headers={"X-RapidAPI-Key": self.rapidapi_key, "X-RapidAPI-Host": HOST},
        )
        response.raise_for_status()
        body = response.json()
        trains = body.get("data") or []
        if not body.get("status") or not trains:
            return f"No trains found from {from_station} to {to_station} on {date}. {body.get('message', '')}".strip()
        return "\n".join(
            f"- {t.get('train_number')} {t.get('train_name')}: departs {t.get('from_std')}, "
            f"arrives {t.get('to_sta')}, duration {t.get('duration')}"
            for t in trains[:15]
        )
```

### `src/a2a_system/mcp_server/providers/flights.py`

```python
"""Flights from AviationStack. The free tier only allows plain HTTP."""

import httpx2

URL = "http://api.aviationstack.com/v1/flights"


class FlightProvider:
    def __init__(self, http: httpx2.AsyncClient, api_key: str = ""):
        self.http, self.api_key = http, api_key

    async def search(self, departure_iata: str = "", arrival_iata: str = "", flight_number: str = "") -> str:
        if not self.api_key:
            return "Flight search is not configured (AVIATIONSTACK_API_KEY is missing)."
        filters = {"dep_iata": departure_iata.upper(), "arr_iata": arrival_iata.upper(),
                   "flight_iata": flight_number.upper().replace(" ", "")}
        params = {"access_key": self.api_key, "limit": 10, **{k: v for k, v in filters.items() if v}}
        response = await self.http.get(URL, params=params)
        body = response.json()
        if "error" in body:
            return f"Flight search failed: {body['error'].get('message', body['error'])}"
        flights = body.get("data") or []
        if not flights:
            return "No flights found."
        return "\n".join(
            f"- {f['flight'].get('iata')} ({f['airline'].get('name')}) {f.get('flight_status')}: "
            f"{f['departure'].get('iata')} {f['departure'].get('scheduled')} -> "
            f"{f['arrival'].get('iata')} {f['arrival'].get('scheduled')}"
            for f in flights
        )
```

`build_server` registers one MCP tool per provider method. The MCP SDK builds each tool's input schema from
the type hints and its description from the docstring, so those are worth writing well. The LLM reads them.

### `src/a2a_system/mcp_server/server.py`

```python
"""The MCP server: each tool is a thin wrapper around a provider."""

import httpx2
from mcp.server.mcpserver import MCPServer

from a2a_system.config import Settings
from a2a_system.mcp_server.providers.flights import FlightProvider
from a2a_system.mcp_server.providers.news import NewsProvider
from a2a_system.mcp_server.providers.search import WebSearchProvider
from a2a_system.mcp_server.providers.trains import TrainProvider
from a2a_system.mcp_server.providers.weather import WeatherProvider


def build_server(http: httpx2.AsyncClient, settings: Settings) -> MCPServer:
    weather = WeatherProvider(http)
    news = NewsProvider(http)
    search = WebSearchProvider(http, settings.tavily_api_key)
    trains = TrainProvider(http, settings.rapidapi_key)
    flights = FlightProvider(http, settings.aviationstack_api_key)

    server = MCPServer("information-tools", instructions="Live weather, news, web search, Indian trains, flights.")

    @server.tool()
    async def get_weather(city: str, days: int = 3) -> str:
        """Current weather and a daily forecast (1-7 days) for a city."""
        return await weather.forecast(city, days)

    @server.tool()
    async def get_news(topic: str = "", limit: int = 5) -> str:
        """Latest news headlines with links. Leave topic empty for top stories."""
        return await news.headlines(topic, limit)

    @server.tool()
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web. Returns titles, links and snippets."""
        return await search.search(query, max_results)

    @server.tool()
    async def search_trains(from_station: str, to_station: str, date: str) -> str:
        """Indian Railways trains between two station codes (e.g. NDLS, BCT) on a date (YYYY-MM-DD)."""
        return await trains.between_stations(from_station, to_station, date)

    @server.tool()
    async def search_flights(departure_iata: str = "", arrival_iata: str = "", flight_number: str = "") -> str:
        """Live and scheduled flights filtered by airport IATA codes (e.g. DEL, BOM) and/or flight number (e.g. AI101)."""
        return await flights.search(departure_iata, arrival_iata, flight_number)

    return server
```

**Checkpoint:** `pytest tests/mcp_server`

---

## 9. OpenAI-compatible gateway

This lets any OpenAI client talk to Sprout. It is just an A2A client:

- If the request has `user`, that becomes the A2A `contextId` and only the newest user message is sent,
  because Sprout's Redis memory already has the rest. Without `user`, the whole transcript goes as one
  message to a fresh context.
- Streaming maps each A2A `artifactUpdate` to a `chat.completion.chunk`, ends with a `finish_reason: "stop"`
  chunk, then sends `data: [DONE]`. Responses are built with the `openai` package's own types, so their shape is right.

### `src/a2a_system/gateway/models.py`

```python
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
```

### `src/a2a_system/gateway/app.py`

```python
"""OpenAI-compatible API (/v1/models, /v1/chat/completions) that forwards to Sprout over A2A."""

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from a2a_system.gateway.models import ChatCompletionRequest
from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.models import Message, Part, Role, TaskState, get_text, task_text

MODEL_ID = "sprout"


def to_a2a_message(request: ChatCompletionRequest) -> Message:
    if request.user:
        # Sprout keeps this conversation in Redis, so only the newest user turn is sent.
        text = next((m.text for m in reversed(request.messages) if m.role == "user"), "")
    else:
        # No conversation id: send the whole transcript as one message to a fresh context.
        text = "\n".join(f"{m.role}: {m.text}" for m in request.messages)
    return Message(message_id=uuid4().hex, role=Role.USER, parts=[Part(text=text)], context_id=request.user)


def build_gateway(a2a: A2AClient, sprout_url: str) -> FastAPI:
    # ponytail: no auth, it is only published on localhost by docker compose
    app = FastAPI(title="Sprout OpenAI gateway")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "a2a-system"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        message = to_a2a_message(request)
        completion_id = f"chatcmpl-{uuid4().hex}"
        if request.stream:
            return StreamingResponse(_stream(message, completion_id, request.model), media_type="text/event-stream")
        task = await a2a.send_message(sprout_url, message)
        return ChatCompletion.model_validate({
            "id": completion_id, "object": "chat.completion", "created": int(time.time()), "model": request.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": task_text(task)}}],
        }).model_dump(exclude_unset=True)

    async def _stream(message: Message, completion_id: str, model: str) -> AsyncIterator[str]:
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
            data = ChatCompletionChunk.model_validate({
                "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            })
            return f"data: {data.model_dump_json(exclude_unset=True)}\n\n"

        yield chunk({"role": "assistant"})
        async for event in a2a.stream_message(sprout_url, message):
            if event.artifact_update and (text := get_text(event.artifact_update.artifact.parts)):
                yield chunk({"content": text})
            elif event.status_update and event.status_update.status.state is TaskState.FAILED:
                if event.status_update.status.message:
                    yield chunk({"content": get_text(event.status_update.status.message.parts)})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return app
```

**Checkpoint:** `pytest tests/gateway`. One test drives your gateway with the real OpenAI SDK.

---

## 10. Entry point

`__main__.py` is the only place where real Redis, LLM and HTTP clients are created. Everything else receives
them through constructors, which is why the tests can swap in fakes.

The MCP server binds `0.0.0.0`. On a non-localhost host the MCP SDK leaves DNS-rebinding protection off,
which other containers need to reach it as `mcp-server:8000`. `stateless_http=True` fits our "one session
per tool call" client.

### `src/a2a_system/__main__.py`

```python
"""Entry point: `python -m a2a_system <sprout|information|coding|mcp-server|gateway>`.

This is the only place where real dependencies (Redis, the LLM, HTTP clients) are created and wired together.
"""

import asyncio
import logging
import sys

import httpx2
import uvicorn
from fastapi import FastAPI
from openai import AsyncOpenAI
from redis.asyncio import Redis

from a2a_system.agents.base import LLMAgent
from a2a_system.agents.coding import CodingAgent
from a2a_system.agents.information import InformationAgent
from a2a_system.agents.sprout import SproutAgent
from a2a_system.config import Settings
from a2a_system.gateway.app import build_gateway
from a2a_system.llm.client import LLMClient
from a2a_system.mcp_server.server import build_server
from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.request_handler import RequestHandler
from a2a_system.protocol.server import build_app
from a2a_system.storage import ConversationStore, TaskStore

SERVICES = ("sprout", "information", "coding", "mcp-server", "gateway")


def build_agent_app(service: str, settings: Settings) -> FastAPI:
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = sys.argv[1] if len(sys.argv) > 1 else ""
    if service not in SERVICES:
        sys.exit(f"usage: python -m a2a_system <{'|'.join(SERVICES)}>")
    settings = Settings()

    if service == "mcp-server":
        server = build_server(httpx2.AsyncClient(timeout=30, follow_redirects=True), settings)
        # Binding 0.0.0.0 (not localhost) also leaves DNS-rebinding protection off, so "mcp-server:8000" is accepted.
        asyncio.run(server.run_streamable_http_async(host="0.0.0.0", port=settings.port, stateless_http=True))
    elif service == "gateway":
        uvicorn.run(build_gateway(A2AClient(), settings.sprout_url), host="0.0.0.0", port=settings.port)
    else:
        uvicorn.run(build_agent_app(service, settings), host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
```

**Checkpoint:** `pytest`. The whole suite should now be green. Then run it for real:

```bash
cp .env.example .env        # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
docker compose up --build
```

See the README for curl and OpenAI SDK examples.
