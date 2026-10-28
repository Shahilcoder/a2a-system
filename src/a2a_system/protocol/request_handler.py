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

# Python's generic type parameter syntax
# Only part difficult for me to understand: https://www.python.org/dev/peps/pep-3109/
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

    def stream(self, params: Any):
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
        task = Task(id=uuid4().hex, context_id=message.context_id or uuid4().hex, status=TaskStatus(state=TaskState.SUBMITTED))
        message = message.model_copy(update={"task_id": task.id, "context_id": task.context_id})
        task.history = [message]
        await self.tasks.save(task)
        yield task

        ctx = RequestContext(task_id=task.id, context_id=task.context_id, message=message)

        try:
            # samajh nhi aya
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


# samajh nhi aya
def _merge_artifact(task: Task, event: TaskArtifactUpdateEvent) -> None:
    for i, existing in enumerate(task.artifacts):
        if existing.artifact_id == event.artifact.artifact_id:
            if event.append:
                existing.parts.extend(event.artifact.parts)
            else:
                task.artifacts[i] = event.artifact
            return
    task.artifacts.append(event.artifact)
