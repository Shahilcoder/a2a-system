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

    # OpenAI streaming response format
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
