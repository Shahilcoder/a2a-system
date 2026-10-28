import json

import httpx2
import pytest
from openai import AsyncOpenAI

from a2a_system.gateway.app import build_gateway
from a2a_system.protocol.models import (
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


class FakeA2A:
    """Stands in for A2AClient: records messages and replies with a canned answer."""

    def __init__(self, answer=("Hello", " there!")):
        self.answer = answer
        self.sent = []

    async def send_message(self, url, message):
        self.sent.append((url, message))
        return Task(id="t", context_id="c", status=TaskStatus(state=TaskState.COMPLETED),
                    artifacts=[Artifact(artifact_id="a", parts=[Part(text="".join(self.answer))])])

    async def stream_message(self, url, message):
        self.sent.append((url, message))
        yield StreamResponse(task=Task(id="t", context_id="c", status=TaskStatus(state=TaskState.SUBMITTED)))
        for i, text in enumerate(self.answer):
            artifact = Artifact(artifact_id="a", parts=[Part(text=text)])
            yield StreamResponse(artifact_update=TaskArtifactUpdateEvent(task_id="t", context_id="c",
                                                                         artifact=artifact, append=i > 0))
        yield StreamResponse(status_update=TaskStatusUpdateEvent(task_id="t", context_id="c",
                                                                 status=TaskStatus(state=TaskState.COMPLETED)))


@pytest.fixture
def a2a():
    return FakeA2A()


@pytest.fixture
async def client(a2a):
    app = build_gateway(a2a, "http://sprout:8000")
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gw") as http:
        yield http


async def test_models(client):
    body = (await client.get("/v1/models")).json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "sprout"


async def test_non_streaming_completion(client, a2a):
    request = {"model": "sprout", "messages": [{"role": "user", "content": "hi"}]}
    body = (await client.post("/v1/chat/completions", json=request)).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello there!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    url, message = a2a.sent[0]
    assert url == "http://sprout:8000"
    assert message.context_id is None


async def test_streaming_completion(client):
    request = {"model": "sprout", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    response = await client.post("/v1/chat/completions", json=request)
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line[len("data: "):] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello there!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_user_field_becomes_context_and_only_last_turn_is_sent(client, a2a):
    request = {"model": "sprout", "user": "shahil-1", "messages": [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]}]}
    await client.post("/v1/chat/completions", json=request)
    _, message = a2a.sent[0]
    assert message.context_id == "shahil-1"
    assert message.parts[0].text == "second"


async def test_without_user_the_transcript_is_sent(client, a2a):
    request = {"model": "sprout", "messages": [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]}
    await client.post("/v1/chat/completions", json=request)
    _, message = a2a.sent[0]
    assert message.parts[0].text == "system: be brief\nuser: hi"


async def test_works_with_the_openai_sdk(a2a):
    app = build_gateway(a2a, "http://sprout:8000")
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
    openai = AsyncOpenAI(api_key="unused", base_url="http://gw/v1", http_client=http)

    completion = await openai.chat.completions.create(model="sprout", messages=[{"role": "user", "content": "hi"}])
    assert completion.choices[0].message.content == "Hello there!"

    stream = await openai.chat.completions.create(model="sprout", messages=[{"role": "user", "content": "hi"}],
                                                  stream=True)
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in stream]) == "Hello there!"
