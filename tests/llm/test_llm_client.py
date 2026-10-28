import json

import httpx2
from openai import AsyncOpenAI

from a2a_system.llm.client import LLMClient, TextDelta, ToolCall


def sse(*chunks: dict) -> str:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    return "".join(f"data: {json.dumps({**base, **chunk})}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def delta(**fields) -> dict:
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def llm_returning(body: str, requests: list | None = None) -> LLMClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return LLMClient(AsyncOpenAI(api_key="test", base_url="http://llm/v1", http_client=http), "test-model")


async def test_streams_text_deltas():
    requests = []
    llm = llm_returning(sse(delta(role="assistant", content=""), delta(content="Hel"), delta(content="lo"),
                            {"choices": []}), requests)
    items = [item async for item in llm.stream([{"role": "user", "content": "hi"}])]
    assert items == [TextDelta("Hel"), TextDelta("lo")]
    assert requests[0]["model"] == "test-model"
    assert requests[0]["stream"] is True
    assert "tools" not in requests[0]


async def test_accumulates_tool_call_deltas_by_index():
    body = sse(
        delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function",
                           "function": {"name": "get_weather", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"city": '}}]),
        delta(tool_calls=[{"index": 1, "id": "call_b", "type": "function",
                           "function": {"name": "get_news", "arguments": "{}"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '"Delhi"}'}}]),
    )
    requests = []
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    items = [item async for item in llm_returning(body, requests).stream([], tools)]
    assert items == [ToolCall("call_a", "get_weather", '{"city": "Delhi"}'), ToolCall("call_b", "get_news", "{}")]
    assert requests[0]["tools"] == tools


async def test_tool_call_without_index_defaults_to_zero():
    body = sse(delta(tool_calls=[{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]))
    items = [item async for item in llm_returning(body).stream([])]
    assert items == [ToolCall("x", "f", "{}")]
