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
