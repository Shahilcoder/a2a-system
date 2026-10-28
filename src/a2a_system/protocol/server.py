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
