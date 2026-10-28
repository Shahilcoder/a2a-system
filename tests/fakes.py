"""Test doubles shared by several layers. Imported only by tests that need them."""

from collections.abc import AsyncIterator

import httpx2
from mcp.server.mcpserver import MCPServer

from a2a_system.protocol.executor import AgentExecutor, RequestContext
from a2a_system.protocol.models import AgentCard, AgentInterface, AgentSkill, TaskState
from a2a_system.protocol.request_handler import RequestHandler
from a2a_system.protocol.server import build_app
from a2a_system.storage import TaskStore

ECHO_CARD = AgentCard(
    name="Echo Agent",
    description="Repeats what you say.",
    version="1.0.0",
    supported_interfaces=[AgentInterface(url="http://test")],
    skills=[AgentSkill(id="echo", name="Echo", description="Echoes text back.", tags=["echo"])],
)


class EchoExecutor(AgentExecutor):
    """Streams 'echo: <text>' as two artifact chunks, then completes."""

    async def execute(self, ctx: RequestContext) -> AsyncIterator:
        yield ctx.status(TaskState.WORKING)
        yield ctx.artifact_chunk("answer", "echo: ", append=False)
        yield ctx.artifact_chunk("answer", ctx.user_text, append=True, last_chunk=True)
        yield ctx.status(TaskState.COMPLETED)


class FailingExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext) -> AsyncIterator:
        yield ctx.status(TaskState.WORKING)
        raise RuntimeError("boom")


def echo_http_client(redis, executor: AgentExecutor | None = None) -> httpx2.AsyncClient:
    """An HTTP client whose requests go straight into an in-process echo agent at http://test."""
    app = build_app(ECHO_CARD, RequestHandler(executor or EchoExecutor(), TaskStore(redis, "echo")))
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test")


def make_mcp_server() -> MCPServer:
    server = MCPServer("test-tools")

    @server.tool()
    async def add(a: int, b: int = 2) -> str:
        """Add two numbers."""
        return f"sum={a + b}"

    @server.tool()
    async def explode() -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    return server
