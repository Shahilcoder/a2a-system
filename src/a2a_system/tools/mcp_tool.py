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
