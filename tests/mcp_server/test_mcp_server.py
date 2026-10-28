import httpx2
from mcp import Client

from a2a_system.config import Settings
from a2a_system.mcp_server.server import build_server


async def test_server_exposes_information_tools():
    server = build_server(httpx2.AsyncClient(), Settings(_env_file=None))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"get_weather", "get_news", "web_search", "search_trains", "search_flights"}
    assert tools["get_weather"].input_schema["required"] == ["city"]
    assert tools["get_weather"].description


async def test_tools_delegate_to_providers():
    server = build_server(httpx2.AsyncClient(), Settings(_env_file=None, rapidapi_key=""))
    async with Client(server) as client:
        result = await client.call_tool("search_trains", {"from_station": "NDLS", "to_station": "BCT",
                                                          "date": "2026-11-02"})
    assert "not configured" in result.content[0].text
