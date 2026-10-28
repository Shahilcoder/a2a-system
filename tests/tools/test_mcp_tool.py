from a2a_system.tools.mcp_tool import McpTool, load_mcp_tools
from tests.fakes import make_mcp_server as make_server


async def test_load_lists_tools_with_schemas():
    tools = {tool.name: tool for tool in await load_mcp_tools(make_server())}
    assert set(tools) == {"add", "explode"}
    add = tools["add"]
    assert isinstance(add, McpTool)
    assert add.description == "Add two numbers."
    assert add.parameters["required"] == ["a"]
    assert add.schema()["function"]["name"] == "add"


async def test_call_returns_text():
    tools = {tool.name: tool for tool in await load_mcp_tools(make_server())}
    assert await tools["add"].call(a=1) == "sum=3"


async def test_call_reports_tool_errors():
    tools = {tool.name: tool for tool in await load_mcp_tools(make_server())}
    assert (await tools["explode"].call()).startswith("Tool error:")
