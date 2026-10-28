from a2a_system.protocol.client import A2AClient
from a2a_system.tools.remote_agent_tool import RemoteAgentTool
from tests.fakes import ECHO_CARD, FailingExecutor, echo_http_client


def test_name_and_schema_come_from_the_card():
    tool = RemoteAgentTool(ECHO_CARD, "http://test", A2AClient())
    schema = tool.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ask_echo_agent"
    assert "Repeats what you say." in schema["function"]["description"]
    assert "Echoes text back." in schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["query"]


async def test_call_returns_the_remote_answer(redis):
    async with echo_http_client(redis) as http:
        tool = RemoteAgentTool(ECHO_CARD, "http://test", A2AClient(http))
        assert await tool.call(query="ping") == "echo: ping"


async def test_call_reports_remote_failure(redis):
    async with echo_http_client(redis, FailingExecutor()) as http:
        result = await RemoteAgentTool(ECHO_CARD, "http://test", A2AClient(http)).call(query="ping")
        assert "TASK_STATE_FAILED" in result
        assert "boom" in result
