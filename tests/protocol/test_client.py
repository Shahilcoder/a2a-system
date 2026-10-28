import pytest

import a2a_system.protocol.client as client_module
from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.errors import VersionNotSupportedError
from a2a_system.protocol.models import Message, Part, Role, TaskState
from tests.fakes import echo_http_client


def message(text="hi"):
    return Message(message_id="m1", role=Role.USER, parts=[Part(text=text)])


@pytest.fixture
async def a2a(redis):
    async with echo_http_client(redis) as http:
        yield A2AClient(http)


async def test_get_agent_card(a2a):
    card = await a2a.get_agent_card("http://test")
    assert card.name == "Echo Agent"
    assert card.skills[0].id == "echo"


async def test_send_message(a2a):
    task = await a2a.send_message("http://test", message("ping"))
    assert task.status.state is TaskState.COMPLETED
    assert [p.text for p in task.artifacts[0].parts] == ["echo: ", "ping"]


async def test_stream_message(a2a):
    events = [event async for event in a2a.stream_message("http://test", message("ping"))]
    assert events[0].task is not None
    chunks = [e.artifact_update.artifact.parts[0].text for e in events if e.artifact_update]
    assert "".join(chunks) == "echo: ping"
    assert events[-1].status_update.status.state is TaskState.COMPLETED


async def test_json_rpc_errors_raise(redis, monkeypatch):
    monkeypatch.setattr(client_module, "HEADERS", {})  # without A2A-Version the server answers with -32009
    async with echo_http_client(redis) as http:
        a2a = A2AClient(http)
        with pytest.raises(VersionNotSupportedError):
            await a2a.send_message("http://test", message())
        with pytest.raises(VersionNotSupportedError):
            async for _ in a2a.stream_message("http://test", message()):
                pass
