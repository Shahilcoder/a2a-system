import pytest
from pydantic import ValidationError

from a2a_system.protocol.models import (
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    task_text,
)


def test_message_round_trips_camel_case():
    wire = {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hi"}], "contextId": "c1"}
    message = Message.model_validate(wire)
    assert message.message_id == "m1"
    assert message.context_id == "c1"
    assert message.role is Role.USER
    assert message.dump() == wire


def test_python_names_also_accepted():
    message = Message(message_id="m1", role=Role.AGENT, parts=[Part(text="x")])
    assert message.dump() == {"messageId": "m1", "role": "ROLE_AGENT", "parts": [{"text": "x"}]}


@pytest.mark.parametrize("fields", [{}, {"text": "a", "url": "http://x"}, {"text": "a", "data": {"k": 1}}])
def test_part_needs_exactly_one_content_field(fields):
    with pytest.raises(ValidationError):
        Part(**fields)


@pytest.mark.parametrize("fields", [{"text": ""}, {"raw": "aGk="}, {"url": "http://x/f.png"}, {"data": {"k": 1}}])
def test_valid_parts(fields):
    assert Part(**fields).dump() == fields


def test_part_media_type_alias():
    assert Part(url="http://x", media_type="image/png").dump() == {"url": "http://x", "mediaType": "image/png"}


def test_enum_wire_values():
    assert TaskState.WORKING == "TASK_STATE_WORKING"
    assert TaskState.INPUT_REQUIRED == "TASK_STATE_INPUT_REQUIRED"
    assert Role.USER == "ROLE_USER"


@pytest.mark.parametrize("state", list(TaskState))
def test_terminal_states(state):
    expected = state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
    assert state.is_terminal is expected


def test_task_round_trip():
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.COMPLETED),
                artifacts=[Artifact(artifact_id="a1", parts=[Part(text="hello")])])
    wire = task.dump()
    assert wire["contextId"] == "c1"
    assert wire["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "timestamp" in wire["status"]
    assert wire["artifacts"] == [{"artifactId": "a1", "parts": [{"text": "hello"}]}]
    assert Task.model_validate(wire) == task


def test_stream_response_needs_exactly_one_payload():
    status = TaskStatusUpdateEvent(task_id="t", context_id="c", status=TaskStatus(state=TaskState.WORKING))
    assert set(StreamResponse(status_update=status).dump()) == {"statusUpdate"}
    with pytest.raises(ValidationError):
        StreamResponse()


def test_agent_card_wire_shape():
    card = AgentCard(name="A", description="d", version="1.0.0",
                     supported_interfaces=[AgentInterface(url="http://a")],
                     skills=[AgentSkill(id="s", name="S", description="sd", tags=["t"])])
    wire = card.dump()
    assert wire["supportedInterfaces"] == [{"url": "http://a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]
    assert wire["capabilities"] == {"streaming": True, "pushNotifications": False}
    assert wire["defaultInputModes"] == ["text/plain"]
    assert wire["skills"] == [{"id": "s", "name": "S", "description": "sd", "tags": ["t"]}]


def test_task_text_prefers_artifacts_then_status_message():
    done = Task(id="t", context_id="c", status=TaskStatus(state=TaskState.COMPLETED),
                artifacts=[Artifact(artifact_id="a", parts=[Part(text="he"), Part(text="llo")])])
    assert task_text(done) == "hello"
    error = Message(message_id="m", role=Role.AGENT, parts=[Part(text="it broke")])
    failed = Task(id="t", context_id="c", status=TaskStatus(state=TaskState.FAILED, message=error))
    assert task_text(failed) == "it broke"
