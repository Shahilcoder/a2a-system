import json

import pytest

from a2a_system.protocol.executor import AgentExecutor
from a2a_system.protocol.models import Message, Task, TaskState, TaskStatus
from a2a_system.protocol.request_handler import RequestHandler
from a2a_system.storage import TaskStore
from tests.fakes import EchoExecutor, FailingExecutor, echo_http_client

HEADERS = {"A2A-Version": "1.0"}


def rpc(method, params=None, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


def send_params(text="hi", **message):
    return {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}], **message}}


def sse_events(body: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


@pytest.fixture
async def client(redis):
    async with echo_http_client(redis) as http:
        yield http


async def test_agent_card_route(client):
    response = await client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "Echo Agent"
    assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
    assert card["capabilities"]["streaming"] is True


async def test_send_message_returns_completed_task(client):
    response = await client.post("/", json=rpc("SendMessage", send_params("hello")), headers=HEADERS)
    body = response.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == 1
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"] == [{"artifactId": "answer", "parts": [{"text": "echo: "}, {"text": "hello"}]}]
    assert task["contextId"]
    assert task["history"][0]["taskId"] == task["id"]


async def test_context_id_is_kept_and_task_id_ignored(client):
    params = send_params(contextId="ctx-1", taskId="old-task")
    task = (await client.post("/", json=rpc("SendMessage", params), headers=HEADERS)).json()["result"]["task"]
    assert task["contextId"] == "ctx-1"
    assert task["id"] != "old-task"


async def test_get_task(client):
    sent = (await client.post("/", json=rpc("SendMessage", send_params()), headers=HEADERS)).json()["result"]["task"]
    got = (await client.post("/", json=rpc("GetTask", {"id": sent["id"]}), headers=HEADERS)).json()["result"]
    assert got == sent


async def test_unknown_task_is_task_not_found(client):
    body = (await client.post("/", json=rpc("GetTask", {"id": "nope"}), headers=HEADERS)).json()
    assert body["error"]["code"] == -32001


async def test_cancel_completed_task_is_not_cancelable(client):
    sent = (await client.post("/", json=rpc("SendMessage", send_params()), headers=HEADERS)).json()["result"]["task"]
    body = (await client.post("/", json=rpc("CancelTask", {"id": sent["id"]}), headers=HEADERS)).json()
    assert body["error"]["code"] == -32002


async def test_cancel_working_task(redis, client):
    await TaskStore(redis, "echo").save(Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.WORKING)))
    body = (await client.post("/", json=rpc("CancelTask", {"id": "t1"}), headers=HEADERS)).json()
    assert body["result"]["status"]["state"] == "TASK_STATE_CANCELED"


async def test_unknown_method(client):
    body = (await client.post("/", json=rpc("DoMagic"), headers=HEADERS)).json()
    assert body["error"]["code"] == -32601


@pytest.mark.parametrize("method", ["ListTasks", "SubscribeToTask", "GetExtendedAgentCard",
                                    "CreateTaskPushNotificationConfig"])
async def test_unsupported_methods(client, method):
    body = (await client.post("/", json=rpc(method), headers=HEADERS)).json()
    assert body["error"]["code"] == -32004


async def test_missing_version_header(client):
    body = (await client.post("/", json=rpc("SendMessage", send_params()))).json()
    assert body["error"]["code"] == -32009
    assert body["id"] == 1


async def test_parse_error(client):
    body = (await client.post("/", content=b"{not json", headers=HEADERS)).json()
    assert body["error"]["code"] == -32700


async def test_invalid_request(client):
    body = (await client.post("/", json={"id": 1, "method": "SendMessage"}, headers=HEADERS)).json()
    assert body["error"]["code"] == -32600


async def test_invalid_params(client):
    body = (await client.post("/", json=rpc("SendMessage", {"message": {"parts": []}}), headers=HEADERS)).json()
    assert body["error"]["code"] == -32602


async def test_streaming_events_in_order(client):
    response = await client.post("/", json=rpc("SendStreamingMessage", send_params("yo")), headers=HEADERS)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [e["result"] for e in sse_events(response.text)]
    assert [next(iter(e)) for e in events] == ["task", "statusUpdate", "artifactUpdate", "artifactUpdate",
                                               "statusUpdate"]
    assert events[0]["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    first, last = events[2]["artifactUpdate"], events[3]["artifactUpdate"]
    assert first["append"] is False and first.get("lastChunk", False) is False
    assert last["append"] is True and last["lastChunk"] is True
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    task_id = events[0]["task"]["id"]
    assert all(e[k]["taskId"] == task_id for e in events[1:] for k in e)


async def test_streaming_bad_params_is_plain_json_error(client):
    response = await client.post("/", json=rpc("SendStreamingMessage", {}), headers=HEADERS)
    assert response.json()["error"]["code"] == -32602


async def test_executor_error_fails_the_task(redis):
    async with echo_http_client(redis, FailingExecutor()) as client:
        response = await client.post("/", json=rpc("SendStreamingMessage", send_params()), headers=HEADERS)
        last = sse_events(response.text)[-1]["result"]["statusUpdate"]["status"]
        assert last["state"] == "TASK_STATE_FAILED"
        assert "boom" in last["message"]["parts"][0]["text"]

        task = (await client.post("/", json=rpc("SendMessage", send_params()), headers=HEADERS)).json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"


async def test_cancel_stops_a_running_task(redis):
    handler = RequestHandler(EchoExecutor(), TaskStore(redis, "echo"))
    stream = handler.stream(send_params())
    task_id = (await anext(stream))["task"]["id"]
    assert (await anext(stream))["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"

    await handler.cancel_task(task_id)
    rest = [event async for event in stream]
    assert [e["statusUpdate"]["status"]["state"] for e in rest] == ["TASK_STATE_CANCELED"]
    assert (await handler.get_task(task_id)).status.state is TaskState.CANCELED


async def test_handler_is_executor_agnostic(redis):
    class Silent(AgentExecutor):
        async def execute(self, ctx):
            yield ctx.status(TaskState.COMPLETED, "nothing to say")

    handler = RequestHandler(Silent(), TaskStore(redis, "s"))
    task = await handler.send_message(Message.model_validate(send_params()["message"]))
    assert task.status.state is TaskState.COMPLETED
    assert task.status.message.parts[0].text == "nothing to say"
