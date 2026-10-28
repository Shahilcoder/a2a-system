import json
from typing import Any

import httpx2

from a2a_system.agents.base import MAX_TOOL_ROUNDS, LLMAgent
from a2a_system.agents.coding import CodingAgent
from a2a_system.agents.information import InformationAgent
from a2a_system.agents.sprout import SproutAgent
from a2a_system.llm.client import TextDelta, ToolCall
from a2a_system.protocol.client import A2AClient
from a2a_system.protocol.executor import RequestContext
from a2a_system.protocol.models import AgentSkill, Message, Part, Role, TaskState
from a2a_system.storage import ConversationStore
from a2a_system.tools.base import Tool
from tests.fakes import ECHO_CARD, make_mcp_server


class ScriptedLLM:
    """Replays one scripted response per stream() call and records what it was sent."""

    def __init__(self, *responses: list):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def stream(self, messages, tools=None):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        for item in self.responses.pop(0):
            yield item


class Clock(Tool):
    name, description = "clock", "Tells the time."
    parameters = {"type": "object", "properties": {"zone": {"type": "string"}}}

    def __init__(self):
        self.calls = []

    async def call(self, **arguments):
        self.calls.append(arguments)
        return "12:00"


class ToolAgent(LLMAgent):
    name, description, system_prompt = "Tool Agent", "Has a clock.", "You are a test agent."
    skills = [AgentSkill(id="t", name="T", description="t", tags=["t"])]

    def __init__(self, llm, conversations, tools):
        super().__init__(llm, conversations)
        self._tools = tools

    async def tools(self):
        return self._tools


def context(text="what time is it?", context_id="ctx"):
    return RequestContext(task_id="task", context_id=context_id,
                          message=Message(message_id="m", role=Role.USER, parts=[Part(text=text)]))


async def run(agent, ctx):
    return [event async for event in agent.execute(ctx)]


async def test_tool_loop_then_streamed_answer(redis):
    clock = Clock()
    llm = ScriptedLLM([ToolCall("call_1", "clock", '{"zone": "IST"}')], [TextDelta("It is "), TextDelta("noon.")])
    agent = ToolAgent(llm, ConversationStore(redis, "t"), [clock])
    events = await run(agent, context())

    assert clock.calls == [{"zone": "IST"}]
    assert llm.calls[0]["tools"] == [clock.schema()]
    second = llm.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "clock", "arguments": '{"zone": "IST"}'}}]}
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "12:00"}

    assert events[0].status.state is TaskState.WORKING
    assert events[-1].status.state is TaskState.COMPLETED
    chunks = [e for e in events if hasattr(e, "artifact")]
    assert "".join(c.artifact.parts[0].text for c in chunks) == "It is noon."
    assert len({c.artifact.artifact_id for c in chunks}) == 1
    assert [c.append for c in chunks] == [False, True, True]
    assert [c.last_chunk for c in chunks] == [False, False, True]


async def test_history_is_saved_and_reused(redis):
    conversations = ConversationStore(redis, "t")
    llm = ScriptedLLM([ToolCall("c", "clock", "{}")], [TextDelta("Noon.")], [TextDelta("You asked the time.")])
    agent = ToolAgent(llm, conversations, [Clock()])
    await run(agent, context("time?"))
    assert await conversations.history("ctx") == [{"role": "user", "content": "time?"},
                                                  {"role": "assistant", "content": "Noon."}]
    await run(agent, context("what did I ask?"))
    messages = llm.calls[-1]["messages"]
    assert messages[0]["role"] == "system" and "You are a test agent." in messages[0]["content"]
    assert messages[1:] == [{"role": "user", "content": "time?"}, {"role": "assistant", "content": "Noon."},
                            {"role": "user", "content": "what did I ask?"}]


async def test_tool_failures_and_unknown_tools_go_back_to_the_llm(redis):
    class Broken(Clock):
        name = "broken"

        async def call(self, **arguments):
            raise RuntimeError("no battery")

    llm = ScriptedLLM([ToolCall("a", "broken", "{}"), ToolCall("b", "nope", "{}")], [TextDelta("Sorry!")])
    events = await run(ToolAgent(llm, ConversationStore(redis, "t"), [Broken()]), context())
    results = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "no battery" in results[0]
    assert "nope" in results[1]
    assert events[-1].status.state is TaskState.COMPLETED


async def test_tool_rounds_are_capped(redis):
    rounds = [[ToolCall(str(i), "clock", "{}")] for i in range(MAX_TOOL_ROUNDS)]
    llm = ScriptedLLM(*rounds, [TextDelta("done")])
    events = await run(ToolAgent(llm, ConversationStore(redis, "t"), [Clock()]), context())
    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert not llm.calls[-1]["tools"]  # the last round offers no tools, forcing an answer
    assert events[-1].status.state is TaskState.COMPLETED


def test_card(redis):
    card = CodingAgent(ScriptedLLM(), ConversationStore(redis, "c")).card("http://localhost:8003")
    assert card.name == "Coding Agent"
    assert card.supported_interfaces[0].url == "http://localhost:8003"
    assert card.skills


async def test_coding_agent_has_no_tools(redis):
    assert await CodingAgent(ScriptedLLM(), ConversationStore(redis, "c")).tools() == []


async def test_information_agent_loads_mcp_tools(redis):
    agent = InformationAgent(ScriptedLLM(), ConversationStore(redis, "i"), make_mcp_server())
    assert {tool.name for tool in await agent.tools()} == {"add", "explode"}


async def test_sprout_turns_remote_cards_into_tools(redis):
    down = {"coder"}
    fetched = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        fetched.append(request.url.host)
        if request.url.host in down:
            raise httpx2.ConnectError("refused")
        return httpx2.Response(200, json=ECHO_CARD.model_copy(update={"name": request.url.host}).dump())

    a2a = A2AClient(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    sprout = SproutAgent(ScriptedLLM(), ConversationStore(redis, "s"), a2a, ["http://echo:8000", "http://coder:8000"])
    tools = await sprout.tools()
    assert [tool.name for tool in tools] == ["ask_echo"]  # an unreachable agent is skipped
    assert tools[0].url == "http://echo:8000"

    down.clear()
    fetched.clear()
    assert [tool.name for tool in await sprout.tools()] == ["ask_echo", "ask_coder"]
    assert fetched == ["coder"]  # successful lookups are cached, failed ones are retried
