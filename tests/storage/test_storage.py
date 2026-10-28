from a2a_system.protocol.models import Task, TaskState, TaskStatus
from a2a_system.storage import ConversationStore, TaskStore


async def test_task_store_round_trip(redis):
    store = TaskStore(redis, "sprout")
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.WORKING))
    await store.save(task)
    assert await store.get("t1") == task
    assert await store.get("missing") is None
    assert await redis.exists("sprout:task:t1")
    assert 0 < await redis.ttl("sprout:task:t1")


async def test_conversation_store_appends_in_order(redis):
    store = ConversationStore(redis, "sprout")
    await store.append("c1", {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})
    await store.append("c1", {"role": "user", "content": "bye"})
    assert await store.history("c1") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    assert await store.history("other") == []
    assert 0 < await redis.ttl("sprout:conversation:c1")


async def test_conversation_store_keeps_latest_messages(redis):
    store = ConversationStore(redis, "a", max_messages=2)
    await store.append("c", *({"role": "user", "content": str(i)} for i in range(5)))
    assert [m["content"] for m in await store.history("c")] == ["3", "4"]


async def test_agents_do_not_share_keys(redis):
    await ConversationStore(redis, "a").append("c", {"role": "user", "content": "x"})
    assert await ConversationStore(redis, "b").history("c") == []
