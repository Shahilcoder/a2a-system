"""Redis persistence: A2A tasks and per-context conversation memory."""

import json

from redis.asyncio import Redis

from a2a_system.protocol.models import Task

WEEK = 7 * 24 * 3600


class TaskStore:
    def __init__(self, redis: Redis, agent: str, ttl: int = WEEK):
        self.redis, self.agent, self.ttl = redis, agent, ttl

    def _key(self, task_id: str) -> str:
        return f"{self.agent}:task:{task_id}"

    async def get(self, task_id: str) -> Task | None:
        raw = await self.redis.get(self._key(task_id))
        return Task.model_validate_json(raw) if raw else None

    async def save(self, task: Task) -> None:
        await self.redis.set(self._key(task.id), task.model_dump_json(exclude_none=True), ex=self.ttl)


class ConversationStore:
    """Chat history as OpenAI-style messages ({"role": ..., "content": ...}), one list per contextId."""

    def __init__(self, redis: Redis, agent: str, ttl: int = WEEK, max_messages: int = 40):
        self.redis, self.agent, self.ttl, self.max_messages = redis, agent, ttl, max_messages

    def _key(self, context_id: str) -> str:
        return f"{self.agent}:conversation:{context_id}"

    async def history(self, context_id: str) -> list[dict]:
        return [json.loads(item) for item in await self.redis.lrange(self._key(context_id), 0, -1)]

    async def append(self, context_id: str, *messages: dict) -> None:
        key = self._key(context_id)
        await self.redis.rpush(key, *(json.dumps(message) for message in messages))
        await self.redis.ltrim(key, -self.max_messages, -1)
        await self.redis.expire(key, self.ttl)
