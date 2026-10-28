import pytest
from fakeredis import FakeAsyncRedis


@pytest.fixture
async def redis():
    client = FakeAsyncRedis()
    yield client
    await client.aclose()
