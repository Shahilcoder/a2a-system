import httpx2
import pytest

from a2a_system.agents.app import build_agent
from a2a_system.config import Settings


@pytest.mark.parametrize("service, name", [("sprout", "Sprout"), ("information", "Information Agent"),
                                           ("coding", "Coding Agent")])
async def test_each_agent_service_serves_its_card(service, name):
    app = build_agent(service, Settings(_env_file=None, agent_public_url="http://localhost:9999"))
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://t") as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
    assert card["name"] == name
    assert card["supportedInterfaces"][0]["url"] == "http://localhost:9999"
