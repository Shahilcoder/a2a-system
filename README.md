# A2A System: Sprout

A small multi-agent system built on a hand-rolled implementation of the [A2A protocol v1.0](https://a2a-protocol.org/latest/specification/)
(JSON-RPC binding plus SSE). Sprout is the only agent you talk to. It delegates to specialist agents over A2A:

- **Sprout** (host): a friendly chat agent. Each remote agent's Agent Card becomes a function-calling tool for it.
- **Information agent**: answers with live data from an MCP server (weather, news, web search, Indian trains, flights).
- **Coding agent**: reasons about and writes code. It never runs anything.
- **MCP server**: our own MCP server (`mcp` SDK v2, Streamable HTTP) that wraps the external APIs.
- **Gateway**: an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) in front of Sprout.

```
 user ──A2A──▶ sprout:8001 ──A2A──▶ information-agent:8002 ──MCP──▶ mcp-server:8004 ──▶ Open-Meteo / Google News / Tavily|DDG / IRCTC / AviationStack
 OpenAI SDK ─▶ gateway:8080 ─A2A─▶ sprout        └──A2A──▶ coding-agent:8003
                                      all agents ──▶ redis (tasks + conversation memory)
                                      all agents ──▶ LLM_BASE_URL (any OpenAI-compatible API)
```

The source is typed in by hand from [IMPLEMENTATION.md](IMPLEMENTATION.md). It goes layer by layer, with a
test checkpoint after each section.

## Run

```bash
cp .env.example .env          # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (the model needs tool calling)
docker compose up --build
```

| Service | Host port | Run as |
|---|---|---|
| sprout | 8001 | `python -m a2a_system sprout` |
| information-agent | 8002 | `python -m a2a_system information` |
| coding-agent | 8003 | `python -m a2a_system coding` |
| mcp-server | 8004 (`/mcp`) | `python -m a2a_system mcp-server` |
| gateway | 8080 | `python -m a2a_system gateway` |

All services use one image. `./src` and `./tests` are mounted, so code changes only need a restart
(`docker compose restart sprout`), not a rebuild. To use a different model for one agent, set `LLM_MODEL`
in that service's `environment` in `docker-compose.yml`.

## Try it

```bash
# Agent Card
curl localhost:8001/.well-known/agent-card.json

# A2A SendMessage (the A2A-Version header is required)
curl -H 'A2A-Version: 1.0' localhost:8001/ -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":
  {"messageId":"1","role":"ROLE_USER","parts":[{"text":"weather in Delhi?"}],"contextId":"me"}}}'

# A2A streaming (SSE)
curl -N -H 'A2A-Version: 1.0' localhost:8001/ -d '{"jsonrpc":"2.0","id":2,"method":"SendStreamingMessage","params":{"message":
  {"messageId":"2","role":"ROLE_USER","parts":[{"text":"write a haiku about trains"}]}}}'

# OpenAI-compatible gateway
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sprout","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="sprout",
    user="shahil",  # becomes the A2A contextId, so Sprout remembers the conversation
    messages=[{"role": "user", "content": "Any trains from NDLS to BCT on 2026-11-02?"}],
)
print(reply.choices[0].message.content)
```

## Tests

```bash
docker compose --profile test run --rm tests              # whole suite
docker compose --profile test run --rm tests tests/agents # one layer
uv sync && uv run pytest                                  # or locally
```

The tests don't need network access or API keys: Redis is `fakeredis`, HTTP is `httpx2.MockTransport` or
an in-process ASGI app, the MCP server runs in memory, and the LLM is scripted.

## Configuration

| Variable | Used by | Notes |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | agents | Any OpenAI-compatible API with tools and streaming. For a model on your Mac use `http://host.docker.internal:<port>/v1` |
| `TAVILY_API_KEY` | mcp-server | Optional. Without it, web search uses DuckDuckGo |
| `RAPIDAPI_KEY` | mcp-server | RapidAPI "IRCTC" (`irctc1`) for train search |
| `AVIATIONSTACK_API_KEY` | mcp-server | Flight search (free tier, HTTP only) |
| `REDIS_URL`, `AGENT_PUBLIC_URL`, `REMOTE_AGENT_URLS`, `MCP_SERVER_URL`, `SPROUT_URL` | per service | Set in `docker-compose.yml` |

## Protocol coverage

Implemented: `SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`, and the Agent Card at
`/.well-known/agent-card.json`. `ListTasks`, `SubscribeToTask`, the push-notification config methods and
`GetExtendedAgentCard` return `UnsupportedOperationError (-32004)`. A request without `A2A-Version: 1.0`
gets `VersionNotSupportedError (-32009)`.

## Deliberate shortcuts

Search the code for `ponytail:` to find them.

- `CancelTask` takes effect at the executor's next event. An LLM or tool call that is already running finishes first.
- Sprout calls remote agents without streaming. Each MCP tool call opens a new MCP session.
- The gateway has no auth. It is only meant for local Docker use.
- Each message starts a new task, so multi-turn tasks (`INPUT_REQUIRED`) aren't used. Conversation memory is per `contextId` instead.

## To verify

- The RapidAPI `irctc1` endpoint path, params and response fields in `mcp_server/providers/trains.py` were
  written from memory. Check them against your RapidAPI dashboard.
# a2a-system
