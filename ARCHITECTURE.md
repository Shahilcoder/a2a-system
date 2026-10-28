# Architecture

## The big picture

You only ever talk to **Sprout**. Sprout is a friendly chat agent. When it needs help, it asks another agent over **A2A** (the Agent-to-Agent protocol).

- **Sprout**: chats with you and decides who to ask.
- **Information agent**: looks up live facts using tools from the MCP server.
- **Coding agent**: writes and explains code. It never runs code.
- **MCP server**: wraps outside APIs (weather, news, search, trains, flights) as tools.
- **Gateway**: lets any OpenAI client (SDK, chat app) talk to Sprout.
- **Redis**: remembers tasks and conversations.

```mermaid
flowchart LR
    User([You]) -- A2A --> Sprout
    OpenAI([OpenAI SDK / chat app]) --> Gateway -- A2A --> Sprout
    Sprout -- A2A --> Info[Information agent]
    Sprout -- A2A --> Coding[Coding agent]
    Info -- MCP --> MCP[MCP server]
    MCP --> APIs[(Open-Meteo, Google News,<br/>Tavily / DuckDuckGo,<br/>IRCTC, AviationStack)]
    Sprout & Info & Coding --> Redis[(Redis)]
    Sprout & Info & Coding --> LLM[(LLM API)]
```

Every box is the same Docker image started with a different command, for example `python -m a2a_system sprout`.

## How agents find each other

Each agent publishes an **Agent Card** at `/.well-known/agent-card.json`. The card holds its name, what it can do (its skills) and how to call it.

On its first request, Sprout reads the cards of the other agents and turns each one into a **tool** for its LLM: `ask_information_agent` and `ask_coding_agent`. From then on, the LLM decides when to delegate, the same way it decides to call any other function.

## Inside one agent

All three agents are built from the same layers:

```mermaid
flowchart TD
    HTTP["server.py<br/>HTTP routes: Agent Card + JSON-RPC"] --> Handler["RequestHandler<br/>creates the task, saves every update"]
    Handler --> Agent["LLMAgent.execute()<br/>yields status + answer chunks"]
    Agent --> Loop["Tool loop<br/>LLM → tools → LLM (max 5 rounds)"]
    Loop --> T1["RemoteAgentTool<br/>(Sprout: asks another agent)"]
    Loop --> T2["McpTool<br/>(Information: calls the MCP server)"]
    Handler <--> Redis[(Redis)]
```

- **server.py** receives the request and checks the `A2A-Version: 1.0` header.
- **RequestHandler** creates a task, runs the agent and saves each update to Redis.
- **LLMAgent** loads your chat history, runs the LLM, calls tools when asked, and streams the answer back piece by piece.

## A request, step by step

You ask Sprout: *"What's the weather in Delhi?"*

```mermaid
sequenceDiagram
    actor You
    participant S as Sprout
    participant I as Information agent
    participant M as MCP server
    participant W as Open-Meteo
    participant L as LLM

    You->>S: SendStreamingMessage("weather in Delhi?")
    S-->>You: task created (SUBMITTED, then WORKING)
    S->>L: chat history + question + tools
    L-->>S: call ask_information_agent("weather in Delhi")
    S->>I: SendMessage("weather in Delhi")
    I->>L: question + MCP tools
    L-->>I: call get_weather(city="Delhi")
    I->>M: get_weather
    M->>W: geocode + forecast
    W-->>M: data
    M-->>I: "Delhi: 27°C, clear sky…"
    I->>L: tool result
    L-->>I: final answer
    I-->>S: task COMPLETED + answer
    S->>L: tool result
    L-->>S: answer, streamed in pieces
    S-->>You: answer chunks (artifact updates)
    S-->>You: COMPLETED
```

Two details:

- Sprout **streams** its answer to you. The call from Sprout to the Information agent waits for the whole answer instead.
- The Information agent doesn't see your chat history. Sprout rewrites follow-ups ("and tomorrow?") into full questions before it asks.

## Task lifecycle

Every message you send starts a new **task**. Its state moves like this:

```mermaid
stateDiagram-v2
    [*] --> SUBMITTED
    SUBMITTED --> WORKING
    WORKING --> COMPLETED: answer done
    WORKING --> FAILED: error
    WORKING --> CANCELED: CancelTask
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELED --> [*]
```

A stream always ends with one of the final states, so a client never waits forever.

## The OpenAI gateway

The gateway translates between OpenAI's chat API and A2A.

```mermaid
sequenceDiagram
    participant C as OpenAI client
    participant G as Gateway
    participant S as Sprout

    C->>G: POST /v1/chat/completions (stream=true)
    G->>S: SendStreamingMessage
    loop each answer chunk
        S-->>G: artifactUpdate
        G-->>C: chat.completion.chunk
    end
    S-->>G: COMPLETED
    G-->>C: data: [DONE]
```

If the request sets `user`, the gateway uses it as the conversation id, so Sprout remembers earlier messages.

## Memory

Redis stores two things. Each key starts with the agent's name, so the agents can share one Redis without clashing.

| Key | Holds |
|---|---|
| `sprout:task:<id>` | One task: its state and its answer |
| `sprout:conversation:<contextId>` | The last 40 chat messages (your questions and the final answers) |

Tool calls are not saved, only what you said and what the agent answered.
