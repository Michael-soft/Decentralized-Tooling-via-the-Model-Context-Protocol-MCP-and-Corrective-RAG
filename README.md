# Stage 2 task — Decentralised Thinking Agent (FastMCP + LangChain)

Production-grade MCP monorepo: a FastMCP server exposing a **Reflection Tool** (MCP Sampling) and a **Hierarchical CRAG Resource** (ToT + Tavily fallback), consumed by a LangChain ReAct agent over `streamable-http`.

Built on **LangChain 1.x** (`langchain-classic`, `langchain-tavily`, `langsmith`).

```
DecentralizedToolingviatheModelContextProtocolAndCorrective/
├── pyproject.toml          ← uv workspace root
├── pyrightconfig.json      ← IDE Python env (.venv)
├── REFLECTION.md
├── agent_system.log        ← sample execution log (generated at runtime)
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       ├── __init__.py
│       ├── server.py          ← FastMCP server (tool + resource)
│       └── knowledge_base.py  ← hierarchical knowledge base + search
└── agent_client/
    ├── pyproject.toml
    └── src/agent_client/
        ├── __init__.py
        ├── logger.py      ← dual-stream logging (CLIENT + SERVER → agent_system.log)
        ├── mcp_client.py  ← FastMCP client, sampling handler, @tool wrappers
        └── main.py        ← ReAct agent entrypoint
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

---

## Environment Variables

Create a `.env` file in the repo root **or** export these before running.

> **Note:** `uv run` does not auto-load `.env`. Load it in each terminal session (see Running below).

```bash
export GROQ_API_KEY="gsk_..."          # Required by agent_client
export TAVILY_API_KEY="tvly-..."       # Required by mcp_server (fallback) + agent_client
export MCP_SERVER_URL="http://localhost:8000/mcp"   # Default; change for remote server
```

---

## Setup — Install All Packages

```bash
# From the repo root
uv sync --all-packages
```

This installs both `mcp-server` and `agent-client` into the workspace `.venv`.

---

## Running

### Load environment (each terminal)

**PowerShell:**

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
```

**Bash:**

```bash
set -a && source .env && set +a
```

### Terminal 1 — Start the MCP Server

```bash
uv run --package mcp-server mcp-server
```

Expected output:

```
[2026-05-20 09:00:00] [SERVER] [INFO] ThinkingAgentServer starting up
[2026-05-20 09:00:00] [SERVER] [INFO] Endpoint   : http://0.0.0.0:8000/mcp
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify server health:

```bash
curl http://localhost:8000/health   # → OK
```

### Terminal 2 — Run the Agent Client

```bash
uv run --package agent-client agent-client
```

The agent will:

1. Connect to the MCP server via `streamable-http`.
2. Pull the ReAct prompt from LangSmith Hub (`hwchase17/react`).
3. Run a demo query through the full ReAct loop.
4. Print `Thought / Action / Observation` logs to stdout.
5. Append all `[CLIENT]` and `[SERVER]` entries to `agent_system.log`.

---

## Custom Query

Edit `DEMO_QUERY` in `agent_client/src/agent_client/main.py`, or call programmatically:

```python
from agent_client.main import build_agent_executor, run_query, print_trace

executor = build_agent_executor()
result   = run_query(executor, "Your question here")
print_trace(result)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    agent_client                         │
│                                                         │
│  LangChain ReAct Agent (llama-3.1-8b-instant / Groq)   │
│       │                                                 │
│       ├─ tavily_search            (local tool)          │
│       ├─ remote_crag_tool      ──────────────────────┐  │
│       └─ remote_reflection_tool ────────────────────┐│  │
│                                                     ││  │
│  FastMCP Client (streamable-http)                   ││  │
│  ┌─ sampling_handler → ChatGroq 70b ◄──────────┐   ││  │
│  └─ log_handler      → server_log + file        │   ││  │
└──────────────────────────────────────────────── │ ──┘│──┘
                         HTTP / MCP               │    │
┌──────────────────────────────────────────────── │ ───┘──┐
│                    mcp_server                   │       │
│                                                 │       │
│  FastMCP Server (streamable-http :8000/mcp)     │       │
│                                                 │       │
│  @tool  reflection_tool                         │       │
│    └─ Stage 1: Critique  → ctx.session          │       │
│         .create_message() ──────────────────────┘       │
│    └─ Stage 2: Correction → ctx.session                 │
│         .create_message() ──────────────────────────────┘
│                                                         │
│  @resource knowledge://domain/{query}                   │
│    └─ Multi-query expansion (Sampling)                  │
│    └─ Hierarchical search (knowledge_base.py)           │
│    └─ ToT evaluation ×3 (rule-based: Specificity/Completeness/Novelty) │
│    └─ Tavily fallback (if avg ToT < 0.6)                │
└─────────────────────────────────────────────────────────┘
```

**LangChain stack (agent_client):**

| Package | Role |
|---|---|
| `langchain-classic` | `AgentExecutor`, `create_react_agent` |
| `langchain-tavily` | `TavilySearch` web tool |
| `langsmith` | Pull ReAct prompt from Hub |
| `langchain-groq` | Groq LLM for ReAct loop |

---

## Log Format

```
[2026-05-20 09:00:01] [CLIENT] [INFO]  Sending query to server...
[2026-05-20 09:00:02] [SERVER] [DEBUG] Initiating ToT Evaluation on Resource...
[2026-05-20 09:00:03] [SERVER] [INFO]  ToT scores — Specificity:0.85 Completeness:0.78 Accuracy:0.91
[2026-05-20 09:00:04] [CLIENT] [INFO]  Sampling request received from server | request_id=req-001
```

Both streams are written to `agent_system.log` in the working directory.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Connection refused` on port 8000 | Start server first in Terminal 1 |
| `GROQ_API_KEY not set` / `TAVILY_API_KEY` missing | Load `.env` in the terminal before running |
| `AttributeError: module 'langchainhub' has no attribute 'pull'` | Use current code — prompts are pulled via `langsmith` |
| Import could not be resolved (IDE) | Select `.venv\Scripts\python.exe` as interpreter |
| `413 TPM exceeded` (Groq free tier) | The agent uses compact prompts; wait 60s and retry |
| `Tavily fallback unavailable` | Set `TAVILY_API_KEY` on the server environment |
| `ValidationError field required` | Ensure tool_input is passed as a JSON string |
