**Hierarchical Log Persistence and Graph Knowledge Mapping.** 

A production-grade **uv monorepo** spanning three decoupled processes:

1. **`mcp_server`** — FastMCP server exposing a **Reflection Tool** (MCP Sampling) and a
   **Hierarchical CRAG Resource** (LLM Tree-of-Thought grading via Sampling + Tavily
   fallback) over `streamable-http`.
2. **`agent_client`** — a LangChain agent (`create_agent`) that consumes the server, and
   (Stage 3) persists every MCP interaction into an **embedded, vector-enabled LangGraph
   SQLite log store** using hierarchical namespaces.
3. **`analysis_dashboard`** — a **decoupled Log Analysis Agent** (LangChain `create_agent`)
   that semantically searches the logs, projects causal execution paths into a **Neo4j
   Aura DB** knowledge graph, computes performance trends (matplotlib/seaborn), and serves
   an interactive **Streamlit** diagnostic dashboard.

Built on **LangChain 1.x** (`langchain`, `langchain-classic`, `langchain-tavily`,
`langsmith`, `langgraph`), **Groq** for inference, **fastembed** for local embeddings.

```
Decentralized-Tooling-via-the-Model-Context-Protocol-MCP-and-Corrective-RAG/
├── pyproject.toml              ← uv workspace root (3 members)
├── .env.example                ← all config keys (copy to .env)
├── REFLECTION.md               ← Stage 2 reflection
├── REFLECTION_STAGE3.md        ← Stage 3 reflection
├── mcp_agent_system.log        ← flat dual-stream log (sample multi-turn run)
├── mcp_agent_log.db            ← vector-enabled SQLite log store
├── analysis_agent.log          ← analysis-agent execution log
├── charts/                     ← generated trend charts (PNG)
├── mcp_server/
│   └── src/mcp_server/
│       ├── server.py               ← FastMCP server (tool + resource)
│       └── knowledge_base.py       ← hierarchical knowledge base
├── agent_client/
│   └── src/agent_client/
│       ├── logger.py               ← dual-stream flat logging
│       ├── embeddings.py           ← local FastEmbed wrapper (384-dim)
│       ├── session.py              ← per-run session UUID
│       ├── log_store.py            ← vector SQLite store + LogEntry schema + guardrails
│       ├── mcp_client.py           ← FastMCP client, sampling handler, instrumented @tools
│       └── main.py                 ← create_agent entrypoint (multi-turn, vector-logged)
└── analysis_dashboard/
    └── src/analysis_dashboard/
        ├── embeddings.py           ← mirror of the client embedding config
        ├── store_reader.py         ← read-side store accessor (int-key guardrail)
        ├── log_retrieval.py        ← semantic_log_search @tool
        ├── graph_mapper.py         ← Neo4j projection + sync_knowledge_graph @tool
        ├── trend_tools.py          ← latency / token / error analytics + charts
        ├── agent.py                ← Log Analysis Agent (create_agent) + CLI
        └── app.py                  ← Streamlit diagnostic dashboard
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://python.org) (uv can also manage this) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

External services:

- **Groq** API key — inference for both the ReAct loop and the analysis agent.
- **Tavily** API key — web grounding (client tool + server CRAG fallback).
- **Neo4j Aura DB** (free tier) — knowledge-graph projection target.
- **Embeddings** are *local* (`fastembed`, BAAI/bge-small-en-v1.5) — **no key required**.

---

## Setup

```bash
# From the repo root — installs all three packages into one .venv
uv sync --all-packages
```

### Neo4j Aura DB (free) setup

1. Go to <https://neo4j.com/product/auradb/> → create a **free** instance.
2. On creation, download/copy the credentials. You need:
   - `NEO4J_URI`  (looks like `neo4j+s://<id>.databases.neo4j.io`)
   - `NEO4J_USERNAME` (default `neo4j`)
   - `NEO4J_PASSWORD`
3. Paste them into `.env` (below). The graph mapper degrades gracefully and reports
   `not_configured` if these are blank — it never emits a broken connection string.

### Environment variables

Copy `.env.example` → `.env` and fill in real values:

```bash
cp .env.example .env
# then edit .env with your GROQ / TAVILY / NEO4J credentials
```

> **Note:** `uv run` does **not** auto-load `.env`. Load it per terminal (commands below).
> `main.py` and `agent.py` also attempt a best-effort `dotenv` load as a fallback.

---

## Running — split terminal sessions

Load the environment **once per terminal**, then start each process.

### Load environment

```bash
# Bash / sh / zsh
set -a && . ./.env && set +a
```

```powershell
# PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
        $v = $matches[2].Trim().Trim('"')
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $v, 'Process')
    }
}
```

```cmd
:: Windows CMD — load .env (strip surrounding quotes)
for /f "usebackq tokens=1,* delims==" %i in (".env") do set %i=%~j
```

### Terminal 1 — MCP Server (core service)

```bash
# bash / sh
uv run --package mcp-server mcp-server
```
```powershell
# PowerShell
uv run --package mcp-server mcp-server
```
```cmd
:: CMD
uv run --package mcp-server mcp-server
```

Verify health: `curl http://localhost:8000/health`  → `OK`

### Terminal 2 — Agent Client (generates the vector log trace)

```bash
# bash / sh — runs the multi-turn demo, writing mcp_agent_system.log + mcp_agent_log.db
uv run --package agent-client agent-client
```
```powershell
# PowerShell
uv run --package agent-client agent-client
```
```cmd
:: CMD
uv run --package agent-client agent-client
```

### Terminal 3 — Observability control plane (Streamlit dashboard)

```bash
# bash / sh
uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```
```powershell
# PowerShell
uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```
```cmd
:: CMD
uv run --package analysis-dashboard streamlit run analysis_dashboard\src\analysis_dashboard\app.py
```

Open <http://localhost:8501> and ask, e.g.:
*"Check error frequency, then sync the knowledge graph to Neo4j and report the committed nodes/edges."*

### (Optional) Run the Log Analysis Agent headless (CLI)

```bash
# bash / sh — runs a diagnostic demo, writing analysis_agent.log + charts/
uv run --package analysis-dashboard analysis-agent
```
```powershell
uv run --package analysis-dashboard analysis-agent
```
```cmd
uv run --package analysis-dashboard analysis-agent
```

---

## Stage 3 architecture

```
┌──────────────── OPERATIONAL PLANE (Terminals 1 + 2) ────────────────┐
│  agent_client (create_agent / Groq)                                 │
│    ├─ tavily_search                                                 │
│    ├─ remote_crag_tool ───────────► mcp_server  knowledge://domain  │
│    └─ remote_reflection_tool ─────► mcp_server  reflection_tool     │
│           │                              │                          │
│           │   FastMCP streamable-http    │  MCP Sampling (no LLM     │
│           │                              ▼  on server) ─────────────┤
│   every interaction (tool / resource / sampling) is persisted to    │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │  mcp_agent_log.db  — LangGraph SqliteStore (vector index)   │   │
│   │  namespaces: ("logs","mcp","server","tools","reflection_tool")│ │
│   │  schema: session_id · mcp_interaction_type · content + meta │   │
│   └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────│──────────────────────────────┘
                                        │  (read-only, separate process)
┌──────────────── ANALYSIS PLANE (Terminal 3) ──────────────────────┐
│  analysis_dashboard — Log Analysis Agent (create_agent / Groq)     │
│    ├─ semantic_log_search      (vector similarity over the store)  │
│    ├─ analyze_tool_latency     ┐                                   │
│    ├─ analyze_token_consumption├─ matplotlib/seaborn → charts/*.png│
│    ├─ analyze_error_frequency  ┘                                   │
│    └─ sync_knowledge_graph ───► Neo4j Aura DB                      │
│            (:Session)-[:TRIGGERED]->(:AgentAction)                 │
│              -[:ROUTED_TO]->(:MCPServerCall)-[:DEPENDS_ON]->(:…)   │
│    Streamlit UI: NL chat · reasoning trace · charts · graph commits│
└────────────────────────────────────────────────────────────────────┘
```

### Knowledge-graph schema

| Node | Meaning |
|---|---|
| `(:Session)` | One multi-turn execution trace (by `session_id`) |
| `(:AgentAction)` | Client-side decisions: ReAct tool calls + MCP sampling work |
| `(:MCPServerCall)` | Server-side tool / resource executions |

| Edge | Meaning |
|---|---|
| `[:TRIGGERED]` | Session → AgentAction |
| `[:ROUTED_TO]` | AgentAction → MCPServerCall (client tool routed to the server) |
| `[:DEPENDS_ON]` | MCPServerCall → AgentAction (server delegated back via Sampling) |

Inspect in the Aura console:

```cypher
MATCH (s:Session)-[:TRIGGERED]->(a:AgentAction)-[:ROUTED_TO]->(c:MCPServerCall)
RETURN s, a, c LIMIT 50;
```

---

## Deliverable artifacts (generated at runtime)

| File | Produced by |
|---|---|
| `mcp_agent_system.log` | `agent-client` — flat dual-stream `[CLIENT]`/`[SERVER]` log |
| `mcp_agent_log.db` | `agent-client` — vector-enabled SQLite log store |
| `analysis_agent.log` | `analysis-agent` / dashboard — analysis execution log |
| `charts/*.png` | trend tools — latency / token / error charts |

---

## Screenshots

Evidence of a successful end-to-end run. The **primary deliverable is the Streamlit
dashboard** captured in a real browser as a **full-window** screenshot showing the
**executing machine's clock** — the dashboard renders its own `🕒 Executing-machine time`
banner at the top of the page, so the timestamp is visible both in-app and in the OS menu
bar. See [`screenshots/`](screenshots/) for the capture checklist and the full index.

| # | Capture | Shows |
|---|---|---|
| 1 | `01-dashboard-full-ui.png` | Full Streamlit dashboard (clock banner, sidebar, reasoning trace, trend chart, diagnosis) — OS clock visible |
| 2 | `02-dashboard-graph-sync.png` | Dashboard Neo4j sync notification (committed nodes/edges) — OS clock visible |
| 3 | `03-agent-client-trace.png` | `agent_client` multi-turn tool-calling trace |
| 4 | `04-mcp-server-protocol.png` | `mcp_server` streamable-http protocol + MCP Sampling round-trips |
| 5 | `05-neo4j-aura-graph.png` | Neo4j Aura projected `(:Session)-[:TRIGGERED]->(:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)` graph |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Connection refused` on port 8000 | Start the MCP server (Terminal 1) first |
| `GROQ_API_KEY` / `TAVILY_API_KEY` not set | Load `.env` in the terminal before running |
| Neo4j sync returns `not_configured` | Set `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` in `.env` |
| `Unable to retrieve routing information` (Neo4j) | Check the `neo4j+s://` URI and that the Aura instance is **Running** |
| First run is slow | One-time `fastembed` model download (~80 MB), then cached |
| `tool_use_failed` (Groq schema) | Numeric tool args are string-tolerant; retry the query |
| `413 TPM exceeded` (Groq free tier) | Wait 60s and retry — prompts are kept compact |

---

## Stage 2 (carried forward)

The Stage 2 pipeline is documented in `REFLECTION.md`. The agent — now built with the
modern `create_agent` factory (native tool-calling, no legacy `create_react_agent` /
`AgentExecutor`) — gathers facts via Tavily, retrieves graded domain knowledge via the
CRAG resource, and verifies its draft via the 2-stage Reflection tool — all over
`streamable-http`, with the server holding no API keys (every LLM call, including the
Tree-of-Thought CRAG grading, is delegated back to the client via MCP Sampling).
