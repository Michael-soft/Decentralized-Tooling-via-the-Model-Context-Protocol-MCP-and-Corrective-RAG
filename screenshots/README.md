# Stage 3 — Execution Screenshots

Evidence of a successful end-to-end run of the distributed MCP system and the
decoupled **Streamlit** Log Analysis dashboard.

> **Recapture checklist (addresses grading feedback).** The dashboard captures
> must (a) show the **Streamlit dashboard itself** — not the VS Code preview
> pane or a terminal — (b) capture the **full browser window UI** (sidebar +
> chat + reasoning trace + chart), and (c) show the **executing machine's
> clock**. The dashboard now renders a server-side **"Executing-machine time"**
> banner at the top of the page, so a single full-window browser screenshot
> carries the clock both in the app banner *and* in the OS menu bar.

## How to capture

1. Run all three processes (see the root `README.md`), then open the dashboard
   at <http://localhost:8501> **in a real browser** (Chrome/Safari/Firefox) —
   not the IDE's embedded preview.
2. Ask one of the example questions (e.g. *"Analyze tool latency trends and show
   me the chart."*) and let the agent finish so a chart + diagnosis render.
3. Capture the **entire browser window**, including the OS menu-bar clock, so
   both the in-app `🕒 Executing-machine time` banner and the OS clock are
   visible in the same frame.

## Required captures

| # | File | What it must show |
|---|---|---|
| 1 | `01-dashboard-full-ui.png` | **Full Streamlit dashboard** in the browser — title, the `🕒 Executing-machine time` banner, sidebar system status, the agent reasoning trace, a rendered trend chart, and the diagnosis. **OS clock visible.** |
| 2 | `02-dashboard-graph-sync.png` | Dashboard after a *"sync the knowledge graph to Neo4j"* query — the green `Neo4j sync committed — N nodes, M edges` notification with the node/edge breakdown. **OS clock visible.** |
| 3 | `03-agent-client-trace.png` | `agent_client` terminal — the multi-turn tool-calling trace ending with `Agent session complete`. |
| 4 | `04-mcp-server-protocol.png` | `mcp_server` terminal — `POST /mcp 200 OK`, `ListToolsRequest`, the `reflection_tool` schema, and MCP Sampling round-trips (including the ToT grading branches). |
| 5 | `05-neo4j-aura-graph.png` | Neo4j Aura console — the projected property graph: `(:Session)-[:TRIGGERED]->(:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)`. |

> Items 1–2 are the primary deliverable (the dashboard run). Items 3–5 are the
> supporting operational evidence.
