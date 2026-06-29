# Stage 4 — Execution Screenshots

Evidence of a successful **explainability audit execution loop** on the upgraded
edgeless Streamlit XAI control room. Each capture is a full-window browser shot;
the dashboard renders its own `🕒 Executing-machine time` banner so the timestamp
is visible in-app alongside the OS menu-bar clock.

| # | File(s) | Shows |
|---|---|---|
| 1 | `01-audit-execution-loop.png`, `01-audit-execution-loop-2.png` … `-4.png` | **Explainability Audit Hub** after running an Audit Report — proxy-SHAP horizontal bar chart, proxy-LIME token annotations, graph-relational context, and the audit summary. |
| 2 | `02-resilience-panel.png`, `02-resilience-panel-2.png` | Sidebar **Resilience tracking** panel — historical fallback / self-healing / retry / hardcoded counts read from the `resilience.*` log namespace. |
| 3 | `03-edgeless-diagnostics.png`, `03-edgeless-diagnostics-2.png` … `-7.png` | **Diagnostics** tab — the edgeless `Command(goto)` hop trace (`initial_ingest_node → … → synthesis_node`) plus rendered trend charts. |

> Terminal-side resilience evidence (retry → self-heal → hardcoded recoveries) and the
> MCP streamable-http / Sampling round-trips are captured in the root `mcp_agent_system.log`
> and `mcp_agent_log.db` deliverables rather than as screenshots.

## How these were captured

1. `set -a && . ./.env && set +a && uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py`
2. Opened <http://localhost:8501> in a real browser (not the IDE preview).
3. Ran an Explainability Audit Report from the Audit Hub, exercised the Diagnostics
   tab, and refreshed the resilience panel — capturing the full window with the OS
   clock visible in each shot.
