"""
agent.py
────────
The decoupled Log Analysis Agent — Stage 4 edgeless edition.

Stage 3 built this agent with LangChain's ``create_agent`` factory. Stage 4
guts that topology and rebuilds it as an **edgeless LangGraph StateGraph**
(see ``edgeless_agent.py``): a single ``START → initial_ingest_node`` edge, with
every other hop routed dynamically via ``Command(goto=...)``. This module is now
a thin façade so the Streamlit dashboard and CLI share one construction idiom.

Capabilities (Stage 3 + Stage 4):
  • semantic_log_search        — vector similarity over the MCP logs
  • analyze_* trend tools      — latency / token / error charts
  • sync_knowledge_graph       — Neo4j Aura causal projection
  • explainability audit       — graph-context + proxy SHAP/LIME (Stage 4)

Run
  uv run --package analysis-dashboard analysis-agent
"""

from __future__ import annotations

import os

from .edgeless_agent import build_edgeless_graph, run_edgeless
from .logger import analysis_log

# Best-effort .env load (uv run does not auto-load it).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def build_analysis_agent():
    """Compile and return the edgeless StateGraph Log Analysis Agent."""
    analysis_log.info("Building edgeless Log Analysis Agent (StateGraph + Command routing)")
    return build_edgeless_graph()


def run_analysis(agent, query: str, audit_session: str | None = None,
                 audit_target: str | None = None) -> dict:
    """Invoke the edgeless graph on one request and log its dynamic hop trace."""
    analysis_log.info("=" * 70)
    analysis_log.info(f"Analysis request: {query!r}")
    state = run_edgeless(agent, query, audit_session, audit_target)

    hops = " → ".join(state.get("visited", []))
    analysis_log.info(f"Edgeless route: {hops}")
    for step in state.get("trace", []):
        analysis_log.debug(f"  [{step.get('node')}] {step.get('summary')}")
    analysis_log.info(f"Final analysis answer: {str(state.get('final', ''))[:300]}")
    return state


# Demo queries exercise routing across every node, ending with an XAI audit.
DEMO_ANALYSIS_QUERIES = [
    "Search the logs for any slow or anomalous MCP sampling interactions and summarise what you find.",
    "Analyze tool latency trends and token consumption, and tell me which component is the heaviest.",
    "Check the error frequency across components, then sync the knowledge graph to Neo4j and report exactly which nodes and edges were committed.",
    "Run an explainability audit on the most recent failure mode and explain which feature and token most influenced it.",
]


def main() -> None:
    """CLI entry point — drives the edgeless graph over a diagnostic demo."""
    analysis_log.info("Log Analysis Agent starting up (edgeless StateGraph, decoupled process)")
    analysis_log.info(f"Reading log store: {os.environ.get('MCP_LOG_DB_PATH', 'mcp_agent_log.db')}")

    agent = build_analysis_agent()
    for query in DEMO_ANALYSIS_QUERIES:
        state = run_analysis(agent, query)
        sep = "═" * 70
        print(f"\n{sep}\nQUERY: {query}")
        print(f"ROUTE: {' → '.join(state.get('visited', []))}\n{sep}")
        print(state.get("final", "(no response)"))

    analysis_log.info("Analysis session complete — logs → analysis_agent.log")


if __name__ == "__main__":
    main()
