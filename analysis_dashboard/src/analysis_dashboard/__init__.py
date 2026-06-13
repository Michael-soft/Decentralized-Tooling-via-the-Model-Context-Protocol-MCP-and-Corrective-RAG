"""
analysis_dashboard
──────────────────
Stage 3 decoupled observability control-plane.

Components
  log_retrieval.py — semantic vector search @tool over the shared LangGraph store.
  graph_mapper.py  — Neo4j Aura DB causal knowledge-graph projection.
  trend_tools.py   — latency / token / error trend analytics + chart generation.
  agent.py         — Log Analysis Agent assembled via LangChain create_agent.
  app.py           — Streamlit human-in-the-loop diagnostic dashboard.

This package runs as an entirely separate process from the operational
MCP client/server. It only shares the on-disk SQLite log store (read-only).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
