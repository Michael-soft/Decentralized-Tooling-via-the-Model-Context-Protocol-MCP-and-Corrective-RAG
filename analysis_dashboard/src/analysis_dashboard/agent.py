"""
agent.py
────────
The decoupled Log Analysis Agent.

Built with LangChain's `create_agent` factory and equipped with:
  • semantic_log_search   — vector similarity search over the MCP logs
  • sync_knowledge_graph  — project causal paths into Neo4j Aura DB
  • analyze_tool_latency  — latency moving-average trend chart
  • analyze_token_consumption — token-usage chart
  • analyze_error_frequency   — error-frequency chart

This process is entirely separate from the operational MCP client/server; it
only shares the on-disk SQLite vector store (read-only). Run it directly for a
CLI demo, or drive it from the Streamlit dashboard.

Run
  uv run --package analysis-dashboard analysis-agent
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

from .graph_mapper import make_graph_tool
from .log_retrieval import semantic_log_search
from .logger import analysis_log
from .trend_tools import TREND_TOOLS

# Best-effort .env load (uv run does not auto-load it).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


SYSTEM_PROMPT = (
    "You are the Log Analysis Agent — an autonomous observability diagnostician "
    "for a distributed Model Context Protocol (MCP) system. You investigate the "
    "persisted interaction traces of an MCP client/server and surface anomalies, "
    "performance trends, and causal relationships.\n\n"
    "TOOLS:\n"
    "• semantic_log_search — find log entries by meaning (anomalies, errors, slow paths).\n"
    "• sync_knowledge_graph — project the client→server→sampling execution paths into "
    "the Neo4j Aura DB property graph. Use when asked to map/sync/build the graph.\n"
    "• analyze_tool_latency / analyze_token_consumption / analyze_error_frequency — "
    "compute operational metrics and generate charts (saved to disk).\n\n"
    "GUIDELINES:\n"
    "1. Prefer semantic_log_search to ground answers in real log evidence.\n"
    "2. When asked about performance, latency, tokens, or errors, call the matching "
    "analytics tool and reference the generated chart path.\n"
    "3. When asked to map/update the knowledge graph, call sync_knowledge_graph and "
    "report exactly which nodes and edges were committed.\n"
    "4. Be concise, cite the evidence (namespaces, scores, counts), and end with a "
    "clear diagnostic conclusion."
)


def build_llm() -> ChatGroq:
    """Tool-calling LLM for the analysis agent's reasoning loop."""
    return ChatGroq(
        model=os.environ.get("ANALYSIS_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=1500,
    )


def build_analysis_agent():
    """Assemble the Log Analysis Agent graph via create_agent."""
    from langchain.agents import create_agent

    tools = [semantic_log_search, make_graph_tool(), *TREND_TOOLS]
    analysis_log.info(f"Building Log Analysis Agent | tools={[t.name for t in tools]}")
    return create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def run_analysis(agent, query: str) -> dict:
    """Invoke the agent on a single question and log its structural reasoning."""
    analysis_log.info("=" * 70)
    analysis_log.info(f"Analysis query: {query!r}")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    messages = result.get("messages", [])
    for msg in messages:
        role = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                analysis_log.info(f"TOOL CALL | {tc['name']} args={tc.get('args')}")
        elif role == "ToolMessage":
            analysis_log.debug(f"TOOL RESULT | {getattr(msg, 'name', '?')} | "
                               f"{str(msg.content)[:200]}")
        elif getattr(msg, "content", None):
            analysis_log.debug(f"{role} | {str(msg.content)[:200]}")

    final = messages[-1].content if messages else ""
    analysis_log.info(f"Final analysis answer: {str(final)[:300]}")
    return result


# Demo queries exercise every tool so analysis_agent.log is well-populated.
DEMO_ANALYSIS_QUERIES = [
    "Search the logs for any slow or anomalous MCP sampling interactions and summarise what you find.",
    "Analyze tool latency trends and token consumption, and tell me which component is the heaviest.",
    "Check the error frequency across components, then sync the knowledge graph to Neo4j and report exactly which nodes and edges were committed.",
]


def main() -> None:
    """CLI entry point — runs a diagnostic demo over the persisted logs."""
    analysis_log.info("Log Analysis Agent starting up (decoupled process)")
    analysis_log.info(f"Reading log store: {os.environ.get('MCP_LOG_DB_PATH', 'mcp_agent_log.db')}")

    agent = build_analysis_agent()
    for query in DEMO_ANALYSIS_QUERIES:
        result = run_analysis(agent, query)
        final = result.get("messages", [])[-1].content if result.get("messages") else ""
        sep = "═" * 70
        print(f"\n{sep}\nQUERY: {query}\n{sep}")
        print(final)

    analysis_log.info("Analysis session complete — logs → analysis_agent.log")


if __name__ == "__main__":
    main()
