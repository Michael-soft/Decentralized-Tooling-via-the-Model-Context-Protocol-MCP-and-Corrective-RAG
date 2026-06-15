"""
main.py
───────
Stage 2 — AI Agent (MCP Client) entrypoint.

Initialises a LangChain agent via the modern ``create_agent`` factory
(LangChain 1.x / LangGraph) equipped with three tools:
  1. tavily_search               — real-world web grounding (local Tavily).
  2. remote_crag_tool            — hierarchical CRAG resource over MCP HTTP.
  3. remote_reflection_tool      — 2-stage critique + correction over MCP HTTP.

The agent uses native tool-calling (not text-format ReAct) — the same
``create_agent`` factory used by the decoupled analysis dashboard, so both
processes share one agent-construction idiom.

All MCP communication uses streamable-http transport. The sampling handler
in mcp_client.py ensures the server's Reflection tool delegates LLM calls
back to this client's Groq model — the server itself holds no API keys.

Run
  uv run --package agent-client agent-client
  # or directly:
  python -m agent_client.main
"""

from __future__ import annotations

import json
import os
import time
import uuid

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from .log_store import LogEntry, estimate_tokens, get_log_store
from .logger import client_log
from .mcp_client import remote_crag_tool, remote_reflection_tool
from .session import get_session_id

# Best-effort .env loading (uv run does not auto-load it).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm() -> ChatGroq:
    """Fast, deterministic tool-calling LLM for the agent's reasoning loop."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────────────────────

def _build_tools() -> list:
    """
    Assemble the agent's tool registry.

    Tool order signals priority to the agent:
      1. tavily_search          — gather real-world facts first.
      2. remote_crag_tool       — domain knowledge with ToT grading.
      3. remote_reflection_tool — verify draft before the final answer.
    """
    tavily = TavilySearch(
        max_results=2,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        description=(
            "Search the web for current, factual information. Use for recent events, "
            "statistics, or anything requiring up-to-date real-world data."
        ),
    )
    return [tavily, remote_crag_tool, remote_reflection_tool]


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a Thinking Agent connected to a remote MCP server, equipped with "
    "three tools. Reason step by step and call tools to ground every answer.\n\n"
    "For complex queries follow this order:\n"
    "  1. tavily_search       — gather live, real-world facts.\n"
    "  2. remote_crag_tool    — retrieve graded domain knowledge.\n"
    "  3. remote_reflection_tool — verify and correct your draft before answering.\n\n"
    "Call remote_reflection_tool once you have a draft, passing your draft as "
    "`draft_answer` and the user's question as `original_query`. After it returns, "
    "write your final answer immediately and concisely."
)


def build_agent():
    """
    Construct and return a ready-to-run agent.

    Uses LangChain's modern ``create_agent`` factory (LangGraph runtime) with
    native tool-calling. The system prompt encodes the meta-cognitive tool
    ordering above. Returns a compiled graph whose ``.invoke`` takes
    ``{"messages": [...]}`` and returns ``{"messages": [...]}``.
    """
    llm   = _build_llm()
    tools = _build_tools()

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tool_steps(messages: list) -> list[tuple[str, Any, str]]:
    """
    Flatten a create_agent message list into ordered (tool, args, observation)
    triples — the modern equivalent of AgentExecutor's intermediate_steps.

    Tool calls live on AIMessage.tool_calls; their results arrive as
    ToolMessage objects keyed by tool_call_id.
    """
    results_by_id: dict[str, str] = {}
    for msg in messages:
        if type(msg).__name__ == "ToolMessage":
            results_by_id[getattr(msg, "tool_call_id", "")] = str(getattr(msg, "content", ""))

    steps: list[tuple[str, Any, str]] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            observation = results_by_id.get(tc.get("id", ""), "")
            steps.append((name, args, observation))
    return steps


def run_query(agent, query: str) -> dict:
    """
    Invoke the agent on a single query and emit structured log output.

    Args:
        agent : Compiled create_agent graph.
        query : Natural language question from the user.

    Returns:
        The full result dict including the message list.
    """
    client_log.info(f"Query submitted to agent: {query!r}")
    store = get_log_store()

    # Session-scoped marker entry — becomes the (:Session) anchor in Neo4j.
    store.record(
        LogEntry(
            session_id=get_session_id(),
            mcp_interaction_type="tool_invocation",
            content=f"SESSION QUERY: {query}",
            component="agent.session",
            tool_name="session_start",
            token_estimate=estimate_tokens(query),
        ),
        store.namespace("agent", "session"),
        f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
    )

    start  = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    total_ms = (time.perf_counter() - start) * 1000.0

    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else ""
    client_log.info("Agent completed execution")
    client_log.info(f"Final Answer: {final_answer}")

    # Log a structured trace of every tool call + persist each as an AgentAction.
    steps = _extract_tool_steps(messages)
    client_log.info(f"Total tool calls: {len(steps)}")
    per_step_ms = total_ms / max(len(steps), 1)
    for i, (tool_name, args, observation) in enumerate(steps, 1):
        obs_preview = str(observation)[:200]
        client_log.debug(
            f"Step {i} | tool={tool_name} | "
            f"input={str(args)[:120]} | "
            f"obs={obs_preview}"
        )
        # Persist the agent's decision to invoke this tool (graph: AgentAction).
        store.record(
            LogEntry(
                session_id=get_session_id(),
                mcp_interaction_type="tool_invocation",
                content=(
                    f"AGENT ACTION step={i} tool={tool_name}\n"
                    f"INPUT: {str(args)[:400]}\n"
                    f"OBSERVATION: {str(observation)[:800]}"
                ),
                component=f"agent.planning.{tool_name}",
                tool_name=tool_name,
                target=tool_name,
                latency_ms=round(per_step_ms, 2),
                token_estimate=estimate_tokens(str(observation)),
            ),
            store.namespace("agent", "planning", tool_name),
            f"{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:6]}",
        )

    return result


def print_trace(result: dict) -> None:
    """Pretty-print the agent trace to stdout for submission evidence."""
    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else ""

    sep = "═" * 70
    print(f"\n{sep}")
    print("                      FINAL ANSWER")
    print(sep)
    print(final_answer)
    print(sep)

    steps = _extract_tool_steps(messages)
    print(f"\n{sep}")
    print("                  TOOL-CALL TRACE")
    print(sep)
    for i, (tool_name, args, observation) in enumerate(steps, 1):
        print(f"\n─── Step {i} {'─'*50}")
        print(f"  TOOL  : {tool_name}")
        print(f"  INPUT : {json.dumps(args, indent=2) if isinstance(args, dict) else str(args)[:300]}")
        obs = str(observation)
        print(f"  OBS   : {obs[:600]}{'...' if len(obs) > 600 else ''}")
    print(f"\n{sep}")
    print(f"Total tool calls executed: {len(steps)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUERY = (
    "What is the Model Context Protocol (MCP) and how does it differ from "
    "standard LangChain tool use? Which approach is better suited for "
    "enterprise-grade agentic systems requiring modularity and observability?"
)

# A complex, multi-turn test execution (deliverable: mcp_agent_system.log).
# Each query exercises the full client→server→sampling path so the trace is
# rich enough for the analysis agent to map and trend.
DEMO_QUERIES = [
    DEMO_QUERY,
    (
        "Explain Corrective RAG (CRAG) and Tree-of-Thought grading. How does "
        "hierarchical retrieval improve answer quality over naive RAG?"
    ),
    (
        "Why is observability critical for distributed multi-agent MCP systems, "
        "and how do structured vector logs improve anomaly detection?"
    ),
]


def main() -> None:
    """uv script entry point — runs a complex multi-turn observability trace."""
    client_log.info("=" * 70)
    client_log.info("Stage 3 Thinking Agent starting up (vector-logged)")
    client_log.info(f"Session ID  : {get_session_id()}")
    client_log.info(f"MCP Server URL: {os.environ.get('MCP_SERVER_URL', 'http://localhost:8000/mcp')}")
    client_log.info(f"Log DB        : {os.environ.get('MCP_LOG_DB_PATH', 'mcp_agent_log.db')}")
    client_log.info("=" * 70)

    # Warm the store (and trigger the FastEmbed model load) up front.
    get_log_store()

    agent = build_agent()

    multi = os.environ.get("MCP_MULTI_TURN", "1") not in ("0", "false", "False")
    queries = DEMO_QUERIES if multi else [DEMO_QUERY]

    for turn, query in enumerate(queries, 1):
        client_log.info("─" * 70)
        client_log.info(f"TURN {turn}/{len(queries)} | session={get_session_id()}")
        result = run_query(agent, query)
        print_trace(result)

    client_log.info("Agent session complete — flat logs → mcp_agent_system.log, "
                    "vector logs → mcp_agent_log.db")


if __name__ == "__main__":
    main()
