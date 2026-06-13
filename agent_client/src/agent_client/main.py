"""
main.py
───────
Stage 2 — AI Agent (MCP Client) entrypoint.

Initialises a LangChain ReAct agent (create_react_agent factory) equipped
with three tools:
  1. tavily_search               — real-world web grounding (local Tavily).
  2. remote_crag_tool            — hierarchical CRAG resource over MCP HTTP.
  3. remote_reflection_tool      — 2-stage critique + correction over MCP HTTP.

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

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langsmith import Client as LangSmithClient

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
    """Fast, deterministic outer-loop LLM for the ReAct Thought/Action cycle."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────────────────────

def _build_tools() -> list:
    """
    Assemble the agent's tool registry.

    Tool order signals priority to the ReAct agent:
      1. tavily_search          — gather real-world facts first.
      2. remote_crag_tool       — domain knowledge with ToT grading.
      3. remote_reflection_tool — verify draft before Final Answer.
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

_META_DIRECTIVE = (
    "You are a Thinking Agent connected to a remote MCP server. "
    "For complex queries follow this order: "
    "(1) tavily_search — gather live facts, "
    "(2) remote_crag_tool — retrieve graded domain knowledge, "
    "(3) remote_reflection_tool — verify your draft before answering. "
    "After remote_reflection_tool returns, write Final Answer immediately."
)


def build_agent_executor() -> AgentExecutor:
    """
    Construct and return a ready-to-run AgentExecutor.

    Uses LangChain's create_react_agent factory with the hwchase17/react
    base prompt augmented by the meta-cognitive directive above.
    """
    llm   = _build_llm()
    tools = _build_tools()

    base_prompt: PromptTemplate = LangSmithClient().pull_prompt(
        "hwchase17/react",
        dangerously_pull_public_prompt=True,
    )
    augmented_template = base_prompt.template.replace(
        "Begin!", f"{_META_DIRECTIVE}\n\nBegin!"
    )
    react_prompt = PromptTemplate.from_template(augmented_template)

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    return AgentExecutor(
        agent                     = agent,
        tools                     = tools,
        verbose                   = True,
        handle_parsing_errors     = True,
        max_iterations            = 12,
        max_execution_time        = 180,
        return_intermediate_steps = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_query(executor: AgentExecutor, query: str) -> dict:
    """
    Invoke the agent on a single query and emit structured log output.

    Args:
        executor : Configured AgentExecutor instance.
        query    : Natural language question from the user.

    Returns:
        The full result dict including intermediate_steps.
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
    result = executor.invoke({"input": query})
    total_ms = (time.perf_counter() - start) * 1000.0

    client_log.info("Agent completed execution")
    client_log.info(f"Final Answer: {result.get('output', '')}")

    # Log a structured trace of every step + persist each as an AgentAction.
    steps = result.get("intermediate_steps", [])
    client_log.info(f"Total intermediate steps: {len(steps)}")
    per_step_ms = total_ms / max(len(steps), 1)
    for i, (action, observation) in enumerate(steps, 1):
        obs_preview = str(observation)[:200]
        client_log.debug(
            f"Step {i} | tool={action.tool} | "
            f"input={str(action.tool_input)[:120]} | "
            f"obs={obs_preview}"
        )
        # Persist the agent's decision to invoke this tool (graph: AgentAction).
        store.record(
            LogEntry(
                session_id=get_session_id(),
                mcp_interaction_type="tool_invocation",
                content=(
                    f"AGENT ACTION step={i} tool={action.tool}\n"
                    f"INPUT: {str(action.tool_input)[:400]}\n"
                    f"OBSERVATION: {str(observation)[:800]}"
                ),
                component=f"agent.planning.{action.tool}",
                tool_name=action.tool,
                target=action.tool,
                latency_ms=round(per_step_ms, 2),
                token_estimate=estimate_tokens(str(observation)),
            ),
            store.namespace("agent", "planning", action.tool),
            f"{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:6]}",
        )

    return result


def print_trace(result: dict) -> None:
    """Pretty-print the ReAct trace to stdout for submission evidence."""
    sep = "═" * 70
    print(f"\n{sep}")
    print("                      FINAL ANSWER")
    print(sep)
    print(result.get("output", ""))
    print(sep)

    print(f"\n{sep}")
    print("                  INTERMEDIATE STEPS TRACE")
    print(sep)
    for i, (action, observation) in enumerate(
        result.get("intermediate_steps", []), 1
    ):
        print(f"\n─── Step {i} {'─'*50}")
        print(f"  TOOL  : {action.tool}")
        inp = action.tool_input
        print(f"  INPUT : {json.dumps(inp, indent=2) if isinstance(inp, dict) else str(inp)[:300]}")
        obs = str(observation)
        print(f"  OBS   : {obs[:600]}{'...' if len(obs) > 600 else ''}")
    print(f"\n{sep}")
    print(f"Total steps executed: {len(result.get('intermediate_steps', []))}")


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

    executor = build_agent_executor()

    multi = os.environ.get("MCP_MULTI_TURN", "1") not in ("0", "false", "False")
    queries = DEMO_QUERIES if multi else [DEMO_QUERY]

    for turn, query in enumerate(queries, 1):
        client_log.info("─" * 70)
        client_log.info(f"TURN {turn}/{len(queries)} | session={get_session_id()}")
        result = run_query(executor, query)
        print_trace(result)

    client_log.info("Agent session complete — flat logs → mcp_agent_system.log, "
                    "vector logs → mcp_agent_log.db")


if __name__ == "__main__":
    main()
