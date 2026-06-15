"""
mcp_client.py
─────────────
FastMCP client layer for the Stage 2 agent.

Responsibilities
  1. Connect to the FastMCP server via streamable-http.
  2. Handle MCP Sampling requests from the server — execute them using
     the local Groq LLM and return generated text back over the transport.
  3. Forward server log notifications to the dual-stream logger (server_log).
  4. Expose two LangChain @tool-decorated wrappers the ReAct agent calls:
       • remote_reflection_tool — calls server reflection_tool over MCP/HTTP.
       • remote_crag_tool       — reads server CRAG resource over MCP/HTTP.

Design notes
  • Pydantic models (ReflectionRequest, ReflectionResponse, CRAGResponse) are
    defined at the top and ACTIVELY USED inside each @tool for input validation
    and output parsing — not just as documentation stubs.
  • _sampling_llm is lazy-initialised so GROQ_API_KEY is read at call-time.
  • FastMCP Client context manager returns the same client; use directly.
  • LangChain @tool functions are synchronous; _run_async() bridges asyncio.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any
from urllib.parse import quote

from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.sampling import SamplingMessage, SamplingParams
from langchain_core.tools import tool
from mcp.shared.context import RequestContext
from pydantic import BaseModel, Field

from .log_store import LogEntry, estimate_tokens, get_log_store
from .logger import client_log, server_log
from .session import get_session_id


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic models — structured contracts for inter-tool data
#  Defined at module top; actively used inside @tool wrappers below.
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionRequest(BaseModel):
    """Validated input contract for remote_reflection_tool."""
    draft_answer:   str = Field(description="The agent's current draft answer to critique.")
    original_query: str = Field(description="The user's original question — truth anchor.")

    def to_tool_input(self) -> str:
        """Serialize to the single JSON string the ReAct @tool expects."""
        return self.model_dump_json()


class ReflectionResponse(BaseModel):
    """Validated output contract from the server's reflection_tool."""
    critique:         str  = Field(description="Detailed critique of the draft answer.")
    is_sufficient:    bool = Field(description="True if no correction was needed.")
    corrected_answer: str  = Field(description="Improved, fact-grounded answer.")

    @classmethod
    def parse_tool_output(cls, raw: str) -> "ReflectionResponse":
        """Parse and validate the JSON string returned by the server tool."""
        try:
            return cls(**json.loads(raw))
        except Exception:
            return cls(
                critique="Parse failed — raw output returned",
                is_sufficient=False,
                corrected_answer=raw,
            )


class CRAGResponse(BaseModel):
    """Validated output contract from the server's CRAG resource."""
    query:            str       = Field(description="Original query string.")
    expanded_queries: list[str] = Field(default_factory=list)
    tot_scores:       dict      = Field(default_factory=dict)
    avg_tot_score:    float     = Field(default=0.0)
    fallback_used:    bool      = Field(default=False)
    combined_context: str       = Field(default="")

    @classmethod
    def parse_resource_output(cls, raw: str) -> "CRAGResponse":
        """Parse and validate the JSON payload from the CRAG resource."""
        try:
            return cls(**json.loads(raw))
        except Exception:
            return cls(query="unknown", combined_context=raw)


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy client factory — reads MCP_SERVER_URL at call-time, not import-time
# ─────────────────────────────────────────────────────────────────────────────

_client_instance: Client | None = None


def _get_client() -> Client:
    """Return the singleton FastMCP Client, creating it on first call."""
    global _client_instance
    if _client_instance is None:
        url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")
        client_log.info(f"Initialising FastMCP Client → {url}")
        _client_instance = Client(
            transport        = url,
            sampling_handler = sampling_handler,
            log_handler      = log_handler,
            auto_initialize  = False,
        )
    return _client_instance


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy LLM factory — ChatGroq constructed on first sampling call
# ─────────────────────────────────────────────────────────────────────────────

_sampling_llm_instance = None


def _get_sampling_llm():
    """Return the singleton Groq LLM used exclusively for sampling requests."""
    global _sampling_llm_instance
    if _sampling_llm_instance is None:
        from langchain_groq import ChatGroq
        _sampling_llm_instance = ChatGroq(
            model       = "llama-3.3-70b-versatile",
            temperature = 0.3,
            max_tokens  = 512,
        )
        client_log.info("Sampling LLM initialised: llama-3.3-70b-versatile")
    return _sampling_llm_instance


# ─────────────────────────────────────────────────────────────────────────────
#  Vector-store persistence helper
#  Every MCP interaction is mirrored into the hierarchical vector log store
#  (Stage 3) in addition to the flat dual-stream file (Stage 2). Failures here
#  must never break the operational agent — they are swallowed and logged.
# ─────────────────────────────────────────────────────────────────────────────

def _persist(
    *,
    interaction_type: str,
    content: str,
    namespace: tuple[str, ...],
    component: str,
    tool_name: str | None = None,
    target: str | None = None,
    request_id: str | None = None,
    latency_ms: float = 0.0,
    status: str = "success",
    error: str | None = None,
    requested_schema: dict | None = None,
) -> None:
    """Build a validated LogEntry and write it to the hierarchical store."""
    try:
        entry = LogEntry(
            session_id=get_session_id(),
            mcp_interaction_type=interaction_type,           # type: ignore[arg-type]
            content=content,
            component=component,
            tool_name=tool_name,
            target=target,
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
            token_estimate=estimate_tokens(content),
            status=status,                                   # type: ignore[arg-type]
            error=error,
            requested_schema=requested_schema or {},
            # Data-guardrail demonstrator: an integer-keyed config map. JSON will
            # stringify these keys; the store round-trip casts them back to int.
            config_map={0: interaction_type, 1: component, 2: status},
        )
        key = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        get_log_store().record(entry, namespace, key)
    except Exception as exc:
        client_log.error(f"_persist failed ({interaction_type}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Callback Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def sampling_handler(
    messages: list[SamplingMessage],
    params:   SamplingParams,
    context:  RequestContext,
) -> str:
    """
    Handle MCP Sampling requests routed from the server's reflection_tool.

    The server constructs Critique/Correction prompts then delegates LLM
    execution here via ctx.session.create_message(). This handler invokes
    the local Groq LLM and returns the generated text back over the transport.
    The server holds no API keys — all LLM calls originate here.
    """
    client_log.info(
        f"Sampling request received from server | request_id={context.request_id}"
    )
    system = (
        getattr(params, "systemPrompt", None)
        or getattr(params, "system_prompt", None)
    )
    human_parts: list[str] = []
    for msg in messages:
        content = msg.content
        if hasattr(content, "text"):
            human_parts.append(content.text)
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "text"):
                    human_parts.append(block.text)
        else:
            human_parts.append(str(content))

    from langchain_core.messages import HumanMessage, SystemMessage
    lc_messages: list = []
    if system:
        lc_messages.append(SystemMessage(content=system))
    human_text = "\n".join(human_parts)
    lc_messages.append(HumanMessage(content=human_text))

    # Capture any nested requestedSchema / structural metadata the server sent.
    try:
        req_schema = params.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    except Exception:
        req_schema = {"systemPrompt": system}
    req_id = str(getattr(context, "request_id", ""))

    start = time.perf_counter()
    try:
        llm       = _get_sampling_llm()
        response  = await llm.ainvoke(lc_messages)
        generated = response.content
        latency   = (time.perf_counter() - start) * 1000.0
        client_log.info(f"Sampling response generated | tokens≈{len(generated.split())}")
        _persist(
            interaction_type="sampling_request",
            content=f"PROMPT: {human_text}\n\nRESPONSE: {generated}",
            namespace=get_log_store().namespace("mcp", "client", "sampling"),
            component="mcp.client.sampling",
            tool_name="create_message",
            target="agent_client.groq_llm",
            request_id=req_id,
            latency_ms=latency,
            requested_schema=req_schema,
        )
        return generated
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000.0
        client_log.error(f"Sampling handler error: {exc}")
        _persist(
            interaction_type="sampling_request",
            content=f"PROMPT: {human_text}\n\nERROR: {exc}",
            namespace=get_log_store().namespace("mcp", "client", "sampling"),
            component="mcp.client.sampling",
            tool_name="create_message",
            request_id=req_id,
            latency_ms=latency,
            status="error",
            error=str(exc),
            requested_schema=req_schema,
        )
        return f"[Sampling error: {exc}]"


async def log_handler(message: LogMessage) -> None:
    """
    Forward MCP server log notifications into the dual-stream logger.
    Server ctx.info() / ctx.debug() calls arrive here and are written
    to both stdout and agent_system.log with a [SERVER] prefix.
    """
    import logging as _logging
    level_map  = _logging.getLevelNamesMapping()
    level_name = message.level.upper() if hasattr(message, "level") else "INFO"
    level_int  = level_map.get(level_name, _logging.INFO)
    msg_text   = (
        message.data.get("msg", str(message.data))
        if isinstance(message.data, dict)
        else str(message.data)
    )
    server_log.log(level_int, msg_text)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level async helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_server_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Open a FastMCP Client session, call a tool, return text content."""
    client_log.info(f"Calling server tool '{tool_name}'")
    mcp_client = _get_client()
    start = time.perf_counter()
    status, error = "success", None
    text = ""
    try:
        async with mcp_client:
            await mcp_client.initialize()
            result = await mcp_client.call_tool(tool_name, args)
        if isinstance(result, list) and result:
            block = result[0]
            text = block.text if hasattr(block, "text") else str(block)
        else:
            text = str(result)
        return text
    except Exception as exc:
        status, error, text = "error", str(exc), f"[tool error: {exc}]"
        raise
    finally:
        _persist(
            interaction_type="tool_invocation",
            content=f"TOOL {tool_name} args={json.dumps(args)[:500]}\nRESULT: {text[:1500]}",
            namespace=get_log_store().namespace("mcp", "server", "tools", tool_name),
            component=f"mcp.server.tools.{tool_name}",
            tool_name=tool_name,
            target="mcp_server",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            status=status,
            error=error,
        )


async def _read_server_resource(uri: str) -> str:
    """Open a FastMCP Client session, read a resource URI, return text content."""
    client_log.info(f"Reading server resource '{uri}'")
    mcp_client = _get_client()
    start = time.perf_counter()
    status, error = "success", None
    text = ""
    try:
        async with mcp_client:
            await mcp_client.initialize()
            result = await mcp_client.read_resource(uri)
        if isinstance(result, list) and result:
            block = result[0]
            text = block.text if hasattr(block, "text") else str(block)
        else:
            text = str(result)
        return text
    except Exception as exc:
        status, error, text = "error", str(exc), f"[resource error: {exc}]"
        raise
    finally:
        _persist(
            interaction_type="resource_read",
            content=f"RESOURCE {uri}\nRESULT: {text[:1500]}",
            namespace=get_log_store().namespace("mcp", "server", "resources", "crag"),
            component="mcp.server.resources.crag",
            tool_name="crag_knowledge_resource",
            target="mcp_server",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            status=status,
            error=error,
        )


def _run_async(coro) -> str:
    """
    Bridge an async coroutine into a synchronous LangChain @tool call.
    Uses ThreadPoolExecutor if an event loop is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain @tool wrappers — Pydantic used for input validation + output parsing
# ─────────────────────────────────────────────────────────────────────────────

@tool
def remote_reflection_tool(draft_answer: str, original_query: str) -> str:
    """
    Remote 2-stage Reflection via the MCP server (streamable-http).

    Validates input with ReflectionRequest, calls the server reflection_tool,
    then parses and validates the response with ReflectionResponse.
    The server delegates both LLM calls back to this client's Groq model
    via MCP Sampling — the server holds no API keys.

    WHEN TO USE:
      - After forming a preliminary answer from search or CRAG retrieval.
      - Before writing the final answer on any factual or high-stakes query.

    Args:
        draft_answer:   Your current best answer to be critiqued.
        original_query: The user's original question (the truth anchor).

    Returns:
        corrected_answer string after critique and correction.
    """
    client_log.info("remote_reflection_tool invoked by agent")

    # ── Pydantic input validation ──────────────────────────────────────────────
    # Native tool-calling delivers typed args; ReflectionRequest validates them
    # and re-serialises the single JSON string the server tool contract expects.
    req       = ReflectionRequest(draft_answer=draft_answer, original_query=original_query)
    validated = req.to_tool_input()
    client_log.debug(f"ReflectionRequest validated | query='{req.original_query[:60]}'")

    # ── Remote tool call over MCP/HTTP ─────────────────────────────────────────
    raw_result = _run_async(
        _call_server_tool("reflection_tool", {"tool_input": validated})
    )

    # ── Pydantic output validation ─────────────────────────────────────────────
    response = ReflectionResponse.parse_tool_output(raw_result)
    client_log.info(
        f"ReflectionResponse received | is_sufficient={response.is_sufficient}"
    )

    # Return the corrected answer as plain text for the ReAct agent
    return response.corrected_answer


@tool
def remote_crag_tool(query: str) -> str:
    """
    Hierarchical CRAG knowledge retrieval via the MCP server resource.

    Reads knowledge://domain/{query} which triggers on the server:
    (1) Multi-query expansion — 3 semantic variants via MCP Sampling.
    (2) Hierarchical retrieval — L1 topic routing → L2/L3 chunk scoring.
    (3) Tree-of-Thought grading — 3-branch LLM reasoning via MCP Sampling
        (Specificity, Completeness, Novelty rated per branch then voted).
    (4) Tavily fallback — if avg ToT score < 0.6 or no chunks found.
    Response is validated with CRAGResponse before returning to the agent.

    WHEN TO USE:
      - Domain questions on: agentic AI, MCP, RAG/CRAG, LangChain,
        Groq, ReAct, Tree-of-Thought, observability, FastMCP.

    Args:
        query: Natural language question about the knowledge domain.

    Returns:
        combined_context string (internal chunks + optional web results).
    """
    client_log.info(f"remote_crag_tool invoked | query='{query}'")

    encoded = quote(query, safe="")
    uri     = f"knowledge://domain/{encoded}"
    raw     = _run_async(_read_server_resource(uri))

    # ── Pydantic output validation ─────────────────────────────────────────────
    crag = CRAGResponse.parse_resource_output(raw)
    client_log.info(
        f"CRAGResponse validated | fallback_used={crag.fallback_used} "
        f"| avg_tot={crag.avg_tot_score} | queries={len(crag.expanded_queries)}"
    )

    return crag.combined_context
