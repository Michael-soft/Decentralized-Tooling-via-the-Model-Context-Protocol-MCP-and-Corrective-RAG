"""
log_retrieval.py
────────────────
Semantic retrieval tool for the Log Analysis Agent.

Exposes a LangChain `@tool` that queries the shared LangGraph SQLite vector
store by similarity, so the agent can surface semantic anomalies, traces, or
unmapped errors across execution histories — without knowing exact keywords.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .logger import analysis_log
from .store_reader import LOG_ROOT, get_reader


@tool
def semantic_log_search(query: str, limit: int = 6) -> str:
    """
    Semantic vector search over the persisted MCP interaction logs.

    Use this to investigate the system's execution history by MEANING rather
    than exact text — e.g. "slow reflection sampling", "CRAG resource errors",
    "token-heavy turns", "anomalous server calls".

    Args:
        query: Natural-language description of what to look for.
        limit: Max number of matching log entries to return (default 6).

    Returns:
        JSON string: a ranked list of matching log entries with similarity
        score, namespace, interaction type, component, latency, and status.
    """
    analysis_log.info(f"semantic_log_search | query={query!r} | limit={limit}")
    reader = get_reader()
    hits = reader.semantic_search(query, namespace_prefix=(LOG_ROOT,), limit=limit)

    summary = []
    for h in hits:
        e = h["entry"]
        summary.append(
            {
                "score": round(h["score"], 3) if h.get("score") is not None else None,
                "namespace": ".".join(h["namespace"]),
                "session_id": e.get("session_id"),
                "interaction_type": e.get("mcp_interaction_type"),
                "component": e.get("component"),
                "tool_name": e.get("tool_name"),
                "latency_ms": e.get("latency_ms"),
                "status": e.get("status"),
                "content_preview": (e.get("content", "")[:240]),
            }
        )
    analysis_log.info(f"semantic_log_search | {len(summary)} hits returned")
    return json.dumps({"query": query, "matches": summary}, indent=2)
