"""
graph_context.py
────────────────
Stage 4 — Graph-Relational Context Hydration for the explainability engine.

The audit pipeline cross-references BOTH storage systems built in Stage 3:
it reads target trace IDs from the vector store, then extracts the adjacent
structural subgraph (parent ``(:Session)``, sibling ``(:AgentAction)`` states,
and subsequent ``(:MCPServerCall)`` executions) so the relationship context can
be appended as raw prompt injection for the SHAP/LIME explainability layer.

Two extraction paths:
  • Live Neo4j Aura  — runs Cypher against the projected property graph.
  • Log-derived      — when Aura is unreachable/unconfigured, the same
                       adjacency is reconstructed from the SQLite log metadata
                       via ``graph_mapper.build_graph_payload`` so audits still
                       run fully offline (graceful degradation).
"""

from __future__ import annotations

from typing import Any, Optional

from .graph_mapper import build_graph_payload, neo4j_config
from .logger import analysis_log
from .store_reader import LOG_ROOT, get_reader


# ─────────────────────────────────────────────────────────────────────────────
#  Trace discovery (read target trace IDs from the vector store)
# ─────────────────────────────────────────────────────────────────────────────
def list_sessions(limit: int = 2000) -> list[dict[str, Any]]:
    """Return distinct session_ids with interaction + error counts."""
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT,), limit=limit)
    sessions: dict[str, dict[str, Any]] = {}
    for rec in entries:
        e = rec["entry"]
        sid = e.get("session_id", "unknown")
        s = sessions.setdefault(sid, {"session_id": sid, "interactions": 0, "errors": 0})
        s["interactions"] += 1
        if e.get("status") == "error":
            s["errors"] += 1
    return sorted(sessions.values(), key=lambda s: s["errors"], reverse=True)


def find_failure_traces(limit: int = 2000) -> list[dict[str, Any]]:
    """
    Return individual trace IDs (log keys) whose status is 'error' — the
    targeted runtime failure modes the audit explains.
    """
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT,), limit=limit)
    traces = []
    for rec in entries:
        e = rec["entry"]
        if e.get("status") == "error":
            traces.append(
                {
                    "key": rec["key"],
                    "namespace": ".".join(rec["namespace"]),
                    "session_id": e.get("session_id"),
                    "component": e.get("component"),
                    "tool_name": e.get("tool_name"),
                    "error": e.get("error"),
                    "content_preview": (e.get("content", "") or "")[:160],
                }
            )
    return traces


def resilience_stats(limit: int = 2000) -> dict[str, Any]:
    """
    Count historical resilience events recorded by the operational client under
    the ``("logs","resilience",*)`` namespace — fallback activations, self-
    healing reinjections, retries, and hardcoded absolute fallbacks. Powers the
    dashboard's Resilience Tracking panel (Stage 4 §5).
    """
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT, "resilience"), limit=limit)
    counts = {"fallbacks": 0, "self_healing": 0, "retries": 0, "hardcoded": 0, "total": 0}
    recent: list[dict[str, Any]] = []
    kind_map = {
        "fallback_activation": "fallbacks",
        "self_healing_iteration": "self_healing",
        "retry_attempt": "retries",
        "hardcoded_fallback": "hardcoded",
    }
    for rec in entries:
        e = rec["entry"]
        kind = e.get("tool_name", "")
        bucket = kind_map.get(kind)
        if bucket:
            counts[bucket] += 1
            counts["total"] += 1
            recent.append(
                {
                    "kind": kind,
                    "component": e.get("component"),
                    "timestamp": e.get("timestamp"),
                    "content": (e.get("content", "") or "")[:160],
                }
            )
    recent.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    counts["recent"] = recent[:15]
    return counts


def entries_for_session(session_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    """All log records belonging to one session, in stored order."""
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT,), limit=limit)
    return [r for r in entries if r["entry"].get("session_id") == session_id]


# ─────────────────────────────────────────────────────────────────────────────
#  Subgraph extraction
# ─────────────────────────────────────────────────────────────────────────────
_CYPHER_SUBGRAPH = """
MATCH (s:Session {id: $sid})
OPTIONAL MATCH (s)-[:TRIGGERED]->(a:AgentAction)
OPTIONAL MATCH (a)-[:ROUTED_TO]->(c:MCPServerCall)
OPTIONAL MATCH (c2:MCPServerCall)-[:DEPENDS_ON]->(a)
RETURN s.id AS session_id,
       collect(DISTINCT a {.key, .name, .component, .interaction_type,
                           .latency_ms, .status}) AS agent_actions,
       collect(DISTINCT c {.key, .name, .component, .latency_ms, .status}) AS routed_calls,
       collect(DISTINCT c2 {.key, .name, .component}) AS sampling_calls
"""


def _neo4j_subgraph(session_id: str) -> Optional[dict[str, Any]]:
    """Query Neo4j Aura for the session's adjacent subgraph; None on failure."""
    cfg = neo4j_config()
    if cfg is None:
        return None
    try:
        from neo4j import GraphDatabase

        analysis_log.info(f"Graph hydration → Neo4j Aura | session={session_id}")
        driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
        try:
            driver.verify_connectivity()
            with driver.session(database=cfg["database"]) as sess:
                rec = sess.run(_CYPHER_SUBGRAPH, sid=session_id).single()
        finally:
            driver.close()
        if not rec:
            return None
        agent_actions = [a for a in rec["agent_actions"] if a]
        routed_calls = [c for c in rec["routed_calls"] if c]
        sampling_calls = [c for c in rec["sampling_calls"] if c]
        return {
            "source": "neo4j_aura",
            "session_id": session_id,
            "session_present": True,
            "agent_actions": agent_actions,
            "server_calls": routed_calls,
            "sampling_dependencies": sampling_calls,
            "counts": {
                "agent_actions": len(agent_actions),
                "server_calls": len(routed_calls),
                "sampling_dependencies": len(sampling_calls),
            },
        }
    except Exception as exc:
        analysis_log.warning(f"Neo4j hydration failed ({exc}) — using log-derived subgraph")
        return None


def _log_derived_subgraph(session_id: str) -> dict[str, Any]:
    """Reconstruct the session's adjacency from SQLite log metadata."""
    analysis_log.info(f"Graph hydration → log-derived | session={session_id}")
    entries = entries_for_session(session_id)
    payload = build_graph_payload(entries)
    agent_nodes = [n for n in payload["agent_nodes"] if n["session_id"] == session_id]
    server_nodes = [n for n in payload["server_nodes"] if n["session_id"] == session_id]
    return {
        "source": "log_derived",
        "session_id": session_id,
        "session_present": session_id in {s["id"] for s in payload["sessions"]},
        "agent_actions": [
            {k: n.get(k) for k in ("key", "name", "component", "interaction_type",
                                   "latency_ms", "status")}
            for n in agent_nodes
        ],
        "server_calls": [
            {k: n.get(k) for k in ("key", "name", "component", "latency_ms", "status")}
            for n in server_nodes
        ],
        "sampling_dependencies": [
            {"key": d["agent_key"], "server_key": d["server_key"]}
            for d in payload["depends_on"]
        ],
        "edges": {
            "TRIGGERED": len([t for t in payload["triggered"]
                              if t["session_id"] == session_id]),
            "ROUTED_TO": len(payload["routed"]),
            "DEPENDS_ON": len(payload["depends_on"]),
        },
        "counts": {
            "agent_actions": len(agent_nodes),
            "server_calls": len(server_nodes),
            "sampling_dependencies": len(payload["depends_on"]),
        },
    }


def fetch_graph_context(session_id: str) -> dict[str, Any]:
    """
    Extract the structural subgraph adjacent to a session/trace.

    Prefers live Neo4j Aura; transparently falls back to a log-derived
    reconstruction so the explainability audit always has graph context.
    """
    sub = _neo4j_subgraph(session_id)
    if sub is None:
        sub = _log_derived_subgraph(session_id)
    sub["context_text"] = format_graph_context(sub)
    return sub


def format_graph_context(sub: dict[str, Any]) -> str:
    """Render the subgraph as a raw prompt-injection context block."""
    lines = [
        "── GRAPH-RELATIONAL CONTEXT (subgraph hydration) ──",
        f"source: {sub.get('source')}  session: {sub.get('session_id')}",
        f"parent (:Session) present: {sub.get('session_present')}",
        f"adjacent (:AgentAction) states: {sub['counts']['agent_actions']}",
        f"subsequent (:MCPServerCall) executions: {sub['counts']['server_calls']}",
        f"(:DEPENDS_ON) sampling delegations: {sub['counts']['sampling_dependencies']}",
    ]
    for a in sub.get("agent_actions", [])[:8]:
        lines.append(
            f"  • AgentAction {a.get('name')} [{a.get('component')}] "
            f"status={a.get('status')} latency={a.get('latency_ms')}ms"
        )
    for c in sub.get("server_calls", [])[:8]:
        lines.append(
            f"  • MCPServerCall {c.get('name')} [{c.get('component')}] "
            f"status={c.get('status')}"
        )
    return "\n".join(lines)
