"""
graph_mapper.py
───────────────
Neo4j Aura DB causal knowledge-graph projection.

Reads the structured metadata persisted by the operational MCP client and
projects the asynchronous client→server→sampling execution paths into a
Neo4j property graph using idempotent, batched (UNWIND) Cypher.

Property graph schema (Stage 3 guardrail)
─────────────────────────────────────────
Nodes
  (:Session      {id, ...})            — one multi-turn execution trace
  (:AgentAction  {key, name, ...})     — client-side decisions / sampling work
  (:MCPServerCall{key, name, ...})     — server-side tool / resource executions
Edges
  (:Session)-[:TRIGGERED]->(:AgentAction)        — the agent acted within a session
  (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)  — a client tool routed to the server
  (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction) — server call delegated back via Sampling

If Neo4j credentials are absent the mapper degrades gracefully: it reports a
clear "not configured" status instead of emitting a broken connection string.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from .logger import analysis_log
from .store_reader import LOG_ROOT, get_reader

# Maps a client-side AgentAction tool to the server-side execution it invokes.
TOOL_ROUTING = {
    "remote_reflection_tool": "reflection_tool",
    "remote_crag_tool": "crag_knowledge_resource",
}

# Server calls that delegate work back to the client LLM via MCP Sampling.
SERVER_CALLS_USING_SAMPLING = {"reflection_tool", "crag_knowledge_resource"}


def _parse_ts(value: str) -> float:
    """Parse an ISO-8601 timestamp to epoch seconds; 0.0 on failure."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


def neo4j_config() -> Optional[dict[str, str]]:
    """Return Neo4j connection config from env, or None if not configured."""
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip()
    pwd = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not (uri and user and pwd):
        return None
    return {"uri": uri, "user": user, "password": pwd,
            "database": os.environ.get("NEO4J_DATABASE", "neo4j")}


# ─────────────────────────────────────────────────────────────────────────────
#  Build node/edge lists from log metadata (pure, testable, no DB needed)
# ─────────────────────────────────────────────────────────────────────────────
def build_graph_payload(entries: list[dict[str, Any]]) -> dict[str, list]:
    """
    Transform raw log entries into node + edge batches for Cypher UNWIND.

    Classification:
      • component starts with "mcp.server"          → MCPServerCall
      • interaction_type == "sampling_request"      → AgentAction (client LLM work)
      • otherwise (component starts with "agent")   → AgentAction
    """
    sessions: set[str] = set()
    agent_nodes: dict[str, dict] = {}
    server_nodes: dict[str, dict] = {}

    for rec in entries:
        e = rec["entry"]
        key = rec["key"]
        sid = e.get("session_id", "unknown")
        sessions.add(sid)

        node = {
            "key": key,
            "session_id": sid,
            "name": e.get("tool_name") or e.get("component", "unknown"),
            "component": e.get("component", ""),
            "interaction_type": e.get("mcp_interaction_type", ""),
            "latency_ms": float(e.get("latency_ms", 0.0) or 0.0),
            "token_estimate": int(e.get("token_estimate", 0) or 0),
            "status": e.get("status", "success"),
            "timestamp": e.get("timestamp", ""),
            "request_id": e.get("request_id") or "",
        }

        component = e.get("component", "")
        if component.startswith("mcp.server"):
            server_nodes[key] = node
        else:
            # agent.* and sampling requests are client-side AgentActions
            agent_nodes[key] = node

    # ── Edges ────────────────────────────────────────────────────────────────
    # Matching is causal, not cartesian: routing is paired in execution order
    # per (session, tool) and sampling dependencies attach to the nearest
    # server call in time. This keeps the graph 1:1 instead of exploding.
    triggered = [
        {"session_id": n["session_id"], "key": n["key"]} for n in agent_nodes.values()
    ]

    def _ts(node: dict) -> float:
        return _parse_ts(node.get("timestamp", ""))

    # ROUTED_TO — pair the i-th client tool invocation with the i-th matching
    # server execution, per session, in chronological order.
    routed: list[dict] = []
    sessions_seen = {n["session_id"] for n in agent_nodes.values()}
    for sid in sessions_seen:
        for client_tool, server_name in TOOL_ROUTING.items():
            agents = sorted(
                (a for a in agent_nodes.values()
                 if a["session_id"] == sid and a["name"] == client_tool),
                key=_ts,
            )
            servers = sorted(
                (s for s in server_nodes.values()
                 if s["session_id"] == sid and s["name"] == server_name),
                key=_ts,
            )
            for a, s in zip(agents, servers):
                routed.append({"agent_key": a["key"], "server_key": s["key"]})

    # DEPENDS_ON — each sampling request attaches to the temporally nearest
    # server call (in the same session) that delegated work back to the client.
    depends_on: list[dict] = []
    sampling_actions = [
        a for a in agent_nodes.values() if a["interaction_type"] == "sampling_request"
    ]
    for smp in sampling_actions:
        candidates = [
            s for s in server_nodes.values()
            if s["session_id"] == smp["session_id"]
            and s["name"] in SERVER_CALLS_USING_SAMPLING
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda s: abs(_ts(s) - _ts(smp)))
        depends_on.append({"server_key": nearest["key"], "agent_key": smp["key"]})

    return {
        "sessions": [{"id": sid} for sid in sessions],
        "agent_nodes": list(agent_nodes.values()),
        "server_nodes": list(server_nodes.values()),
        "triggered": triggered,
        "routed": routed,
        "depends_on": depends_on,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Cypher projection
# ─────────────────────────────────────────────────────────────────────────────
_CYPHER = [
    # Sessions
    """
    UNWIND $sessions AS s
    MERGE (:Session {id: s.id})
    """,
    # Agent actions + TRIGGERED
    """
    UNWIND $agent_nodes AS a
    MERGE (n:AgentAction {key: a.key})
    SET n.name = a.name, n.component = a.component,
        n.interaction_type = a.interaction_type, n.latency_ms = a.latency_ms,
        n.token_estimate = a.token_estimate, n.status = a.status,
        n.timestamp = a.timestamp, n.session_id = a.session_id
    WITH a, n
    MATCH (s:Session {id: a.session_id})
    MERGE (s)-[:TRIGGERED]->(n)
    """,
    # Server calls
    """
    UNWIND $server_nodes AS c
    MERGE (n:MCPServerCall {key: c.key})
    SET n.name = c.name, n.component = c.component,
        n.interaction_type = c.interaction_type, n.latency_ms = c.latency_ms,
        n.token_estimate = c.token_estimate, n.status = c.status,
        n.timestamp = c.timestamp, n.session_id = c.session_id
    """,
    # AgentAction -[:ROUTED_TO]-> MCPServerCall
    """
    UNWIND $routed AS r
    MATCH (a:AgentAction {key: r.agent_key})
    MATCH (c:MCPServerCall {key: r.server_key})
    MERGE (a)-[:ROUTED_TO]->(c)
    """,
    # MCPServerCall -[:DEPENDS_ON]-> AgentAction (sampling)
    """
    UNWIND $depends_on AS d
    MATCH (c:MCPServerCall {key: d.server_key})
    MATCH (a:AgentAction {key: d.agent_key})
    MERGE (c)-[:DEPENDS_ON]->(a)
    """,
]


def project_to_neo4j(payload: dict[str, list]) -> dict[str, Any]:
    """Execute the batched Cypher projection. Returns a commit summary."""
    cfg = neo4j_config()
    if cfg is None:
        analysis_log.warning("Neo4j not configured — skipping live projection")
        return {
            "status": "not_configured",
            "message": "Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD to enable graph sync.",
            "planned_nodes": len(payload["sessions"])
            + len(payload["agent_nodes"])
            + len(payload["server_nodes"]),
            "planned_edges": len(payload["triggered"])
            + len(payload["routed"])
            + len(payload["depends_on"]),
        }

    from neo4j import GraphDatabase

    analysis_log.info(f"Connecting to Neo4j Aura → {cfg['uri']}")
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    try:
        driver.verify_connectivity()
        with driver.session(database=cfg["database"]) as session:
            for stmt in _CYPHER:
                session.run(stmt, **payload)
        summary = {
            "status": "committed",
            "uri": cfg["uri"],
            "nodes": {
                "Session": len(payload["sessions"]),
                "AgentAction": len(payload["agent_nodes"]),
                "MCPServerCall": len(payload["server_nodes"]),
            },
            "edges": {
                "TRIGGERED": len(payload["triggered"]),
                "ROUTED_TO": len(payload["routed"]),
                "DEPENDS_ON": len(payload["depends_on"]),
            },
        }
        summary["nodes_total"] = sum(summary["nodes"].values())
        summary["edges_total"] = sum(summary["edges"].values())
        analysis_log.info(
            f"Neo4j projection committed | nodes={summary['nodes_total']} "
            f"edges={summary['edges_total']}"
        )
        return summary
    except Exception as exc:
        analysis_log.error(f"Neo4j projection failed: {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        driver.close()


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain tool
# ─────────────────────────────────────────────────────────────────────────────
def sync_knowledge_graph_impl() -> dict[str, Any]:
    """Read all logs, build the causal graph payload, and project to Neo4j."""
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT,), limit=2000)
    analysis_log.info(f"sync_knowledge_graph | {len(entries)} log entries read")
    payload = build_graph_payload(entries)
    return project_to_neo4j(payload)


def make_graph_tool():
    """Factory so the agent and dashboard share one tool definition."""
    from langchain_core.tools import tool

    @tool
    def sync_knowledge_graph() -> str:
        """
        Project the persisted MCP interaction logs into the Neo4j Aura DB
        causal knowledge graph and report what was committed.

        Builds (:Session)-[:TRIGGERED]->(:AgentAction)-[:ROUTED_TO]->
        (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction) from the structural
        log metadata. Use this whenever the user asks to map, sync, build,
        or update the knowledge graph of system interactions.

        Returns:
            JSON commit summary: node/edge counts per label and overall status.
        """
        import json

        result = sync_knowledge_graph_impl()
        return json.dumps(result, indent=2)

    return sync_knowledge_graph
