"""
edgeless_agent.py
─────────────────
Stage 4 — Edgeless LangGraph StateGraph Log Analysis Agent.

The Stage 3 ``create_agent`` topology is gutted and rebuilt as an *edgeless*
``StateGraph``: there is exactly ONE structural transition —
``builder.add_edge(START, "initial_ingest_node")``. Every subsequent hop is
performed dynamically: each functional node computes its own destination and
returns a ``langgraph.types.Command(goto=..., update=...)``, passing control
peer-to-peer with no hardcoded ``add_edge`` map between processing nodes.

Routing model
─────────────
``initial_ingest_node`` classifies the request into a dynamic *plan* (an
ordered queue of node names). Each functional node does its work, pops the next
hop off the plan, and ``goto``s it (or ``synthesis_node`` when the plan drains).
``synthesis_node`` composes the final diagnosis and ``goto``s ``END``. The plan
lives in graph state and is recomputed per request, so control flow is data-
driven rather than wired.
"""

from __future__ import annotations

import json
import os
import re
from operator import add
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .explainability import (
    render_shap_chart,
    run_explainability_audit,
    write_audit_report,
)
from .graph_mapper import sync_knowledge_graph_impl
from .log_retrieval import semantic_log_search
from .logger import analysis_log
from .trend_tools import (
    analyze_error_frequency,
    analyze_token_consumption,
    analyze_tool_latency,
)

# ── Node names (string constants; NOT used to build static edges) ────────────
INGEST = "initial_ingest_node"
SEARCH = "semantic_search_node"
TREND = "trend_analysis_node"
GRAPH = "graph_sync_node"
XAI = "explainability_node"
SYNTH = "synthesis_node"


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class AnalysisState(TypedDict, total=False):
    """Concurrently-updated graph state shared across edgeless hops."""

    query: str
    plan: list[str]                       # dynamic routing queue (peer-to-peer)
    intent: dict[str, bool]
    audit_session: Optional[str]
    audit_target: Optional[str]
    visited: Annotated[list[str], add]    # ordered hop trace
    trace: Annotated[list[dict], add]     # per-node rendered summary
    findings: Annotated[dict, _merge]     # tool/audit results keyed by node
    final: str


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic intent → plan classifier (computed in-node, not wired)
# ─────────────────────────────────────────────────────────────────────────────
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_KEY_RE = re.compile(r"\b\d{13}-[0-9a-zA-Z]{2,8}(?:-[0-9a-zA-Z]{4,8})?\b")


def _classify(query: str) -> tuple[list[str], dict[str, bool]]:
    q = query.lower()
    wants_audit = any(k in q for k in (
        "audit", "explain", "why", "shap", "lime", "importance", "feature",
        "interpret", "attribut", "explainab", "root cause", "post-hoc",
    ))
    wants_search = any(k in q for k in (
        "search", "find", "anomal", "slow", "look for", "investigate", "trace", "log",
    ))
    wants_trend = any(k in q for k in (
        "latency", "token", "error frequency", "trend", "performance",
        "consumption", "chart", "metric", "throughput",
    ))
    wants_graph = any(k in q for k in (
        "graph", "neo4j", "sync", "map ", "project", "knowledge graph", "aura",
    ))

    if wants_audit:
        plan = [SEARCH, XAI]
    else:
        plan = []
        if wants_search:
            plan.append(SEARCH)
        if wants_trend:
            plan.append(TREND)
        if wants_graph:
            plan.append(GRAPH)
        if not plan:
            plan = [SEARCH, TREND]
    return plan, {
        "search": wants_search, "trend": wants_trend,
        "graph": wants_graph, "audit": wants_audit,
    }


def _route(state: AnalysisState, this_node: str, findings: dict,
           trace_summary: str, plan_override: Optional[list[str]] = None) -> Command:
    """Pop the next hop off the dynamic plan and goto it (peer-to-peer)."""
    plan = list(state.get("plan", []) if plan_override is None else plan_override)
    nxt = plan.pop(0) if plan else SYNTH
    analysis_log.info(f"[edgeless] {this_node} → goto={nxt} | plan_remaining={plan}")
    return Command(
        goto=nxt,
        update={
            "plan": plan,
            "visited": [this_node],
            "trace": [{"node": this_node, "summary": trace_summary}],
            "findings": findings,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Nodes — each returns Command(goto=...) computed dynamically
# ─────────────────────────────────────────────────────────────────────────────
def initial_ingest_node(state: AnalysisState) -> Command:
    """The single START target: classify intent, build the plan, route onward."""
    query = state.get("query", "")
    plan, intent = _classify(query)
    sess = state.get("audit_session") or (
        _UUID_RE.search(query).group(0) if _UUID_RE.search(query) else None
    )
    target = state.get("audit_target") or (
        _KEY_RE.search(query).group(0) if _KEY_RE.search(query) else None
    )
    analysis_log.info(f"[edgeless] ingest | intent={intent} | plan={plan} | session={sess}")
    nxt = plan[0] if plan else SYNTH
    return Command(
        goto=nxt,
        update={
            "plan": plan[1:],
            "intent": intent,
            "audit_session": sess,
            "audit_target": target,
            "visited": [INGEST],
            "trace": [{"node": INGEST, "summary": f"intent={intent}; plan={plan}"}],
            "findings": {},
        },
    )


def semantic_search_node(state: AnalysisState) -> Command:
    """Vector-search the logs for evidence, then route dynamically."""
    raw = semantic_log_search.invoke({"query": state.get("query", ""), "limit": 6})
    try:
        n = len(json.loads(raw).get("matches", []))
    except Exception:
        n = 0
    return _route(state, SEARCH, {"semantic_search": raw},
                  f"semantic_log_search → {n} matches")


def trend_analysis_node(state: AnalysisState) -> Command:
    """Run the operational trend tools relevant to the intent, then route."""
    intent = state.get("intent", {})
    out: dict[str, Any] = {}
    q = state.get("query", "").lower()
    # Always produce latency + error views; tokens when asked or generic.
    out["latency"] = analyze_tool_latency.invoke({"window": "3"})
    out["errors"] = analyze_error_frequency.invoke({})
    if "token" in q or not intent.get("trend"):
        out["tokens"] = analyze_token_consumption.invoke({})
    return _route(state, TREND, {"trends": out},
                  f"trend tools run: {list(out.keys())}")


def graph_sync_node(state: AnalysisState) -> Command:
    """Project the logs into the Neo4j causal graph, then route."""
    result = sync_knowledge_graph_impl()
    return _route(state, GRAPH, {"graph_sync": result},
                  f"graph sync → {result.get('status')}")


def explainability_node(state: AnalysisState) -> Command:
    """Run the local explainability audit (SHAP/LIME + graph context), then route."""
    report = run_explainability_audit(
        session_id=state.get("audit_session"),
        target_key=state.get("audit_target"),
    )
    chart = ""
    report_path = ""
    if report.get("proxy_shap"):
        chart = render_shap_chart(report)
        report_path = write_audit_report(report)
    summary = report.get("summary", report.get("message", "audit produced no target"))
    return _route(
        state, XAI,
        {"explainability": report, "shap_chart": chart, "report_path": report_path},
        f"explainability audit → {summary[:90]}",
    )


def synthesis_node(state: AnalysisState) -> Command:
    """Compose the final diagnosis from all findings, then goto END."""
    findings = state.get("findings", {})
    final = _synthesize(state.get("query", ""), findings)
    return Command(
        goto=END,
        update={
            "visited": [SYNTH],
            "trace": [{"node": SYNTH, "summary": "final diagnosis composed"}],
            "final": final,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Synthesis (LLM with deterministic fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _build_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.environ.get("ANALYSIS_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=900,
        max_retries=int(os.environ.get("MAX_RETR_ATTEMPTS", "3")),
    )


def _compact_findings(findings: dict) -> str:
    parts: list[str] = []
    if "semantic_search" in findings:
        parts.append(f"SEMANTIC SEARCH:\n{str(findings['semantic_search'])[:900]}")
    if "trends" in findings:
        for k, v in findings["trends"].items():
            parts.append(f"TREND[{k}]:\n{str(v)[:500]}")
    if "graph_sync" in findings:
        parts.append(f"GRAPH SYNC:\n{json.dumps(findings['graph_sync'])[:500]}")
    if "explainability" in findings:
        rep = findings["explainability"]
        parts.append(
            "EXPLAINABILITY AUDIT:\n"
            + json.dumps(
                {
                    "failure_mode": rep.get("failure_mode"),
                    "proxy_shap": rep.get("proxy_shap"),
                    "proxy_lime": {
                        "most_influential_token": rep.get("proxy_lime", {}).get("most_influential_token"),
                        "top": rep.get("proxy_lime", {}).get("token_importances", [])[:5],
                    },
                    "graph_context": rep.get("graph_context", {}).get("counts"),
                },
                indent=2,
            )[:1100]
        )
    return "\n\n".join(parts) or "(no findings)"


def _synthesize(query: str, findings: dict) -> str:
    compact = _compact_findings(findings)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = _build_llm()
        resp = llm.invoke(
            [
                SystemMessage(content=(
                    "You are the edgeless Log Analysis Agent. Produce a concise, "
                    "evidence-grounded diagnostic answer. Cite SHAP/LIME drivers and "
                    "graph context when present. End with a clear conclusion."
                )),
                HumanMessage(content=f"USER REQUEST:\n{query}\n\nFINDINGS:\n{compact}"),
            ]
        )
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        analysis_log.warning(f"synthesis LLM failed ({exc}) — deterministic summary")
        return f"Diagnostic summary (LLM unavailable: {exc}).\n\n{compact}"


# ─────────────────────────────────────────────────────────────────────────────
#  Graph construction — EXACTLY ONE static edge: START → initial_ingest_node
# ─────────────────────────────────────────────────────────────────────────────
def build_edgeless_graph():
    """Compile the edgeless StateGraph. Hops happen via Command(goto), not edges."""
    builder = StateGraph(AnalysisState)
    builder.add_node(INGEST, initial_ingest_node)
    builder.add_node(SEARCH, semantic_search_node)
    builder.add_node(TREND, trend_analysis_node)
    builder.add_node(GRAPH, graph_sync_node)
    builder.add_node(XAI, explainability_node)
    builder.add_node(SYNTH, synthesis_node)

    # The ONLY structural transition rule in the entire compilation.
    builder.add_edge(START, INGEST)

    analysis_log.info(
        "Edgeless StateGraph compiled | 1 static edge (START→initial_ingest_node); "
        "all other hops via Command(goto)"
    )
    return builder.compile()


def run_edgeless(graph, query: str, audit_session: Optional[str] = None,
                 audit_target: Optional[str] = None) -> dict[str, Any]:
    """Invoke the edgeless graph and return the terminal state."""
    return graph.invoke(
        {
            "query": query,
            "plan": [],
            "findings": {},
            "audit_session": audit_session,
            "audit_target": audit_target,
        }
    )
