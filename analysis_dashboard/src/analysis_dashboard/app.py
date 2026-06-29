"""
app.py
──────
Stage 4 — Edgeless XAI Analytical Control Room (Streamlit).

Upgrades the Stage 3 diagnostic dashboard with:
  • Diagnostics tab — natural-language chat driving the EDGELESS LangGraph
    analysis agent, rendering its dynamic Command-routed hop trace + charts.
  • Explainability Audit Hub — pick an execution track (session) or failure
    trace ID and run a localized "Explainability Audit Report":
      – step-by-step trace explanation,
      – proxy LIME token-importance text annotations,
      – proxy SHAP horizontal feature-weight bar chart.
  • Resilience Tracking panel — historical counts of how often the core MCP
    client fell back (RunnableWithFallbacks) or self-healed (LLM reinjection).
  • Executing-machine clock banner — server-side timestamp for screenshot evidence.

Run
  uv run --package analysis-dashboard streamlit run \
      analysis_dashboard/src/analysis_dashboard/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from analysis_dashboard.agent import build_analysis_agent
from analysis_dashboard.edgeless_agent import run_edgeless
from analysis_dashboard.explainability import (
    render_shap_chart,
    run_explainability_audit,
    write_audit_report,
)
from analysis_dashboard.graph_context import (
    find_failure_traces,
    list_sessions,
    resilience_stats,
)
from analysis_dashboard.graph_mapper import neo4j_config

st.set_page_config(page_title="Edgeless XAI Control Room", page_icon="🔭", layout="wide")


@st.cache_resource(show_spinner="Booting edgeless analysis graph + embedding model…")
def _get_agent():
    return build_analysis_agent()


# ─────────────────────────────────────────────────────────────────────────────
#  Renderers
# ─────────────────────────────────────────────────────────────────────────────
def _render_chart_if_present(content: str) -> None:
    try:
        data = json.loads(content)
    except Exception:
        return
    cp = data.get("chart_path")
    if cp and Path(cp).exists():
        st.image(cp, caption=cp, width="stretch")


def _render_findings(findings: dict) -> None:
    """Render charts / graph-sync notifications produced during a graph run."""
    trends = findings.get("trends", {})
    for _, payload in trends.items():
        _render_chart_if_present(payload)
    gs = findings.get("graph_sync")
    if gs:
        status = gs.get("status")
        if status == "committed":
            st.success(f"Neo4j sync committed — {gs.get('nodes_total',0)} nodes, "
                       f"{gs.get('edges_total',0)} edges")
        elif status == "not_configured":
            st.warning(f"Neo4j not configured — {gs.get('message','')}")
        else:
            st.error(f"Graph sync: {gs.get('message', status)}")
    if findings.get("explainability"):
        render_audit(findings["explainability"], chart_path=findings.get("shap_chart"))


def _lime_annotation(report: dict) -> None:
    """Text annotations of proxy-LIME token importances (Stage 4 §5)."""
    lime = report.get("proxy_lime", {})
    toks = lime.get("token_importances", [])
    st.markdown(f"**Proxy LIME** · model `{lime.get('model')}` · "
                f"base error-confidence `{lime.get('base_confidence')}` · "
                f"{lime.get('n_samples')} perturbations")
    if not toks:
        st.caption("No textual tokens to attribute for this trace.")
        return
    mx = max(abs(t["coefficient"]) for t in toks) or 1.0
    for t in toks:
        coef = t["coefficient"]
        bar = "█" * max(1, int(abs(coef) / mx * 18))
        arrow = "🔺" if coef >= 0 else "🔻"
        st.markdown(f"`{t['token']:<18}` {arrow} `{coef:+.4f}`  {bar}")


def render_audit(report: dict, chart_path: str | None = None) -> None:
    """Render a full explainability audit report in the UI."""
    if report.get("status") in ("no_data", "no_target"):
        st.warning(f"Audit could not run: {report.get('message')}")
        return

    st.markdown(f"#### 🎯 Targeted failure mode\n`{report.get('failure_mode')}`")
    tgt = report.get("target", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Session", str(report.get("session_id", ""))[:8])
    c2.metric("Component", str(tgt.get("component", "—")))
    c3.metric("Status", str(tgt.get("status", "—")))

    # Step-by-step / graph context
    gc = report.get("graph_context", {})
    with st.expander("🧭 Graph-relational context (subgraph hydration)", expanded=True):
        st.caption(f"source: {gc.get('source')} · adjacent nodes: "
                   f"{gc.get('counts', {})}")
        st.code(gc.get("context_text", ""), language="text")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("##### Proxy SHAP — structured feature influence")
        shap = report.get("proxy_shap", {})
        st.caption(f"model `{shap.get('model')}` · predicted error-confidence "
                   f"`{shap.get('predicted_error_proba')}` · most influential: "
                   f"**{shap.get('most_influential_feature')}**")
        cpath = chart_path or render_shap_chart(report)
        if cpath and Path(cpath).exists():
            st.image(cpath, caption=cpath, width="stretch")
        st.json(shap.get("values", {}))
    with colB:
        st.markdown("##### Proxy LIME — token importance")
        _lime_annotation(report)

    st.markdown("##### 🩺 Audit summary")
    st.info(report.get("summary", ""))


def _run_and_render(agent, query: str) -> None:
    """Drive the edgeless graph and stream its dynamic routing + artifacts."""
    with st.chat_message("assistant"):
        with st.spinner("Routing through the edgeless analysis graph…"):
            state = run_edgeless(agent, query)
        route = " → ".join(state.get("visited", []))
        st.caption(f"**Edgeless route (Command goto):** {route}")
        with st.expander("Dynamic hop trace", expanded=False):
            for step in state.get("trace", []):
                st.markdown(f"- **`{step.get('node')}`** — {step.get('summary')}")
        _render_findings(state.get("findings", {}))
        final = state.get("final", "(no response)")
        st.markdown("### 🩺 Diagnosis")
        st.markdown(final)
        st.session_state.history.append(("assistant", final))


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔭 Edgeless XAI Analytical Control Room")
st.caption("Stage 4 — edgeless LangGraph analysis agent · graph-contextual explainability audits")

_now = datetime.now().astimezone()
st.info(
    f"🕒 **Executing-machine time:** {_now.strftime('%a %d %b %Y, %H:%M:%S %Z')}  "
    f"·  run rendered live by Streamlit"
)

# ── Sidebar: system status + resilience tracking ─────────────────────────────
with st.sidebar:
    st.header("System status")
    db_path = os.environ.get("MCP_LOG_DB_PATH", "mcp_agent_log.db")
    st.metric("Log store", "online" if Path(db_path).exists() else "missing")
    st.text(f"DB: {db_path}")
    cfg = neo4j_config()
    if cfg:
        st.success(f"Neo4j Aura configured\n{cfg['uri']}")
    else:
        st.warning("Neo4j not configured\n(graph hydration falls back to log-derived)")

    st.divider()
    st.subheader("🛡️ Resilience tracking")
    if st.button("Refresh resilience stats", width="stretch"):
        st.session_state.pop("_res", None)
    if "_res" not in st.session_state:
        try:
            st.session_state["_res"] = resilience_stats()
        except Exception as exc:
            st.session_state["_res"] = {"error": str(exc)}
    res = st.session_state["_res"]
    if "error" in res:
        st.caption(f"(unavailable: {res['error'][:80]})")
    else:
        r1, r2 = st.columns(2)
        r1.metric("Fallback activations", res.get("fallbacks", 0))
        r2.metric("Self-healing loops", res.get("self_healing", 0))
        r3, r4 = st.columns(2)
        r3.metric("Retry attempts", res.get("retries", 0))
        r4.metric("Hardcoded fallbacks", res.get("hardcoded", 0))
        with st.expander("Recent resilience events"):
            for ev in res.get("recent", []):
                st.caption(f"`{ev['kind']}` · {ev['component']} · {ev['timestamp'][:19]}")

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

agent = _get_agent()

tab_diag, tab_audit = st.tabs(["💬 Diagnostics", "🔬 Explainability Audit Hub"])

# ── Tab 1: edgeless diagnostics chat ─────────────────────────────────────────
with tab_diag:
    st.subheader("Edgeless diagnostic agent")
    cols = st.columns(4)
    examples = [
        "Find anomalous or slow MCP sampling interactions in the logs.",
        "Analyze tool latency trends and show me the chart.",
        "Check error frequency, then sync the knowledge graph to Neo4j.",
        "Run an explainability audit on the most recent failure mode.",
    ]
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.pending = ex

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Ask about health, trends, errors, the graph, or request an audit…")
    if st.session_state.pending:
        prompt = st.session_state.pending
        st.session_state.pending = None
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        _run_and_render(agent, prompt)

# ── Tab 2: Explainability Audit Hub ──────────────────────────────────────────
with tab_audit:
    st.subheader("Audit Activation Hub")
    st.caption("Pick an execution track or failure trace, then run a localized "
               "explainability audit (graph context + proxy SHAP/LIME).")

    try:
        sessions = list_sessions()
        failures = find_failure_traces()
    except Exception as exc:
        sessions, failures = [], []
        st.error(f"Could not read the log store: {exc}")

    colL, colR = st.columns(2)
    with colL:
        sess_labels = ["(auto: highest-error session)"] + [
            f"{s['session_id'][:12]}…  ·  {s['interactions']} ints · {s['errors']} errs"
            for s in sessions
        ]
        sess_pick = st.selectbox("Execution track (session)", sess_labels)
        chosen_session = None
        if sess_pick != sess_labels[0]:
            chosen_session = sessions[sess_labels.index(sess_pick) - 1]["session_id"]
    with colR:
        fail_labels = ["(auto: highest-anomaly trace)"] + [
            f"{f['component']} · {f['tool_name']} · {f['key'][-10:]}"
            for f in failures
        ]
        fail_pick = st.selectbox("Targeted failure trace ID", fail_labels)
        chosen_target = None
        if fail_pick != fail_labels[0]:
            chosen_target = failures[fail_labels.index(fail_pick) - 1]["key"]

    st.caption(f"Discovered {len(sessions)} sessions · {len(failures)} error traces.")

    if st.button("▶ Run Explainability Audit Report", type="primary", width="stretch"):
        with st.spinner("Hydrating graph context + computing proxy SHAP/LIME…"):
            report = run_explainability_audit(
                session_id=chosen_session, target_key=chosen_target
            )
            if report.get("proxy_shap"):
                write_audit_report(report)
        st.session_state["_last_audit"] = report

    if st.session_state.get("_last_audit"):
        st.divider()
        render_audit(st.session_state["_last_audit"])
        st.download_button(
            "⬇ Download explainability_audit_report.json",
            data=json.dumps(st.session_state["_last_audit"], indent=2),
            file_name="explainability_audit_report.json",
            mime="application/json",
        )
