"""
app.py
──────
Streamlit human-in-the-loop diagnostic dashboard for the Log Analysis Agent.

Provides:
  • A natural-language chat interface to the decoupled Log Analysis Agent.
  • Live rendering of the agent's step-by-step structural reasoning (each
    tool call + result).
  • Inline display of generated trend charts (latency / tokens / errors).
  • Clear notifications of which Neo4j nodes and edges were committed during
    a graph-sync.

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
from analysis_dashboard.graph_mapper import neo4j_config

st.set_page_config(page_title="MCP Observability Control Plane", page_icon="🔭", layout="wide")


@st.cache_resource(show_spinner="Booting Log Analysis Agent + embedding model…")
def _get_agent():
    return build_analysis_agent()


def _render_tool_result(name: str, content: str) -> None:
    """Render a single tool result: charts inline, graph syncs as notifications."""
    try:
        data = json.loads(content)
    except Exception:
        st.code(content[:2000])
        return

    # Knowledge-graph sync notification
    if name == "sync_knowledge_graph" or {"nodes", "edges"} & set(data.keys()):
        status = data.get("status")
        if status == "committed":
            st.success(
                f" Neo4j sync committed — {data.get('nodes_total', 0)} nodes, "
                f"{data.get('edges_total', 0)} edges → {data.get('uri', '')}"
            )
            c1, c2 = st.columns(2)
            c1.json({"nodes": data.get("nodes", {})})
            c2.json({"edges": data.get("edges", {})})
        elif status == "not_configured":
            st.warning(f" Neo4j not configured — {data.get('message', '')} "
                       f"(planned {data.get('planned_nodes', 0)} nodes / "
                       f"{data.get('planned_edges', 0)} edges)")
        else:
            st.error(f" Graph sync error — {data.get('message', data)}")
        return

    # Chart-producing analytics tools
    chart_path = data.get("chart_path")
    st.json({k: v for k, v in data.items() if k != "by_component"})
    if chart_path and Path(chart_path).exists():
        st.image(chart_path, caption=chart_path, width="stretch")


def _run_and_render(agent, query: str) -> None:
    """Invoke the agent and stream its reasoning + artifacts into the UI."""
    with st.chat_message("assistant"):
        with st.spinner("Analysing logs…"):
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])

        # Structural reasoning trace
        with st.expander(" Agent reasoning trace (step-by-step)", expanded=True):
            for msg in messages:
                role = type(msg).__name__
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        st.markdown(f"**🔧 Tool call → `{tc['name']}`**")
                        if tc.get("args"):
                            st.code(json.dumps(tc["args"], indent=2), language="json")
                elif role == "ToolMessage":
                    st.markdown(f"** Result ← `{getattr(msg, 'name', '?')}`**")
                    _render_tool_result(getattr(msg, "name", ""), str(msg.content))
                elif getattr(msg, "content", None) and role == "AIMessage":
                    st.markdown(f"> {str(msg.content)[:1200]}")

        # Final answer
        final = messages[-1].content if messages else "(no response)"
        st.markdown("### 🩺 Diagnosis")
        st.markdown(final)
        st.session_state.history.append(("assistant", final))


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔭 MCP Observability Control Plane")
st.caption("Stage 3 — decoupled Log Analysis Agent over the hierarchical vector log store")

# Executing-machine clock — rendered server-side so a single dashboard
# screenshot carries its own timestamp evidence (in addition to the OS clock).
_now = datetime.now().astimezone()
st.info(
    f"🕒 **Executing-machine time:** {_now.strftime('%a %d %b %Y, %H:%M:%S %Z')}  "
    f"·  run rendered live by Streamlit"
)

with st.sidebar:
    st.header("System status")
    db_path = os.environ.get("MCP_LOG_DB_PATH", "mcp_agent_log.db")
    st.metric("Log store", "online" if Path(db_path).exists() else "missing")
    st.text(f"DB: {db_path}")
    st.text("Embeddings: BAAI/bge-small-en-v1.5")
    cfg = neo4j_config()
    if cfg:
        st.success(f"Neo4j Aura configured\n{cfg['uri']}")
    else:
        st.warning("Neo4j not configured\n(set NEO4J_* in .env)")

    st.divider()
    st.subheader("Example questions")
    examples = [
        "Find anomalous or slow MCP sampling interactions in the logs.",
        "Analyze tool latency trends and show me the chart.",
        "Which component consumes the most tokens?",
        "Check error frequency, then sync the knowledge graph to Neo4j and report committed nodes/edges.",
    ]
    for ex in examples:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

agent = _get_agent()

# Replay history
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask the Log Analysis Agent about system health, trends, errors, or the graph…")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    _run_and_render(agent, prompt)
