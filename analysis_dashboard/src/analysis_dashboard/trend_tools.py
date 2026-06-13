"""
trend_tools.py
──────────────
Operational trend-analysis toolkit for the Log Analysis Agent.

Parses the structured logs into a tabular frame and computes operational
metrics — tool-latency moving averages, token-consumption patterns, and
error-frequency counts over time — rendering each as a matplotlib/seaborn
chart. Charts are written to disk (default `charts/`) so the Streamlit
dashboard can display them and the user can keep them as evidence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — safe for servers / Streamlit
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from .logger import analysis_log  # noqa: E402
from .store_reader import LOG_ROOT, get_reader  # noqa: E402

sns.set_theme(style="whitegrid")

CHART_DIR = Path(os.environ.get("CHART_DIR", "charts"))


def _ensure_chart_dir(save_path: str | None) -> Path:
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CHART_DIR / f"chart_{stamp}.png"


def _load_frame() -> pd.DataFrame:
    """Load all persisted log entries into a tidy DataFrame."""
    reader = get_reader()
    entries = reader.list_entries(namespace_prefix=(LOG_ROOT,), limit=2000)
    rows: list[dict[str, Any]] = []
    for rec in entries:
        e = rec["entry"]
        rows.append(
            {
                "session_id": e.get("session_id"),
                "interaction_type": e.get("mcp_interaction_type"),
                "component": e.get("component"),
                "tool_name": e.get("tool_name") or e.get("component"),
                "latency_ms": float(e.get("latency_ms", 0.0) or 0.0),
                "token_estimate": int(e.get("token_estimate", 0) or 0),
                "status": e.get("status", "success"),
                "timestamp": e.get("timestamp", ""),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        df["seq"] = range(1, len(df) + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Tool 1 — tool latency moving average
# ─────────────────────────────────────────────────────────────────────────────
@tool
def analyze_tool_latency(window: str = "3", save_path: str = "") -> str:
    """
    Compute and chart tool-call latency trends with a moving average.

    Use to investigate performance regressions or slow components. Plots each
    interaction's latency in execution order, overlaid with a rolling-mean
    trend line, and reports per-tool average latency.

    Args:
        window: Moving-average window size (number of interactions, e.g. "3").
        save_path: Optional explicit path to save the chart PNG (else charts/).

    Returns:
        JSON string with per-tool latency stats and the saved chart path.
    """
    # llama tool-calling sometimes emits ints as strings — coerce defensively.
    try:
        win = max(1, int(float(window)))
    except (TypeError, ValueError):
        win = 3
    analysis_log.info(f"analyze_tool_latency | window={win}")
    df = _load_frame()
    if df.empty:
        return json.dumps({"error": "no log data available"})

    df = df.sort_values("seq")
    df["latency_ma"] = df["latency_ms"].rolling(window=win, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="seq", y="latency_ms", marker="o", label="latency (ms)", ax=ax)
    sns.lineplot(data=df, x="seq", y="latency_ma", color="red",
                 label=f"{win}-step moving avg", ax=ax)
    ax.set_title("Tool / Interaction Latency Trend")
    ax.set_xlabel("Interaction sequence")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()
    out = _ensure_chart_dir(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    per_tool = (
        df.groupby("tool_name")["latency_ms"]
        .agg(["count", "mean", "max"])
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )
    analysis_log.info(f"analyze_tool_latency | chart saved → {out}")
    return json.dumps(
        {
            "metric": "tool_latency_moving_average",
            "window": win,
            "interactions": int(len(df)),
            "overall_avg_ms": round(float(df["latency_ms"].mean()), 2),
            "per_tool": per_tool,
            "chart_path": str(out),
        },
        indent=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tool 2 — token consumption patterns
# ─────────────────────────────────────────────────────────────────────────────
@tool
def analyze_token_consumption(save_path: str = "") -> str:
    """
    Chart token-consumption patterns by interaction type and tool.

    Use to find token-heavy components driving cost/latency. Produces a bar
    chart of total estimated tokens per tool, split by interaction type.

    Args:
        save_path: Optional explicit path to save the chart PNG (else charts/).

    Returns:
        JSON string with token totals per tool/type and the saved chart path.
    """
    analysis_log.info("analyze_token_consumption")
    df = _load_frame()
    if df.empty:
        return json.dumps({"error": "no log data available"})

    grouped = (
        df.groupby(["tool_name", "interaction_type"])["token_estimate"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=grouped, x="tool_name", y="token_estimate",
                hue="interaction_type", ax=ax)
    ax.set_title("Token Consumption by Tool & Interaction Type")
    ax.set_xlabel("Tool")
    ax.set_ylabel("Total estimated tokens")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = _ensure_chart_dir(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    analysis_log.info(f"analyze_token_consumption | chart saved → {out}")
    return json.dumps(
        {
            "metric": "token_consumption",
            "total_tokens": int(df["token_estimate"].sum()),
            "by_tool_type": grouped.to_dict(orient="records"),
            "chart_path": str(out),
        },
        indent=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tool 3 — error frequency over time
# ─────────────────────────────────────────────────────────────────────────────
@tool
def analyze_error_frequency(save_path: str = "") -> str:
    """
    Count and chart error frequency across components over the execution trace.

    Use to detect failure hot-spots. Produces a bar chart of success vs error
    counts per component and reports the overall error rate.

    Args:
        save_path: Optional explicit path to save the chart PNG (else charts/).

    Returns:
        JSON string with error counts per component and the saved chart path.
    """
    analysis_log.info("analyze_error_frequency")
    df = _load_frame()
    if df.empty:
        return json.dumps({"error": "no log data available"})

    counts = (
        df.groupby(["component", "status"]).size().reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x="component", y="count", hue="status", ax=ax)
    ax.set_title("Success vs Error Frequency by Component")
    ax.set_xlabel("Component")
    ax.set_ylabel("Interaction count")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    out = _ensure_chart_dir(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    total = int(len(df))
    errors = int((df["status"] == "error").sum())
    analysis_log.info(f"analyze_error_frequency | chart saved → {out}")
    return json.dumps(
        {
            "metric": "error_frequency",
            "total_interactions": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "by_component": counts.to_dict(orient="records"),
            "chart_path": str(out),
        },
        indent=2,
    )


TREND_TOOLS = [analyze_tool_latency, analyze_token_consumption, analyze_error_frequency]
