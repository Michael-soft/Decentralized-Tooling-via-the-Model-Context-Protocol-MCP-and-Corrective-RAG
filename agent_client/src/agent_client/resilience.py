"""
resilience.py
─────────────
Stage 4 — Resilient client orchestration & self-healing chains.

This module hardens the operational MCP client against two distinct failure
boundaries, exactly as the Stage 4 brief requires:

  1. Infrastructure Fault Isolation  → LangChain ``RunnableWithRetry``
     (``.with_retry``). Transient communication faults — socket drops, DNS
     blips, HTTP 429 rate limits — are retried with native exponential backoff
     and jitter, bounded by ``MAX_RETR_ATTEMPTS``, before any exception bubbles.

  2. Declarative Self-Healing        → LangChain ``RunnableWithFallbacks``
     (``.with_fallbacks(..., exception_key="error_trace")``). If a tool runs at
     the network layer but returns an internal application error / invalid
     payload / runtime exception, a chain of self-correcting fallback runnables
     attempts an LLM-driven reinjection loop to resolve it.

  3. Hardcoded Absolute Fallback     → the final runnable in the
     ``.with_fallbacks`` list. It can never raise: it captures the execution
     state, logs the catastrophic failure to the hierarchical SQLite store, and
     returns a structured, deterministic error payload to the caller.

Every fallback activation and self-healing iteration is persisted to the vector
log store under the ``("logs","resilience",*)`` namespace, so the decoupled
analysis dashboard can report historical resilience counts (Stage 4 §5).

A controlled, env-gated fault injector (``MCP_FAULT_INJECT``) lets a demo run
deterministically exercise — and recover from — each failure boundary so the
deliverable logs contain verifiable fallback-activation traces.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable

from langchain_core.runnables import Runnable, RunnableLambda

from .log_store import LogEntry, estimate_tokens, get_log_store
from .logger import client_log
from .session import get_session_id

# ── Configuration ────────────────────────────────────────────────────────────
#: Retry budget for the RunnableWithRetry infrastructure layer.
MAX_RETR_ATTEMPTS = int(os.environ.get("MAX_RETR_ATTEMPTS", "3"))

#: The exception_key injected into the payload for self-healing fallbacks.
EXCEPTION_KEY = "error_trace"

#: Self-healing reinjection budget (LLM correction iterations per fallback).
SELF_HEAL_MAX_ITERS = int(os.environ.get("SELF_HEAL_MAX_ITERS", "2"))


# ─────────────────────────────────────────────────────────────────────────────
#  Failure taxonomy
# ─────────────────────────────────────────────────────────────────────────────
class TransientNetworkError(ConnectionError):
    """Infrastructure-layer fault (socket/DNS/429) — eligible for retry."""


class ToolApplicationError(RuntimeError):
    """Application-layer fault (bad payload / runtime) — eligible for self-heal."""


def _transient_exception_types() -> tuple[type[BaseException], ...]:
    """Collect the transient exception types the retry layer should catch."""
    types: list[type[BaseException]] = [
        TransientNetworkError,
        ConnectionError,
        TimeoutError,
        OSError,
    ]
    try:  # httpx powers the FastMCP streamable-http transport
        import httpx

        types += [
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ]
    except Exception:  # pragma: no cover - httpx always present in practice
        pass
    try:  # Groq SDK transient errors (rate limits, 5xx, connection)
        import groq

        types += [
            groq.RateLimitError,
            groq.APIConnectionError,
            groq.APITimeoutError,
            groq.InternalServerError,
        ]
    except Exception:  # pragma: no cover
        pass
    return tuple(types)


#: Exception types the RunnableWithRetry layer treats as transient infra flakes.
TRANSIENT_EXCEPTIONS = _transient_exception_types()


# ─────────────────────────────────────────────────────────────────────────────
#  Resilience counters + persistence (read back by the analysis dashboard)
# ─────────────────────────────────────────────────────────────────────────────
RESILIENCE_COUNTERS: dict[str, int] = {
    "retries": 0,
    "fallbacks": 0,
    "self_healing": 0,
    "hardcoded": 0,
}


def _persist_resilience_event(
    kind: str,
    component: str,
    content: str,
    *,
    status: str = "error",
    error: str | None = None,
    latency_ms: float = 0.0,
) -> None:
    """Mirror a resilience event into the hierarchical vector log store."""
    try:
        store = get_log_store()
        entry = LogEntry(
            session_id=get_session_id(),
            mcp_interaction_type="tool_invocation",
            content=content,
            component=f"resilience.{component}",
            tool_name=kind,
            target="agent_client.resilience",
            latency_ms=round(latency_ms, 2),
            token_estimate=estimate_tokens(content),
            status=status,  # type: ignore[arg-type]
            error=error,
            config_map={0: kind, 1: component, 2: status},
        )
        store.record(
            entry,
            store.namespace("resilience", component),
            f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
        )
    except Exception as exc:  # observability must never crash the agent
        client_log.error(f"resilience persist failed ({kind}): {exc}")


def record_retry(component: str, detail: str) -> None:
    RESILIENCE_COUNTERS["retries"] += 1
    client_log.warning(f"[RESILIENCE] retry attempt | {component} | {detail}")
    _persist_resilience_event(
        "retry_attempt", component, f"RETRY ATTEMPT ({component}): {detail}"
    )


def record_fallback(component: str, detail: str) -> None:
    RESILIENCE_COUNTERS["fallbacks"] += 1
    client_log.warning(f"[RESILIENCE] fallback activated | {component} | {detail}")
    _persist_resilience_event(
        "fallback_activation", component, f"FALLBACK ACTIVATED ({component}): {detail}"
    )


def record_self_heal(component: str, detail: str) -> None:
    RESILIENCE_COUNTERS["self_healing"] += 1
    client_log.warning(f"[RESILIENCE] self-healing reinjection | {component} | {detail}")
    _persist_resilience_event(
        "self_healing_iteration",
        component,
        f"SELF-HEALING REINJECTION ({component}): {detail}",
    )


def record_hardcoded(component: str, detail: str) -> None:
    RESILIENCE_COUNTERS["hardcoded"] += 1
    client_log.error(f"[RESILIENCE] HARDCODED absolute fallback | {component} | {detail}")
    _persist_resilience_event(
        "hardcoded_fallback",
        component,
        f"CATASTROPHIC FALLBACK ({component}): {detail}",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Controlled fault injection (env-gated, for verifiable demo evidence)
#  MCP_FAULT_INJECT ∈ {"", "transient", "application", "all"}
# ─────────────────────────────────────────────────────────────────────────────
_FAULT_STATE: dict[str, int] = {}


def _fault_mode() -> str:
    return os.environ.get("MCP_FAULT_INJECT", "").strip().lower()


def set_fault_mode(mode: str) -> None:
    """Set the active fault-injection mode at runtime (for the demo harness)."""
    os.environ["MCP_FAULT_INJECT"] = mode


def reset_fault_state() -> None:
    """Clear the per-component injection memory so a new scenario re-arms."""
    _FAULT_STATE.clear()


def maybe_inject_transient(component: str) -> None:
    """
    Raise a simulated transient fault on the FIRST invocation per component so
    the RunnableWithRetry layer demonstrably catches, backs off, and recovers
    on a subsequent attempt. No-op unless MCP_FAULT_INJECT enables it.
    """
    if _fault_mode() not in ("transient", "all"):
        return
    n = _FAULT_STATE.get(component, 0)
    if n == 0:
        _FAULT_STATE[component] = n + 1
        record_retry(component, "injected transient fault (HTTP 429-like) — retrying")
        raise TransientNetworkError(
            f"[injected] simulated transient network fault on {component} (HTTP 429-like)"
        )


def should_inject_application_error(component: str, healed: bool) -> bool:
    """
    Signal the primary runnable to raise an application error until the
    self-healing loop has re-driven it with a corrected payload. No-op unless
    MCP_FAULT_INJECT enables it.
    """
    return (not healed) and _fault_mode() in ("application", "all")


# ─────────────────────────────────────────────────────────────────────────────
#  Resilient LLM (primary LLM wrapped with RunnableWithRetry)
# ─────────────────────────────────────────────────────────────────────────────
def build_resilient_llm(
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Runnable:
    """
    Return the client's primary LLM wrapped with LangChain's native
    ``RunnableWithRetry`` — exponential backoff + jitter, bounded by
    ``MAX_RETR_ATTEMPTS`` — so transient Groq faults (429 / 5xx / connection)
    are absorbed before bubbling. Used for MCP Sampling and self-healing.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=model or os.environ.get("RESILIENT_LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm.with_retry(
        retry_if_exception_type=TRANSIENT_EXCEPTIONS,
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_RETR_ATTEMPTS,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Resilient tool chain factory
#  primary(.with_retry) → [self_heal, hardcoded] via .with_fallbacks
# ─────────────────────────────────────────────────────────────────────────────
def _hardcoded_fallback(name: str, payload: dict[str, Any]) -> str:
    """
    Final, non-raising fallback. Captures state, logs the catastrophic failure,
    and returns a structured, deterministic error payload to the caller.
    """
    err = payload.get(EXCEPTION_KEY)
    detail = f"all retries + self-healing exhausted: {type(err).__name__}: {err}"
    record_hardcoded(name, detail)
    return json.dumps(
        {
            "status": "degraded",
            "tool": name,
            "error_type": type(err).__name__ if err else "UnknownError",
            "error": str(err),
            "message": (
                f"'{name}' failed after exhausting infrastructure retries and the "
                f"self-healing fallback chain. Returning a deterministic safe payload "
                f"so the agent never experiences an unhandled crash."
            ),
            "result": None,
            "captured_state": {
                k: (str(v)[:200] if v is not None else None)
                for k, v in payload.items()
                if k != EXCEPTION_KEY
            },
        },
        indent=2,
    )


def build_resilient_tool_chain(
    *,
    name: str,
    primary_fn: Callable[[dict[str, Any]], str],
    self_heal_fn: Callable[[dict[str, Any]], str],
) -> Runnable:
    """
    Compose the three-layer resilient runnable for one MCP tool/resource call.

    Args:
        name: Tool name (for logging + structured payloads).
        primary_fn: Performs the real call. Raises a transient exception on an
            infrastructure fault (→ retried) or ``ToolApplicationError`` on an
            application fault (→ self-healed). Receives a payload dict.
        self_heal_fn: Receives the payload with ``EXCEPTION_KEY`` populated;
            runs an LLM-driven reinjection loop and may re-drive ``primary_fn``
            with a corrected payload. May raise if it cannot resolve.

    Returns:
        A ``Runnable[dict -> str]``: primary wrapped with RunnableWithRetry,
        chained via ``.with_fallbacks([self_heal, hardcoded], exception_key=...)``.
    """
    primary = RunnableLambda(primary_fn, name=f"{name}.primary").with_retry(
        retry_if_exception_type=TRANSIENT_EXCEPTIONS,
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_RETR_ATTEMPTS,
    )
    self_heal = RunnableLambda(self_heal_fn, name=f"{name}.self_heal")
    hardcoded = RunnableLambda(
        lambda payload: _hardcoded_fallback(name, payload), name=f"{name}.hardcoded"
    )

    # Declarative self-healing: primary → LLM self-heal → hardcoded absolute.
    return primary.with_fallbacks(
        fallbacks=[self_heal, hardcoded],
        exception_key=EXCEPTION_KEY,
        exceptions_to_handle=(Exception,),
    )


def llm_reinjection(
    *,
    component: str,
    system_directive: str,
    failure_context: str,
) -> str:
    """
    One LLM-driven self-healing reinjection step.

    Feeds the captured error trace back into the resilient LLM and asks it to
    produce a corrected/best-effort result. Records a self-healing iteration.
    Returns the model's text (best-effort); raises only if the LLM itself
    fails after its own retry budget (→ caller's hardcoded fallback).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    record_self_heal(component, failure_context[:160])
    llm = build_resilient_llm(temperature=0.2, max_tokens=512)
    response = llm.invoke(
        [
            SystemMessage(content=system_directive),
            HumanMessage(content=failure_context),
        ]
    )
    return response.content if hasattr(response, "content") else str(response)
