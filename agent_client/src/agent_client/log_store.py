"""
log_store.py
────────────
Embedded, vector-enabled hierarchical log persistence for the MCP client.

This is the Stage 3 structural upgrade over the Stage 2 flat `*.log` file.
Every MCP interaction (tool invocation, resource read, sampling request) is
persisted into a `langgraph.store.sqlite.SqliteStore` backed by a local
embedding model, so the traces become *semantically searchable* by the
decoupled Log Analysis Agent.

Design
──────
• Hierarchical namespaces — the store's native `(namespace, key)` tuple is
  used to express component ownership as dot-separated domains, e.g.
      ("logs", "agent", "planning", "reflexive_loop")
      ("logs", "mcp", "server", "tools", "reflection_tool")
• Validated schema (`LogEntry`) — every entry carries:
      session_id            : UUID for the multi-turn execution trace
      mcp_interaction_type  : tool_invocation | resource_read | sampling_request
      content               : raw text / stringified JSON → vector index field
  plus structural metadata used for graph mapping and trend analysis.
• Vector serialization — the store is configured with `index={"fields":
  ["content"], "embed": <FastEmbed>}`, so `content` is embedded on write
  and similarity-searchable on read.
• Data guardrail — JSON serialization silently coerces integer dict keys to
  strings. `config_map` deliberately uses integer keys; `to_store_value()` /
  `from_store_value()` round-trip them so they are cast cleanly back to int,
  preventing type-mismatch faults across the decoupled agent boundary.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .embeddings import EMBED_DIMS, get_embeddings
from .logger import client_log

# ── Interaction-type contract ────────────────────────────────────────────────
McpInteractionType = Literal["tool_invocation", "resource_read", "sampling_request"]

#: Root of every hierarchical namespace tuple.
LOG_ROOT = "logs"

#: On-disk SQLite file (deliverable: mcp_agent_log.db). Overridable via env.
DEFAULT_DB_PATH = os.environ.get("MCP_LOG_DB_PATH", "mcp_agent_log.db")


# ─────────────────────────────────────────────────────────────────────────────
#  Validated log schema
# ─────────────────────────────────────────────────────────────────────────────
class LogEntry(BaseModel):
    """
    Structured, validated contract for a single persisted MCP interaction.

    The `content` field is the one indexed for vector similarity search; the
    remaining fields are structural metadata consumed by the analysis agent
    for Neo4j graph projection and operational trend calculations.
    """

    # ── Required schema (per task spec) ──────────────────────────────────────
    session_id: str = Field(description="UUID tracking the multi-turn execution trace.")
    mcp_interaction_type: McpInteractionType = Field(
        description="tool_invocation | resource_read | sampling_request."
    )
    content: str = Field(description="Raw text / stringified JSON → vector index field.")

    # ── Structural metadata (graph + trend analysis) ─────────────────────────
    timestamp: str = Field(default="", description="UTC ISO-8601 capture time.")
    component: str = Field(default="agent", description="Dot-path scope, e.g. mcp.server.tools.reflection_tool.")
    tool_name: Optional[str] = Field(default=None, description="Tool/resource invoked.")
    target: Optional[str] = Field(default=None, description="Server-side target the action routed to.")
    request_id: Optional[str] = Field(default=None, description="Correlates sampling requests to server tools.")
    latency_ms: float = Field(default=0.0, description="Wall-clock duration of the interaction.")
    token_estimate: int = Field(default=0, description="Approx tokens in the content payload.")
    status: Literal["success", "error"] = Field(default="success")
    error: Optional[str] = Field(default=None)

    # nested requestedSchema keys + structural arrays must be respected
    requested_schema: dict[str, Any] = Field(default_factory=dict)

    # ── Data guardrail demonstrator ──────────────────────────────────────────
    # Integer keys deliberately used; JSON would coerce them to strings, so the
    # store round-trip explicitly casts them back to int (see serializers below).
    config_map: dict[int, str] = Field(
        default_factory=dict,
        description="Integer-keyed config map; keys are restored to int on read.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _default_timestamp(cls, v: str) -> str:
        return v or datetime.now(timezone.utc).isoformat()

    # ── Serialization round-trip with the integer-key guardrail ──────────────
    def to_store_value(self) -> dict[str, Any]:
        """
        Produce a JSON-safe dict for `store.put`.

        `config_map`'s integer keys are stringified here (JSON requirement) but
        tagged so `from_store_value()` can cast them back cleanly to int.
        """
        data = self.model_dump()
        # Cast int keys → str for JSON storage, preserving original type intent.
        data["config_map"] = {str(k): v for k, v in self.config_map.items()}
        return data

    @classmethod
    def from_store_value(cls, value: dict[str, Any]) -> "LogEntry":
        """
        Rehydrate a `LogEntry` from a stored dict, casting `config_map` keys
        back to integers to avoid type-mismatch faults downstream.
        """
        raw = dict(value)
        raw["config_map"] = _restore_int_keys(raw.get("config_map", {}))
        return cls(**raw)


def _restore_int_keys(mapping: dict[Any, Any]) -> dict[int, str]:
    """
    Data guardrail: cast string-coerced integer keys back to `int`.

    JSON has no integer-key concept, so any int key written through a JSON
    boundary returns as a string. Digits-only keys are restored to int; any
    genuinely non-numeric key is left untouched rather than silently dropped.
    """
    restored: dict[int, str] = {}
    for key, val in mapping.items():
        if isinstance(key, int):
            restored[key] = val
        elif isinstance(key, str) and key.lstrip("-").isdigit():
            restored[int(key)] = val
        else:
            # Non-numeric key — keep as-is (best-effort) and warn loudly.
            client_log.warning(f"config_map non-integer key encountered: {key!r}")
            restored[key] = val  # type: ignore[index]
    return restored


def estimate_tokens(text: str) -> int:
    """Cheap word-based token approximation (no tokenizer dependency)."""
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent store singleton (writer side — operational MCP client)
# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalLogStore:
    """
    Thin wrapper around a vector-enabled `SqliteStore`.

    Opens the store once for the lifetime of the process (kept warm via the
    context manager's `__enter__`), and closes it cleanly at interpreter exit.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        from langgraph.store.sqlite import SqliteStore

        self.db_path = db_path
        self._lock = threading.Lock()
        index = {
            "dims": EMBED_DIMS,
            "embed": get_embeddings(),
            "fields": ["content"],   # vector index built from the content field
        }
        self._cm = SqliteStore.from_conn_string(db_path, index=index)
        self.store = self._cm.__enter__()
        self.store.setup()
        atexit.register(self.close)
        client_log.info(f"Hierarchical vector log store ready → {db_path} (dims={EMBED_DIMS})")

    # ── Namespace helpers ────────────────────────────────────────────────────
    @staticmethod
    def namespace(*parts: str) -> tuple[str, ...]:
        """Build a hierarchical namespace tuple rooted at `logs`."""
        return (LOG_ROOT, *parts)

    # ── Write path ───────────────────────────────────────────────────────────
    def record(self, entry: LogEntry, namespace: tuple[str, ...], key: str) -> None:
        """Persist a validated `LogEntry` under a hierarchical namespace."""
        with self._lock:
            try:
                self.store.put(namespace, key, entry.to_store_value())
                client_log.debug(
                    f"Log persisted | ns={'.'.join(namespace)} | key={key} | "
                    f"type={entry.mcp_interaction_type} | status={entry.status}"
                )
            except Exception as exc:  # never let observability crash the agent
                client_log.error(f"Log store write failed: {exc}")

    # ── Read path (used by analysis agent / verification) ────────────────────
    def semantic_search(
        self,
        query: str,
        namespace_prefix: tuple[str, ...] = (LOG_ROOT,),
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search; returns rehydrated entries + scores."""
        results = self.store.search(namespace_prefix, query=query, limit=limit)
        out: list[dict[str, Any]] = []
        for item in results:
            entry = LogEntry.from_store_value(item.value)
            out.append(
                {
                    "namespace": list(item.namespace),
                    "key": item.key,
                    "score": float(item.score) if item.score is not None else None,
                    "entry": entry.model_dump(),
                }
            )
        return out

    def list_all(self, namespace_prefix: tuple[str, ...] = (LOG_ROOT,), limit: int = 500) -> list[dict[str, Any]]:
        """Non-semantic dump of every entry under a prefix (for graph/trend tools)."""
        results = self.store.search(namespace_prefix, query=None, limit=limit)
        return [
            {
                "namespace": list(item.namespace),
                "key": item.key,
                "entry": LogEntry.from_store_value(item.value).model_dump(),
            }
            for item in results
        ]

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass


# ── Process-wide singleton accessor ──────────────────────────────────────────
_store_instance: Optional[HierarchicalLogStore] = None


def get_log_store(db_path: str = DEFAULT_DB_PATH) -> HierarchicalLogStore:
    """Return the singleton vector log store, creating it on first use."""
    global _store_instance
    if _store_instance is None:
        _store_instance = HierarchicalLogStore(db_path)
    return _store_instance
