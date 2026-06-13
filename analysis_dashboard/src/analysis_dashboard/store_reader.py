"""
store_reader.py
───────────────
Read-side accessor for the shared hierarchical vector log store.

The operational MCP client writes interaction traces into `mcp_agent_log.db`
via `langgraph.store.sqlite.SqliteStore`. This module opens the *same* file
from the decoupled analysis process and exposes:

  • semantic_search(query)  — vector similarity over the `content` field
  • list_entries(prefix)    — full structured dump for graph / trend tooling

Crucially, it re-applies the integer-key data guardrail when rehydrating
entries: JSON stored `config_map` keys as strings, and they are cast back to
int here — demonstrating type-safe serialization across the decoupled agent
boundary (a core Stage 3 requirement).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from .embeddings import EMBED_DIMS, get_embeddings
from .logger import analysis_log

DEFAULT_DB_PATH = os.environ.get("MCP_LOG_DB_PATH", "mcp_agent_log.db")
LOG_ROOT = "logs"


def restore_int_keys(mapping: dict[Any, Any]) -> dict[Any, str]:
    """Cast string-coerced integer keys back to int (data guardrail)."""
    restored: dict[Any, str] = {}
    for key, val in mapping.items():
        if isinstance(key, int):
            restored[key] = val
        elif isinstance(key, str) and key.lstrip("-").isdigit():
            restored[int(key)] = val
        else:
            restored[key] = val
    return restored


class LogStoreReader:
    """Opens the shared SQLite vector store for read/query access."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        from langgraph.store.sqlite import SqliteStore

        self.db_path = db_path
        index = {"dims": EMBED_DIMS, "embed": get_embeddings(), "fields": ["content"]}
        self._cm = SqliteStore.from_conn_string(db_path, index=index)
        self.store = self._cm.__enter__()
        self.store.setup()
        analysis_log.info(f"Log store opened (read) → {db_path}")

    def _hydrate(self, item: Any, include_score: bool = False) -> dict[str, Any]:
        value = dict(item.value)
        value["config_map"] = restore_int_keys(value.get("config_map", {}))
        rec = {
            "namespace": list(item.namespace),
            "key": item.key,
            "entry": value,
        }
        if include_score:
            rec["score"] = float(item.score) if getattr(item, "score", None) is not None else None
        return rec

    def semantic_search(
        self,
        query: str,
        namespace_prefix: tuple[str, ...] = (LOG_ROOT,),
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search over persisted log content."""
        results = self.store.search(namespace_prefix, query=query, limit=limit)
        return [self._hydrate(r, include_score=True) for r in results]

    def list_entries(
        self,
        namespace_prefix: tuple[str, ...] = (LOG_ROOT,),
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Full structured dump of all entries under a namespace prefix."""
        results = self.store.search(namespace_prefix, query=None, limit=limit)
        return [self._hydrate(r) for r in results]

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_reader(db_path: Optional[str] = None) -> LogStoreReader:
    """Return a process-wide singleton reader (lazy)."""
    return LogStoreReader(db_path or DEFAULT_DB_PATH)
