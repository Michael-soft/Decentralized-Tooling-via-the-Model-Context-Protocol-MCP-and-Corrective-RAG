"""
session.py
──────────
Process-wide session identity for the multi-turn execution trace.

Every log entry persisted to the vector store carries this `session_id`
(a UUID) so the decoupled analysis agent can stitch a complete client→
server→sampling causal path back together and project it as a single
(:Session) node in Neo4j.
"""

from __future__ import annotations

import os
import uuid

_SESSION_ID: str = os.environ.get("MCP_SESSION_ID") or str(uuid.uuid4())


def get_session_id() -> str:
    """Return the current process session UUID."""
    return _SESSION_ID


def new_session() -> str:
    """Rotate to a fresh session UUID (e.g. between independent demo runs)."""
    global _SESSION_ID
    _SESSION_ID = str(uuid.uuid4())
    return _SESSION_ID
