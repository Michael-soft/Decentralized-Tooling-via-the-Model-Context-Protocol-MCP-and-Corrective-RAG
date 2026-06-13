"""
logger.py
─────────
Dual-stream logging for the agent client.

Produces two named loggers:
  CLIENT — records the agent's own orchestration events.
  SERVER — receives log entries forwarded from the MCP server via
           the log_handler callback and writes them with a [SERVER] prefix.

Both streams are written simultaneously to:
  • stdout          (human-readable, coloured by level)
  • agent_system.log (append-mode, persistent across runs)

Log format
  [2026-05-16 10:14:22] [CLIENT] [INFO]  Sending query to server...
  [2026-05-16 10:14:23] [SERVER] [DEBUG] Initiating ToT Evaluation on Resource...
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FILE = Path(os.environ.get("MCP_FLAT_LOG", "mcp_agent_system.log"))
_FMT     = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATE    = "%Y-%m-%d %H:%M:%S"


def _build_logger(name: str) -> logging.Logger:
    """Create a named logger with both a stream and a file handler."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False          # prevent double-printing via root logger

    if not log.handlers:
        formatter = logging.Formatter(_FMT, datefmt=_DATE)

        # stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        log.addHandler(sh)

        # persistent file handler — appends so multi-run history is retained
        fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log


# Public singletons — import these everywhere
client_log: logging.Logger = _build_logger("CLIENT")
server_log: logging.Logger = _build_logger("SERVER")
