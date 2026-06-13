"""
logger.py (analysis side)
─────────────────────────
Dedicated logger for the decoupled Log Analysis Agent.

Writes to stdout and to `analysis_agent.log` (a Stage 3 deliverable) with an
[ANALYSIS] prefix, keeping this process's observability stream clearly
separated from the operational MCP client/server streams.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FILE = Path(os.environ.get("ANALYSIS_LOG", "analysis_agent.log"))
_FMT = "[%(asctime)s] [ANALYSIS] [%(levelname)s] %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def _build_logger(name: str = "ANALYSIS") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not log.handlers:
        fmt = logging.Formatter(_FMT, datefmt=_DATE)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


analysis_log: logging.Logger = _build_logger()
