"""Structured logging.

Two sinks: a human-readable console stream, and a JSONL file per run. The JSONL
stream is what makes agent behaviour auditable after the fact -- every tool call,
reliability score and correction decision lands there with its instance id, so a
surprising number in a results table can be traced back to the decision that produced
it.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["setup_logging", "get_logger", "JsonlWriter"]

_CONFIGURED = False


class _ConsoleFormatter(logging.Formatter):
    """Compact console format; keeps long runs readable."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        name = record.name.replace("TS-AgentBench.", "")
        return f"{ts} {record.levelname:<7} {name:<22} {record.getMessage()}"


def setup_logging(
    log_file: Path | None = None, level: str = "INFO", quiet: bool = False
) -> None:
    """Configure the root project logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger("TS-AgentBench")
    if _CONFIGURED:
        root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(_ConsoleFormatter())
        root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Logger namespaced under the project root."""
    return logging.getLogger(f"TS-AgentBench.{name}")


class JsonlWriter:
    """Append-only JSONL sink for machine-readable event records.

    Opened in append mode and flushed per record so that a run killed halfway still
    leaves a usable trace -- important when a long experiment is interrupted.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(record, default=_fallback, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fallback(obj: Any) -> Any:
    """Make numpy scalars and paths JSON-serialisable."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover
            pass
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)
