"""Tool interface.

A "tool" is a diagnostic the agent may choose to run. Two properties make the tool
layer a research object rather than a utility module:

**Declared cost.** Every tool states what it costs to run. H4 asks whether selective
tool use can match exhaustive tool use more cheaply, and that question is only
meaningful if cost is measured rather than assumed.

**History-only inputs.** A tool sees ``history`` and never the forecast target. This is
enforced by the signature itself: :meth:`Tool.run` simply is not given the future. The
single most seductive leak in this project would be a diagnostic that peeks at the
target, and the type signature makes that impossible to do by accident.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.registry import Registry

__all__ = ["Tool", "ToolResult", "TOOLS", "run_tools"]

TOOLS: Registry["Tool"] = Registry("tool")


@dataclass
class ToolResult:
    """Output of one tool invocation."""

    tool_name: str
    values: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    cost: float = 0.0
    seconds: float = 0.0
    failed: bool = False
    error: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class Tool(ABC):
    """Base class for diagnostics.

    Subclasses implement :meth:`_run` and may raise; the public :meth:`run` records
    timing, cost and failure. A failed diagnostic must not abort an agent -- it should
    degrade the evidence available to the decision, exactly as it would in deployment.
    """

    name: str = "tool"
    #: Relative computational cost, in arbitrary units consistent across tools.
    cost: float = 1.0
    #: One-line description shown to the LLM when it selects tools.
    description: str = ""

    @abstractmethod
    def _run(self, history: np.ndarray, seasonal_period: int, **kwargs) -> dict[str, Any]:
        """Compute the diagnostic. May assume finite, non-empty ``history``."""

    def summarize(self, values: dict[str, Any]) -> str:
        """Short natural-language summary, used in LLM prompts."""
        return ", ".join(f"{k}={_fmt(v)}" for k, v in values.items())

    def run(self, history: np.ndarray, seasonal_period: int = 1, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        history = np.asarray(history, dtype=float)
        finite = history[np.isfinite(history)]
        if finite.size < 3:
            return ToolResult(
                tool_name=self.name,
                cost=self.cost,
                seconds=time.perf_counter() - t0,
                failed=True,
                error="insufficient finite history",
            )
        try:
            values = self._run(finite, max(1, int(seasonal_period)), **kwargs)
            return ToolResult(
                tool_name=self.name,
                values=values,
                summary=self.summarize(values),
                cost=self.cost,
                seconds=time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't abort the agent
            return ToolResult(
                tool_name=self.name,
                cost=self.cost,
                seconds=time.perf_counter() - t0,
                failed=True,
                error=f"{type(exc).__name__}: {exc}",
            )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    return str(value)


def run_tools(
    names: list[str], history: np.ndarray, seasonal_period: int = 1, **kwargs
) -> dict[str, ToolResult]:
    """Run a set of tools by name, returning results keyed by tool name."""
    return {
        name: TOOLS.build(name).run(history, seasonal_period, **kwargs)
        for name in names
    }
