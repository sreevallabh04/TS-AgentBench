"""Diagnostic tools.

Importing this package registers every tool in :data:`tools.base.TOOLS`. See the note
in ``forecasting/__init__.py``: eager registration is required for multiprocessing
workers, which re-import modules in a fresh interpreter.
"""

from tools.base import TOOLS, Tool, ToolResult, run_tools  # noqa: F401

# Imported for their registration side-effects.
from tools import diagnostics as _diagnostics  # noqa: F401,E402
from tools import backtest as _backtest  # noqa: F401,E402

__all__ = ["TOOLS", "Tool", "ToolResult", "run_tools"]
