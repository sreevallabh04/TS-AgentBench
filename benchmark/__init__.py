"""Benchmark construction: corrective actions, the counterfactual tensor, oracle labels.

Importing this package registers every corrective action in
:data:`benchmark.actions.ACTIONS`.

Note that :mod:`benchmark.oracle` is *not* imported here. It is evaluation-only and
must stay out of any import path reachable from ``agents/``; see
``tests/test_leakage.py``.
"""

from benchmark.actions import ACTIONS, ACTION_ORDER, Action, ActionContext  # noqa: F401

__all__ = ["ACTIONS", "ACTION_ORDER", "Action", "ActionContext"]
