"""Forecasting models.

Importing this package registers every forecaster in
:data:`forecasting.base.FORECASTERS`.

This eager registration is deliberate and load-bearing. The registry is populated by
import side-effects, and joblib's ``loky`` backend spawns *fresh* interpreter
processes that re-import modules from scratch. If registration lived only in the
caller's ``__main__``, every worker would start with an empty registry and the whole
tensor build would fail. Registering here means any module that touches
``forecasting.base`` gets a fully populated registry in every process.

The neural models are safe to import eagerly because ``torch`` is imported lazily
inside their methods, so this costs nothing when they are not used. The same applies to
the foundation-model baseline, whose optional dependency and weights load only on first
use.
"""

from forecasting.base import FORECASTERS, ForecastResult, Forecaster  # noqa: F401

# Imported for their registration side-effects.
from forecasting import classical as _classical  # noqa: F401,E402
from forecasting import ml as _ml  # noqa: F401,E402
from forecasting import deep as _deep  # noqa: F401,E402
from forecasting import foundation as _foundation  # noqa: F401,E402

__all__ = ["FORECASTERS", "Forecaster", "ForecastResult"]
