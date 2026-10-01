"""Time-series foundation model baseline (Chronos).

Registered unconditionally so that configs can name it, but the package and weights are
imported lazily. If ``chronos-forecasting`` is not installed the forecaster raises a
clear, actionable error and the base class converts that into a recorded fallback
rather than crashing a run -- which matters because this is the one baseline with a
large external download.

    python scripts/setup_foundation_model.py        # fetch weights once

Why it is optional rather than core: on CPU, Chronos costs roughly an order of
magnitude more per forecast than ARIMA, so including it in every cell of the
counterfactual tensor would dominate the compute budget for a baseline that is not the
subject of the study. The honest arrangement is to run it as a zero-shot baseline on a
defined subsample, report it as such, and state the restriction rather than quietly
omitting the comparison.

Chronos is a *zero-shot* model: it is not fitted to the series. That is the point of a
foundation model, and it makes it a genuinely different kind of baseline from
everything else in the pool.
"""

from __future__ import annotations

import numpy as np

from forecasting.base import FORECASTERS, Forecaster

__all__ = ["ChronosForecaster"]

_INSTALL_HINT = (
    "chronos-forecasting is not installed. Install it with\n"
    "    python -m pip install chronos-forecasting\n"
    "or run\n"
    "    python scripts/setup_foundation_model.py\n"
    "It is deliberately excluded from requirements.txt so that the benchmark stays "
    "installable without a large model download."
)


@FORECASTERS.register("chronos")
class ChronosForecaster(Forecaster):
    """Zero-shot forecasting with a pretrained Chronos model.

    Defaults to the ``bolt-small`` checkpoint, which is the largest variant that
    remains tractable on CPU. The pipeline is cached at class level because loading
    weights per forecast would dwarf inference cost.
    """

    name = "chronos"
    cost = 30.0

    _pipeline = None
    _loaded_model: str | None = None

    def __init__(self, model_name: str = "amazon/chronos-bolt-small", **params):
        super().__init__(**params)
        self.model_name = model_name

    @classmethod
    def _get_pipeline(cls, model_name: str):
        if cls._pipeline is not None and cls._loaded_model == model_name:
            return cls._pipeline
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(_INSTALL_HINT) from exc

        import torch

        cls._pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32
        )
        cls._loaded_model = model_name
        return cls._pipeline

    def _forecast(self, history, horizon, seasonal_period, rng):
        import torch

        pipeline = self._get_pipeline(self.model_name)
        context = torch.tensor(np.asarray(history, dtype=np.float32))

        # Chronos returns quantile levels; we ask for a dense grid and treat them as
        # sample paths so the output matches every other model's interface and CRPS
        # stays comparable.
        levels = np.linspace(0.05, 0.95, 19)
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=int(horizon),
            quantile_levels=list(levels),
        )
        q = quantiles[0].numpy()          # (horizon, n_levels)
        point = np.asarray(mean[0], dtype=float).reshape(-1)[:horizon]

        # Interpolate the quantile function at uniform probabilities to obtain samples.
        n_samples = 200
        probs = (np.arange(n_samples) + 0.5) / n_samples
        samples = np.empty((n_samples, horizon), dtype=float)
        for h in range(horizon):
            samples[:, h] = np.interp(probs, levels, q[h])
        return point, samples
