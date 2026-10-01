"""Forecaster interface and shared machinery.

Design decisions that matter for the science:

**Forecasters are stateless per call.** Each rolling origin refits from scratch on
``history``. This is slower than incremental updating but removes an entire class of
leakage bug, where state fitted on a later origin bleeds into an earlier one.

**Every model produces the same output shape**: a point forecast, prediction intervals
at the requested coverage levels, and a sample path matrix. Uniform probabilistic
output is what makes CRPS comparable across a naive baseline and a neural network --
without it, probabilistic comparisons quietly become comparisons of who had an
analytic interval implemented.

**Missing values are imputed once, in the base class.** Individual models are not
trusted to handle NaN consistently, and silently dropping NaN would shift the time
index and corrupt seasonality.

**Failures degrade, they do not crash.** A single ill-conditioned ARIMA fit must not
kill an overnight run, so :meth:`Forecaster.forecast` catches model errors and falls
back to a seasonal naive forecast with the failure recorded in the result metadata.
That fallback is counted and reported, since a method that silently falls back on half
its instances is not the method it claims to be.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core.registry import Registry

__all__ = [
    "ForecastResult",
    "Forecaster",
    "FORECASTERS",
    "impute_missing",
    "residual_samples",
    "intervals_from_samples",
]

FORECASTERS: Registry["Forecaster"] = Registry("forecaster")

N_SAMPLES = 200  # sample paths per forecast; drives CRPS and empirical intervals


@dataclass
class ForecastResult:
    """Output of a single forecast.

    ``samples`` has shape ``(N_SAMPLES, horizon)`` and is the canonical probabilistic
    object; intervals are derived from it so that intervals and CRPS can never
    disagree about the same predictive distribution.
    """

    point: np.ndarray
    samples: np.ndarray
    model_name: str
    lower: dict[float, np.ndarray] = field(default_factory=dict)
    upper: dict[float, np.ndarray] = field(default_factory=dict)
    fit_seconds: float = 0.0
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.point.shape[0])

    def interval_width(self, level: float = 0.8) -> float:
        """Mean width of the prediction interval at ``level``; a reliability signal."""
        if level not in self.lower:
            return float("nan")
        return float(np.mean(self.upper[level] - self.lower[level]))


def impute_missing(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Linearly interpolate interior NaNs; edge-fill the rest.

    Returns the imputed array and the number of values imputed, so downstream code can
    treat heavily-imputed history as a reliability signal rather than pretending the
    data were complete.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isnan(x)
    n_missing = int(mask.sum())
    if n_missing == 0:
        return x, 0
    if mask.all():
        return np.zeros_like(x), n_missing

    idx = np.arange(x.size)
    out = x.copy()
    out[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return out, n_missing


def residual_samples(
    point: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = N_SAMPLES,
    accumulate: bool = True,
) -> np.ndarray:
    """Bootstrap sample paths around a point forecast from in-sample residuals.

    ``accumulate=True`` random-walks the residuals so that uncertainty widens with the
    horizon, which is the right default for models whose errors compound (naive,
    drift, AR). Models with their own analytic variance path override this.
    """
    h = int(point.shape[0])
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 2:
        # Degenerate history: fall back to a small scale rather than zero-width
        # intervals, which would claim perfect certainty.
        scale = float(np.std(point)) or 1.0
        residuals = np.array([-scale, scale], dtype=float)

    draws = rng.choice(residuals, size=(n_samples, h), replace=True)
    if accumulate:
        draws = np.cumsum(draws, axis=1) / np.sqrt(np.arange(1, h + 1))
    return point[None, :] + draws


def intervals_from_samples(
    samples: np.ndarray, levels: Sequence[float]
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Empirical central prediction intervals from sample paths."""
    lower: dict[float, np.ndarray] = {}
    upper: dict[float, np.ndarray] = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lower[level] = np.quantile(samples, tail, axis=0)
        upper[level] = np.quantile(samples, 1.0 - tail, axis=0)
    return lower, upper


class Forecaster(ABC):
    """Base class for all forecasting models.

    Subclasses implement :meth:`_forecast`, which may assume clean (imputed), finite
    history and may raise freely -- the public :meth:`forecast` wrapper handles
    imputation, timing, interval construction and failure fallback.
    """

    name: str = "base"
    #: Declared relative cost, used by the cost-aware analysis (H4) and to decide
    #: which models are restricted to a subsample on CPU.
    cost: float = 1.0
    requires_seasonality: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params

    # -- subclass API ------------------------------------------------------- #

    @abstractmethod
    def _forecast(
        self,
        history: np.ndarray,
        horizon: int,
        seasonal_period: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)``; ``samples`` may be ``None`` to use residuals.

        If ``samples`` is ``None`` the wrapper builds them from
        :meth:`_in_sample_residuals`.
        """

    def _in_sample_residuals(
        self, history: np.ndarray, seasonal_period: int
    ) -> np.ndarray:
        """Residuals used for bootstrap intervals when a model supplies no samples.

        The default is the seasonal difference, which is a deliberately conservative
        proxy: it overstates uncertainty for good models rather than understating it,
        and an over-wide interval is a less dangerous failure than an over-confident
        one.
        """
        m = max(1, int(seasonal_period))
        if history.size > m + 1:
            return np.diff(history) if m == 1 else history[m:] - history[:-m]
        return np.diff(history) if history.size > 1 else np.array([0.0])

    # -- public API --------------------------------------------------------- #

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        seasonal_period: int = 1,
        coverage_levels: Sequence[float] = (0.8, 0.95),
        seed: int = 0,
    ) -> ForecastResult:
        """Produce a forecast, never raising on model failure."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        clean, n_imputed = impute_missing(np.asarray(history, dtype=float))
        if clean.size == 0:
            raise ValueError("cannot forecast from empty history")

        fallback = False
        error_msg: str | None = None
        try:
            with warnings.catch_warnings():
                # Statistical libraries warn liberally about convergence; we record
                # the outcome via residual diagnostics instead of drowning the logs.
                warnings.simplefilter("ignore")
                point, samples = self._forecast(clean, horizon, seasonal_period, rng)
            point = np.asarray(point, dtype=float).reshape(-1)
            if point.shape[0] != horizon or not np.all(np.isfinite(point)):
                raise ValueError(
                    f"{self.name} produced an invalid point forecast "
                    f"(shape={point.shape}, finite={np.all(np.isfinite(point))})"
                )
        except Exception as exc:  # noqa: BLE001 - deliberate: degrade, don't crash
            fallback = True
            error_msg = f"{type(exc).__name__}: {exc}"
            point = _seasonal_naive_point(clean, horizon, seasonal_period)
            samples = None

        if samples is None:
            resid = self._in_sample_residuals(clean, seasonal_period)
            samples = residual_samples(point, resid, rng)
        samples = np.asarray(samples, dtype=float)

        lower, upper = intervals_from_samples(samples, coverage_levels)
        return ForecastResult(
            point=point,
            samples=samples,
            model_name=self.name,
            lower=lower,
            upper=upper,
            fit_seconds=time.perf_counter() - t0,
            fallback=fallback,
            metadata={
                "n_imputed": n_imputed,
                "history_length": int(clean.size),
                "error": error_msg,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


def _seasonal_naive_point(
    history: np.ndarray, horizon: int, seasonal_period: int
) -> np.ndarray:
    """Fallback used when a model fails. Always defined for non-empty history."""
    m = max(1, int(seasonal_period))
    if history.size >= m:
        season = history[-m:]
        reps = int(np.ceil(horizon / m))
        return np.tile(season, reps)[:horizon]
    return np.full(horizon, float(history[-1]))
