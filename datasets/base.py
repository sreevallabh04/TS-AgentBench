"""Core data structures for TS-AgentBench.

Two ideas carry most of the weight here.

**Ground-truth labels travel with the series.** A synthetic generator knows exactly
where it put a level shift, which points are outliers, and which failure mode it was
built to induce. Carrying that provenance through to evaluation is what makes the
causal analysis in H2/H3 possible; on real data those fields are absent and the
analysis says so rather than inventing them.

**An instance is a (series, origin, horizon) triple**, not a series. The correction
decision is made per forecast, so the unit of analysis has to be per forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "FailureMode",
    "SeriesLabels",
    "TimeSeries",
    "TemporalSplit",
    "Instance",
    "make_split",
    "iter_instances",
]


class FailureMode(str, Enum):
    """Failure modes a forecast can exhibit.

    Deliberately a small, closed set: each mode has at least one matched corrective
    action in :mod:`benchmark.actions`, which is what makes failure-mode-conditioned
    repair meaningful rather than decorative.
    """

    NONE = "none"
    TREND_MISSPEC = "trend_misspecification"
    SEASONAL_MISSPEC = "seasonality_misspecification"
    LEVEL_SHIFT = "level_shift"
    ANOMALY_CONTAMINATION = "anomaly_contamination"
    VARIANCE_SHIFT = "variance_shift"
    SHORT_HISTORY = "insufficient_history"
    DRIFT = "drift_nonstationarity"
    MODEL_MISMATCH = "model_class_mismatch"


@dataclass
class SeriesLabels:
    """Ground-truth structure of a series.

    Populated by synthetic generators. For real data every field stays at its default
    and :attr:`is_ground_truth` is False -- downstream analysis checks this flag before
    treating a label as truth.
    """

    is_ground_truth: bool = False
    has_trend: bool | None = None
    has_seasonality: bool | None = None
    seasonal_periods: tuple[int, ...] = ()
    is_stationary: bool | None = None
    changepoints: tuple[int, ...] = ()
    anomaly_indices: tuple[int, ...] = ()
    regime_labels: np.ndarray | None = None
    missing_indices: tuple[int, ...] = ()
    expected_failure_modes: tuple[FailureMode, ...] = ()
    # Free-form generator parameters, recorded so any series can be regenerated.
    generator_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "is_ground_truth": self.is_ground_truth,
            "has_trend": self.has_trend,
            "has_seasonality": self.has_seasonality,
            "seasonal_periods": list(self.seasonal_periods),
            "is_stationary": self.is_stationary,
            "changepoints": list(self.changepoints),
            "n_anomalies": len(self.anomaly_indices),
            "n_missing": len(self.missing_indices),
            "expected_failure_modes": [m.value for m in self.expected_failure_modes],
            "generator_params": dict(self.generator_params),
        }
        return d


@dataclass
class TimeSeries:
    """A single (possibly multivariate) time series with provenance.

    ``values`` has shape ``(T,)`` for univariate series and ``(T, D)`` for
    multivariate ones. The forecast target is always column 0; extra columns are
    covariates available to models that accept them.
    """

    series_id: str
    values: np.ndarray
    seasonal_period: int = 1
    freq: str = "unknown"
    dataset: str = "unknown"
    source: str = "synthetic"  # "synthetic" | "real"
    family: str | None = None  # synthetic generator family, if applicable
    labels: SeriesLabels = field(default_factory=SeriesLabels)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim not in (1, 2):
            raise ValueError(
                f"{self.series_id}: values must be 1-D or 2-D, got shape "
                f"{self.values.shape}"
            )
        if self.values.shape[0] == 0:
            raise ValueError(f"{self.series_id}: empty series")
        if self.seasonal_period < 1:
            raise ValueError(f"{self.series_id}: seasonal_period must be >= 1")

    @property
    def n_timesteps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_variates(self) -> int:
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])

    @property
    def is_multivariate(self) -> bool:
        return self.n_variates > 1

    @property
    def target(self) -> np.ndarray:
        """The univariate series being forecast (column 0 if multivariate)."""
        return self.values if self.values.ndim == 1 else self.values[:, 0]

    @property
    def has_missing(self) -> bool:
        return bool(np.isnan(self.values).any())

    def __len__(self) -> int:
        return self.n_timesteps

    def with_values(self, values: np.ndarray) -> "TimeSeries":
        """Copy carrying new values; used by robustness perturbations."""
        return replace(self, values=np.asarray(values, dtype=float))

    def describe(self) -> dict[str, Any]:
        t = self.target
        finite = t[np.isfinite(t)]
        return {
            "series_id": self.series_id,
            "dataset": self.dataset,
            "source": self.source,
            "family": self.family,
            "n_timesteps": self.n_timesteps,
            "n_variates": self.n_variates,
            "seasonal_period": self.seasonal_period,
            "freq": self.freq,
            "n_missing": int(np.isnan(t).sum()),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
        }


@dataclass(frozen=True)
class TemporalSplit:
    """Index boundaries of a strictly ordered temporal split.

    Half-open intervals: train is ``[0, train_end)``, validation
    ``[train_end, val_end)``, test ``[val_end, test_end)``.

    Frozen because a split that can be mutated after the fact is a leakage vector.
    """

    train_end: int
    val_end: int
    test_end: int

    def __post_init__(self) -> None:
        if not 0 < self.train_end < self.val_end < self.test_end:
            raise ValueError(
                "split boundaries must be strictly increasing and positive: "
                f"train_end={self.train_end}, val_end={self.val_end}, "
                f"test_end={self.test_end}"
            )

    @property
    def n_train(self) -> int:
        return self.train_end

    @property
    def n_val(self) -> int:
        return self.val_end - self.train_end

    @property
    def n_test(self) -> int:
        return self.test_end - self.val_end


@dataclass(frozen=True)
class Instance:
    """One forecasting task: predict ``horizon`` steps from ``origin``.

    ``origin`` is an exclusive index -- history is ``values[:origin]`` and the target
    is ``values[origin : origin + horizon]``. The off-by-one here is the single most
    dangerous index in the codebase, so it is stated once and enforced by
    :meth:`history` / :meth:`target_slice`.
    """

    instance_id: str
    series_id: str
    origin: int
    horizon: int
    split: str  # "val" | "test"

    def history(self, series: TimeSeries) -> np.ndarray:
        return series.target[: self.origin]

    def target_slice(self, series: TimeSeries) -> np.ndarray:
        return series.target[self.origin : self.origin + self.horizon]

    def history_multivariate(self, series: TimeSeries) -> np.ndarray:
        return series.values[: self.origin]


def make_split(
    n: int,
    val_length: int,
    test_length: int,
    min_train_length: int = 32,
) -> TemporalSplit:
    """Build a temporal split for a series of length ``n``.

    Shrinks the validation and test windows proportionally when a series is too short
    to honour the requested sizes, rather than silently eating into training data.
    Raises if even the shrunk split would leave less than ``min_train_length``
    observations for training -- such a series is dropped rather than half-evaluated.
    """
    if n < min_train_length + 2:
        raise ValueError(
            f"series of length {n} is too short for min_train_length="
            f"{min_train_length}"
        )

    val_len, test_len = int(val_length), int(test_length)
    budget = n - min_train_length
    if val_len + test_len > budget:
        # Preserve the requested val:test ratio while fitting the budget.
        total = val_len + test_len
        scale = budget / total
        val_len = max(1, int(val_len * scale))
        test_len = max(1, budget - val_len)

    train_end = n - val_len - test_len
    if train_end < min_train_length:
        raise ValueError(
            f"cannot build split for series of length {n}: training window would be "
            f"{train_end} < {min_train_length}"
        )
    return TemporalSplit(train_end, train_end + val_len, n)


def iter_instances(
    series: TimeSeries,
    split: TemporalSplit,
    horizons: Sequence[int],
    n_val_origins: int,
    n_test_origins: int,
    origin_stride: int,
) -> Iterator[Instance]:
    """Yield rolling-origin instances for validation and test windows.

    Origins are placed so that the full horizon fits inside the corresponding window:
    a target must never extend past ``val_end`` (for validation instances) or past the
    end of the series (for test instances). Validation instances therefore never see
    a single test observation, which is what lets the risk estimator and every
    threshold be fit without leakage.
    """
    for horizon in horizons:
        # --- validation origins: target must end at or before val_end ---
        last_val_origin = split.val_end - horizon
        for k in range(n_val_origins):
            origin = last_val_origin - k * origin_stride
            if origin < split.train_end or origin <= 0:
                break
            yield Instance(
                instance_id=f"{series.series_id}|val|h{horizon}|o{origin}",
                series_id=series.series_id,
                origin=origin,
                horizon=horizon,
                split="val",
            )

        # --- test origins: target must end at or before the series end ---
        last_test_origin = split.test_end - horizon
        for k in range(n_test_origins):
            origin = last_test_origin - k * origin_stride
            if origin < split.val_end or origin <= 0:
                break
            yield Instance(
                instance_id=f"{series.series_id}|test|h{horizon}|o{origin}",
                series_id=series.series_id,
                origin=origin,
                horizon=horizon,
                split="test",
            )
