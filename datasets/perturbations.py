"""Controlled perturbations for the robustness analysis.

Each perturbation is a single, monotone axis applied to an otherwise identical series,
so that "performance degrades under X" is a causal statement about X rather than a
correlation across heterogeneous datasets. The same underlying series is used at every
severity level, and the seed is derived from the series id, so the only thing that
changes between levels is the perturbation strength.

Perturbations are applied to **history only, never to the forecast target**. Corrupting
the target would change the quantity being predicted and would make the resulting error
uninterpretable -- we want to measure how a *degraded view of the past* affects the
forecast, not how well a method predicts corrupted data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from core.registry import Registry
from core.seed import seeded_rng
from datasets.base import TimeSeries

__all__ = ["PERTURBATIONS", "apply_perturbation", "perturbation_names", "SEVERITIES"]

PERTURBATIONS: Registry[Callable] = Registry("perturbation")

#: Severity grid shared by every axis, so results are comparable across axes.
SEVERITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def _protected_tail(series: TimeSeries, protect: int) -> int:
    """Index after which values must not be modified (the evaluation region)."""
    return max(1, series.n_timesteps - int(protect))


@PERTURBATIONS.register("gaussian_noise")
def gaussian_noise(series: TimeSeries, severity: float, protect: int,
                   rng: np.random.Generator) -> TimeSeries:
    """Add Gaussian noise scaled to a fraction of the series' own standard deviation."""
    values = series.values.copy()
    cut = _protected_tail(series, protect)
    scale = float(np.nanstd(series.target)) * float(severity)
    if scale <= 0:
        return series
    noise = rng.normal(0.0, scale, size=values[:cut].shape)
    values[:cut] = values[:cut] + noise
    return series.with_values(values)


@PERTURBATIONS.register("missing_observations")
def missing_observations(series: TimeSeries, severity: float, protect: int,
                         rng: np.random.Generator) -> TimeSeries:
    """Delete a fraction of historical observations at random."""
    values = series.values.copy()
    cut = _protected_tail(series, protect)
    rate = 0.30 * float(severity)
    n_drop = int(cut * rate)
    if n_drop <= 0:
        return series
    idx = rng.choice(cut, size=min(n_drop, cut - 8), replace=False)
    if values.ndim == 1:
        values[idx] = np.nan
    else:
        values[idx, 0] = np.nan
    return series.with_values(values)


@PERTURBATIONS.register("artificial_anomalies")
def artificial_anomalies(series: TimeSeries, severity: float, protect: int,
                         rng: np.random.Generator) -> TimeSeries:
    """Inject additive outliers into the history."""
    values = series.values.copy()
    cut = _protected_tail(series, protect)
    rate = 0.10 * float(severity)
    n = int(cut * rate)
    if n <= 0:
        return series
    sd = float(np.nanstd(series.target)) or 1.0
    idx = rng.choice(cut, size=min(n, cut - 1), replace=False)
    magnitude = rng.uniform(4.0, 8.0, size=idx.size) * sd * rng.choice([-1.0, 1.0], idx.size)
    if values.ndim == 1:
        values[idx] += magnitude
    else:
        values[idx, 0] += magnitude
    return series.with_values(values)


@PERTURBATIONS.register("regime_shift")
def regime_shift(series: TimeSeries, severity: float, protect: int,
                 rng: np.random.Generator) -> TimeSeries:
    """Insert an abrupt level shift shortly before the evaluation region.

    Placed just before the protected tail on purpose: this is the hardest and most
    operationally relevant case, where the break is recent enough that most of the
    fitting window predates it.
    """
    values = series.values.copy()
    cut = _protected_tail(series, protect)
    sd = float(np.nanstd(series.target)) or 1.0
    magnitude = 5.0 * sd * float(severity) * rng.choice([-1.0, 1.0])
    if magnitude == 0:
        return series
    start = max(1, cut - int(0.25 * cut))
    if values.ndim == 1:
        values[start:] += magnitude
    else:
        values[start:, 0] += magnitude
    return series.with_values(values)


@PERTURBATIONS.register("gradual_drift")
def gradual_drift(series: TimeSeries, severity: float, protect: int,
                  rng: np.random.Generator) -> TimeSeries:
    """Superimpose a slowly growing linear drift over the second half."""
    values = series.values.copy()
    n = series.n_timesteps
    sd = float(np.nanstd(series.target)) or 1.0
    start = n // 2
    slope = (3.0 * sd * float(severity)) / max(1, n - start)
    ramp = np.zeros(n, dtype=float)
    ramp[start:] = slope * np.arange(n - start, dtype=float)
    if values.ndim == 1:
        values = values + ramp
    else:
        values[:, 0] = values[:, 0] + ramp
    return series.with_values(values)


@PERTURBATIONS.register("seasonal_change")
def seasonal_change(series: TimeSeries, severity: float, protect: int,
                    rng: np.random.Generator) -> TimeSeries:
    """Change the seasonal amplitude partway through the series."""
    values = series.values.copy()
    n = series.n_timesteps
    m = max(2, series.seasonal_period)
    start = n // 2
    factor = 1.0 + 1.5 * float(severity)
    target = values if values.ndim == 1 else values[:, 0]
    # Amplify deviations from a local mean, which is where the seasonal signal lives.
    local_mean = float(np.nanmean(target[start:]))
    scaled = local_mean + (target[start:] - local_mean) * factor
    if values.ndim == 1:
        values[start:] = scaled
    else:
        values[start:, 0] = scaled
    return series.with_values(values)


@PERTURBATIONS.register("short_history")
def short_history(series: TimeSeries, severity: float, protect: int,
                  rng: np.random.Generator) -> TimeSeries:
    """Truncate the available history, keeping the evaluation region intact."""
    n = series.n_timesteps
    keep_fraction = 1.0 - 0.7 * float(severity)
    keep = max(int(protect) + 32, int(n * keep_fraction))
    if keep >= n:
        return series
    return series.with_values(series.values[-keep:])


def perturbation_names() -> list[str]:
    return PERTURBATIONS.names()


def apply_perturbation(
    series: TimeSeries,
    name: str,
    severity: float,
    protect: int,
    base_seed: int = 0,
) -> TimeSeries:
    """Apply one perturbation at one severity.

    ``protect`` is the number of trailing observations left untouched; it must cover
    the validation and test windows so that forecast targets are never corrupted.
    """
    if severity <= 0.0:
        return series
    rng = seeded_rng(base_seed, "perturbation", name, series.series_id, severity)
    fn = PERTURBATIONS.get(name)
    out = fn(series, float(severity), int(protect), rng)
    out.labels.generator_params["perturbation"] = name
    out.labels.generator_params["severity"] = float(severity)
    return out
