"""Controlled synthetic time-series generators.

These exist for one reason: on synthetic data we *know* the answer. We know where the
level shift is, which points are outliers, whether the series is stationary, and which
failure mode a forecaster is likely to hit. Real data cannot give us that, so every
causal claim in the paper (H2, H3) rests on these generators, and the real datasets
supply external validity instead.

Two design decisions are load-bearing:

**Perturbations are positioned relative to the end of the series.** Whether a level
shift falls before or after the forecast origin completely determines whether any
corrective action could have helped. A shift in the distant past is learnable; a shift
inside the forecast window is not. Families that inject structural breaks therefore
place them in a configurable late window, and record the exact index, so the analysis
can condition on it rather than averaging two opposite situations together.

**Every generator records its parameters.** ``SeriesLabels.generator_params`` holds
enough to regenerate the series exactly, which keeps the benchmark reproducible
without shipping large data files.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from core.registry import Registry
from datasets.base import FailureMode, SeriesLabels, TimeSeries

__all__ = ["SYNTHETIC_FAMILIES", "generate_series", "generate_family", "family_names"]

# A generator takes (rng, n, period, **params) and returns (values, labels).
GeneratorFn = Callable[..., tuple[np.ndarray, SeriesLabels]]
SYNTHETIC_FAMILIES: Registry[tuple[np.ndarray, SeriesLabels]] = Registry(
    "synthetic family"
)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def _linear_trend(n: int, slope: float, intercept: float = 0.0) -> np.ndarray:
    return intercept + slope * np.arange(n, dtype=float)


def _damped_trend(n: int, slope: float, phi: float) -> np.ndarray:
    """Trend whose increments decay geometrically.

    This is the classic trap for linear extrapolation: the recent history looks
    linear, so a naive trend model over-extrapolates. It is the generator behind
    ``TREND_MISSPEC``.
    """
    steps = slope * phi ** np.arange(n, dtype=float)
    return np.cumsum(steps)


def _seasonal(n: int, period: int, amplitude: float, rng: np.random.Generator,
              n_harmonics: int = 2) -> np.ndarray:
    """Sum of harmonics with random phases -- richer than a pure sine."""
    t = np.arange(n, dtype=float)
    out = np.zeros(n, dtype=float)
    for h in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        out += (amplitude / h) * np.sin(2 * np.pi * h * t / period + phase)
    return out


def _ar_process(
    n: int, coeffs: np.ndarray, sigma: float, rng: np.random.Generator,
    burn_in: int = 200,
) -> np.ndarray:
    """Simulate an AR(p) process, discarding a burn-in so it starts stationary."""
    p = len(coeffs)
    total = n + burn_in
    x = np.zeros(total, dtype=float)
    eps = rng.normal(0.0, sigma, size=total)
    for t in range(p, total):
        x[t] = float(np.dot(coeffs, x[t - p : t][::-1])) + eps[t]
    return x[burn_in:]


def _stable_ar_coeffs(rng: np.random.Generator, order: int, max_root: float = 0.9
                      ) -> np.ndarray:
    """Draw AR coefficients guaranteed stationary.

    Sampling coefficients directly would frequently produce explosive processes.
    Instead we sample the *roots* of the characteristic polynomial inside the unit
    circle and expand. For ``prod(z - r_i) = z^p + c_1 z^(p-1) + ...`` matched against
    ``z^p - a_1 z^(p-1) - ...``, the AR coefficients are ``a = -c``.
    """
    roots = rng.uniform(-max_root, max_root, size=order)
    poly = np.poly(roots)
    return -poly[1:]


def _late_index(rng: np.random.Generator, n: int, lo_frac: float, hi_frac: float) -> int:
    """Pick an index inside a late window of the series."""
    lo, hi = int(n * lo_frac), int(n * hi_frac)
    lo, hi = max(1, lo), max(2, min(n - 1, hi))
    if hi <= lo:
        return max(1, min(n - 1, lo))
    return int(rng.integers(lo, hi))


def _scale(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give series realistic, heterogeneous levels and scales.

    Without this every series sits near zero with unit variance, which makes
    scale-sensitive metrics and features behave unrealistically well.
    """
    level = rng.uniform(10.0, 1000.0)
    scale = rng.uniform(0.5, 5.0)
    return level + scale * values


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #


@SYNTHETIC_FAMILIES.register("trend_linear")
def _trend_linear(rng, n, period, noise=1.0, **_):
    slope = rng.uniform(0.05, 0.5) * rng.choice([-1.0, 1.0])
    signal = _linear_trend(n, slope) + rng.normal(0, noise, n)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=True,
        has_seasonality=False,
        is_stationary=False,
        expected_failure_modes=(FailureMode.NONE,),
        generator_params={"slope": float(slope), "noise": float(noise)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("trend_damped")
def _trend_damped(rng, n, period, noise=1.0, **_):
    """Saturating trend: punishes linear extrapolation -> TREND_MISSPEC."""
    slope = rng.uniform(1.0, 4.0)
    phi = rng.uniform(0.95, 0.995)
    signal = _damped_trend(n, slope, phi) + rng.normal(0, noise, n)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=True,
        has_seasonality=False,
        is_stationary=False,
        expected_failure_modes=(FailureMode.TREND_MISSPEC,),
        generator_params={"slope": float(slope), "phi": float(phi)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("seasonal_single")
def _seasonal_single(rng, n, period, noise=1.0, **_):
    amp = rng.uniform(2.0, 10.0)
    signal = _seasonal(n, period, amp, rng) + rng.normal(0, noise, n)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=True,
        expected_failure_modes=(FailureMode.NONE,),
        generator_params={"amplitude": float(amp), "period": int(period)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("seasonal_multiple")
def _seasonal_multiple(rng, n, period, noise=1.0, **_):
    """Two nested seasonal cycles -- single-period models are misspecified."""
    p2 = period * int(rng.choice([2, 3, 4]))
    a1, a2 = rng.uniform(2.0, 8.0), rng.uniform(2.0, 8.0)
    signal = (
        _seasonal(n, period, a1, rng)
        + _seasonal(n, p2, a2, rng, n_harmonics=1)
        + rng.normal(0, noise, n)
    )
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(int(period), int(p2)),
        is_stationary=True,
        expected_failure_modes=(FailureMode.SEASONAL_MISSPEC,),
        generator_params={"periods": [int(period), int(p2)]},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("ar_stationary")
def _ar_stationary(rng, n, period, noise=1.0, **_):
    order = int(rng.integers(1, 4))
    coeffs = _stable_ar_coeffs(rng, order)
    signal = _ar_process(n, coeffs, noise, rng)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=False,
        is_stationary=True,
        expected_failure_modes=(FailureMode.NONE,),
        generator_params={"ar_order": order, "coeffs": [float(c) for c in coeffs]},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("near_unit_root")
def _near_unit_root(rng, n, period, noise=1.0, **_):
    """Strong persistence: long-term dependence, hard to distinguish from a trend."""
    phi = rng.uniform(0.96, 0.999)
    signal = _ar_process(n, np.array([phi]), noise, rng, burn_in=500)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=False,
        is_stationary=False,
        expected_failure_modes=(FailureMode.DRIFT, FailureMode.TREND_MISSPEC),
        generator_params={"phi": float(phi)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("level_shift")
def _level_shift(rng, n, period, noise=1.0, shift_lo=0.55, shift_hi=0.95, **_):
    """Abrupt structural break placed late in the series.

    The position window straddles the validation/test boundary deliberately: some
    shifts are learnable from validation data, some are not, and conditioning on that
    is exactly what H2 needs.
    """
    base = _seasonal(n, period, rng.uniform(1.0, 4.0), rng) + rng.normal(0, noise, n)
    cp = _late_index(rng, n, shift_lo, shift_hi)
    magnitude = rng.uniform(4.0, 15.0) * rng.choice([-1.0, 1.0])
    signal = base.copy()
    signal[cp:] += magnitude
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=False,
        changepoints=(int(cp),),
        expected_failure_modes=(FailureMode.LEVEL_SHIFT,),
        generator_params={
            "changepoint": int(cp),
            "magnitude": float(magnitude),
            "relative_position": float(cp / n),
        },
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("gradual_drift")
def _gradual_drift(rng, n, period, noise=1.0, **_):
    """Concept drift: the generating trend slowly rotates in the second half."""
    start = _late_index(rng, n, 0.4, 0.6)
    drift_slope = rng.uniform(0.05, 0.4) * rng.choice([-1.0, 1.0])
    signal = _seasonal(n, period, rng.uniform(1.0, 4.0), rng) + rng.normal(0, noise, n)
    ramp = np.zeros(n, dtype=float)
    ramp[start:] = drift_slope * np.arange(n - start, dtype=float)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=True,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=False,
        changepoints=(int(start),),
        expected_failure_modes=(FailureMode.DRIFT,),
        generator_params={"drift_start": int(start), "drift_slope": float(drift_slope)},
    )
    return _scale(signal + ramp, rng), labels


@SYNTHETIC_FAMILIES.register("anomaly_contaminated")
def _anomaly_contaminated(rng, n, period, noise=1.0, anomaly_rate=0.03, **_):
    """Additive and innovational outliers.

    Additive outliers corrupt a single observation; innovational outliers propagate
    through the process. Both are present because they call for different repairs --
    which is the point of a typed action space.
    """
    coeffs = _stable_ar_coeffs(rng, 1)
    signal = _ar_process(n, coeffs, noise, rng)
    signal += _seasonal(n, period, rng.uniform(1.0, 4.0), rng)

    n_anom = max(1, int(n * anomaly_rate))
    # Keep outliers out of the first few points so models have clean initial state.
    idx = rng.choice(np.arange(period, n), size=min(n_anom, n - period), replace=False)
    sd = float(np.std(signal)) or 1.0
    innovational = rng.random(len(idx)) < 0.3
    for i, is_inno in zip(idx, innovational):
        magnitude = rng.uniform(4.0, 9.0) * sd * rng.choice([-1.0, 1.0])
        if is_inno:
            # Propagates forward with geometric decay.
            decay = 0.7 ** np.arange(n - i, dtype=float)
            signal[i:] += magnitude * decay
        else:
            signal[i] += magnitude

    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=True,
        anomaly_indices=tuple(int(i) for i in np.sort(idx)),
        expected_failure_modes=(FailureMode.ANOMALY_CONTAMINATION,),
        generator_params={
            "anomaly_rate": float(anomaly_rate),
            "n_innovational": int(innovational.sum()),
        },
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("variance_shift")
def _variance_shift(rng, n, period, noise=1.0, **_):
    """Volatility regime change: the point forecast may be fine, the interval is not."""
    cp = _late_index(rng, n, 0.5, 0.9)
    ratio = rng.uniform(3.0, 8.0)
    sd = np.full(n, noise, dtype=float)
    sd[cp:] *= ratio
    signal = _seasonal(n, period, rng.uniform(1.0, 4.0), rng) + rng.normal(0, 1, n) * sd
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=False,
        changepoints=(int(cp),),
        expected_failure_modes=(FailureMode.VARIANCE_SHIFT,),
        generator_params={"changepoint": int(cp), "variance_ratio": float(ratio)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("regime_switching")
def _regime_switching(rng, n, period, noise=1.0, **_):
    """Markov-switching AR with two regimes; records per-timestep regime labels."""
    p_stay = rng.uniform(0.95, 0.99)
    means = np.array([rng.uniform(-5, 0), rng.uniform(2, 8)])
    phis = np.array([rng.uniform(0.2, 0.6), rng.uniform(-0.5, 0.2)])

    regimes = np.zeros(n, dtype=int)
    state = int(rng.integers(0, 2))
    for t in range(n):
        if rng.random() > p_stay:
            state = 1 - state
        regimes[t] = state

    signal = np.zeros(n, dtype=float)
    eps = rng.normal(0, noise, n)
    for t in range(1, n):
        s = regimes[t]
        signal[t] = means[s] + phis[s] * (signal[t - 1] - means[s]) + eps[t]

    switches = tuple(int(i) for i in np.flatnonzero(np.diff(regimes) != 0) + 1)
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=False,
        is_stationary=False,
        changepoints=switches,
        regime_labels=regimes,
        expected_failure_modes=(FailureMode.LEVEL_SHIFT, FailureMode.DRIFT),
        generator_params={"p_stay": float(p_stay), "n_switches": len(switches)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("short_history")
def _short_history(rng, n, period, noise=1.0, **_):
    """Deliberately truncated series: too little history to identify structure."""
    short_n = max(period * 3, int(n * rng.uniform(0.12, 0.22)))
    short_n = min(short_n, n)
    signal = (
        _linear_trend(short_n, rng.uniform(-0.3, 0.3))
        + _seasonal(short_n, period, rng.uniform(2.0, 6.0), rng)
        + rng.normal(0, noise, short_n)
    )
    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=True,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=False,
        expected_failure_modes=(FailureMode.SHORT_HISTORY,),
        generator_params={"length": int(short_n)},
    )
    return _scale(signal, rng), labels


@SYNTHETIC_FAMILIES.register("missing_blocks")
def _missing_blocks(rng, n, period, noise=1.0, missing_rate=0.08, **_):
    """Contiguous blocks of missing observations (sensor dropout)."""
    signal = (
        _seasonal(n, period, rng.uniform(2.0, 6.0), rng)
        + _linear_trend(n, rng.uniform(-0.1, 0.1))
        + rng.normal(0, noise, n)
    )
    signal = _scale(signal, rng)

    # Missingness is confined to the first 70% of the series. Forecasting with gaps in
    # the *history* is the realistic problem; a NaN in a forecast *target* would
    # instead be an evaluation artifact, silently dropping instances and biasing the
    # comparison toward whichever arm happened to be evaluated on the easy remainder.
    gap_limit = int(n * 0.70)
    n_missing = int(n * missing_rate)
    missing: list[int] = []
    guard = 0
    while len(set(missing)) < n_missing and guard < 1000:
        guard += 1
        block = int(rng.integers(2, max(3, period)))
        start = int(rng.integers(period, max(period + 1, gap_limit - block)))
        missing.extend(range(start, min(start + block, gap_limit)))
    missing_idx = sorted(set(missing))[:n_missing]
    signal[missing_idx] = np.nan

    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=True,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=False,
        missing_indices=tuple(int(i) for i in missing_idx),
        expected_failure_modes=(FailureMode.NONE,),
        generator_params={"missing_rate": float(missing_rate)},
    )
    return signal, labels


@SYNTHETIC_FAMILIES.register("multivariate_var")
def _multivariate_var(rng, n, period, noise=1.0, n_variates=3, **_):
    """VAR(1) with genuine cross-series dependence.

    Column 0 is the forecast target; the remaining columns are informative covariates,
    so a model that ignores them is leaving signal on the table.
    """
    d = int(n_variates)
    # Scale a random matrix so its spectral radius is < 1 (stationary VAR).
    A = rng.normal(0, 0.4, size=(d, d))
    radius = max(abs(np.linalg.eigvals(A)))
    A *= rng.uniform(0.5, 0.85) / (radius if radius > 0 else 1.0)

    burn = 200
    x = np.zeros((n + burn, d), dtype=float)
    eps = rng.normal(0, noise, size=(n + burn, d))
    for t in range(1, n + burn):
        x[t] = A @ x[t - 1] + eps[t]
    values = x[burn:]
    values[:, 0] += _seasonal(n, period, rng.uniform(1.0, 4.0), rng)

    level = rng.uniform(10.0, 200.0)
    values = level + rng.uniform(0.5, 3.0) * values

    labels = SeriesLabels(
        is_ground_truth=True,
        has_trend=False,
        has_seasonality=True,
        seasonal_periods=(period,),
        is_stationary=True,
        expected_failure_modes=(FailureMode.MODEL_MISMATCH,),
        generator_params={"n_variates": d, "spectral_radius": float(radius)},
    )
    return values, labels


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def family_names() -> list[str]:
    return SYNTHETIC_FAMILIES.names()


def generate_series(
    family: str,
    series_id: str,
    rng: np.random.Generator,
    n_timesteps: int = 400,
    seasonal_period: int = 12,
    noise: float = 1.0,
    **params,
) -> TimeSeries:
    """Generate one labelled synthetic series from ``family``."""
    fn = SYNTHETIC_FAMILIES.get(family)
    values, labels = fn(rng, n_timesteps, seasonal_period, noise=noise, **params)
    labels.generator_params.update(
        {
            "family": family,
            "n_timesteps_requested": int(n_timesteps),
            "seasonal_period": int(seasonal_period),
            "noise": float(noise),
        }
    )
    return TimeSeries(
        series_id=series_id,
        values=values,
        seasonal_period=int(seasonal_period),
        freq="synthetic",
        dataset=f"synthetic_{family}",
        source="synthetic",
        family=family,
        labels=labels,
    )


def generate_family(
    family: str,
    n_series: int,
    base_seed: int,
    n_timesteps: int = 400,
    seasonal_period: int = 12,
    noise_range: tuple[float, float] = (0.5, 2.0),
    **params,
) -> list[TimeSeries]:
    """Generate ``n_series`` series from one family.

    Each series derives its own RNG from ``(base_seed, family, index)`` rather than
    drawing from a shared stream, so generation is order-independent and safe to
    parallelise -- see :func:`core.seed.derive_seed`.
    """
    from core.seed import seeded_rng

    out: list[TimeSeries] = []
    for i in range(n_series):
        rng = seeded_rng(base_seed, "synthetic", family, i)
        noise = float(rng.uniform(*noise_range))
        out.append(
            generate_series(
                family=family,
                series_id=f"syn_{family}_{i:04d}",
                rng=rng,
                n_timesteps=n_timesteps,
                seasonal_period=seasonal_period,
                noise=noise,
                **params,
            )
        )
    return out
