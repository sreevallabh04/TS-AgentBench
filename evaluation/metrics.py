"""Forecasting, probabilistic and decision metrics.

Conventions fixed once, here, so every arm is scored identically:

* **MASE is the primary metric.** It is scale-free, defined for series containing
  zeros (unlike MAPE), and its denominator is computed on the *training history only*.
  Using the full series to scale would leak test information into the denominator --
  a subtle leak that changes every reported number.
* **CRPS is computed from sample paths** using the energy form, so it is available for
  every model including ones with no closed-form predictive distribution.
* **Lower is better for every metric here**, without exception, so aggregation and
  ranking code never needs to special-case direction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "geometric_mean", "GEO_OFFSET",
    "mae", "rmse", "mase", "smape", "wape", "mase_denominator",
    "crps_from_samples", "coverage", "mean_interval_width", "winkler_score",
    "expected_calibration_error", "risk_coverage_curve", "aurc",
    "point_metrics", "probabilistic_metrics",
]

_EPS = 1e-12

#: Additive offset used by :func:`geometric_mean`.
#:
#: MASE takes the value 0 on instances a model predicts exactly, and log(0) is
#: undefined. Clipping instead of offsetting makes the result depend on an arbitrary
#: floor: clipping at 1e-6 versus 1e-9 moves the geometric mean of our test set from
#: 0.398 to 0.336, purely because log(1e-6) and log(1e-9) differ by 7. Relative
#: comparisons between policies are unaffected -- they shift together -- but an
#: absolute figure computed that way is not meaningful and must not be reported.
#:
#: An offset is the standard remedy. 1e-3 is small relative to typical MASE values
#: (median ~0.6 here) yet large enough that exact zeros contribute a bounded term.
GEO_OFFSET = 1e-3


def geometric_mean(values: np.ndarray, offset: float = GEO_OFFSET) -> float:
    """Offset geometric mean: ``exp(mean(log(x + offset))) - offset``.

    The right central tendency for a heavy-tailed, strictly non-negative error
    distribution. On this benchmark MASE spans 0 to 187 with a median of 0.61, so the
    arithmetic mean describes a handful of extreme instances rather than typical
    behaviour; the geometric mean does not.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if (x < 0).any():
        raise ValueError("geometric_mean requires non-negative values")
    return float(np.exp(np.mean(np.log(x + offset))) - offset)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mase_denominator(train: np.ndarray, seasonal_period: int = 1) -> float:
    """Mean absolute seasonal difference of the training history.

    Falls back to the first difference when the history is shorter than one season,
    and to 1.0 for a constant series, so MASE is always finite.
    """
    train = np.asarray(train, dtype=float)
    train = train[np.isfinite(train)]
    m = max(1, int(seasonal_period))
    if train.size > m:
        d = float(np.mean(np.abs(train[m:] - train[:-m])))
    elif train.size > 1:
        d = float(np.mean(np.abs(np.diff(train))))
    else:
        d = 0.0
    return d if d > _EPS else 1.0


def mase(actual: np.ndarray, pred: np.ndarray, denom: float) -> float:
    """Mean absolute scaled error. ``denom`` must come from training data only."""
    return mae(actual, pred) / max(float(denom), _EPS)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE in percent, on the 0-200 convention.

    Undefined where actual and prediction are both zero; those points are dropped
    rather than counted as zero error, which would flatter the model.
    """
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    valid = denom > _EPS
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(a[valid] - p[valid]) / denom[valid]) * 100.0)


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error: total error over total actual volume."""
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    total = float(np.sum(np.abs(a)))
    return float(np.sum(np.abs(a - p)) / max(total, _EPS))


# --------------------------------------------------------------------------- #
# Probabilistic metrics
# --------------------------------------------------------------------------- #


def crps_from_samples(actual: np.ndarray, samples: np.ndarray) -> float:
    """Continuous ranked probability score from sample paths.

    Uses the energy form ``CRPS = E|X - y| - 0.5 * E|X - X'|``. The second term is
    evaluated in O(n log n) via the sorted-sample identity
    ``E|X - X'| = 2/n^2 * sum_i (2i - n + 1) * x_(i)``; the naive pairwise version is
    O(n^2) and would dominate the runtime of the whole evaluation.

    ``samples`` has shape ``(n_samples, horizon)``; the score is averaged over the
    horizon.
    """
    actual = np.asarray(actual, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    n, h = samples.shape

    term1 = np.mean(np.abs(samples - actual[None, :]), axis=0)

    srt = np.sort(samples, axis=0)
    weights = (2.0 * np.arange(1, n + 1) - n - 1).reshape(-1, 1)
    term2 = (2.0 / (n * n)) * np.sum(weights * srt, axis=0)

    return float(np.mean(term1 - 0.5 * term2))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage: fraction of actuals inside the interval (PICP)."""
    a = np.asarray(actual, dtype=float)
    return float(np.mean((a >= np.asarray(lower)) & (a <= np.asarray(upper))))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def winkler_score(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float
) -> float:
    """Interval score: width plus a penalty for exclusions.

    Reported alongside coverage because coverage alone is trivially gamed -- an
    infinitely wide interval has perfect coverage. Winkler penalises that directly.
    """
    a = np.asarray(actual, dtype=float)
    lo, up = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    alpha = 1.0 - float(level)
    score = up - lo
    score = score + np.where(a < lo, (2.0 / alpha) * (lo - a), 0.0)
    score = score + np.where(a > up, (2.0 / alpha) * (a - up), 0.0)
    return float(np.mean(score))


# --------------------------------------------------------------------------- #
# Calibration and selective prediction
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    """ECE between predicted probabilities and binary outcomes.

    Defaults to equal-*mass* (quantile) bins rather than equal-width. With the skewed
    risk distributions produced here, equal-width bins leave most bins nearly empty
    and the resulting ECE is dominated by sampling noise in those bins.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(outcomes, dtype=float).ravel()
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return float("nan")

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            return float(abs(p.mean() - y.mean()))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / p.size) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def risk_coverage_curve(
    losses: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Selective-prediction risk-coverage curve.

    Instances are ranked by confidence (high first); at each coverage level the risk
    is the mean loss over the retained instances. A useful confidence signal produces
    a curve that rises with coverage -- the forecasts it was least sure about really
    were the worse ones.
    """
    losses = np.asarray(losses, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    ok = np.isfinite(losses) & np.isfinite(confidence)
    losses, confidence = losses[ok], confidence[ok]
    if losses.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(-confidence, kind="stable")
    ordered = losses[order]
    n = ordered.size
    coverages = np.arange(1, n + 1, dtype=float) / n
    risks = np.cumsum(ordered) / np.arange(1, n + 1)
    return coverages, risks


def aurc(losses: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    cov, risk = risk_coverage_curve(losses, confidence)
    if cov.size == 0:
        return float("nan")
    return float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(
        np.trapz(risk, cov)
    )


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #


def point_metrics(
    actual: np.ndarray, pred: np.ndarray, denom: float
) -> dict[str, float]:
    return {
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "mase": mase(actual, pred, denom),
        "smape": smape(actual, pred),
        "wape": wape(actual, pred),
    }


def probabilistic_metrics(
    actual: np.ndarray,
    samples: np.ndarray,
    lower: dict[float, np.ndarray],
    upper: dict[float, np.ndarray],
    denom: float = 1.0,
) -> dict[str, float]:
    """CRPS plus per-level coverage, width and Winkler score.

    CRPS is also reported scaled by the MASE denominator, so it is comparable across
    series of different magnitudes -- raw CRPS is scale-dependent and aggregating it
    across heterogeneous series would simply rank series by size.
    """
    out: dict[str, float] = {
        "crps": crps_from_samples(actual, samples),
    }
    out["crps_scaled"] = out["crps"] / max(float(denom), _EPS)
    for level in sorted(lower):
        tag = int(round(level * 100))
        out[f"coverage_{tag}"] = coverage(actual, lower[level], upper[level])
        out[f"width_{tag}"] = mean_interval_width(lower[level], upper[level])
        out[f"winkler_{tag}"] = winkler_score(actual, lower[level], upper[level], level)
        out[f"coverage_error_{tag}"] = abs(out[f"coverage_{tag}"] - level)
    return out
