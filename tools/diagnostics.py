"""Statistical diagnostic tools.

These produce the evidence the agent reasons over and the features the reliability
estimator is trained on. Everything here is computed from history alone.

Where a classical test has a well-known weakness, the implementation says so in a
comment rather than presenting the number as authoritative -- for instance ADF has low
power against near-unit-root alternatives, which is precisely the regime our
``near_unit_root`` family generates, so the agent should not treat a single p-value as
settled evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.base import TOOLS, Tool

__all__ = [
    "ACFTool",
    "STLDecompositionTool",
    "SeasonalityDetectionTool",
    "StationarityTool",
    "ChangePointTool",
    "AnomalyTool",
    "DistributionShiftTool",
    "TSFeaturesTool",
]


def _safe_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrelation up to ``nlags``, robust to constant series."""
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = float(np.dot(x[: x.size - k], x[k:]) / denom) if k < x.size else 0.0
    return out


@TOOLS.register("acf")
class ACFTool(Tool):
    """Autocorrelation and partial autocorrelation structure."""

    name = "acf"
    cost = 0.1
    description = (
        "Autocorrelation (ACF) and partial autocorrelation (PACF): reveals "
        "persistence, AR order, and seasonal lags."
    )

    def _run(self, history, seasonal_period, **kwargs) -> dict[str, Any]:
        from statsmodels.tsa.stattools import pacf

        nlags = int(min(max(2 * seasonal_period, 10), max(2, history.size // 3)))
        acf_vals = _safe_acf(history, nlags)
        try:
            pacf_vals = pacf(history, nlags=min(nlags, history.size // 2 - 1))
        except Exception:
            pacf_vals = np.zeros(nlags + 1)

        # Approximate 95% white-noise band.
        band = 1.96 / np.sqrt(history.size)
        significant = [int(k) for k in range(1, len(acf_vals)) if abs(acf_vals[k]) > band]
        return {
            "acf": [float(v) for v in acf_vals],
            "pacf": [float(v) for v in pacf_vals],
            "acf_lag1": float(acf_vals[1]) if len(acf_vals) > 1 else 0.0,
            "acf_at_season": (
                float(acf_vals[seasonal_period])
                if seasonal_period < len(acf_vals)
                else 0.0
            ),
            "n_significant_lags": len(significant),
            "significant_lags": significant[:10],
            "white_noise_band": float(band),
        }

    def summarize(self, values):
        return (
            f"ACF(1)={values['acf_lag1']:.2f}, "
            f"ACF(season)={values['acf_at_season']:.2f}, "
            f"{values['n_significant_lags']} significant lags"
        )


@TOOLS.register("stl")
class STLDecompositionTool(Tool):
    """Seasonal-trend decomposition and component strengths.

    Trend and seasonal *strength* follow the standard definition from the forecasting
    literature: one minus the ratio of the remainder variance to the variance of the
    remainder plus the component. Both lie in [0, 1] and are directly comparable
    across series of different scales.
    """

    name = "stl"
    cost = 0.5
    description = (
        "STL decomposition into trend, seasonal and remainder, with strength scores "
        "for trend and seasonality."
    )

    def _run(self, history, seasonal_period, **kwargs) -> dict[str, Any]:
        from statsmodels.tsa.seasonal import STL

        m = seasonal_period
        if m > 1 and history.size >= 2 * m + 1:
            res = STL(history, period=m, robust=True).fit()
            trend, seasonal, resid = res.trend, res.seasonal, res.resid
        else:
            # No usable season: fall back to a moving-average trend.
            win = max(3, min(history.size // 4 * 2 + 1, 25))
            kernel = np.ones(win) / win
            trend = np.convolve(history, kernel, mode="same")
            seasonal = np.zeros_like(history)
            resid = history - trend

        var_r = float(np.var(resid))
        trend_strength = max(0.0, 1.0 - var_r / max(float(np.var(resid + trend)), 1e-12))
        seasonal_strength = (
            max(0.0, 1.0 - var_r / max(float(np.var(resid + seasonal)), 1e-12))
            if np.any(seasonal)
            else 0.0
        )

        # Slope of the trend component over its final season -- what a naive linear
        # extrapolation would latch onto.
        tail = trend[-max(3, m) :]
        recent_slope = (
            float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size > 2 else 0.0
        )
        return {
            "trend_strength": float(min(1.0, trend_strength)),
            "seasonal_strength": float(min(1.0, seasonal_strength)),
            "recent_trend_slope": recent_slope,
            "resid_std": float(np.std(resid)),
            "series_std": float(np.std(history)),
        }

    def summarize(self, values):
        return (
            f"trend strength={values['trend_strength']:.2f}, "
            f"seasonal strength={values['seasonal_strength']:.2f}, "
            f"recent slope={values['recent_trend_slope']:.3g}"
        )


@TOOLS.register("seasonality")
class SeasonalityDetectionTool(Tool):
    """Detect candidate seasonal periods from the periodogram and the ACF.

    Two independent detectors are used because they fail differently: the periodogram
    is reliable for strong, stable cycles and the ACF is better for short series. When
    they disagree, that disagreement is itself informative and is reported rather than
    resolved arbitrarily.
    """

    name = "seasonality"
    cost = 0.2
    description = (
        "Detects candidate seasonal periods via spectral density and ACF peaks; "
        "flags multiple seasonality."
    )

    def _run(self, history, seasonal_period, **kwargs) -> dict[str, Any]:
        from scipy.signal import find_peaks, periodogram

        x = history - history.mean()
        n = x.size

        freqs, power = periodogram(x, detrend="linear")
        spectral_periods: list[int] = []
        if freqs.size > 2:
            order = np.argsort(power[1:])[::-1] + 1
            for idx in order[:5]:
                if freqs[idx] > 0:
                    period = int(round(1.0 / freqs[idx]))
                    if 2 <= period <= n // 2:
                        spectral_periods.append(period)

        nlags = min(n // 2 - 1, 3 * max(seasonal_period, 2))
        acf_vals = _safe_acf(x, max(2, nlags))
        peaks, _ = find_peaks(acf_vals[1:], height=1.96 / np.sqrt(n))
        acf_periods = [int(p + 1) for p in peaks[:5]]

        candidates = sorted(set(spectral_periods) | set(acf_periods))
        return {
            "spectral_periods": spectral_periods[:3],
            "acf_periods": acf_periods[:3],
            "candidate_periods": candidates[:5],
            "assumed_period": int(seasonal_period),
            # Multiple well-separated candidates suggest nested cycles, which a
            # single-period model cannot represent.
            "multiple_seasonality": bool(len({p for p in candidates if p > 2}) >= 2),
            "period_mismatch": bool(
                seasonal_period > 1
                and candidates
                and seasonal_period not in candidates
            ),
        }

    def summarize(self, values):
        return (
            f"candidates={values['candidate_periods']}, "
            f"assumed={values['assumed_period']}, "
            f"mismatch={values['period_mismatch']}"
        )


@TOOLS.register("stationarity")
class StationarityTool(Tool):
    """ADF and KPSS tests, reported jointly.

    ADF tests the null of a unit root; KPSS tests the null of stationarity. They are
    reported together because each alone is weak: ADF in particular has low power
    against near-unit-root alternatives, which is exactly the regime that most often
    breaks a forecaster. Agreement between them is far more informative than either
    p-value on its own.
    """

    name = "stationarity"
    cost = 0.3
    description = "ADF and KPSS stationarity tests with a combined verdict."

    def _run(self, history, seasonal_period, **kwargs) -> dict[str, Any]:
        from statsmodels.tsa.stattools import adfuller, kpss

        out: dict[str, Any] = {}
        try:
            adf_stat, adf_p, *_ = adfuller(history, autolag="AIC")
            out["adf_pvalue"] = float(adf_p)
            out["adf_rejects_unit_root"] = bool(adf_p < 0.05)
        except Exception:
            out["adf_pvalue"] = float("nan")
            out["adf_rejects_unit_root"] = False

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # KPSS warns on p-value clipping
                kpss_stat, kpss_p, *_ = kpss(history, regression="c", nlags="auto")
            out["kpss_pvalue"] = float(kpss_p)
            out["kpss_rejects_stationarity"] = bool(kpss_p < 0.05)
        except Exception:
            out["kpss_pvalue"] = float("nan")
            out["kpss_rejects_stationarity"] = False

        stationary = out["adf_rejects_unit_root"] and not out["kpss_rejects_stationarity"]
        nonstationary = (
            not out["adf_rejects_unit_root"] and out["kpss_rejects_stationarity"]
        )
        out["verdict"] = (
            "stationary" if stationary
            else "non-stationary" if nonstationary
            else "inconclusive"
        )
        return out

    def summarize(self, values):
        return (
            f"{values['verdict']} (ADF p={values['adf_pvalue']:.3g}, "
            f"KPSS p={values['kpss_pvalue']:.3g})"
        )


@TOOLS.register("changepoint")
class ChangePointTool(Tool):
    """Detect structural breaks in level with the PELT algorithm.

    The reported quantity that matters most is not the number of changepoints but
    ``steps_since_last_changepoint``: a break in the distant past is learnable, while
    one immediately before the forecast origin means the model was largely fitted to a
    regime that no longer holds.

    Three implementation choices were made empirically, against the synthetic families
    whose changepoints are known by construction (see ``tests/test_tools.py``):

    * **Standardise first.** The default ``rbf`` kernel cost is unusable on raw data:
      at a series level of ~750 all kernel distances saturate and the detector returns
      nothing at all, including on a textbook level shift.
    * **Remove the linear trend.** A steady trend is otherwise decomposed into a
      staircase of spurious mean shifts.
    * **``l2`` cost at a BIC-style penalty of ``5*log(n)``.** Benchmarked against the
      ``normal`` cost, which detects variance changes too but produced four times the
      false-alarm rate on families with no true break.

    Consequently this tool detects *level* changes only. Variance changes are reported
    separately by :class:`DistributionShiftTool`, which is the right split: a mean
    shift calls for refitting on the post-break segment, whereas a variance shift calls
    for wider intervals. Collapsing both into one "change score" would destroy the
    distinction the corrective action space relies on.

    Known limitation: strongly persistent series (near unit root) genuinely wander
    between levels, and this detector reports those excursions as breaks. That is a
    property of the process rather than a bug, but it means the signal is noisier for
    such series, and the paper reports detection quality per family.
    """

    name = "changepoint"
    cost = 2.0
    description = (
        "PELT change-point detection: locates structural breaks in level and reports "
        "recency relative to the forecast origin."
    )

    def _run(self, history, seasonal_period, penalty_mult: float = 5.0, **kwargs):
        import ruptures as rpt

        n = history.size
        min_size = max(5, min(seasonal_period, n // 6))
        if n < 3 * min_size:
            return {
                "changepoints": [],
                "n_changepoints": 0,
                "steps_since_last_changepoint": int(n),
                "relative_recency": 1.0,
                "last_segment_mean_shift": 0.0,
            }

        # Standardise, then remove the linear trend, then re-standardise so the
        # penalty is interpretable on a unit-variance scale.
        z = (history - history.mean()) / (float(history.std()) or 1.0)
        t = np.arange(z.size, dtype=float)
        z = z - np.polyval(np.polyfit(t, z, 1), t)
        z = z / (float(z.std()) or 1.0)

        algo = rpt.Pelt(model="l2", min_size=min_size, jump=max(1, n // 100)).fit(
            z.reshape(-1, 1)
        )
        # ruptures always returns n as the final boundary; drop it.
        bkps = [
            int(b)
            for b in algo.predict(pen=float(penalty_mult) * np.log(max(2, n)))
            if b < n
        ]

        last = bkps[-1] if bkps else 0
        since = n - last if bkps else n
        shift = 0.0
        if bkps and last > 0:
            before = history[max(0, last - min_size) : last]
            after = history[last:]
            if before.size and after.size:
                scale = float(np.std(history)) or 1.0
                shift = float(abs(after.mean() - before.mean()) / scale)

        return {
            "changepoints": bkps[:20],
            "n_changepoints": len(bkps),
            "steps_since_last_changepoint": int(since),
            # 0 means "a break just happened"; 1 means "no recent break".
            "relative_recency": float(min(1.0, since / max(1, n))),
            "last_segment_mean_shift": shift,
        }

    def summarize(self, values):
        return (
            f"{values['n_changepoints']} changepoints, last "
            f"{values['steps_since_last_changepoint']} steps ago "
            f"(shift={values['last_segment_mean_shift']:.2f} sd)"
        )


@TOOLS.register("anomaly")
class AnomalyTool(Tool):
    """Detect outliers via robust z-scores of STL remainders.

    Median/MAD rather than mean/std: the mean and standard deviation are themselves
    corrupted by the outliers we are trying to find, which is how naive detectors miss
    exactly the contamination that matters.
    """

    name = "anomaly"
    cost = 0.6
    description = (
        "Robust outlier detection on decomposition remainders; reports overall and "
        "recent-window anomaly rates."
    )

    def _run(self, history, seasonal_period, threshold: float = 3.5, **kwargs):
        m = seasonal_period
        if m > 1 and history.size >= 2 * m + 1:
            from statsmodels.tsa.seasonal import STL

            try:
                resid = STL(history, period=m, robust=True).fit().resid
            except Exception:
                resid = history - np.median(history)
        else:
            resid = np.diff(history, prepend=history[0])

        median = float(np.median(resid))
        mad = float(np.median(np.abs(resid - median)))
        # 0.6745 converts MAD to a standard-deviation-equivalent for normal data.
        scale = mad / 0.6745 if mad > 1e-12 else (float(np.std(resid)) or 1.0)
        z = np.abs(resid - median) / scale

        idx = np.flatnonzero(z > threshold)
        recent_window = max(seasonal_period, min(history.size, 24))
        recent = idx[idx >= history.size - recent_window]
        return {
            "anomaly_indices": [int(i) for i in idx[:50]],
            "n_anomalies": int(idx.size),
            "anomaly_rate": float(idx.size / history.size),
            "recent_anomaly_rate": float(recent.size / recent_window),
            "max_zscore": float(z.max()) if z.size else 0.0,
            # An anomaly at the final observation is especially dangerous: every
            # naive/drift-style forecast anchors on that point.
            "anomaly_at_last_point": bool(z[-1] > threshold) if z.size else False,
        }

    def summarize(self, values):
        return (
            f"{values['n_anomalies']} anomalies "
            f"({values['anomaly_rate']:.1%}), recent rate "
            f"{values['recent_anomaly_rate']:.1%}"
        )


@TOOLS.register("distribution_shift")
class DistributionShiftTool(Tool):
    """Compare the recent window against earlier history.

    Reports a two-sample KS statistic, a population stability index, and separate
    mean/variance ratios. Keeping mean and variance apart matters: a pure variance
    shift calls for wider intervals, while a mean shift calls for refitting. Collapsing
    them into one "drift score" would destroy exactly the distinction the corrective
    action space needs.
    """

    name = "distribution_shift"
    cost = 0.3
    description = (
        "KS test, PSI and mean/variance ratios between the recent window and earlier "
        "history."
    )

    def _run(self, history, seasonal_period, window: int | None = None, **kwargs):
        from scipy import stats

        n = history.size
        win = int(window or max(seasonal_period * 2, min(n // 4, 48)))
        win = max(4, min(win, n // 2))
        recent, past = history[-win:], history[:-win]
        if past.size < 4:
            return {
                "ks_statistic": 0.0,
                "ks_pvalue": 1.0,
                "psi": 0.0,
                "mean_ratio": 1.0,
                "variance_ratio": 1.0,
                "shift_detected": False,
            }

        ks_stat, ks_p = stats.ks_2samp(recent, past)

        # Population stability index over deciles of the historical distribution.
        edges = np.quantile(past, np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        p_hist, _ = np.histogram(past, bins=edges)
        p_recent, _ = np.histogram(recent, bins=edges)
        p1 = np.clip(p_hist / max(1, p_hist.sum()), 1e-6, None)
        p2 = np.clip(p_recent / max(1, p_recent.sum()), 1e-6, None)
        psi = float(np.sum((p2 - p1) * np.log(p2 / p1)))

        past_sd = float(np.std(past)) or 1e-9
        mean_ratio = float(abs(recent.mean() - past.mean()) / past_sd)
        var_ratio = float((np.std(recent) + 1e-9) / past_sd)

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "psi": psi,
            "mean_ratio": mean_ratio,
            "variance_ratio": var_ratio,
            "window": int(win),
            "shift_detected": bool(ks_p < 0.05 or psi > 0.25),
        }

    def summarize(self, values):
        return (
            f"KS={values['ks_statistic']:.2f} (p={values['ks_pvalue']:.3g}), "
            f"PSI={values['psi']:.2f}, var ratio={values['variance_ratio']:.2f}"
        )


@TOOLS.register("ts_features")
class TSFeaturesTool(Tool):
    """Cheap scale-free summary features.

    These are the FFORMPP/FFORMA-style features, and they serve a second purpose: the
    features-only control condition for H3 is built from exactly this tool, so the
    reliability estimator has to beat them to claim it is using forecast-specific
    evidence rather than series descriptors alone.
    """

    name = "ts_features"
    cost = 0.2
    description = (
        "Scale-free summary features: variation, skew, kurtosis, spectral entropy, "
        "stability, lumpiness, crossing rate."
    )

    def _run(self, history, seasonal_period, **kwargs):
        from scipy import stats
        from scipy.signal import periodogram

        n = history.size
        mean = float(np.mean(history))
        sd = float(np.std(history))

        # Spectral entropy: high means noise-like and hard to forecast.
        _, power = periodogram(history - mean, detrend="linear")
        power = power[1:]
        total = power.sum()
        if total > 0:
            p = power / total
            p = p[p > 0]
            spectral_entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))
        else:
            spectral_entropy = 1.0

        # Stability/lumpiness: variance of tiled means/variances (Hyndman features).
        block = max(2, min(seasonal_period if seasonal_period > 1 else 10, n // 3))
        n_blocks = n // block
        if n_blocks >= 2:
            tiles = history[: n_blocks * block].reshape(n_blocks, block)
            stability = float(np.var(tiles.mean(axis=1)))
            lumpiness = float(np.var(tiles.var(axis=1)))
            scale = (sd**2) or 1.0
            stability, lumpiness = stability / scale, lumpiness / (scale**2)
        else:
            stability = lumpiness = 0.0

        diffs = np.diff(history)
        return {
            "length": int(n),
            "mean": mean,
            "std": sd,
            "coefficient_of_variation": float(sd / abs(mean)) if abs(mean) > 1e-9 else 0.0,
            "skewness": float(stats.skew(history)),
            "kurtosis": float(stats.kurtosis(history)),
            "spectral_entropy": spectral_entropy,
            "stability": stability,
            "lumpiness": lumpiness,
            # How often the series crosses its own mean: a crude nonlinearity/noise cue.
            "crossing_rate": float(
                np.mean(np.diff(np.sign(history - mean)) != 0) if n > 1 else 0.0
            ),
            "diff_std_ratio": float(np.std(diffs) / sd) if sd > 1e-9 and n > 1 else 0.0,
            "n_missing_imputed": 0,
        }

    def summarize(self, values):
        return (
            f"n={values['length']}, CV={values['coefficient_of_variation']:.2f}, "
            f"spec. entropy={values['spectral_entropy']:.2f}"
        )
