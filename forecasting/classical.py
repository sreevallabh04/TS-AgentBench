"""Classical and statistical forecasters.

These are the backbone of the benchmark. On CPU they are fast enough to run on every
instance, which is what makes the counterfactual tensor affordable, and they remain
genuinely competitive -- the M4 and Monash results are a standing reminder that a
well-specified statistical model is a hard baseline, not a formality.

Each model exposes the same probabilistic interface, so a naive baseline and an ARIMA
are compared on identical terms including CRPS.
"""

from __future__ import annotations

import numpy as np

from forecasting.base import FORECASTERS, Forecaster, residual_samples

__all__ = [
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "MeanForecaster",
    "ThetaForecaster",
    "ETSForecaster",
    "ARIMAForecaster",
    "DampedTrendForecaster",
]


@FORECASTERS.register("naive")
class NaiveForecaster(Forecaster):
    """Repeat the last observation. The random-walk benchmark."""

    name = "naive"
    cost = 0.01

    def _forecast(self, history, horizon, seasonal_period, rng):
        return np.full(horizon, float(history[-1])), None

    def _in_sample_residuals(self, history, seasonal_period):
        return np.diff(history) if history.size > 1 else np.array([0.0])


@FORECASTERS.register("seasonal_naive")
class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season. The denominator of MASE in seasonal settings."""

    name = "seasonal_naive"
    cost = 0.01
    requires_seasonality = True

    def _forecast(self, history, horizon, seasonal_period, rng):
        m = max(1, int(seasonal_period))
        if history.size < m:
            return np.full(horizon, float(history[-1])), None
        season = history[-m:]
        reps = int(np.ceil(horizon / m))
        return np.tile(season, reps)[:horizon], None


@FORECASTERS.register("drift")
class DriftForecaster(Forecaster):
    """Extrapolate the straight line through the first and last observation."""

    name = "drift"
    cost = 0.01

    def _forecast(self, history, horizon, seasonal_period, rng):
        n = history.size
        if n < 2:
            return np.full(horizon, float(history[-1])), None
        slope = (history[-1] - history[0]) / (n - 1)
        steps = np.arange(1, horizon + 1, dtype=float)
        return history[-1] + slope * steps, None


@FORECASTERS.register("mean")
class MeanForecaster(Forecaster):
    """Historical mean. Strong on pure noise, catastrophic on trends -- both useful."""

    name = "mean"
    cost = 0.01

    def _forecast(self, history, horizon, seasonal_period, rng):
        return np.full(horizon, float(np.mean(history))), None

    def _in_sample_residuals(self, history, seasonal_period):
        return history - float(np.mean(history))


@FORECASTERS.register("theta")
class ThetaForecaster(Forecaster):
    """The Theta method (M3 winner), implemented as SES + half the linear drift.

    Uses the classical decomposition: the theta-0 line is the OLS trend and theta-2 is
    simple exponential smoothing, combined with equal weights. Seasonality is handled
    by multiplicative deseasonalisation when a season is present.
    """

    name = "theta"
    cost = 0.05

    def _forecast(self, history, horizon, seasonal_period, rng):
        m = max(1, int(seasonal_period))
        x = history.astype(float)
        n = x.size

        # --- deseasonalise multiplicatively when there is enough data ---
        seasonal_idx = np.ones(m, dtype=float)
        deseasonalised = x
        if m > 1 and n >= 2 * m and np.all(x > 0):
            n_full = (n // m) * m
            mat = x[-n_full:].reshape(-1, m)
            seasonal_idx = mat.mean(axis=0) / mat.mean()
            seasonal_idx = np.where(seasonal_idx <= 0, 1.0, seasonal_idx)
            # Align the seasonal index with the series' phase.
            phase = np.arange(n) % m
            offset = (n - n_full) % m
            deseasonalised = x / seasonal_idx[(phase - offset) % m]

        # --- SES level ---
        alpha = 0.3
        level = deseasonalised[0]
        for value in deseasonalised[1:]:
            level = alpha * value + (1 - alpha) * level

        # --- OLS drift, halved (the theta-0 line contributes half the slope) ---
        t = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, deseasonalised, 1)[0]) if n > 2 else 0.0
        steps = np.arange(1, horizon + 1, dtype=float)
        point = level + 0.5 * slope * steps

        if m > 1 and not np.allclose(seasonal_idx, 1.0):
            future_phase = (np.arange(n, n + horizon) - (n - (n // m) * m)) % m
            point = point * seasonal_idx[future_phase % m]
        return point, None


@FORECASTERS.register("ets")
class ETSForecaster(Forecaster):
    """Exponential smoothing via statsmodels, with automatic configuration.

    Trend and seasonal components are enabled only when the history can support them;
    fitting a seasonal ETS to two cycles of data produces confident nonsense.
    """

    name = "ets"
    cost = 0.3

    def _forecast(self, history, horizon, seasonal_period, rng):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        m = max(1, int(seasonal_period))
        n = history.size
        use_seasonal = m > 1 and n >= 2 * m + 4
        use_trend = n >= 10

        model = ExponentialSmoothing(
            history,
            trend="add" if use_trend else None,
            damped_trend=bool(use_trend),
            seasonal="add" if use_seasonal else None,
            seasonal_periods=m if use_seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        point = np.asarray(fit.forecast(horizon), dtype=float)
        resid = np.asarray(fit.resid, dtype=float)
        samples = residual_samples(point, resid, rng)
        return point, samples


@FORECASTERS.register("arima")
class ARIMAForecaster(Forecaster):
    """ARIMA with a small automatic order search.

    A full auto.arima stepwise search is too slow to run on every instance of the
    counterfactual tensor, so we score a compact candidate set by AIC. The candidate
    set includes a differenced and an undifferenced branch, which is the choice that
    actually matters for the trend/drift families.
    """

    name = "arima"
    cost = 1.0

    _CANDIDATES = ((1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (1, 1, 0), (2, 0, 0))

    def _forecast(self, history, horizon, seasonal_period, rng):
        from statsmodels.tsa.arima.model import ARIMA

        best_fit, best_aic = None, np.inf
        for order in self._CANDIDATES:
            if history.size <= sum(order) + 3:
                continue
            try:
                fit = ARIMA(history, order=order).fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit
            except Exception:  # noqa: BLE001 - candidate search, keep going
                continue

        if best_fit is None:
            raise RuntimeError("no ARIMA candidate converged")

        forecast = best_fit.get_forecast(steps=horizon)
        point = np.asarray(forecast.predicted_mean, dtype=float)
        # Use the model's own variance path rather than bootstrapped residuals: it
        # encodes how uncertainty grows with the horizon for this specific process.
        sd = np.sqrt(np.maximum(forecast.var_pred_mean, 1e-12))
        samples = point[None, :] + rng.normal(0, 1, size=(200, horizon)) * sd[None, :]
        return point, samples


@FORECASTERS.register("damped_trend")
class DampedTrendForecaster(Forecaster):
    """Holt's linear trend with explicit damping.

    Kept as a distinct model because damping is also a *corrective action* (A4): when
    the diagnosis is trend over-extrapolation, this is the matched repair.
    """

    name = "damped_trend"
    cost = 0.2

    def __init__(self, phi: float = 0.9, **params):
        super().__init__(**params)
        self.phi = float(phi)

    def _forecast(self, history, horizon, seasonal_period, rng):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if history.size < 10:
            raise ValueError("insufficient history for a damped trend fit")
        fit = ExponentialSmoothing(
            history,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True, damping_trend=self.phi)
        point = np.asarray(fit.forecast(horizon), dtype=float)
        samples = residual_samples(point, np.asarray(fit.resid, dtype=float), rng)
        return point, samples
