"""Machine-learning forecasters (linear, random forest, gradient boosting).

Two modelling choices are worth stating explicitly, because both affect how strong
these baselines are and a reviewer will ask.

**Differencing.** Tree ensembles cannot extrapolate beyond the range of their training
targets, so on a trending series a random forest trained on raw levels is guaranteed to
flatline. That is a real property of the model class, but comparing an undifferenced
tree to ARIMA would be a straw-man baseline rather than an honest one. We therefore fit
tree models on first differences and integrate the predictions back, which is standard
practice and makes them genuinely competitive. The linear model is fitted on levels,
where extrapolation is well defined.

**Recursive multi-step.** Predictions are fed back as inputs for the next step. The
alternative -- a separate model per horizon step -- multiplies fitting cost by the
horizon, which the counterfactual tensor cannot absorb. Recursive forecasting
accumulates error over the horizon; that is a documented property of the baseline, not
a bug, and it is identical across all ML models so the comparison stays fair.
"""

from __future__ import annotations

import numpy as np

from forecasting.base import FORECASTERS, Forecaster, residual_samples

__all__ = ["LinearARForecaster", "RandomForestForecaster", "XGBoostForecaster"]


def _lag_matrix(x: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised dataset of lagged features.

    Row ``i`` holds ``[x[i-1], x[i-2], ..., x[i-n_lags]]`` and predicts ``x[i]``.
    """
    n = x.size
    if n <= n_lags + 1:
        raise ValueError(f"need > {n_lags + 1} observations, got {n}")
    rows = n - n_lags
    X = np.empty((rows, n_lags), dtype=float)
    for lag in range(1, n_lags + 1):
        X[:, lag - 1] = x[n_lags - lag : n - lag]
    y = x[n_lags:]
    return X, y


def _seasonal_features(index: np.ndarray, period: int, n_harmonics: int = 2
                       ) -> np.ndarray:
    """Fourier terms for a seasonal period.

    Preferred over one-hot seasonal dummies: far fewer parameters for long periods
    (e.g. 168-hour weekly cycles) and smooth by construction.
    """
    if period <= 1:
        return np.empty((index.size, 0), dtype=float)
    feats = []
    for h in range(1, n_harmonics + 1):
        feats.append(np.sin(2 * np.pi * h * index / period))
        feats.append(np.cos(2 * np.pi * h * index / period))
    return np.column_stack(feats)


class _RecursiveMLForecaster(Forecaster):
    """Shared fit/predict loop for lag-feature regressors."""

    name = "ml_base"
    difference = True

    def _make_model(self, rng: np.random.Generator):  # pragma: no cover - abstract
        raise NotImplementedError

    def _n_lags(self, history_size: int, seasonal_period: int) -> int:
        # Enough lags to see a full season, bounded so short series stay fittable.
        target = max(2 * seasonal_period, 8) if seasonal_period > 1 else 12
        return int(max(2, min(target, history_size // 3)))

    def _forecast(self, history, horizon, seasonal_period, rng):
        m = max(1, int(seasonal_period))
        work = np.diff(history) if self.difference else history.astype(float)
        anchor = float(history[-1])

        n_lags = self._n_lags(work.size, m)
        X, y = _lag_matrix(work, n_lags)

        # Seasonal position of each training target, then of each forecast step.
        offset = history.size - work.size  # 1 when differenced, else 0
        train_index = np.arange(n_lags, work.size) + offset
        S = _seasonal_features(train_index, m)
        if S.size:
            X = np.column_stack([X, S])

        model = self._make_model(rng)
        model.fit(X, y)

        # --- recursive rollout ---
        buffer = list(work[-n_lags:])
        preds = np.empty(horizon, dtype=float)
        for step in range(horizon):
            lags = np.array(buffer[-n_lags:][::-1], dtype=float)[None, :]
            idx = np.array([history.size + step], dtype=float)
            Sf = _seasonal_features(idx, m)
            feat = np.column_stack([lags, Sf]) if Sf.size else lags
            value = float(model.predict(feat)[0])
            preds[step] = value
            buffer.append(value)

        point = anchor + np.cumsum(preds) if self.difference else preds

        # Residual bootstrap from in-sample fit. Not accumulated for undifferenced
        # models (their errors do not compound the same way).
        resid = y - model.predict(X)
        samples = residual_samples(point, resid, rng, accumulate=self.difference)
        return point, samples


@FORECASTERS.register("linear_ar")
class LinearARForecaster(_RecursiveMLForecaster):
    """Ridge-regularised autoregression on levels with Fourier seasonality."""

    name = "linear_ar"
    cost = 0.05
    difference = False

    def _make_model(self, rng):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(self.params.get("alpha", 1.0)))


@FORECASTERS.register("random_forest")
class RandomForestForecaster(_RecursiveMLForecaster):
    """Random forest on differenced lags."""

    name = "random_forest"
    cost = 0.8

    def _make_model(self, rng):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(self.params.get("n_estimators", 50)),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_jobs=1,  # parallelism lives at the experiment level, not inside a model
        )


@FORECASTERS.register("xgboost")
class XGBoostForecaster(_RecursiveMLForecaster):
    """Gradient-boosted trees on differenced lags."""

    name = "xgboost"
    cost = 0.6

    def _make_model(self, rng):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(self.params.get("n_estimators", 100)),
            max_depth=int(self.params.get("max_depth", 3)),
            learning_rate=float(self.params.get("learning_rate", 0.08)),
            subsample=float(self.params.get("subsample", 0.9)),
            colsample_bytree=float(self.params.get("colsample_bytree", 0.9)),
            reg_lambda=float(self.params.get("reg_lambda", 1.0)),
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
        )
