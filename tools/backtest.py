"""Backtesting, residual analysis and model-disagreement tools.

These three are the most important reliability signals in the project, and they share
one property that makes the method defensible: **they are computed entirely inside the
history**, by holding out the most recent observations that the agent already has.
They are therefore available at decision time in deployment, not just in hindsight.

This is the direct answer to the Huang et al. result that intrinsic self-correction
fails without external feedback. A backtest is external feedback -- it is a
measurement, not the model's opinion of itself -- and it costs nothing but compute.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forecasting.base import FORECASTERS
from tools.base import TOOLS, Tool

__all__ = ["BacktestTool", "ErrorAnalysisTool", "ModelDisagreementTool", "rolling_backtest"]


def _mase_denominator(history: np.ndarray, seasonal_period: int) -> float:
    """Mean absolute seasonal difference -- the MASE scaling term."""
    m = max(1, int(seasonal_period))
    if history.size > m:
        denom = float(np.mean(np.abs(history[m:] - history[:-m])))
    else:
        denom = float(np.mean(np.abs(np.diff(history)))) if history.size > 1 else 0.0
    return denom if denom > 1e-12 else 1.0


def rolling_backtest(
    history: np.ndarray,
    model_name: str,
    horizon: int,
    seasonal_period: int = 1,
    n_folds: int = 3,
    stride: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Rolling-origin backtest confined to ``history``.

    Folds are carved out of the *end* of the history, so the most recent fold is the
    closest available proxy for the forecast the agent is about to make. Every fold
    refits from scratch on data strictly preceding its own origin.
    """
    history = np.asarray(history, dtype=float)
    n = history.size
    stride = int(stride or max(1, horizon))
    model = FORECASTERS.build(model_name)

    errors: list[float] = []
    scaled: list[float] = []
    biases: list[float] = []
    fold_details: list[dict[str, float]] = []

    for fold in range(n_folds):
        origin = n - horizon - fold * stride
        # Require enough remaining history to fit anything meaningful.
        if origin < max(8, 2 * seasonal_period):
            break
        train, actual = history[:origin], history[origin : origin + horizon]
        if actual.size < horizon:
            break

        result = model.forecast(
            train, horizon, seasonal_period, coverage_levels=(0.8,), seed=seed + fold
        )
        err = result.point - actual
        mae = float(np.mean(np.abs(err)))
        denom = _mase_denominator(train, seasonal_period)

        errors.append(mae)
        scaled.append(mae / denom)
        biases.append(float(np.mean(err)))
        fold_details.append(
            {"origin": int(origin), "mae": mae, "mase": mae / denom,
             "bias": float(np.mean(err))}
        )

    if not errors:
        return {
            "n_folds": 0,
            "backtest_mae": float("nan"),
            "backtest_mase": float("nan"),
            "backtest_mase_std": float("nan"),
            "backtest_bias": float("nan"),
            "error_trend": 0.0,
            "folds": [],
        }

    scaled_arr = np.asarray(scaled)
    # Positive error_trend means accuracy is degrading as origins approach the present
    # -- a strong hint that the model is drifting out of regime. folds[0] is the most
    # recent origin, so we regress against reversed order.
    error_trend = (
        float(np.polyfit(np.arange(scaled_arr.size), scaled_arr[::-1], 1)[0])
        if scaled_arr.size > 2
        else 0.0
    )
    return {
        "n_folds": len(errors),
        "backtest_mae": float(np.mean(errors)),
        "backtest_mase": float(np.mean(scaled_arr)),
        "backtest_mase_std": float(np.std(scaled_arr)),
        "backtest_mase_max": float(np.max(scaled_arr)),
        "backtest_bias": float(np.mean(biases)),
        # Systematic one-directional error, normalised: near 1 means every fold missed
        # the same way, which usually indicates misspecification rather than noise.
        "bias_ratio": float(abs(np.mean(biases)) / (np.mean(errors) + 1e-12)),
        "error_trend": error_trend,
        "folds": fold_details,
    }


@TOOLS.register("backtest")
class BacktestTool(Tool):
    """Rolling-origin backtest of one model, inside the history."""

    name = "backtest"
    cost = 5.0
    description = (
        "Rolling-origin backtest of a candidate model on held-out recent history; "
        "returns scaled error, its variability, bias and trend."
    )

    def _run(
        self,
        history,
        seasonal_period,
        model_name: str = "theta",
        horizon: int = 8,
        n_folds: int = 3,
        seed: int = 0,
        **kwargs,
    ):
        out = rolling_backtest(
            history, model_name, horizon, seasonal_period, n_folds, seed=seed
        )
        out["model_name"] = model_name
        return out

    def summarize(self, values):
        if not values.get("n_folds"):
            return "backtest unavailable (insufficient history)"
        return (
            f"{values['model_name']}: MASE={values['backtest_mase']:.3f} "
            f"(sd {values['backtest_mase_std']:.3f}) over "
            f"{values['n_folds']} folds, bias ratio={values['bias_ratio']:.2f}"
        )


@TOOLS.register("error_analysis")
class ErrorAnalysisTool(Tool):
    """Residual diagnostics for a fitted model.

    Structure left in the residuals is evidence of misspecification: a well-specified
    model leaves white noise. Ljung-Box tests for leftover autocorrelation, and a
    variance-ratio check between the first and second half of the residuals detects
    heteroskedasticity that would invalidate the prediction intervals.
    """

    name = "error_analysis"
    cost = 3.0
    description = (
        "Residual diagnostics: Ljung-Box autocorrelation test, bias, and "
        "heteroskedasticity checks indicating model misspecification."
    )

    def _run(
        self,
        history,
        seasonal_period,
        model_name: str = "theta",
        horizon: int = 8,
        seed: int = 0,
        **kwargs,
    ):
        from statsmodels.stats.diagnostic import acorr_ljungbox

        # One-step-ahead residuals from an expanding-window walk over the tail. This
        # is more informative than in-sample residuals, which are optimistically
        # small for models with many parameters.
        n = history.size
        start = max(max(8, 2 * seasonal_period), int(n * 0.7))
        model = FORECASTERS.build(model_name)
        resid: list[float] = []
        # Cap the walk so the diagnostic stays affordable inside the tensor.
        step = max(1, (n - start) // 20)
        for origin in range(start, n, step):
            r = model.forecast(
                history[:origin], 1, seasonal_period, coverage_levels=(0.8,), seed=seed
            )
            resid.append(float(history[origin] - r.point[0]))

        if len(resid) < 5:
            return {
                "n_residuals": len(resid),
                "ljung_box_pvalue": float("nan"),
                "residual_autocorrelated": False,
                "residual_bias": float(np.mean(resid)) if resid else 0.0,
                "heteroskedastic": False,
                "variance_ratio": 1.0,
            }

        r = np.asarray(resid, dtype=float)
        nlags = min(10, max(1, len(r) // 5))
        try:
            lb = acorr_ljungbox(r, lags=[nlags], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = float("nan")

        half = len(r) // 2
        v1 = float(np.var(r[:half])) + 1e-12
        v2 = float(np.var(r[half:])) + 1e-12
        var_ratio = v2 / v1

        return {
            "n_residuals": int(len(r)),
            "ljung_box_pvalue": lb_p,
            # Low p-value: structure remains -> the model is leaving signal behind.
            "residual_autocorrelated": bool(np.isfinite(lb_p) and lb_p < 0.05),
            "residual_bias": float(np.mean(r)),
            "residual_std": float(np.std(r)),
            "bias_ratio": float(abs(np.mean(r)) / (np.mean(np.abs(r)) + 1e-12)),
            "variance_ratio": float(var_ratio),
            "heteroskedastic": bool(var_ratio > 2.0 or var_ratio < 0.5),
        }

    def summarize(self, values):
        return (
            f"Ljung-Box p={values['ljung_box_pvalue']:.3g}, "
            f"autocorrelated={values['residual_autocorrelated']}, "
            f"heteroskedastic={values['heteroskedastic']}"
        )


@TOOLS.register("model_disagreement")
class ModelDisagreementTool(Tool):
    """Spread of forecasts across a pool of models.

    Disagreement is a genuinely different signal from any single model's own interval:
    a model can be confidently wrong, but an entire pool rarely agrees confidently on
    a value that is far from the truth. Reported scale-free (as a fraction of the
    series' own variability) so it is comparable across series.
    """

    name = "model_disagreement"
    cost = 4.0
    description = (
        "Dispersion of point forecasts across a model pool; high disagreement "
        "indicates the data do not determine a single answer."
    )

    def _run(
        self,
        history,
        seasonal_period,
        model_names: tuple[str, ...] = ("naive", "seasonal_naive", "theta", "drift"),
        horizon: int = 8,
        seed: int = 0,
        **kwargs,
    ):
        points = []
        used = []
        for name in model_names:
            try:
                r = FORECASTERS.build(name).forecast(
                    history, horizon, seasonal_period, coverage_levels=(0.8,), seed=seed
                )
                if not r.fallback:
                    points.append(r.point)
                    used.append(name)
            except Exception:
                continue

        if len(points) < 2:
            return {
                "n_models": len(points),
                "disagreement": 0.0,
                "normalized_disagreement": 0.0,
                "max_pairwise_gap": 0.0,
                "models_used": used,
            }

        P = np.vstack(points)
        spread = float(np.mean(np.std(P, axis=0)))
        scale = float(np.std(history)) or 1.0
        gaps = float(np.max(np.max(P, axis=0) - np.min(P, axis=0)))
        return {
            "n_models": len(points),
            "disagreement": spread,
            "normalized_disagreement": float(spread / scale),
            "max_pairwise_gap": float(gaps / scale),
            "models_used": used,
        }

    def summarize(self, values):
        return (
            f"{values['n_models']} models, normalised disagreement="
            f"{values['normalized_disagreement']:.3f}"
        )
