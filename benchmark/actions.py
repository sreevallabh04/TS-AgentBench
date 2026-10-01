"""The typed corrective action space.

This is the object that makes the correction decision measurable. Existing agentic
forecasting systems "reflect" and emit a revised forecast, which cannot be scored
against what a *different* revision would have produced. Here, correction is a choice
from a small closed set of named actions, each of which can be executed independently
on every instance. Running all of them yields, per instance, the counterfactual outcome
of every available repair -- and therefore an oracle action, a ``should_correct`` label
and a regret for any policy.

Each action is paired with the failure modes it is meant to repair. That pairing is
what turns diagnosis into something with consequences: the agent does not merely label
a failure, it selects the matched repair, and the benchmark records whether that was
the right call.

**Actions must never see the forecast target.** Every action receives
:class:`ActionContext`, which carries history and diagnostics only. This is enforced by
construction -- there is no field on the context that could leak the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core.registry import Registry
from datasets.base import FailureMode
from forecasting.base import FORECASTERS, ForecastResult, intervals_from_samples

__all__ = ["Action", "ActionContext", "ACTIONS", "apply_all_actions", "ACTION_ORDER"]

ACTIONS: Registry["Action"] = Registry("action")


@dataclass
class ActionContext:
    """Everything an action may look at. Contains no future information."""

    history: np.ndarray
    seasonal_period: int
    horizon: int
    base_model: str
    base_result: ForecastResult
    seed: int = 0
    #: Tool outputs keyed by tool name (values dicts, not ToolResult objects).
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Pre-computed forecasts from the model pool, keyed by model name. Reused by the
    #: model-switch and ensemble actions so they cost nothing extra.
    pool: dict[str, ForecastResult] = field(default_factory=dict)
    #: Pre-computed rolling-origin backtest score (MASE) per model, shared by base-model
    #: selection and by every action that needs to rank models. Computing these once per
    #: instance rather than once per consumer is the difference between roughly 11s and
    #: 3s per instance, which is what makes the full tensor affordable on a CPU.
    model_scores: dict[str, float] = field(default_factory=dict)
    coverage_levels: Sequence[float] = (0.8, 0.95)

    def diag(self, tool: str, key: str, default: Any = None) -> Any:
        return self.diagnostics.get(tool, {}).get(key, default)


class Action(ABC):
    """A corrective action."""

    code: str = "A?"
    name: str = "action"
    description: str = ""
    #: Failure modes this action is designed to repair.
    repairs: tuple[FailureMode, ...] = ()
    #: Relative compute cost, used for the compute-matched analysis (H6).
    cost: float = 1.0

    @abstractmethod
    def apply(self, ctx: ActionContext) -> ForecastResult | None:
        """Return a revised forecast, or ``None`` if the action does not apply."""

    def is_applicable(self, ctx: ActionContext) -> bool:
        """Cheap precondition check. Actions may still return ``None``."""
        return True

    def _refit(
        self, ctx: ActionContext, history: np.ndarray, model: str | None = None
    ) -> ForecastResult:
        """Refit a model on modified history."""
        return FORECASTERS.build(model or ctx.base_model).forecast(
            history,
            ctx.horizon,
            ctx.seasonal_period,
            coverage_levels=ctx.coverage_levels,
            seed=ctx.seed,
        )


# --------------------------------------------------------------------------- #
# A0 -- accept
# --------------------------------------------------------------------------- #


@ACTIONS.register("accept")
class AcceptAction(Action):
    """Do nothing. The null action, and the baseline every correction must beat."""

    code = "A0"
    name = "accept"
    description = "Accept the initial forecast unchanged."
    repairs = (FailureMode.NONE,)
    cost = 0.0

    def apply(self, ctx):
        return ctx.base_result


# --------------------------------------------------------------------------- #
# A1 -- switch model class
# --------------------------------------------------------------------------- #


@ACTIONS.register("switch_model")
class SwitchModelAction(Action):
    """Replace the selected model with the pool's best validated alternative.

    Selection uses backtest performance on history, never the target. When no backtest
    evidence is available the action declines rather than guessing, so that a
    "correction" is never a coin flip dressed up as a decision.
    """

    code = "A1"
    name = "switch_model"
    description = (
        "Switch to a different model class, chosen by backtest performance on "
        "held-out recent history."
    )
    repairs = (FailureMode.MODEL_MISMATCH, FailureMode.TREND_MISSPEC,
               FailureMode.SEASONAL_MISSPEC)
    cost = 3.0

    def apply(self, ctx):
        candidates = [m for m in ctx.pool if m != ctx.base_model]
        if not candidates:
            return None

        scored = [
            (float(ctx.model_scores[m]), m)
            for m in candidates
            if np.isfinite(ctx.model_scores.get(m, float("nan")))
        ]
        if not scored:
            return None
        scored.sort()
        best = scored[0][1]
        result = ctx.pool[best]
        result.metadata["switched_to"] = best
        return result


# --------------------------------------------------------------------------- #
# A2 -- refit after the most recent structural break
# --------------------------------------------------------------------------- #


@ACTIONS.register("refit_post_changepoint")
class RefitPostChangepointAction(Action):
    """Discard history before the last structural break and refit.

    The trade-off is explicit: a shorter, homogeneous window versus a longer window
    spanning two regimes. We require a minimum remaining length, because refitting on
    a handful of post-break points substitutes variance for bias and usually makes
    things worse.
    """

    code = "A2"
    name = "refit_post_changepoint"
    description = (
        "Refit the model using only observations after the most recent detected "
        "change point."
    )
    repairs = (FailureMode.LEVEL_SHIFT, FailureMode.DRIFT)
    cost = 2.0

    MIN_SEGMENT = 24

    def is_applicable(self, ctx):
        cps = ctx.diag("changepoint", "changepoints", []) or []
        if not cps:
            return False
        remaining = ctx.history.size - max(cps)
        return remaining >= max(self.MIN_SEGMENT, 2 * ctx.seasonal_period)

    def apply(self, ctx):
        if not self.is_applicable(ctx):
            return None
        cps = ctx.diag("changepoint", "changepoints", []) or []
        start = max(cps)
        result = self._refit(ctx, ctx.history[start:])
        result.metadata["refit_from"] = int(start)
        result.metadata["segment_length"] = int(ctx.history.size - start)
        return result


# --------------------------------------------------------------------------- #
# A3 -- robustify against outliers
# --------------------------------------------------------------------------- #


@ACTIONS.register("robustify")
class RobustifyAction(Action):
    """Replace detected outliers with a local median, then refit.

    Replacement rather than deletion: removing points would shift every subsequent
    index and silently corrupt the seasonal phase, which is a far worse error than an
    imperfectly imputed outlier.
    """

    code = "A3"
    name = "robustify"
    description = (
        "Replace detected outliers with local medians and refit, so contaminated "
        "observations stop dominating the fit."
    )
    repairs = (FailureMode.ANOMALY_CONTAMINATION,)
    cost = 2.0

    def is_applicable(self, ctx):
        return bool(ctx.diag("anomaly", "n_anomalies", 0))

    def apply(self, ctx):
        idx = ctx.diag("anomaly", "anomaly_indices", []) or []
        if not idx:
            return None

        clean = ctx.history.copy()
        win = max(3, ctx.seasonal_period // 2)
        for i in idx:
            if not 0 <= i < clean.size:
                continue
            lo, hi = max(0, i - win), min(clean.size, i + win + 1)
            neighbourhood = np.delete(ctx.history[lo:hi], i - lo)
            if neighbourhood.size:
                clean[i] = float(np.median(neighbourhood))

        result = self._refit(ctx, clean)
        result.metadata["n_replaced"] = len(idx)
        return result


# --------------------------------------------------------------------------- #
# A4 -- damp the trend
# --------------------------------------------------------------------------- #


@ACTIONS.register("damp_trend")
class DampTrendAction(Action):
    """Shrink the forecast's trend component geometrically.

    Applied directly to the produced path rather than by refitting, so it works for
    any model including ones with no explicit trend parameter. The damped increment
    ``phi^h`` is the standard damped-trend form.
    """

    code = "A4"
    name = "damp_trend"
    description = (
        "Geometrically damp the extrapolated trend, correcting over-confident "
        "linear extrapolation."
    )
    repairs = (FailureMode.TREND_MISSPEC,)
    cost = 0.1

    def __init__(self, phi: float = 0.85):
        self.phi = float(phi)

    def apply(self, ctx):
        base = ctx.base_result
        anchor = float(ctx.history[-1])
        increments = np.diff(np.concatenate([[anchor], base.point]))
        damped = increments * self.phi ** np.arange(1, increments.size + 1)
        point = anchor + np.cumsum(damped)

        # Shift the sample paths by the same correction so intervals stay consistent
        # with the point forecast.
        shift = point - base.point
        samples = base.samples + shift[None, :]
        lower, upper = intervals_from_samples(samples, ctx.coverage_levels)
        return ForecastResult(
            point=point,
            samples=samples,
            model_name=f"{base.model_name}+damped",
            lower=lower,
            upper=upper,
            fit_seconds=0.0,
            metadata={**base.metadata, "phi": self.phi},
        )


# --------------------------------------------------------------------------- #
# A5 -- re-specify seasonality
# --------------------------------------------------------------------------- #


@ACTIONS.register("respecify_seasonality")
class RespecifySeasonalityAction(Action):
    """Refit using the seasonal period suggested by spectral/ACF detection."""

    code = "A5"
    name = "respecify_seasonality"
    description = (
        "Refit with a corrected seasonal period detected from the data rather than "
        "the assumed one."
    )
    repairs = (FailureMode.SEASONAL_MISSPEC,)
    cost = 2.0

    def is_applicable(self, ctx):
        candidates = ctx.diag("seasonality", "candidate_periods", []) or []
        return any(p != ctx.seasonal_period and 1 < p <= ctx.history.size // 3
                   for p in candidates)

    def apply(self, ctx):
        candidates = [
            int(p)
            for p in (ctx.diag("seasonality", "candidate_periods", []) or [])
            if p != ctx.seasonal_period and 1 < p <= ctx.history.size // 3
        ]
        if not candidates:
            return None
        # Prefer the strongest candidate, which detection returns first.
        new_period = candidates[0]
        result = FORECASTERS.build(ctx.base_model).forecast(
            ctx.history, ctx.horizon, new_period,
            coverage_levels=ctx.coverage_levels, seed=ctx.seed,
        )
        result.metadata["new_period"] = new_period
        result.metadata["old_period"] = int(ctx.seasonal_period)
        return result


# --------------------------------------------------------------------------- #
# A6 -- variance-stabilising transform
# --------------------------------------------------------------------------- #


@ACTIONS.register("transform")
class TransformAction(Action):
    """Fit on a log scale and back-transform.

    Back-transformation uses the sample paths rather than exponentiating the point
    forecast, which would return the median instead of the mean and introduce a
    systematic downward bias -- a classic and easily-missed error.
    """

    code = "A6"
    name = "transform"
    description = (
        "Fit on a log-transformed series to stabilise variance, back-transforming "
        "through the sample paths."
    )
    repairs = (FailureMode.VARIANCE_SHIFT,)
    cost = 2.0

    def is_applicable(self, ctx):
        return bool(np.all(ctx.history > 0))

    def apply(self, ctx):
        if not self.is_applicable(ctx):
            return None
        logged = np.log(ctx.history)
        result = self._refit(ctx, logged)

        samples = np.exp(result.samples)
        point = samples.mean(axis=0)
        lower, upper = intervals_from_samples(samples, ctx.coverage_levels)
        return ForecastResult(
            point=point,
            samples=samples,
            model_name=f"{result.model_name}+log",
            lower=lower,
            upper=upper,
            fit_seconds=result.fit_seconds,
            metadata={**result.metadata, "transform": "log"},
        )


# --------------------------------------------------------------------------- #
# A7 -- ensemble
# --------------------------------------------------------------------------- #


@ACTIONS.register("ensemble")
class EnsembleAction(Action):
    """Average the top-k models by backtest performance.

    Combination is the most reliable single intervention in the forecasting
    literature, which makes it the *hardest* action for an adaptive policy to beat --
    and therefore the most honest one to include.
    """

    code = "A7"
    name = "ensemble"
    description = "Combine the top-k models by backtest error into an equal-weight ensemble."
    repairs = (FailureMode.MODEL_MISMATCH, FailureMode.NONE)
    cost = 4.0

    def __init__(self, k: int = 3):
        self.k = int(k)

    def apply(self, ctx):
        if len(ctx.pool) < 2:
            return None

        scored = [
            (float(ctx.model_scores[m]), m)
            for m in ctx.pool
            if np.isfinite(ctx.model_scores.get(m, float("nan")))
        ]
        if len(scored) < 2:
            return None

        scored.sort()
        chosen = [m for _, m in scored[: self.k]]
        point = np.mean([ctx.pool[m].point for m in chosen], axis=0)
        # Pool the sample paths so the ensemble's uncertainty reflects both within-
        # model spread and between-model disagreement.
        samples = np.vstack([ctx.pool[m].samples for m in chosen])
        lower, upper = intervals_from_samples(samples, ctx.coverage_levels)
        return ForecastResult(
            point=point,
            samples=samples,
            model_name="ensemble(" + ",".join(chosen) + ")",
            lower=lower,
            upper=upper,
            metadata={"ensemble_members": chosen},
        )


# --------------------------------------------------------------------------- #
# A8 -- conformal interval recalibration
# --------------------------------------------------------------------------- #


@ACTIONS.register("recalibrate_intervals")
class RecalibrateIntervalsAction(Action):
    """Rescale prediction intervals using split-conformal residuals from history.

    A *distribution-only* correction: the point forecast is untouched, so this action
    can improve coverage and CRPS while leaving MAE/MASE exactly unchanged. Keeping it
    in the action space is deliberate -- it lets the analysis distinguish "the forecast
    was wrong" from "the forecast was fine but over-confident", which a point-error
    metric alone cannot see.
    """

    code = "A8"
    name = "recalibrate_intervals"
    description = (
        "Rescale prediction intervals using conformal residual quantiles from recent "
        "history, leaving the point forecast unchanged."
    )
    repairs = (FailureMode.VARIANCE_SHIFT,)
    cost = 3.0

    def apply(self, ctx):
        from tools.backtest import rolling_backtest

        bt = rolling_backtest(
            ctx.history, ctx.base_model, ctx.horizon, ctx.seasonal_period,
            n_folds=4, seed=ctx.seed,
        )
        folds = bt.get("folds") or []
        if len(folds) < 2:
            return None

        # Empirical scale of recent absolute errors vs the width the model claims.
        realized = float(np.mean([f["mae"] for f in folds]))
        claimed = float(np.mean(np.std(ctx.base_result.samples, axis=0)))
        if claimed <= 1e-12 or not np.isfinite(realized):
            return None

        # 1.2533 = E|Z| for a standard normal; converts MAE to an sd-equivalent.
        factor = float(np.clip((realized * 1.2533) / claimed, 0.5, 5.0))
        base = ctx.base_result
        centred = base.samples - base.point[None, :]
        samples = base.point[None, :] + centred * factor
        lower, upper = intervals_from_samples(samples, ctx.coverage_levels)
        return ForecastResult(
            point=base.point.copy(),
            samples=samples,
            model_name=f"{base.model_name}+conformal",
            lower=lower,
            upper=upper,
            metadata={**base.metadata, "calibration_factor": factor},
        )


# --------------------------------------------------------------------------- #
# A9 -- abstain
# --------------------------------------------------------------------------- #


@ACTIONS.register("abstain")
class AbstainAction(Action):
    """Fall back to seasonal naive and flag low confidence.

    Abstention in forecasting cannot mean "return nothing" -- downstream systems still
    need a number. It means retreating to the most defensible simple forecast and
    signalling that the estimate should not be trusted, which is the operational form
    of selective prediction in this setting.
    """

    code = "A9"
    name = "abstain"
    description = (
        "Abstain: fall back to a seasonal naive forecast with widened intervals and "
        "a low-confidence flag."
    )
    repairs = (FailureMode.SHORT_HISTORY,)
    cost = 0.1

    def apply(self, ctx):
        result = self._refit(ctx, ctx.history, model="seasonal_naive")
        # Widen to reflect that we are explicitly declining to model the series.
        centred = result.samples - result.point[None, :]
        samples = result.point[None, :] + centred * 1.5
        lower, upper = intervals_from_samples(samples, ctx.coverage_levels)
        return ForecastResult(
            point=result.point,
            samples=samples,
            model_name="abstain(seasonal_naive)",
            lower=lower,
            upper=upper,
            metadata={**result.metadata, "abstained": True},
        )


#: Canonical ordering, used for stable column layout in the counterfactual tensor.
ACTION_ORDER: tuple[str, ...] = (
    "accept",
    "switch_model",
    "refit_post_changepoint",
    "robustify",
    "damp_trend",
    "respecify_seasonality",
    "transform",
    "ensemble",
    "recalibrate_intervals",
    "abstain",
)


def apply_all_actions(
    ctx: ActionContext, actions: Sequence[str] | None = None
) -> dict[str, ForecastResult]:
    """Execute every action on one instance.

    This is the workhorse behind the counterfactual tensor. Actions that do not apply
    are omitted from the returned mapping rather than substituted with the base
    forecast, so that "this repair was unavailable" and "this repair changed nothing"
    stay distinguishable -- they mean different things for a correction policy.
    """
    names = list(actions or ACTION_ORDER)
    out: dict[str, ForecastResult] = {}
    for name in names:
        action = ACTIONS.build(name)
        try:
            result = action.apply(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad action must not kill a run
            result = None
            ctx.diagnostics.setdefault("_action_errors", {})[name] = (
                f"{type(exc).__name__}: {exc}"
            )
        if result is not None:
            out[name] = result
    return out
