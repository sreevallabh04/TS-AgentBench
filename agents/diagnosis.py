"""Failure-mode diagnosis from reliability signals.

Existing agentic forecasters "reflect" generically: they notice something is wrong and
produce a revised forecast. Here, diagnosis is explicit and typed -- the agent commits
to *which* failure mode it believes it is facing, which in turn restricts the candidate
repairs to those matched to that mode.

Two things make this worth doing rather than simply trying every action:

* **It is falsifiable.** On synthetic data the true failure mode is known, so diagnosis
  accuracy is measurable rather than asserted.
* **It is cheap.** Restricting candidates to matched repairs is what lets the adaptive
  policy use a fraction of the compute of an exhaustive search (H4).

The rules below are deliberately transparent rather than learned. A learned diagnoser
would likely score better, but it would also make the ablations uninterpretable: when
"no change-point detector" changes the outcome, we want to know that it was the
change-point evidence that mattered, not that a black box re-balanced itself. The LLM
router in :mod:`agents.llm_agent` is evaluated as an alternative to these rules, which
is one of the paper's comparisons rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from datasets.base import FailureMode

__all__ = ["diagnose", "Diagnosis", "candidate_actions", "MODE_TO_ACTIONS"]


#: Repairs matched to each failure mode, in preference order. ``accept`` is always
#: appended by the policy as the null option.
MODE_TO_ACTIONS: dict[FailureMode, tuple[str, ...]] = {
    FailureMode.TREND_MISSPEC: ("damp_trend", "switch_model", "ensemble"),
    FailureMode.SEASONAL_MISSPEC: ("respecify_seasonality", "switch_model", "ensemble"),
    FailureMode.LEVEL_SHIFT: ("refit_post_changepoint", "switch_model", "abstain"),
    FailureMode.ANOMALY_CONTAMINATION: ("robustify", "switch_model", "ensemble"),
    FailureMode.VARIANCE_SHIFT: ("transform", "recalibrate_intervals", "ensemble"),
    FailureMode.SHORT_HISTORY: ("abstain", "ensemble", "switch_model"),
    FailureMode.DRIFT: ("refit_post_changepoint", "damp_trend", "ensemble"),
    FailureMode.MODEL_MISMATCH: ("switch_model", "ensemble", "transform"),
    FailureMode.NONE: ("ensemble",),
}


@dataclass
class Diagnosis:
    """A ranked diagnosis with the evidence that produced it."""

    mode: FailureMode
    score: float
    scores: dict[FailureMode, float]
    evidence: list[str]

    @property
    def ranked_modes(self) -> list[FailureMode]:
        return sorted(self.scores, key=lambda m: -self.scores[m])


def _get(signals: Mapping[str, float], key: str, default: float = 0.0) -> float:
    value = signals.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default  # NaN -> default


def diagnose(
    signals: Mapping[str, float],
    disabled: Sequence[str] = (),
) -> Diagnosis:
    """Score each failure mode from the signal vector.

    ``disabled`` lists signal-family prefixes switched off by an ablation (e.g.
    ``"changepoint."``); rules depending on a disabled family contribute nothing,
    which is how the ablations remove evidence rather than merely renaming it.
    """

    def enabled(prefix: str) -> bool:
        return not any(prefix.startswith(d) for d in disabled)

    scores: dict[FailureMode, float] = {m: 0.0 for m in FailureMode}
    evidence: list[str] = []

    # --- level shift / regime change ---
    if enabled("changepoint."):
        recency = _get(signals, "changepoint.relative_recency", 1.0)
        shift = _get(signals, "changepoint.last_segment_mean_shift", 0.0)
        n_cp = _get(signals, "changepoint.n_changepoints", 0.0)
        if n_cp > 0 and recency < 0.5:
            # A break close to the origin is the dangerous case: most of the fitting
            # window predates the current regime.
            score = (1.0 - recency) * min(2.0, 0.5 + shift)
            scores[FailureMode.LEVEL_SHIFT] += score
            evidence.append(
                f"change point {recency:.0%} of the way back with a "
                f"{shift:.2f}sd level shift"
            )
        if n_cp >= 3:
            scores[FailureMode.DRIFT] += 0.5
            evidence.append(f"{int(n_cp)} change points: unstable regime")

    # --- anomaly contamination ---
    if enabled("anomaly."):
        rate = _get(signals, "anomaly.anomaly_rate", 0.0)
        recent = _get(signals, "anomaly.recent_anomaly_rate", 0.0)
        at_last = _get(signals, "anomaly.anomaly_at_last_point", 0.0)
        if rate > 0.02 or recent > 0.05:
            scores[FailureMode.ANOMALY_CONTAMINATION] += min(3.0, 20.0 * rate + 5.0 * recent)
            evidence.append(f"anomaly rate {rate:.1%} (recent {recent:.1%})")
        if at_last > 0.5:
            # Every naive/drift-style forecast anchors on the final observation.
            scores[FailureMode.ANOMALY_CONTAMINATION] += 1.0
            evidence.append("final observation is an outlier")

    # --- seasonality misspecification ---
    if enabled("seasonality."):
        if _get(signals, "seasonality.period_mismatch", 0.0) > 0.5:
            scores[FailureMode.SEASONAL_MISSPEC] += 1.2
            evidence.append("detected period differs from the assumed one")
        if _get(signals, "seasonality.multiple_seasonality", 0.0) > 0.5:
            scores[FailureMode.SEASONAL_MISSPEC] += 0.8
            evidence.append("multiple seasonal cycles present")

    # --- trend misspecification ---
    trend_strength = _get(signals, "stl.trend_strength", 0.0) if enabled("stl.") else 0.0
    if trend_strength > 0.6:
        bias = _get(signals, "backtest.bias_ratio", 0.0) if enabled("backtest.") else 0.0
        scores[FailureMode.TREND_MISSPEC] += trend_strength * (0.5 + bias)
        evidence.append(f"trend strength {trend_strength:.2f}, backtest bias ratio {bias:.2f}")

    # --- variance shift ---
    if enabled("distribution_shift."):
        var_ratio = _get(signals, "distribution_shift.variance_ratio", 1.0)
        if var_ratio > 1.8 or var_ratio < 0.55:
            scores[FailureMode.VARIANCE_SHIFT] += min(2.0, abs(np.log(max(var_ratio, 1e-6))))
            evidence.append(f"variance ratio {var_ratio:.2f} between recent and past")
        if _get(signals, "distribution_shift.psi", 0.0) > 0.25:
            scores[FailureMode.DRIFT] += 0.8
            evidence.append(f"PSI {_get(signals, 'distribution_shift.psi', 0.0):.2f}")

    # --- non-stationarity / drift ---
    if enabled("stationarity."):
        if _get(signals, "stationarity.kpss_rejects_stationarity", 0.0) > 0.5 and \
           _get(signals, "stationarity.adf_rejects_unit_root", 0.0) < 0.5:
            scores[FailureMode.DRIFT] += 1.0
            evidence.append("ADF and KPSS agree the series is non-stationary")
    if enabled("backtest.") and _get(signals, "backtest.error_trend", 0.0) > 0.05:
        scores[FailureMode.DRIFT] += 0.7
        evidence.append("backtest error is growing towards the present")

    # --- insufficient history ---
    per_season = _get(signals, "context.history_per_season", 99.0)
    if per_season < 3.0:
        scores[FailureMode.SHORT_HISTORY] += (3.0 - per_season)
        evidence.append(f"only {per_season:.1f} seasonal cycles of history")

    # --- model-class mismatch ---
    if enabled("disagreement."):
        disagreement = _get(signals, "disagreement.normalized", 0.0)
        if disagreement > 0.3:
            scores[FailureMode.MODEL_MISMATCH] += min(2.0, disagreement)
            evidence.append(f"model pool disagreement {disagreement:.2f}")
    if enabled("selection.") and _get(signals, "selection.margin", 1.0) < 0.02:
        scores[FailureMode.MODEL_MISMATCH] += 0.5
        evidence.append("model selection was nearly a tie")

    scores.pop(FailureMode.NONE, None)
    if not scores or max(scores.values()) <= 0.0:
        return Diagnosis(FailureMode.NONE, 0.0, scores, ["no failure signature detected"])

    best = max(scores, key=lambda m: scores[m])
    return Diagnosis(best, float(scores[best]), scores, evidence)


def candidate_actions(diagnosis: Diagnosis, max_actions: int = 3,
                      top_modes: int = 2) -> list[str]:
    """Candidate repairs for a diagnosis, ordered by preference.

    Draws from the top few modes rather than only the single best, because the
    diagnosis is itself uncertain and committing entirely to a marginal winner throws
    away the second hypothesis for no benefit.
    """
    seen: list[str] = []
    for mode in diagnosis.ranked_modes[:top_modes]:
        if diagnosis.scores.get(mode, 0.0) <= 0.0:
            continue
        for action in MODE_TO_ACTIONS.get(mode, ()):
            if action not in seen:
                seen.append(action)
    if not seen:
        seen = list(MODE_TO_ACTIONS[FailureMode.NONE])
    return seen[:max_actions]

