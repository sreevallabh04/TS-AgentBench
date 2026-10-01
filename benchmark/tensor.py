"""The counterfactual outcome tensor.

This is contribution C1. For every instance we execute *every* corrective action and
record what each one would have produced. The result is a table of counterfactual
outcomes that turns the correction decision into something measurable:

* the **oracle action** -- which repair was actually best,
* a **should_correct** label -- whether any repair beat accepting,
* the **regret** of any policy, computable offline without refitting anything.

The practical consequence is that every correction policy, ablation and robustness
sweep in the paper is a *replay* over this table rather than a fresh experiment. The
expensive stochastic work happens exactly once, which is what makes a CPU-only machine
sufficient and what makes the results exactly reproducible.

**One deliberate experimental control.** All arms share the same base-model selection
procedure (backtest-best over the pool, computed from history alone). Arms differ only
in their *correction policy*. Letting each arm also choose its own model would
confound the variable under study -- a difference in final accuracy could then come
from better model selection rather than from better correction decisions, and the
paper's question is specifically about the latter.

**Leakage.** Everything the agent may see is computed from ``history``. The target is
touched in exactly one place -- :func:`_score` -- which runs *after* all forecasts
exist and whose outputs feed evaluation only. Oracle labels derived here live behind
:mod:`benchmark.oracle`, which agent code must not import; ``tests/test_leakage.py``
enforces that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from benchmark.actions import ACTION_ORDER, ACTIONS, ActionContext, apply_all_actions
from core.logging import get_logger
from core.seed import derive_seed
from datasets.base import Instance, TimeSeries
from evaluation.metrics import mase_denominator, point_metrics, probabilistic_metrics
from forecasting.base import FORECASTERS, ForecastResult

log = get_logger("tensor")

__all__ = ["build_tensor", "compute_instance", "InstanceOutcome", "DEFAULT_TOOLS"]

#: Diagnostics computed for every instance. These become both the evidence shown to
#: the agent and the feature vector for the reliability estimator.
DEFAULT_TOOLS: tuple[str, ...] = (
    "ts_features",
    "stl",
    "acf",
    "seasonality",
    "stationarity",
    "changepoint",
    "anomaly",
    "distribution_shift",
)


@dataclass
class InstanceOutcome:
    """All results for one instance: outcomes per action, plus signals."""

    outcomes: list[dict[str, Any]]
    signals: dict[str, Any]


# --------------------------------------------------------------------------- #
# Base-model selection (history only)
# --------------------------------------------------------------------------- #


def select_base_model(
    history: np.ndarray,
    seasonal_period: int,
    models: Sequence[str],
    horizon: int,
    seed: int = 0,
    n_folds: int = 2,
    base_selection: str = "backtest",
) -> tuple[str, dict[str, float]]:
    """Pick the model with the lowest rolling-origin backtest error on history.

    This is the conventional automated-forecasting procedure and doubles as Baseline
    A's selector. Ties break on the declared model order, which keeps selection
    deterministic.
    """
    from tools.backtest import rolling_backtest

    scores: dict[str, float] = {}
    for model in models:
        bt = rolling_backtest(
            history, model, horizon, seasonal_period, n_folds=n_folds, seed=seed
        )
        score = bt.get("backtest_mase", float("nan"))
        scores[model] = float(score) if np.isfinite(score) else float("inf")

    if base_selection.startswith("fixed:"):
        # A conventional single-model pipeline. The pool is still scored, because the
        # model-switch and ensemble actions need those rankings -- only the *base*
        # is fixed.
        forced = base_selection.split(":", 1)[1]
        if forced in models:
            return forced, scores

    best = min(models, key=lambda m: (scores[m], list(models).index(m)))
    if not np.isfinite(scores[best]):
        best = "seasonal_naive" if "seasonal_naive" in models else list(models)[0]
    return best, scores


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


def _flatten_diagnostics(diags: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Flatten tool outputs into a numeric feature vector.

    List- and string-valued entries are dropped: they are useful as evidence in an
    LLM prompt but are not features a gradient-boosted model can consume.
    """
    feats: dict[str, float] = {}
    for tool, values in diags.items():
        if tool.startswith("_"):
            continue
        for key, value in values.items():
            if isinstance(value, bool):
                feats[f"{tool}.{key}"] = float(value)
            elif isinstance(value, (int, float)) and np.isfinite(value):
                feats[f"{tool}.{key}"] = float(value)
    return feats


def compute_signals(
    series: TimeSeries,
    instance: Instance,
    history: np.ndarray,
    base_model: str,
    model_scores: dict[str, float],
    pool: dict[str, ForecastResult],
    diagnostics: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Assemble the reliability-signal vector for one instance.

    Every entry is computable at decision time in deployment. That constraint is what
    separates a usable reliability estimator from a post-hoc explanation of errors.
    """
    from tools.backtest import rolling_backtest

    base_result = pool[base_model]
    sig: dict[str, Any] = _flatten_diagnostics(diagnostics)

    # --- backtest evidence for the selected model ---
    bt = rolling_backtest(
        history, base_model, instance.horizon, series.seasonal_period,
        n_folds=3, seed=seed,
    )
    for key in (
        "backtest_mase", "backtest_mase_std", "backtest_mase_max",
        "bias_ratio", "error_trend", "n_folds",
    ):
        value = bt.get(key, float("nan"))
        sig[f"backtest.{key}"] = float(value) if value is not None else float("nan")

    # --- uncertainty the model itself claims ---
    denom = mase_denominator(history, series.seasonal_period)
    width80 = base_result.interval_width(0.8)
    sig["uncertainty.interval_width_80"] = float(width80)
    # Scaled by the MASE denominator so it is comparable across series.
    sig["uncertainty.interval_width_scaled"] = float(width80 / max(denom, 1e-12))
    sig["uncertainty.sample_sd_mean"] = float(
        np.mean(np.std(base_result.samples, axis=0))
    )

    # --- disagreement across the pool ---
    points = np.vstack([r.point for r in pool.values()])
    scale = float(np.std(history)) or 1.0
    sig["disagreement.sd"] = float(np.mean(np.std(points, axis=0)))
    sig["disagreement.normalized"] = float(sig["disagreement.sd"] / scale)
    sig["disagreement.max_gap"] = float(
        np.max(np.max(points, axis=0) - np.min(points, axis=0)) / scale
    )
    finite_scores = [v for v in model_scores.values() if np.isfinite(v)]
    sig["selection.best_backtest_mase"] = float(min(finite_scores)) if finite_scores else float("nan")
    sig["selection.score_spread"] = (
        float(np.std(finite_scores)) if len(finite_scores) > 1 else 0.0
    )
    # A narrow margin means the selector was nearly indifferent -- itself a risk cue.
    sig["selection.margin"] = (
        float(np.sort(finite_scores)[1] - np.sort(finite_scores)[0])
        if len(finite_scores) > 1 else 0.0
    )

    # --- context ---
    sig["context.horizon"] = float(instance.horizon)
    sig["context.history_length"] = float(history.size)
    sig["context.seasonal_period"] = float(series.seasonal_period)
    sig["context.history_per_season"] = float(
        history.size / max(1, series.seasonal_period)
    )
    sig["context.base_model_fallback"] = float(base_result.fallback)

    # --- identity and ground truth (evaluation only, never a model input) ---
    sig.update(
        {
            "instance_id": instance.instance_id,
            "series_id": series.series_id,
            "dataset": series.dataset,
            "family": series.family or "real",
            "source": series.source,
            "split": instance.split,
            "origin": int(instance.origin),
            "horizon": int(instance.horizon),
            "base_model": base_model,
            "mase_denominator": float(denom),
            "label_is_ground_truth": bool(series.labels.is_ground_truth),
            "label_failure_modes": ",".join(
                m.value for m in series.labels.expected_failure_modes
            ),
            "label_n_changepoints": len(series.labels.changepoints),
            "label_n_anomalies": len(series.labels.anomaly_indices),
            # Distance from the forecast origin to the nearest true break. This is the
            # variable that decides whether *any* correction could have helped, and is
            # the key conditioning variable for H2.
            "label_steps_since_true_cp": (
                float(
                    min(
                        (instance.origin - cp for cp in series.labels.changepoints
                         if cp <= instance.origin),
                        default=float(instance.origin),
                    )
                )
                if series.labels.changepoints else float(instance.origin)
            ),
            "label_cp_inside_horizon": bool(
                any(
                    instance.origin < cp <= instance.origin + instance.horizon
                    for cp in series.labels.changepoints
                )
            ),
        }
    )
    return sig


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _score(
    result: ForecastResult,
    actual: np.ndarray,
    denom: float,
    coverage_levels: Sequence[float],
) -> dict[str, float]:
    """Score one forecast against the realised target.

    The only place in the tensor pipeline that reads the future.
    """
    scores = point_metrics(actual, result.point, denom)
    scores.update(
        probabilistic_metrics(actual, result.samples, result.lower, result.upper, denom)
    )
    return scores


# --------------------------------------------------------------------------- #
# Per-instance computation
# --------------------------------------------------------------------------- #


def compute_instance(
    series: TimeSeries,
    instance: Instance,
    models: Sequence[str],
    actions: Sequence[str] = ACTION_ORDER,
    tools: Sequence[str] = DEFAULT_TOOLS,
    coverage_levels: Sequence[float] = (0.8, 0.95),
    base_seed: int = 0,
    base_selection: str = "backtest",
) -> InstanceOutcome:
    """Compute pool forecasts, diagnostics, all action outcomes and signals."""
    from tools.base import TOOLS

    seed = derive_seed(base_seed, instance.instance_id)
    history = instance.history(series)
    actual = instance.target_slice(series)
    m = series.seasonal_period

    # --- 1. model pool (history only) ---
    pool: dict[str, ForecastResult] = {}
    for model in models:
        pool[model] = FORECASTERS.build(model).forecast(
            history, instance.horizon, m, coverage_levels=coverage_levels, seed=seed
        )

    # --- 2. base model selection (history only) ---
    base_model, model_scores = select_base_model(
        history, m, models, instance.horizon, seed=seed, base_selection=base_selection
    )
    base_result = pool[base_model]

    # --- 3. diagnostics (history only) ---
    diagnostics: dict[str, dict[str, Any]] = {}
    tool_seconds: dict[str, float] = {}
    for tool_name in tools:
        res = TOOLS.build(tool_name).run(history, m)
        diagnostics[tool_name] = {} if res.failed else res.values
        tool_seconds[tool_name] = res.seconds

    # --- 4. every corrective action ---
    ctx = ActionContext(
        history=history,
        seasonal_period=m,
        horizon=instance.horizon,
        base_model=base_model,
        base_result=base_result,
        seed=seed,
        diagnostics=diagnostics,
        pool=pool,
        model_scores=model_scores,
        coverage_levels=coverage_levels,
    )
    t0 = time.perf_counter()
    action_results = apply_all_actions(ctx, actions)
    action_seconds = time.perf_counter() - t0

    # --- 5. score everything against the realised target ---
    denom = mase_denominator(history, m)
    rows: list[dict[str, Any]] = []
    for action_name in actions:
        result = action_results.get(action_name)
        row: dict[str, Any] = {
            "instance_id": instance.instance_id,
            "series_id": series.series_id,
            "dataset": series.dataset,
            "family": series.family or "real",
            "source": series.source,
            "split": instance.split,
            "horizon": int(instance.horizon),
            "origin": int(instance.origin),
            "base_model": base_model,
            "action": action_name,
            "action_code": ACTIONS.build(action_name).code,
            "applicable": result is not None,
            "action_cost": float(ACTIONS.build(action_name).cost),
        }
        if result is None:
            # Unavailable repair: distinct from "made no difference".
            row.update({k: float("nan") for k in
                        ("mae", "rmse", "mase", "smape", "wape", "crps", "crps_scaled")})
        else:
            row.update(_score(result, actual, denom, coverage_levels))
            row["result_model"] = result.model_name
            row["fallback"] = bool(result.fallback)
        rows.append(row)

    # --- 6. signals ---
    signals = compute_signals(
        series, instance, history, base_model, model_scores, pool, diagnostics, seed
    )
    signals["cost.action_seconds"] = float(action_seconds)
    signals["cost.tool_seconds_total"] = float(sum(tool_seconds.values()))
    for tool_name, secs in tool_seconds.items():
        signals[f"cost.tool_seconds.{tool_name}"] = float(secs)

    return InstanceOutcome(outcomes=rows, signals=signals)


# --------------------------------------------------------------------------- #
# Tensor construction
# --------------------------------------------------------------------------- #


def build_tensor(
    series_list: Sequence[TimeSeries],
    instances_by_series: dict[str, list[Instance]],
    models: Sequence[str],
    actions: Sequence[str] = ACTION_ORDER,
    tools: Sequence[str] = DEFAULT_TOOLS,
    coverage_levels: Sequence[float] = (0.8, 0.95),
    base_seed: int = 0,
    n_jobs: int = -1,
    progress: bool = True,
    base_selection: str = "backtest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the counterfactual outcome tensor and the signal table.

    Returns ``(outcomes, signals)``: one row per (instance, action) and one row per
    instance respectively.
    """
    import os

    from joblib import Parallel, delayed

    # Pin BLAS to one thread per worker. Each joblib worker is already a separate
    # process, so multi-threaded BLAS inside them oversubscribes the CPU (8 workers x
    # 8 BLAS threads on 8 cores) and slows everything down.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    tasks: list[tuple[TimeSeries, Instance]] = [
        (s, inst)
        for s in series_list
        for inst in instances_by_series.get(s.series_id, [])
    ]
    if not tasks:
        raise ValueError("no instances to compute")

    log.info(
        "building tensor: %d instances x %d actions (%d models, %d tools)",
        len(tasks), len(actions), len(models), len(tools),
    )

    def _one(series: TimeSeries, instance: Instance) -> InstanceOutcome | None:
        try:
            return compute_instance(
                series, instance, models, actions, tools, coverage_levels, base_seed,
                base_selection,
            )
        except Exception as exc:  # noqa: BLE001 - one bad instance must not kill a run
            log.warning("instance %s failed: %s: %s",
                        instance.instance_id, type(exc).__name__, exc)
            return None

    results = Parallel(
        n_jobs=n_jobs, backend="loky", verbose=5 if progress else 0
    )(delayed(_one)(s, i) for s, i in tasks)

    outcomes = [row for r in results if r is not None for row in r.outcomes]
    signals = [r.signals for r in results if r is not None]
    n_failed = sum(1 for r in results if r is None)
    if n_failed:
        log.warning("%d/%d instances failed and were dropped", n_failed, len(tasks))

    return pd.DataFrame(outcomes), pd.DataFrame(signals)
