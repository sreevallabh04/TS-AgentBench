"""Component tests: config, generators, forecasters, tools, actions, policies.

Where a component encodes a scientific claim, the test checks the claim rather than
the plumbing. For example, the change-point detector is tested against families whose
break locations are known by construction, because a detector that returns plausible
output on unlabelled data tells us nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ConfigError, config_hash, load_config  # noqa: E402
from core.registry import Registry, RegistryError  # noqa: E402
from core.seed import derive_seed, seeded_rng  # noqa: E402
from datasets.synthetic import family_names, generate_family  # noqa: E402


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def test_registry_rejects_duplicate_names():
    reg: Registry[object] = Registry("thing")
    reg.register("a", object)
    with pytest.raises(RegistryError):
        reg.register("a", dict)


def test_registry_reports_available_names_on_miss():
    reg: Registry[object] = Registry("thing")
    reg.register("alpha", object)
    with pytest.raises(RegistryError, match="alpha"):
        reg.get("missing")


def test_derived_seeds_are_stable_and_distinct():
    assert derive_seed(0, "a", 1) == derive_seed(0, "a", 1)
    assert derive_seed(0, "a", 1) != derive_seed(0, "a", 2)
    assert derive_seed(0, "a", 1) != derive_seed(1, "a", 1)


def test_seeded_rng_is_reproducible_across_calls():
    a = seeded_rng(7, "x").normal(size=5)
    b = seeded_rng(7, "x").normal(size=5)
    assert np.allclose(a, b)


def test_config_inheritance_and_override():
    """Check the mechanism, not a tunable value.

    Asserting a specific `series_per_family` would make the test fail whenever the
    smoke config is legitimately resized, which is noise rather than signal.
    """
    base = load_config("configs/base.yaml")
    smoke = load_config("configs/smoke.yaml")

    # inherited: not restated in smoke.yaml
    assert smoke.eval.primary_metric == base.eval.primary_metric
    assert smoke.eval.large_error_quantile == base.eval.large_error_quantile
    # overridden: smoke deliberately shrinks the dataset and disables real data
    assert smoke.data.series_per_family < base.data.series_per_family
    assert smoke.data.include_real is False

    overridden = load_config("configs/smoke.yaml", ["seeds=[7,8]"])
    assert overridden.seeds == [7, 8]


def test_config_hash_is_stable_and_sensitive():
    a = config_hash(load_config("configs/smoke.yaml"))
    b = config_hash(load_config("configs/smoke.yaml"))
    c = config_hash(load_config("configs/smoke.yaml", ["seeds=[9]"]))
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    "override",
    [
        "unknown_key=1",                       # typo protection
        "forecast.horizons=[999]",             # horizon exceeds the test window
        "agent.target_correction_rate=1.5",    # out of range
        "eval.bootstrap_unit=galaxy",          # invalid enum
    ],
)
def test_config_rejects_invalid_values(override):
    with pytest.raises(ConfigError):
        load_config("configs/smoke.yaml", [override])


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", family_names())
def test_every_family_produces_usable_series(family):
    series = generate_family(family, 2, base_seed=0, n_timesteps=240, seasonal_period=12)
    assert len(series) == 2
    for s in series:
        finite = s.target[np.isfinite(s.target)]
        assert finite.size >= 30
        assert np.isfinite(finite).all()
        assert s.labels.is_ground_truth


def test_generation_is_deterministic_given_a_seed():
    a = generate_family("level_shift", 2, base_seed=42)[0].target
    b = generate_family("level_shift", 2, base_seed=42)[0].target
    c = generate_family("level_shift", 2, base_seed=43)[0].target
    assert np.allclose(a, b, equal_nan=True)
    assert not np.allclose(a, c, equal_nan=True)


def test_level_shift_records_its_changepoint():
    for s in generate_family("level_shift", 5, base_seed=0, n_timesteps=300):
        assert len(s.labels.changepoints) == 1
        cp = s.labels.changepoints[0]
        assert 0 < cp < len(s)


def test_variance_shift_actually_changes_variance():
    for s in generate_family("variance_shift", 5, base_seed=0, n_timesteps=300):
        cp = s.labels.changepoints[0]
        before = np.std(s.target[:cp])
        after = np.std(s.target[cp:])
        assert after > before, "variance_shift did not increase dispersion"


def test_multivariate_family_is_multivariate():
    s = generate_family("multivariate_var", 1, base_seed=0, n_timesteps=200)[0]
    assert s.is_multivariate and s.n_variates >= 2
    assert s.target.ndim == 1  # column 0 is the forecast target


# --------------------------------------------------------------------------- #
# Forecasters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    ["naive", "seasonal_naive", "drift", "mean", "theta", "ets", "arima",
     "linear_ar", "random_forest", "xgboost"],
)
def test_forecaster_contract(model):
    from forecasting.base import FORECASTERS

    s = generate_family("seasonal_single", 1, base_seed=0, n_timesteps=200)[0]
    result = FORECASTERS.build(model).forecast(
        s.target[:150], horizon=12, seasonal_period=12, coverage_levels=(0.8, 0.95),
        seed=0,
    )
    assert result.point.shape == (12,)
    assert np.isfinite(result.point).all()
    assert result.samples.shape[1] == 12
    for level in (0.8, 0.95):
        assert np.all(result.upper[level] >= result.lower[level] - 1e-9)
    # A 95% interval must contain the 80% interval.
    assert np.all(
        (result.upper[0.95] - result.lower[0.95])
        >= (result.upper[0.8] - result.lower[0.8]) - 1e-9
    )


def test_forecasters_degrade_instead_of_raising():
    """A pathological input must produce a fallback forecast, not an exception."""
    from forecasting.base import FORECASTERS

    constant = np.ones(40)
    for model in ("arima", "ets", "theta", "xgboost"):
        result = FORECASTERS.build(model).forecast(constant, 6, 12, seed=0)
        assert result.point.shape == (6,)
        assert np.isfinite(result.point).all()


def test_missing_values_are_imputed_not_dropped():
    """Dropping NaNs would shift the time index and corrupt seasonal phase."""
    from forecasting.base import FORECASTERS, impute_missing

    x = np.arange(60, dtype=float)
    x[10:14] = np.nan
    imputed, n = impute_missing(x)
    assert n == 4
    assert imputed.size == x.size
    assert np.isfinite(imputed).all()
    assert np.allclose(imputed[10:14], [10.0, 11.0, 12.0, 13.0])

    result = FORECASTERS.build("theta").forecast(x, 6, 12, seed=0)
    assert result.metadata["n_imputed"] == 4


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def test_changepoint_detector_finds_known_breaks():
    """Validated against families whose breaks are known by construction."""
    from tools.base import TOOLS

    tool = TOOLS.build("changepoint")
    detected, errors = 0, []
    for s in generate_family("level_shift", 10, base_seed=11, n_timesteps=300):
        cp = s.labels.changepoints[0]
        history = s.target[: min(len(s), cp + 40)]
        found = tool.run(history, s.seasonal_period).get("changepoints") or []
        if found:
            detected += 1
            errors.append(min(abs(np.array(found) - cp)))
    assert detected >= 7, "change-point detector missed most known level shifts"
    assert np.median(errors) <= 10, "change points are localised too imprecisely"


def test_changepoint_detector_is_quiet_on_stable_series():
    from tools.base import TOOLS

    tool = TOOLS.build("changepoint")
    for family in ("seasonal_single", "ar_stationary", "trend_linear"):
        counts = [
            tool.run(s.target[:200], s.seasonal_period).get("n_changepoints", 0)
            for s in generate_family(family, 4, base_seed=7, n_timesteps=240)
        ]
        assert np.mean(counts) <= 0.5, f"false change points on {family}"


def test_anomaly_detector_responds_to_contamination():
    from tools.base import TOOLS

    tool = TOOLS.build("anomaly")
    clean = [
        tool.run(s.target[:200], 12).get("anomaly_rate", 0.0)
        for s in generate_family("seasonal_single", 4, base_seed=0, n_timesteps=240)
    ]
    dirty = [
        tool.run(s.target[:200], 12).get("anomaly_rate", 0.0)
        for s in generate_family("anomaly_contaminated", 4, base_seed=0, n_timesteps=240)
    ]
    assert np.mean(dirty) > np.mean(clean)


def test_all_tools_return_results_without_raising():
    from tools.base import TOOLS

    s = generate_family("level_shift", 1, base_seed=0, n_timesteps=240)[0]
    for name in TOOLS.names():
        result = TOOLS.build(name).run(s.target[:200], s.seasonal_period)
        assert result.tool_name == name
        assert result.seconds >= 0.0
        if not result.failed:
            assert result.summary


def test_tools_handle_degenerate_input_gracefully():
    from tools.base import TOOLS

    for name in TOOLS.names():
        result = TOOLS.build(name).run(np.ones(4), 12)
        assert result.failed or isinstance(result.values, dict)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def action_context():
    from benchmark.actions import ActionContext
    from forecasting.base import FORECASTERS
    from tools.base import TOOLS

    from benchmark.tensor import select_base_model

    s = generate_family("level_shift", 1, base_seed=2, n_timesteps=240)[0]
    history, horizon = s.target[:200], 12
    models = ("naive", "seasonal_naive", "theta", "drift")
    pool = {
        m: FORECASTERS.build(m).forecast(history, horizon, s.seasonal_period, seed=0)
        for m in models
    }
    diagnostics = {
        n: TOOLS.build(n).run(history, s.seasonal_period).values
        for n in ("changepoint", "anomaly", "seasonality", "distribution_shift")
    }
    # model_scores must be populated: switch_model and ensemble rank models with it and
    # decline when it is empty, so omitting it would silently leave two of the ten
    # actions untested.
    _, scores = select_base_model(history, s.seasonal_period, models, horizon, seed=0)
    return ActionContext(
        history=history, seasonal_period=s.seasonal_period, horizon=horizon,
        base_model="theta", base_result=pool["theta"], seed=0,
        diagnostics=diagnostics, pool=pool, model_scores=scores,
    )


def test_actions_produce_valid_forecasts(action_context):
    from benchmark.actions import ACTION_ORDER, apply_all_actions

    results = apply_all_actions(action_context)
    assert "accept" in results
    for name, result in results.items():
        assert name in ACTION_ORDER
        assert result.point.shape == (action_context.horizon,)
        assert np.isfinite(result.point).all(), f"{name} produced non-finite values"


def test_accept_action_is_the_identity(action_context):
    from benchmark.actions import ACTIONS

    result = ACTIONS.build("accept").apply(action_context)
    assert np.allclose(result.point, action_context.base_result.point)


def test_recalibrate_leaves_the_point_forecast_untouched(action_context):
    """A distribution-only repair: it may change CRPS and coverage but never MASE."""
    from benchmark.actions import ACTIONS

    result = ACTIONS.build("recalibrate_intervals").apply(action_context)
    if result is not None:
        assert np.allclose(result.point, action_context.base_result.point)


def test_damp_trend_reduces_extrapolated_slope(action_context):
    from benchmark.actions import ACTIONS

    damped = ACTIONS.build("damp_trend").apply(action_context)
    base_slope = abs(np.polyfit(np.arange(12), action_context.base_result.point, 1)[0])
    damped_slope = abs(np.polyfit(np.arange(12), damped.point, 1)[0])
    assert damped_slope <= base_slope + 1e-9


def test_inapplicable_actions_return_none():
    """Unavailable is distinct from ineffective; the tensor relies on the difference."""
    from benchmark.actions import ACTIONS, ActionContext
    from forecasting.base import FORECASTERS

    history = np.linspace(10, 20, 80)
    base = FORECASTERS.build("theta").forecast(history, 6, 12, seed=0)
    ctx = ActionContext(
        history=history, seasonal_period=12, horizon=6, base_model="theta",
        base_result=base, diagnostics={"changepoint": {"changepoints": []}},
        pool={"theta": base},
    )
    assert ACTIONS.build("refit_post_changepoint").apply(ctx) is None


# --------------------------------------------------------------------------- #
# Diagnosis and policies
# --------------------------------------------------------------------------- #


def test_diagnosis_identifies_a_recent_level_shift():
    from agents.diagnosis import diagnose
    from datasets.base import FailureMode

    dx = diagnose(
        {
            "changepoint.n_changepoints": 1,
            "changepoint.relative_recency": 0.05,
            "changepoint.last_segment_mean_shift": 2.0,
        }
    )
    assert dx.mode == FailureMode.LEVEL_SHIFT
    assert dx.evidence


def test_ablation_removes_the_corresponding_evidence():
    from agents.diagnosis import diagnose
    from datasets.base import FailureMode

    signals = {
        "changepoint.n_changepoints": 1,
        "changepoint.relative_recency": 0.05,
        "changepoint.last_segment_mean_shift": 2.0,
    }
    ablated = diagnose(signals, disabled=["changepoint."])
    assert ablated.mode != FailureMode.LEVEL_SHIFT


def test_candidate_actions_are_matched_to_the_diagnosis():
    from agents.diagnosis import candidate_actions, diagnose

    dx = diagnose({"anomaly.anomaly_rate": 0.15, "anomaly.recent_anomaly_rate": 0.2})
    assert "robustify" in candidate_actions(dx)


def test_never_correct_policy_always_accepts():
    from agents.policies import NeverCorrectPolicy, ValidationEvidence

    signals = pd.DataFrame(
        {"instance_id": ["a", "b"], "series_id": ["s", "s"], "horizon": [1, 1]}
    )
    outcomes = pd.DataFrame(
        {
            "instance_id": ["a", "b"], "series_id": ["s", "s"], "horizon": [1, 1],
            "action": ["accept", "accept"], "split": ["val", "val"],
            "mase": [1.0, 1.0], "applicable": [True, True],
        }
    )
    decisions = NeverCorrectPolicy().decide(signals, ValidationEvidence(outcomes))
    assert (decisions["chosen_action"] == "accept").all()
    assert not decisions["corrected"].any()
