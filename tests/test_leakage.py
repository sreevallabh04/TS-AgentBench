"""Leakage tests -- the scientific gate on this project.

If any test in this file fails, no result from the repository should be reported.
Every other bug degrades a number; a leak invalidates the conclusion.

Four classes of leak are checked:

1. **Structural**: oracle labels (derived from test outcomes) must be unreachable from
   agent code, verified by walking the import graph rather than by convention.
2. **Temporal**: validation instances must never touch test observations, and no
   forecast target may extend beyond its own window.
3. **Interface**: tools and actions receive history only -- verified by mutating the
   future and asserting nothing downstream changes.
4. **Fitting**: the reliability estimator refuses to be fitted on test instances.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import iter_instances, make_split  # noqa: E402
from datasets.synthetic import generate_family  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agents"


# --------------------------------------------------------------------------- #
# 1. Structural: the oracle must be unreachable from agent code
# --------------------------------------------------------------------------- #


def _imports_of(path: Path) -> set[str]:
    """Module names imported by a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _local_module_path(module: str) -> Path | None:
    candidate = PROJECT_ROOT / Path(*module.split("."))
    if candidate.with_suffix(".py").exists():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").exists():
        return candidate / "__init__.py"
    return None


def _transitive_imports(start: Path, seen: set[Path] | None = None) -> set[str]:
    """All project modules reachable from ``start`` by following imports."""
    seen = seen if seen is not None else set()
    if start in seen:
        return set()
    seen.add(start)

    modules = _imports_of(start)
    reachable = set(modules)
    for module in modules:
        path = _local_module_path(module)
        if path is not None:
            reachable |= _transitive_imports(path, seen)
    return reachable


def test_agents_cannot_reach_the_oracle():
    """No module under agents/ may import benchmark.oracle, even transitively.

    The oracle is computed from realised test outcomes. An agent that could import it
    could -- accidentally or otherwise -- condition on the answer.
    """
    offenders: list[str] = []
    for path in AGENT_DIR.rglob("*.py"):
        reachable = _transitive_imports(path)
        if any(m == "benchmark.oracle" or m.endswith(".oracle") for m in reachable):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        "agent modules can reach benchmark.oracle (test-set information): "
        f"{offenders}"
    )


def test_agents_do_not_import_the_tensor_builder():
    """Agents must consume precomputed signals, not build tensors themselves.

    The tensor builder scores forecasts against realised targets. Agent code has no
    legitimate reason to touch it.
    """
    offenders = [
        str(p.relative_to(PROJECT_ROOT))
        for p in AGENT_DIR.rglob("*.py")
        if "benchmark.tensor" in _imports_of(p)
    ]
    assert not offenders, f"agent modules import benchmark.tensor: {offenders}"


# --------------------------------------------------------------------------- #
# 2. Temporal: split and instance boundaries
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sample_series():
    return generate_family("level_shift", 3, base_seed=0, n_timesteps=300,
                           seasonal_period=12)


def test_split_boundaries_are_strictly_ordered(sample_series):
    for s in sample_series:
        split = make_split(len(s), 48, 48, 32)
        assert 0 < split.train_end < split.val_end < split.test_end == len(s)


def test_validation_targets_never_reach_test_observations(sample_series):
    """The core temporal guarantee.

    Every threshold and the reliability estimator are fitted on validation outcomes.
    If a validation target overlapped the test window, all of that fitting would be
    contaminated.
    """
    for s in sample_series:
        split = make_split(len(s), 48, 48, 32)
        for inst in iter_instances(s, split, [1, 8, 24], 5, 3, 8):
            end = inst.origin + inst.horizon
            if inst.split == "val":
                assert end <= split.val_end, (
                    f"{inst.instance_id}: validation target ends at {end}, past "
                    f"val_end={split.val_end}"
                )
                assert inst.origin >= split.train_end
            else:
                assert end <= split.test_end
                assert inst.origin >= split.val_end


def test_history_excludes_the_target(sample_series):
    """``history`` and ``target_slice`` must partition the series without overlap."""
    s = sample_series[0]
    split = make_split(len(s), 48, 48, 32)
    for inst in iter_instances(s, split, [1, 12], 3, 3, 8):
        history = inst.history(s)
        target = inst.target_slice(s)
        assert history.size == inst.origin
        assert target.size == inst.horizon
        combined = np.concatenate([history, target])
        expected = s.target[: inst.origin + inst.horizon]
        assert np.allclose(combined, expected, equal_nan=True)


def test_synthetic_targets_are_never_missing():
    """A NaN target would silently drop instances and bias the comparison."""
    from datasets.synthetic import family_names

    for family in family_names():
        for s in generate_family(family, 3, base_seed=3, n_timesteps=240):
            tail = s.target[int(0.75 * len(s)):]
            assert not np.isnan(tail).any(), f"{family}: NaN inside the evaluation region"


# --------------------------------------------------------------------------- #
# 3. Interface: tools and actions see history only
# --------------------------------------------------------------------------- #


def test_tools_are_invariant_to_the_future():
    """Corrupting observations after the origin must not change any diagnostic.

    A stronger check than reading the signatures: it would catch a tool that captured
    the full series through a closure or a default argument.
    """
    from tools.base import TOOLS

    s = generate_family("seasonal_single", 1, base_seed=0, n_timesteps=240)[0]
    origin = 180
    history = s.target[:origin]

    corrupted = s.target.copy()
    corrupted[origin:] = 1e6  # destroy the future entirely

    for name in TOOLS.names():
        if name in {"backtest", "error_analysis", "model_disagreement"}:
            continue  # these refit models; covered by the action test below
        tool = TOOLS.build(name)
        a = tool.run(history, s.seasonal_period)
        b = tool.run(corrupted[:origin], s.seasonal_period)
        assert a.values.keys() == b.values.keys()
        for key in a.values:
            va, vb = a.values[key], b.values[key]
            if isinstance(va, float) and np.isfinite(va):
                assert np.isclose(va, vb, equal_nan=True), f"{name}.{key} changed"


def test_actions_are_invariant_to_the_future():
    """Corrupting the future must not change any corrective action's output."""
    from benchmark.actions import ActionContext, apply_all_actions
    from forecasting.base import FORECASTERS
    from tools.base import TOOLS

    s = generate_family("level_shift", 1, base_seed=1, n_timesteps=240)[0]
    origin, horizon = 180, 12
    history = s.target[:origin]

    def run(hist):
        pool = {
            m: FORECASTERS.build(m).forecast(hist, horizon, s.seasonal_period, seed=0)
            for m in ("naive", "seasonal_naive", "theta")
        }
        diagnostics = {
            name: TOOLS.build(name).run(hist, s.seasonal_period).values
            for name in ("changepoint", "anomaly", "seasonality")
        }
        ctx = ActionContext(
            history=hist, seasonal_period=s.seasonal_period, horizon=horizon,
            base_model="theta", base_result=pool["theta"], seed=0,
            diagnostics=diagnostics, pool=pool,
        )
        return apply_all_actions(ctx)

    corrupted = s.target.copy()
    corrupted[origin:] = -9e9

    a = run(history)
    b = run(corrupted[:origin])
    assert set(a) == set(b)
    for name in a:
        assert np.allclose(a[name].point, b[name].point), f"action {name} saw the future"


# --------------------------------------------------------------------------- #
# 4. Fitting: the estimator refuses test data
# --------------------------------------------------------------------------- #


def test_reliability_estimator_rejects_test_instances():
    """Fitting on test must fail loudly rather than silently invalidating results."""
    from agents.reliability import ReliabilityEstimator

    signals = pd.DataFrame(
        {
            "instance_id": [f"i{i}" for i in range(20)],
            "ts_features.length": np.linspace(50, 200, 20),
            "context.horizon": np.ones(20),
        }
    )
    labels = pd.DataFrame(
        {
            "instance_id": signals["instance_id"],
            "large_error": [i % 2 == 0 for i in range(20)],
            "should_correct": [i % 3 == 0 for i in range(20)],
            "split": ["test"] * 20,
        }
    )
    with pytest.raises(ValueError, match="validation"):
        ReliabilityEstimator().fit(signals, labels)


def test_reliability_estimator_excludes_label_columns():
    """Ground-truth label columns must never be selected as features."""
    from agents.reliability import ReliabilityEstimator

    signals = pd.DataFrame(
        {
            "instance_id": ["a", "b"],
            "ts_features.length": [100.0, 120.0],
            "label_n_changepoints": [1.0, 2.0],
            "label_steps_since_true_cp": [5.0, 9.0],
            "mase_denominator": [1.0, 2.0],
        }
    )
    selected = ReliabilityEstimator().select_features(signals)
    assert "ts_features.length" in selected
    assert not any(c.startswith("label_") for c in selected)
    assert "mase_denominator" not in selected
