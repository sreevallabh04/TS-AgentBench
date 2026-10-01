"""Series-level inference tests.

These are validated against constructed cases with known answers, because the whole
point of this module is that it produces *different* answers from the instance-level
version it replaced (deviation D7), and "different" is only reassuring if the new
answers are demonstrably right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.series_level import (  # noqa: E402
    cliffs_delta, compare_all_to_reference, paired_series_test, per_series_means,
)


def test_per_series_means_collapses_to_one_value_per_series():
    values = pd.Series({"i1": 1.0, "i2": 3.0, "i3": 10.0})
    mapping = pd.Series({"i1": "s1", "i2": "s1", "i3": "s2"})
    out = per_series_means(values, mapping)
    assert len(out) == 2
    assert out["s1"] == pytest.approx(2.0)
    assert out["s2"] == pytest.approx(10.0)


def test_identical_policies_give_zero_effect_and_half_win_rate():
    """A policy compared against itself must be exactly neutral.

    The win rate in particular has to count ties as half, or a policy that never acts
    would appear to lose to itself.
    """
    values = pd.Series(np.linspace(0.5, 2.0, 40), index=[f"s{i}" for i in range(40)])
    out = paired_series_test(values, values, n_boot=200, seed=0)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["win_rate"] == pytest.approx(0.5)
    assert out["ci_lo"] == pytest.approx(0.0)
    assert out["ci_hi"] == pytest.approx(0.0)
    assert out["n_series"] == 40


def test_constant_improvement_is_detected_with_an_interval_excluding_zero():
    reference = pd.Series(np.full(50, 1.0), index=[f"s{i}" for i in range(50)])
    treatment = reference - 0.2
    out = paired_series_test(treatment, reference, n_boot=2000, seed=0)
    assert out["mean_diff"] == pytest.approx(-0.2)
    assert out["pct_diff"] == pytest.approx(-20.0)
    assert out["win_rate"] == pytest.approx(1.0)
    assert out["ci_hi"] < 0
    assert out["significant"] is True


def test_pure_noise_gives_an_interval_containing_zero():
    rng = np.random.default_rng(0)
    index = [f"s{i}" for i in range(200)]
    reference = pd.Series(rng.normal(1.0, 0.2, 200), index=index)
    treatment = pd.Series(rng.normal(1.0, 0.2, 200), index=index)
    out = paired_series_test(treatment, reference, n_boot=2000, seed=0)
    assert out["ci_lo"] < 0 < out["ci_hi"]
    assert out["significant"] is False


def test_series_level_interval_is_wider_than_the_instance_level_one():
    """The reason this module exists.

    Instances within a series are dependent. Ten near-identical instances from each of
    twenty series carry roughly twenty series' worth of information, not two hundred
    instances' worth. Treating them as independent must produce a narrower -- that is,
    anti-conservative -- interval, and the series-level computation must not.
    """
    rng = np.random.default_rng(3)
    n_series, per_series = 20, 10
    series_effects = rng.normal(0.0, 0.5, n_series)

    rows = []
    for s in range(n_series):
        for k in range(per_series):
            rows.append({
                "instance_id": f"i{s}_{k}",
                "series_id": f"s{s}",
                # Almost all the variance lives between series, not within them.
                "treatment": 1.0 + series_effects[s] + rng.normal(0, 0.01),
                "reference": 1.0 + rng.normal(0, 0.01),
            })
    df = pd.DataFrame(rows).set_index("instance_id")
    mapping = df["series_id"]

    series_out = paired_series_test(
        per_series_means(df["treatment"], mapping),
        per_series_means(df["reference"], mapping),
        n_boot=4000, seed=0,
    )
    series_width = series_out["ci_hi"] - series_out["ci_lo"]

    # The naive instance-level bootstrap, for comparison.
    diff = (df["treatment"] - df["reference"]).to_numpy()
    rng2 = np.random.default_rng(0)
    idx = rng2.integers(0, diff.size, size=(4000, diff.size))
    boot = diff[idx].mean(axis=1)
    instance_width = float(np.quantile(boot, 0.975) - np.quantile(boot, 0.025))

    assert series_width > instance_width * 2, (
        f"series-level interval ({series_width:.4f}) should be much wider than the "
        f"instance-level one ({instance_width:.4f}) under strong within-series "
        f"dependence"
    )


def test_win_rate_and_mean_can_disagree():
    """Heavy tails let a policy improve the mean while losing on most series.

    This is deviation D2, and later D7, in miniature. Both statistics are reported
    because either alone is misleading here.
    """
    # Loses slightly on 70 series, wins enormously on 30.
    treatment = pd.Series(
        [1.01] * 70 + [0.1] * 30, index=[f"s{i}" for i in range(100)]
    )
    reference = pd.Series([1.0] * 100, index=[f"s{i}" for i in range(100)])
    out = paired_series_test(treatment, reference, n_boot=2000, seed=0)
    assert out["mean_diff"] < 0          # the mean says "better"
    assert out["win_rate"] < 0.5         # the win rate says "worse on most series"


def test_cliffs_delta_sign_and_bounds():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cliffs_delta(a, b) == pytest.approx(-1.0)
    assert cliffs_delta(b, a) == pytest.approx(1.0)
    assert cliffs_delta(a, a) == pytest.approx(0.0)


def test_empty_input_returns_nan_rather_than_raising():
    empty = pd.Series(dtype=float)
    out = paired_series_test(empty, empty, n_boot=10)
    assert out["n_series"] == 0
    assert np.isnan(out["mean_diff"])
    assert out["significant"] is False


def test_compare_all_to_reference_covers_every_policy():
    index = [f"i{i}" for i in range(30)]
    mapping = pd.Series([f"s{i // 3}" for i in range(30)], index=index)
    wide = pd.DataFrame(
        {
            "never_correct": np.full(30, 1.0),
            "better": np.full(30, 0.8),
            "worse": np.full(30, 1.2),
        },
        index=index,
    )
    out = compare_all_to_reference(wide, mapping, "never_correct", n_boot=500, seed=0)
    assert set(out["policy"]) == {"never_correct", "better", "worse"}
    assert out.set_index("policy").loc["better", "pct_diff"] == pytest.approx(-20.0)
    assert out.set_index("policy").loc["worse", "pct_diff"] == pytest.approx(20.0)
    assert out.set_index("policy").loc["never_correct", "mean_diff"] == pytest.approx(0.0)
    # 30 instances, 3 per series -> 10 series is the inference unit, not 30.
    assert set(out["n_series"]) == {10}
