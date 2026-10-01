"""Metric correctness tests.

Metrics define every number in the paper, so they are validated against closed-form
answers wherever one exists rather than against their own previous output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (  # noqa: E402
    aurc, coverage, crps_from_samples, expected_calibration_error, mae, mase,
    mase_denominator, mean_interval_width, rmse, smape, wape, winkler_score,
)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #


def test_point_metrics_are_zero_for_perfect_forecasts():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(x, x) == 0.0
    assert rmse(x, x) == 0.0
    assert smape(x, x) == 0.0
    assert wape(x, x) == 0.0


def test_rmse_penalises_large_errors_more_than_mae():
    actual = np.zeros(4)
    concentrated = np.array([4.0, 0.0, 0.0, 0.0])
    spread = np.array([1.0, 1.0, 1.0, 1.0])
    assert mae(actual, concentrated) == mae(actual, spread)
    assert rmse(actual, concentrated) > rmse(actual, spread)


def test_smape_is_bounded_at_200():
    assert smape(np.array([1.0]), np.array([-1.0])) == pytest.approx(200.0)


def test_smape_skips_undefined_points():
    """Points where actual and prediction are both zero are dropped, not scored as 0."""
    assert smape(np.array([0.0, 2.0]), np.array([0.0, 4.0])) == pytest.approx(
        smape(np.array([2.0]), np.array([4.0]))
    )


def test_mase_denominator_uses_seasonal_differences():
    m = 12
    x = np.tile(np.arange(m, dtype=float), 8)  # perfectly seasonal
    assert mase_denominator(x, m) == pytest.approx(1.0)  # guarded floor for zero diff


def test_mase_denominator_never_returns_zero():
    """A constant series must not produce a divide-by-zero MASE."""
    assert mase_denominator(np.ones(50), 12) == 1.0
    assert mase_denominator(np.array([5.0]), 12) == 1.0


def test_seasonal_naive_has_mase_of_about_one_on_random_walk():
    """The defining property of MASE: the seasonal naive benchmark scores about 1."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=600))
    train, actual = x[:500], x[500:512]
    pred = train[-12:]
    denom = mase_denominator(train, 12)
    assert 0.3 < mase(actual, pred, denom) < 3.0


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #


def _crps_normal(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


@pytest.mark.parametrize("mu,sigma,y", [(0, 1, 0.0), (0, 1, 1.5), (5, 2, 3.0)])
def test_crps_matches_the_gaussian_closed_form(mu, sigma, y):
    rng = np.random.default_rng(0)
    samples = rng.normal(mu, sigma, size=(100_000, 1))
    assert crps_from_samples(np.array([y]), samples) == pytest.approx(
        _crps_normal(mu, sigma, y), abs=0.02
    )


def test_crps_fast_path_matches_the_quadratic_definition():
    """The O(n log n) sorted identity must equal the O(n^2) pairwise definition."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(200, 3))
    y = rng.normal(size=3)
    naive = np.mean(
        [
            np.mean(np.abs(samples[:, j] - y[j]))
            - 0.5 * np.mean(np.abs(samples[:, j][:, None] - samples[:, j][None, :]))
            for j in range(3)
        ]
    )
    assert crps_from_samples(y, samples) == pytest.approx(naive, abs=1e-9)


def test_crps_is_a_proper_score():
    """The true predictive distribution must score best."""
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 300)
    truth = crps_from_samples(y, rng.normal(0, 1, size=(400, 300)))
    shifted = crps_from_samples(y, rng.normal(2, 1, size=(400, 300)))
    too_wide = crps_from_samples(y, rng.normal(0, 4, size=(400, 300)))
    assert truth < shifted
    assert truth < too_wide


# --------------------------------------------------------------------------- #
# Intervals and calibration
# --------------------------------------------------------------------------- #


def test_coverage_counts_points_inside_the_interval():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(actual, np.zeros(4), np.full(4, 2.0)) == pytest.approx(0.5)


def test_interval_width_is_the_mean_gap():
    assert mean_interval_width(np.zeros(3), np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_winkler_penalises_width_and_misses():
    y = np.array([1.0])
    tight_and_right = winkler_score(y, np.array([0.9]), np.array([1.1]), 0.8)
    very_wide = winkler_score(y, np.array([-10.0]), np.array([10.0]), 0.8)
    tight_and_wrong = winkler_score(y, np.array([5.0]), np.array([5.2]), 0.8)
    assert tight_and_right < very_wide
    assert tight_and_right < tight_and_wrong


def test_ece_is_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.02


def test_ece_is_large_for_inverted_probabilities():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(float)
    assert expected_calibration_error(1 - p, y) > 0.3


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #


def test_aurc_rewards_informative_confidence():
    rng = np.random.default_rng(5)
    losses = rng.exponential(1.0, 2000)
    informative = -losses          # perfectly ranks the bad cases
    uninformative = rng.normal(size=2000)
    assert aurc(losses, informative) < aurc(losses, uninformative)


def test_aurc_handles_empty_input():
    assert np.isnan(aurc(np.array([]), np.array([])))
