"""Few-shot prompting and the action-choice span statistics.

Two pieces of machinery added in response to review. The few-shot prefix exists to
close a confound --- the arithmetic gate is fitted on validation while the LLM arms are
zero-shot --- so the tests here are mostly about the prefix not introducing a *new*
confound in the process: examples must come only from validation, must not be balanced
when the population is not, and must not quietly teach a single repair.

The span statistics turn "the model's pick is 72% worse than the best action available"
into a number with an interval. They are checked against constructed cases whose answers
are known by construction, because a normalised regret that is subtly wrong would be
very hard to notice in a results table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.llm_agent import build_evidence_prompt, build_fewshot_prefix  # noqa: E402
from scripts.action_choice import bootstrap_span_ratios  # noqa: E402
from scripts.llm_arms import few_shot_examples  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _signals(n: int, prefix: str = "v") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "instance_id": [f"{prefix}{i}" for i in range(n)],
        "series_id": [f"{prefix}s{i // 2}" for i in range(n)],
        "context.history_length": rng.integers(50, 500, n),
        "stl.trend_strength": rng.random(n),
        "backtest.backtest_mase": rng.random(n) * 2,
    })


def _labels(n: int, actions: list[str], prefix: str = "v") -> pd.DataFrame:
    """``should_correct`` is true exactly when the oracle action is not ``accept``."""
    picked = [actions[i % len(actions)] for i in range(n)]
    return pd.DataFrame({
        "instance_id": [f"{prefix}{i}" for i in range(n)],
        "oracle_action": picked,
        "should_correct": [a != "accept" for a in picked],
    })


# --------------------------------------------------------------------------- #
# Example selection
# --------------------------------------------------------------------------- #

def test_examples_are_drawn_only_from_validation():
    sig, lab = _signals(40), _labels(40, ["accept", "ensemble", "robustify"])
    shots = few_shot_examples(sig, lab, 6, seed=0)
    assert len(shots) == 6
    # Every exemplar's diagnostics must be a row that exists in the validation frame.
    valid = set(sig.set_index("instance_id").index)
    assert all(row.name in valid for row, _, _ in shots)


def test_mix_follows_the_base_rate_rather_than_being_balanced():
    """A balanced prefix would itself hint at how often to intervene."""
    # Base rate 0.75: three of every four oracle actions are a real repair.
    sig = _signals(80)
    lab = _labels(80, ["accept", "ensemble", "robustify", "switch_model"])
    shots = few_shot_examples(sig, lab, 8, seed=0)
    positives = sum(1 for _, needs, _ in shots if needs)
    assert positives == 6, f"expected 6 of 8 positive at a 0.75 base rate, got {positives}"


def test_positives_are_spread_across_distinct_oracle_actions():
    """Otherwise the prefix teaches one repair instead of how to choose."""
    sig = _signals(120)
    lab = _labels(120, ["accept", "ensemble", "robustify", "switch_model", "damp_trend"])
    shots = few_shot_examples(sig, lab, 8, seed=0)
    repairs = {a for _, needs, a in shots if needs}
    assert len(repairs) >= 4, f"positives collapsed onto {repairs}"


def test_zero_requested_examples_is_the_zero_shot_default():
    sig, lab = _signals(20), _labels(20, ["accept", "ensemble"])
    assert few_shot_examples(sig, lab, 0, seed=0) == ()
    assert build_fewshot_prefix(()) == ""


def test_selection_is_deterministic_under_a_fixed_seed():
    sig, lab = _signals(60), _labels(60, ["accept", "ensemble", "robustify"])
    a = few_shot_examples(sig, lab, 6, seed=3)
    b = few_shot_examples(sig, lab, 6, seed=3)
    assert [r.name for r, _, _ in a] == [r.name for r, _, _ in b]


# --------------------------------------------------------------------------- #
# Prefix rendering
# --------------------------------------------------------------------------- #

def test_prefix_leaks_no_identifiers_or_labels_beyond_the_decision():
    sig, lab = _signals(40), _labels(40, ["accept", "ensemble", "robustify"])
    shots = few_shot_examples(sig, lab, 5, seed=0)
    prefix = build_fewshot_prefix(shots, ask_gate=True)
    for banned in ("instance_id", "series_id", "should_correct", "v0", "vs0"):
        assert banned not in prefix, f"{banned!r} leaked into the prompt"


def test_gate_examples_carry_the_gate_field_and_choice_examples_do_not():
    sig, lab = _signals(40), _labels(40, ["accept", "ensemble"])
    shots = few_shot_examples(sig, lab, 4, seed=0)
    assert "needs_correction" in build_fewshot_prefix(shots, ask_gate=True)
    assert "needs_correction" not in build_fewshot_prefix(shots, ask_gate=False)


def test_prefix_changes_the_prompt_and_therefore_the_cache_key():
    """A few-shot run must not be able to read a zero-shot run's cached answers."""
    sig, lab = _signals(40), _labels(40, ["accept", "ensemble"])
    shots = few_shot_examples(sig, lab, 4, seed=0)
    query = _signals(1, prefix="t").iloc[0]
    zero = build_evidence_prompt(query, ask_gate=True)
    few = build_evidence_prompt(query, ask_gate=True, few_shot=shots)
    assert few != zero
    assert few.endswith(zero), "the query block must be unchanged, only prefixed"
    assert few.count("TIME SERIES DIAGNOSTICS") == len(shots) + 1


# --------------------------------------------------------------------------- #
# Span statistics
# --------------------------------------------------------------------------- #

def test_headroom_recovered_is_zero_when_the_choice_equals_the_coin():
    coin = np.array([1.0, 1.2, 0.9, 1.4])
    best = np.array([0.5, 0.6, 0.4, 0.7])
    out = bootstrap_span_ratios(coin.copy(), coin, best, 200, 0)
    assert out["headroom_recovered_pct"] == pytest.approx(0.0, abs=1e-9)


def test_headroom_recovered_is_total_when_the_choice_is_always_the_best():
    coin = np.array([1.0, 1.2, 0.9, 1.4])
    best = np.array([0.5, 0.6, 0.4, 0.7])
    out = bootstrap_span_ratios(best.copy(), coin, best, 200, 0)
    assert out["headroom_recovered_pct"] == pytest.approx(100.0, abs=1e-9)
    assert out["oracle_gap_pct"] == pytest.approx(0.0, abs=1e-9)


def test_headroom_is_negative_when_the_choice_is_worse_than_the_coin():
    coin = np.array([1.0, 1.0, 1.0, 1.0])
    best = np.array([0.5, 0.5, 0.5, 0.5])
    chosen = np.array([1.1, 1.1, 1.1, 1.1])
    out = bootstrap_span_ratios(chosen, coin, best, 200, 0)
    assert out["headroom_recovered_pct"] < 0
    assert out["oracle_gap_pct"] == pytest.approx(120.0)


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    best = rng.random(200) * 0.5
    coin = best + 0.5 + rng.random(200) * 0.2
    chosen = coin + rng.normal(0, 0.05, 200)
    out = bootstrap_span_ratios(chosen, coin, best, 2000, 0)
    assert out["oracle_gap_ci_lo"] <= out["oracle_gap_pct"] <= out["oracle_gap_ci_hi"]
    assert out["headroom_ci_lo"] <= out["headroom_recovered_pct"] <= out["headroom_ci_hi"]


def test_empty_input_yields_no_statistics_rather_than_a_crash():
    assert bootstrap_span_ratios(np.array([]), np.array([]), np.array([]), 10, 0) == {}
