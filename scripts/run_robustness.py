#!/usr/bin/env python
"""Robustness study: how do policies degrade under controlled perturbations?

    python scripts/run_robustness.py --config configs/robustness.yaml

For each perturbation axis and severity level, a fresh counterfactual tensor is built
on perturbed copies of the *same* base series, then every policy is replayed against
it. Holding the underlying series fixed and varying only the perturbation is what makes
"performance degrades because of X" a causal statement rather than a correlation.

This is the most expensive script in the project -- it builds one tensor per
(axis, severity) cell -- so it defaults to a reduced series count and is intended to be
run once, overnight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.policies import (  # noqa: E402
    AlwaysCorrectPolicy, NeverCorrectPolicy, RGCPolicy, ValidationEvidence,
)
from agents.reliability import ReliabilityEstimator  # noqa: E402
from benchmark.oracle import add_oracle_labels, compute_policy_regret  # noqa: E402
from benchmark.tensor import build_tensor  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR  # noqa: E402
from datasets.base import iter_instances, make_split  # noqa: E402
from datasets.perturbations import SEVERITIES, apply_perturbation, perturbation_names  # noqa: E402
from datasets.synthetic import generate_family  # noqa: E402

log = get_logger("robustness")


def build_base_series(cfg, n_per_family: int, seed: int):
    """A fixed pool of clean series, perturbed identically at every severity."""
    from datasets.synthetic import family_names

    series = []
    for family in family_names():
        series += generate_family(
            family, n_per_family, base_seed=seed,
            n_timesteps=cfg.data.max_series_length // 2, seasonal_period=12,
        )
    return series


def evaluate_cell(cfg, series, seed: int) -> pd.DataFrame | None:
    """Build a tensor for one perturbation cell and replay the policies on it."""
    instances = {}
    for s in series:
        try:
            split = make_split(
                len(s), cfg.split.val_length, cfg.split.test_length,
                cfg.split.min_train_length,
            )
        except ValueError:
            continue
        items = list(
            iter_instances(
                s, split, cfg.forecast.horizons, cfg.split.n_val_origins,
                cfg.split.n_test_origins, cfg.split.origin_stride,
            )
        )
        if items:
            instances[s.series_id] = items

    usable = [s for s in series if s.series_id in instances]
    if not usable:
        return None

    outcomes, signals = build_tensor(
        usable, instances, list(cfg.forecast.models),
        coverage_levels=tuple(cfg.forecast.coverage_levels),
        base_seed=seed, n_jobs=cfg.n_jobs, progress=False,
    )
    if outcomes.empty:
        return None

    primary = cfg.eval.primary_metric
    val_outcomes = outcomes[outcomes["split"] == "val"]
    test_outcomes = outcomes[outcomes["split"] == "test"]
    if val_outcomes.empty or test_outcomes.empty:
        return None

    val_labels = add_oracle_labels(val_outcomes, primary, cfg.eval.oracle_delta,
                                   cfg.eval.large_error_quantile)
    test_labels = add_oracle_labels(test_outcomes, primary, cfg.eval.oracle_delta,
                                    cfg.eval.large_error_quantile)
    val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]
    test_signals = signals[signals["instance_id"].isin(test_labels["instance_id"])]
    if val_signals.empty or test_signals.empty:
        return None

    try:
        estimator = ReliabilityEstimator(random_state=seed).fit(val_signals, val_labels)
    except ValueError as exc:
        log.warning("estimator could not be fitted for this cell: %s", exc)
        return None

    gain = dict(zip(test_signals["instance_id"], estimator.predict_gain(test_signals)))
    risk = dict(zip(test_signals["instance_id"], estimator.predict_risk(test_signals)))
    threshold = ReliabilityEstimator.threshold_for_rate(
        estimator.predict_gain(val_signals), cfg.agent.target_correction_rate
    )
    evidence = ValidationEvidence(outcomes, primary)

    policies = [
        NeverCorrectPolicy(),
        AlwaysCorrectPolicy(),
        RGCPolicy(gain_threshold=threshold, gain_scores=gain, risk_scores=risk,
                  acceptance_delta=cfg.agent.acceptance_delta),
    ]

    frames = []
    for policy in policies:
        decisions = policy.decide(test_signals, evidence, seed)
        scored = compute_policy_regret(
            test_outcomes, decisions.set_index("instance_id")["chosen_action"],
            test_labels, primary,
        )
        scored["policy"] = policy.name
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--series-per-family", type=int, default=6)
    parser.add_argument("--axes", nargs="*", default=None)
    parser.add_argument("--severities", nargs="*", type=float, default=list(SEVERITIES))
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    axes = args.axes or perturbation_names()
    base = build_base_series(cfg, args.series_per_family, args.seed)
    # Protect the evaluation region: perturbations must never touch a forecast target.
    protect = cfg.split.val_length + cfg.split.test_length

    log.info(
        "robustness grid: %d axes x %d severities on %d base series",
        len(axes), len(args.severities), len(base),
    )

    frames = []
    for axis in axes:
        for severity in args.severities:
            perturbed = [
                apply_perturbation(s, axis, severity, protect, args.seed) for s in base
            ]
            result = evaluate_cell(cfg, perturbed, args.seed)
            if result is None:
                log.warning("cell %s@%.2f produced no usable results", axis, severity)
                continue
            result["perturbation"] = axis
            result["severity"] = severity
            frames.append(result)
            summary = result.groupby("policy")["chosen_metric"].mean()
            log.info(
                "%-22s sev=%.2f  %s", axis, severity,
                "  ".join(f"{k}={v:.3f}" for k, v in summary.items()),
            )

    if not frames:
        log.error("no robustness results produced")
        return 1

    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out or RESULTS_DIR / "robustness.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    pivot = df.pivot_table(
        index=["perturbation", "severity"], columns="policy", values="chosen_metric"
    )
    print("\n=== Robustness: mean MASE by perturbation and severity ===")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
