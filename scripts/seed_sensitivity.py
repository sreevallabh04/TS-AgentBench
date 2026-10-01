#!/usr/bin/env python
"""Measure sensitivity to model-internal randomness.

    python scripts/seed_sensitivity.py --config configs/smoke.yaml --seeds 0 1 2

The main experiment fixes model-internal randomness when the counterfactual tensor is
built, and varies seeds only over the *policy*, the bootstrap and subsampling. That is a
deliberate cost decision: rebuilding the tensor per seed would multiply the most
expensive stage fivefold for a source of variation the study is not about.

It is still a decision that needs checking rather than assuming. This script rebuilds
the tensor under several model seeds on a reduced dataset and reports how much the
headline comparison moves. If the spread here is comparable to the effect being claimed,
the main results are not resolvable at this scale and the paper must say so.
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
from core.config import load_config  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR  # noqa: E402
from experiments.runner import get_tensor  # noqa: E402

log = get_logger("seed_sensitivity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    primary = cfg.eval.primary_metric

    rows = []
    for seed in args.seeds:
        # A distinct seed produces a distinct tensor cache key, so each iteration
        # genuinely rebuilds with different model-internal randomness.
        outcomes, signals = get_tensor(cfg, seed)
        val_outcomes = outcomes[outcomes["split"] == "val"]
        test_outcomes = outcomes[outcomes["split"] == "test"]
        if val_outcomes.empty or test_outcomes.empty:
            log.warning("seed %d produced an incomplete tensor; skipping", seed)
            continue

        val_labels = add_oracle_labels(val_outcomes, primary, cfg.eval.oracle_delta,
                                       cfg.eval.large_error_quantile)
        test_labels = add_oracle_labels(test_outcomes, primary, cfg.eval.oracle_delta,
                                        cfg.eval.large_error_quantile)
        val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]
        test_signals = signals[signals["instance_id"].isin(test_labels["instance_id"])]

        try:
            estimator = ReliabilityEstimator(random_state=seed).fit(val_signals, val_labels)
        except ValueError as exc:
            log.warning("seed %d: estimator could not be fitted (%s)", seed, exc)
            continue

        gain = dict(zip(test_signals["instance_id"], estimator.predict_gain(test_signals)))
        risk = dict(zip(test_signals["instance_id"], estimator.predict_risk(test_signals)))
        threshold = ReliabilityEstimator.threshold_for_rate(
            estimator.predict_gain(val_signals), cfg.agent.target_correction_rate
        )
        evidence = ValidationEvidence(outcomes, primary)

        for policy in (
            NeverCorrectPolicy(),
            AlwaysCorrectPolicy(),
            RGCPolicy(gain_threshold=threshold, gain_scores=gain, risk_scores=risk,
                      acceptance_delta=cfg.agent.acceptance_delta),
        ):
            decisions = policy.decide(test_signals, evidence, seed)
            scored = compute_policy_regret(
                test_outcomes, decisions.set_index("instance_id")["chosen_action"],
                test_labels, primary,
            )
            rows.append(
                {
                    "model_seed": seed,
                    "policy": policy.name,
                    "mase": float(scored["chosen_metric"].mean()),
                    "regret": float(scored["regret"].mean()),
                    "correction_rate": float(scored["corrected"].mean()),
                }
            )
        log.info("seed %d complete", seed)

    if not rows:
        log.error("no results produced")
        return 1

    df = pd.DataFrame(rows)
    out = Path(args.out or RESULTS_DIR / "seed_sensitivity.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    summary = df.groupby("policy")["mase"].agg(["mean", "std", "min", "max"])
    print("\n=== Sensitivity to model-internal seed ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    if "never_correct" in summary.index and "rgc" in summary.index:
        effect = abs(summary.loc["rgc", "mean"] - summary.loc["never_correct", "mean"])
        spread = float(summary["std"].max())
        print(f"\nRGC vs. never-correct effect: {effect:.4f}")
        print(f"Largest across-seed standard deviation: {spread:.4f}")
        if spread >= effect:
            print(
                "\nThe across-seed spread is at least as large as the effect. At this "
                "scale the comparison is not resolvable, and the manuscript must "
                "report that rather than the point estimate."
            )
        else:
            print("\nThe effect exceeds across-seed variation at this scale.")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
