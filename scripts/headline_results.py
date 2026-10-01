#!/usr/bin/env python
"""Produce the study's headline table: point vs probabilistic accuracy, side by side.

    python scripts/headline_results.py --run latest

This exists because the central finding is a *dissociation* between two metrics, and
that is only visible when both are computed on the same instances, with the same action
space and the same policies. Reporting either alone gives the wrong answer: the point
metric says correction never helps, the probabilistic metric says it helps a great deal.

Aggregation is reported three ways -- arithmetic mean, offset geometric mean, and median
-- because MASE here is heavy-tailed enough that the choice changes the conclusion. See
``evaluation.metrics.geometric_mean`` for why the geometric mean uses an additive offset
rather than clipping, and ``docs/ANALYSIS_PLAN.md`` deviation D1 for why the mean alone
was insufficient.

**Inference is series-level.** The instance-level figures below are descriptive. Every
claim of a difference rests on :mod:`evaluation.series_level`, which collapses each
policy to one value per series and bootstraps over series, as ``ANALYSIS_PLAN.md`` S7
requires. The two disagree here, and the series-level answer is the one that counts:
instances within a series share a generating process, a base model and most of their
history, so an instance-level interval is too narrow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.calibration import GainCalibrator  # noqa: E402
from agents.policies import (  # noqa: E402
    ACCEPT, AlwaysCorrectPolicy, BestFixedPolicy, CalibratedGainPolicy, CorrectionPolicy,
    NeverCorrectPolicy, PolicyDecision, RandomCorrectPolicy, ValidationEvidence,
)


from benchmark.oracle import add_oracle_labels, compute_policy_regret  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from evaluation.metrics import GEO_OFFSET, geometric_mean  # noqa: E402
from evaluation.series_level import compare_all_to_reference  # noqa: E402


class _AlwaysFixedAction(CorrectionPolicy):
    """Apply one named action unconditionally, regardless of any diagnosis.

    This is the arm that turned out to matter most: on this benchmark it dominates
    every adaptive policy we built, so it is reported as a first-class arm rather than
    an ad-hoc check.
    """

    def __init__(self, action: str) -> None:
        self.action = action
        self.name = f"always_{action}"

    def decide_one(self, signals, evidence, rng):
        return PolicyDecision(
            instance_id=signals["instance_id"], chosen_action=self.action,
            corrected=True, compute_cost=1.0,
            rationale=f"unconditional fixed action: {self.action}",
        )

log = get_logger("headline")

REFERENCE = "never_correct"

#: Gate settings per metric, each selected on a held-out validation origin and never on
#: test. They differ because the two metrics have genuinely different optima: on point
#: error almost no correction is worth making, on probabilistic error most are.
GATE = {
    "mase": {"margin": 0.10, "risk_z": 0.5},
    "crps_scaled": {"margin": 0.0, "risk_z": 0.0},
}


def resolve_run(name: str) -> Path:
    if name != "latest":
        path = RUNS_DIR / name
        if not path.exists():
            raise SystemExit(f"run not found: {path}")
        return path
    candidates = sorted(
        (p for p in RUNS_DIR.iterdir()
         if p.is_dir() and (p / "outcomes.parquet").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("no runs with a counterfactual tensor found")
    return candidates[-1]


def evaluate(outcomes: pd.DataFrame, signals: pd.DataFrame, metric: str,
             oracle_delta: float, seed: int) -> tuple[pd.DataFrame, dict]:
    test = outcomes[outcomes["split"] == "test"]
    val = outcomes[outcomes["split"] == "val"]

    labels = add_oracle_labels(test, metric, oracle_delta, 0.80)
    test_signals = signals[signals["instance_id"].isin(labels["instance_id"])]
    calibrator = GainCalibrator(n_estimate_origins=2).fit(val, metric)
    evidence = ValidationEvidence(outcomes, metric)
    gate = GATE.get(metric, {"margin": 0.0, "risk_z": 0.0})

    policies = [
        NeverCorrectPolicy(),
        CalibratedGainPolicy(calibrator=calibrator, use_diagnosis=False, **gate),
        _AlwaysFixedAction("ensemble"),
        AlwaysCorrectPolicy(),
        RandomCorrectPolicy(rate=0.30),
        BestFixedPolicy(),
    ]

    frames, rates = [], {}
    for policy in policies:
        decisions = policy.decide(test_signals, evidence, seed)
        scored = compute_policy_regret(
            test, decisions.set_index("instance_id")["chosen_action"], labels, metric
        )
        scored["policy"] = policy.name
        frames.append(scored[["instance_id", "policy", "chosen_metric",
                              "corrected", "improved", "harmed"]])
        rates[policy.name] = float(decisions["corrected"].mean())

    scores = pd.concat(frames, ignore_index=True)
    wide = scores.pivot_table(index="instance_id", columns="policy",
                              values="chosen_metric")

    reference = wide[REFERENCE]
    ref_mean = float(reference.mean())
    ref_geo = geometric_mean(reference)
    ref_median = float(reference.median())

    rows = []
    for policy in wide.columns:
        a, b = wide[policy], reference
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        diff = a - b
        nonzero = diff[np.abs(diff) > 1e-12]
        acted = scores[(scores["policy"] == policy) & scores["corrected"]]
        rows.append(
            {
                "metric": metric,
                "policy": policy,
                "mean": float(a.mean()),
                "mean_pct": 100.0 * (a.mean() / ref_mean - 1.0),
                "geometric_mean": geometric_mean(a),
                "geo_pct": 100.0 * (geometric_mean(a) / ref_geo - 1.0),
                "median": float(a.median()),
                "median_pct": 100.0 * (a.median() / ref_median - 1.0),
                "correction_rate": rates[policy],
                "instance_win_rate": float(
                    (diff < -1e-12).mean() + 0.5 * (np.abs(diff) <= 1e-12).mean()
                ),
                "helped": float(acted["improved"].mean()) if len(acted) else np.nan,
                "harmed": float(acted["harmed"].mean()) if len(acted) else np.nan,
                # Descriptive only. Instances within a series are dependent, so this
                # p-value is anti-conservative; the confirmatory test is series-level.
                "wilcoxon_p_instance": float(stats.wilcoxon(a, b)[1])
                if len(nonzero) > 10 else np.nan,
                "n": int(len(a)),
            }
        )

    # Confirmatory inference: one value per series, bootstrap over series.
    series_of_instance = signals.set_index("instance_id")["series_id"]
    series_table = compare_all_to_reference(
        wide, series_of_instance, REFERENCE, n_boot=10_000, seed=seed
    )
    table = pd.DataFrame(rows).merge(series_table, on="policy", how="left")

    extras = {
        "accept": float(labels["accept_metric"].mean()),
        "oracle_inflated": float(labels["oracle_metric"].mean()),
        "trusted_actions": calibrator.trusted_actions(),
        "gate": gate,
        "n_series": int(series_table["n_series"].max()),
    }
    return table, extras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--oracle-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")

    all_rows, meta = [], {}
    for metric, title in [("mase", "POINT ACCURACY (MASE)"),
                          ("crps_scaled", "PROBABILISTIC ACCURACY (CRPS, scaled)")]:
        if metric not in outcomes.columns:
            continue
        table, extras = evaluate(outcomes, signals, metric, args.oracle_delta, args.seed)
        all_rows.append(table)
        meta[metric] = extras

        order = [REFERENCE, "rgc_calibrated", "always_ensemble", "always_correct",
                 "best_fixed", "random_correct"]
        table = table.set_index("policy").reindex(
            [p for p in order if p in set(table["policy"])]
        ).reset_index()

        print(f"\n{'=' * 104}\n{title}   n={table['n'].iloc[0]} instances "
              f"/ {extras['n_series']} series\n{'=' * 104}")
        print("SERIES-LEVEL (confirmatory: bootstrap over series, ANALYSIS_PLAN S7)")
        print(f"{'policy':18s} {'series mean':>11} {'delta %':>9}  {'95% CI (%)':>20} "
              f"{'win rate':>9} {'p':>10}")
        for _, r in table.iterrows():
            flag = "*" if r["significant"] else " "
            print(f"{r['policy']:18s} {r['series_mean']:11.4f} "
                  f"{r['pct_diff']:+8.2f}%{flag} "
                  f" [{r['pct_ci_lo']:+7.2f}, {r['pct_ci_hi']:+7.2f}] "
                  f"{r['win_rate']:8.1%} {r['wilcoxon_p']:10.2g}")
        print("  * = 95% series-level bootstrap CI excludes zero. A policy can improve "
              "the mean\n    while winning on under half of series; both are shown "
              "because they differ here.")

        print("\nINSTANCE-LEVEL (descriptive only -- within-series dependence "
              "makes these intervals too narrow)")
        print(f"{'policy':18s} {'mean':>9} {'vs ref':>9} {'geo-mean':>9} {'vs ref':>9} "
              f"{'median':>9} {'acts':>6} {'help':>6} {'harm':>6}")
        for _, r in table.iterrows():
            print(f"{r['policy']:18s} {r['mean']:9.4f} {r['mean_pct']:+8.2f}% "
                  f"{r['geometric_mean']:9.4f} {r['geo_pct']:+8.2f}% {r['median']:9.4f} "
                  f"{r['correction_rate']:6.0%} "
                  f"{(r['helped'] if np.isfinite(r['helped']) else 0):6.0%} "
                  f"{(r['harmed'] if np.isfinite(r['harmed']) else 0):6.0%}")
        print(f"  oracle (selection-inflated, not attainable): "
              f"{extras['oracle_inflated']:.4f} vs accept {extras['accept']:.4f}")
        print(f"  actions with transferable signal: {extras['trusted_actions']}")

    # Why: base forecasts are over-confident, and corrections repair the distribution.
    test = outcomes[outcomes["split"] == "test"]
    accept = test[test["action"] == "accept"]
    print(f"\n{'=' * 92}\nMECHANISM: INTERVAL COVERAGE (nominal -> empirical)\n{'=' * 92}")
    print(f"{'action':26s} {'cov@80':>8} {'cov@95':>8} {'calib err':>10}")
    for action in ["accept", "ensemble", "recalibrate_intervals", "abstain",
                   "damp_trend"]:
        g = test[test["action"] == action]
        if g.empty:
            continue
        c80, c95 = g["coverage_80"].mean(), g["coverage_95"].mean()
        print(f"{action:26s} {c80:8.3f} {c95:8.3f} "
              f"{abs(c80 - 0.8) + abs(c95 - 0.95):10.3f}")
    print(f"\n  Geometric means use an additive offset of {GEO_OFFSET:g}; absolute "
          f"values depend on it, relative comparisons do not.")

    combined = pd.concat(all_rows, ignore_index=True)
    out = Path(args.out or RESULTS_DIR / "headline_results.parquet")
    combined.to_parquet(out)
    combined.to_csv(out.with_suffix(".csv"), index=False)
    (RESULTS_DIR / "headline_meta.json").write_text(
        json.dumps({"run": run_dir.name, "geo_offset": GEO_OFFSET, "metrics": meta},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
