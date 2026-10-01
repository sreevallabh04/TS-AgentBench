#!/usr/bin/env python
"""Ablation study: which component actually produces the effect?

    python scripts/run_ablations.py --config configs/experiment.yaml

Each ablation removes exactly one component and re-runs the policy replay against the
*same* cached counterfactual tensor. Because the tensor is fixed, every configuration
sees identical forecasts and identical action outcomes -- the only thing that varies is
the decision policy, which is what an ablation is supposed to isolate. Rebuilding the
tensor per ablation would reintroduce model-fitting noise and confound the comparison.

Also runs the correction-policy comparison the study design calls for: always, never,
random, threshold, and the proposed adaptive policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.policies import RGCPolicy, ValidationEvidence  # noqa: E402
from agents.reliability import ReliabilityEstimator  # noqa: E402
from benchmark.oracle import add_oracle_labels, compute_policy_regret  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR  # noqa: E402
from experiments.runner import get_tensor  # noqa: E402

log = get_logger("ablations")

#: Each ablation names the RGC stage or signal family it removes.
ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_self_critique": {"use_diagnosis": False},
    "no_uncertainty": {"use_abstention": False, "disabled_signals": ("uncertainty.",)},
    "no_model_disagreement": {"disabled_signals": ("disagreement.",)},
    "no_changepoint": {"disabled_signals": ("changepoint.",)},
    "no_anomaly": {"disabled_signals": ("anomaly.",)},
    "no_adaptive_tools": {"use_diagnosis": False},
    "no_backtesting": {"use_validation": False, "disabled_signals": ("backtest.",)},
    "no_correction_loop": {"gain_threshold": 2.0},  # gate can never open
}

#: Signal families removed from the *estimator's* feature set, not just the diagnoser.
ESTIMATOR_ABLATIONS = {
    "no_uncertainty": ("uncertainty.",),
    "no_model_disagreement": ("disagreement.",),
    "no_changepoint": ("changepoint.",),
    "no_anomaly": ("anomaly.",),
    "no_backtesting": ("backtest.",),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run", default=None,
                        help="name of a completed run under results/runs/; its "
                             "outcomes.parquet and signals.parquet are used directly, "
                             "so nothing is refit and the replay takes minutes")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    if args.run:
        # Ablations only vary the decision policy; the tensor is identical to the main
        # run's, so read it from the run directory rather than rebuilding it, which
        # recomputes every diagnostic and takes hours.
        from core.paths import RUNS_DIR
        run_dir = RUNS_DIR / args.run
        outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
        signals = pd.read_parquet(run_dir / "signals.parquet")
        log.info("replaying tensor from %s", run_dir.name)
    else:
        outcomes, signals = get_tensor(cfg, cfg.seeds[0])

    val_outcomes = outcomes[outcomes["split"] == "val"]
    test_outcomes = outcomes[outcomes["split"] == "test"]
    primary = cfg.eval.primary_metric

    val_labels = add_oracle_labels(
        val_outcomes, primary, cfg.eval.oracle_delta, cfg.eval.large_error_quantile
    )
    test_labels = add_oracle_labels(
        test_outcomes, primary, cfg.eval.oracle_delta, cfg.eval.large_error_quantile
    )
    val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]
    test_signals = signals[signals["instance_id"].isin(test_labels["instance_id"])]
    evidence = ValidationEvidence(outcomes, primary)

    rows = []
    per_instance = []
    for name, overrides in ABLATIONS.items():
        # Refit the estimator without the ablated signal family, so the ablation
        # removes the evidence everywhere rather than only from the rule-based
        # diagnoser. Otherwise the gate would smuggle the signal back in.
        ablate = ESTIMATOR_ABLATIONS.get(name, ())
        estimator = ReliabilityEstimator(
            ablate_signals=ablate, random_state=args.seed
        ).fit(val_signals, val_labels)

        gain = dict(zip(test_signals["instance_id"], estimator.predict_gain(test_signals)))
        risk = dict(zip(test_signals["instance_id"], estimator.predict_risk(test_signals)))
        threshold = ReliabilityEstimator.threshold_for_rate(
            estimator.predict_gain(val_signals), cfg.agent.target_correction_rate
        )

        params = {
            "gain_threshold": threshold,
            "gain_scores": gain,
            "risk_scores": risk,
            "acceptance_delta": cfg.agent.acceptance_delta,
            **overrides,
        }
        policy = RGCPolicy(name=name, **params)
        decisions = policy.decide(test_signals, evidence, args.seed)
        scored = compute_policy_regret(
            test_outcomes, decisions.set_index("instance_id")["chosen_action"],
            test_labels, primary,
        )
        corrected = scored[scored["corrected"]]
        per_instance.append(
            scored[["instance_id", "series_id", "chosen_metric"]].assign(ablation=name))
        rows.append(
            {
                "ablation": name,
                "mase": float(scored["chosen_metric"].mean()),
                "regret": float(scored["regret"].mean()),
                "correction_rate": float(scored["corrected"].mean()),
                "helped": float(corrected["improved"].mean()) if len(corrected) else np.nan,
                "harmed": float(corrected["harmed"].mean()) if len(corrected) else np.nan,
                "decision_accuracy": float(scored["decision_correct"].mean()),
                "compute_cost": float(decisions["compute_cost"].mean()),
                "n_features": len(estimator.feature_names_),
            }
        )
        log.info("%-24s MASE=%.4f regret=%.4f rate=%.1f%%",
                 name, rows[-1]["mase"], rows[-1]["regret"],
                 100 * rows[-1]["correction_rate"])

    df = pd.DataFrame(rows)
    base = float(df.loc[df["ablation"] == "full", "mase"].iloc[0])
    df["delta_vs_full_pct"] = 100.0 * (df["mase"] - base) / base

    # Series-level intervals. The instance-level delta above is kept for continuity, but
    # the interval is what says whether a component's removal did anything at all.
    from evaluation.series_level import paired_series_test, per_series_means
    inst = pd.concat(per_instance, ignore_index=True)
    inst.to_parquet((Path(args.out) if args.out else RESULTS_DIR / "ablations.parquet")
                    .with_name("ablations_instances.parquet"))
    series_of = inst.drop_duplicates("instance_id").set_index("instance_id")["series_id"]
    full = inst[inst["ablation"] == "full"].set_index("instance_id")["chosen_metric"]
    full_series = per_series_means(full, series_of)
    ci = {}
    for name, block in inst.groupby("ablation"):
        arm = block.set_index("instance_id")["chosen_metric"]
        st = paired_series_test(per_series_means(arm, series_of), full_series,
                                n_boot=10000, seed=args.seed)
        ci[name] = (st["pct_diff"], st["pct_ci_lo"], st["pct_ci_hi"], st["significant"])
    df["series_delta_pct"] = df["ablation"].map(lambda a: ci[a][0])
    df["series_ci_lo"] = df["ablation"].map(lambda a: ci[a][1])
    df["series_ci_hi"] = df["ablation"].map(lambda a: ci[a][2])
    df["series_significant"] = df["ablation"].map(lambda a: bool(ci[a][3]))

    out = Path(args.out or RESULTS_DIR / "ablations.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)

    print("\n=== Ablation study (removing one component at a time) ===")
    print(
        df[["ablation", "mase", "delta_vs_full_pct", "correction_rate",
            "helped", "harmed", "compute_cost"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\nSaved to {out}")
    print(
        "\nA positive delta means removing the component made things worse, i.e. the "
        "component was contributing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
