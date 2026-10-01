#!/usr/bin/env python
"""H5: is a calibrated risk estimate better calibrated than the agent's own confidence?

    python scripts/confidence_calibration.py --run latest

The LLM arms are asked, in the same breath as the decision, for a number between 0 and 1
expressing how reliable they believe the forecast to be. That number is a claim about the
same event the reliability estimator predicts --- whether this forecast will turn out to
have a large error --- so the two can be scored against each other on identical instances
against an identical label. This is pre-registered hypothesis H5, and it is the last one
in the plan that the counterfactual tensor makes answerable without a new experiment.

The comparison is deliberately narrow. Both predictors are asked for P(large error):

* the **language model**, via ``1 - confidence``, where confidence is its stated belief
  that the forecast as it stands is reliable;
* the **reliability estimator**, via its isotonic-calibrated risk head.

They are scored three ways, because calibration and ranking are different virtues and a
predictor can be good at one and useless at the other:

* **ECE** --- expected calibration error, equal-width bins. Does a stated 0.8 mean 0.8?
* **Brier score** --- the proper scoring rule, which penalises both miscalibration and
  poor discrimination, so it cannot be gamed by a predictor that hedges to the base rate.
* **AURC** --- area under the risk-coverage curve. If you abstained on the forecasts each
  predictor was least sure about, how fast would your error fall? This is the quantity a
  practitioner actually cares about, and it is pure ranking: it is unaffected by
  miscalibration that a monotone rescaling would fix.

A constant predictor emitting the base rate is included as the floor. It has near-zero
ECE by construction and no ranking ability at all, which is exactly why ECE alone is not
allowed to settle the question.

Intervals are paired bootstraps over **series**, matching the rest of the paper.
Uses only cached responses and the precomputed tensor: no API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.policies import ValidationEvidence  # noqa: E402
from benchmark.oracle import add_oracle_labels  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402

log = get_logger("confidence_calibration")


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


def ece(prob: np.ndarray, truth: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error over equal-width bins.

    Bins with no instances contribute nothing rather than counting as perfectly
    calibrated, which is the usual way this statistic is quietly flattered.
    """
    prob = np.clip(prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(prob[mask].mean() - truth[mask].mean())
    return float(total)


def brier(prob: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.clip(prob, 0.0, 1.0) - truth) ** 2))


def aurc(prob: np.ndarray, truth: np.ndarray) -> float:
    """Area under the risk-coverage curve, lower is better.

    Instances are ordered from the ones the predictor is most confident about to the
    least, and the cumulative error rate is read off as coverage grows. A predictor that
    ranks perfectly puts every error last and gets the smallest area.
    """
    order = np.argsort(prob, kind="mergesort")  # ascending predicted risk
    errors = truth[order]
    cumulative = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(np.mean(cumulative))


def paired_bootstrap(
    per_series: pd.DataFrame, stat: str, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap the LLM-minus-estimator difference by resampling series."""
    diff = (per_series[f"llm_{stat}"] - per_series[f"est_{stat}"]).to_numpy()
    n = diff.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(diff.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--metric", default="mase")
    parser.add_argument("--oracle-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="must match the llm_arms run: it is part of the cache key")
    parser.add_argument("--reasoning-effort", default=None,
                        help="must match the llm_arms run being analysed: it is part of "
                             "the response cache key")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    from dotenv import load_dotenv
    load_dotenv()

    from agents.llm import LLMClient
    from agents.llm_agent import LLMCritiqueGatePolicy
    from agents.reliability import ReliabilityEstimator
    from scripts.llm_arms import series_spread_sample

    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")
    test = outcomes[outcomes["split"] == "test"]
    val = outcomes[outcomes["split"] == "val"]

    labels = add_oracle_labels(test, args.metric, args.oracle_delta, 0.80)
    val_labels = add_oracle_labels(val, args.metric, args.oracle_delta, 0.80)
    test_signals = signals[signals["instance_id"].isin(labels["instance_id"])]
    val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]

    sample = series_spread_sample(test_signals, 1, args.n, args.seed)

    # cache_only: H5 must cost nothing and must compare the same decisions the arms made.
    client = LLMClient(provider=args.provider, model=args.model, cache_only=True,
                       max_output_tokens=args.max_output_tokens,
                       reasoning_effort=args.reasoning_effort,
                       requests_per_minute=0)
    evidence = ValidationEvidence(outcomes, args.metric)

    # The gate arm is the one asked for a reliability judgement in so many words, which
    # is what H5 names. The single-pass arm reports confidence in the *resulting*
    # forecast after its chosen repair, a different quantity, so it is not used here.
    decisions = LLMCritiqueGatePolicy(client=client, n_workers=1).decide(
        sample, evidence, args.seed).set_index("instance_id")

    estimator = ReliabilityEstimator(random_state=args.seed).fit(val_signals, val_labels)
    est_risk = pd.Series(estimator.predict_risk(sample),
                         index=sample["instance_id"].values)

    truth = labels.set_index("instance_id")["large_error"].astype(float)
    llm_risk = 1.0 - decisions["confidence"].astype(float)

    common = (llm_risk.dropna().index
              .intersection(est_risk.index)
              .intersection(truth.index))
    if len(common) < 50:
        raise SystemExit(
            f"only {len(common)} instances carry both a cached confidence and a label; "
            "run scripts/llm_arms.py for this model first"
        )
    llm_risk, est_risk, truth = (llm_risk.loc[common], est_risk.loc[common],
                                 truth.loc[common])
    base_rate = float(truth.mean())
    constant = pd.Series(base_rate, index=common)

    series_of = signals.set_index("instance_id")["series_id"].reindex(common)

    rows = []
    for name, prob in (("LLM verbalised confidence", llm_risk),
                       ("Reliability estimator (calibrated)", est_risk),
                       (f"Constant at base rate ({base_rate:.2f})", constant)):
        p, y = prob.to_numpy(), truth.to_numpy()
        rows.append({
            "predictor": name,
            "ece": ece(p, y),
            "brier": brier(p, y),
            "aurc": aurc(p, y),
            "mean_prediction": float(p.mean()),
            "n": int(len(common)),
        })
    table = pd.DataFrame(rows)

    # Per-series statistics, so the interval on the difference resamples the same unit
    # the rest of the paper resamples.
    per_series = []
    for series_id, idx in pd.Series(common, index=series_of.values).groupby(level=0):
        ids = idx.to_numpy()
        y = truth.loc[ids].to_numpy()
        if y.size == 0:
            continue
        per_series.append({
            "series_id": series_id,
            "llm_ece": ece(llm_risk.loc[ids].to_numpy(), y),
            "est_ece": ece(est_risk.loc[ids].to_numpy(), y),
            "llm_brier": brier(llm_risk.loc[ids].to_numpy(), y),
            "est_brier": brier(est_risk.loc[ids].to_numpy(), y),
        })
    per_series = pd.DataFrame(per_series)

    diffs = {}
    for stat in ("ece", "brier"):
        point, lo, hi = paired_bootstrap(per_series, stat, args.n_boot, args.seed)
        diffs[stat] = {"diff": point, "ci_lo": lo, "ci_hi": hi,
                       "estimator_better": bool(lo > 0)}

    print(f"\n{'=' * 92}")
    print(f"H5: CALIBRATED RISK ESTIMATE VS THE AGENT'S OWN CONFIDENCE   "
          f"metric={args.metric}")
    print(f"{'=' * 92}")
    print(f"Both predict P(large error) on the same {len(common)} instances against the "
          f"same label.")
    print(f"Base rate of large errors: {base_rate:.1%}. Lower is better in every column.\n")
    print(f"{'predictor':38s} {'ECE':>8} {'Brier':>8} {'AURC':>8} {'mean p':>8}")
    for _, r in table.iterrows():
        print(f"{r['predictor']:38s} {r['ece']:8.4f} {r['brier']:8.4f} "
              f"{r['aurc']:8.4f} {r['mean_prediction']:8.3f}")
    print("\n  Paired difference, LLM minus estimator, bootstrapped over "
          f"{per_series.shape[0]} series:")
    for stat, d in diffs.items():
        verdict = "estimator better" if d["estimator_better"] else "not separated"
        print(f"    {stat.upper():6s} {d['diff']:+8.4f} "
              f"[{d['ci_lo']:+7.4f}, {d['ci_hi']:+7.4f}]  {verdict}")
    print("\n  The constant predictor is the reason ECE is reported alongside Brier and")
    print("  AURC: it attains near-zero ECE while ranking nothing, so a low ECE on its")
    print("  own is not evidence that a predictor is useful.")

    supported = bool(diffs["ece"]["estimator_better"])
    print(f"\n  H5 VERDICT: {'supported' if supported else 'not supported'} "
          f"(pre-registered criterion: 95% CI on the ECE difference excludes zero)")

    out = Path(args.out or RESULTS_DIR / "confidence_calibration.parquet")
    table.to_parquet(out)
    out.with_suffix(".json").write_text(json.dumps({
        "metric": args.metric, "model": args.model, "n": int(len(common)),
        "n_series": int(per_series.shape[0]), "base_rate": base_rate,
        "rows": table.to_dict(orient="records"), "differences": diffs,
        "h5_supported": supported,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
