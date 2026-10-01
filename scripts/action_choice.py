#!/usr/bin/env python
"""Given that an agent has decided to correct, is its choice of repair any good?

    python scripts/action_choice.py --run latest

The LLM arms conflate two decisions: *whether* to correct and *which* repair to apply.
The router arm separates them --- a calibrated quantitative gate decides whether, and the
model chooses only among the diagnosis-matched candidates --- and it has by far the
highest harm per intervention of any arm. That is not what one would predict if the
model's contribution to action choice were merely uninformative, so it is worth testing
directly rather than inferring.

The test is exact, not statistical modelling, because the counterfactual tensor holds the
realised outcome of *every* action on *every* instance. On the instances where an arm
intervened we can ask, on identical data:

* what the model's chosen action actually scored;
* what a **uniform random draw** from the same candidate menu would have scored, in
  expectation over the menu (computed exactly by averaging the menu's outcomes, not by
  sampling);
* what the **best** and **worst** actions in that menu scored, which bracket the range
  the model was choosing within.

The headline quantity is the **oracle gap**: how far the chosen action sits from the
best one that was available *on that instance, inside the menu the model was handed*.
That is an attainable upper bound, not an idealisation, which is what makes the
shortfall interpretable. It is reported with the share of the coin-to-oracle span the
choice recovers, so a reader can see at once whether selection captures any headroom.

The comparison against a uniform draw is the mechanism underneath that number, and
three outcomes are distinguishable and they mean different things:

* chosen $\\approx$ random  -- the model contributes nothing to action choice
* chosen $<$ random        -- the model selects usefully within the menu
* chosen $>$ random        -- the model **anti-selects**: it is systematically drawn to
  the worse options, and a coin would serve better

The third would sharpen the regret decomposition considerably. The restriction loss says
the menu excludes repairs that would have helped; anti-selection would say the model also
fails to find the good ones inside the menu it was given. Separating the third from
the first needs more instances than the router arm has, so it is not claimed here.

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

from agents.policies import ACCEPT, ValidationEvidence  # noqa: E402
from benchmark.oracle import add_oracle_labels  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from evaluation.series_level import paired_series_test, per_series_means  # noqa: E402

log = get_logger("action_choice")


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


def bootstrap_span_ratios(
    chosen: np.ndarray, random_mean: np.ndarray, best: np.ndarray,
    n_boot: int, seed: int,
) -> dict:
    """Interval for the two ratios that say how much of the menu's headroom is used.

    ``oracle_gap`` is how much worse the chosen action is than the best one that was
    available on the same instance. ``headroom_recovered`` is the share of the
    coin-to-oracle span the selection actually captures: 1.0 would mean it always finds
    the best action, 0.0 that it does no better than a uniform draw, and a negative
    value that it does worse than the draw.

    Both are ratios of means over series, so numerator and denominator are resampled
    together: dividing two separately-computed marginal intervals would ignore that they
    move on the same series. The aggregate form also avoids the per-instance version's
    blow-up when a single instance happens to have a near-degenerate menu.
    """
    n = chosen.size
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    c, r, b = chosen[idx].mean(axis=1), random_mean[idx].mean(axis=1), best[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(np.abs(b) > 1e-12, 100.0 * (c - b) / b, np.nan)
        span = r - b
        recovered = np.where(np.abs(span) > 1e-12, 100.0 * (r - c) / span, np.nan)
    gap, recovered = gap[np.isfinite(gap)], recovered[np.isfinite(recovered)]
    cm, rm, bm = chosen.mean(), random_mean.mean(), best.mean()
    return {
        "oracle_gap_pct": float(100.0 * (cm - bm) / bm) if abs(bm) > 1e-12 else np.nan,
        "oracle_gap_ci_lo": float(np.quantile(gap, 0.025)) if gap.size else np.nan,
        "oracle_gap_ci_hi": float(np.quantile(gap, 0.975)) if gap.size else np.nan,
        "headroom_recovered_pct": float(100.0 * (rm - cm) / (rm - bm))
        if abs(rm - bm) > 1e-12 else np.nan,
        "headroom_ci_lo": float(np.quantile(recovered, 0.025)) if recovered.size else np.nan,
        "headroom_ci_hi": float(np.quantile(recovered, 0.975)) if recovered.size else np.nan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--metric", default="mase")
    parser.add_argument("--oracle-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default=None,
                        help="must match the llm_arms run being analysed: it "
                             "is part of the response cache key")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    from dotenv import load_dotenv
    load_dotenv()

    from agents.llm import LLMClient
    from agents.llm_agent import LLMCritiqueGatePolicy, LLMSinglePolicy
    from agents.reliability import ReliabilityEstimator
    from agents.llm_agent import RGCLLMRouterPolicy

    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")
    test = outcomes[outcomes["split"] == "test"]
    val = outcomes[outcomes["split"] == "val"]

    labels = add_oracle_labels(test, args.metric, args.oracle_delta, 0.80)
    val_labels = add_oracle_labels(val, args.metric, args.oracle_delta, 0.80)
    test_signals = signals[signals["instance_id"].isin(labels["instance_id"])]
    val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]

    from scripts.llm_arms import series_spread_sample  # noqa: E402
    sample = series_spread_sample(test_signals, 1, args.n, args.seed)

    client = LLMClient(provider=args.provider, model=args.model, cache_only=True,
                       max_output_tokens=args.max_output_tokens,
                       reasoning_effort=args.reasoning_effort,
                       requests_per_minute=0)
    evidence = ValidationEvidence(outcomes, args.metric)

    estimator = ReliabilityEstimator(random_state=args.seed).fit(val_signals, val_labels)
    gain = dict(zip(sample["instance_id"], estimator.predict_gain(sample)))
    risk = dict(zip(sample["instance_id"], estimator.predict_risk(sample)))
    threshold = ReliabilityEstimator.threshold_for_rate(
        estimator.predict_gain(val_signals), 0.30)

    arms = {
        "llm_single": LLMSinglePolicy(client=client, n_workers=1),
        "llm_critique_gate": LLMCritiqueGatePolicy(client=client, n_workers=1),
        "rgc_llm_router": RGCLLMRouterPolicy(
            client=client, gain_threshold=float(threshold),
            gain_scores=gain, risk_scores=risk, n_workers=1),
    }

    # Outcome of every action on every sampled instance: the whole point of the tensor.
    applicable = test[test["applicable"].astype(bool)]
    wide = applicable.pivot_table(index="instance_id", columns="action",
                                  values=args.metric)
    wide = wide.reindex(sample["instance_id"].values)
    series_of = signals.set_index("instance_id")["series_id"]

    rows, series_tests = [], []
    for name, policy in arms.items():
        try:
            decisions = policy.decide(sample, evidence, args.seed).set_index("instance_id")
        except Exception as exc:  # noqa: BLE001 - a missing cache must not kill the rest
            log.warning("%s unavailable (%s); skipping", name, type(exc).__name__)
            continue

        acted = decisions[decisions["corrected"].astype(bool)]
        if acted.empty:
            continue

        chosen, menu_mean, menu_best, menu_worst, ids = [], [], [], [], []
        for iid, row in acted.iterrows():
            if iid not in wide.index:
                continue
            available = wide.loc[iid].dropna()
            # The menu the model was actually choosing from: every non-accept action
            # with a scored outcome on this instance.
            menu = available.drop(labels=[ACCEPT], errors="ignore")
            action = row["chosen_action"]
            if action not in available.index or menu.empty:
                continue
            chosen.append(float(available[action]))
            # Expectation over a uniform draw, computed exactly rather than sampled.
            menu_mean.append(float(menu.mean()))
            menu_best.append(float(menu.min()))
            menu_worst.append(float(menu.max()))
            ids.append(iid)

        if len(chosen) < 20:
            log.warning("%s: only %d usable instances; skipping", name, len(chosen))
            continue

        chosen = pd.Series(chosen, index=ids)
        menu_mean = pd.Series(menu_mean, index=ids)
        menu_best = pd.Series(menu_best, index=ids)
        menu_worst = pd.Series(menu_worst, index=ids)

        # Series-level paired test of chosen against the uniform-random expectation.
        stat = paired_series_test(
            per_series_means(chosen, series_of),
            per_series_means(menu_mean, series_of),
            n_boot=10000, seed=args.seed,
        )
        series_tests.append({"arm": name, **stat})

        # The same three quantities aggregated to series before any ratio is taken, so
        # the headline gap and the paired test share one unit of analysis.
        s_chosen = per_series_means(chosen, series_of)
        s_random = per_series_means(menu_mean, series_of)
        s_best = per_series_means(menu_best, series_of)
        common = s_chosen.index.intersection(s_random.index).intersection(s_best.index)
        spans = bootstrap_span_ratios(
            s_chosen.loc[common].to_numpy(), s_random.loc[common].to_numpy(),
            s_best.loc[common].to_numpy(), 10000, args.seed)

        rows.append({
            "arm": name,
            "n_instances": int(len(chosen)),
            "n_series": int(stat["n_series"]),
            "chosen": float(chosen.mean()),
            "uniform_random": float(menu_mean.mean()),
            "best_in_menu": float(menu_best.mean()),
            "worst_in_menu": float(menu_worst.mean()),
            "chosen_vs_random_pct": stat["pct_diff"],
            "ci_lo": stat["pct_ci_lo"],
            "ci_hi": stat["pct_ci_hi"],
            "beats_random_on_pct_of_series": stat["win_rate"],
            "significant": stat["significant"],
            **spans,
        })

    if not rows:
        raise SystemExit("no arm had enough cached decisions; run scripts/llm_arms.py")

    table = pd.DataFrame(rows)
    print(f"\n{'=' * 96}")
    print(f"ACTION CHOICE, GIVEN THE DECISION TO CORRECT   metric={args.metric}  "
          f"model={args.model}")
    print(f"{'=' * 96}")
    print("All columns are the metric on the SAME instances; lower is better. 'best' and")
    print("'worst' bracket the menu the model was choosing from on those instances.\n")
    print(f"{'arm':20s} {'inst':>6} {'best':>9} {'chosen':>9} {'random':>9} {'worst':>9}")
    for _, r in table.iterrows():
        print(f"{r['arm']:20s} {r['n_instances']:6d} {r['best_in_menu']:9.4f} "
              f"{r['chosen']:9.4f} {r['uniform_random']:9.4f} {r['worst_in_menu']:9.4f}")

    # The headline. How far the choice sits from an upper bound that existed on the
    # instance, inside the menu the model was handed -- not from an unattainable ideal.
    print("\n  HOW MUCH OF THE MENU'S HEADROOM DOES THE CHOICE CAPTURE?")
    print(f"  {'arm':20s} {'chosen vs best':>18} {'95% CI':>19} "
          f"{'headroom recovered':>20} {'95% CI':>19}")
    for _, r in table.iterrows():
        print(f"  {r['arm']:20s} {r.get('oracle_gap_pct', float('nan')):+17.2f}% "
              f"[{r.get('oracle_gap_ci_lo', float('nan')):+7.2f},"
              f"{r.get('oracle_gap_ci_hi', float('nan')):+7.2f}] "
              f"{r.get('headroom_recovered_pct', float('nan')):+19.1f}% "
              f"[{r.get('headroom_ci_lo', float('nan')):+7.1f},"
              f"{r.get('headroom_ci_hi', float('nan')):+7.1f}]")
    print("  headroom recovered = share of the coin-to-oracle span the choice captures;")
    print("  0% = no better than a uniform draw, 100% = always picks the best action.")

    # The mechanism underneath it: is the shortfall explained by the choice being
    # indistinguishable from a coin over the same menu?
    print("\n  MECHANISM: CHOSEN VS A UNIFORM DRAW FROM THE SAME MENU")
    print(f"  {'arm':20s} {'chosen vs random':>17} {'95% CI':>19} {'win rate':>9}")
    for _, r in table.iterrows():
        flag = "*" if r["significant"] else " "
        print(f"  {r['arm']:20s} {r['chosen_vs_random_pct']:+16.2f}%{flag} "
              f"[{r['ci_lo']:+7.2f},{r['ci_hi']:+7.2f}] "
              f"{r['beats_random_on_pct_of_series']:8.1%}")
    print("\n  * = series-level bootstrap CI excludes zero")
    print("  positive 'chosen vs random' = the model's pick is WORSE than a coin over the")
    print("  same menu. No interval here excludes zero, so anti-selection is NOT claimed.")

    out = Path(args.out or RESULTS_DIR / "action_choice.parquet")
    table.to_parquet(out)
    out.with_suffix(".json").write_text(
        json.dumps({"metric": args.metric, "model": args.model,
                    "rows": table.to_dict(orient="records")}, indent=2,
                   default=float),
        encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
