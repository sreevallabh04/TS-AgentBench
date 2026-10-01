#!/usr/bin/env python
"""Run the LLM correction arms against the precomputed counterfactual tensor.

    python scripts/llm_arms.py --run latest --max-per-series 2

This is the experiment the manuscript's central critique is actually about. Every prior
system it discusses delegates the correction decision to a language model -- either
picking a repair in one pass, or deciding *whether* to repair by verbal self-critique.
Evaluating that pattern is the point, so it has to be run, not argued about.

It is cheap to run only because of the counterfactual tensor: every action's outcome is
already scored on every instance, so an LLM arm costs one decision per instance and zero
refits. The same tensor scores the non-LLM arms, so all policies are compared on
identical instances with identical outcome data, and the only thing that varies is who
decides.

Arms
----
``llm_single``          one pass over the evidence, pick an action (Self-Refine style)
``llm_critique_gate``   the same evidence, but the model first judges whether the
                        forecast needs correcting at all -- the reflection step whose
                        value is the paper's question
``rgc_llm_router``      quantitative gate decides *whether*, the model decides *which*,
                        which separates the value of gating from the value of routing
``never_correct`` / ``always_ensemble`` / ``rgc_calibrated`` are re-scored on the same
subsample so the comparison is like-for-like rather than against numbers computed on a
different instance set.

Costs are recorded per arm (calls, tokens, latency) because a claim about whether
correction is worth it is also a claim about what it costs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.calibration import GainCalibrator  # noqa: E402
from agents.policies import (  # noqa: E402
    AlwaysCorrectPolicy, CalibratedGainPolicy, CorrectionPolicy, NeverCorrectPolicy,
    PolicyDecision, ValidationEvidence,
)
from agents.reliability import ReliabilityEstimator  # noqa: E402
from benchmark.oracle import add_oracle_labels, compute_policy_regret  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from evaluation.metrics import geometric_mean  # noqa: E402
from evaluation.series_level import compare_all_to_reference  # noqa: E402

log = get_logger("llm_arms")

REFERENCE = "never_correct"

#: Same gate settings as ``scripts/headline_results.py``; selected on a held-out
#: validation origin and never on test.
GATE = {
    "mase": {"margin": 0.10, "risk_z": 0.5},
    "crps_scaled": {"margin": 0.0, "risk_z": 0.0},
}


class _AlwaysFixedAction(CorrectionPolicy):
    """Apply one named action unconditionally. The static control."""

    def __init__(self, action: str) -> None:
        self.action = action
        self.name = f"always_{action}"

    def decide_one(self, signals, evidence, rng):
        return PolicyDecision(
            instance_id=signals["instance_id"], chosen_action=self.action,
            corrected=True, compute_cost=1.0,
            rationale=f"unconditional fixed action: {self.action}",
        )


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


def series_spread_sample(
    signals: pd.DataFrame, max_per_series: int, n_cap: int, seed: int
) -> pd.DataFrame:
    """Subsample instances so that **every series is represented**.

    The API budget forces a subsample, and how it is drawn decides how much statistical
    power survives. Because the inference unit is the series (``ANALYSIS_PLAN.md`` S7),
    power comes from the number of *series* covered, not the number of instances: a draw
    of 800 instances concentrated in 200 series bootstraps over 200 units, while the same
    budget spread one-or-two-per-series covers all 578 and matches the main analysis
    exactly. We therefore cap instances per series rather than sampling instances
    independently, which also keeps the LLM arms comparable to the non-LLM arms on the
    same series set.

    Within each series, instances are drawn at random; the per-series cap keeps the draw
    balanced across families and horizons automatically, because every series contributes
    equally regardless of how many origins it happens to have.
    """
    rng = np.random.default_rng(seed)
    series_ids = sorted(signals["series_id"].unique())

    # When the budget cannot cover one instance per series, drop whole *series* rather
    # than thinning instances within them. Keeping 200 series at one origin each gives
    # 200 independent bootstrap units; keeping 578 series at a third of an origin each
    # is not a thing, and spending the same budget on fewer origins from more series
    # would leave most series unscored for some arms and scored for others.
    if n_cap and n_cap < len(series_ids):
        chosen = rng.choice(len(series_ids), size=n_cap, replace=False)
        keep = {series_ids[i] for i in sorted(chosen)}
        signals = signals[signals["series_id"].isin(keep)]
        per_series = 1
    else:
        per_series = max_per_series

    parts = []
    for _, group in signals.groupby("series_id", sort=True):
        take = min(per_series, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(idx)])
    out = pd.concat(parts)

    if n_cap and len(out) > n_cap:
        out = out.groupby("series_id", group_keys=False).head(1)
    return out.sort_index()


def few_shot_examples(
    val_signals: pd.DataFrame, val_labels: pd.DataFrame, k: int, seed: int
) -> tuple:
    """Pick k solved validation instances to show the model before it decides.

    Closing a real asymmetry: the arithmetic gate the LLM arms are measured against is
    fitted on this same validation split, while the LLM sees nothing but its prompt. A
    null result from a zero-shot model is therefore ambiguous between "cannot read the
    evidence" and "was never shown what a good decision looks like". Supplying labelled
    examples from the estimator's own training data removes the second reading.

    Two constraints shape the draw. The mix of correct and incorrect answers follows the
    validation base rate rather than being balanced, because a balanced prefix would
    itself hint at how often to intervene --- and intervention rate is the quantity under
    study. Among instances where correcting was right, examples are spread round-robin
    across distinct oracle actions, so the prefix does not quietly teach one repair.

    Examples come only from validation. Nothing from the test split enters a prompt.
    """
    if k <= 0:
        return ()
    sig = val_signals.set_index("instance_id")
    lab = val_labels.set_index("instance_id")
    lab = lab.loc[[i for i in lab.index if i in sig.index]]
    if lab.empty:
        return ()

    rng = np.random.default_rng(seed)
    rate = float(lab["should_correct"].astype(bool).mean())
    n_pos = int(np.clip(round(k * rate), 1, k - 1)) if k > 1 else 1
    pos = lab[lab["should_correct"].astype(bool)]
    neg = lab[~lab["should_correct"].astype(bool)]

    by_action = {a: list(g.index) for a, g in pos.groupby("oracle_action")}
    for ids in by_action.values():
        rng.shuffle(ids)
    order = sorted(by_action, key=lambda a: -len(by_action[a]))
    picked: list = []
    while len(picked) < min(n_pos, len(pos)) and any(by_action[a] for a in order):
        for action in order:
            if by_action[action] and len(picked) < min(n_pos, len(pos)):
                picked.append(by_action[action].pop())
    n_neg = min(k - len(picked), len(neg))
    if n_neg > 0:
        picked += list(rng.choice(neg.index.to_numpy(), size=n_neg, replace=False))
    rng.shuffle(picked)  # interleave, so position in the prefix does not encode the answer

    examples = tuple(
        (sig.loc[i], bool(lab.loc[i, "should_correct"]), str(lab.loc[i, "oracle_action"]))
        for i in picked
    )
    log.info("few-shot prefix: %d examples from validation (%d correct-to-act, "
             "actions %s)", len(examples), sum(1 for e in examples if e[1]),
             sorted({e[2] for e in examples}))
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--n", type=int, default=0,
                        help="cap on series covered; 0 = every series. Below the "
                             "series count, whole series are dropped so the "
                             "bootstrap keeps independent units")
    parser.add_argument("--max-per-series", type=int, default=2,
                        help="instances drawn per series (power scales with series)")
    parser.add_argument("--oracle-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=6000)
    parser.add_argument("--rpm", type=float, default=14.0,
                        help="requests/minute; 14 fits the Gemini free tier, raise "
                             "it when billing is enabled")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent requests; pointless above the rpm limit")
    parser.add_argument("--max-error-rate", type=float, default=0.02,
                        help="abort rather than report arms whose decisions were "
                             "silently degraded to 'accept' by provider failures")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"],
                        help="running two providers gives a cross-family replication, "
                             "which is what rules out 'it was just that model'")
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset of arms to score; use when a "
                             "token budget cannot cover them all")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="lower values stretch a token-per-day budget; Groq "
                             "appears to reserve against this, not actual usage")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"],
                        help="Groq reasoning models only; 'low' keeps the answer and "
                             "cuts reasoning tokens, which are the binding constraint")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="prepend N solved validation instances to every LLM "
                             "prompt; 0 (the default) is the zero-shot setting the "
                             "main results use")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()

    from dotenv import load_dotenv
    load_dotenv()
    key_env = "GROQ_API_KEY" if args.provider == "groq" else "GEMINI_API_KEY"
    if not os.environ.get(key_env, "").strip() and not args.cache_only:
        raise SystemExit(
            f"{key_env} is not set. This script exists to run the LLM arms; refusing "
            f"to produce a file that looks like it covers them but does not."
        )

    from agents.llm import LLMClient
    from agents.llm_agent import (
        LLMCritiqueGatePolicy, LLMSinglePolicy, RGCLLMRouterPolicy,
    )

    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")
    log.info("run %s: %d outcome rows, %d instances", run_dir.name,
             len(outcomes), len(signals))

    test = outcomes[outcomes["split"] == "test"]
    val = outcomes[outcomes["split"] == "val"]

    default_model = ("openai/gpt-oss-120b" if args.provider == "groq"
                     else os.environ.get("TSAB_LLM_MODEL", "gemini-3.5-flash-lite"))
    client = LLMClient(
        provider=args.provider,
        model=args.model or default_model,
        # Reasoning models spend most of their completion budget on the chain of
        # thought; a budget sized for the JSON answer alone returns an empty string.
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_calls=args.max_calls,
        cache_only=args.cache_only,
        requests_per_minute=args.rpm,
    )
    log.info("provider=%s model=%s keys=%d rpm=%.1f",
             client.provider, client.model, len(client._api_keys()), args.rpm)

    series_of_instance = signals.set_index("instance_id")["series_id"]
    all_rows, all_series_rows, cost_rows = [], [], []

    # Worked examples are built once, from the primary metric, and reused unchanged for
    # every metric. Rebuilding them per metric would tell the model which metric it is
    # about to be scored on --- information the zero-shot arms do not get and no deployed
    # agent has. One decision per instance, scored under both metrics, is what keeps the
    # few-shot arm exactly parallel to its zero-shot control. Empty unless --few-shot.
    shots: tuple = ()
    if args.few_shot:
        fs_labels = add_oracle_labels(val, "mase", args.oracle_delta, 0.80)
        fs_signals = signals[signals["instance_id"].isin(fs_labels["instance_id"])]
        shots = few_shot_examples(fs_signals, fs_labels, args.few_shot, args.seed)

    for metric in ("mase", "crps_scaled"):
        if metric not in outcomes.columns:
            continue

        labels = add_oracle_labels(test, metric, args.oracle_delta, 0.80)
        val_labels = add_oracle_labels(val, metric, args.oracle_delta, 0.80)
        test_signals = signals[signals["instance_id"].isin(labels["instance_id"])]
        val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]

        # The LLM arms see a stratified subsample; every arm is then scored on exactly
        # that subsample, so no arm benefits from seeing more instances than another.
        sample = series_spread_sample(
            test_signals, args.max_per_series, args.n, args.seed
        )
        log.info("[%s] scoring all arms on %d of %d test instances (%d series)",
                 metric, len(sample), len(test_signals), sample["series_id"].nunique())

        estimator = ReliabilityEstimator(random_state=args.seed).fit(
            val_signals, val_labels
        )
        gain = dict(zip(sample["instance_id"], estimator.predict_gain(sample)))
        risk = dict(zip(sample["instance_id"], estimator.predict_risk(sample)))
        gain_threshold = ReliabilityEstimator.threshold_for_rate(
            estimator.predict_gain(val_signals), 0.30
        )

        calibrator = GainCalibrator(n_estimate_origins=2).fit(val, metric)
        evidence = ValidationEvidence(outcomes, metric)
        gate = GATE.get(metric, {"margin": 0.0, "risk_z": 0.0})

        policies = [
            NeverCorrectPolicy(),
            _AlwaysFixedAction("ensemble"),
            CalibratedGainPolicy(calibrator=calibrator, use_diagnosis=False, **gate),
            AlwaysCorrectPolicy(),
            LLMSinglePolicy(client=client, n_workers=args.workers, few_shot=shots),
            LLMCritiqueGatePolicy(client=client, n_workers=args.workers,
                                  few_shot=shots),
            RGCLLMRouterPolicy(
                client=client, gain_threshold=float(gain_threshold),
                gain_scores=gain, risk_scores=risk, n_workers=args.workers,
                few_shot=shots,
            ),
        ]
        # A replication run under a tighter token budget can cover the two arms that
        # carry the argument (one-pass choice, and verbal self-critique) without the
        # router, rather than covering nothing. Dropping an arm is visible in the
        # output; silently running it on degraded data would not be.
        if args.arms:
            wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
            unknown = wanted - {p.name for p in policies}
            if unknown:
                raise SystemExit(f"unknown arm(s): {sorted(unknown)}")
            policies = [p for p in policies if p.name in wanted]
            log.info("restricted to arms: %s", [p.name for p in policies])

        frames, rates, costs = [], {}, {}
        for policy in policies:
            before = (client.n_calls, client.n_cache_hits, client.total_prompt_tokens,
                      client.total_completion_tokens, client.total_latency_s)
            t0 = time.time()
            decisions = policy.decide(sample, evidence, args.seed)
            elapsed = time.time() - t0

            # Anything that makes _parse fall back to ACCEPT biases the intervention
            # rate downward, and the intervention rate is the single quantity the
            # LLM findings rest on. There are two such paths and they are easy to
            # conflate:
            #
            #   "llm error ..."     the provider call failed (quota, 5xx, timeout)
            #   "unparseable ..."   the call succeeded but the reply was not usable,
            #                       which on a reasoning model usually means the
            #                       completion budget was spent on the chain of
            #                       thought before the JSON was emitted
            #
            # The second is not a provider failure and an earlier version of this
            # guard missed it entirely. Both are counted, and reported separately so
            # the cause is visible rather than inferred.
            if getattr(policy, "uses_llm", False):
                rationale = decisions["rationale"].astype(str)
                call_failed = rationale.str.contains("llm error", case=False)
                unparsed = rationale.str.contains("unparseable", case=False)
                degraded = float((call_failed | unparsed).mean())
                log.info("[%s] %-20s provider failures %.1f%%, unparsed replies %.1f%%",
                         metric, policy.name, 100 * call_failed.mean(),
                         100 * unparsed.mean())
                if degraded > args.max_error_rate:
                    raise SystemExit(
                        f"{policy.name}: {degraded:.1%} of decisions fell back to "
                        f"'accept' without the model choosing to "
                        f"({100 * call_failed.mean():.1f}% provider failures, "
                        f"{100 * unparsed.mean():.1f}% unparseable replies; limit "
                        f"{args.max_error_rate:.1%}). Reporting this arm would "
                        f"understate how often the model chooses to correct. If the "
                        f"unparsed share is high, raise --max-output-tokens: a "
                        f"reasoning model needs room for its trace *and* the answer. "
                        f"Cached responses are reused, so a rerun resumes."
                    )

            scored = compute_policy_regret(
                test, decisions.set_index("instance_id")["chosen_action"], labels, metric
            )
            scored = scored[scored["instance_id"].isin(sample["instance_id"])]
            scored["policy"] = policy.name
            # `regret` and `should_correct` carry the decision-quality evidence for H7,
            # whose pre-registered test is the quantitative gate against an LLM
            # critique gate -- a comparison that was unavailable until these arms ran.
            keep = ["instance_id", "policy", "chosen_metric", "corrected",
                    "improved", "harmed"]
            keep += [c for c in ("regret", "should_correct") if c in scored.columns]
            frames.append(scored[keep])
            rates[policy.name] = float(decisions["corrected"].mean())
            costs[policy.name] = {
                "metric": metric,
                "policy": policy.name,
                "llm_calls": client.n_calls - before[0],
                "cache_hits": client.n_cache_hits - before[1],
                "prompt_tokens": client.total_prompt_tokens - before[2],
                "completion_tokens": client.total_completion_tokens - before[3],
                "wall_seconds": elapsed,
                "correction_rate": rates[policy.name],
            }
            log.info("[%s] %-20s acts=%4.0f%%  calls=%d  hits=%d  %.1fs",
                     metric, policy.name, 100 * rates[policy.name],
                     costs[policy.name]["llm_calls"],
                     costs[policy.name]["cache_hits"], elapsed)

        cost_rows.extend(costs.values())
        scores = pd.concat(frames, ignore_index=True)
        wide = scores.pivot_table(index="instance_id", columns="policy",
                                  values="chosen_metric")

        # --- instance-level summary (descriptive) ---
        reference = wide[REFERENCE]
        ref_mean, ref_geo = float(reference.mean()), geometric_mean(reference)
        for policy in wide.columns:
            a = wide[policy].dropna()
            rows_for_policy = scores[scores["policy"] == policy]
            acted = rows_for_policy[rows_for_policy["corrected"]]

            # Decision-quality terms: H7 asks which mechanism *decides* better, which
            # is a question about precision and regret, not about mean error.
            decision: dict[str, float] = {}
            if "should_correct" in rows_for_policy.columns:
                corrected = rows_for_policy["corrected"].astype(bool)
                warranted = rows_for_policy["should_correct"].astype(bool)
                tp = int((corrected & warranted).sum())
                fp = int((corrected & ~warranted).sum())
                fn = int((~corrected & warranted).sum())
                precision = tp / (tp + fp) if (tp + fp) else np.nan
                recall = tp / (tp + fn) if (tp + fn) else np.nan
                decision = {
                    "decision_precision": precision,
                    "decision_recall": recall,
                    "decision_f1": (2 * precision * recall / (precision + recall)
                                    if np.isfinite(precision) and np.isfinite(recall)
                                    and (precision + recall) else np.nan),
                }
            if "regret" in rows_for_policy.columns:
                decision["regret"] = float(rows_for_policy["regret"].mean())

            all_rows.append({
                "metric": metric,
                "policy": policy,
                "mean": float(a.mean()),
                "mean_pct": 100.0 * (a.mean() / ref_mean - 1.0),
                "geometric_mean": geometric_mean(a),
                "geo_pct": 100.0 * (geometric_mean(a) / ref_geo - 1.0),
                "median": float(a.median()),
                "correction_rate": rates[policy],
                "helped": float(acted["improved"].mean()) if len(acted) else np.nan,
                "harmed": float(acted["harmed"].mean()) if len(acted) else np.nan,
                "n": int(len(a)),
                **decision,
            })

        # --- series-level inference (confirmatory; see ANALYSIS_PLAN.md S7) ---
        series_table = compare_all_to_reference(
            wide, series_of_instance, REFERENCE, n_boot=10_000, seed=args.seed
        )
        series_table.insert(0, "metric", metric)
        all_series_rows.append(series_table)

    instance_table = pd.DataFrame(all_rows)
    series_table = pd.concat(all_series_rows, ignore_index=True)
    cost_table = pd.DataFrame(cost_rows)

    order = [REFERENCE, "always_ensemble", "rgc_calibrated", "llm_single",
             "llm_critique_gate", "rgc_llm_router", "always_correct"]

    for metric in instance_table["metric"].unique():
        sub = series_table[series_table["metric"] == metric].set_index("policy")
        sub = sub.reindex([p for p in order if p in sub.index])
        ins = instance_table[instance_table["metric"] == metric].set_index("policy")
        print(f"\n{'=' * 104}\n{metric.upper()}   "
              f"n={int(ins['n'].iloc[0])} instances / {int(sub['n_series'].iloc[0])} series"
              f"\n{'=' * 104}")
        print(f"{'policy':20s} {'series mean':>11} {'delta %':>9} "
              f"{'95% CI (%)':>20} {'win rate':>9} {'acts':>6} {'p':>10}")
        for policy, r in sub.iterrows():
            flag = "*" if r["significant"] else " "
            print(f"{policy:20s} {r['series_mean']:11.4f} {r['pct_diff']:+8.2f}%{flag}"
                  f"  [{r['pct_ci_lo']:+7.2f}, {r['pct_ci_hi']:+7.2f}] "
                  f"{r['win_rate']:8.1%} "
                  f"{ins.loc[policy, 'correction_rate']:6.0%} "
                  f"{r['wilcoxon_p']:10.2g}")
        print("  * = series-level bootstrap CI excludes zero")

    print(f"\n{'=' * 104}\nLLM COST\n{'=' * 104}")
    print(cost_table.groupby("policy")[
        ["llm_calls", "cache_hits", "prompt_tokens", "completion_tokens", "wall_seconds"]
    ].sum().to_string())
    print(f"\ntotal live calls this session: {client.n_calls}, "
          f"cache hits: {client.n_cache_hits}, errors: {client.n_errors}, "
          f"tokens: {client.total_prompt_tokens + client.total_completion_tokens}")

    out = Path(args.out or RESULTS_DIR / "llm_arms.parquet")
    instance_table.to_parquet(out)
    # Sibling files are named from `out`'s stem, not hardcoded: a smoke run given
    # --out must not overwrite the real artifacts that the manuscript reads.
    series_table.to_parquet(out.with_name(f"{out.stem}_series.parquet"))
    cost_table.to_parquet(out.with_name(f"{out.stem}_cost.parquet"))
    out.with_name(f"{out.stem}_meta.json").write_text(
        json.dumps({
            "run": run_dir.name,
            "provider": client.provider,
            "model": client.model,
            "n_subsample": args.n,
            "seed": args.seed,
            "live_calls": client.n_calls,
            "cache_hits": client.n_cache_hits,
            "errors": client.n_errors,
            "prompt_tokens": client.total_prompt_tokens,
            "completion_tokens": client.total_completion_tokens,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
