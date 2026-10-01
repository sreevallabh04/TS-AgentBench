#!/usr/bin/env python
"""Does verbal self-critique carry any information about whether to correct?

    python scripts/critique_discrimination.py --run latest

The headline observation in the LLM section is that a self-critique gate intervenes at
almost exactly the rate an uncritiqued single pass intervenes at (about 70% either way).
That is *consistent* with the critique being inert, but it does not establish it: two
arms can share an intervention rate and still be selecting very different instances, one
of them well.

This script runs the decisive test instead of the suggestive one. It asks whether the
critique's yes/no **discriminates** --- whether the instances it flags are actually the
instances where correction pays:

* **AUROC of the critique decision against the ground-truth ``should_correct`` label.**
  0.5 means the critique carries literally no information about whether correction was
  warranted. This is the number the claim should rest on.
* **Agreement with the uncritiqued arm.** If the critique step changes nothing about
  *which* instances get corrected, not merely how many, it is inert in the strong sense.
* **Conditional outcome rates.** Among instances the critique approved, how often did
  correcting actually help versus harm --- and is that any better than the unconditioned
  intervention rate? If not, the critique adds no value over deciding at random at the
  same rate.
* **A rate-matched random control**, which is the right null: any gate that fires 70% of
  the time inherits some of the base rate, so the comparison must be against a coin that
  fires 70% of the time, not against never correcting.

Everything is computed from cached responses, so this makes no API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.calibration import GainCalibrator  # noqa: E402
from agents.policies import ValidationEvidence  # noqa: E402
from benchmark.oracle import add_oracle_labels, compute_policy_regret  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402

log = get_logger("critique_discrimination")


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


def main(argv: list[str] | None = None) -> int:
    from sklearn.metrics import roc_auc_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--metric", default="mase")
    parser.add_argument("--oracle-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=0,
                        help="series cap; must match the llm_arms run being analysed")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="must match the llm_arms run: it is part of the cache key")
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

    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")
    test = outcomes[outcomes["split"] == "test"]

    labels = add_oracle_labels(test, args.metric, args.oracle_delta, 0.80)
    test_signals = signals[signals["instance_id"].isin(labels["instance_id"])]

    # The same one-instance-per-series draw the LLM arms used, so the cache hits.
    from scripts.llm_arms import series_spread_sample  # noqa: E402
    sample = series_spread_sample(test_signals, 1, args.n, args.seed)

    # cache_only: this analysis must never spend budget, and a cache miss here means
    # the arms were run on a different sample, which would silently compare the wrong
    # instances.
    client = LLMClient(provider=args.provider, model=args.model, cache_only=True,
                       max_output_tokens=args.max_output_tokens,
                       reasoning_effort=args.reasoning_effort,
                       requests_per_minute=0)
    evidence = ValidationEvidence(outcomes, args.metric)

    single = LLMSinglePolicy(client=client, n_workers=1).decide(
        sample, evidence, args.seed)
    critique = LLMCritiqueGatePolicy(client=client, n_workers=1).decide(
        sample, evidence, args.seed)

    truth = labels.set_index("instance_id")["should_correct"].astype(int)
    single = single.set_index("instance_id")
    critique = critique.set_index("instance_id")
    common = single.index.intersection(critique.index).intersection(truth.index)
    single, critique, truth = single.loc[common], critique.loc[common], truth.loc[common]

    c_act = critique["corrected"].astype(bool)
    s_act = single["corrected"].astype(bool)

    # --- 1. Does the critique discriminate at all? ---
    auroc = float(roc_auc_score(truth, c_act.astype(int))) if truth.nunique() > 1 else np.nan
    auroc_single = float(roc_auc_score(truth, s_act.astype(int))) if truth.nunique() > 1 else np.nan

    # --- 2. Does it change *which* instances, not just how many? ---
    agreement = float((c_act == s_act).mean())
    both = float((c_act & s_act).mean())
    jaccard = float((c_act & s_act).sum() / max((c_act | s_act).sum(), 1))

    # --- 3. Conditional outcomes, and a rate-matched random control ---
    scored_c = compute_policy_regret(
        test, critique["chosen_action"], labels, args.metric
    ).set_index("instance_id").reindex(common)
    scored_s = compute_policy_regret(
        test, single["chosen_action"], labels, args.metric
    ).set_index("instance_id").reindex(common)

    rng = np.random.default_rng(args.seed)
    rate = float(c_act.mean())
    random_act = pd.Series(rng.random(len(common)) < rate, index=common)

    def outcome_block(name, acted, scored, outcomes_comparable=True):
        """Summarise one arm's selection.

        ``outcomes_comparable`` is false for the random control: it is evaluated against
        the critique arm's action assignments, so on instances the critique declined the
        recorded action is ``accept`` and the helped/harmed rates are diluted by rows
        where nothing was done. Only its *selection precision* is a like-for-like
        comparison, so the other columns are withheld rather than printed misleadingly.
        """
        acted = acted.reindex(common).fillna(False)
        sub = scored[acted]
        return {
            "arm": name,
            "act_rate": float(acted.mean()),
            "precision_vs_should_correct": float(truth[acted].mean())
            if acted.any() else np.nan,
            "helped": float(sub["improved"].mean())
            if outcomes_comparable and len(sub) else np.nan,
            "harmed": float(sub["harmed"].mean())
            if outcomes_comparable and len(sub) else np.nan,
        }

    # --- 4. The matched contrast: arithmetic signal vs. language model, same label ---
    #
    # The triage AUROC reported elsewhere (0.79) targets `large_error`, not
    # `should_correct`, so quoting it against the critique's 0.53 would compare two
    # different questions. Scoring the reliability estimator against `should_correct` on
    # these exact instances makes the contrast like-for-like: one arithmetic signal and
    # one language model, judging the identical decision on the identical data.
    arithmetic_auroc = float("nan")
    try:
        from agents.reliability import ReliabilityEstimator
        val = outcomes[outcomes["split"] == "val"]
        val_labels = add_oracle_labels(val, args.metric, args.oracle_delta, 0.80)
        val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]
        estimator = ReliabilityEstimator(random_state=args.seed).fit(
            val_signals, val_labels)
        sample_in_common = sample[sample["instance_id"].isin(common)]
        gain_hat = pd.Series(estimator.predict_gain(sample_in_common),
                             index=sample_in_common["instance_id"].values)
        gain_hat = gain_hat.reindex(common).dropna()
        if gain_hat.size > 20 and truth.loc[gain_hat.index].nunique() > 1:
            arithmetic_auroc = float(
                roc_auc_score(truth.loc[gain_hat.index], gain_hat.values))
    except Exception as exc:  # noqa: BLE001 - a missing comparator must not kill the run
        log.warning("arithmetic comparator unavailable: %s", exc)

    # --- 5. The same contrast at matched footing --------------------------------
    #
    # `auroc` above scores a binary yes/no. For a binary predictor roc_auc_score is
    # exactly balanced accuracy, so comparing it against a continuous score credits the
    # estimator for ranking that a yes/no cannot express, and part of the gap is an
    # artefact of the output type rather than of the judgement. Two matched comparisons
    # remove that artefact, and both are reported.
    def balanced_accuracy(pred: pd.Series, y: pd.Series) -> float:
        pred = pred.astype(bool)
        y = y.astype(bool)
        tpr = float(pred[y].mean()) if y.any() else np.nan
        tnr = float((~pred[~y]).mean()) if (~y).any() else np.nan
        return float((tpr + tnr) / 2.0)

    truth_bool = truth.astype(bool)
    auroc_llm_confidence = float("nan")
    balacc_critique = balanced_accuracy(c_act, truth_bool)
    balacc_arithmetic_matched = float("nan")
    try:
        # (a) The model's own confidence is continuous, so scoring it gives the language
        # model the same kind of credit the estimator gets. Confidence is its belief that
        # the forecast is reliable, so 1 - confidence is its implied vote for correcting.
        conf = critique["confidence"].astype(float)
        conf = conf.reindex(common).dropna()
        if conf.size > 20 and truth.loc[conf.index].nunique() > 1:
            auroc_llm_confidence = float(
                roc_auc_score(truth.loc[conf.index], 1.0 - conf.to_numpy()))

        # (b) The estimator forced to the LLM's own operating point: same fraction of
        # instances flagged, compared on the same statistic.
        if np.isfinite(arithmetic_auroc) and gain_hat.size:
            rate = float(c_act.mean())
            cutoff = float(np.quantile(gain_hat.to_numpy(), 1.0 - rate))
            matched = (gain_hat >= cutoff).reindex(common).fillna(False)
            balacc_arithmetic_matched = balanced_accuracy(
                matched, truth_bool.loc[matched.index])
    except Exception as exc:  # noqa: BLE001 - a missing comparator must not kill the run
        log.warning("matched comparison unavailable: %s", exc)

    base_rate = float(truth.mean())
    rows = [
        outcome_block("llm_critique_gate (critique said yes)", c_act, scored_c),
        outcome_block("llm_single (no critique)", s_act, scored_s),
    ]
    # The random control is scored against the critique arm's own chosen actions, so
    # only *which instances* differ, not what was done to them.
    rows.append(outcome_block(f"random at the same {rate:.0%} rate",
                              random_act, scored_c, outcomes_comparable=False))
    table = pd.DataFrame(rows)

    print(f"\n{'=' * 88}")
    print(f"DOES VERBAL SELF-CRITIQUE DISCRIMINATE?   metric={args.metric}  "
          f"model={args.model}  n={len(common)}")
    print(f"{'=' * 88}")
    print(f"Base rate of instances where correction was warranted: {base_rate:.1%}\n")
    print(f"  AUROC, critique decision vs. should_correct : {auroc:.4f}")
    print(f"  AUROC, single-pass decision vs should_correct: {auroc_single:.4f}")
    print(f"  AUROC, arithmetic gain estimator (same label, same instances): "
          f"{arithmetic_auroc:.4f}")
    print(f"  (0.500 = the decision carries no information about whether to correct)\n")
    print("  MATCHED FOOTING. The three numbers above are not all the same kind of")
    print("  quantity: a binary yes/no scored as AUROC is balanced accuracy, while the")
    print("  estimator is a continuous score and is credited for ranking. Both matched")
    print("  comparisons are therefore reported.")
    print(f"    (a) continuous vs continuous, AUROC")
    print(f"        LLM verbalised confidence : {auroc_llm_confidence:.4f}")
    print(f"        arithmetic estimator      : {arithmetic_auroc:.4f}")
    print(f"    (b) same operating point, balanced accuracy at the LLM's own rate")
    print(f"        LLM critique decision     : {balacc_critique:.4f}")
    print(f"        arithmetic estimator      : {balacc_arithmetic_matched:.4f}\n")
    print(f"  Agreement between the two arms on which instances to act on: {agreement:.1%}")
    print(f"  Jaccard overlap of the two intervention sets:                {jaccard:.1%}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    lift = auroc - 0.5
    # The verdict is stated in terms a reader can check against the table rather than
    # as a bare pass/fail, because "is 0.53 meaningfully above 0.50" is a judgement the
    # reader is entitled to make themselves.
    lift = auroc - 0.5
    precision_critique = float(table.iloc[0]["precision_vs_should_correct"])
    precision_random = float(table.iloc[-1]["precision_vs_should_correct"])
    verdict = (
        f"AUROC {auroc:.3f} ({lift:+.3f} vs chance); selection precision "
        f"{precision_critique:.1%} vs {precision_random:.1%} for a coin firing at the "
        f"same rate and a {base_rate:.1%} base rate; {jaccard:.0%} of its intervention "
        f"set is shared with the uncritiqued arm"
    )
    print(f"\n  VERDICT: {verdict}")

    out = Path(args.out or RESULTS_DIR / "critique_discrimination.parquet")
    table.to_parquet(out)
    out.with_suffix(".json").write_text(json.dumps({
        "metric": args.metric, "model": args.model, "n": int(len(common)),
        "base_rate": base_rate, "auroc_critique": auroc,
        "auroc_single": auroc_single, "auroc_arithmetic": arithmetic_auroc,
        "auroc_llm_confidence": auroc_llm_confidence,
        "balacc_critique": balacc_critique,
        "balacc_arithmetic_matched": balacc_arithmetic_matched,
        "agreement": agreement, "jaccard": jaccard,
        "rows": table.to_dict(orient="records"), "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
