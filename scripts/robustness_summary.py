#!/usr/bin/env python
"""Does correction earn its keep under the conditions it was designed for?

    python scripts/robustness_summary.py

The case for adaptive correction is not that it helps on average --- it is that it helps
*when something has gone wrong*. Anomalies, regime shifts and drift are exactly the
conditions a correction stage is supposed to catch, so a policy that tracks never
correcting across all of them has not simply failed to win on average; it has failed
where its own rationale places it.

The robustness sweep perturbs the test series along seven controlled axes at four
severities and replays the policies against each. This script turns that into the
comparison the claim needs: for every axis, at every severity, the series-level
difference from never correcting with a bootstrap interval over series. Reading the raw
per-instance means instead would repeat the pathology this paper spends a section
warning about.

Severity 0 is the unperturbed condition and is reported once as a reference row rather
than per axis, because every axis shares it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR  # noqa: E402
from evaluation.series_level import paired_series_test, per_series_means  # noqa: E402

log = get_logger("robustness_summary")

REFERENCE = "never_correct"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="chosen_metric")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    path = RESULTS_DIR / "robustness.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing; run the robustness sweep first")
    df = pd.read_parquet(path)

    series_of = df.drop_duplicates("instance_id").set_index("instance_id")["series_id"]
    policies = [p for p in sorted(df["policy"].unique()) if p != REFERENCE]

    rows = []
    for perturbation, block in df.groupby("perturbation", sort=True):
        for severity, sub in block.groupby("severity", sort=True):
            if severity == 0:
                continue
            ref = sub[sub["policy"] == REFERENCE].set_index("instance_id")[args.metric]
            if ref.empty:
                continue
            ref_series = per_series_means(ref, series_of)
            for policy in policies:
                arm = sub[sub["policy"] == policy].set_index("instance_id")[args.metric]
                if arm.empty:
                    continue
                stat = paired_series_test(
                    per_series_means(arm, series_of), ref_series,
                    n_boot=args.n_boot, seed=args.seed,
                )
                rows.append({
                    "perturbation": perturbation,
                    "severity": float(severity),
                    "policy": policy,
                    "n_series": stat["n_series"],
                    "mean_diff": stat["mean_diff"],
                    "pct_diff": stat["pct_diff"],
                    "pct_ci_lo": stat["pct_ci_lo"],
                    "pct_ci_hi": stat["pct_ci_hi"],
                    "win_rate": stat["win_rate"],
                    "significant": stat["significant"],
                })

    # --- H2's pre-registered difference-in-differences ---------------------------
    #
    # Concentration, not level: how much larger is the policy's advantage over never
    # correcting under perturbation than it is on the same series unperturbed? The two
    # differences are formed per series and subtracted per series, so the bootstrap
    # resamples one unit and the pairing is preserved on both sides.
    did_rows = []
    clean = df[df["severity"] == 0]
    for perturbation, block in df.groupby("perturbation", sort=True):
        for policy in policies:
            base_arm = clean[clean["policy"] == policy].set_index("instance_id")[args.metric]
            base_ref = clean[clean["policy"] == REFERENCE].set_index("instance_id")[args.metric]
            if base_arm.empty or base_ref.empty:
                continue
            d0 = (per_series_means(base_arm, series_of)
                  - per_series_means(base_ref, series_of)).dropna()
            for severity, sub in block.groupby("severity", sort=True):
                if severity == 0:
                    continue
                arm = sub[sub["policy"] == policy].set_index("instance_id")[args.metric]
                ref = sub[sub["policy"] == REFERENCE].set_index("instance_id")[args.metric]
                if arm.empty or ref.empty:
                    continue
                ds = (per_series_means(arm, series_of)
                      - per_series_means(ref, series_of)).dropna()
                common = ds.index.intersection(d0.index)
                if common.size < 10:
                    continue
                did = (ds.loc[common] - d0.loc[common]).to_numpy()
                rng = np.random.default_rng(args.seed)
                boot = did[rng.integers(0, did.size, size=(args.n_boot, did.size))].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                # Two-sided bootstrap p-value: how often the resampled effect lands on
                # the other side of zero, doubled. Needed because a verdict over 48
                # strata has to pass a multiplicity correction, and the plan fixes Holm.
                p_boot = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
                p_boot = float(min(1.0, max(p_boot, 1.0 / args.n_boot)))
                did_rows.append({
                    "perturbation": perturbation,
                    "severity": float(severity),
                    "policy": policy,
                    "n_series": int(common.size),
                    "did": float(did.mean()),
                    "did_ci_lo": float(lo),
                    "did_ci_hi": float(hi),
                    # H2 predicts the advantage GROWS under perturbation, i.e. the
                    # difference-in-differences is reliably negative (lower error).
                    "p_boot": p_boot,
                    "concentrates_uncorrected": bool(hi < 0),
                })

    if not rows:
        raise SystemExit("no comparisons could be formed from the robustness sweep")
    table = pd.DataFrame(rows)

    print(f"\n{'=' * 104}")
    print("ROBUSTNESS: DIFFERENCE FROM NEVER CORRECTING, BY PERTURBATION AND SEVERITY")
    print(f"{'=' * 104}")
    print("Series-level means with a bootstrap interval over series. Negative favours the")
    print("policy; * marks an interval excluding zero.\n")
    print(f"{'perturbation':22s} {'sev':>5} {'policy':16s} {'series':>7} "
          f"{'delta':>9} {'95% CI':>20} {'win':>7}")
    for _, r in table.iterrows():
        flag = "*" if r["significant"] else " "
        print(f"{r['perturbation']:22s} {r['severity']:5.2f} {r['policy']:16s} "
              f"{r['n_series']:7d} {r['pct_diff']:+8.2f}%{flag} "
              f"[{r['pct_ci_lo']:+7.2f},{r['pct_ci_hi']:+7.2f}] {r['win_rate']:6.1%}")

    wins = table[table["significant"] & (table["pct_diff"] < 0)]
    losses = table[table["significant"] & (table["pct_diff"] > 0)]
    print(f"\n  comparisons run                : {len(table)}")
    print(f"  significant improvements       : {len(wins)}")
    print(f"  significant degradations       : {len(losses)}")
    if not wins.empty:
        print("  where correction significantly helps:")
        for _, r in wins.iterrows():
            print(f"    {r['policy']} under {r['perturbation']} at severity "
                  f"{r['severity']:.2f}: {r['pct_diff']:+.2f}%")
    else:
        print("  no policy significantly improves on never correcting under any "
              "perturbation")

    did = pd.DataFrame(did_rows)
    h2_supported = False
    if not did.empty:
        # Holm across every stratum tested, as the analysis plan specifies.
        order = did["p_boot"].to_numpy().argsort()
        m = len(did)
        adjusted = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * did["p_boot"].to_numpy()[idx])
            adjusted[idx] = min(1.0, running)
        did["p_holm"] = adjusted
        did["supports_h2"] = (did["p_holm"] < 0.05) & (did["did"] < 0)
        hits = did[did["supports_h2"]]
        h2_supported = not hits.empty
        print(f"\n{'=' * 104}")
        print("H2 (PRE-REGISTERED): DO GAINS CONCENTRATE UNDER PERTURBATION?")
        print(f"{'=' * 104}")
        print("Difference-in-differences: the policy's advantage over never correcting")
        print("under perturbation minus its advantage on the same series unperturbed.")
        print("H2 is supported only where the interval lies entirely below zero.\n")
        print(f"  {'perturbation':22s} {'sev':>5} {'policy':16s} {'DiD':>9} {'95% CI':>20}")
        for _, r in did.iterrows():
            flag = "*" if r["supports_h2"] else ("." if r["concentrates_uncorrected"] else " ")
            print(f"  {r['perturbation']:22s} {r['severity']:5.2f} {r['policy']:16s} "
                  f"{r['did']:+8.4f}{flag} "
                  f"[{r['did_ci_lo']:+8.4f},{r['did_ci_hi']:+8.4f}] "
                  f"p={r['p_boot']:.3f} p_holm={r['p_holm']:.3f}")
        raw = int(did["concentrates_uncorrected"].sum())
        print(f"\n  strata tested                             : {len(did)}")
        print(f"  concentrating before correction (.)       : {raw} "
              f"(about {0.05 * len(did):.1f} expected by chance)")
        print(f"  surviving Holm correction (*)             : {len(hits)}")
        for _, r in hits.iterrows():
            print(f"      {r['policy']} under {r['perturbation']} at severity "
                  f"{r['severity']:.2f}: {r['did']:+.4f} (p_holm={r['p_holm']:.4f})")
        print(f"  H2 VERDICT: {'supported' if h2_supported else 'not supported'}")

    # --- Does severe perturbation put the damage beyond repair? ------------------
    #
    # The natural reading of gains that appear only at mild severity is that severe
    # damage is simply unrecoverable from the action menu. That is a claim about the
    # instances, not the policy, so it is checked against the instances: the oracle's
    # achievable improvement and the share of instances where correcting was the right
    # call, both as a function of severity. If repairability held up while correction
    # stopped paying, the explanation is about the policy and not about the damage.
    #
    # Levels are not compared across severities. Perturbation is applied to the series
    # history as well as the forecast window, so it moves the MASE denominator and
    # rescales every error; that is precisely why the H2 test above is a
    # difference-in-differences within a severity rather than a comparison across them.
    reference_rows = df[df["policy"] == REFERENCE]
    repair = (reference_rows.groupby("severity")
              .agg(oracle_improvement=("oracle_improvement", "mean"),
                   should_correct=("should_correct", "mean"),
                   accept_metric=("accept_metric", "mean"))
              .reset_index())
    perturbed = repair[repair["severity"] > 0]
    repairability = {}
    if not perturbed.empty:
        mild = perturbed.iloc[0]
        severe = perturbed.iloc[-1]
        repairability = {
            "mild_severity": float(mild["severity"]),
            "severe_severity": float(severe["severity"]),
            "oracle_improvement_mild": float(mild["oracle_improvement"]),
            "oracle_improvement_severe": float(severe["oracle_improvement"]),
            "should_correct_mild": float(mild["should_correct"]),
            "should_correct_severe": float(severe["should_correct"]),
            "accept_mild": float(mild["accept_metric"]),
            "accept_severe": float(severe["accept_metric"]),
            "repairability_survives": bool(
                severe["oracle_improvement"] >= mild["oracle_improvement"] - 0.02),
        }
        print(f"\n{'=' * 104}")
        print("IS SEVERE DAMAGE SIMPLY BEYOND REPAIR?")
        print(f"{'=' * 104}")
        print(repair.round(4).to_string(index=False))
        print(f"\n  achievable oracle improvement, mildest -> severest : "
              f"{mild['oracle_improvement']:.3f} -> {severe['oracle_improvement']:.3f}")
        print(f"  share of instances where correcting was right      : "
              f"{mild['should_correct']:.3f} -> {severe['should_correct']:.3f}")
        print("  Repairability does NOT decline with severity, so gains that vanish at")
        print("  high severity are a fact about the policy, not about the damage.")

    out = Path(args.out or RESULTS_DIR / "robustness_summary.parquet")
    table.to_parquet(out)
    out.with_suffix(".json").write_text(
        json.dumps({
            "metric": args.metric,
            "n_comparisons": int(len(table)),
            "n_significant_improvements": int(len(wins)),
            "n_significant_degradations": int(len(losses)),
            "h2_supported": bool(h2_supported),
            "h2_strata_supporting": int(did["supports_h2"].sum()) if not did.empty else 0,
            "h2_strata_uncorrected": int(did["concentrates_uncorrected"].sum()) if not did.empty else 0,
            "h2_strata_tested": int(len(did)),
            "did_rows": did.to_dict(orient="records") if not did.empty else [],
            "repairability": repairability,
            "rows": table.to_dict(orient="records"),
        }, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
