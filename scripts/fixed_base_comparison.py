#!/usr/bin/env python
"""Does correction pay when the base forecast was not adaptively chosen?

    python scripts/fixed_base_comparison.py --fixed fixed_base_20260923-164259_8a8949c5

This is the confound named in the Limitations and pre-registered in deviation D11. Every
headline result rests on a base forecast that was itself selected per instance by
rolling-origin backtest over a nine-model pool, which is a strong adaptive procedure.
"Correction does not help" and "correction had nothing left to fix" are not separated
anywhere in the main experiment. The fixed-base run holds the base at ETS and changes
nothing else, so the two runs differ in one factor and the comparison can separate them.

This script was written and committed **before the fixed-base run finished**, so the
comparison it performs could not be chosen to suit the answer. It does three things:

1. **Verifies the single-factor property rather than asserting it.** The two runs must
   cover the same instances, and their diagnostic signals --- which are computed from
   series history alone and therefore cannot depend on the base model --- must agree. If
   they do not, the comparison is not single-factor and the script says so and stops.

2. **Compares like with like.** Each run's policies are scored against *its own*
   never-correct reference, at the series level, exactly as the headline table is built.
   Levels are never compared across runs: the two have different base forecasts and
   therefore different references, and subtracting across them would be the cross-design
   error this paper warns about elsewhere.

3. **Reports the contrast that answers D11.** For each policy, the advantage over never
   correcting under an adaptive base, beside the same advantage under a fixed base. If
   correction pays only on the fixed base, the confound is real and the headline claims
   are scoped to adaptively-chosen bases. If it pays on neither, the confound is
   discharged.
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
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from evaluation.series_level import paired_series_test, per_series_means  # noqa: E402

log = get_logger("fixed_base_comparison")

REFERENCE = "never_correct"
#: Signal columns are derived from series history only. If the base model changed them,
#: something other than the base forecast differs between the runs.
SIGNAL_TOLERANCE = 1e-9


def latest(prefix: str) -> Path:
    candidates = sorted(
        (p for p in RUNS_DIR.iterdir()
         if p.is_dir() and p.name.startswith(prefix)
         and (p / "outcomes.parquet").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"no completed run found with prefix {prefix!r}")
    return candidates[-1]


def check_single_factor(a_dir: Path, b_dir: Path) -> list[str]:
    """Confirm the two runs differ in the base forecast and nothing else.

    Asserting a single-factor design in prose is cheap. This checks it: the instance
    sets must match, and the history-derived diagnostics must be numerically identical,
    because nothing about them can legitimately depend on which base model was chosen.
    """
    problems: list[str] = []
    sa = pd.read_parquet(a_dir / "signals.parquet")
    sb = pd.read_parquet(b_dir / "signals.parquet")

    ia, ib = set(sa["instance_id"]), set(sb["instance_id"])
    if ia != ib:
        problems.append(
            f"instance sets differ: {len(ia - ib)} only in {a_dir.name}, "
            f"{len(ib - ia)} only in {b_dir.name}")
        return problems

    common = sorted(ia)
    sa = sa.set_index("instance_id").loc[common]
    sb = sb.set_index("instance_id").loc[common]

    # Signals that describe the fitted forecast (backtest error, model disagreement,
    # selection margin, interval width) legitimately differ: they are properties of the
    # base model. Everything upstream of the forecast must not.
    history_only = [c for c in sa.columns
                    if c.split(".")[0] in {"context", "stl", "ts_features", "acf",
                                           "changepoint", "stationarity",
                                           "distribution_shift", "anomaly", "seasonality"}
                    and pd.api.types.is_numeric_dtype(sa[c])]
    for col in history_only:
        if col not in sb.columns:
            problems.append(f"signal {col!r} missing from {b_dir.name}")
            continue
        x, y = sa[col].to_numpy(dtype=float), sb[col].to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        delta = np.abs(np.where(both_nan, 0.0, x - y))
        worst = float(np.nanmax(delta)) if delta.size else 0.0
        if np.isfinite(worst) and worst > SIGNAL_TOLERANCE:
            problems.append(
                f"history-derived signal {col!r} differs between runs "
                f"(max |delta| = {worst:.3g}); the runs are not single-factor")
    log.info("checked %d history-derived signals across %d instances",
             len(history_only), len(common))
    return problems


def arm_table(run_dir: Path, metric: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Series-level difference from never correcting, inside one run."""
    scores = pd.read_parquet(run_dir / "scores.parquet")
    if metric not in scores.columns:
        raise SystemExit(
            f"{run_dir.name}: column {metric!r} not in scores.parquet "
            f"(available: {sorted(scores.columns)[:8]}...)")
    # scores.parquet carries its own series mapping, so no join with signals is needed;
    # `chosen_metric` is the run's primary metric under the policy in that row.
    series_of = scores.drop_duplicates("instance_id").set_index("instance_id")["series_id"]

    ref = scores[scores["policy"] == REFERENCE].set_index("instance_id")[metric]
    if ref.empty:
        raise SystemExit(f"{run_dir.name}: no {REFERENCE} arm to compare against")
    ref_series = per_series_means(ref, series_of)

    rows = []
    for policy in sorted(scores["policy"].unique()):
        if policy == REFERENCE:
            continue
        arm = scores[scores["policy"] == policy].set_index("instance_id")[metric]
        if arm.empty:
            continue
        stat = paired_series_test(per_series_means(arm, series_of), ref_series,
                                  n_boot=n_boot, seed=seed)
        rows.append({
            "policy": policy,
            "n_series": stat["n_series"],
            "pct_diff": stat["pct_diff"],
            "pct_ci_lo": stat["pct_ci_lo"],
            "pct_ci_hi": stat["pct_ci_hi"],
            "win_rate": stat["win_rate"],
            "significant": stat["significant"],
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", default=None,
                        help="run directory name for the backtest-selected base")
    parser.add_argument("--fixed", default=None,
                        help="run directory name for the fixed ETS base")
    parser.add_argument("--metric", default="chosen_metric",
                        help="column in scores.parquet holding the "
                             "realised error under each policy")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-single-factor-check", action="store_true",
                        help="report the comparison even if the runs are not "
                             "single-factor; the output is then labelled as such")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    a_dir = RUNS_DIR / args.adaptive if args.adaptive else latest("main_")
    b_dir = RUNS_DIR / args.fixed if args.fixed else latest("fixed_base_")
    for d in (a_dir, b_dir):
        if not (d / "scores.parquet").exists():
            raise SystemExit(
                f"{d.name} has no scores.parquet -- the run did not complete. "
                "Deviation D11 requires this be reported as not completed rather "
                "than quietly omitted.")

    problems = check_single_factor(a_dir, b_dir)
    if problems:
        print("\nSINGLE-FACTOR CHECK FAILED:")
        for p in problems[:20]:
            print(f"  {p}")
        if not args.skip_single_factor_check:
            raise SystemExit(
                "\nRefusing to report a single-factor comparison between runs that are "
                "not single-factor. Re-run with --skip-single-factor-check only if the "
                "difference is understood and will be disclosed.")
    else:
        print("single-factor check: OK -- same instances, identical history diagnostics")

    adaptive = arm_table(a_dir, args.metric, args.n_boot, args.seed)
    fixed = arm_table(b_dir, args.metric, args.n_boot, args.seed)
    merged = adaptive.merge(fixed, on="policy", how="outer",
                            suffixes=("_adaptive", "_fixed"))

    print(f"\n{'=' * 104}")
    print(f"CORRECTION UNDER AN ADAPTIVE BASE VS A FIXED ETS BASE   metric={args.metric}")
    print(f"{'=' * 104}")
    print(f"adaptive base : {a_dir.name}")
    print(f"fixed base    : {b_dir.name}")
    print("\nEach column is the series-level difference from never correcting WITHIN its")
    print("own run. Levels are not comparable across runs; the two have different base")
    print("forecasts and therefore different references. Negative favours correcting.\n")
    print(f"{'policy':20s} {'adaptive base':>28} {'fixed ETS base':>28}")
    for _, r in merged.iterrows():
        def cell(suffix):
            pct = r.get(f"pct_diff_{suffix}")
            if pct is None or not np.isfinite(pct):
                return f"{'--':>28}"
            flag = "*" if r.get(f"significant_{suffix}") else " "
            return (f"{pct:+8.2f}%{flag} [{r[f'pct_ci_lo_{suffix}']:+6.2f},"
                    f"{r[f'pct_ci_hi_{suffix}']:+6.2f}]")
        print(f"{r['policy']:20s} {cell('adaptive'):>28} {cell('fixed'):>28}")
    print("\n  * = series-level bootstrap interval excludes zero")

    # The D11 verdict, stated mechanically so it cannot drift.
    def helps(frame: pd.DataFrame) -> list[str]:
        if frame.empty:
            return []
        good = frame[frame["significant"] & (frame["pct_diff"] < 0)]
        return sorted(good["policy"].tolist())

    helps_adaptive, helps_fixed = helps(adaptive), helps(fixed)
    gained = [p for p in helps_fixed if p not in helps_adaptive]
    print(f"\n  policies that significantly beat never correcting:")
    print(f"    under an adaptive base : {helps_adaptive or 'none'}")
    print(f"    under a fixed ETS base : {helps_fixed or 'none'}")
    if gained:
        reading = ("A: correction pays on a fixed base and not on an adaptive one, so "
                   "the base pipeline's strength is the operative variable and the "
                   "headline claims are scoped to adaptively-selected bases")
    else:
        reading = ("B: correction does not pay under either base, so the confound is "
                   "discharged and the result does not depend on the base forecast "
                   "having been adaptively chosen")
    print(f"\n  D11 READING {reading}")

    out = Path(args.out or RESULTS_DIR / "fixed_base_comparison.parquet")
    merged.to_parquet(out)
    out.with_suffix(".json").write_text(json.dumps({
        "metric": args.metric,
        "adaptive_run": a_dir.name,
        "fixed_run": b_dir.name,
        "single_factor_ok": not problems,
        "single_factor_problems": problems,
        "helps_under_adaptive_base": helps_adaptive,
        "helps_under_fixed_base": helps_fixed,
        "policies_gained_under_fixed_base": gained,
        "d11_reading": "A" if gained else "B",
        "rows": merged.to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
