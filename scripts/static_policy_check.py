#!/usr/bin/env python
"""Two checks on the paper's one positive result: was it picked on test, and is it real?

    python scripts/static_policy_check.py --run main_20260918-233847_3fa46848

The headline says the only policy of any kind that reliably improves on not correcting
is an unconditional static ensemble. A reviewer raised two objections, and both are
checkable rather than arguable:

1. **Post-hoc selection.** ``always_ensemble`` was not among the pre-registered arms. If it
   is simply the best of ten static actions *on test*, declaring it the winner is the
   selection-inflated-oracle problem again. The check: choose the single static action
   with the lowest mean error on VALIDATION origins, apply it to every test instance, and
   report what it does. If validation independently picks the ensemble, the result is
   realizable; if it picks something else, the ensemble result is a post-hoc pick.

2. **Generator dependence.** 420 of the 578 series come from our own generators. The check:
   the same comparison on real series only and on synthetic series only.

Series-level paired bootstrap throughout.
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

log = get_logger("static_policy_check")


def effect(test: pd.DataFrame, action: str, metric: str, series_of: pd.Series,
           ids=None, n_boot: int = 10000, seed: int = 0) -> dict:
    sub = test if ids is None else test[test["instance_id"].isin(ids)]
    acc = sub[sub["action"] == "accept"].set_index("instance_id")[metric]
    arm = (sub[sub["action"] == action].set_index("instance_id")[metric]
           .reindex(acc.index).fillna(acc))  # inapplicable -> accept, as deployed
    st = paired_series_test(per_series_means(arm, series_of),
                            per_series_means(acc, series_of), n_boot=n_boot, seed=seed)
    return {k: st[k] for k in ("pct_diff", "pct_ci_lo", "pct_ci_hi", "significant",
                               "n_series", "win_rate")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    setup_logging()

    o = pd.read_parquet(RUNS_DIR / args.run / "outcomes.parquet")
    o.loc[~o["applicable"].astype(bool), ["mase", "crps_scaled"]] = np.nan
    val, test = o[o["split"] == "val"], o[o["split"] == "test"]
    first = test.drop_duplicates("instance_id").set_index("instance_id")
    series_of, source = first["series_id"], first["source"]

    report = {}
    for metric in ("mase", "crps_scaled"):
        ranking = val.groupby("action")[metric].mean().dropna().sort_values()
        chosen = str(ranking.index[0])
        rec = {
            "validation_ranking": {a: float(v) for a, v in ranking.items()},
            "validation_chosen_action": chosen,
            "validation_chosen_effect": effect(test, chosen, metric, series_of,
                                               n_boot=args.n_boot, seed=args.seed),
            "ensemble_effect": effect(test, "ensemble", metric, series_of,
                                      n_boot=args.n_boot, seed=args.seed),
            "ensemble_is_validation_choice": chosen == "ensemble",
            "ensemble_by_source": {
                src: effect(test, "ensemble", metric, series_of,
                            ids=source[source == src].index,
                            n_boot=args.n_boot, seed=args.seed)
                for src in ("real", "synthetic")
            },
        }
        report[metric] = rec

        print(f"\n=== {metric} ===")
        print("validation ranking of static actions (lower is better):")
        for a, v in list(rec["validation_ranking"].items())[:5]:
            print(f"   {a:24s} {v:.4f}")
        ve = rec["validation_chosen_effect"]
        print(f"validation chooses: {chosen}  -> on test {ve['pct_diff']:+.2f}% "
              f"[{ve['pct_ci_lo']:+.2f}, {ve['pct_ci_hi']:+.2f}]")
        ee = rec["ensemble_effect"]
        print(f"ensemble (as reported)    -> on test {ee['pct_diff']:+.2f}% "
              f"[{ee['pct_ci_lo']:+.2f}, {ee['pct_ci_hi']:+.2f}]   "
              f"validation's own choice: {rec['ensemble_is_validation_choice']}")
        for src, e in rec["ensemble_by_source"].items():
            print(f"   ensemble, {src:9s} only: {e['pct_diff']:+.2f}% "
                  f"[{e['pct_ci_lo']:+.2f}, {e['pct_ci_hi']:+.2f}]  n_series={e['n_series']}")

    out = RESULTS_DIR / "static_policy_check.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
