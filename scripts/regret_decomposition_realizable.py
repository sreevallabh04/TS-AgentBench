#!/usr/bin/env python
"""The regret decomposition without the winner's curse, and with intervals.

    python scripts/regret_decomposition_realizable.py --run main_20260918-233847_3fa46848

``scripts/regret_decomposition.py`` compares an unrestricted oracle (the minimum over ten
actions) against a matched-candidate oracle (the minimum over two or three), and a
canonical fix against that matched oracle. Every one of those minima is taken over TEST
outcomes. The expected minimum of k noisy draws falls as k grows even when no action
carries any real signal, so both named losses are inflated by order statistics alone:
ten-versus-three for the restriction loss, three-versus-one for the first-choice loss.
That is exactly the selection-inflated-oracle pathology the manuscript warns about in its
measurement-pathologies section, and the original decomposition committed it.

This script removes the selection. Every choice is made on VALIDATION origins and scored
on held-out TEST origins, using the same rule as the manuscript's realizable oracle
(``benchmark.oracle.add_realizable_oracle``): per (series, horizon), the action with the
lowest mean validation error. Four stages, all selection-free on the test outcome:

  accept              the reference
  canonical fix       true failure mode -> its first matched action (no selection at all)
  validated matched   true failure mode -> the best of its 2-3 matched candidates,
                      chosen on validation
  validated any       the best of all ten actions, chosen on validation
                      (= the realizable oracle, restricted to the synthetic instances
                      where the true failure mode is known)

Every stage and every loss term carries a series-level bootstrap interval, because the
decomposition's original table was instance-level means on heavy-tailed errors, which is
the mistake deviation D7 corrected everywhere else.

The test-selected ceilings are reported alongside, clearly labelled, so the size of the
winner's curse can be read directly off the difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.diagnosis import MODE_TO_ACTIONS  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from datasets.base import FailureMode  # noqa: E402

log = get_logger("regret_decomposition_realizable")

GROUP = ["series_id", "horizon"]
STAGES = ["accept", "canonical_fix", "validated_matched", "validated_any",
          "test_matched", "test_any"]


def matched_candidates(modes_raw: str) -> tuple[str | None, list[str]]:
    """The canonical first action and the full matched-candidate list for a mode set."""
    first, cands = None, []
    for mode_str in [m for m in str(modes_raw or "").split(",") if m]:
        try:
            mode = FailureMode(mode_str)
        except ValueError:
            continue
        matched = list(MODE_TO_ACTIONS.get(mode, ()))
        for a in matched:
            if a not in cands:
                cands.append(a)
        if first is None and matched:
            first = matched[0]
    return first, cands


def stage_values(outcomes: pd.DataFrame, signals: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per synthetic test instance, one column per stage."""
    df = outcomes.copy()
    df.loc[~df["applicable"].astype(bool), metric] = np.nan

    synthetic = signals[signals["source"] == "synthetic"]
    modes = synthetic.set_index("instance_id")["label_failure_modes"]
    syn_ids = set(synthetic["instance_id"])

    val = df[(df["split"] == "val") & df["instance_id"].isin(syn_ids)]
    test = df[(df["split"] == "test") & df["instance_id"].isin(syn_ids)]

    # Mean validation error per (series, horizon, action): the realizable selection table.
    val_scores = (val.groupby(GROUP + ["action"])[metric].mean()
                  .dropna().reset_index())

    # The failure mode is a property of the synthetic series' generator, so the
    # candidate list is defined per group from its test instances' labels.
    test_modes = test.drop_duplicates("instance_id").copy()
    test_modes["modes"] = test_modes["instance_id"].map(modes)
    group_modes = (test_modes.groupby(GROUP)["modes"].first())

    choice_matched, choice_any, canonical = {}, {}, {}
    for key, block in val_scores.groupby(GROUP):
        first, cands = matched_candidates(group_modes.get(key, ""))
        ranked = block.sort_values(metric)
        choice_any[key] = ranked["action"].iloc[0]
        inside = ranked[ranked["action"].isin(cands)]
        choice_matched[key] = inside["action"].iloc[0] if len(inside) else "accept"
        canonical[key] = first or "accept"

    rows = []
    for (iid, sid, h), g in test.groupby(["instance_id", "series_id", "horizon"]):
        by = g.set_index("action")[metric]
        acc = by.get("accept", np.nan)
        if not np.isfinite(acc):
            continue
        key = (sid, h)
        _, cands = matched_candidates(modes.get(iid, ""))

        def realised(action):
            v = by.get(action, np.nan)
            return acc if not np.isfinite(v) else float(v)  # inapplicable -> accept

        m_vals = by.reindex(cands).dropna()
        rows.append({
            "instance_id": iid, "series_id": sid,
            "accept": float(acc),
            "canonical_fix": realised(canonical.get(key, "accept")),
            "validated_matched": realised(choice_matched.get(key, "accept")),
            "validated_any": realised(choice_any.get(key, "accept")),
            # Test-selected ceilings, reported only to size the winner's curse.
            "test_matched": float(m_vals.min()) if len(m_vals) else float(acc),
            "test_any": float(by.dropna().min()),
        })
    return pd.DataFrame(rows)


def bootstrap(per_series: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Series-level intervals for each stage vs accept and for each loss term.

    Every quantity is a ratio of means over series, recomputed inside each resample, so
    numerator and denominator move together.
    """
    rng = np.random.default_rng(seed)
    n = len(per_series)
    idx = rng.integers(0, n, size=(n_boot, n))
    arr = {c: per_series[c].to_numpy() for c in STAGES}
    acc_boot = arr["accept"][idx].mean(axis=1)
    acc_mean = arr["accept"].mean()

    def pct(col):
        point = 100.0 * (arr[col].mean() / acc_mean - 1.0)
        boot = 100.0 * (arr[col][idx].mean(axis=1) / acc_boot - 1.0)
        return point, boot

    out = {}
    boots = {}
    for c in STAGES:
        point, boot = pct(c)
        boots[c] = boot
        out[f"stage:{c}"] = (point, *np.quantile(boot, [0.025, 0.975]))

    def loss(name, a, b):
        point = out[f"stage:{a}"][0] - out[f"stage:{b}"][0]
        boot = boots[a] - boots[b]
        out[name] = (point, *np.quantile(boot, [0.025, 0.975]))

    # Realizable (validation-selected) decomposition: the one the paper may rely on.
    loss("realizable total achievable", "validated_any", "accept")
    loss("realizable first-choice loss", "canonical_fix", "validated_matched")
    loss("realizable restriction loss", "validated_matched", "validated_any")
    # Test-selected ceilings, kept only to show how much was winner's curse.
    loss("ceiling total achievable", "test_any", "accept")
    loss("ceiling first-choice loss", "canonical_fix", "test_matched")
    loss("ceiling restriction loss", "test_matched", "test_any")
    return {k: {"point": float(v[0]), "ci_lo": float(v[1]), "ci_hi": float(v[2]),
                "excludes_zero": bool(v[1] > 0 or v[2] < 0)} for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    run_dir = RUNS_DIR / args.run
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")

    report = {}
    for metric in ("mase", "crps_scaled"):
        inst = stage_values(outcomes, signals, metric)
        per_series = inst.groupby("series_id")[STAGES].mean()
        stats = bootstrap(per_series, args.n_boot, args.seed)
        report[metric] = {"n_instances": int(len(inst)),
                          "n_series": int(len(per_series)), "stats": stats}

        print(f"\n{'=' * 100}")
        print(f"REALIZABLE DECOMPOSITION  metric={metric}  synthetic, ground-truth diagnosis  "
              f"n={len(inst)} instances / {len(per_series)} series")
        print(f"{'=' * 100}")
        print("Every % is relative to accept, series-level, with a 95% bootstrap interval "
              "over series.\n")
        print(f"  {'quantity':34s} {'point':>9} {'95% CI':>22}")
        for k, v in stats.items():
            flag = "*" if v["excludes_zero"] else " "
            print(f"  {k:34s} {v['point']:+8.2f}%{flag} [{v['ci_lo']:+8.2f}, {v['ci_hi']:+8.2f}]")
        rt = stats["realizable total achievable"]
        ct = stats["ceiling total achievable"]
        print(f"\n  achievable headroom, test-selected ceiling : {ct['point']:+.2f}%")
        print(f"  achievable headroom, validation-selected   : {rt['point']:+.2f}%")
        print(f"  share of the ceiling that was winner's curse: "
              f"{100.0 * (1 - rt['point'] / ct['point']):.0f}%"
              if ct["point"] else "")

    out = Path(args.out or RESULTS_DIR / "regret_decomposition_realizable.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
