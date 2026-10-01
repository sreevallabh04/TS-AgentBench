#!/usr/bin/env python
"""Decompose the gap between accepting and the oracle into named sources of loss.

    python scripts/regret_decomposition.py --run latest

This is the analysis that explains why diagnosis-conditioned correction -- the dominant
pattern in the agentic forecasting literature (diagnose a failure mode, apply its
textbook repair) -- underperforms a plain static ensemble in this benchmark. It uses
ground-truth failure-mode labels, available only on synthetic instances, to build an
idealised diagnosis-conditioned policy and compares four points on the same scale:

  accept                        <- the reference
  first-matched action          <- diagnose the TRUE mode, apply MODE_TO_ACTIONS[mode][0]
                                    (the "diagnose then apply the canonical fix" pattern)
  best-of-matched candidates    <- diagnose the TRUE mode, pick the best of its 2-3
                                    matched candidates using TEST outcomes (an oracle
                                    restricted to the hand-designed candidate list)
  best-of-all actions           <- the unrestricted oracle (no restriction at all)

The gap between successive rows attributes the total achievable improvement to two
independent failure sources of the diagnose-then-repair pattern:

  * "first-choice loss": using a fixed canonical mapping instead of validating among
    the matched candidates
  * "restriction loss": the candidate list itself excludes the action that would have
    helped most

Both intermediate policies use TEST-set information to select the action and are
therefore ceilings, not deployable policies -- exactly like the existing unrestricted
oracle. They isolate where headroom is lost, which no deployable policy can recover
more of than this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.diagnosis import MODE_TO_ACTIONS  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from datasets.base import FailureMode  # noqa: E402
from evaluation.metrics import geometric_mean  # noqa: E402

log = get_logger("regret_decomposition")


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


def decompose(outcomes: pd.DataFrame, signals: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the four-point decomposition for one metric, synthetic instances only.

    Ground-truth failure-mode labels (``label_failure_modes``) exist only where
    ``label_is_ground_truth`` is true, i.e. synthetic series. Real data cannot support
    this analysis without begging the question of what the "true" failure mode is.
    """
    test = outcomes[outcomes["split"] == "test"].copy()
    test.loc[~test["applicable"].astype(bool), metric] = np.nan

    synthetic_ids = set(signals.loc[signals["source"] == "synthetic", "instance_id"])
    test = test[test["instance_id"].isin(synthetic_ids)]
    mode_by_instance = signals.set_index("instance_id")["label_failure_modes"].to_dict()

    accept, first_matched, best_matched, best_any = {}, {}, {}, {}
    for instance_id, group in test.groupby("instance_id"):
        modes_raw = str(mode_by_instance.get(instance_id, "") or "")
        modes = [m for m in modes_raw.split(",") if m]

        candidates: set[str] = set()
        first_action: str | None = None
        for mode_str in modes:
            try:
                mode = FailureMode(mode_str)
            except ValueError:
                continue
            matched = MODE_TO_ACTIONS.get(mode, ())
            candidates.update(matched)
            if first_action is None and matched:
                first_action = matched[0]

        by_action = group.set_index("action")[metric]
        accept_value = by_action.get("accept", np.nan)
        accept[instance_id] = accept_value

        first_matched[instance_id] = (
            by_action.get(first_action, accept_value) if first_action else accept_value
        )
        matched_values = by_action.reindex(list(candidates)).dropna()
        best_matched[instance_id] = (
            float(matched_values.min()) if len(matched_values) else accept_value
        )
        finite = by_action.dropna()
        best_any[instance_id] = float(finite.min()) if len(finite) else accept_value

    df = pd.DataFrame(
        {
            "accept": pd.Series(accept),
            "first_matched_action": pd.Series(first_matched),
            "best_of_matched_candidates": pd.Series(best_matched),
            "best_of_all_actions": pd.Series(best_any),
        }
    ).dropna()

    ref_mean = df["accept"].mean()
    ref_geo = geometric_mean(df["accept"])
    rows = []
    for column in df.columns:
        values = df[column]
        rows.append(
            {
                "metric": metric,
                "stage": column,
                "mean": float(values.mean()),
                "mean_pct": 100.0 * (values.mean() / ref_mean - 1.0),
                "geometric_mean": geometric_mean(values),
                "geo_pct": 100.0 * (geometric_mean(values) / ref_geo - 1.0),
                "n": int(len(values)),
            }
        )
    table = pd.DataFrame(rows)

    # Named loss terms, in mean-percent units relative to accept.
    pct = table.set_index("stage")["mean_pct"]
    losses = pd.DataFrame(
        [
            {
                "metric": metric,
                "loss": "total achievable (unrestricted oracle vs. accept)",
                "value_pct": pct["best_of_all_actions"],
            },
            {
                "metric": metric,
                "loss": "restriction loss (matched-candidate oracle vs. unrestricted oracle)",
                "value_pct": pct["best_of_matched_candidates"] - pct["best_of_all_actions"],
            },
            {
                "metric": metric,
                "loss": "first-choice loss (canonical fix vs. best-of-matched, true diagnosis)",
                "value_pct": pct["first_matched_action"] - pct["best_of_matched_candidates"],
            },
        ]
    )
    return table, losses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")

    tables, losses = [], []
    for metric in ("mase", "crps_scaled"):
        if metric not in outcomes.columns:
            continue
        table, loss = decompose(outcomes, signals, metric)
        tables.append(table)
        losses.append(loss)

        print(f"\n=== {metric}: regret decomposition (synthetic, ground-truth diagnosis, "
              f"n={table['n'].iloc[0]}) ===")
        print(table[["stage", "mean", "mean_pct", "geometric_mean", "geo_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nnamed loss terms (mean %, relative to accept):")
        print(loss[["loss", "value_pct"]].to_string(index=False,
              float_format=lambda v: f"{v:+.2f}"))

    combined_table = pd.concat(tables, ignore_index=True)
    combined_loss = pd.concat(losses, ignore_index=True)

    out = Path(args.out or RESULTS_DIR / "regret_decomposition.parquet")
    combined_table.to_parquet(out)
    combined_loss.to_parquet(out.with_name("regret_decomposition_losses.parquet"))
    print(f"\nSaved to {out} and {out.with_name('regret_decomposition_losses.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
