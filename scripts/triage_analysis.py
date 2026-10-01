#!/usr/bin/env python
"""Selective-prediction value of the failure-risk estimator: a triage table.

    python scripts/triage_analysis.py --run latest

Automated correction is largely a wash in this benchmark (see the headline results),
but a well-calibrated failure-risk estimate is independently valuable for routing
forecasts to human review, which is closer to how most production forecasting systems
actually use anomaly/risk scores today. This script reports precision, recall and lift
for reviewing the top-k% of forecasts by predicted risk -- the standard way to
communicate the operating value of a ranking model to a non-technical stakeholder, and a
positive, deployable finding to set alongside the cautionary results on automated
correction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.reliability import ReliabilityEstimator  # noqa: E402
from benchmark.oracle import add_oracle_labels  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402

log = get_logger("triage")

#: Review-budget grid reported in the table. Chosen to span "spot check" (5%) through
#: "review half the workload" (50%).
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


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
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    run_dir = resolve_run(args.run)
    outcomes = pd.read_parquet(run_dir / "outcomes.parquet")
    signals = pd.read_parquet(run_dir / "signals.parquet")

    test = outcomes[outcomes["split"] == "test"]
    val = outcomes[outcomes["split"] == "val"]
    test_labels = add_oracle_labels(test, args.metric, 0.10, 0.80)
    val_labels = add_oracle_labels(val, args.metric, 0.10, 0.80)
    test_signals = signals[signals["instance_id"].isin(test_labels["instance_id"])]
    val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]

    estimator = ReliabilityEstimator(random_state=0).fit(val_signals, val_labels)
    risk = pd.Series(
        estimator.predict_risk(test_signals), index=test_signals["instance_id"].values
    )
    target = test_labels.set_index("instance_id")["large_error"].astype(int)
    common = risk.index.intersection(target.index)
    risk, target = risk[common], target[common]

    base_rate = float(target.mean())
    auroc = float(roc_auc_score(target, risk))
    order = risk.sort_values(ascending=False).index

    rows = []
    for budget in BUDGETS:
        n = max(1, int(len(order) * budget))
        flagged = order[:n]
        true_positives = int(target.loc[flagged].sum())
        precision = true_positives / n
        recall = true_positives / max(1, int(target.sum()))
        rows.append(
            {
                "review_budget": budget,
                "n_reviewed": n,
                "precision": precision,
                "recall": recall,
                "lift": precision / base_rate if base_rate > 0 else float("nan"),
            }
        )
    table = pd.DataFrame(rows)

    print(f"n={len(risk)}, base rate of large-error instances = {base_rate:.1%}, "
          f"AUROC = {auroc:.3f}\n")
    print(table.to_string(
        index=False,
        formatters={
            "review_budget": lambda v: f"{v:.0%}",
            "precision": lambda v: f"{v:.1%}",
            "recall": lambda v: f"{v:.1%}",
            "lift": lambda v: f"{v:.2f}x",
        },
    ))

    table.attrs["auroc"] = auroc
    table.attrs["base_rate"] = base_rate
    table.attrs["n"] = len(risk)

    out = Path(args.out or RESULTS_DIR / "triage.parquet")
    table.to_parquet(out)
    meta = pd.DataFrame([{"auroc": auroc, "base_rate": base_rate, "n": len(risk)}])
    meta.to_parquet(out.with_name("triage_meta.parquet"))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
