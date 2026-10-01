#!/usr/bin/env python
"""TS-AgentBench experiment entry point.

    python run_experiment.py --config configs/smoke.yaml
    python run_experiment.py --config configs/experiment.yaml
    python run_experiment.py --config configs/experiment.yaml --set seeds=[0,1] --force-tensor

Every run writes a self-contained directory under ``results/runs/`` containing the
manifest (config, code version, environment), the counterfactual tensor, per-policy
decisions and scores, the statistical comparison, and a log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project importable when invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import ConfigError, load_config  # noqa: E402
from core.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a TS-AgentBench experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config (or a name under configs/).",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set seeds=[0,1] --set n_jobs=4. "
             "Repeatable.",
    )
    parser.add_argument(
        "--force-tensor", action="store_true",
        help="Rebuild the counterfactual tensor even if a cached one exists.",
    )
    parser.add_argument("--run-id", default=None, help="Override the generated run id.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Validate and print the resolved config, then exit without running.",
    )
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)

    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    if args.print_config:
        import json

        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        return 0

    from experiments.runner import run_experiment

    try:
        result = run_experiment(cfg, force_tensor=args.force_tensor, run_id=args.run_id)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    comparison = result.get("comparison")
    print(f"\nRun directory: {result['run_dir']}")
    if comparison is not None and not comparison.empty:
        cols = ["policy", "mean_metric", "relative_change_pct", "ci_low", "ci_high",
                "p_adjusted", "significant"]
        print("\nPolicy comparison (vs reference arm, lower metric is better):")
        print(comparison[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
