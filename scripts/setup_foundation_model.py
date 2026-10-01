#!/usr/bin/env python
"""Install and cache the time-series foundation model baseline.

    python scripts/setup_foundation_model.py
    python scripts/setup_foundation_model.py --model amazon/chronos-bolt-tiny

Kept out of requirements.txt on purpose: it pulls a package plus model weights, and the
benchmark must stay installable and runnable without that download. Everything except
the `chronos` forecaster works without it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_install:
        try:
            import chronos  # noqa: F401
            print("chronos-forecasting already installed")
        except ImportError:
            print("installing chronos-forecasting ...")
            rc = subprocess.call(
                [sys.executable, "-m", "pip", "install", "chronos-forecasting"]
            )
            if rc != 0:
                print("installation failed", file=sys.stderr)
                return rc

    print(f"downloading and warming weights: {args.model}")
    import numpy as np

    from forecasting.base import FORECASTERS

    model = FORECASTERS.build("chronos", model_name=args.model)
    series = np.sin(np.arange(200) * 2 * np.pi / 12) * 5 + 100
    result = model.forecast(series, horizon=12, seasonal_period=12, seed=0)

    if result.fallback:
        print(f"FAILED: {result.metadata.get('error')}", file=sys.stderr)
        return 1
    print(f"ok -- produced a {result.horizon}-step forecast, "
          f"first value {result.point[0]:.3f}")
    print("\nEnable it by adding `chronos` to forecast.models in a config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
