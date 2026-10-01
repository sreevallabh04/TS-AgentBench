#!/usr/bin/env python
"""Probe whether the LLM has memorised the benchmark's real datasets.

    python scripts/contamination_probe.py --datasets ett_h1 monash_tourism_monthly

Several datasets used here are long-standing and public, so a frontier model may have
seen them during training. If it has, an LLM arm could appear to "reason" about a series
while actually recalling it, and any conclusion drawn from those arms would be invalid.

The probe is a paired comparison. For each series we ask the model to continue a
context, twice:

* **intact** -- the true consecutive history;
* **shuffled** -- the same values in a random order, which destroys all temporal
  structure while preserving the marginal distribution exactly.

Memorisation predicts a large advantage on intact contexts, because the continuation can
be recalled rather than inferred. Genuine (non-memorised) temporal reasoning also
predicts an advantage, so a positive result is not proof of contamination on its own --
but the *size* of the gap, compared against synthetic series the model cannot have seen,
is informative. Synthetic series act as the negative control: any gap there is the
baseline benefit of real temporal structure.

Interpretation, stated in advance so it is not chosen after seeing the numbers:
a real-data gap substantially larger than the synthetic gap is evidence of
contamination, and the affected datasets should be excluded from LLM-arm claims.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging import get_logger, setup_logging  # noqa: E402
from core.seed import seeded_rng  # noqa: E402

log = get_logger("contamination")

SYSTEM = (
    "You continue numeric sequences. Reply with exactly the requested number of "
    "comma-separated numbers and nothing else."
)


def make_prompt(context: np.ndarray, n_predict: int) -> str:
    values = ", ".join(f"{v:.4f}" for v in context)
    return (
        f"Continue this sequence with exactly {n_predict} more numbers.\n\n"
        f"{values}\n\n"
        f"Reply with {n_predict} comma-separated numbers only."
    )


def parse_numbers(text: str, n: int) -> np.ndarray | None:
    cleaned = text.replace("\n", " ").replace("[", "").replace("]", "")
    parts = [p.strip() for p in cleaned.split(",")]
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            continue
    if len(values) < n:
        return None
    return np.array(values[:n], dtype=float)


def scaled_error(actual: np.ndarray, predicted: np.ndarray, context: np.ndarray) -> float:
    """MAE scaled by the context's first-difference magnitude, so scales are comparable."""
    denom = float(np.mean(np.abs(np.diff(context)))) or 1.0
    return float(np.mean(np.abs(actual - predicted)) / denom)


def probe_series(client, series, n_context: int, n_predict: int, seed: int
                 ) -> tuple[float, float] | None:
    """Return (intact_error, shuffled_error) for one series."""
    target = series.target[np.isfinite(series.target)]
    if target.size < n_context + n_predict:
        return None

    context = target[-(n_context + n_predict): -n_predict]
    actual = target[-n_predict:]

    rng = seeded_rng(seed, "contamination", series.series_id)
    shuffled = context.copy()
    rng.shuffle(shuffled)

    results = []
    for ctx in (context, shuffled):
        response = client.generate(make_prompt(ctx, n_predict), system=SYSTEM)
        predicted = parse_numbers(response.text, n_predict)
        if predicted is None:
            return None
        results.append(scaled_error(actual, predicted, ctx))
    return results[0], results[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["ett_h1"])
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-context", type=int, default=64)
    parser.add_argument("--n-predict", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    setup_logging()
    from dotenv import load_dotenv

    from agents.llm import LLMClient
    from datasets.real import load_real_dataset
    from datasets.synthetic import generate_family

    load_dotenv()
    client = LLMClient(model=args.model or "gemini-3.5-flash-lite", temperature=0.0,
                       max_output_tokens=256)

    groups: dict[str, list] = {}
    for key in args.datasets:
        try:
            groups[key] = load_real_dataset(key, max_series=args.n_series,
                                            max_length=512, seed=args.seed)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", key, exc)

    # Negative control: the model cannot have memorised freshly seeded synthetic data.
    groups["synthetic_control"] = generate_family(
        "seasonal_single", args.n_series, base_seed=args.seed + 991, n_timesteps=400
    )

    summary: dict[str, tuple[float, float, int]] = {}
    for name, series_list in groups.items():
        intact, shuffled = [], []
        for series in series_list:
            result = probe_series(client, series, args.n_context, args.n_predict,
                                  args.seed)
            if result is None:
                continue
            intact.append(result[0])
            shuffled.append(result[1])
        if intact:
            summary[name] = (float(np.mean(intact)), float(np.mean(shuffled)),
                             len(intact))

    print("\n=== Contamination probe ===")
    print(f"{'dataset':28s} {'intact':>9s} {'shuffled':>9s} {'gap':>9s} {'n':>4s}")
    control_gap = None
    for name, (a, b, n) in summary.items():
        gap = b - a
        if name == "synthetic_control":
            control_gap = gap
        print(f"{name:28s} {a:9.3f} {b:9.3f} {gap:9.3f} {n:4d}")

    if control_gap is not None:
        print(
            f"\nSynthetic control gap = {control_gap:.3f}. This is the benefit the "
            "model gets from genuine temporal structure alone."
        )
        for name, (a, b, n) in summary.items():
            if name == "synthetic_control":
                continue
            gap = b - a
            if gap > 2.0 * max(control_gap, 1e-6):
                print(
                    f"  WARNING: {name} shows a gap of {gap:.3f}, more than twice the "
                    "control. Treat LLM-arm results on this dataset as potentially "
                    "contaminated and exclude it from LLM claims."
                )

    print(f"\nLLM usage: {client.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
