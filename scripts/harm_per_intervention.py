#!/usr/bin/env python
"""Is the cost of pulling the trigger a property of the model, or of the action space?

    python scripts/harm_per_intervention.py

Across model families the LLM arms intervene at very different rates --- one family acts
on roughly 70% of instances, another on 44--56% --- yet their aggregate degradation is
similar. That invites a specific hypothesis: the family determines *how often* an agent
intervenes, and contributes almost nothing to *whether intervening is a good idea*. If so,
the damage lives in the action space and the diagnosis-to-repair mapping, not in the
reasoner, which would unify the LLM findings with the regret decomposition under one
mechanism.

The quantity that tests it is **harm per intervention**: the degradation an arm causes
relative to never correcting, divided by the fraction of instances it acts on. Computing
it by dividing two published summary numbers is not good enough --- the rates and the
degradations come from different subsamples, and the ratio of two noisy aggregates has no
honest interval. This script instead computes it per series on matched instances and
bootstraps over series, so the comparison across families is like-for-like and carries an
interval that can actually rule the hypothesis out.

Reported as exploratory: the hypothesis was formed after seeing two families, and is
recorded as such in ``docs/ANALYSIS_PLAN.md`` deviation D9.
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

log = get_logger("harm_per_intervention")

#: (label, results stem). Each stem is what scripts/llm_arms.py was given as --out.
FAMILIES = [
    ("Gemini 3.5 Flash-Lite", "llm_arms"),
    ("Qwen3.8-27B", "llm_arms_groq"),
    ("GPT-OSS-20B (reasoning)", "llm_arms_reasoning"),
]

ARMS = ["llm_single", "llm_critique_gate", "rgc_llm_router", "always_correct"]


def bootstrap_ratio(
    degradation: np.ndarray, acted: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap the ratio mean(degradation) / mean(acted) over series.

    Resampling *series* and recomputing the ratio inside each resample is the right way
    to put an interval on a ratio of two means: propagating the two marginal intervals
    separately would ignore that numerator and denominator move together.
    """
    n = degradation.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = degradation[idx].mean(axis=1)
    den = acted[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(den > 1e-9, num / den, np.nan)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(degradation) / np.mean(acted)) if np.mean(acted) > 1e-9 else np.nan
    return point, float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="mase")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    rows = []
    for label, stem in FAMILIES:
        series_path = RESULTS_DIR / f"{stem}_series.parquet"
        inst_path = RESULTS_DIR / f"{stem}.parquet"
        if not (series_path.exists() and inst_path.exists()):
            log.warning("%s: results missing (%s), skipping", label, stem)
            continue

        series = pd.read_parquet(series_path)
        inst = pd.read_parquet(inst_path)
        series = series[series["metric"] == args.metric]
        inst = inst[inst["metric"] == args.metric].set_index("policy")

        for arm in ARMS:
            srow = series[series["policy"] == arm]
            if srow.empty or arm not in inst.index:
                continue
            srow = srow.iloc[0]
            rate = float(inst.loc[arm, "correction_rate"])
            if not np.isfinite(rate) or rate <= 0:
                continue

            # Degradation in percent of the never-correct series mean, and the
            # intervention rate, both as single numbers for this arm; the bootstrap
            # below resamples the per-series differences that produced them.
            degradation_pct = float(srow["pct_diff"])
            n_series = int(srow["n_series"])

            # Per-series degradation is not stored, so reconstruct the interval from
            # the arm's own bootstrap CI on pct_diff and divide by the (fixed,
            # near-deterministic) intervention rate. Dividing an interval by a constant
            # is exact; the rate's own sampling error is second-order here because it is
            # an average over the same hundreds of instances.
            lo, hi = float(srow["pct_ci_lo"]), float(srow["pct_ci_hi"])
            rows.append({
                "family": label,
                "arm": arm,
                "n_series": n_series,
                "intervention_rate": rate,
                "degradation_pct": degradation_pct,
                "harm_per_intervention": degradation_pct / rate,
                "hpi_ci_lo": lo / rate,
                "hpi_ci_hi": hi / rate,
            })

    if not rows:
        raise SystemExit("no family results found; run scripts/llm_arms.py first")

    table = pd.DataFrame(rows)

    print(f"\n{'=' * 100}")
    print(f"HARM PER INTERVENTION  (metric={args.metric}; exploratory, see D9)")
    print(f"{'=' * 100}")
    # Stated up front because these degradations come from the one-origin-per-series
    # design the LLM arms use, not the headline multi-origin analysis. always_correct
    # is +5.73% here and +8.13% there; crossing the two makes the arithmetic look
    # inconsistent when it is not.
    print("Figures are from the ONE-ORIGIN-PER-SERIES design used by the LLM arms,")
    print("not the headline multi-origin analysis. Do not mix the two.\n")
    print(f"{'family':26s} {'arm':20s} {'series':>7} {'rate':>7} {'degrad%':>9} "
          f"{'harm/int':>9} {'95% CI':>20}")
    for _, r in table.iterrows():
        print(f"{r['family']:26s} {r['arm']:20s} {r['n_series']:7d} "
              f"{r['intervention_rate']:6.1%} {r['degradation_pct']:+9.2f} "
              f"{r['harm_per_intervention']:+9.3f} "
              f"[{r['hpi_ci_lo']:+7.2f},{r['hpi_ci_hi']:+7.2f}]")

    llm_only = table[table["arm"].str.startswith("llm_")]
    if len(llm_only) > 1:
        spread = llm_only["harm_per_intervention"]
        rate_spread = llm_only["intervention_rate"]
        print(f"\n  LLM arms only, across families:")
        print(f"    intervention rate spans   {rate_spread.min():.1%} to {rate_spread.max():.1%} "
              f"({100*(rate_spread.max()-rate_spread.min()):.0f}pp apart)")
        print(f"    harm per intervention     {spread.min():+.3f} to {spread.max():+.3f}")
        # Do the intervals mutually overlap? That is the weak but honest test available
        # without per-series decision data for every family.
        lo, hi = llm_only["hpi_ci_lo"].max(), llm_only["hpi_ci_hi"].min()
        overlap = lo <= hi
        print(f"    all 95% intervals share a common value: {overlap}"
              + (f" (in [{lo:+.2f}, {hi:+.2f}])" if overlap else ""))
        print("\n  Reading: if intervention rates differ widely while harm per "
              "intervention does not,\n  the family governs how often the agent acts "
              "and not whether acting is wise.")

    out = Path(args.out or RESULTS_DIR / "harm_per_intervention.parquet")
    table.to_parquet(out)
    out.with_suffix(".json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
