"""Series-level paired inference: the unit of analysis the pre-registration mandates.

``docs/ANALYSIS_PLAN.md`` §7 fixes the bootstrap unit as the **series**, not the
instance, because instances drawn from the same series are not independent: they share
a generating process, a base model, a scaling denominator and, under rolling origins,
most of their history. An instance-level test treats 3,284 correlated observations as
3,284 independent ones and reports a standard error that is too small by roughly the
square root of the within-series design effect. On this benchmark that is a factor of
two to three, which is the difference between an interval that excludes zero and one
that does not.

Every comparison the manuscript makes therefore goes through :func:`paired_series_test`,
which:

* collapses each policy to one value per series (the mean over that series' instances),
* pairs the two policies on the series they share,
* reports the mean paired difference with a **series-level bootstrap** interval,
* reports a Wilcoxon signed-rank p-value over series,
* and reports the **win rate** -- the fraction of series on which the policy is better.

The win rate is reported next to the mean because on heavy-tailed data the two can
disagree: a policy can improve the mean by a wide margin while losing on most series,
which is a materially different claim than "it is better". Deviation D2 in the analysis
plan records the first time that bit us.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "per_series_means",
    "paired_series_test",
    "cliffs_delta",
    "compare_all_to_reference",
]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta for paired samples, negative when ``a`` is smaller (better).

    Computed from the rank of the pooled sample in O(n log n) rather than the O(n^2)
    pairwise definition.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    rank_sum_a = float(ranks[:n].sum())
    # P(a > b) - P(a < b), derived from the Mann-Whitney U statistic.
    u = rank_sum_a - n * (n + 1) / 2.0
    return float(2.0 * u / (n * m) - 1.0)


def per_series_means(
    values: pd.Series, series_of_instance: pd.Series
) -> pd.Series:
    """Collapse instance-level values to one mean per series.

    ``values`` is indexed by ``instance_id``; ``series_of_instance`` maps
    ``instance_id -> series_id``.
    """
    df = pd.DataFrame({"value": values})
    df["series_id"] = series_of_instance.reindex(df.index).values
    return df.dropna(subset=["series_id"]).groupby("series_id")["value"].mean()


def paired_series_test(
    treatment: pd.Series,
    reference: pd.Series,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired comparison of two policies at the series level.

    Both arguments are indexed by ``series_id``. Lower is better for every metric in
    this project, so a negative ``mean_diff`` means the treatment improved on the
    reference.

    The bootstrap resamples **series**, which is the independence unit. The returned
    interval is a percentile interval on the mean paired difference.
    """
    joined = pd.concat([treatment.rename("t"), reference.rename("r")], axis=1).dropna()
    if joined.empty:
        return {k: float("nan") for k in (
            "mean_diff", "ci_lo", "ci_hi", "pct_diff", "wilcoxon_p", "win_rate",
            "cliffs_delta", "n_series",
        )} | {"n_series": 0, "significant": False}

    diff = (joined["t"] - joined["r"]).to_numpy()
    n = diff.size

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])

    nonzero = diff[np.abs(diff) > 1e-12]
    p = float(stats.wilcoxon(joined["t"], joined["r"])[1]) if nonzero.size > 10 else float("nan")

    ref_mean = float(joined["r"].mean())
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        # Percent change in the series-mean metric, the scale the manuscript quotes.
        "pct_diff": 100.0 * float(diff.mean()) / ref_mean if ref_mean else float("nan"),
        "pct_ci_lo": 100.0 * float(ci_lo) / ref_mean if ref_mean else float("nan"),
        "pct_ci_hi": 100.0 * float(ci_hi) / ref_mean if ref_mean else float("nan"),
        "wilcoxon_p": p,
        # Ties split evenly so the win rate of a policy against itself is exactly 0.5.
        "win_rate": float((diff < -1e-12).mean() + 0.5 * (np.abs(diff) <= 1e-12).mean()),
        "cliffs_delta": cliffs_delta(joined["t"].to_numpy(), joined["r"].to_numpy()),
        "n_series": int(n),
        # The decision rule the analysis plan fixes: the interval, not the p-value.
        "significant": bool(ci_lo < 0 and ci_hi < 0) or bool(ci_lo > 0 and ci_hi > 0),
    }


def compare_all_to_reference(
    wide: pd.DataFrame,
    series_of_instance: pd.Series,
    reference: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Series-level comparison of every column of ``wide`` against ``reference``.

    ``wide`` is instances x policies. Returns one row per policy.
    """
    by_series = {
        policy: per_series_means(wide[policy], series_of_instance)
        for policy in wide.columns
    }
    ref = by_series[reference]
    rows = []
    for policy, values in by_series.items():
        result = paired_series_test(values, ref, n_boot=n_boot, alpha=alpha, seed=seed)
        rows.append({"policy": policy, "series_mean": float(values.mean()), **result})
    return pd.DataFrame(rows)
