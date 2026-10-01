"""Statistical machinery for the confirmatory analysis.

Implements exactly what ``docs/ANALYSIS_PLAN.md`` pre-registered, and nothing more:
paired non-parametric tests across series, Diebold-Mariano for forecast pairs,
series-level bootstrap confidence intervals, Holm correction across the hypothesis
family, and Cliff's delta effect sizes.

One choice deserves emphasis because it changes conclusions rather than decorating
them: **the bootstrap resamples series, not instances.** Instances from the same
series share a model fit, a history and a regime, so they are far from independent.
Resampling instances would shrink every confidence interval by roughly the square root
of the number of instances per series and would manufacture significance that is not
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TestResult",
    "paired_wilcoxon",
    "diebold_mariano",
    "bootstrap_ci",
    "bootstrap_diff_ci",
    "cliffs_delta",
    "holm_correction",
    "tost_equivalence",
    "compare_policies",
]


@dataclass
class TestResult:
    """A single statistical comparison, with everything needed to report it."""

    name: str
    statistic: float
    p_value: float
    effect_size: float = float("nan")
    effect_name: str = ""
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n: int = 0
    p_adjusted: float = float("nan")
    significant: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #


def paired_wilcoxon(a: np.ndarray, b: np.ndarray, name: str = "") -> TestResult:
    """Wilcoxon signed-rank test on paired observations (``a`` vs ``b``).

    Non-parametric by design: forecast-error distributions are heavily right-skewed,
    and a t-test on raw MASE would be driven by a handful of catastrophic series.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return TestResult(name, float("nan"), float("nan"), n=int(a.size),
                          note="too few paired observations")
    if np.allclose(a, b):
        return TestResult(name, 0.0, 1.0, effect_size=0.0, effect_name="cliffs_delta",
                          n=int(a.size), note="identical samples")
    stat, p = stats.wilcoxon(a, b)
    delta = cliffs_delta(a, b)
    return TestResult(
        name=name, statistic=float(stat), p_value=float(p),
        effect_size=delta, effect_name="cliffs_delta", n=int(a.size),
    )


def diebold_mariano(
    errors_a: np.ndarray, errors_b: np.ndarray, horizon: int = 1, power: int = 1,
    name: str = "",
) -> TestResult:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    Tests equal predictive accuracy of two forecasts on the same target. The HLN
    correction matters here: our per-series samples are short, and the uncorrected
    statistic over-rejects badly at these sample sizes.
    """
    ea, eb = np.asarray(errors_a, dtype=float), np.asarray(errors_b, dtype=float)
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb = ea[ok], eb[ok]
    n = ea.size
    if n < 8:
        return TestResult(name, float("nan"), float("nan"), n=int(n),
                          note="too few observations for DM")

    d = np.abs(ea) ** power - np.abs(eb) ** power
    d_bar = float(np.mean(d))
    if np.allclose(d, 0.0):
        return TestResult(name, 0.0, 1.0, n=int(n), note="identical loss differentials")

    # Long-run variance with Newey-West style truncation at horizon - 1.
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for lag in range(1, min(horizon, n - 1)):
        cov = float(np.cov(d[lag:], d[:-lag], ddof=0)[0, 1])
        var += 2.0 * cov
    var = max(var / n, 1e-300)

    dm = d_bar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * hln
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_corrected), df=n - 1))
    return TestResult(
        name=name, statistic=float(dm_corrected), p_value=float(p), n=int(n),
        effect_size=d_bar, effect_name="mean_loss_differential",
    )


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    Computed by rank rather than by the O(n^2) pairwise comparison, so it stays cheap
    on the full instance set. Negative means ``a`` tends to be smaller -- which, for
    error metrics, means ``a`` is better.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    rank_sum_a = float(np.sum(ranks[:n]))
    # U statistic -> delta
    u = rank_sum_a - n * (n + 1) / 2.0
    return float(2.0 * u / (n * m) - 1.0)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    statistic=np.mean,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for a statistic, resampling ``groups`` (clusters) when given.

    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    values = values[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    point = float(statistic(values))

    if groups is None:
        idx = rng.integers(0, values.size, size=(n_resamples, values.size))
        draws = np.array([statistic(values[i]) for i in idx])
    else:
        unique = np.unique(groups)
        index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
        draws = np.empty(n_resamples, dtype=float)
        for b in range(n_resamples):
            picked = rng.choice(unique, size=unique.size, replace=True)
            sample = np.concatenate([index_by_group[g] for g in picked])
            draws[b] = statistic(values[sample])

    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(low), float(high)


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, groups: np.ndarray | None = None,
    n_resamples: int = 10000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the *paired* mean difference ``a - b``.

    Paired resampling preserves the within-instance correlation between arms, which is
    substantial here because all arms share the same base forecast. Resampling the two
    arms independently would inflate the interval enormously.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    g = np.asarray(groups)[ok] if groups is not None else None
    return bootstrap_ci(a - b, g, n_resamples, alpha, np.mean, seed)


# --------------------------------------------------------------------------- #
# Multiplicity and equivalence
# --------------------------------------------------------------------------- #


def holm_correction(results: Sequence[TestResult], alpha: float = 0.05
                    ) -> list[TestResult]:
    """Holm-Bonferroni step-down correction across a family of tests.

    Applied to the seven confirmatory hypotheses as one family, per the analysis plan.
    Exploratory per-stratum breakdowns are reported unadjusted and labelled as such.
    """
    finite = [r for r in results if np.isfinite(r.p_value)]
    order = sorted(range(len(finite)), key=lambda i: finite[i].p_value)
    m = len(finite)
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (m - rank) * finite[i].p_value)
        running = max(running, adjusted)  # enforce monotonicity
        finite[i].p_adjusted = running
        finite[i].significant = running < alpha
    for r in results:
        if not np.isfinite(r.p_value):
            r.p_adjusted = float("nan")
            r.significant = False
    return list(results)


def tost_equivalence(
    a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05, name: str = ""
) -> TestResult:
    """Two one-sided tests for equivalence of paired means within +/- ``margin``.

    Used for H4, where the claim is that adaptive tool use is *no worse* than
    exhaustive use. A non-significant difference is not evidence of equivalence, so
    the equivalence claim needs its own test rather than an absent one.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = d.size
    if n < 3:
        return TestResult(name, float("nan"), float("nan"), n=int(n),
                          note="too few observations for TOST")

    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    if se <= 0:
        p = 0.0 if abs(float(np.mean(d))) < margin else 1.0
        return TestResult(name, 0.0, p, n=int(n), note="zero variance")

    mean_d = float(np.mean(d))
    t_low = (mean_d + margin) / se
    t_high = (mean_d - margin) / se
    p_low = 1.0 - stats.t.cdf(t_low, df=n - 1)
    p_high = stats.t.cdf(t_high, df=n - 1)
    p = float(max(p_low, p_high))
    return TestResult(
        name=name, statistic=mean_d, p_value=p, n=int(n),
        effect_size=mean_d, effect_name="mean_difference",
        ci_low=mean_d - 1.96 * se, ci_high=mean_d + 1.96 * se,
        significant=p < alpha,
        note=f"equivalence within +/-{margin}",
    )


# --------------------------------------------------------------------------- #
# High-level comparison
# --------------------------------------------------------------------------- #


def compare_policies(
    scored: pd.DataFrame,
    reference: str = "never_correct",
    metric: str = "chosen_metric",
    policy_col: str = "policy",
    group_col: str = "series_id",
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare every policy against a reference arm, with Holm correction.

    ``scored`` must contain one row per (policy, instance). Comparisons are paired on
    ``instance_id``, so every arm is judged on exactly the same instances.
    """
    if reference not in set(scored[policy_col]):
        raise ValueError(f"reference policy {reference!r} not in results")

    wide = scored.pivot_table(
        index=["instance_id", group_col], columns=policy_col, values=metric
    )
    ref = wide[reference]

    results: list[TestResult] = []
    rows: list[dict] = []
    for policy in wide.columns:
        if policy == reference:
            continue
        pair = wide[[policy, reference]].dropna()
        if pair.empty:
            continue
        a = pair[policy].to_numpy()
        b = pair[reference].to_numpy()
        groups = pair.index.get_level_values(group_col).to_numpy()

        test = paired_wilcoxon(a, b, name=f"{policy}_vs_{reference}")
        diff, low, high = bootstrap_diff_ci(a, b, groups, n_resamples, alpha, seed)
        test.ci_low, test.ci_high = low, high
        results.append(test)

        # Report the mean effect and the rank effect separately.
        #
        # These can disagree in *sign* on heavy-tailed error distributions, and for
        # MASE they routinely do: a handful of instances with enormous error dominate
        # the mean while leaving the median and the win rate untouched. Printing a
        # mean-based effect next to a rank-based p-value produced exactly that
        # contradiction here -- a policy shown as "+0.4% worse" alongside
        # "significant: True" from a test that was in fact detecting an improvement.
        # Both summaries are legitimate; presenting them as one number is not.
        d = a - b
        nonzero = d[np.abs(d) > 1e-12]
        rows.append(
            {
                "policy": policy,
                "reference": reference,
                "mean_metric": float(np.mean(a)),
                "reference_metric": float(np.mean(b)),
                "mean_difference": diff,
                "relative_change_pct": 100.0 * diff / float(np.mean(b))
                if np.mean(b) else float("nan"),
                "ci_low": low,
                "ci_high": high,
                # Outlier-robust companions to the mean.
                "median_metric": float(np.median(a)),
                "reference_median": float(np.median(b)),
                "median_difference": float(np.median(nonzero)) if nonzero.size else 0.0,
                "trimmed_mean_metric": float(stats.trim_mean(a, 0.10)),
                "win_rate": float(
                    (d < -1e-12).mean() + 0.5 * (np.abs(d) <= 1e-12).mean()
                ),
                "n_better": int((d < -1e-12).sum()),
                "n_worse": int((d > 1e-12).sum()),
                # True when the mean and the rank test point in opposite directions --
                # a flag that the mean is being driven by the tail.
                "direction_conflict": bool(
                    np.sign(diff) != 0
                    and np.sign(diff) != np.sign(test.effect_size or 0.0)
                    and abs(test.effect_size or 0.0) > 1e-9
                ),
                "n_instances": int(len(pair)),
                "n_series": int(len(np.unique(groups))),
            }
        )

    holm_correction(results, alpha)
    for row, test in zip(rows, results):
        row.update(
            {
                "p_value": test.p_value,
                "p_adjusted": test.p_adjusted,
                "significant": test.significant,
                "cliffs_delta": test.effect_size,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_metric").reset_index(drop=True)
