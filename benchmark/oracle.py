"""Oracle labels derived from the counterfactual tensor.

**EVALUATION ONLY. Agent code must never import this module.**

Everything here is computed from realised test outcomes. It is the ground truth
against which correction policies are scored, and it is exactly the information a
deployed agent cannot have. ``tests/test_leakage.py`` walks the import graph of
``agents/`` and fails the build if this module appears anywhere in it.

The labels defined here are what make the correction decision measurable:

``should_correct``
    Did *any* available action beat accepting the initial forecast, by a margin
    exceeding ``delta``? This is the target a correction policy is trying to predict.

``oracle_action`` / ``oracle_mase``
    The best available repair and its error -- the ceiling any policy could reach.

``regret``
    How much worse a chosen action was than the oracle action. This is the primary
    decision-quality metric, because it scores *the choice* rather than the forecast.

``large_error``
    Whether the accepted forecast's error exceeded a quantile threshold. The target
    for the reliability estimator (H3).

Selection bias in the oracle -- read this before interpreting any oracle number
------------------------------------------------------------------------------

``oracle_action`` takes the minimum error over ~10 candidate actions on a *single*
realised outcome. The minimum of ten noisy draws is biased downward: some of the
apparent improvement is the winner's curse rather than a real repair. Empirically,
with a 2% margin this labels over 90% of instances as "should correct", which makes
the label nearly degenerate and badly overstates achievable headroom.

Two things follow, and both are implemented here rather than papered over:

1. ``delta`` defaults to a substantial *relative* margin, so an action must beat
   accepting by more than plausible noise before the instance is labelled correctable.
2. :func:`add_realizable_oracle` provides an **achievable** comparator: the best
   action chosen on the *validation* origins of the same series and then applied to
   test. It involves no selection on the evaluation outcome and is therefore unbiased.

The gap between the two is the selection bias itself, and the manuscript reports it
explicitly. Any paper quoting only the unrestricted oracle is quoting an upper bound
that no policy -- including a perfect one -- could actually attain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "add_oracle_labels",
    "add_realizable_oracle",
    "compute_policy_regret",
    "ACCEPT_ACTION",
    "DEFAULT_DELTA",
]

ACCEPT_ACTION = "accept"

#: Minimum relative improvement over accepting for an instance to count as
#: correctable. Chosen to keep the ``should_correct`` label informative rather than
#: near-constant; see the selection-bias note in the module docstring.
DEFAULT_DELTA = 0.10


def add_oracle_labels(
    outcomes: pd.DataFrame,
    metric: str = "mase",
    delta: float = DEFAULT_DELTA,
    large_error_quantile: float = 0.80,
) -> pd.DataFrame:
    """Derive per-instance oracle labels from the outcome tensor.

    Parameters
    ----------
    outcomes:
        One row per (instance, action), as produced by :func:`benchmark.tensor.build_tensor`.
    metric:
        Column to optimise. Lower is better for every supported metric.
    delta:
        Minimum *relative* improvement over accepting for an action to count as a
        genuine correction. Without a margin, floating-point noise would label roughly
        half of all instances as "should correct", which would make the label
        meaningless and inflate every policy's apparent recall.
    large_error_quantile:
        Quantile of accepted-forecast error above which an instance is labelled a
        failure, for H3.

    Returns
    -------
    One row per instance with oracle labels and per-action regret columns.
    """
    if metric not in outcomes.columns:
        raise KeyError(f"metric {metric!r} not present in outcomes")

    df = outcomes.copy()
    # Unavailable actions must not be selectable as the oracle.
    df.loc[~df["applicable"].astype(bool), metric] = np.nan

    accept = (
        df[df["action"] == ACCEPT_ACTION]
        .set_index("instance_id")[metric]
        .rename("accept_metric")
    )

    records: list[dict] = []
    for instance_id, group in df.groupby("instance_id", sort=False):
        valid = group.dropna(subset=[metric])
        if valid.empty:
            continue

        accept_value = float(accept.get(instance_id, np.nan))
        best_row = valid.loc[valid[metric].idxmin()]
        best_value = float(best_row[metric])
        best_action = str(best_row["action"])

        # Relative improvement, guarded for near-zero accept error.
        if np.isfinite(accept_value) and accept_value > 1e-9:
            improvement = (accept_value - best_value) / accept_value
        else:
            improvement = 0.0

        should_correct = bool(np.isfinite(accept_value) and improvement > delta)

        first = group.iloc[0]
        records.append(
            {
                "instance_id": instance_id,
                "series_id": first["series_id"],
                "dataset": first["dataset"],
                "family": first["family"],
                "source": first["source"],
                "split": first["split"],
                "horizon": int(first["horizon"]),
                "base_model": first["base_model"],
                "accept_metric": accept_value,
                "oracle_action": best_action if should_correct else ACCEPT_ACTION,
                "oracle_metric": best_value if should_correct else accept_value,
                "best_action_any": best_action,
                "best_metric_any": best_value,
                "should_correct": should_correct,
                "oracle_improvement": float(improvement),
                "n_applicable_actions": int(valid.shape[0]),
                # How many actions would have made things worse. A high value means
                # correcting blindly is dangerous here, which is precisely the
                # situation an adaptive policy must detect.
                "n_harmful_actions": int(
                    (valid[metric] > accept_value).sum()
                    if np.isfinite(accept_value) else 0
                ),
                "harmful_fraction": float(
                    (valid[metric] > accept_value).mean()
                    if np.isfinite(accept_value) else np.nan
                ),
            }
        )

    labels = pd.DataFrame(records)
    if labels.empty:
        return labels

    # `large_error` is defined per (split, horizon): error magnitude is not comparable
    # across horizons, and a global quantile would simply label all long-horizon
    # instances as failures.
    labels["large_error"] = False
    for (_, _), idx in labels.groupby(["split", "horizon"]).groups.items():
        sub = labels.loc[idx, "accept_metric"]
        finite = sub[np.isfinite(sub)]
        if finite.empty:
            continue
        threshold = float(np.quantile(finite, large_error_quantile))
        labels.loc[idx, "large_error"] = sub > threshold
        labels.loc[idx, "large_error_threshold"] = threshold

    return labels


def compute_policy_regret(
    outcomes: pd.DataFrame,
    chosen: pd.Series | dict[str, str],
    labels: pd.DataFrame,
    metric: str = "mase",
) -> pd.DataFrame:
    """Score a policy's action choices against the oracle.

    ``chosen`` maps ``instance_id -> action name``. Returns per-instance realised
    metric, regret versus the oracle, and whether the correction helped or hurt.

    This function is the reason the tensor exists: evaluating a new correction policy
    costs one table lookup per instance, with no model refitting at all.
    """
    if isinstance(chosen, dict):
        chosen = pd.Series(chosen, name="chosen_action")
    chosen = chosen.rename("chosen_action")

    lookup = (
        outcomes.set_index(["instance_id", "action"])[metric]
        .rename("chosen_metric")
    )

    df = labels.set_index("instance_id").join(chosen, how="inner")
    keys = list(zip(df.index, df["chosen_action"]))
    df["chosen_metric"] = [lookup.get(k, np.nan) for k in keys]

    # A policy that picks an inapplicable action falls back to accepting: that is what
    # a real system would do, and silently dropping those instances would flatter the
    # policy by removing its own mistakes from the comparison.
    fell_back = ~np.isfinite(df["chosen_metric"])
    df.loc[fell_back, "chosen_metric"] = df.loc[fell_back, "accept_metric"]
    df["fell_back"] = fell_back

    df["regret"] = df["chosen_metric"] - df["oracle_metric"]
    df["corrected"] = df["chosen_action"] != ACCEPT_ACTION
    df["improved"] = df["chosen_metric"] < df["accept_metric"] - 1e-12
    df["harmed"] = df["chosen_metric"] > df["accept_metric"] + 1e-12
    df["decision_correct"] = df["corrected"] == df["should_correct"]
    return df.reset_index()


def add_realizable_oracle(
    outcomes: pd.DataFrame,
    metric: str = "mase",
    group_keys: tuple[str, ...] = ("series_id", "horizon"),
) -> pd.DataFrame:
    """The best *fixed* action per group, chosen on validation and applied to test.

    This is the honest counterpart to :func:`add_oracle_labels`. Because the action is
    selected using validation origins and scored on test origins, no selection happens
    on the evaluation outcome, so the resulting error is an unbiased estimate of what a
    policy with perfect per-series (but not per-instance) knowledge would achieve.

    It doubles as a strong non-adaptive baseline: any per-instance adaptive policy has
    to beat "pick one good repair for this series and stick with it" to justify its
    complexity. That is a considerably harder bar than beating "never correct", and it
    is the bar most of the self-correction literature has not been measured against.

    Returns one row per test instance with the chosen action and its realised metric.
    """
    df = outcomes.copy()
    df.loc[~df["applicable"].astype(bool), metric] = np.nan

    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if val.empty or test.empty:
        return pd.DataFrame()

    keys = list(group_keys)
    # Mean validation error of each action within each group.
    val_scores = (
        val.groupby(keys + ["action"], sort=False)[metric]
        .mean()
        .reset_index()
        .dropna(subset=[metric])
    )
    if val_scores.empty:
        return pd.DataFrame()

    best = (
        val_scores.sort_values(metric)
        .groupby(keys, sort=False)
        .first()
        .reset_index()[keys + ["action"]]
        .rename(columns={"action": "realizable_action"})
    )

    merged = test.merge(best, on=keys, how="left")
    chosen = merged[merged["action"] == merged["realizable_action"]]
    out = chosen[["instance_id", "realizable_action", metric]].rename(
        columns={metric: "realizable_metric"}
    )

    # Groups whose chosen action turned out inapplicable on test fall back to accept,
    # which is what a deployed system would do.
    accept = (
        test[test["action"] == ACCEPT_ACTION]
        .set_index("instance_id")[metric]
        .rename("accept_metric")
    )
    out = out.set_index("instance_id")
    missing = accept.index.difference(out.index)
    if len(missing):
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {
                        "realizable_action": ACCEPT_ACTION,
                        "realizable_metric": accept.loc[missing],
                    }
                ),
            ]
        )
    out["realizable_metric"] = out["realizable_metric"].fillna(accept)
    return out.reset_index()
