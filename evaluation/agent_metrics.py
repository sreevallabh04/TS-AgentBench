"""Agent-level metrics: diagnosis, tool selection, correction decisions, cost.

These are the metrics that distinguish this benchmark from a forecasting benchmark.
Forecast error tells you how the agent *did*; these tell you whether it *decided*
correctly, which is the object of study.

Diagnosis accuracy is measurable only where the failure mode is known by construction,
i.e. on synthetic families. On real data the "true" failure mode would have to come from
the same detectors the agent uses, which would make the metric circular. Every function
here therefore filters on the ground-truth flag rather than quietly scoring everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "diagnosis_accuracy",
    "correction_decision_metrics",
    "tool_selection_metrics",
    "cost_summary",
]


def diagnosis_accuracy(
    decisions: pd.DataFrame,
    signals: pd.DataFrame,
    ground_truth_only: bool = True,
) -> pd.DataFrame:
    """Compare the agent's diagnosed failure mode against the generator's.

    A series may carry several expected failure modes (a near-unit-root series is both
    drifting and trend-misspecified), so a diagnosis counts as correct if it names any
    of them. Scoring against a single arbitrary "primary" mode would penalise correct
    answers.
    """
    if "diagnosed_mode" not in decisions.columns:
        return pd.DataFrame()

    cols = ["instance_id", "label_failure_modes", "label_is_ground_truth", "family"]
    available = [c for c in cols if c in signals.columns]
    merged = decisions.merge(signals[available], on="instance_id", how="inner")
    if ground_truth_only and "label_is_ground_truth" in merged.columns:
        merged = merged[merged["label_is_ground_truth"].astype(bool)]
    if merged.empty:
        return pd.DataFrame()

    def correct(row) -> bool:
        truth = {m for m in str(row.get("label_failure_modes", "")).split(",") if m}
        predicted = str(row.get("diagnosed_mode", "none"))
        if not truth or truth == {"none"}:
            # No failure was induced, so "none" is the correct answer.
            return predicted in ("none", "not_diagnosed")
        return predicted in truth

    merged["diagnosis_correct"] = merged.apply(correct, axis=1)

    rows = []
    for (policy, family), group in merged.groupby(["policy", "family"], dropna=False):
        rows.append(
            {
                "policy": policy,
                "family": family,
                "n": int(len(group)),
                "accuracy": float(group["diagnosis_correct"].mean()),
                "most_common_diagnosis": (
                    group["diagnosed_mode"].mode().iloc[0]
                    if not group["diagnosed_mode"].mode().empty else "n/a"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "accuracy"])


def correction_decision_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall, F1 and regret for the decision to correct.

    Computable only because the benchmark stores counterfactual outcomes: without them
    there is no ``should_correct`` label to score against.
    """
    rows = []
    for policy, group in scored.groupby("policy"):
        corrected = group["corrected"].astype(bool)
        should = group["should_correct"].astype(bool)

        tp = int((corrected & should).sum())
        fp = int((corrected & ~should).sum())
        fn = int((~corrected & should).sum())
        tn = int((~corrected & ~should).sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
            else np.nan
        )
        acted = group[corrected]
        rows.append(
            {
                "policy": policy,
                "n": int(len(group)),
                "correction_rate": float(corrected.mean()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": float((tp + tn) / max(1, len(group))),
                "helped_rate": float(acted["improved"].mean()) if len(acted) else np.nan,
                "harmed_rate": float(acted["harmed"].mean()) if len(acted) else np.nan,
                # The headline decision metric: how much worse than the best available
                # action, on average.
                "mean_regret": float(group["regret"].mean()),
                "median_regret": float(group["regret"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_regret").reset_index(drop=True)


def tool_selection_metrics(
    decisions: pd.DataFrame, exhaustive_cost: float | None = None
) -> pd.DataFrame:
    """Cost of selective versus exhaustive tool use (H4).

    ``exhaustive_cost`` defaults to the maximum observed per-instance cost, which is the
    cost of the arm that evaluates every action.
    """
    if "compute_cost" not in decisions.columns:
        return pd.DataFrame()

    ceiling = exhaustive_cost or float(
        pd.to_numeric(decisions["compute_cost"], errors="coerce").max()
    )
    rows = []
    for policy, group in decisions.groupby("policy"):
        cost = pd.to_numeric(group["compute_cost"], errors="coerce").fillna(0.0)
        llm = pd.to_numeric(group.get("n_llm_calls"), errors="coerce").fillna(0.0) \
            if "n_llm_calls" in group else pd.Series([0.0] * len(group))
        rows.append(
            {
                "policy": policy,
                "mean_action_evaluations": float(cost.mean()),
                "mean_llm_calls": float(llm.mean()),
                "cost_vs_exhaustive": float(cost.mean() / ceiling) if ceiling else np.nan,
                "n_candidates_mean": float(
                    pd.to_numeric(group.get("n_candidates_evaluated"), errors="coerce")
                    .fillna(0).mean()
                ) if "n_candidates_evaluated" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_action_evaluations").reset_index(drop=True)


def cost_summary(decisions: pd.DataFrame, llm_stats: dict | None = None) -> dict:
    """Aggregate computational cost, for the reproducibility and cost reporting."""
    out: dict[str, float] = {}
    if "compute_cost" in decisions.columns:
        cost = pd.to_numeric(decisions["compute_cost"], errors="coerce")
        out["total_action_evaluations"] = float(cost.sum())
        out["mean_action_evaluations"] = float(cost.mean())
    if "n_llm_calls" in decisions.columns:
        out["total_llm_calls"] = float(
            pd.to_numeric(decisions["n_llm_calls"], errors="coerce").fillna(0).sum()
        )
    if llm_stats:
        out.update({k: float(v) for k, v in llm_stats.items()})
    return out
