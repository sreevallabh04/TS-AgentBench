"""Publication figures.

Every figure is generated from a results directory; none contains a hardcoded number.
Regenerating the paper after a rerun is therefore a single command, and a figure can
never drift out of sync with the experiment that produced it.

Style choices are deliberately plain: greyscale-safe colours, no chartjunk, readable at
single-column width. Journals print in greyscale more often than authors expect, and a
figure whose message depends on hue is a figure that stops working in print.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "set_style", "fig_policy_comparison", "fig_risk_coverage", "fig_calibration",
    "fig_correction_outcomes", "fig_stratum_gains", "fig_cost_accuracy",
    "fig_feature_importance", "fig_robustness", "fig_oracle_gap",
    "fig_example_forecast", "save",
]

# Colour-blind-safe and distinguishable in greyscale by ordering.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
HIGHLIGHT = "#C44E52"
NEUTRAL = "#8C8C8C"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,          # journals typically require >=300 dpi
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.autolayout": True,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, path: Path, also_pdf: bool = True) -> Path:
    """Save a figure as PNG (for review) and PDF (for submission)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main results
# --------------------------------------------------------------------------- #


def fig_policy_comparison(
    comparison: pd.DataFrame, reference_metric: float, metric_name: str = "MASE"
) -> plt.Figure:
    """Mean metric per policy with bootstrap CIs on the difference from the reference.

    Plotted as a *difference* from the reference arm rather than as absolute levels:
    the arms share a base forecast, so absolute bars would differ by a sliver and hide
    the very effect the figure exists to show.
    """
    df = comparison.sort_values("mean_difference").copy()
    y = np.arange(len(df))
    colours = [HIGHLIGHT if p.startswith("rgc") else NEUTRAL for p in df["policy"]]

    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(df) + 1.6))
    ax.barh(y, df["mean_difference"], color=colours, alpha=0.85, height=0.62)
    ax.errorbar(
        df["mean_difference"], y,
        xerr=[df["mean_difference"] - df["ci_low"], df["ci_high"] - df["mean_difference"]],
        fmt="none", ecolor="black", elinewidth=1.0, capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["policy"])
    ax.set_xlabel(f"Change in {metric_name} vs. never-correcting (negative is better)")
    ax.set_title(
        f"Correction policies vs. never correcting\n"
        f"(reference {metric_name} = {reference_metric:.3f}; bars are 95% bootstrap CIs)"
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        if row.get("significant"):
            ax.text(
                row["ci_high"], yi, "  *", va="center", ha="left",
                fontsize=11, fontweight="bold",
            )
    return fig


def fig_correction_outcomes(scores: pd.DataFrame) -> plt.Figure:
    """For each policy: how often a correction helped, hurt, or changed nothing.

    This is the figure that distinguishes a selective policy from a busy one. Two arms
    can correct equally often and differ entirely in whether those corrections were
    worth making.
    """
    rows = []
    for policy, group in scores.groupby("policy"):
        corrected = group[group["corrected"]]
        n = len(group)
        if n == 0:
            continue
        rows.append(
            {
                "policy": policy,
                "helped": len(corrected[corrected["improved"]]) / n,
                "neutral": len(corrected[~corrected["improved"] & ~corrected["harmed"]]) / n,
                "harmed": len(corrected[corrected["harmed"]]) / n,
                "accepted": (n - len(corrected)) / n,
            }
        )
    df = pd.DataFrame(rows).set_index("policy").sort_values("harmed")

    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(df) + 1.6))
    left = np.zeros(len(df))
    for label, colour in [
        ("accepted", "#D9D9D9"), ("helped", "#55A868"),
        ("neutral", "#BFBFBF"), ("harmed", "#C44E52"),
    ]:
        ax.barh(df.index, df[label], left=left, label=label, color=colour, height=0.62)
        left = left + df[label].to_numpy()
    ax.set_xlabel("Fraction of test instances")
    ax.set_xlim(0, 1)
    ax.set_title("What each policy's corrections actually did")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    return fig


def fig_stratum_gains(scores: pd.DataFrame, reference: str = "never_correct",
                      metric: str = "chosen_metric") -> plt.Figure:
    """Per-family change vs the reference arm -- the H2 figure.

    If self-correction earns its cost anywhere, it should be in the shift, anomaly and
    regime-change families. Plotting every family together is what makes that claim
    falsifiable rather than selective.
    """
    wide = scores.pivot_table(index=["instance_id", "family"], columns="policy",
                              values=metric)
    if reference not in wide.columns:
        raise KeyError(f"reference policy {reference!r} missing")

    policies = [c for c in wide.columns if c != reference]
    families = sorted({f for _, f in wide.index})
    fig, ax = plt.subplots(figsize=(max(6.4, 0.7 * len(families)), 3.8))

    width = 0.8 / max(1, len(policies))
    x = np.arange(len(families))
    for i, policy in enumerate(policies):
        gains = []
        for family in families:
            sub = wide.xs(family, level="family")[[policy, reference]].dropna()
            gains.append(
                100.0 * (sub[policy].mean() - sub[reference].mean()) / sub[reference].mean()
                if len(sub) and sub[reference].mean() else np.nan
            )
        ax.bar(x + i * width - 0.4 + width / 2, gains, width,
               label=policy, color=PALETTE[i % len(PALETTE)])

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_ylabel("Change in MASE vs. never correcting (%)")
    ax.set_title("Where correction helps and where it hurts (negative is better)")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    return fig


# --------------------------------------------------------------------------- #
# Reliability, calibration, selective prediction
# --------------------------------------------------------------------------- #


def fig_risk_coverage(scores: pd.DataFrame, metric: str = "chosen_metric",
                      confidence_col: str = "confidence") -> plt.Figure:
    """Risk-coverage curves: does the confidence signal rank failures correctly?"""
    from evaluation.metrics import risk_coverage_curve

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, (policy, group) in enumerate(scores.groupby("policy")):
        conf = pd.to_numeric(group.get(confidence_col), errors="coerce")
        losses = pd.to_numeric(group[metric], errors="coerce")
        ok = conf.notna() & losses.notna()
        if ok.sum() < 10:
            continue
        cov, risk = risk_coverage_curve(losses[ok].to_numpy(), conf[ok].to_numpy())
        ax.plot(cov, risk, label=policy, color=PALETTE[i % len(PALETTE)], linewidth=1.4)

    ax.set_xlabel("Coverage (fraction of forecasts retained)")
    ax.set_ylabel("Mean MASE among retained")
    ax.set_title("Selective prediction: risk vs. coverage")
    ax.legend(frameon=False, fontsize=7)
    return fig


def fig_calibration(risk: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
                    ) -> plt.Figure:
    """Reliability diagram for the failure-risk estimator.

    Equal-mass bins: with a skewed risk distribution, equal-width bins leave most bins
    nearly empty and the diagram becomes a picture of sampling noise.
    """
    from evaluation.metrics import expected_calibration_error

    risk = np.asarray(risk, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(risk) & np.isfinite(outcomes)
    risk, outcomes = risk[ok], outcomes[ok]

    edges = np.unique(np.quantile(risk, np.linspace(0, 1, n_bins + 1)))
    centres, observed, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (risk >= lo) & (risk <= hi)
        if m.sum() < 5:
            continue
        centres.append(risk[m].mean())
        observed.append(outcomes[m].mean())
        counts.append(int(m.sum()))

    ece = expected_calibration_error(risk, outcomes, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0, label="perfect")
    ax.plot(centres, observed, "o-", color=PALETTE[0], linewidth=1.4,
            markersize=5, label="estimator")
    for x, y, n in zip(centres, observed, counts):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 7),
                    fontsize=6, ha="center", color=NEUTRAL)
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title(f"Reliability estimator calibration (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return fig


def fig_cost_accuracy(scores: pd.DataFrame, metric: str = "chosen_metric",
                      cost_col: str = "compute_cost") -> plt.Figure:
    """Accuracy against compute -- the H4 figure.

    The interesting region is the lower-left: equal or better accuracy for less
    compute. A method that only wins by spending more is not a better decision rule.
    """
    rows = []
    for policy, group in scores.groupby("policy"):
        rows.append(
            {
                "policy": policy,
                "metric": float(pd.to_numeric(group[metric], errors="coerce").mean()),
                "cost": float(pd.to_numeric(group.get(cost_col), errors="coerce").fillna(0).mean()),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, row in df.iterrows():
        colour = HIGHLIGHT if str(row["policy"]).startswith("rgc") else PALETTE[i % len(PALETTE)]
        ax.scatter(row["cost"], row["metric"], s=70, color=colour, zorder=3)
        ax.annotate(row["policy"], (row["cost"], row["metric"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("Mean compute cost per instance (action evaluations)")
    ax.set_ylabel("Mean MASE")
    ax.set_title("Accuracy vs. compute: is adaptivity cheaper?")
    return fig


def fig_feature_importance(importance: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Which reliability signals drive the correction decision."""
    df = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(df) + 1.4))
    ax.barh(df["feature"], df["importance"], xerr=df.get("std"),
            color=PALETTE[0], alpha=0.85, height=0.7,
            error_kw={"elinewidth": 0.8, "ecolor": "black"})
    ax.set_xlabel("Permutation importance (drop in AUROC)")
    ax.set_title("Signals driving the correction gate")
    return fig


# --------------------------------------------------------------------------- #
# Robustness and diagnostics
# --------------------------------------------------------------------------- #


def fig_robustness(robustness: pd.DataFrame, metric: str = "chosen_metric"
                   ) -> plt.Figure:
    """Degradation curves across perturbation severity, one panel per axis."""
    axes_names = sorted(robustness["perturbation"].unique())
    n = len(axes_names)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows),
                            squeeze=False, sharex=True)

    for idx, name in enumerate(axes_names):
        ax = axs[idx // ncols][idx % ncols]
        sub = robustness[robustness["perturbation"] == name]
        for i, (policy, group) in enumerate(sub.groupby("policy")):
            agg = group.groupby("severity")[metric].mean()
            ax.plot(agg.index, agg.to_numpy(), "o-", markersize=3.5,
                    label=policy, color=PALETTE[i % len(PALETTE)], linewidth=1.3)
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("severity")
        if idx % ncols == 0:
            ax.set_ylabel("MASE")
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Robustness under controlled perturbations", y=1.01)
    return fig


def fig_oracle_gap(labels: pd.DataFrame, realizable: pd.DataFrame,
                   policy_means: dict[str, float]) -> plt.Figure:
    """Accept vs. achievable oracle vs. unrestricted oracle.

    Makes the selection bias visible. The unrestricted oracle picks the best of ten
    noisy outcomes after the fact and is therefore an unattainable bound; the
    realizable oracle chooses on validation and is attainable. Reporting only the
    former would overstate the available headroom several-fold.
    """
    merged = labels.merge(realizable, on="instance_id", how="inner")
    bars = {
        "accept (no correction)": merged["accept_metric"].mean(),
        **policy_means,
        "realizable oracle\n(chosen on validation)": merged["realizable_metric"].mean(),
        "unrestricted oracle\n(chosen on test)": merged["oracle_metric"].mean(),
    }
    names = list(bars)
    values = [bars[n] for n in names]
    colours = [NEUTRAL] * len(names)
    colours[-1] = "#D9D9D9"
    colours[-2] = "#BFBFBF"
    for i, n in enumerate(names):
        if n.startswith("rgc"):
            colours[i] = HIGHLIGHT

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(len(names)), values, color=colours, width=0.62)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean MASE")
    ax.set_title("Achievable vs. apparent correction headroom")
    ax.axhline(bars["accept (no correction)"], color="black", linestyle="--",
               linewidth=0.9)
    return fig


def fig_example_forecast(series, instance, results: dict, diagnostics: dict | None = None
                         ) -> plt.Figure:
    """One instance: history, truth, competing forecasts, intervals and diagnostics."""
    history = instance.history(series)
    actual = instance.target_slice(series)
    context = min(len(history), 120)
    t_hist = np.arange(-context, 0)
    t_fut = np.arange(0, len(actual))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(t_hist, history[-context:], color="black", linewidth=1.1, label="history")
    ax.plot(t_fut, actual, color="black", linewidth=1.8, linestyle="--", label="actual")

    for i, (name, result) in enumerate(results.items()):
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(t_fut, result.point, linewidth=1.4, color=colour, label=name)
        if 0.8 in result.lower:
            ax.fill_between(t_fut, result.lower[0.8], result.upper[0.8],
                            color=colour, alpha=0.15, linewidth=0)

    if diagnostics:
        for cp in (diagnostics.get("changepoint", {}).get("changepoints") or []):
            offset = cp - len(history)
            if -context <= offset < 0:
                ax.axvline(offset, color=HIGHLIGHT, linestyle=":", linewidth=1.1)
        for idx in (diagnostics.get("anomaly", {}).get("anomaly_indices") or [])[:40]:
            offset = idx - len(history)
            if -context <= offset < 0:
                ax.plot(offset, history[idx], "x", color=HIGHLIGHT, markersize=5)

    ax.axvline(0, color=NEUTRAL, linewidth=0.9)
    ax.set_xlabel("Steps relative to forecast origin")
    ax.set_ylabel("Value")
    ax.set_title(f"{series.series_id} (h={instance.horizon})")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return fig
