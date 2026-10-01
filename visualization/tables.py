"""LaTeX table generation.

Tables are emitted as ``\\input``-able fragments containing only a ``tabular``
environment, so the manuscript controls placement, captions and labels while the
numbers come exclusively from results files. Nothing in ``paper/`` is typed by hand,
which is what makes "the paper cannot disagree with the experiments" a structural
property rather than a promise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "write_tabular", "table_main_results", "table_ablations", "table_hypotheses",
    "table_dataset_summary", "table_decision_quality", "table_cost",
    "table_diagnosis_accuracy", "table_headline",
    "table_regret_decomposition", "table_triage", "table_llm_arms", "table_coverage",
    "table_discrimination", "table_action_choice",
    "table_confidence_calibration", "table_robustness", "table_fixed_base",
    "table_regret_decomposition_realizable",
    "escape_latex", "fmt", "fmt_p",
]

_LATEX_SPECIALS = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape_latex(text: object) -> str:
    out = str(text)
    for char, replacement in _LATEX_SPECIALS.items():
        out = out.replace(char, replacement)
    return out


def fmt(value: object, digits: int = 3, pct: bool = False) -> str:
    """Format a number for a table, rendering missing values as an em dash."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return escape_latex(value)
    if not np.isfinite(f):
        return "---"
    return f"{f * 100:.1f}\\%" if pct else f"{f:.{digits}f}"


def fmt_p(value: object) -> str:
    """Format a p-value, switching to scientific notation when it would round to zero.

    Fixed-decimal formatting turns p = 8.9e-13 into ``0.0000``, which reads as a
    rounding artifact rather than as strong evidence and is exactly the kind of detail
    a reviewer notices.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return escape_latex(value)
    if not np.isfinite(f):
        return "---"
    if f < 1e-4:
        mantissa, exponent = f"{f:.1e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{f:.4f}"


def write_tabular(
    df: pd.DataFrame,
    path: Path,
    column_format: str | None = None,
    bold_min: str | None = None,
    escape: bool = True,
) -> Path:
    """Write a DataFrame as a bare LaTeX ``tabular`` fragment.

    ``bold_min`` bolds the row achieving the minimum of that column, which is how the
    best method is marked without hand-editing the table later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body = df.copy()
    best_idx = None
    if bold_min and bold_min in body.columns:
        numeric = pd.to_numeric(body[bold_min], errors="coerce")
        if numeric.notna().any():
            best_idx = numeric.idxmin()

    cells = body.astype(str)
    if escape:
        cells = cells.applymap(escape_latex)
    if best_idx is not None and best_idx in cells.index:
        cells.loc[best_idx] = cells.loc[best_idx].map(lambda v: f"\\textbf{{{v}}}")

    fmt_spec = column_format or ("l" + "r" * (len(df.columns) - 1))
    lines = [
        f"\\begin{{tabular}}{{{fmt_spec}}}",
        "\\toprule",
        " & ".join(escape_latex(c) if escape else str(c) for c in df.columns) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in cells.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Result tables
# --------------------------------------------------------------------------- #


def table_main_results(comparison: pd.DataFrame, scores: pd.DataFrame,
                       path: Path) -> Path:
    """Main comparison: metric, change vs reference, CI, adjusted p, effect size."""
    per_policy = scores.groupby("policy").agg(
        mase=("chosen_metric", "mean"),
        regret=("regret", "mean"),
        correction_rate=("corrected", "mean"),
    )
    rows = []
    for _, row in comparison.iterrows():
        policy = row["policy"]
        extra = per_policy.loc[policy] if policy in per_policy.index else {}
        rows.append(
            {
                "Policy": policy.replace("_", " "),
                "MASE": fmt(row["mean_metric"], 4),
                r"$\Delta$ vs. never (\%)": fmt(row["relative_change_pct"], 2),
                "95\\% CI": f"[{fmt(row['ci_low'], 3)}, {fmt(row['ci_high'], 3)}]",
                "$p_{\\text{adj}}$": fmt_p(row["p_adjusted"]),
                "Cliff's $\\delta$": fmt(row["cliffs_delta"], 3),
                "Corr. rate": fmt(extra.get("correction_rate", np.nan), 2),
                "Regret": fmt(extra.get("regret", np.nan), 3),
            }
        )
    return write_tabular(pd.DataFrame(rows), path, escape=False)


def table_decision_quality(scores: pd.DataFrame, path: Path) -> Path:
    """Decision-level quality: the metrics only a counterfactual benchmark can report."""
    rows = []
    for policy, group in scores.groupby("policy"):
        corrected = group[group["corrected"]]
        tp = int(((group["corrected"]) & (group["should_correct"])).sum())
        fp = int(((group["corrected"]) & (~group["should_correct"])).sum())
        fn = int(((~group["corrected"]) & (group["should_correct"])).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and np.isfinite(precision) and np.isfinite(recall)
            else np.nan
        )
        rows.append(
            {
                "Policy": policy.replace("_", " "),
                "Corr. rate": fmt(group["corrected"].mean(), 2),
                "Precision": fmt(precision, 3),
                "Recall": fmt(recall, 3),
                "F1": fmt(f1, 3),
                "Helped": fmt(corrected["improved"].mean() if len(corrected) else np.nan, 3),
                "Harmed": fmt(corrected["harmed"].mean() if len(corrected) else np.nan, 3),
                "Regret": fmt(group["regret"].mean(), 3),
            }
        )
    df = pd.DataFrame(rows).sort_values("Regret")
    return write_tabular(df, path, escape=False)


def table_cost(scores: pd.DataFrame, path: Path) -> Path:
    """Accuracy against computational cost -- the evidence for H4."""
    rows = []
    for policy, group in scores.groupby("policy"):
        rows.append(
            {
                "Policy": policy.replace("_", " "),
                "MASE": fmt(group["chosen_metric"].mean(), 4),
                "Action evals": fmt(
                    pd.to_numeric(group.get("compute_cost"), errors="coerce").mean(), 2
                ),
                "LLM calls": fmt(
                    pd.to_numeric(group.get("n_llm_calls"), errors="coerce").fillna(0).mean(), 2
                ),
            }
        )
    return write_tabular(pd.DataFrame(rows).sort_values("MASE"), path, escape=False)


def table_ablations(ablation_results: pd.DataFrame, path: Path,
                    full_name: str = "full") -> Path:
    """Ablation table: the change caused by removing each component."""
    df = ablation_results.copy()
    baseline = df.loc[df["ablation"] == full_name, "mase"]
    base_value = float(baseline.iloc[0]) if len(baseline) else np.nan

    rows = []
    for _, row in df.iterrows():
        delta = (
            100.0 * (row["mase"] - base_value) / base_value
            if np.isfinite(base_value) and base_value else np.nan
        )
        # Series-level interval when available (run_ablations.py writes it); the bare
        # instance-level delta on heavy-tailed errors is what deviation D7 corrected.
        if "series_delta_pct" in row and np.isfinite(float(row.get("series_delta_pct", np.nan))):
            star = r"$^\dagger$" if row.get("series_significant") else ""
            delta_cell = (fmt(row["series_delta_pct"], 2) + star + "~["
                          + fmt(row["series_ci_lo"], 2) + ", "
                          + fmt(row["series_ci_hi"], 2) + "]")
        else:
            delta_cell = fmt(delta, 2)
        rows.append(
            {
                "Configuration": escape_latex(str(row["ablation"]).replace("_", " ")),
                "MASE": fmt(row["mase"], 4),
                r"$\Delta$ vs. full (\%) [95\% CI]": delta_cell,
                "Corr. rate": fmt(row.get("correction_rate"), 2),
                "Cost": fmt(row.get("compute_cost"), 2),
            }
        )
    return write_tabular(pd.DataFrame(rows), path, column_format="lrrrr", escape=False)


def table_hypotheses(results: list[dict], path: Path) -> Path:
    """Pre-registered hypotheses and their outcomes.

    Every hypothesis appears with its verdict, including the ones that failed. This
    table is the reason the paper cannot quietly drop an inconvenient prediction.
    """
    # The frame is written with escape=False so the p-value header can carry math,
    # which means every free-text cell has to be escaped by hand. A policy name like
    # random_correct otherwise reaches LaTeX as a subscript and fails the build.
    rows = [
        {
            "ID": r["id"],
            "Hypothesis": escape_latex(r["statement"]),
            "Test": escape_latex(r["test"]),
            "Result": fmt(r.get("statistic"), 3),
            "$p_{\\text{adj}}$": fmt(r.get("p_adjusted"), 3),
            "Verdict": escape_latex(r["verdict"]),
            "Qualification": escape_latex(r.get("qualifier", "")),
        }
        for r in results
    ]
    return write_tabular(
        pd.DataFrame(rows), path,
        column_format="lp{3.4cm}p{2.9cm}rrlp{3.6cm}", escape=False
    )


def table_headline(headline: pd.DataFrame, path: Path) -> Path:
    """The dissociation table: point and probabilistic accuracy side by side.

    Both panels cover the same instances, action space and policies; only the scored
    quantity differs. Presenting them together is the point -- either alone supports the
    opposite conclusion.

    The reported effect is the **series-level** paired difference with its bootstrap
    interval, because that is the confirmatory statistic (``ANALYSIS_PLAN.md`` S7 and
    deviation D7). The win rate sits beside it deliberately: on heavy-tailed error data a
    policy can improve the mean while winning on under half of series, and a reader who
    sees only one of the two numbers will draw a conclusion the data does not support.
    """
    order = ["never_correct", "rgc_calibrated", "always_ensemble", "always_correct",
             "best_fixed", "random_correct"]
    label = {"mase": "Point (MASE)", "crps_scaled": "Probabilistic (CRPS)"}
    has_series = "pct_diff" in headline.columns
    rows = []
    for metric in ["mase", "crps_scaled"]:
        block = headline[headline["metric"] == metric]
        if block.empty:
            continue
        for policy in order:
            r = block[block["policy"] == policy]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {
                "Metric": label.get(metric, metric),
                "Policy": str(policy).replace("_", " "),
                "Acts": fmt(r["correction_rate"], 2),
            }
            if has_series:
                # A dagger marks the interval excluding zero -- the decision rule the
                # pre-registration fixes. The p-value is shown too, but it is not what
                # the claims rest on.
                star = r"$^\dagger$" if bool(r.get("significant")) else ""
                row.update({
                    "Series mean": fmt(r["series_mean"], 4),
                    r"$\Delta$": fmt(r["pct_diff"], 2) + r"\%" + star,
                    r"95\% CI": (
                        "[" + fmt(r["pct_ci_lo"], 2) + ", "
                        + fmt(r["pct_ci_hi"], 2) + "]"
                    ),
                    "Win rate": fmt(100.0 * r["win_rate"], 1) + r"\%",
                    "$p$": fmt_p(r["wilcoxon_p"]),
                })
            else:
                row.update({
                    "Mean": fmt(r["mean"], 4),
                    r"Mean $\Delta$": fmt(r["mean_pct"], 2) + r"\%",
                    "Geo. mean": fmt(r["geometric_mean"], 4),
                    r"Geo. $\Delta$": fmt(r["geo_pct"], 2) + r"\%",
                    "Median": fmt(r["median"], 4),
                })
            rows.append(row)
    n_cols = len(rows[0]) if rows else 9
    return write_tabular(pd.DataFrame(rows), path,
                         column_format="ll" + "r" * (n_cols - 2), escape=False)


def table_coverage(outcomes: pd.DataFrame, path: Path) -> Path:
    """Per-action interval coverage against CRPS -- the evidence that CRPS is not gamed.

    The argument this table settles: a probabilistic improvement could in principle be
    manufactured by widening intervals until coverage looks good. ``abstain`` does
    exactly that and is included precisely so a reader can see what it costs on a proper
    scoring rule. Everything is computed on the instances where the action applies, and
    CRPS is expressed relative to accepting on the same instances so the comparison is
    paired rather than across different instance sets.
    """
    test = outcomes[(outcomes["split"] == "test")
                    & (outcomes["applicable"].astype(bool))]
    accept = test[test["action"] == "accept"].set_index("instance_id")
    rows = []
    for action in ["accept", "ensemble", "recalibrate_intervals", "abstain",
                   "damp_trend"]:
        group = test[test["action"] == action].set_index("instance_id")
        if group.empty:
            continue
        common = accept.index.intersection(group.index)
        base = float(accept.loc[common, "crps_scaled"].mean())
        crps = float(group.loc[common, "crps_scaled"].mean())
        c80 = float(group["coverage_80"].mean())
        c95 = float(group["coverage_95"].mean())
        rows.append({
            "Action": action.replace("_", " "),
            "Cov.\\ @80\\%": fmt(c80, 3),
            "Cov.\\ @95\\%": fmt(c95, 3),
            "Calib.\\ err.": fmt(abs(c80 - 0.80) + abs(c95 - 0.95), 3),
            "Width @80\\%": fmt(float(group["width_80"].mean()), 1),
            "CRPS": fmt(crps, 4),
            r"CRPS $\Delta$": fmt(100.0 * (crps / base - 1.0), 1) + r"\%",
        })
    return write_tabular(pd.DataFrame(rows), path, column_format="lrrrrrr",
                         escape=False)


def table_discrimination(families, path: Path) -> Path:
    """Whether a verbal self-critique's yes/no carries information about correcting.

    ``families`` is a list of ``(label, json_dict)`` from
    ``scripts/critique_discrimination.py``. The column that matters is the AUROC gap
    between the critique arm and the same model deciding without being asked to reflect:
    that gap, not the absolute AUROC, is what the reflection step contributes.
    """
    rows = []
    for label, d in families:
        if not d:
            continue
        by_arm = {r["arm"].split(" (")[0]: r for r in d.get("rows", [])}
        crit = by_arm.get("llm_critique_gate", {})
        single = by_arm.get("llm_single", {})
        rand = next((r for r in d.get("rows", []) if r["arm"].startswith("random")), {})
        rows.append({
            "Model": label,
            "$n$": fmt(d.get("n"), 0),
            "Base rate": fmt(100.0 * d.get("base_rate", float("nan")), 1) + r"\%",
            "AUROC (critique)": fmt(d.get("auroc_critique"), 3),
            "AUROC (no critique)": fmt(d.get("auroc_single"), 3),
            r"$\Delta$ AUROC": fmt(
                d.get("auroc_critique", float("nan"))
                - d.get("auroc_single", float("nan")), 3),
            # The matched contrast: an arithmetic signal judging the same decision,
            # on the same label and the same instances.
            "AUROC (arithmetic)": fmt(d.get("auroc_arithmetic"), 3),
            "Precision": fmt(100.0 * crit.get("precision_vs_should_correct",
                                               float("nan")), 1) + r"\%",
            "Random": fmt(100.0 * rand.get("precision_vs_should_correct",
                                           float("nan")), 1) + r"\%",
            "Overlap": fmt(100.0 * d.get("jaccard", float("nan")), 0) + r"\%",
        })
    if not rows:
        return write_tabular(pd.DataFrame(), path, escape=False)
    return write_tabular(pd.DataFrame(rows), path,
                         column_format="l" + "r" * (len(rows[0]) - 1), escape=False)


def table_action_choice(rows, path: Path) -> Path:
    """Given the decision to correct, how good is the repair the model picks?

    Produced by ``scripts/action_choice.py``. Quantities are rows and arms are columns
    because the argument is vertical: the three middle rows are the best action in the
    menu, the action the model chose, and a uniform draw from that same menu on the same
    instance, and the whole finding is visible in the gap between the first two.

    Everything is the same metric on the same instances, so the rows are directly
    comparable. The attainable oracle is a maximum taken over the menu after the fact and
    is therefore an optimistic bound --- it says how much headroom existed, not how much
    any selector could have taken. The uniform draw is the unbiased anchor.
    """
    arms = [("llm_single", "LLM single"),
            ("llm_critique_gate", "LLM critique gate"),
            ("rgc_llm_router", "RGC + LLM router")]
    by_arm = {r["arm"]: r for r in (rows or [])}
    present = [(k, lab) for k, lab in arms if k in by_arm]
    if not present:
        return write_tabular(pd.DataFrame(), path, escape=False)

    def ci(r, point, lo, hi, digits=2):
        if r.get(point) is None or not np.isfinite(float(r.get(point, np.nan))):
            return "---"
        return (fmt(r[point], digits) + r"\%~[" + fmt(r.get(lo), digits) + ", "
                + fmt(r.get(hi), digits) + "]")

    spec = [
        ("Instances intervened on", lambda r: fmt(r["n_instances"], 0)),
        ("Best in menu (test-selected ceiling)", lambda r: fmt(r["best_in_menu"], 4)),
        (r"\textbf{Model's chosen action}", lambda r: r"\textbf{" + fmt(r["chosen"], 4) + "}"),
        ("Uniform draw from same menu", lambda r: fmt(r["uniform_random"], 4)),
        ("Worst action in menu", lambda r: fmt(r["worst_in_menu"], 4)),
        (r"\midrule Chosen vs.\ uniform draw",
         lambda r: ci(r, "chosen_vs_random_pct", "ci_lo", "ci_hi")),
    ]
    table = pd.DataFrame(
        [{"": name, **{lab: get(by_arm[k]) for k, lab in present}} for name, get in spec])
    return write_tabular(table, path,
                         column_format="l" + "r" * len(present), escape=False)


def table_confidence_calibration(data, path: Path) -> Path:
    """H5: the agent's stated confidence against a calibrated estimate of the same
    event, on the same instances and the same label.

    Produced by ``scripts/confidence_calibration.py``. The constant predictor is in
    the table on purpose: it attains a perfect ECE while ranking nothing, which is
    the standing argument for not reading ECE on its own.
    """
    rows = data.get("rows") or []
    if not rows:
        return write_tabular(pd.DataFrame(), path, escape=False)
    table = pd.DataFrame([{
        "Predictor": escape_latex(r["predictor"]),
        "ECE": fmt(r.get("ece"), 4),
        "Brier": fmt(r.get("brier"), 4),
        "AURC": fmt(r.get("aurc"), 4),
        "Mean $p$": fmt(r.get("mean_prediction"), 3),
    } for r in rows])
    return write_tabular(table, path, column_format="lrrrr", escape=False)


def table_robustness(data, path: Path, max_rows: int = 16) -> Path:
    """Correction against never correcting, per perturbation axis and severity.

    Produced by ``scripts/robustness_summary.py``. Only the strongest severity of
    each axis is shown, because that is where a correction stage is supposed to earn
    its place and because showing all four severities would fill a page to make one
    point. The full sweep is in the released results.
    """
    rows = data.get("rows") or []
    if not rows:
        return write_tabular(pd.DataFrame(), path, escape=False)
    frame = pd.DataFrame(rows)
    worst = frame.groupby("perturbation")["severity"].transform("max")
    frame = frame[frame["severity"] == worst]
    out = pd.DataFrame([{
        "Perturbation": escape_latex(str(r["perturbation"]).replace("_", " ")),
        "Severity": fmt(r["severity"], 2),
        "Policy": escape_latex(str(r["policy"]).replace("_", " ")),
        "Series": fmt(r["n_series"], 0),
        r"$\Delta$ vs never": fmt(r["pct_diff"], 2) + r"\%"
                             + (r"$^\dagger$" if r.get("significant") else ""),
        r"95\% CI": "[" + fmt(r["pct_ci_lo"], 2) + ", " + fmt(r["pct_ci_hi"], 2) + "]",
        "Win rate": fmt(100.0 * r["win_rate"], 1) + r"\%",
    } for _, r in frame.iterrows()][:max_rows])
    return write_tabular(out, path, column_format="lllrrrr", escape=False)


def table_fixed_base(data, path: Path) -> Path:
    """Correction under an adaptive base beside the same policies under a fixed base.

    Produced by ``scripts/fixed_base_comparison.py``. Each column is the series-level
    difference from never correcting *within its own run*; the two runs have different
    base forecasts and therefore different references, so levels are never shown and
    never compared across columns. Only policies present in both runs are listed.
    """
    rows = data.get("rows") or []
    order = ["always_correct", "best_fixed", "threshold_single", "random_correct", "rgc"]
    by = {r["policy"]: r for r in rows}

    def cell(r, suffix):
        pct = r.get(f"pct_diff_{suffix}")
        if pct is None or not np.isfinite(float(pct)):
            return "---"
        star = r"$^\dagger$" if r.get(f"significant_{suffix}") else ""
        return (fmt(pct, 2) + r"\%" + star + "~[" + fmt(r[f"pct_ci_lo_{suffix}"], 2)
                + ", " + fmt(r[f"pct_ci_hi_{suffix}"], 2) + "]")

    out = []
    for pol in order:
        r = by.get(pol)
        if not r or r.get("pct_diff_adaptive") is None or r.get("pct_diff_fixed") is None:
            continue
        out.append({"Policy": escape_latex(pol.replace("_", " ")),
                    "Adaptive (backtest) base": cell(r, "adaptive"),
                    "Fixed ETS base": cell(r, "fixed")})
    if not out:
        return write_tabular(pd.DataFrame(), path, escape=False)
    return write_tabular(pd.DataFrame(out), path, column_format="lrr", escape=False)


def table_regret_decomposition_realizable(data, path: Path) -> Path:
    """The decomposition twice: selected on test (a ceiling) and on validation.

    Produced by ``scripts/regret_decomposition_realizable.py``. The left column is what a
    decomposition built from test-selected minima reports; the right column is the same
    quantity with every choice made on validation origins. The gap between them is the
    winner's curse, and it is the reason the right column is the one to read.
    """
    rows = []
    spec = [
        ("Canonical fix vs accept", "stage:canonical_fix", "stage:canonical_fix"),
        ("Best of matched candidates vs accept", "stage:test_matched", "stage:validated_matched"),
        ("Best of all actions vs accept (headroom)", "stage:test_any", "stage:validated_any"),
        ("First-choice loss (pp)", "ceiling first-choice loss", "realizable first-choice loss"),
        ("Restriction loss (pp)", "ceiling restriction loss", "realizable restriction loss"),
    ]

    def cell(v):
        if not v:
            return "---"
        star = r"$^\dagger$" if v.get("excludes_zero") else ""
        return (fmt(v["point"], 2) + star + "~[" + fmt(v["ci_lo"], 2) + ", "
                + fmt(v["ci_hi"], 2) + "]")

    for metric_key, mlabel in (("mase", "MASE"), ("crps_scaled", "CRPS")):
        stats = (data.get(metric_key) or {}).get("stats") or {}
        for label, test_key, val_key in spec:
            rows.append({"Metric": mlabel, "Quantity": label,
                         "Selected on test (ceiling)": cell(stats.get(test_key)),
                         "Selected on validation": cell(stats.get(val_key))})
    if not rows:
        return write_tabular(pd.DataFrame(), path, escape=False)
    return write_tabular(pd.DataFrame(rows), path, column_format="llrr", escape=False)


def table_llm_arms(families, path: Path) -> Path:
    """LLM correction arms against the same counterfactual outcomes.

    Produced by ``scripts/llm_arms.py``. These are the arms the manuscript's critique is
    about -- a model picking a repair in one pass, and a model deciding by verbal
    self-critique whether to repair at all.

    ``families`` is a list of ``(label, series_table, cost_table)``, one per model
    family. Reporting more than one in a single table is the point: one model failing to
    beat never-correcting invites "you picked a weak model", and two unrelated families
    failing the same way does not. The non-LLM rows repeat within each family block
    because each family runs on its own subsample, so their reference values differ and
    must not be compared across blocks.
    """
    order = ["never_correct", "always_ensemble", "rgc_calibrated", "llm_single",
             "llm_critique_gate", "rgc_llm_router", "always_correct"]
    label = {"mase": "Point (MASE)", "crps_scaled": "Probabilistic (CRPS)"}
    multi = len(families) > 1
    rows = []
    for family, series, cost in families:
        if series is None or getattr(series, "empty", True):
            continue
        calls = (cost.groupby("policy")["llm_calls"].sum().to_dict()
                 if cost is not None and not cost.empty else {})
        for metric in ["mase", "crps_scaled"]:
            block = series[series["metric"] == metric]
            if block.empty:
                continue
            for policy in order:
                r = block[block["policy"] == policy]
                if r.empty:
                    continue
                r = r.iloc[0]
                star = r"$^\dagger$" if bool(r.get("significant")) else ""
                row = {"Metric": label.get(metric, metric)}
                if multi:
                    row["Model"] = family
                row.update({
                    "Policy": str(policy).replace("_", " "),
                    "Series": fmt(r.get("n_series"), 0),
                    "Series mean": fmt(r["series_mean"], 4),
                    r"$\Delta$": fmt(r["pct_diff"], 2) + r"\%" + star,
                    r"95\% CI": ("[" + fmt(r["pct_ci_lo"], 2) + ", "
                                 + fmt(r["pct_ci_hi"], 2) + "]"),
                    "Win rate": fmt(100.0 * r["win_rate"], 1) + r"\%",
                    "LLM calls": fmt(calls.get(policy, 0), 0),
                })
                rows.append(row)
    if not rows:
        return write_tabular(pd.DataFrame(), path, escape=False)
    lead = "lll" if multi else "ll"
    n_cols = len(rows[0])
    return write_tabular(pd.DataFrame(rows), path,
                         column_format=lead + "r" * (n_cols - len(lead)), escape=False)


def table_regret_decomposition(table: pd.DataFrame, path: Path) -> Path:
    """Where the achievable improvement goes, under ground-truth diagnosis.

    Produced by ``scripts/regret_decomposition.py``. Four stages per metric, from
    accepting through the unrestricted oracle, isolating how much of the total
    achievable improvement a fixed diagnose-then-apply-canonical-fix policy forfeits
    even when the diagnosis itself is perfect.
    """
    stage_label = {
        "accept": "Accept",
        "first_matched_action": "Canonical fix (true diagnosis)",
        "best_of_matched_candidates": "Best of matched candidates (oracle)",
        "best_of_all_actions": "Unrestricted oracle",
    }
    metric_label = {"mase": "MASE", "crps_scaled": "CRPS"}
    order = ["accept", "first_matched_action", "best_of_matched_candidates",
             "best_of_all_actions"]
    rows = []
    for metric in ["mase", "crps_scaled"]:
        block = table[table["metric"] == metric].set_index("stage")
        for stage in order:
            if stage not in block.index:
                continue
            r = block.loc[stage]
            rows.append(
                {
                    "Metric": metric_label.get(metric, metric),
                    "Stage": stage_label.get(stage, stage),
                    "Mean": fmt(r["mean"], 4),
                    r"Mean $\Delta$": fmt(r["mean_pct"], 2) + r"\%",
                    "Geo. mean": fmt(r["geometric_mean"], 4),
                    r"Geo. $\Delta$": fmt(r["geo_pct"], 2) + r"\%",
                }
            )
    return write_tabular(pd.DataFrame(rows), path, column_format="llrrrr", escape=False)


def table_triage(triage: pd.DataFrame, auroc: float, base_rate: float, path: Path
                 ) -> Path:
    """Selective-prediction / triage value of the failure-risk estimator.

    Produced by ``scripts/triage_analysis.py``. Reports precision, recall and lift for
    reviewing the top-k% of forecasts by predicted risk -- the standard way to state
    the operating value of a ranking model to a non-technical stakeholder.
    """
    rows = []
    for _, r in triage.iterrows():
        rows.append(
            {
                "Review budget": fmt(r["review_budget"], 0, pct=True),
                "Reviewed": int(r["n_reviewed"]),
                "Precision": fmt(r["precision"], 3),
                "Recall": fmt(r["recall"], 3),
                "Lift": fmt(r["lift"], 2) + r"$\times$",
            }
        )
    df = pd.DataFrame(rows)
    written = write_tabular(df, path, column_format="lrrrr", escape=False)
    # A one-line caption stub with AUROC/base rate is embedded via the caller's LaTeX
    # caption, not here, since those are scalars rather than table rows.
    return written


def table_diagnosis_accuracy(accuracy: pd.DataFrame, path: Path) -> Path:
    """Failure-mode diagnosis accuracy, synthetic families only.

    Restricted to synthetic data because that is where the true failure mode is known
    by construction. Scoring real data against the agent's own detectors would be
    circular.
    """
    rows = [
        {
            "Policy": str(r["policy"]).replace("_", " "),
            "Family": str(r["family"]).replace("_", " "),
            "n": int(r["n"]),
            "Accuracy": fmt(r["accuracy"], 3),
            "Most common diagnosis": str(r["most_common_diagnosis"]).replace("_", " "),
        }
        for _, r in accuracy.iterrows()
    ]
    return write_tabular(pd.DataFrame(rows), path, column_format="llrrl", escape=False)


def table_dataset_summary(signals: pd.DataFrame, path: Path) -> Path:
    """Benchmark composition, generated from what was actually evaluated."""
    rows = []
    for (source, family), group in signals.groupby(["source", "family"], dropna=False):
        rows.append(
            {
                "Source": str(source),
                "Family / dataset": str(family).replace("_", " "),
                "Series": int(group["series_id"].nunique()),
                "Instances": int(len(group)),
                "Median length": fmt(group["context.history_length"].median(), 0),
                "Ground-truth labels": "yes"
                if bool(group["label_is_ground_truth"].iloc[0]) else "no",
            }
        )
    df = pd.DataFrame(rows).sort_values(["Source", "Family / dataset"])
    return write_tabular(df, path, escape=False)
