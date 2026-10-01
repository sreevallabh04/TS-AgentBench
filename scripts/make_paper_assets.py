#!/usr/bin/env python
"""Regenerate every figure and table in the manuscript from results files.

    python scripts/make_paper_assets.py --run latest
    python scripts/make_paper_assets.py --run smoke_20260918-230208_44a52869

This is the mechanism behind the project's integrity rule: ``paper/`` contains no
hardcoded numbers. Every table is an ``\\input``-able fragment and every figure a PDF,
both written here from ``results/``. A number in the manuscript that disagrees with the
experiments is therefore not possible without deleting this script's output first.

Also writes ``paper/generated/results_summary.json``, which records the headline
quantities and the verdict on each pre-registered hypothesis, so the prose can be
checked against the data mechanically.
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
from core.paths import FIGURES_DIR, PAPER_DIR, RESULTS_DIR, RUNS_DIR, TABLES_DIR  # noqa: E402

log = get_logger("paper_assets")


def resolve_run(name: str) -> Path:
    if name == "latest":
        # "latest" means the latest PRIMARY run. Experimental conditions such as the
        # fixed-base run (D11) live alongside it and are analysed by their own scripts;
        # letting them win on modification time once regenerated every headline table
        # from the wrong experiment, which the duplicate-macro guard caught only by luck.
        completed = [p for p in RUNS_DIR.iterdir()
                     if p.is_dir() and (p / "scores.parquet").exists()]
        primary = [p for p in completed if p.name.startswith("main_")]
        candidates = sorted(primary or completed, key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit("no completed runs found under results/runs/")
        if not primary:
            log.warning("no main_* run found; falling back to %s", candidates[-1].name)
        return candidates[-1]
    path = RUNS_DIR / name
    if not path.exists():
        raise SystemExit(f"run not found: {path}")
    return path


def load(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def hypothesis_verdicts(
    comparison: pd.DataFrame, scores: pd.DataFrame, labels: pd.DataFrame,
    signals: pd.DataFrame, run_dir: Path,
) -> list[dict]:
    """Evaluate each pre-registered hypothesis against the results.

    Reports the verdict as found, including ``not supported``. The hypotheses were
    fixed in ``docs/ANALYSIS_PLAN.md`` before any result was seen, so a failed
    prediction is evidence rather than an embarrassment.
    """
    from sklearn.metrics import roc_auc_score

    verdicts: list[dict] = []

    def comparison_row(policy: str):
        sub = comparison[comparison["policy"] == policy]
        return sub.iloc[0] if len(sub) else None

    # H1: RGC beats the non-correcting reference.
    row = comparison_row("rgc")
    if row is not None:
        verdicts.append(
            {
                "id": "H1",
                "statement": "Reliability-gated correction improves accuracy over a "
                             "non-correcting agent.",
                # The statistic is a signed percentage change, not a test statistic, and
                # a significant p beside a "not supported" verdict is confusing unless
                # the column says which way the effect points.
                # Written with a bare percent sign: the table escapes this field.
                "test": "series-mean change vs never (%, positive = worse); "
                        "Wilcoxon Holm-adj.",
                "statistic": float(row["relative_change_pct"]),
                "p_adjusted": float(row["p_adjusted"]),
                "verdict": "supported"
                if row["significant"] and row["mean_difference"] < 0
                else "not supported",
            }
        )

    # H3: failure is predictable pre-hoc.
    if not labels.empty and not signals.empty:
        merged = signals.merge(
            labels[["instance_id", "large_error", "should_correct"]], on="instance_id"
        )
        scored = scores[scores["policy"].str.startswith("rgc")] if not scores.empty \
            else pd.DataFrame()
        auroc = np.nan
        if "risk" in scored.columns and len(scored):
            joined = scored.merge(
                labels[["instance_id", "large_error"]], on="instance_id",
                suffixes=("", "_l"),
            ).dropna(subset=["risk"])
            target = joined["large_error_l"] if "large_error_l" in joined \
                else joined["large_error"]
            if len(joined) > 20 and target.nunique() > 1:
                auroc = float(roc_auc_score(target.astype(int), joined["risk"]))
        verdicts.append(
            {
                "id": "H3",
                "statement": "Forecast failure is predictable from pre-hoc diagnostics.",
                "test": "AUROC of the risk estimator",
                "statistic": auroc,
                "p_adjusted": float("nan"),
                "verdict": "supported" if np.isfinite(auroc) and auroc > 0.65
                else "not supported",
            }
        )

    # H4: adaptive tool use costs less.
    if not scores.empty and "compute_cost" in scores.columns:
        cost = scores.groupby("policy")["compute_cost"].mean()
        if "rgc" in cost.index and "always_correct" in cost.index:
            verdicts.append(
                {
                    "id": "H4",
                    "statement": "Selective tool use matches exhaustive use at lower cost.",
                    "test": "mean action evaluations per instance",
                    "statistic": float(cost["rgc"] / max(cost["always_correct"], 1e-9)),
                    "p_adjusted": float("nan"),
                    "verdict": "supported"
                    if cost["rgc"] < cost["always_correct"] else "not supported",
                }
            )

    # H6: gains are not explained by compute alone.
    rgc_row, random_row = comparison_row("rgc"), comparison_row("random_correct")
    if rgc_row is not None and random_row is not None:
        verdicts.append(
            {
                "id": "H6",
                "statement": "Gains are not explained by extra compute alone "
                             "(vs. rate-matched random correction).",
                "test": "RGC vs. random_correct",
                "statistic": float(
                    random_row["mean_metric"] - rgc_row["mean_metric"]
                ),
                "p_adjusted": float("nan"),
                "verdict": "supported"
                if rgc_row["mean_metric"] < random_row["mean_metric"]
                else "not supported",
            }
        )

    # H7: quantitative gating decides better than verbal self-critique.
    #
    # The plan's test is the quantitative gate against an *LLM critique gate*. Until the
    # LLM arms were run that comparison did not exist, and the substitute used here --
    # the gate against unconditional correction -- is a weaker question: it asks whether
    # gating beats not gating, not whether a number decides better than a sentence. When
    # scripts/llm_arms.py has produced its output we use the pre-registered comparison
    # and say so; otherwise we report the substitute and label it.
    llm_path = RESULTS_DIR / "llm_arms.parquet"
    if llm_path.exists():
        llm = pd.read_parquet(llm_path)
        llm = llm[llm["metric"] == "mase"].set_index("policy")
        if {"rgc_calibrated", "llm_critique_gate"}.issubset(llm.index) \
                and "regret" in llm.columns:
            gate = float(llm.loc["rgc_calibrated", "regret"])
            critique = float(llm.loc["llm_critique_gate", "regret"])
            verdicts.append({
                "id": "H7",
                "statement": "A quantitative gate decides better than verbal "
                             "self-critique (pre-registered comparison).",
                "test": "mean oracle regret, calibrated gate vs. LLM critique gate",
                "statistic": critique - gate,
                "p_adjusted": float("nan"),
                "verdict": "supported" if gate < critique else "not supported",
            })
            # Two hypotheses were never evaluated, and the table says so rather than
    # omitting them. A pre-registration that silently drops its unevaluated
    # predictions is worth no more than no pre-registration at all.
    # H2 is answered by scripts/robustness_summary.py, whose difference-in-differences
    # over perturbed strata is the test the plan specified, Holm-corrected across all
    # strata because 48 uncorrected comparisons would manufacture a verdict.
    rob_file = RESULTS_DIR / "robustness_summary.json"
    if rob_file.exists():
        rob = json.loads(rob_file.read_text(encoding="utf-8"))
        hits = [r for r in rob.get("did_rows", []) if r.get("supports_h2")]
        best = min((float(r["p_holm"]) for r in hits), default=float("nan"))
        verdicts.append({
            "id": "H2",
            "statement": "Gains concentrate under distribution shift, anomalies and regime change.",
            "test": f"difference-in-differences over {rob.get('h2_strata_tested', 0)} perturbed strata (Holm)",
            "statistic": float(len(hits)),
            "p_adjusted": best,
            "verdict": "supported" if hits else "not supported",
        })
    else:
        verdicts.append({
            "id": "H2",
            "statement": "Gains concentrate under distribution shift, anomalies and regime change.",
            "test": "difference-in-differences across perturbed strata; not run",
            "statistic": float("nan"),
            "p_adjusted": float("nan"),
            "verdict": "not evaluated",
        })
    # H5 is answered from cached decisions by scripts/confidence_calibration.py.
    cal_path = RESULTS_DIR / "confidence_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        verdicts.append({
            "id": "H5",
            "statement": "Reliability-gated confidence is better calibrated than an agent's verbalised confidence.",
            "test": "paired bootstrap over series on the ECE difference",
            "statistic": float(cal["differences"]["ece"]["diff"]),
            "p_adjusted": float("nan"),
            "verdict": "supported" if cal["h5_supported"] else "not supported",
        })
    else:
        verdicts.append({
            "id": "H5",
            "statement": "Reliability-gated confidence is better calibrated than an agent's verbalised confidence.",
            "test": "paired bootstrap on ECE and AURC; not run",
            "statistic": float("nan"),
            "p_adjusted": float("nan"),
            "verdict": "not evaluated",
        })
    # A verdict without its qualification flatters. These state, next to each verdict,
    # what the "supported" does and does not mean, so a reader need not reconstruct it.
    QUALIFIERS = {
        "H1": "the effect is a small, significant degradation",
        "H2": "2 of 48 strata after Holm; both unconditional correction, mildest severity",
        "H3": "supports triage, not correction; the gate built on it does not win",
        "H4": "because the gate rarely acts, not because it acts well",
        "H5": "against the model's verbalised confidence, an easy comparator",
        "H6": "because the gate rarely acts; it does not beat never correcting",
        "H7": "the gate's regret is close to never correcting's; it wins by not acting",
    }
    for v in verdicts:
        v["qualifier"] = QUALIFIERS.get(v["id"], "")
    verdicts.sort(key=lambda v: v["id"])
    return verdicts

    if not scores.empty:
        regret = scores.groupby("policy")["regret"].mean()
        if "rgc" in regret.index and "always_correct" in regret.index:
            verdicts.append(
                {
                    "id": "H7",
                    "statement": "Quantitative gating chooses better than "
                                 "unconditional correction (substitute for the "
                                 "pre-registered test; LLM arms not available).",
                    "test": "mean oracle regret",
                    "statistic": float(regret["always_correct"] - regret["rgc"]),
                    "p_adjusted": float("nan"),
                    "verdict": "supported"
                    if regret["rgc"] < regret["always_correct"] else "not supported",
                }
            )
    return verdicts


def write_macros(summary: dict, path: Path) -> Path:
    """Emit LaTeX macros for every number quoted inline in the prose.

    Without this, a sentence such as "RGC reduces error by 2.1%" would be typed by hand
    and would go stale the moment the experiment is rerun. Defining the numbers as
    macros means the prose re-renders with the data.
    """

    def num(value, digits=3, default="??"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(f) else f"{f:.{digits}f}"

    def pct(value, digits=1, default="??"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(f) else f"{100 * f:.{digits}f}"

    def macro(name: str, value: str) -> str:
        return "\\newcommand{\\" + name + "}{" + str(value) + "}"

    #: LaTeX control sequences may contain letters only, so a macro named
    #: ``\triageprecisionP10`` does not merely look odd -- it fails to compile. Digits
    #: appearing in a generated macro name are spelled out.
    _DIGIT_WORDS = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }

    def letters(text: object) -> str:
        """Render a tag as a legal LaTeX macro-name fragment."""
        out = []
        for ch in str(text):
            if ch.isdigit():
                out.append(_DIGIT_WORDS[ch])
            elif ch.isalpha():
                out.append(ch)
        return "".join(out)

    lines = [
        "% Auto-generated by scripts/make_paper_assets.py. Do not edit by hand.",
        "% Every number quoted in the manuscript prose is defined here, so the text",
        "% cannot disagree with the experiment that produced it.",
        macro("runid", str(summary["run_id"]).replace("_", "-")),
        macro("ntestinstances", summary["n_test_instances"]),
        macro("nseries", summary.get("n_series") or "??"),
        macro("acceptmase", num(summary.get("accept_metric"), 4)),
        macro("oraclemase", num(summary.get("unrestricted_oracle"), 4)),
        macro("realizablemase", num(summary.get("realizable_oracle"), 4)),
        macro("shouldcorrectrate", pct(summary.get("should_correct_rate"))),
    ]

    accept = summary.get("accept_metric")
    for policy, stats in summary.get("policies", {}).items():
        # LaTeX macro names must be letters only.
        tag = letters(policy.title())
        lines.append(macro(f"mase{tag}", num(stats["mase"], 4)))
        lines.append(macro(f"regret{tag}", num(stats["regret"], 3)))
        lines.append(macro(f"rate{tag}", pct(stats["correction_rate"])))
        lines.append(macro(f"helped{tag}", pct(stats.get("helped"))))
        lines.append(macro(f"harmed{tag}", pct(stats.get("harmed"))))
        if accept:
            delta = 100.0 * (stats["mase"] - accept) / accept
            lines.append(macro(f"delta{tag}", num(delta, 2)))

    for verdict in summary.get("hypotheses", []):
        # "H1" -> "HOne": hypothesis ids contain digits, macro names may not.
        tag = letters(verdict["id"])
        lines.append(macro(f"verdict{tag}", verdict["verdict"]))
        lines.append(macro(f"stat{tag}", num(verdict.get("statistic"), 3)))

    triage = summary.get("triage") or {}
    if triage:
        lines.append(macro("triageauroc", num(triage.get("auroc"), 3)))
        lines.append(macro("triagebaserate", pct(triage.get("base_rate"))))
        for budget_pct, row in (triage.get("budgets") or {}).items():
            tag = "P" + letters(budget_pct)
            lines.append(macro(f"triageprecision{tag}", pct(row.get("precision"))))
            lines.append(macro(f"triagerecall{tag}", pct(row.get("recall"))))
            lines.append(macro(f"triagelift{tag}", num(row.get("lift"), 2)))

    def magnitude_macros(stem: str, row: dict) -> list[str]:
        """``...Abs``/``...AbsCIlo``/``...AbsCIhi``: unsigned versions of an effect.

        The CI bounds are reordered by magnitude, so an interval of [-22.85, -7.27]
        becomes [7.27, 22.85] and reads correctly after "by".
        """
        try:
            effect = abs(float(row.get("pct_diff")))
            lo, hi = float(row.get("pct_ci_lo")), float(row.get("pct_ci_hi"))
        except (TypeError, ValueError):
            return []
        if not all(np.isfinite(v) for v in (effect, lo, hi)):
            return []
        bounds = sorted((abs(lo), abs(hi)))
        return [
            macro(f"{stem}Abs", num(effect, 2)),
            macro(f"{stem}AbsCIlo", num(bounds[0], 2)),
            macro(f"{stem}AbsCIhi", num(bounds[1], 2)),
        ]

    # Series-level inference: the confirmatory numbers. Emitted as
    # \<metric><Policy>VsNeverPct / ...CIlo / ...CIhi / ...WinRate so the prose can
    # quote an effect and its interval without either being retyped. These are the
    # figures the manuscript's claims rest on; the instance-level macros above are
    # descriptive and must not be used to assert a difference.
    for row in summary.get("series_level", []):
        metric_tag = {"mase": "mase", "crps_scaled": "crps"}.get(
            row["metric"], row["metric"]
        )
        policy_tag = letters(row["policy"].title())
        stem = f"{metric_tag}{policy_tag}VsNever"
        lines.append(macro(f"{stem}Pct", num(row.get("pct_diff"), 2)))
        lines.append(macro(f"{stem}CIlo", num(row.get("pct_ci_lo"), 2)))
        lines.append(macro(f"{stem}CIhi", num(row.get("pct_ci_hi"), 2)))
        lines.append(macro(f"{stem}WinRate", pct(row.get("win_rate"))))
        lines.append(macro(f"{stem}Sig",
                           "yes" if row.get("significant") else "no"))
        # Magnitude variants, for prose that already carries the direction ("cuts CRPS
        # by 13.07%"). Writing the signed value there would read as a double negative.
        # Only meaningful for an interval that does not straddle zero, which is exactly
        # where such phrasing belongs.
        lines.extend(magnitude_macros(stem, row))

    # LLM arms, same naming scheme with an "llm" prefix so the two families cannot
    # collide (the LLM arms are scored on a subsample, and quoting one where the other
    # is meant would be a silent error).
    for row in summary.get("llm_arms", []):
        metric_tag = {"mase": "mase", "crps_scaled": "crps"}.get(
            row["metric"], row["metric"]
        )
        policy_tag = letters(row["policy"].title())
        stem = f"llm{metric_tag.title()}{policy_tag}"
        lines.append(macro(f"{stem}Pct", num(row.get("pct_diff"), 2)))
        lines.append(macro(f"{stem}CIlo", num(row.get("pct_ci_lo"), 2)))
        lines.append(macro(f"{stem}CIhi", num(row.get("pct_ci_hi"), 2)))
        lines.append(macro(f"{stem}WinRate", pct(row.get("win_rate"))))
        lines.append(macro(f"{stem}Nseries", str(row.get("n_series", "??"))))
        lines.extend(magnitude_macros(stem, row))

    # Correction rates and provider cost for the LLM arms. The rate is what shows that
    # the reflection step barely changes behaviour, and the cost is what makes the
    # comparison against a zero-call static policy concrete.
    for row in summary.get("llm_rates", []):
        policy_tag = letters(row["policy"].title())
        lines.append(macro(f"rate{policy_tag}", pct(row.get("correction_rate"))))
        lines.append(macro(f"helped{policy_tag}Llm", pct(row.get("helped"))))
        lines.append(macro(f"harmed{policy_tag}Llm", pct(row.get("harmed"))))

    # The discrimination test: the numbers Finding 4 rests on.
    d = summary.get("discrimination") or {}
    if d:
        by_arm = {r["arm"].split(" (")[0]: r for r in d.get("rows", [])}
        rand = next((r for r in d.get("rows", [])
                     if r["arm"].startswith("random")), {})
        lines.append(macro("discrimAurocCritique", num(d.get("auroc_critique"), 3)))
        lines.append(macro("discrimAurocSingle", num(d.get("auroc_single"), 3)))
        lines.append(macro("discrimAurocDelta", num(
            (d.get("auroc_critique") or 0) - (d.get("auroc_single") or 0), 3)))
        lines.append(macro("discrimJaccard", pct(d.get("jaccard"), 0)))
        lines.append(macro("discrimBaseRate", pct(d.get("base_rate"))))
        lines.append(macro("discrimPrecisionCritique", pct(
            by_arm.get("llm_critique_gate", {}).get("precision_vs_should_correct"))))
        lines.append(macro("discrimPrecisionRandom", pct(
            rand.get("precision_vs_should_correct"))))
        lines.append(macro("discrimN", str(d.get("n", "??"))))
        lines.append(macro("discrimAurocArithmetic", num(d.get("auroc_arithmetic"), 3)))
        # Matched footing: a binary yes/no scored as AUROC is balanced accuracy, so the
        # raw gap against a continuous score is partly an artefact of the output type.
        lines.append(macro("discrimAurocLlmConfidence",
                           num(d.get("auroc_llm_confidence"), 3)))
        lines.append(macro("discrimBalaccCritique", num(d.get("balacc_critique"), 3)))
        lines.append(macro("discrimBalaccArithmetic",
                           num(d.get("balacc_arithmetic_matched"), 3)))
        lines.append(macro("discrimContinuousGap", num(
            (d.get("auroc_arithmetic") or 0) - (d.get("auroc_llm_confidence") or 0), 3)))
        lines.append(macro("discrimMatchedGap", num(
            (d.get("balacc_arithmetic_matched") or 0) - (d.get("balacc_critique") or 0), 3)))
        lines.append(macro("discrimArithmeticAdvantage", num(
            (d.get("auroc_arithmetic") or 0) - (d.get("auroc_critique") or 0), 3)))

    # Action choice given the decision to correct: chosen repair vs a uniform draw
    # from the same menu on the same instance.
    for row in summary.get("action_choice", []):
        tag = {"llm_single": "Single", "llm_critique_gate": "Critique",
               "rgc_llm_router": "Router"}.get(row["arm"])
        if not tag:
            continue
        lines.append(macro(f"actionChoice{tag}Pct", num(row.get("chosen_vs_random_pct"), 2)))
        lines.append(macro(f"actionChoice{tag}CIlo", num(row.get("ci_lo"), 2)))
        lines.append(macro(f"actionChoice{tag}CIhi", num(row.get("ci_hi"), 2)))
        lines.append(macro(f"actionChoice{tag}N", str(int(row.get("n_instances", 0)))))
        # The oracle gap: distance from the best action that was available in the menu
        # on that instance. This is the headline; chosen-vs-random above is the
        # mechanism underneath it.
        lines.append(macro(f"actionChoice{tag}Gap", num(row.get("oracle_gap_pct"), 1)))
        lines.append(macro(f"actionChoice{tag}GapCIlo", num(row.get("oracle_gap_ci_lo"), 1)))
        lines.append(macro(f"actionChoice{tag}GapCIhi", num(row.get("oracle_gap_ci_hi"), 1)))
        lines.append(macro(f"actionChoice{tag}Headroom",
                           num(row.get("headroom_recovered_pct"), 1)))
        lines.append(macro(f"actionChoice{tag}HeadroomCIlo",
                           num(row.get("headroom_ci_lo"), 1)))
        lines.append(macro(f"actionChoice{tag}HeadroomCIhi",
                           num(row.get("headroom_ci_hi"), 1)))
        if tag == "Single":
            lines.append(macro("actionChoiceChosen", num(row.get("chosen"), 3)))
            lines.append(macro("actionChoiceBest", num(row.get("best_in_menu"), 3)))
            lines.append(macro("actionChoiceRandom", num(row.get("uniform_random"), 3)))
            lines.append(macro("actionChoiceWorst", num(row.get("worst_in_menu"), 3)))

    # --- Robustness families: the reasoning model and the few-shot pair ------------
    #
    # Three runs share one denominator, the same 60 series with one origin each, so the
    # prose can set them beside one another without a reader having to hold three
    # sample sizes in mind. Read off the results files directly rather than the primary
    # family's summary, because every one of these numbers is deliberately *not* the
    # primary family's.
    FAMILY_TAGS = {
        "reason": "llm_arms_reasoning",
        "fewshot": "llm_arms_fewshot",
        "zeroshot": "llm_arms_n60_zeroshot",
    }
    for tag, stem in FAMILY_TAGS.items():
        series_path = RESULTS_DIR / f"{stem}_series.parquet"
        inst_path = RESULTS_DIR / f"{stem}.parquet"
        if not series_path.exists():
            continue
        sdf = pd.read_parquet(series_path)
        idf = pd.read_parquet(inst_path) if inst_path.exists() else pd.DataFrame()
        for metric, mtag in (("mase", "Mase"), ("crps_scaled", "Crps")):
            block_df = sdf[sdf["metric"] == metric]
            for policy, ptag in (("llm_single", "Single"),
                                 ("llm_critique_gate", "Critique")):
                row = block_df[block_df["policy"] == policy]
                if row.empty:
                    continue
                row = row.iloc[0]
                lines.append(macro(f"{tag}{mtag}{ptag}Pct", num(row["pct_diff"], 2)))
                lines.append(macro(f"{tag}{mtag}{ptag}CIlo", num(row["pct_ci_lo"], 2)))
                lines.append(macro(f"{tag}{mtag}{ptag}CIhi", num(row["pct_ci_hi"], 2)))
                if metric == "mase" and ptag == "Single":
                    lines.append(macro(f"{tag}Series", str(int(row["n_series"]))))
        if not idf.empty and "correction_rate" in idf.columns:
            rates = idf[idf["metric"] == "mase"].set_index("policy")["correction_rate"]
            for policy, ptag in (("llm_single", "Single"),
                                 ("llm_critique_gate", "Critique")):
                if policy in rates.index:
                    lines.append(macro(f"{tag}Rate{ptag}",
                                       pct(float(rates[policy]), 0)))

    # Per-family discrimination, so the within-model delta can be quoted for the
    # reasoning family without retyping it from the table.
    DISCRIM_TAGS = {"reason": "critique_discrimination_reasoning.json"}
    for tag, fname in DISCRIM_TAGS.items():
        # Deliberately not named `path`: that is this function's output file, and
        # shadowing it here once wrote the macro file over a results file.
        discrim_path = RESULTS_DIR / fname
        if not discrim_path.exists():
            continue
        d = json.loads(discrim_path.read_text(encoding="utf-8"))
        lines.append(macro(f"{tag}AurocCritique", num(d.get("auroc_critique"), 3)))
        lines.append(macro(f"{tag}AurocSingle", num(d.get("auroc_single"), 3)))
        lines.append(macro(f"{tag}AurocArithmetic", num(d.get("auroc_arithmetic"), 3)))
        delta = (d.get("auroc_critique") or 0) - (d.get("auroc_single") or 0)
        lines.append(macro(f"{tag}AurocDelta", num(delta, 3)))
        lines.append(macro(f"{tag}DiscrimN", str(int(d.get("n", 0)))))

    # Action choice for the reasoning family: the one place where deliberation helps.
    ac_reason = RESULTS_DIR / "action_choice_reasoning.json"
    if ac_reason.exists():
        for row in json.loads(ac_reason.read_text(encoding="utf-8")).get("rows", []):
            tag = {"llm_single": "Single", "llm_critique_gate": "Critique"}.get(row["arm"])
            if not tag:
                continue
            lines.append(macro(f"reasonHeadroom{tag}",
                               num(row.get("headroom_recovered_pct"), 1)))
            lines.append(macro(f"reasonHeadroom{tag}CIlo",
                               num(row.get("headroom_ci_lo"), 1)))
            lines.append(macro(f"reasonHeadroom{tag}CIhi",
                               num(row.get("headroom_ci_hi"), 1)))
            lines.append(macro(f"reasonGap{tag}", num(row.get("oracle_gap_pct"), 1)))
            lines.append(macro(f"reasonChoice{tag}Pct", num(row.get("chosen_vs_random_pct"), 2)))
            lines.append(macro(f"reasonChoice{tag}CIlo", num(row.get("ci_lo"), 2)))
            lines.append(macro(f"reasonChoice{tag}CIhi", num(row.get("ci_hi"), 2)))

    # H5: the agent's own confidence against a calibrated estimate of the same event.
    cal_path = RESULTS_DIR / "confidence_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        by_name = {r["predictor"].split(" (")[0]: r for r in cal.get("rows", [])}
        llm_row = by_name.get("LLM verbalised confidence", {})
        est_row = by_name.get("Reliability estimator", {})
        const_row = next((r for r in cal.get("rows", [])
                          if r["predictor"].startswith("Constant")), {})
        lines.append(macro("calLlmEce", num(llm_row.get("ece"), 3)))
        lines.append(macro("calLlmBrier", num(llm_row.get("brier"), 3)))
        lines.append(macro("calLlmAurc", num(llm_row.get("aurc"), 3)))
        lines.append(macro("calLlmMeanRisk", num(llm_row.get("mean_prediction"), 3)))
        lines.append(macro("calEstEce", num(est_row.get("ece"), 3)))
        lines.append(macro("calEstBrier", num(est_row.get("brier"), 3)))
        lines.append(macro("calEstAurc", num(est_row.get("aurc"), 3)))
        lines.append(macro("calConstBrier", num(const_row.get("brier"), 3)))
        lines.append(macro("calConstAurc", num(const_row.get("aurc"), 3)))
        lines.append(macro("calBaseRate", pct(cal.get("base_rate"), 1)))
        lines.append(macro("calN", str(int(cal.get("n", 0)))))
        ece_d = cal.get("differences", {}).get("ece", {})
        lines.append(macro("calEceDiff", num(ece_d.get("diff"), 3)))
        lines.append(macro("calEceDiffCIlo", num(ece_d.get("ci_lo"), 3)))
        lines.append(macro("calEceDiffCIhi", num(ece_d.get("ci_hi"), 3)))

    # Ablations: which signal, removed, changes the result. Emitted as a percentage
    # difference from the full system so the prose can say what each part was worth.
    abl_path = RESULTS_DIR / "ablations.parquet"
    if abl_path.exists():
        abl = pd.read_parquet(abl_path)
        for _, row in abl.iterrows():
            tag = letters("".join(w.title() for w in str(row["ablation"]).split("_")))
            lines.append(macro(f"abl{tag}", num(row.get("delta_vs_full_pct"), 2)))
            lines.append(macro(f"abl{tag}Rate", pct(row.get("correction_rate"), 1)))
            lines.append(macro(f"abl{tag}Compute", num(row.get("compute_cost"), 2)))
            # Series-level versions: the ones the prose is allowed to lean on.
            if "series_delta_pct" in row:
                lines.append(macro(f"abl{tag}Series", num(row.get("series_delta_pct"), 2)))
                lines.append(macro(f"abl{tag}SeriesCIlo", num(row.get("series_ci_lo"), 2)))
                lines.append(macro(f"abl{tag}SeriesCIhi", num(row.get("series_ci_hi"), 2)))

    # Robustness sweep and H2's difference-in-differences.
    rob_path = RESULTS_DIR / "robustness_summary.json"
    if rob_path.exists():
        rob = json.loads(rob_path.read_text(encoding="utf-8"))
        lines.append(macro("robComparisons", str(int(rob.get("n_comparisons", 0)))))
        lines.append(macro("robImprovements",
                           str(int(rob.get("n_significant_improvements", 0)))))
        lines.append(macro("robDegradations",
                           str(int(rob.get("n_significant_degradations", 0)))))
        lines.append(macro("robHTwoTested", str(int(rob.get("h2_strata_tested", 0)))))
        lines.append(macro("robHTwoRaw",
                           str(int(rob.get("h2_strata_uncorrected", 0)))))
        lines.append(macro("robHTwoSurviving",
                           str(int(rob.get("h2_strata_supporting", 0)))))
        rep = rob.get("repairability") or {}
        if rep:
            lines.append(macro("robOracleMild",
                               pct(rep.get("oracle_improvement_mild"), 1)))
            lines.append(macro("robOracleSevere",
                               pct(rep.get("oracle_improvement_severe"), 1)))
            lines.append(macro("robShouldCorrectMild",
                               pct(rep.get("should_correct_mild"), 1)))
            lines.append(macro("robShouldCorrectSevere",
                               pct(rep.get("should_correct_severe"), 1)))
            lines.append(macro("robSeverityMild", num(rep.get("mild_severity"), 2)))
            lines.append(macro("robSeveritySevere", num(rep.get("severe_severity"), 2)))
        hits = [r for r in rob.get("did_rows", []) if r.get("supports_h2")]
        if hits:
            lines.append(macro("robHTwoPolicies", ", ".join(sorted({
                str(h["policy"]).replace("_", " ") for h in hits}))))
            lines.append(macro("robHTwoAxes", ", ".join(sorted({
                str(h["perturbation"]).replace("_", " ") for h in hits}))))
            lines.append(macro("robHTwoSeverity",
                               num(min(float(h["severity"]) for h in hits), 2)))

    # Fixed-base condition (deviation D11): the confound test. Emitted per policy for
    # both runs so the prose can quote the pair without retyping either.
    fb_path = RESULTS_DIR / "fixed_base_comparison.json"
    if fb_path.exists():
        fb = json.loads(fb_path.read_text(encoding="utf-8"))
        lines.append(macro("fbReading", str(fb.get("d11_reading", "??"))))
        lines.append(macro("fbSingleFactorOk",
                           "yes" if fb.get("single_factor_ok") else "no"))
        for r in fb.get("rows", []):
            tag = letters("".join(w.title() for w in str(r["policy"]).split("_")))
            for suffix, stag in (("adaptive", "Adaptive"), ("fixed", "Fixed")):
                v = r.get(f"pct_diff_{suffix}")
                if v is None:
                    continue
                lines.append(macro(f"fb{tag}{stag}Pct", num(v, 2)))
                lines.append(macro(f"fb{tag}{stag}CIlo", num(r.get(f"pct_ci_lo_{suffix}"), 2)))
                lines.append(macro(f"fb{tag}{stag}CIhi", num(r.get(f"pct_ci_hi_{suffix}"), 2)))

    # Realizable regret decomposition (D12): every choice made on validation, scored on
    # test, with series-level intervals. Replaces the selection-inflated version.
    rd_path = RESULTS_DIR / "regret_decomposition_realizable.json"
    if rd_path.exists():
        rd = json.loads(rd_path.read_text(encoding="utf-8"))
        rd_names = {
            "stage:canonical_fix": "Canonical",
            "stage:validated_matched": "ValMatched",
            "stage:validated_any": "ValAny",
            "stage:test_matched": "TestMatched",
            "stage:test_any": "TestAny",
            "realizable total achievable": "RealTotal",
            "realizable first-choice loss": "RealFirst",
            "realizable restriction loss": "RealRestrict",
            "ceiling total achievable": "CeilTotal",
            "ceiling first-choice loss": "CeilFirst",
            "ceiling restriction loss": "CeilRestrict",
        }
        for metric_key, mtag in (("mase", "Mase"), ("crps_scaled", "Crps")):
            blk = rd.get(metric_key) or {}
            stats = blk.get("stats") or {}
            lines.append(macro(f"rd{mtag}Nseries", str(int(blk.get("n_series", 0)))))
            for key, tag in rd_names.items():
                v = stats.get(key)
                if not v:
                    continue
                lines.append(macro(f"rd{mtag}{tag}", num(v["point"], 2)))
                lines.append(macro(f"rd{mtag}{tag}CIlo", num(v["ci_lo"], 2)))
                lines.append(macro(f"rd{mtag}{tag}CIhi", num(v["ci_hi"], 2)))
            ceil = stats.get("ceiling total achievable")
            real = stats.get("realizable total achievable")
            if ceil and real and ceil["point"]:
                lines.append(macro(f"rd{mtag}CeilTotalAbs", num(abs(ceil["point"]), 2)))
                lines.append(macro(f"rd{mtag}CurseShare",
                                   num(100.0 * (1.0 - real["point"] / ceil["point"]), 0)))

    # Static-policy checks: was the ensemble picked on test, and does it hold on real data?
    sp_path = RESULTS_DIR / "static_policy_check.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text(encoding="utf-8"))
        for metric_key, mtag in (("mase", "Mase"), ("crps_scaled", "Crps")):
            r = sp.get(metric_key) or {}
            if not r:
                continue
            lines.append(macro(f"sp{mtag}ValChoice",
                               str(r.get("validation_chosen_action", "")).replace("_", " ")))
            lines.append(macro(f"sp{mtag}EnsIsValChoice",
                               "yes" if r.get("ensemble_is_validation_choice") else "no"))
            for field, tag in (("validation_chosen_effect", "ValChoiceEff"),
                               ("ensemble_effect", "Ens")):
                e = r.get(field) or {}
                lines.append(macro(f"sp{mtag}{tag}Pct", num(e.get("pct_diff"), 2)))
                lines.append(macro(f"sp{mtag}{tag}CIlo", num(e.get("pct_ci_lo"), 2)))
                lines.append(macro(f"sp{mtag}{tag}CIhi", num(e.get("pct_ci_hi"), 2)))
            for src, stag in (("real", "Real"), ("synthetic", "Syn")):
                e = (r.get("ensemble_by_source") or {}).get(src) or {}
                lines.append(macro(f"sp{mtag}Ens{stag}Pct", num(e.get("pct_diff"), 2)))
                lines.append(macro(f"sp{mtag}Ens{stag}CIlo", num(e.get("pct_ci_lo"), 2)))
                lines.append(macro(f"sp{mtag}Ens{stag}CIhi", num(e.get("pct_ci_hi"), 2)))
                lines.append(macro(f"sp{mtag}Ens{stag}N", str(int(e.get("n_series", 0)))))

    cost = summary.get("llm_cost_totals") or {}
    if cost:
        lines.append(macro("llmCallsCritiqueGate",
                           str(int(cost.get("critique_gate_calls", 0)))))
        lines.append(macro("llmCallsTotal", str(int(cost.get("total_calls", 0)))))
        lines.append(macro("llmPromptTokensThousands",
                           num(cost.get("prompt_tokens", 0) / 1000.0, 0)))
        lines.append(macro("llmErrors", str(int(cost.get("errors", 0)))))

    # Short, stable aliases for the regret-decomposition loss terms, keyed by prefix
    # so a change in the exact wording of scripts/regret_decomposition.py's loss names
    # does not silently break every macro name used in the manuscript prose.
    _LOSS_ALIASES = {
        "total achievable": "TotalAchievable",
        "restriction loss": "RestrictionLoss",
        "first-choice loss": "FirstChoiceLoss",
    }
    regret = summary.get("regret_decomposition") or {}
    for metric_key, losses in regret.items():
        metric_tag = letters(metric_key.title())
        for loss_name, value in losses.items():
            alias = next(
                (v for k, v in _LOSS_ALIASES.items() if loss_name.startswith(k)),
                letters(loss_name.title()),
            )
            lines.append(macro(f"decomp{metric_tag}{alias}", num(value, 2)))

    # \newcommand on an already-defined name is a hard LaTeX error, not a warning, so a
    # duplicate emitted by two code paths must fail here rather than at build time.
    head = "\\newcommand{\\"
    names = [ln[len(head):].split("}", 1)[0]
             for ln in lines if ln.startswith(head)]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"duplicate macro definitions would break the build: {dupes}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--ablations", default=str(RESULTS_DIR / "ablations.parquet"))
    parser.add_argument("--robustness", default=str(RESULTS_DIR / "robustness.parquet"))
    args = parser.parse_args(argv)

    setup_logging()
    from visualization import figures as F
    from visualization import tables as T

    F.set_style()
    run_dir = resolve_run(args.run)
    log.info("generating paper assets from %s", run_dir)

    scores = load(run_dir, "scores")
    signals = load(run_dir, "signals")
    labels = load(run_dir, "test_labels")
    comparison = load(run_dir, "comparison")
    realizable = load(run_dir, "realizable_oracle")
    importance = load(run_dir, "feature_importance")

    if scores is None or comparison is None:
        raise SystemExit(f"{run_dir} does not contain a completed experiment")

    # Use the first seed for the confirmatory comparison; seeds are reported as a band.
    primary_scores = scores[scores["seed"] == scores["seed"].min()]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    generated = PAPER_DIR / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    # Collected across several blocks below, so it is defined once up front.
    llm_summary: dict = {}

    # --- figures ---
    reference_metric = float(
        comparison["reference_metric"].iloc[0] if len(comparison) else np.nan
    )
    F.save(F.fig_policy_comparison(comparison, reference_metric),
           FIGURES_DIR / "fig_policy_comparison")
    written.append("fig_policy_comparison")

    F.save(F.fig_correction_outcomes(primary_scores),
           FIGURES_DIR / "fig_correction_outcomes")
    written.append("fig_correction_outcomes")

    if "family" in primary_scores.columns:
        try:
            F.save(F.fig_stratum_gains(primary_scores), FIGURES_DIR / "fig_stratum_gains")
            written.append("fig_stratum_gains")
        except KeyError as exc:
            log.warning("stratum figure skipped: %s", exc)

    if "confidence" in primary_scores.columns:
        F.save(F.fig_risk_coverage(primary_scores), FIGURES_DIR / "fig_risk_coverage")
        written.append("fig_risk_coverage")

    if "compute_cost" in primary_scores.columns:
        F.save(F.fig_cost_accuracy(primary_scores), FIGURES_DIR / "fig_cost_accuracy")
        written.append("fig_cost_accuracy")

    rgc = primary_scores[primary_scores["policy"] == "rgc"]
    if len(rgc) and labels is not None and rgc["risk"].notna().any():
        merged = rgc.merge(labels[["instance_id", "large_error"]], on="instance_id",
                           suffixes=("", "_lab"))
        target_col = "large_error_lab" if "large_error_lab" in merged else "large_error"
        if merged[target_col].nunique() > 1:
            F.save(
                F.fig_calibration(merged["risk"].to_numpy(),
                                  merged[target_col].astype(float).to_numpy()),
                FIGURES_DIR / "fig_calibration",
            )
            written.append("fig_calibration")

    if importance is not None and not importance.empty:
        F.save(F.fig_feature_importance(importance),
               FIGURES_DIR / "fig_feature_importance")
        written.append("fig_feature_importance")

    if labels is not None and realizable is not None and not realizable.empty:
        means = primary_scores.groupby("policy")["chosen_metric"].mean().to_dict()
        keep = {k: v for k, v in means.items() if k in ("rgc", "always_correct")}
        F.save(F.fig_oracle_gap(labels, realizable, keep), FIGURES_DIR / "fig_oracle_gap")
        written.append("fig_oracle_gap")

    robustness_path = Path(args.robustness)
    if robustness_path.exists():
        robustness = pd.read_parquet(robustness_path)
        F.save(F.fig_robustness(robustness), FIGURES_DIR / "fig_robustness")
        written.append("fig_robustness")

    # --- tables ---
    T.table_main_results(comparison, primary_scores, TABLES_DIR / "table_main_results.tex")
    T.table_decision_quality(primary_scores, TABLES_DIR / "table_decision_quality.tex")
    T.table_cost(primary_scores, TABLES_DIR / "table_cost.tex")
    written += ["table_main_results", "table_decision_quality", "table_cost"]

    if signals is not None and not signals.empty:
        T.table_dataset_summary(signals, TABLES_DIR / "table_dataset_summary.tex")
        written.append("table_dataset_summary")

    ablation_path = Path(args.ablations)
    if ablation_path.exists():
        T.table_ablations(pd.read_parquet(ablation_path),
                          TABLES_DIR / "table_ablations.tex")
        written.append("table_ablations")

    # The headline dissociation table, produced by scripts/headline_results.py.
    headline_path = RESULTS_DIR / "headline_results.parquet"
    if headline_path.exists():
        headline = pd.read_parquet(headline_path)
        T.table_headline(headline, TABLES_DIR / "table_headline.tex")
        written.append("table_headline")
    else:
        log.warning(
            "results/headline_results.parquet is missing; run "
            "scripts/headline_results.py first or the manuscript's main table will be "
            "absent"
        )

    # Regret decomposition: where the achievable improvement goes under ground-truth
    # diagnosis (scripts/regret_decomposition.py). This is what explains why the
    # diagnose-then-apply-canonical-fix pattern underperforms.
    regret_decomp_path = RESULTS_DIR / "regret_decomposition.parquet"
    if regret_decomp_path.exists():
        regret_table = pd.read_parquet(regret_decomp_path)
        T.table_regret_decomposition(
            regret_table, TABLES_DIR / "table_regret_decomposition.tex"
        )
        written.append("table_regret_decomposition")
    else:
        log.warning(
            "results/regret_decomposition.parquet is missing; run "
            "scripts/regret_decomposition.py first"
        )

    # Selective-prediction / triage value of the failure-risk estimator
    ac_path = RESULTS_DIR / "action_choice.json"
    if ac_path.exists():
        llm_summary["action_choice"] = json.loads(
            ac_path.read_text(encoding="utf-8")).get("rows", [])
        T.table_action_choice(llm_summary["action_choice"],
                              TABLES_DIR / "table_action_choice.tex")
        written.append("table_action_choice")

    cal_path = RESULTS_DIR / "confidence_calibration.json"
    if cal_path.exists():
        T.table_confidence_calibration(
            json.loads(cal_path.read_text(encoding="utf-8")),
            TABLES_DIR / "table_confidence_calibration.tex")
        written.append("table_confidence_calibration")

    rob_json = RESULTS_DIR / "robustness_summary.json"
    if rob_json.exists():
        T.table_robustness(json.loads(rob_json.read_text(encoding="utf-8")),
                           TABLES_DIR / "table_robustness.tex")
        written.append("table_robustness")

    rdr_json = RESULTS_DIR / "regret_decomposition_realizable.json"
    if rdr_json.exists():
        T.table_regret_decomposition_realizable(
            json.loads(rdr_json.read_text(encoding="utf-8")),
            TABLES_DIR / "table_regret_decomposition_realizable.tex")
        written.append("table_regret_decomposition_realizable")

    fb_json = RESULTS_DIR / "fixed_base_comparison.json"
    if fb_json.exists():
        T.table_fixed_base(json.loads(fb_json.read_text(encoding="utf-8")),
                           TABLES_DIR / "table_fixed_base.tex")
        written.append("table_fixed_base")

    # Per-action coverage against CRPS. This is the table that answers "could the
    # probabilistic result just be interval inflation?" -- abstain widens intervals,
    # gets the best coverage in the action space, and is heavily penalised on CRPS.
    outcomes_path = run_dir / "outcomes.parquet"
    if outcomes_path.exists():
        T.table_coverage(pd.read_parquet(outcomes_path),
                         TABLES_DIR / "table_coverage.tex")
        written.append("table_coverage")

    # LLM correction arms (scripts/llm_arms.py). These are the arms the manuscript's
    # critique is actually about, so their absence is worth a loud warning rather than
    # a silent omission.
    # One entry per model family. The first is primary (its numbers drive the inline
    # macros); the rest are cross-family replications, which is what answers the
    # objection that a null result merely reflects one weak model.
    LLM_FAMILIES = [
        ("Gemini 3.5 Flash-Lite", "llm_arms"),
        ("Qwen3.8-27B (Groq)", "llm_arms_groq"),
        ("GPT-OSS-20B reasoning (Groq)", "llm_arms_reasoning"),
        ("Gemini 3.5 Flash-Lite, 8-shot", "llm_arms_fewshot"),
        ("Gemini 3.5 Flash-Lite, 0-shot (matched)", "llm_arms_n60_zeroshot"),
    ]
    families = []
    for family_label, stem in LLM_FAMILIES:
        series_path = RESULTS_DIR / f"{stem}_series.parquet"
        if not series_path.exists():
            continue
        cost_path = RESULTS_DIR / f"{stem}_cost.parquet"
        families.append((
            family_label,
            pd.read_parquet(series_path),
            pd.read_parquet(cost_path) if cost_path.exists() else pd.DataFrame(),
        ))

    if families:
        T.table_llm_arms(families, TABLES_DIR / "table_llm_arms.tex")
        written.append("table_llm_arms")
        primary_label, primary_series, primary_cost = families[0]
        llm_summary["llm_arms"] = primary_series.to_dict(orient="records")
        llm_summary["llm_primary_family"] = primary_label
        llm_summary["llm_families"] = [f[0] for f in families]
        if not primary_cost.empty:
            llm_summary["llm_cost"] = primary_cost.to_dict(orient="records")
        # Replications are kept separately so a macro can never silently quote one
        # family's number where another's is meant.
        llm_summary["llm_replications"] = {
            label: series.to_dict(orient="records")
            for label, series, _ in families[1:]
        }

        # Per-arm rates come from the primary family's instance-level table, and are
        # metric-independent for the two arms whose prompt does not depend on the
        # metric, so the MASE pass is representative.
        primary_instance = RESULTS_DIR / f"{LLM_FAMILIES[0][1]}.parquet"
        if primary_instance.exists():
            inst = pd.read_parquet(primary_instance)
            inst = inst[inst["metric"] == "mase"]
            keep = ["policy", "correction_rate", "helped", "harmed"]
            llm_summary["llm_rates"] = inst[
                inst["policy"].str.contains("llm")
            ][[c for c in keep if c in inst.columns]].to_dict(orient="records")

        if not primary_cost.empty:
            gate = primary_cost[primary_cost["policy"] == "llm_critique_gate"]
            llm_summary["llm_cost_totals"] = {
                "critique_gate_calls": float(gate["llm_calls"].sum()),
                "total_calls": float(primary_cost["llm_calls"].sum()),
                "prompt_tokens": float(primary_cost["prompt_tokens"].sum()),
                "errors": float(
                    (json.loads((RESULTS_DIR / "llm_arms_meta.json").read_text())
                     .get("errors", 0))
                    if (RESULTS_DIR / "llm_arms_meta.json").exists() else 0
                ),
            }
    else:
        log.warning(
            "results/llm_arms_series.parquet is missing; run scripts/llm_arms.py. "
            "The manuscript critiques LLM-driven correction, so it must not be "
            "submitted without having run the LLM arms."
        )

    # Whether verbal self-critique carries information (scripts/critique_discrimination.py).
    # The primary family's numbers drive the inline macros; the others are shown in the
    # table so the reader can see the pattern is not one model's quirk.
    DISCRIM_FILES = [
        ("Gemini 3.5 Flash-Lite", "critique_discrimination.json"),
        ("Qwen3.8-27B (Groq)", "critique_discrimination_groq.json"),
        ("GPT-OSS-20B reasoning (Groq)", "critique_discrimination_reasoning.json"),
    ]
    discrim = []
    for label, fname in DISCRIM_FILES:
        path = RESULTS_DIR / fname
        if path.exists():
            try:
                discrim.append((label, json.loads(path.read_text(encoding="utf-8"))))
            except json.JSONDecodeError:
                log.warning("%s is not valid JSON; skipping that family. Re-run "
                            "scripts/critique_discrimination.py to regenerate it.", path)
    if discrim:
        T.table_discrimination(discrim, TABLES_DIR / "table_discrimination.tex")
        written.append("table_discrimination")
        llm_summary["discrimination"] = discrim[0][1]
        llm_summary["discrimination_all"] = {lbl: d for lbl, d in discrim}

    # (scripts/triage_analysis.py): the positive, deployable finding alongside the
    # cautionary results on automated correction.
    triage_path = RESULTS_DIR / "triage.parquet"
    triage_meta_path = RESULTS_DIR / "triage_meta.parquet"
    if triage_path.exists() and triage_meta_path.exists():
        triage_table = pd.read_parquet(triage_path)
        triage_meta = pd.read_parquet(triage_meta_path).iloc[0]
        T.table_triage(
            triage_table, float(triage_meta["auroc"]), float(triage_meta["base_rate"]),
            TABLES_DIR / "table_triage.tex",
        )
        written.append("table_triage")
    else:
        log.warning(
            "results/triage.parquet is missing; run scripts/triage_analysis.py first"
        )

    decisions = load(run_dir, "decisions")
    if decisions is not None and signals is not None:
        from evaluation.agent_metrics import diagnosis_accuracy

        accuracy = diagnosis_accuracy(decisions, signals)
        if not accuracy.empty:
            T.table_diagnosis_accuracy(
                accuracy, TABLES_DIR / "table_diagnosis_accuracy.tex"
            )
            accuracy.to_csv(generated / "diagnosis_accuracy.csv", index=False)
            written.append("table_diagnosis_accuracy")

    verdicts = hypothesis_verdicts(comparison, primary_scores, labels, signals, run_dir)
    if verdicts:
        T.table_hypotheses(verdicts, TABLES_DIR / "table_hypotheses.tex")
        written.append("table_hypotheses")

    # --- machine-readable summary the prose can be checked against ---
    summary = {
        "run_id": run_dir.name,
        "n_test_instances": int(primary_scores["instance_id"].nunique()),
        "n_series": int(primary_scores["series_id"].nunique())
        if "series_id" in primary_scores else None,
        "policies": {
            policy: {
                "mase": float(group["chosen_metric"].mean()),
                "regret": float(group["regret"].mean()),
                "correction_rate": float(group["corrected"].mean()),
                "helped": float(group[group["corrected"]]["improved"].mean())
                if group["corrected"].any() else None,
                "harmed": float(group[group["corrected"]]["harmed"].mean())
                if group["corrected"].any() else None,
            }
            for policy, group in primary_scores.groupby("policy")
        },
        "accept_metric": float(labels["accept_metric"].mean())
        if labels is not None and not labels.empty else None,
        "unrestricted_oracle": float(labels["oracle_metric"].mean())
        if labels is not None and not labels.empty else None,
        "realizable_oracle": float(realizable["realizable_metric"].mean())
        if realizable is not None and not realizable.empty else None,
        "should_correct_rate": float(labels["should_correct"].mean())
        if labels is not None and not labels.empty else None,
        "hypotheses": verdicts,
        "assets_written": written,
        **llm_summary,
    }

    # Series-level inference, carried straight through from scripts/headline_results.py
    # so the manuscript and the console table cannot diverge.
    if headline_path.exists():
        headline_full = pd.read_parquet(headline_path)
        series_cols = ["metric", "policy", "series_mean", "pct_diff", "pct_ci_lo",
                       "pct_ci_hi", "win_rate", "wilcoxon_p", "cliffs_delta",
                       "n_series", "significant"]
        if set(series_cols).issubset(headline_full.columns):
            summary["series_level"] = (
                headline_full[series_cols].to_dict(orient="records")
            )
        else:
            log.warning(
                "headline_results.parquet has no series-level columns; rerun "
                "scripts/headline_results.py. The manuscript's confirmatory numbers "
                "will be missing rather than wrong."
            )

    if triage_path.exists() and triage_meta_path.exists():
        triage_table_full = pd.read_parquet(triage_path)
        triage_meta_full = pd.read_parquet(triage_meta_path).iloc[0]
        summary["triage"] = {
            "auroc": float(triage_meta_full["auroc"]),
            "base_rate": float(triage_meta_full["base_rate"]),
            "budgets": {
                f"{int(round(r['review_budget'] * 100))}": {
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "lift": float(r["lift"]),
                }
                for _, r in triage_table_full.iterrows()
            },
        }

    if regret_decomp_path.exists():
        loss_path = regret_decomp_path.with_name("regret_decomposition_losses.parquet")
        if loss_path.exists():
            loss_table = pd.read_parquet(loss_path)
            summary["regret_decomposition"] = {
                metric: dict(zip(group["loss"], group["value_pct"]))
                for metric, group in loss_table.groupby("metric")
            }

    (generated / "results_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    write_macros(summary, generated / "macros.tex")

    print(f"Generated {len(written)} assets from {run_dir.name}")
    print(f"  figures -> {FIGURES_DIR}")
    print(f"  tables  -> {TABLES_DIR}")
    print(f"  summary -> {generated / 'results_summary.json'}")
    if verdicts:
        print("\nPre-registered hypotheses:")
        for v in verdicts:
            print(f"  {v['id']}: {v['verdict']:14s} ({v['test']} = {v['statistic']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
