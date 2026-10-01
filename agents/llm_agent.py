"""LLM-based correction policies.

Three arms, each isolating a different question:

``llm_single``
    Baseline B. The model sees the diagnostics once and picks a single action, with no
    critique, no second pass and no validation. This is the "LLM agent that does not
    self-correct" the study is measured against.

``llm_critique_gate``
    The model judges *for itself* whether its forecast is unreliable and should be
    revised -- verbal self-critique, the mechanism most of the agentic forecasting
    literature actually uses. Compared against RGC's quantitative gate, this is the
    direct test of H7, in a domain where a cheap external signal exists.

``rgc_llm_router``
    RGC's calibrated gate decides *whether* to correct; the model decides *which*
    repair to apply among the diagnosed candidates. This separates the value of the
    gate from the value of the routing, which the ablations otherwise confound.

Contamination control
---------------------
ETT, M4 and the Monash archive are public and long-standing, so a frontier model may
well have memorised them. Every prompt therefore omits the dataset name, the series
identifier, all dates, and the raw values; the model sees only scale-free diagnostic
statistics. It is being asked to reason about a *diagnosis*, never to recall a series.
``scripts/contamination_probe.py`` tests whether that control holds.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from agents.diagnosis import candidate_actions, diagnose
from agents.llm import LLMClient
from agents.policies import ACCEPT, CorrectionPolicy, PolicyDecision, ValidationEvidence
from core.logging import get_logger

log = get_logger("llm_agent")

__all__ = [
    "LLMSinglePolicy",
    "LLMCritiqueGatePolicy",
    "RGCLLMRouterPolicy",
    "build_evidence_prompt",
    "build_fewshot_prefix",
    "SYSTEM_PROMPT",
]

SYSTEM_PROMPT = (
    "You are a forecasting reliability analyst. You are given diagnostic statistics "
    "for a time series and a forecast that has already been produced for it. Your job "
    "is to judge whether the forecast is likely to be unreliable, and if so which "
    "single corrective action is most likely to improve it.\n\n"
    "You never see the raw series, its name, or any dates. Reason only from the "
    "statistics provided. Prefer accepting the forecast when the evidence is weak: an "
    "unnecessary correction costs compute and can make the forecast worse.\n\n"
    "Reply with JSON only, no prose and no code fences."
)

#: Action descriptions shown to the model. Kept terse and mechanism-focused so the
#: model reasons about what each repair does rather than pattern-matching on names.
ACTION_MENU: dict[str, str] = {
    "accept": "Keep the forecast unchanged.",
    "switch_model": "Refit with a different model class chosen by backtest error.",
    "refit_post_changepoint": "Refit using only data after the last structural break.",
    "robustify": "Replace outliers with local medians, then refit.",
    "damp_trend": "Geometrically damp the extrapolated trend.",
    "respecify_seasonality": "Refit using a seasonal period detected from the data.",
    "transform": "Fit on a log scale to stabilise variance.",
    "ensemble": "Average the best few models by backtest error.",
    "recalibrate_intervals": "Rescale prediction intervals only; point forecast unchanged.",
    "abstain": "Fall back to a seasonal naive forecast and flag low confidence.",
}


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else f"{f:.{digits}g}"


def _diagnostic_lines(signals: pd.Series) -> list[str]:
    """Render diagnostics as a scale-free evidence block.

    Only derived statistics appear -- no identifiers, no dates, no raw observations.
    Kept separate from the action menu and the reply instruction so that a worked
    example can show the same evidence a query shows without repeating either.
    """
    g = signals.get
    lines = [
        "TIME SERIES DIAGNOSTICS (all statistics are scale-free or normalised)",
        "",
        "Structure:",
        f"  history length: {_fmt(g('context.history_length'), 4)} observations",
        f"  seasonal cycles of history available: {_fmt(g('context.history_per_season'))}",
        f"  forecast horizon: {_fmt(g('context.horizon'), 2)} steps",
        f"  trend strength (0-1): {_fmt(g('stl.trend_strength'))}",
        f"  seasonal strength (0-1): {_fmt(g('stl.seasonal_strength'))}",
        f"  spectral entropy (1 = noise-like): {_fmt(g('ts_features.spectral_entropy'))}",
        f"  lag-1 autocorrelation: {_fmt(g('acf.acf_lag1'))}",
        "",
        "Stability:",
        f"  change points detected: {_fmt(g('changepoint.n_changepoints'), 2)}",
        f"  recency of last change point (0 = just now, 1 = none): "
        f"{_fmt(g('changepoint.relative_recency'))}",
        f"  size of last level shift (in sd): {_fmt(g('changepoint.last_segment_mean_shift'))}",
        f"  stationarity (ADF p): {_fmt(g('stationarity.adf_pvalue'))}, "
        f"(KPSS p): {_fmt(g('stationarity.kpss_pvalue'))}",
        f"  recent-vs-past distribution shift (KS): {_fmt(g('distribution_shift.ks_statistic'))}, "
        f"PSI: {_fmt(g('distribution_shift.psi'))}",
        f"  recent/past variance ratio: {_fmt(g('distribution_shift.variance_ratio'))}",
        "",
        "Contamination:",
        f"  anomaly rate: {_fmt(g('anomaly.anomaly_rate'))}, "
        f"recent window: {_fmt(g('anomaly.recent_anomaly_rate'))}",
        f"  final observation is an outlier: {bool(g('anomaly.anomaly_at_last_point', 0))}",
        "",
        "Seasonality detection:",
        f"  assumed period matches detected period: "
        f"{not bool(g('seasonality.period_mismatch', 0))}",
        f"  multiple seasonal cycles present: {bool(g('seasonality.multiple_seasonality', 0))}",
        "",
        "Forecast reliability evidence:",
        f"  backtest error on recent history (MASE): {_fmt(g('backtest.backtest_mase'))}",
        f"  variability of that error across folds: {_fmt(g('backtest.backtest_mase_std'))}",
        f"  systematic bias ratio (0 = unbiased, 1 = one-directional): "
        f"{_fmt(g('backtest.bias_ratio'))}",
        f"  backtest error trend towards the present: {_fmt(g('backtest.error_trend'))}",
        f"  disagreement across the model pool (normalised): "
        f"{_fmt(g('disagreement.normalized'))}",
        f"  margin between best and second-best model: {_fmt(g('selection.margin'))}",
        f"  prediction interval width (scaled): {_fmt(g('uncertainty.interval_width_scaled'))}",
    ]
    return lines


def build_fewshot_prefix(
    examples: Sequence[tuple[pd.Series, bool, str]], ask_gate: bool = False
) -> str:
    """Show the model solved instances before asking it to decide.

    The LLM arms are zero-shot while the arithmetic gate they are measured against is
    fitted on the validation split. That asymmetry is a real confound: a null result
    could mean the model cannot read the evidence, or only that nobody ever showed it
    what a good decision looks like. These exemplars are drawn from the **validation**
    split and labelled by the counterfactual tensor, so the language model receives the
    same supervision, out of the same data, as the estimator it is compared against.

    Only the decision is shown -- no confidence and no rationale -- because the tensor
    knows which action was right and does not know why, and writing a justification on
    its behalf would put words in the data's mouth.

    The set is fixed across instances, the same examples in the same order, so the only
    thing that differs between two queries is the case being asked about.
    """
    if not examples:
        return ""
    out = [
        f"WORKED EXAMPLES: {len(examples)} past instances from a held-out earlier "
        "period, each with the decision that turned out to be correct.",
        "Only the decision is shown for these. Answer the final case in the full format "
        "requested at the end.",
    ]
    for i, (example_signals, needs, action) in enumerate(examples, start=1):
        out += ["", f"--- EXAMPLE {i} ---"]
        out += _diagnostic_lines(example_signals)
        decision = (
            f'{{"needs_correction": {str(bool(needs)).lower()}, "action": "{action}"}}'
            if ask_gate else f'{{"action": "{action}"}}'
        )
        out += ["", f"CORRECT DECISION: {decision}"]
    out += ["", "--- END OF EXAMPLES; THE CASE TO DECIDE FOLLOWS ---"]
    return "\n".join(out)


def build_evidence_prompt(
    signals: pd.Series,
    candidates: Sequence[str] | None = None,
    ask_gate: bool = False,
    few_shot: Sequence[tuple[pd.Series, bool, str]] = (),
) -> str:
    """Assemble the prompt: optional worked examples, evidence, menu, instruction."""
    lines = _diagnostic_lines(signals)

    menu = list(candidates) if candidates else list(ACTION_MENU)
    if ACCEPT not in menu:
        menu = [ACCEPT] + menu
    lines += ["", "AVAILABLE ACTIONS:"]
    lines += [f"  {a}: {ACTION_MENU.get(a, '')}" for a in menu]

    if ask_gate:
        lines += [
            "",
            "Decide whether this forecast should be revised at all, and if so which "
            "action to take.",
            "",
            'Reply with JSON: {"needs_correction": true|false, "action": "<name or '
            'accept>", "confidence": <0-1 that the forecast as it stands is '
            'reliable>, "reason": "<one sentence>"}',
        ]
    else:
        lines += [
            "",
            "Choose exactly one action.",
            "",
            'Reply with JSON: {"action": "<name>", "confidence": <0-1 that the '
            'resulting forecast will be reliable>, "reason": "<one sentence>"}',
        ]
    body = "\n".join(lines)
    prefix = build_fewshot_prefix(few_shot, ask_gate=ask_gate)
    return f"{prefix}\n\n{body}" if prefix else body


def _parse(response, valid: Sequence[str]) -> tuple[str, float, bool, str]:
    """Extract (action, confidence, needs_correction, reason) defensively.

    A malformed or refused response falls back to accepting. Failing closed matters:
    an unparsed reply must not be silently converted into a random intervention, which
    would show up as the LLM arm making erratic corrections it never actually chose.
    """
    data = response.json() if response and not response.error else None
    if not isinstance(data, dict):
        return ACCEPT, float("nan"), False, "unparseable response; accepted by default"

    action = str(data.get("action", ACCEPT)).strip().lower()
    if action not in valid:
        action = ACCEPT
    try:
        confidence = float(data.get("confidence", float("nan")))
    except (TypeError, ValueError):
        confidence = float("nan")
    if np.isfinite(confidence):
        confidence = float(np.clip(confidence, 0.0, 1.0))

    needs = data.get("needs_correction")
    needs = bool(needs) if isinstance(needs, bool) else (action != ACCEPT)
    return action, confidence, needs, str(data.get("reason", ""))[:300]


@dataclass
class _BaseLLMPolicy(CorrectionPolicy):
    """Shared plumbing for the LLM arms."""

    client: LLMClient = None  # type: ignore[assignment]
    uses_llm: bool = True
    max_candidates: int = 4
    #: Instances are decided concurrently. Each decision depends only on its own
    #: instance, so this changes throughput and nothing else; a serial run and a
    #: concurrent run produce identical decisions because temperature is 0 and every
    #: response is cached under a hash of its prompt. Kept modest because provider
    #: rate limits, not local CPU, are the binding constraint.
    n_workers: int = 8
    #: Worked examples prepended to every prompt, as (signals, needs_correction,
    #: action). Empty is the zero-shot default the main results use. The prefix is part
    #: of the prompt and therefore part of the response cache key, so a few-shot run
    #: cannot collide with a zero-shot one.
    few_shot: tuple = ()
    _failures: int = field(default=0, init=False)

    def decide(self, signals, evidence, seed: int = 0):
        """Decide for every instance, issuing provider calls concurrently."""
        if self.n_workers <= 1:
            return super().decide(signals, evidence, seed)

        rows = [row for _, row in signals.iterrows()]
        results: list = [None] * len(rows)

        def run(i: int):
            # Each worker gets its own generator so the sequence any one instance
            # sees does not depend on how the pool happened to interleave.
            results[i] = self.decide_one(
                rows[i], evidence, np.random.default_rng(seed + i)
            )

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            for future in as_completed([pool.submit(run, i) for i in range(len(rows))]):
                future.result()  # re-raise budget/cache errors on the main thread

        out = pd.DataFrame([vars(d) for d in results])
        out["policy"] = self.name
        return out

    def _ask(self, prompt: str, valid: Sequence[str]):
        try:
            response = self.client.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001 - budget/cache errors must surface
            if type(exc).__name__ in ("LLMBudgetExceeded", "CacheMiss"):
                raise
            self._failures += 1
            log.warning("LLM call failed: %s", exc)
            return ACCEPT, float("nan"), False, f"llm error: {exc}", 0
        action, confidence, needs, reason = _parse(response, valid)
        return action, confidence, needs, reason, (0 if response.cached else 1)


@dataclass
class LLMSinglePolicy(_BaseLLMPolicy):
    """Baseline B: one pass, one action, no critique and no validation."""

    name: str = "llm_single"

    def decide_one(self, signals, evidence, rng):
        menu = list(ACTION_MENU)
        prompt = build_evidence_prompt(signals, candidates=menu, ask_gate=False,
                                      few_shot=self.few_shot)
        action, confidence, _, reason, n_calls = self._ask(prompt, menu)
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action,
            corrected=action != ACCEPT,
            confidence=confidence,
            n_candidates_evaluated=1,
            compute_cost=1.0,
            n_llm_calls=n_calls,
            rationale=f"single-pass LLM: {reason}",
        )


@dataclass
class LLMCritiqueGatePolicy(_BaseLLMPolicy):
    """Verbal self-critique decides whether to correct.

    The mechanism under test in H7. Crucially it receives *the same evidence* as RGC --
    the comparison is about how the decision is made, not about who saw more.
    """

    name: str = "llm_critique_gate"

    def decide_one(self, signals, evidence, rng):
        menu = list(ACTION_MENU)
        prompt = build_evidence_prompt(signals, candidates=menu, ask_gate=True,
                                      few_shot=self.few_shot)
        action, confidence, needs, reason, n_calls = self._ask(prompt, menu)
        if not needs:
            action = ACCEPT
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action,
            corrected=action != ACCEPT,
            confidence=confidence,
            n_candidates_evaluated=1,
            compute_cost=1.0,
            n_llm_calls=n_calls,
            rationale=f"LLM self-critique gate: {reason}",
        )


@dataclass
class RGCLLMRouterPolicy(_BaseLLMPolicy):
    """Quantitative gate, LLM routing.

    Isolates the value of the gate from the value of the router: the gate and the
    candidate set are RGC's, and only the final choice among candidates is delegated.
    """

    name: str = "rgc_llm_router"
    gain_threshold: float = 0.5
    acceptance_delta: float = 0.02
    gain_scores: dict[str, float] = field(default_factory=dict)
    risk_scores: dict[str, float] = field(default_factory=dict)
    disabled_signals: tuple[str, ...] = ()

    def decide_one(self, signals, evidence, rng):
        iid = signals["instance_id"]
        gain = float(self.gain_scores.get(iid, np.nan))
        risk = float(self.risk_scores.get(iid, np.nan))

        if not np.isfinite(gain) or gain < self.gain_threshold:
            return PolicyDecision(
                instance_id=iid, chosen_action=ACCEPT, corrected=False,
                risk=risk, gain=gain,
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale=f"quantitative gate closed (predicted gain {gain:.2f})",
            )

        dx = diagnose(signals.to_dict(), disabled=self.disabled_signals)
        candidates = candidate_actions(dx, max_actions=self.max_candidates)
        menu = [ACCEPT] + [c for c in candidates if c != ACCEPT]
        prompt = build_evidence_prompt(signals, candidates=menu, ask_gate=False,
                                      few_shot=self.few_shot)
        action, confidence, _, reason, n_calls = self._ask(prompt, menu)

        return PolicyDecision(
            instance_id=iid,
            chosen_action=action,
            corrected=action != ACCEPT,
            risk=risk,
            gain=gain,
            confidence=confidence if np.isfinite(confidence)
            else 1.0 - (risk if np.isfinite(risk) else 0.5),
            diagnosed_mode=dx.mode.value,
            n_candidates_evaluated=len(candidates),
            compute_cost=float(len(candidates)),
            n_llm_calls=n_calls,
            rationale=f"gate open, LLM routed to {action}: {reason}",
        )
