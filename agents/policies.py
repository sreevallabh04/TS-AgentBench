"""Correction policies: the experimental arms.

Every arm answers the same two questions for each instance -- *should I correct?* and
*with what?* -- and they differ only in how those questions are answered. Base-model
selection, the model pool, the diagnostics and the action space are identical across
arms by construction, so any difference in results is attributable to the correction
policy and nothing else.

All policies are *replays* over the counterfactual tensor. Choosing an action costs a
table lookup rather than a refit, which is what makes the full ablation and robustness
grid affordable and makes every arm exactly reproducible.

**Validation evidence is not leakage.** Policies may consult how each action performed
on the *validation* origins of the same series. Those origins lie strictly in the past
relative to the test origin, so a deployed agent could compute exactly this by
backtesting candidate repairs on recent history. This is the external signal that
Huang et al. (2024) identify as the missing ingredient in intrinsic self-correction,
and obtaining it here costs nothing but compute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from agents.diagnosis import candidate_actions, diagnose
from core.logging import get_logger

log = get_logger("policies")

__all__ = [
    "ValidationEvidence",
    "CorrectionPolicy",
    "NeverCorrectPolicy",
    "AlwaysCorrectPolicy",
    "RandomCorrectPolicy",
    "ThresholdPolicy",
    "BestFixedPolicy",
    "RGCPolicy",
    "CalibratedGainPolicy",
    "PolicyDecision",
    "ACCEPT",
]

ACCEPT = "accept"


@dataclass
class PolicyDecision:
    """One instance's decision, with everything needed to score it."""

    instance_id: str
    chosen_action: str
    corrected: bool
    risk: float = float("nan")
    gain: float = float("nan")
    confidence: float = float("nan")
    diagnosed_mode: str = "none"
    n_candidates_evaluated: int = 0
    compute_cost: float = 0.0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    rationale: str = ""


class ValidationEvidence:
    """Per-series action performance measured on validation origins.

    This is the agent's own backtest of candidate repairs. Built once from the
    validation rows of the tensor and shared by every policy, so no arm gets privileged
    evidence.
    """

    def __init__(self, val_outcomes: pd.DataFrame, metric: str = "mase") -> None:
        df = val_outcomes[val_outcomes["split"] == "val"].copy()
        df.loc[~df["applicable"].astype(bool), metric] = np.nan
        self.metric = metric
        # (series_id, horizon, action) -> mean validation metric
        self._table = (
            df.groupby(["series_id", "horizon", "action"], sort=False)[metric]
            .mean()
            .to_dict()
        )
        # Global fallback for series with no usable validation origins.
        self._global = df.groupby("action")[metric].mean().to_dict()
        # Cached once. Deriving it per call meant rescanning the whole lookup table
        # for every instance, which is quadratic in the benchmark size and dominated
        # the runtime of any arm that calls best_action without an explicit pool.
        self._actions = sorted({action for (_, _, action) in self._table})

    def score(self, series_id: str, horizon: int, action: str) -> float:
        """Validation error of ``action`` for this series; NaN when unknown."""
        value = self._table.get((series_id, int(horizon), action))
        if value is None or not np.isfinite(value):
            value = self._global.get(action, np.nan)
        return float(value) if value is not None else float("nan")

    def relative_gain(self, series_id: str, horizon: int, action: str) -> float:
        """Relative improvement of ``action`` over accepting, on validation."""
        base = self.score(series_id, horizon, ACCEPT)
        cand = self.score(series_id, horizon, action)
        if not (np.isfinite(base) and np.isfinite(cand)) or base <= 1e-12:
            return float("nan")
        return float((base - cand) / base)

    def log_gain(self, series_id: str, horizon: int, action: str,
                 epsilon: float = 1e-3) -> float:
        """Log accuracy ratio of ``action`` versus accepting, on validation.

        The scale the calibrator works on. The relative form divides by an error that
        is sometimes near zero and then explodes; see agents/calibration.py.
        """
        base = self.score(series_id, horizon, ACCEPT)
        cand = self.score(series_id, horizon, action)
        if not (np.isfinite(base) and np.isfinite(cand)):
            return float("nan")
        return float(np.log((base + epsilon) / (cand + epsilon)))

    def best_action(
        self, series_id: str, horizon: int, among: Sequence[str] | None = None
    ) -> tuple[str, float]:
        """Best validated action among ``among`` (default: all seen actions)."""
        pool = list(among) if among else self._actions
        scored = [
            (self.score(series_id, horizon, a), a)
            for a in pool
            if np.isfinite(self.score(series_id, horizon, a))
        ]
        if not scored:
            return ACCEPT, float("nan")
        scored.sort()
        return scored[0][1], scored[0][0]


class CorrectionPolicy(ABC):
    """Base class for correction policies."""

    name: str = "policy"
    uses_llm: bool = False

    @abstractmethod
    def decide_one(
        self,
        signals: pd.Series,
        evidence: ValidationEvidence,
        rng: np.random.Generator,
    ) -> PolicyDecision:
        """Decide for a single instance."""

    def decide(
        self,
        signals: pd.DataFrame,
        evidence: ValidationEvidence,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Decide for every instance; returns one row per instance."""
        rng = np.random.default_rng(seed)
        rows = [
            self.decide_one(row, evidence, rng)
            for _, row in signals.iterrows()
        ]
        out = pd.DataFrame([vars(d) for d in rows])
        out["policy"] = self.name
        return out


# --------------------------------------------------------------------------- #
# Non-adaptive arms
# --------------------------------------------------------------------------- #


class NeverCorrectPolicy(CorrectionPolicy):
    """Accept the initial forecast always.

    The behaviour of a conventional automated pipeline, and of an LLM agent that
    produces one forecast and stops. This is the arm H1 must beat.
    """

    name = "never_correct"

    def decide_one(self, signals, evidence, rng):
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=ACCEPT,
            corrected=False,
            rationale="no correction attempted",
        )


class AlwaysCorrectPolicy(CorrectionPolicy):
    """Correct every instance, picking the best validated repair.

    The Self-Refine / unconditional-reflection analogue, and deliberately the
    *strongest* version of it: rather than picking an arbitrary repair, it picks the
    one with the best validation evidence. Giving this baseline the benefit of the
    doubt matters -- if the adaptive policy wins against a weak always-correct
    baseline, the result says nothing.
    """

    name = "always_correct"

    def __init__(self, exclude: Sequence[str] = (ACCEPT,)):
        self.exclude = set(exclude)

    def decide_one(self, signals, evidence, rng):
        pool = [a for a in _all_actions(evidence) if a not in self.exclude]
        action, _ = evidence.best_action(
            signals["series_id"], signals["horizon"], among=pool
        )
        if action == ACCEPT and pool:
            action = pool[0]
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action,
            corrected=action != ACCEPT,
            n_candidates_evaluated=len(pool),
            compute_cost=float(len(pool)),
            rationale="always correct; best validated repair",
        )


class RandomCorrectPolicy(CorrectionPolicy):
    """Correct a fixed fraction of instances at random, with a random repair.

    The compute-matched control for H6. Its correction *rate* is set to match the
    adaptive policy's, so the two arms spend the same budget and any difference is
    attributable to *which* instances were chosen, not how many.
    """

    name = "random_correct"

    def __init__(self, rate: float = 0.30, exclude: Sequence[str] = (ACCEPT,)):
        self.rate = float(rate)
        self.exclude = set(exclude)

    def decide_one(self, signals, evidence, rng):
        pool = [a for a in _all_actions(evidence) if a not in self.exclude]
        if not pool or rng.random() >= self.rate:
            return PolicyDecision(
                instance_id=signals["instance_id"],
                chosen_action=ACCEPT,
                corrected=False,
                rationale="random gate: no correction",
            )
        action = str(rng.choice(pool))
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action,
            corrected=True,
            n_candidates_evaluated=1,
            compute_cost=1.0,
            rationale="random gate: random repair",
        )


class ThresholdPolicy(CorrectionPolicy):
    """Correct when a single reliability signal exceeds a threshold.

    The obvious, defensible non-learned baseline: watch backtest error and intervene
    when it looks bad. Included because much of the value of a learned gate would be
    illusory if one hand-picked signal did the same job.
    """

    name = "threshold_single"

    def __init__(self, signal: str = "backtest.backtest_mase",
                 threshold: float = 1.0,
                 exclude: Sequence[str] = (ACCEPT,)):
        self.signal = signal
        self.threshold = float(threshold)
        self.exclude = set(exclude)

    @classmethod
    def fit_threshold(
        cls, signals: pd.DataFrame, target_rate: float,
        signal: str = "backtest.backtest_mase",
    ) -> float:
        """Choose the threshold on validation to hit a target correction rate."""
        values = pd.to_numeric(signals.get(signal), errors="coerce").dropna()
        if values.empty:
            return float("inf")
        return float(np.quantile(values, 1.0 - float(np.clip(target_rate, 0, 1))))

    def decide_one(self, signals, evidence, rng):
        value = pd.to_numeric(pd.Series([signals.get(self.signal)]),
                              errors="coerce").iloc[0]
        if not np.isfinite(value) or value <= self.threshold:
            return PolicyDecision(
                instance_id=signals["instance_id"],
                chosen_action=ACCEPT,
                corrected=False,
                risk=float(value) if np.isfinite(value) else float("nan"),
                rationale=f"{self.signal}={value:.3f} <= {self.threshold:.3f}",
            )
        pool = [a for a in _all_actions(evidence) if a not in self.exclude]
        action, _ = evidence.best_action(
            signals["series_id"], signals["horizon"], among=pool
        )
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action if action != ACCEPT else (pool[0] if pool else ACCEPT),
            corrected=True,
            risk=float(value),
            n_candidates_evaluated=len(pool),
            compute_cost=float(len(pool)),
            rationale=f"{self.signal}={value:.3f} > {self.threshold:.3f}",
        )


class BestFixedPolicy(CorrectionPolicy):
    """Apply the single best validated action per series, unconditionally.

    The realizable-oracle baseline. It has perfect per-series knowledge but no
    per-instance adaptivity, so it is the bar an adaptive policy must clear to justify
    deciding instance by instance.
    """

    name = "best_fixed"

    def decide_one(self, signals, evidence, rng):
        action, score = evidence.best_action(signals["series_id"], signals["horizon"])
        return PolicyDecision(
            instance_id=signals["instance_id"],
            chosen_action=action,
            corrected=action != ACCEPT,
            n_candidates_evaluated=1,
            compute_cost=1.0,
            rationale=f"best validated fixed action (val metric {score:.3f})",
        )


# --------------------------------------------------------------------------- #
# The proposed policy
# --------------------------------------------------------------------------- #


@dataclass
class RGCPolicy(CorrectionPolicy):
    """Reliability-Gated Correction -- the proposed method.

    Four stages, each of which an ablation can switch off:

    1. **Gate.** A calibrated estimate of P(some repair beats accepting) is compared
       against a threshold chosen on validation to meet a correction budget. Gating on
       predicted *gain* rather than predicted *error* is deliberate: a forecast can be
       doomed and unfixable, and spending compute there is waste.
    2. **Diagnose.** Signals are mapped to a failure mode, which restricts candidates
       to matched repairs. This is where the compute saving in H4 comes from.
    3. **Validate.** Each candidate is scored on the validation origins of the same
       series. The correction is accepted only if it beats accepting by more than
       ``acceptance_delta``.
    4. **Abstain.** If risk remains high after correction, fall back and flag low
       confidence.

    Stage 3 is the crux. Without it the policy would be choosing repairs on belief; with
    it, every correction is backed by a measurement on data the agent already has. This
    is the concrete answer to the finding that self-correction fails when a model is
    left to judge its own output.
    """

    name: str = "rgc"
    gain_threshold: float = 0.5
    risk_threshold: float = 0.9
    acceptance_delta: float = 0.02
    max_candidates: int = 3
    use_diagnosis: bool = True
    use_validation: bool = True
    use_abstention: bool = True
    disabled_signals: tuple[str, ...] = ()
    risk_scores: dict[str, float] = field(default_factory=dict)
    gain_scores: dict[str, float] = field(default_factory=dict)

    def decide_one(self, signals, evidence, rng):
        iid = signals["instance_id"]
        gain = float(self.gain_scores.get(iid, np.nan))
        risk = float(self.risk_scores.get(iid, np.nan))

        # --- stage 1: gate ---
        if not np.isfinite(gain) or gain < self.gain_threshold:
            return PolicyDecision(
                instance_id=iid,
                chosen_action=ACCEPT,
                corrected=False,
                risk=risk,
                gain=gain,
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale=(
                    f"predicted gain {gain:.2f} < {self.gain_threshold:.2f}: "
                    "correction not expected to pay off"
                ),
            )

        # --- stage 2: diagnose and restrict candidates ---
        if self.use_diagnosis:
            dx = diagnose(signals.to_dict(), disabled=self.disabled_signals)
            candidates = candidate_actions(dx, max_actions=self.max_candidates)
            mode, evidence_text = dx.mode.value, "; ".join(dx.evidence[:3])
        else:
            # Ablation: no diagnosis means every repair is a candidate, which is the
            # exhaustive-search comparison H4 is about.
            candidates = [a for a in _all_actions(evidence) if a != ACCEPT]
            mode, evidence_text = "not_diagnosed", "diagnosis disabled"

        if not candidates:
            return PolicyDecision(
                instance_id=iid, chosen_action=ACCEPT, corrected=False,
                risk=risk, gain=gain, diagnosed_mode=mode,
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale="no candidate repair available",
            )

        # --- stage 3: validate candidates on held-out validation origins ---
        if self.use_validation:
            scored = [
                (evidence.relative_gain(signals["series_id"], signals["horizon"], a), a)
                for a in candidates
            ]
            scored = [(g, a) for g, a in scored if np.isfinite(g)]
            if not scored:
                chosen, validated_gain = candidates[0], float("nan")
            else:
                scored.sort(reverse=True)
                validated_gain, chosen = scored[0]
                # Reject the correction when the measurement does not support it.
                if validated_gain <= self.acceptance_delta:
                    return PolicyDecision(
                        instance_id=iid,
                        chosen_action=ACCEPT,
                        corrected=False,
                        risk=risk, gain=gain, diagnosed_mode=mode,
                        n_candidates_evaluated=len(candidates),
                        compute_cost=float(len(candidates)),
                        confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                        rationale=(
                            f"diagnosed {mode}; best candidate {chosen} improved "
                            f"validation by only {validated_gain:.1%} "
                            f"(<= {self.acceptance_delta:.1%}), correction rejected"
                        ),
                    )
        else:
            chosen, validated_gain = candidates[0], float("nan")

        # --- stage 4: abstain when risk stays high ---
        if self.use_abstention and np.isfinite(risk) and risk > self.risk_threshold:
            chosen = "abstain"

        return PolicyDecision(
            instance_id=iid,
            chosen_action=chosen,
            corrected=chosen != ACCEPT,
            risk=risk,
            gain=gain,
            confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
            diagnosed_mode=mode,
            n_candidates_evaluated=len(candidates),
            compute_cost=float(len(candidates)),
            n_tool_calls=len(candidates),
            rationale=(
                f"diagnosed {mode} ({evidence_text}); applied {chosen}; "
                f"validated gain {validated_gain:.1%}"
            ),
        )


# --------------------------------------------------------------------------- #


def _all_actions(evidence: ValidationEvidence) -> list[str]:
    from benchmark.actions import ACTION_ORDER

    return list(ACTION_ORDER)


@dataclass
class CalibratedGainPolicy(CorrectionPolicy):
    """Correction gated on a calibrated, risk-averse estimate of what a repair is worth.

    This is the corrected form of reliability-gated correction, and it differs from
    :class:`RGCPolicy` in what the gate is asking.

    ``RGCPolicy`` asks *did this repair help on recent history, by more than delta?*
    That question ignores three properties of the data this benchmark exposes: observed
    gain is a biased estimate of future gain, transfer is action-specific, and the loss
    is heavy-tailed so a rare catastrophic repair outweighs many small wins.

    This policy asks instead: *under a pessimistic reading of the evidence, is this
    repair still worth making?* Concretely, for each candidate it takes the gain
    observed on validation origins, maps it through a per-action calibration learned
    inside validation, subtracts ``risk_z`` residual standard deviations, and acts only
    if that lower bound clears ``margin``. Actions with no demonstrated transfer are
    excluded outright rather than being gated on their own noise.

    The result is a policy that acts rarely. That is the intended behaviour: the
    benchmark's own counterfactuals show most repairs are harmful most of the time, so a
    policy that acts often cannot be right.
    """

    name: str = "rgc_calibrated"
    calibrator: object = None          # agents.calibration.GainCalibrator
    margin: float = 0.0
    risk_z: float = 1.0
    max_candidates: int = 4
    use_diagnosis: bool = True
    disabled_signals: tuple[str, ...] = ()
    risk_scores: dict[str, float] = field(default_factory=dict)

    def decide_one(self, signals, evidence, rng):
        iid = signals["instance_id"]
        series_id, horizon = signals["series_id"], signals["horizon"]
        risk = float(self.risk_scores.get(iid, np.nan))

        if self.calibrator is None:
            raise RuntimeError("CalibratedGainPolicy requires a fitted GainCalibrator")

        # Restrict to repairs matched to the diagnosed failure mode, then further to
        # those the calibrator found transferable at all.
        if self.use_diagnosis:
            dx = diagnose(signals.to_dict(), disabled=self.disabled_signals)
            candidates = candidate_actions(dx, max_actions=self.max_candidates)
            mode = dx.mode.value
        else:
            candidates = [a for a in _all_actions(evidence) if a != ACCEPT]
            mode = "not_diagnosed"

        trusted = set(self.calibrator.trusted_actions())
        candidates = [a for a in candidates if a in trusted]

        if not candidates:
            return PolicyDecision(
                instance_id=iid, chosen_action=ACCEPT, corrected=False, risk=risk,
                diagnosed_mode=mode,
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale=(
                    f"diagnosed {mode}; no matched repair has demonstrated transferable "
                    "value, so accepting"
                ),
            )

        scored: list[tuple[float, float, str]] = []
        for action in candidates:
            observed = evidence.log_gain(series_id, horizon, action)
            if not np.isfinite(observed):
                continue
            bound = self.calibrator.lower_bound(action, observed, self.risk_z)
            if np.isfinite(bound):
                scored.append((bound, observed, action))

        if not scored:
            return PolicyDecision(
                instance_id=iid, chosen_action=ACCEPT, corrected=False, risk=risk,
                diagnosed_mode=mode, n_candidates_evaluated=len(candidates),
                compute_cost=float(len(candidates)),
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale="no candidate had usable validation evidence",
            )

        scored.sort(reverse=True)
        bound, observed, action = scored[0]

        if bound <= self.margin:
            return PolicyDecision(
                instance_id=iid, chosen_action=ACCEPT, corrected=False, risk=risk,
                gain=bound, diagnosed_mode=mode,
                n_candidates_evaluated=len(candidates),
                compute_cost=float(len(candidates)),
                confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
                rationale=(
                    f"diagnosed {mode}; best candidate {action} observed "
                    f"{observed:+.1%} on validation, but its calibrated lower bound is "
                    f"{bound:+.1%} <= {self.margin:+.1%}, so the repair is not worth "
                    "making"
                ),
            )

        return PolicyDecision(
            instance_id=iid, chosen_action=action, corrected=True, risk=risk,
            gain=bound, diagnosed_mode=mode,
            n_candidates_evaluated=len(candidates),
            compute_cost=float(len(candidates)),
            n_tool_calls=len(candidates),
            confidence=1.0 - (risk if np.isfinite(risk) else 0.5),
            rationale=(
                f"diagnosed {mode}; {action} observed {observed:+.1%} on validation, "
                f"calibrated lower bound {bound:+.1%} > {self.margin:+.1%}"
            ),
        )
