"""The reliability estimator: a calibrated model of when a forecast will fail.

This is the core of contribution C2, and the component that distinguishes this work
from the LLM-verbal-critique approach that the self-correction literature has shown to
be unreliable (Huang et al., ICLR 2024). The agent does not ask itself whether it
feels confident. It computes a calibrated probability that its forecast is about to
fail, from measurable diagnostics, and gates correction on that number.

The estimator extends offline forecast-stress prediction (FFORMPP; MAST) into a
*control* signal. Three properties are required for that extension to be legitimate:

1. **Fit on validation only.** Never on test, never on the instance being decided.
2. **Features available at decision time.** Every input comes from history-only
   diagnostics, so the estimator is deployable rather than retrospective.
3. **Calibrated output.** A raw classifier score is not a probability, and an
   uncalibrated gate cannot be thresholded to a target correction budget. Isotonic
   regression on held-out validation folds makes the output mean something.

Two heads are fitted:

``risk``
    P(the accepted forecast will be a large error). Used for abstention and reported
    for H3/H5.
``gain``
    P(some corrective action would beat accepting). This is what actually gates
    correction. Keeping the two apart matters: a forecast can be doomed but
    unfixable (a change point inside the horizon), and spending compute to correct it
    is waste. Gating on predicted *gain* rather than predicted *error* is the
    difference between a useful policy and an expensive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from core.logging import get_logger

log = get_logger("reliability")

__all__ = ["ReliabilityEstimator", "FEATURE_PREFIXES", "FEATURES_ONLY_PREFIXES"]

#: Signal families usable as model inputs. Anything outside these prefixes is
#: identity or ground-truth metadata and must never reach the model.
FEATURE_PREFIXES: tuple[str, ...] = (
    "ts_features.",
    "stl.",
    "acf.",
    "seasonality.",
    "stationarity.",
    "changepoint.",
    "anomaly.",
    "distribution_shift.",
    "backtest.",
    "uncertainty.",
    "disagreement.",
    "selection.",
    "context.",
)

#: The FFORMPP-style control condition for H3: series descriptors alone, with no
#: forecast-specific evidence. The full estimator must beat this to claim that
#: forecast diagnostics add anything over knowing what kind of series it is.
FEATURES_ONLY_PREFIXES: tuple[str, ...] = ("ts_features.", "context.")

#: Columns that must never be used as features, enforced at fit time.
_FORBIDDEN = ("label_", "instance_id", "series_id", "dataset", "family", "source",
              "split", "base_model", "mase_denominator", "origin")


@dataclass
class ReliabilityEstimator:
    """Gradient-boosted, isotonic-calibrated failure and gain predictors."""

    ablate_signals: tuple[str, ...] = ()
    feature_prefixes: tuple[str, ...] = FEATURE_PREFIXES
    n_calibration_folds: int = 3
    random_state: int = 0

    feature_names_: list[str] = field(default_factory=list, init=False)
    risk_model_: object | None = field(default=None, init=False)
    gain_model_: object | None = field(default=None, init=False)
    risk_calibrator_: object | None = field(default=None, init=False)
    gain_calibrator_: object | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)
    base_rates_: dict[str, float] = field(default_factory=dict, init=False)

    # -- features ----------------------------------------------------------- #

    def select_features(self, signals: pd.DataFrame) -> list[str]:
        """Choose numeric feature columns, honouring ablation switches.

        Ablations remove whole signal families (e.g. ``changepoint.``) so that the
        paper can attribute gains to specific evidence rather than to the estimator in
        aggregate.
        """
        cols: list[str] = []
        for col in signals.columns:
            if any(col.startswith(bad) for bad in _FORBIDDEN):
                continue
            if not col.startswith(self.feature_prefixes):
                continue
            if any(col.startswith(ab) for ab in self.ablate_signals):
                continue
            if not pd.api.types.is_numeric_dtype(signals[col]):
                continue
            cols.append(col)
        return sorted(cols)

    def _matrix(self, signals: pd.DataFrame) -> np.ndarray:
        X = signals.reindex(columns=self.feature_names_).to_numpy(dtype=float)
        # Trees handle NaN natively in HistGradientBoosting, but infinities do not
        # survive the split search; map them to NaN so they are treated as missing.
        return np.where(np.isfinite(X), X, np.nan)

    # -- fitting ------------------------------------------------------------ #

    def fit(self, signals: pd.DataFrame, labels: pd.DataFrame) -> "ReliabilityEstimator":
        """Fit both heads on the validation split.

        ``signals`` and ``labels`` are joined on ``instance_id``. The caller is
        responsible for passing validation rows only; :meth:`fit` asserts this to make
        an accidental test-set fit loud rather than silent.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier

        data = signals.merge(
            labels[["instance_id", "large_error", "should_correct", "split"]],
            on="instance_id",
            how="inner",
            suffixes=("", "_label"),
        )
        if data.empty:
            raise ValueError("no overlapping instances between signals and labels")

        split_col = "split_label" if "split_label" in data.columns else "split"
        bad = set(data[split_col].unique()) - {"val"}
        if bad:
            raise ValueError(
                f"reliability estimator must be fitted on validation instances only; "
                f"received splits {sorted(bad)}. This guard exists because fitting on "
                f"test would invalidate every downstream result."
            )

        self.feature_names_ = self.select_features(signals)
        if not self.feature_names_:
            raise ValueError("no usable features after ablation")

        X = self._matrix(data)
        y_risk = data["large_error"].astype(int).to_numpy()
        y_gain = data["should_correct"].astype(int).to_numpy()
        self.base_rates_ = {
            "large_error": float(y_risk.mean()),
            "should_correct": float(y_gain.mean()),
        }

        def _fit_head(y: np.ndarray, tag: str):
            # A degenerate label (all one class) cannot be modelled; fall back to the
            # base rate rather than fitting something meaningless.
            if len(np.unique(y)) < 2:
                log.warning("head %r has a single class; falling back to base rate", tag)
                return None
            base = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.06,
                l2_regularization=1.0,
                min_samples_leaf=10,
                random_state=self.random_state,
            )
            n_folds = int(min(self.n_calibration_folds, np.bincount(y).min()))
            if n_folds < 2:
                base.fit(X, y)
                return base
            # Isotonic calibration on cross-validated folds: the raw score ordering is
            # usually good while its scale is not, and the threshold machinery below
            # needs a number that behaves like a probability.
            model = CalibratedClassifierCV(
                base, method="isotonic", cv=n_folds, ensemble=True
            )
            model.fit(X, y)
            return model

        self.risk_model_ = _fit_head(y_risk, "risk")
        self.gain_model_ = _fit_head(y_gain, "gain")
        self.fitted_ = True
        log.info(
            "reliability estimator fitted on %d validation instances, %d features "
            "(base rates: large_error=%.3f, should_correct=%.3f)",
            len(data), len(self.feature_names_),
            self.base_rates_["large_error"], self.base_rates_["should_correct"],
        )
        return self

    # -- prediction --------------------------------------------------------- #

    def _predict(self, model, signals: pd.DataFrame, fallback: float) -> np.ndarray:
        if model is None:
            return np.full(len(signals), fallback, dtype=float)
        return model.predict_proba(self._matrix(signals))[:, 1]

    def predict_risk(self, signals: pd.DataFrame) -> np.ndarray:
        """Calibrated P(large error) for each instance."""
        self._check_fitted()
        return self._predict(
            self.risk_model_, signals, self.base_rates_.get("large_error", 0.2)
        )

    def predict_gain(self, signals: pd.DataFrame) -> np.ndarray:
        """Calibrated P(some corrective action beats accepting)."""
        self._check_fitted()
        return self._predict(
            self.gain_model_, signals, self.base_rates_.get("should_correct", 0.3)
        )

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("ReliabilityEstimator is not fitted")

    # -- thresholding ------------------------------------------------------- #

    @staticmethod
    def threshold_for_rate(scores: np.ndarray, target_rate: float) -> float:
        """Threshold that corrects approximately ``target_rate`` of instances.

        A budget-first formulation. Rather than picking an arbitrary probability
        cut-off, we fix the compute budget and let the data decide the cut-off -- which
        is also what makes the compute-matched comparison against ``random_correct``
        (H6) exactly fair, since both arms then correct the same number of instances.
        """
        scores = np.asarray(scores, dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return 1.0
        rate = float(np.clip(target_rate, 0.0, 1.0))
        return float(np.quantile(finite, 1.0 - rate))

    def feature_importance(self, signals: pd.DataFrame, labels: pd.DataFrame,
                           head: str = "gain", n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance, for the error analysis section.

        Permutation rather than split-gain importance: gain-based importances are
        biased toward high-cardinality features and are not comparable across the
        heterogeneous signal families used here.
        """
        from sklearn.inspection import permutation_importance

        self._check_fitted()
        model = self.gain_model_ if head == "gain" else self.risk_model_
        if model is None:
            return pd.DataFrame(columns=["feature", "importance", "std"])

        target = "should_correct" if head == "gain" else "large_error"
        data = signals.merge(labels[["instance_id", target]], on="instance_id")
        result = permutation_importance(
            model, self._matrix(data), data[target].astype(int).to_numpy(),
            n_repeats=n_repeats, random_state=self.random_state, scoring="roc_auc",
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
