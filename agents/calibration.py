"""Calibrating what a corrective action is actually worth.

This module exists because of a measurement failure we found the hard way, and it is
the core of the corrected method.

The naive policy asks: *did this repair improve things on recent history?* and acts if
the answer is yes by some small margin. That question is the wrong one, for three
reasons this codebase can now demonstrate:

1. **Observed gain is a biased estimate of future gain.** Regressing realised test gain
   on validation gain gives a slope well below one and a strongly negative intercept:
   most of an apparent improvement does not survive to the next window. With a 2%
   acceptance margin the policy acts constantly; the break-even margin is closer to 70%.

2. **Transfer is action-specific.** Some repairs carry genuine signal from one window to
   the next (damped trend, abstention, seasonal re-specification). Others carry
   essentially none (ensembling, robustification, model switching): their observed gain
   on validation is noise, and acting on it is a coin flip. A single global threshold
   treats both alike.

3. **The loss is heavy-tailed.** A rare catastrophic correction costs more than many
   small wins earn. A policy that maximises *expected* gain will still degrade the mean,
   so the gate has to be risk-averse rather than merely positive-in-expectation.

Gain is measured as a **log ratio**, ``log(accept_error / action_error)``, not as the
relative difference ``(accept - action) / accept``. This is not cosmetic. MASE in this
benchmark spans 0.000 to 187, and the relative form divides by an error that is
sometimes near zero, producing gain values in the hundreds. Those few exploded points
dominate every fit: on the relative scale the nested correlation between observed and
realised gain is 0.006 with residual standard deviations above 1000, i.e. no usable
signal at all. On the log scale the same data gives a correlation of 0.43 with residual
standard deviations under 0.7, and a clear per-action structure emerges. The log ratio
is also the natural scale for error ratios -- it is symmetric (halving and doubling the
error are equal and opposite) and it corresponds to the geometric mean.

:class:`GainCalibrator` addresses all three. It learns, per action, the mapping from
observed gain to realised future gain, using a split *inside* the validation window --
early origins estimate, the latest origin measures. It reports a lower confidence bound
rather than a point estimate, so the gate can require that a repair be worth making even
under a pessimistic reading of the evidence.

Everything here is fitted on validation data only. The test split is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.logging import get_logger

log = get_logger("calibration")

__all__ = ["GainCalibrator", "ActionCalibration"]


@dataclass
class ActionCalibration:
    """The learned value of one corrective action."""

    action: str
    slope: float
    intercept: float
    residual_sd: float
    correlation: float
    n: int
    trusted: bool

    def predict(self, observed_gain: float) -> float:
        """Expected future gain given an observed gain on recent history."""
        if not np.isfinite(observed_gain):
            return float("nan")
        if not self.trusted:
            # No demonstrated transfer: the best available estimate of this action's
            # future value is its unconditional mean, which is what intercept holds
            # when slope is pinned to zero.
            return self.intercept
        return self.slope * observed_gain + self.intercept

    def lower_bound(self, observed_gain: float, z: float = 1.0) -> float:
        """Pessimistic estimate: expected gain minus ``z`` residual standard deviations.

        Using a lower bound rather than the point estimate is what protects the mean.
        An action whose expected gain is slightly positive but whose residual spread is
        large will occasionally be catastrophic, and on a heavy-tailed loss those rare
        events dominate.
        """
        predicted = self.predict(observed_gain)
        return predicted - z * self.residual_sd if np.isfinite(predicted) else float("nan")


@dataclass
class GainCalibrator:
    """Per-action calibration of observed gain to realised future gain.

    Parameters
    ----------
    min_samples:
        Minimum nested pairs before an action is modelled at all.
    min_correlation:
        Minimum observed-to-realised correlation for an action to be *trusted*, i.e.
        for its observed gain to be used at all. Below this the action's gain is noise,
        and the calibrator falls back to that action's unconditional mean.
    """

    min_samples: int = 40
    min_correlation: float = 0.20
    #: Number of early validation origins used to *estimate* gain; the remainder
    #: measure it. A 2/2 split balances the noise in both halves -- measuring on a
    #: single origin attenuates the correlation badly (0.35 vs 0.43 here).
    n_estimate_origins: int = 2
    #: Added to both errors before the log, so a zero error cannot produce -inf.
    epsilon: float = 1e-3
    calibrations: dict[str, ActionCalibration] = field(default_factory=dict)
    fitted_: bool = False

    # -- fitting ------------------------------------------------------------ #

    def _log_gain(self, accept_error: float, action_error: float) -> float:
        """Log accuracy ratio: positive means the action reduced error."""
        return float(
            np.log((accept_error + self.epsilon) / (action_error + self.epsilon))
        )

    def _nested_pairs(self, val_outcomes: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Build (observed gain, realised gain) pairs entirely within validation.

        For each (series, horizon) the earliest origins estimate each action's gain and
        the latest origin measures what that gain turned out to be. This mirrors, inside
        the validation window, exactly the extrapolation the policy must perform from
        validation to test -- which is what makes the resulting calibration honest
        rather than an in-sample fit.
        """
        df = val_outcomes.copy()
        df.loc[~df["applicable"].astype(bool), metric] = np.nan

        rows: list[dict] = []
        for (series_id, horizon), group in df.groupby(["series_id", "horizon"],
                                                      sort=False):
            origins = sorted(group["origin"].unique())
            n_est = min(self.n_estimate_origins, len(origins) - 1)
            if n_est < 1:
                continue
            estimate = group[group["origin"].isin(origins[:n_est])]
            measure = group[group["origin"].isin(origins[n_est:])]

            accept_est = estimate.loc[estimate["action"] == "accept", metric].mean()
            accept_meas = measure.loc[measure["action"] == "accept", metric].mean()
            if not (np.isfinite(accept_est) and np.isfinite(accept_meas)):
                continue

            for action in estimate["action"].unique():
                if action == "accept":
                    continue
                m_est = estimate.loc[estimate["action"] == action, metric].mean()
                m_meas = measure.loc[measure["action"] == action, metric].mean()
                if not (np.isfinite(m_est) and np.isfinite(m_meas)):
                    continue
                rows.append(
                    {
                        "action": action,
                        "observed": self._log_gain(accept_est, m_est),
                        "realised": self._log_gain(accept_meas, m_meas),
                    }
                )
        return pd.DataFrame(rows)

    def fit(self, val_outcomes: pd.DataFrame, metric: str = "mase") -> "GainCalibrator":
        pairs = self._nested_pairs(val_outcomes, metric)
        if pairs.empty:
            raise ValueError(
                "no nested validation pairs; the split needs at least two validation "
                "origins per series and horizon"
            )

        for action, group in pairs.groupby("action"):
            observed = group["observed"].to_numpy(dtype=float)
            realised = group["realised"].to_numpy(dtype=float)
            ok = np.isfinite(observed) & np.isfinite(realised)
            observed, realised = observed[ok], realised[ok]
            n = observed.size

            if n < self.min_samples or observed.std() < 1e-9:
                self.calibrations[action] = ActionCalibration(
                    action, 0.0, float(np.mean(realised)) if n else -1.0,
                    float(np.std(realised)) if n else 1.0, 0.0, int(n), False,
                )
                continue

            correlation = float(np.corrcoef(observed, realised)[0, 1])
            # Theil-Sen rather than least squares. Gain ratios produce extreme
            # outliers (a near-zero denominator sends one point to -30), and a single
            # such point flips an OLS slope to strongly negative -- which is precisely
            # what happened with the ensemble and model-switch actions.
            slope, intercept = _theil_sen(observed, realised)
            residuals = realised - (slope * observed + intercept)
            residual_sd = float(np.std(residuals))

            trusted = bool(
                np.isfinite(correlation)
                and correlation >= self.min_correlation
                and slope > 0.0
            )
            if not trusted:
                slope, intercept = 0.0, float(np.median(realised))
                residual_sd = float(np.std(realised))

            self.calibrations[action] = ActionCalibration(
                action, float(slope), float(intercept), residual_sd,
                correlation if np.isfinite(correlation) else 0.0, int(n), trusted,
            )

        self.fitted_ = True
        trusted_actions = [a for a, c in self.calibrations.items() if c.trusted]
        log.info(
            "gain calibrator fitted on %d nested pairs; %d/%d actions show transferable "
            "signal: %s",
            len(pairs), len(trusted_actions), len(self.calibrations),
            sorted(trusted_actions),
        )
        return self

    # -- use ---------------------------------------------------------------- #

    def get(self, action: str) -> ActionCalibration:
        return self.calibrations.get(
            action, ActionCalibration(action, 0.0, -1.0, 1.0, 0.0, 0, False)
        )

    def predicted_gain(self, action: str, observed_gain: float) -> float:
        return self.get(action).predict(observed_gain)

    def lower_bound(self, action: str, observed_gain: float, z: float = 1.0) -> float:
        return self.get(action).lower_bound(observed_gain, z)

    def trusted_actions(self) -> list[str]:
        return sorted(a for a, c in self.calibrations.items() if c.trusted)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "action": c.action, "n": c.n, "correlation": c.correlation,
                    "slope": c.slope, "intercept": c.intercept,
                    "residual_sd": c.residual_sd, "trusted": c.trusted,
                    "breakeven_observed_gain": (
                        -c.intercept / c.slope if c.trusted and c.slope > 1e-9
                        else float("inf")
                    ),
                }
                for c in self.calibrations.values()
            ]
        ).sort_values("correlation", ascending=False).reset_index(drop=True)


def _theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 20000,
               seed: int = 0) -> tuple[float, float]:
    """Robust line fit: median of pairwise slopes.

    Subsamples pairs when the full O(n^2) set would be large, which keeps the fit cheap
    without materially changing the median.
    """
    n = x.size
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-12
    if not ok.any():
        return 0.0, float(np.median(y))
    slopes = (y[j][ok] - y[i][ok]) / dx[ok]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept
