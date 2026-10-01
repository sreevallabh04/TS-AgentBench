"""The experiment runner.

One entry point drives the whole pipeline: build data, build the counterfactual
tensor, derive oracle labels, fit the reliability estimator on validation, replay every
correction policy, score, and run the pre-registered statistical tests.

Two properties are deliberate.

**The tensor is cached by content hash.** Rebuilding it is the only expensive step.
Keyed on the data and model configuration (not the policy configuration), so every
ablation and policy sweep after the first run is near-instant. This is what makes the
full experimental grid affordable on a CPU.

**Seeds vary the policy, not the tensor.** Model-internal randomness is fixed when the
tensor is built; the seeds in the config vary stochastic *policies* (the random-correct
arm), the bootstrap, and subsampling. Rebuilding the tensor per seed would multiply
the cost by five for a source of variation that is not what the paper is about.
Sensitivity to model-internal seeds is measured separately by
``scripts/seed_sensitivity.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import ExperimentConfig, config_hash
from core.logging import JsonlWriter, get_logger, setup_logging
from core.manifest import RunManifest
from core.paths import TENSOR_CACHE_DIR, ensure_dirs, run_dir
from core.seed import set_global_seed

log = get_logger("runner")

__all__ = ["run_experiment", "build_dataset", "ExperimentResult"]


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #


def build_dataset(cfg: ExperimentConfig, seed: int = 0):
    """Assemble the series list and the per-series instance lists."""
    from datasets.base import make_split, iter_instances
    from datasets.synthetic import family_names, generate_family

    series = []

    if cfg.data.include_synthetic:
        families = cfg.data.synthetic_families
        if families == ["all"] or not families:
            families = family_names()
        for family in families:
            series += generate_family(
                family,
                cfg.data.series_per_family,
                base_seed=seed,
                n_timesteps=cfg.data.max_series_length // 2,
                seasonal_period=12,
            )

    if cfg.data.include_real and cfg.data.real_datasets:
        from datasets.real import load_real_dataset

        for key in cfg.data.real_datasets:
            try:
                series += load_real_dataset(
                    key,
                    max_series=cfg.data.max_series_per_dataset,
                    max_length=cfg.data.max_series_length,
                    seed=seed,
                )
            except FileNotFoundError as exc:
                # Missing data is a setup problem, not a silent degradation. Warn
                # loudly so a run is never quietly reported as "real + synthetic"
                # when it was synthetic only.
                log.warning("skipping %s: %s", key, exc)

    instances: dict[str, list] = {}
    dropped = 0
    for s in series:
        try:
            split = make_split(
                len(s), cfg.split.val_length, cfg.split.test_length,
                cfg.split.min_train_length,
            )
        except ValueError:
            dropped += 1
            continue
        items = list(
            iter_instances(
                s, split, cfg.forecast.horizons,
                cfg.split.n_val_origins, cfg.split.n_test_origins,
                cfg.split.origin_stride,
            )
        )
        if items:
            instances[s.series_id] = items

    series = [s for s in series if s.series_id in instances]
    n_instances = sum(len(v) for v in instances.values())
    log.info(
        "dataset: %d series, %d instances (%d series dropped as too short)",
        len(series), n_instances, dropped,
    )
    return series, instances


# --------------------------------------------------------------------------- #
# Tensor caching
# --------------------------------------------------------------------------- #


def _tensor_key(cfg: ExperimentConfig, seed: int) -> str:
    """Hash only the inputs that affect the tensor, not the policy configuration."""
    import hashlib

    payload = json.dumps(
        {
            "data": asdict(cfg.data),
            "split": asdict(cfg.split),
            "forecast": asdict(cfg.forecast),
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_tensor(cfg: ExperimentConfig, seed: int, force: bool = False):
    """Build the counterfactual tensor, or load it from cache."""
    from benchmark.tensor import build_tensor

    TENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _tensor_key(cfg, seed)
    out_path = TENSOR_CACHE_DIR / f"outcomes_{key}.parquet"
    sig_path = TENSOR_CACHE_DIR / f"signals_{key}.parquet"

    if out_path.exists() and sig_path.exists() and not force:
        log.info("loading cached tensor %s", key)
        return pd.read_parquet(out_path), pd.read_parquet(sig_path)

    series, instances = build_dataset(cfg, seed)
    if not series:
        raise RuntimeError(
            "no usable series. Check data.include_synthetic / real_datasets, and run "
            "scripts/prepare_data.py if real datasets are configured."
        )

    models = list(cfg.forecast.models)
    t0 = time.perf_counter()
    outcomes, signals = build_tensor(
        series, instances, models,
        coverage_levels=tuple(cfg.forecast.coverage_levels),
        base_seed=seed, n_jobs=cfg.n_jobs,
        base_selection=cfg.forecast.base_selection,
    )
    log.info("tensor built in %.0fs", time.perf_counter() - t0)
    outcomes.to_parquet(out_path)
    signals.to_parquet(sig_path)
    return outcomes, signals


# --------------------------------------------------------------------------- #
# Policy construction
# --------------------------------------------------------------------------- #


def build_policies(
    cfg: ExperimentConfig,
    gain: dict[str, float],
    risk: dict[str, float],
    gain_threshold: float,
    threshold_value: float,
    correction_rate: float,
    llm_client=None,
    calibrator=None,
) -> list:
    """Instantiate the arms named in the config."""
    from agents.policies import (
        AlwaysCorrectPolicy, BestFixedPolicy, CalibratedGainPolicy,
        NeverCorrectPolicy, RandomCorrectPolicy, RGCPolicy, ThresholdPolicy,
    )

    ablations = {
        "use_self_critique": cfg.agent.use_self_critique,
        "use_uncertainty": cfg.agent.use_uncertainty,
        "use_model_disagreement": cfg.agent.use_model_disagreement,
        "use_changepoint": cfg.agent.use_changepoint,
        "use_anomaly": cfg.agent.use_anomaly,
        "use_adaptive_tools": cfg.agent.use_adaptive_tools,
        "use_backtesting": cfg.agent.use_backtesting,
    }
    # Ablations remove whole evidence families from the diagnoser.
    disabled: list[str] = []
    if not ablations["use_changepoint"]:
        disabled.append("changepoint.")
    if not ablations["use_anomaly"]:
        disabled.append("anomaly.")
    if not ablations["use_model_disagreement"]:
        disabled.append("disagreement.")
    if not ablations["use_uncertainty"]:
        disabled.append("uncertainty.")
    if not ablations["use_backtesting"]:
        disabled.append("backtest.")

    registry = {
        "baselineA_automl": lambda: NeverCorrectPolicy(),
        "never_correct": lambda: NeverCorrectPolicy(),
        "always_correct": lambda: AlwaysCorrectPolicy(),
        "random_correct": lambda: RandomCorrectPolicy(rate=correction_rate),
        "threshold_single": lambda: ThresholdPolicy(threshold=threshold_value),
        "best_fixed": lambda: BestFixedPolicy(),
        "rgc_calibrated": lambda: CalibratedGainPolicy(
            calibrator=calibrator,
            margin=cfg.agent.calibrated_margin,
            risk_z=cfg.agent.calibrated_risk_z,
            use_diagnosis=False,
            disabled_signals=tuple(disabled),
            risk_scores=risk,
        ),
        "rgc": lambda: RGCPolicy(
            gain_threshold=gain_threshold,
            gain_scores=gain,
            risk_scores=risk,
            acceptance_delta=cfg.agent.acceptance_delta,
            use_diagnosis=cfg.agent.use_adaptive_tools,
            use_validation=cfg.agent.use_backtesting,
            use_abstention=cfg.agent.use_uncertainty,
            disabled_signals=tuple(disabled),
        ),
    }

    if llm_client is not None:
        from agents.llm_agent import (
            LLMCritiqueGatePolicy, LLMSinglePolicy, RGCLLMRouterPolicy,
        )

        registry.update(
            {
                "baselineB_llm_single": lambda: LLMSinglePolicy(client=llm_client),
                "llm_single": lambda: LLMSinglePolicy(client=llm_client),
                "llm_critique_gate": lambda: LLMCritiqueGatePolicy(client=llm_client),
                "rgc_llm_router": lambda: RGCLLMRouterPolicy(
                    client=llm_client,
                    gain_threshold=gain_threshold,
                    gain_scores=gain,
                    risk_scores=risk,
                    disabled_signals=tuple(disabled),
                ),
            }
        )

    if calibrator is None:
        registry.pop("rgc_calibrated", None)

    policies = []
    for arm in cfg.agent.arms:
        if arm == "oracle_correct":
            continue  # handled separately; it is a ceiling, not a competitor
        factory = registry.get(arm)
        if factory is None:
            log.warning("arm %r is not available (LLM client missing?); skipping", arm)
            continue
        policies.append(factory())
    return policies


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


class ExperimentResult(dict):
    """Result bundle: DataFrames plus the run directory."""


def run_experiment(
    cfg: ExperimentConfig,
    force_tensor: bool = False,
    run_id: str | None = None,
) -> ExperimentResult:
    """Run one full experiment and write every artifact to ``results/runs/<run_id>``."""
    from agents.policies import ValidationEvidence, ThresholdPolicy
    from agents.reliability import ReliabilityEstimator
    from benchmark.oracle import (
        add_oracle_labels, add_realizable_oracle, compute_policy_regret,
    )
    from evaluation.statistics import compare_policies

    ensure_dirs()
    manifest = RunManifest.create(cfg, run_id)
    directory = run_dir(manifest.run_id)
    setup_logging(directory / "run.log")
    events = JsonlWriter(directory / "events.jsonl")
    log.info("run %s (config hash %s)", manifest.run_id, config_hash(cfg))
    manifest.save()

    try:
        primary = cfg.eval.primary_metric
        base_seed = cfg.seeds[0]
        set_global_seed(base_seed)

        # --- 1. counterfactual tensor ---
        outcomes, signals = get_tensor(cfg, base_seed, force=force_tensor)
        events.write("tensor_built", n_rows=len(outcomes), n_instances=len(signals))

        val_outcomes = outcomes[outcomes["split"] == "val"]
        test_outcomes = outcomes[outcomes["split"] == "test"]
        if val_outcomes.empty or test_outcomes.empty:
            raise RuntimeError("tensor is missing a validation or test split")

        # --- 2. oracle labels (evaluation only) ---
        # NOTE: oracle labels use eval.oracle_delta, never agent.acceptance_delta.
        # The label is ground truth and must not move when a method is retuned.
        val_labels = add_oracle_labels(
            val_outcomes, primary, cfg.eval.oracle_delta,
            cfg.eval.large_error_quantile,
        )
        test_labels = add_oracle_labels(
            test_outcomes, primary, cfg.eval.oracle_delta,
            cfg.eval.large_error_quantile,
        )
        realizable = add_realizable_oracle(outcomes, primary)

        val_signals = signals[signals["instance_id"].isin(val_labels["instance_id"])]
        test_signals = signals[signals["instance_id"].isin(test_labels["instance_id"])]

        # --- 3. reliability estimator, fitted on validation only ---
        estimator = ReliabilityEstimator(random_state=base_seed).fit(
            val_signals, val_labels
        )
        gain_scores = estimator.predict_gain(test_signals)
        risk_scores = estimator.predict_risk(test_signals)
        gain = dict(zip(test_signals["instance_id"], gain_scores))
        risk = dict(zip(test_signals["instance_id"], risk_scores))

        # Thresholds are chosen on validation predictions, never on test.
        val_gain = estimator.predict_gain(val_signals)
        gain_threshold = ReliabilityEstimator.threshold_for_rate(
            val_gain, cfg.agent.target_correction_rate
        )
        threshold_value = ThresholdPolicy.fit_threshold(
            val_signals, cfg.agent.target_correction_rate
        )
        events.write(
            "estimator_fitted",
            n_features=len(estimator.feature_names_),
            gain_threshold=float(gain_threshold),
            **estimator.base_rates_,
        )

        # --- 4. LLM client, if any arm needs it ---
        llm_client = None
        llm_arms = {"baselineB_llm_single", "llm_single", "llm_critique_gate",
                    "rgc_llm_router"}
        if set(cfg.agent.arms) & llm_arms:
            import os

            from dotenv import load_dotenv

            from agents.llm import LLMClient

            load_dotenv()
            # Without a key and without a populated cache, the LLM arms cannot run.
            # Skipping them loudly is better than failing three hours into a run, and
            # far better than reporting a partial experiment as if it were complete.
            if not os.environ.get("GEMINI_API_KEY", "").strip() and not cfg.llm.cache_only:
                log.warning(
                    "GEMINI_API_KEY is not set; skipping LLM arms %s. The non-LLM "
                    "results remain valid, but do not report this run as covering "
                    "LLM baselines.",
                    sorted(set(cfg.agent.arms) & llm_arms),
                )
                cfg.agent.arms = [a for a in cfg.agent.arms if a not in llm_arms]
            else:
                llm_client = LLMClient(
                    provider=cfg.llm.provider,
                    model=cfg.llm.model,
                    temperature=cfg.llm.temperature,
                    max_output_tokens=cfg.llm.max_output_tokens,
                    max_calls=cfg.llm.max_calls,
                    cache_only=cfg.llm.cache_only,
                )

        # LLM arms run on a stratified subsample when the budget requires it.
        llm_signals = test_signals
        if cfg.llm.subsample_instances and len(test_signals) > cfg.llm.subsample_instances:
            llm_signals = _stratified_sample(
                test_signals, cfg.llm.subsample_instances, base_seed
            )
            log.info(
                "LLM arms restricted to a stratified subsample of %d instances",
                len(llm_signals),
            )

        # --- 5. estimate the adaptive correction rate, to match the random arm ---
        evidence = ValidationEvidence(outcomes, primary)
        from agents.policies import RGCPolicy

        probe = RGCPolicy(
            gain_threshold=gain_threshold, gain_scores=gain, risk_scores=risk,
            acceptance_delta=cfg.agent.acceptance_delta,
        )
        correction_rate = float(
            probe.decide(test_signals, evidence, base_seed)["corrected"].mean()
        )
        log.info("adaptive correction rate %.1f%%; matching random arm", 100 * correction_rate)

        # --- 6. run every arm across seeds ---
        # Per-action gain calibration, fitted on validation outcomes only.
        from agents.calibration import GainCalibrator

        try:
            calibrator = GainCalibrator().fit(val_outcomes, primary)
            events.write("calibrator_fitted",
                         trusted_actions=calibrator.trusted_actions())
        except ValueError as exc:
            log.warning("gain calibrator could not be fitted (%s); "
                        "the calibrated arm will be skipped", exc)
            calibrator = None

        policies = build_policies(
            cfg, gain, risk, gain_threshold, threshold_value, correction_rate,
            llm_client, calibrator,
        )
        all_decisions, all_scores = [], []
        for seed in cfg.seeds:
            for policy in policies:
                frame = llm_signals if policy.uses_llm else test_signals
                decisions = policy.decide(frame, evidence, seed)
                decisions["seed"] = seed
                scored = compute_policy_regret(
                    test_outcomes,
                    decisions.set_index("instance_id")["chosen_action"],
                    test_labels,
                    primary,
                )
                scored["policy"] = policy.name
                scored["seed"] = seed
                scored = scored.merge(
                    decisions[["instance_id", "confidence", "risk", "gain",
                               "compute_cost", "n_llm_calls", "diagnosed_mode"]],
                    on="instance_id", how="left",
                )
                all_decisions.append(decisions)
                all_scores.append(scored)
                events.write(
                    "policy_scored", policy=policy.name, seed=seed,
                    mean_metric=float(scored["chosen_metric"].mean()),
                    correction_rate=float(scored["corrected"].mean()),
                )
            # Deterministic policies do not vary with the seed; running them once is
            # enough and avoids inflating the apparent sample size.
            policies = [p for p in policies if _is_stochastic(p)]
            if not policies:
                break

        decisions_df = pd.concat(all_decisions, ignore_index=True)
        scores_df = pd.concat(all_scores, ignore_index=True)

        # --- 7. statistics ---
        reference = "never_correct" if "never_correct" in set(scores_df["policy"]) \
            else "baselineA_automl"
        comparison = compare_policies(
            scores_df[scores_df["seed"] == cfg.seeds[0]],
            reference=reference, metric="chosen_metric",
            n_resamples=cfg.eval.bootstrap_resamples, alpha=cfg.eval.alpha,
            seed=base_seed,
        )

        # --- 7b. repeat the whole comparison on the probabilistic metric ---
        #
        # Point and probabilistic accuracy are not interchangeable here: corrections
        # that leave MASE untouched can change CRPS substantially, because they alter
        # the predictive distribution rather than its centre. Scoring only the point
        # metric would miss the effect entirely.
        secondary = cfg.eval.secondary_metric
        comparison_secondary = pd.DataFrame()
        scores_secondary = pd.DataFrame()
        if secondary and secondary in test_outcomes.columns:
            sec_test_labels = add_oracle_labels(
                test_outcomes, secondary, cfg.eval.oracle_delta,
                cfg.eval.large_error_quantile,
            )
            sec_val_labels = add_oracle_labels(
                val_outcomes, secondary, cfg.eval.oracle_delta,
                cfg.eval.large_error_quantile,
            )
            try:
                sec_cal = GainCalibrator().fit(val_outcomes, secondary)
            except ValueError:
                sec_cal = None
            sec_evidence = ValidationEvidence(outcomes, secondary)
            sec_frames = []
            for policy in build_policies(
                cfg, gain, risk, gain_threshold, threshold_value, correction_rate,
                None, sec_cal,
            ):
                dec = policy.decide(test_signals, sec_evidence, base_seed)
                sc = compute_policy_regret(
                    test_outcomes, dec.set_index("instance_id")["chosen_action"],
                    sec_test_labels, secondary,
                )
                sc["policy"] = policy.name
                sc["seed"] = base_seed
                sec_frames.append(sc)
            if sec_frames:
                scores_secondary = pd.concat(sec_frames, ignore_index=True)
                comparison_secondary = compare_policies(
                    scores_secondary, reference=reference, metric="chosen_metric",
                    n_resamples=cfg.eval.bootstrap_resamples, alpha=cfg.eval.alpha,
                    seed=base_seed,
                )
                events.write("secondary_metric_scored", metric=secondary,
                             n_policies=int(scores_secondary["policy"].nunique()))

        # --- 8. persist ---
        artifacts = {
            "outcomes": outcomes,
            "signals": signals,
            "test_labels": test_labels,
            "val_labels": val_labels,
            "realizable_oracle": realizable,
            "decisions": decisions_df,
            "scores": scores_df,
            "comparison": comparison,
            "comparison_secondary": comparison_secondary,
            "scores_secondary": scores_secondary,
            "feature_importance": estimator.feature_importance(
                test_signals, test_labels, head="gain"
            ),
        }
        for name, frame in artifacts.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame.to_parquet(directory / f"{name}.parquet")

        if llm_client is not None:
            for key, value in llm_client.stats().items():
                manifest.bump(key, value)
        manifest.bump("n_instances_test", len(test_labels))
        manifest.bump("n_policies", len(set(scores_df["policy"])))
        manifest.finish("completed")

        log.info("run complete: %s", directory)
        return ExperimentResult({**artifacts, "run_dir": directory,
                                 "manifest": manifest, "estimator": estimator})

    except Exception as exc:
        manifest.finish("failed", f"{type(exc).__name__}: {exc}")
        log.exception("run failed")
        raise
    finally:
        events.close()


def _is_stochastic(policy) -> bool:
    """Only arms with genuine randomness need repeating across seeds."""
    return policy.name in {"random_correct"}


def _stratified_sample(signals: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample instances stratified by family and horizon.

    Stratification matters because the LLM budget forces a subsample, and an unstratified
    draw would under-represent rare families -- exactly the shift/anomaly strata that H2
    is about.
    """
    strata = signals.groupby(["family", "horizon"], dropna=False)
    per = max(1, n // max(1, strata.ngroups))
    parts = [
        g.sample(min(len(g), per), random_state=seed) for _, g in strata
    ]
    out = pd.concat(parts)
    if len(out) < n:
        remainder = signals.drop(out.index)
        extra = remainder.sample(min(len(remainder), n - len(out)), random_state=seed)
        out = pd.concat([out, extra])
    return out.sort_index()
