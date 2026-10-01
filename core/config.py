"""Configuration system.

Design constraints, in priority order:

1. **Reproducibility.** A config hashes to a stable id, so a run can be traced back to
   the exact settings that produced it.
2. **Composability.** Configs may inherit from a base via ``extends:``, so the twenty
   ablation configs differ from the main config by the three lines that matter rather
   than duplicating everything.
3. **Fail loudly.** An unknown key is an error, not a silently ignored typo. A
   misspelled ``horizion: 24`` that falls back to a default would quietly invalidate an
   experiment, which is exactly the class of bug this project cannot afford.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml

from core.paths import CONFIG_DIR

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "SplitConfig",
    "ForecastConfig",
    "AgentConfig",
    "LLMConfig",
    "EvalConfig",
    "load_config",
    "config_hash",
    "ConfigError",
]


class ConfigError(ValueError):
    """Raised for malformed, unknown or contradictory configuration."""


# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #


@dataclass
class SplitConfig:
    """Temporal split sizes, expressed in observations at the end of each series.

    Splits are strictly ordered ``train < val < test``; see docs/ANALYSIS_PLAN.md §8.
    """

    val_length: int = 24
    test_length: int = 24
    # Number of rolling origins evaluated inside the test window. Origins are spaced
    # by `origin_stride` steps so that a single series yields several instances.
    n_test_origins: int = 3
    origin_stride: int = 8
    # Rolling origins inside the validation window, used to fit the risk estimator and
    # to validate candidate corrections. Never touches test.
    n_val_origins: int = 5
    min_train_length: int = 32


@dataclass
class DataConfig:
    synthetic_families: list[str] = field(default_factory=lambda: ["all"])
    series_per_family: int = 100
    real_datasets: list[str] = field(default_factory=list)
    max_series_per_dataset: int = 150
    max_series_length: int = 1024
    include_synthetic: bool = True
    include_real: bool = True


@dataclass
class ForecastConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 8, 24])
    models: list[str] = field(
        default_factory=lambda: ["naive", "seasonal_naive", "theta", "ets", "arima"]
    )
    # Models that are expensive on CPU are restricted to a subsample of instances.
    heavy_models: list[str] = field(default_factory=list)
    heavy_model_fraction: float = 0.25
    # Nominal coverage levels for prediction intervals.
    coverage_levels: list[float] = field(default_factory=lambda: [0.8, 0.95])
    # How the base forecast is chosen before any correction is considered.
    #   "backtest"      -- lowest rolling-origin backtest error over the whole pool
    #   "fixed:<model>" -- always this model, as a conventional single-model pipeline
    # This is an experimental condition, not a tuning knob: backtest selection is
    # itself a strong adaptive procedure, so it leaves correction far less to repair
    # than a fixed-model pipeline would. Running both separates "correction does not
    # work" from "correction had nothing to fix".
    base_selection: str = "backtest"


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    # Hard ceiling on live calls; the runner aborts rather than overspending.
    max_calls: int = 5000
    # When true, only cached responses are served and a cache miss is an error. This is
    # the mode used to reproduce published results without an API key.
    cache_only: bool = False
    # Number of instances routed to LLM arms (stratified subsample). 0 means all.
    subsample_instances: int = 0


@dataclass
class AgentConfig:
    arms: list[str] = field(
        default_factory=lambda: [
            "baselineA_automl",
            "baselineB_llm_single",
            "always_correct",
            "random_correct",
            "threshold_single",
            "rgc",
            "oracle_correct",
        ]
    )
    # Risk gate: correct when predicted failure risk exceeds tau. Fit on validation.
    risk_threshold: float | None = None  # None -> chosen by conformal risk control
    target_correction_rate: float = 0.30
    # A candidate correction is accepted only if it improves validation MASE by delta.
    acceptance_delta: float = 0.02
    abstention_quantile: float = 0.95
    # Gate settings for the calibrated arm. Selected on a held-out validation origin;
    # see docs/ANALYSIS_PLAN.md deviation D4.
    calibrated_margin: float = 0.10
    calibrated_risk_z: float = 0.5
    # Components that ablations switch off.
    use_self_critique: bool = True
    use_uncertainty: bool = True
    use_model_disagreement: bool = True
    use_changepoint: bool = True
    use_anomaly: bool = True
    use_adaptive_tools: bool = True
    use_backtesting: bool = True
    use_correction_loop: bool = True


@dataclass
class EvalConfig:
    primary_metric: str = "mase"
    # Scored alongside the primary metric, with its own oracle labels, calibration and
    # comparison table. Point and probabilistic accuracy answer different questions and
    # here they give different answers, so reporting only one would be misleading.
    secondary_metric: str = "crps_scaled"
    bootstrap_resamples: int = 10000
    bootstrap_unit: str = "series"  # resample series, not instances
    alpha: float = 0.05
    large_error_quantile: float = 0.80  # defines the `large_error` label for H3
    n_calibration_bins: int = 15
    # Minimum relative improvement for an instance to be *labelled* correctable.
    # Distinct from agent.acceptance_delta, which is the policy's own decision
    # threshold: one defines ground truth, the other is a method hyperparameter, and
    # tying them together would let a method redefine the label it is scored against.
    oracle_delta: float = 0.10


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_jobs: int = -1
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        """Check invariants that would otherwise silently corrupt an experiment."""
        errs: list[str] = []

        if not self.seeds:
            errs.append("seeds must not be empty")
        if any(h < 1 for h in self.forecast.horizons):
            errs.append("horizons must be >= 1")
        if not self.forecast.models:
            errs.append("at least one forecasting model is required")
        if self.split.val_length < 1 or self.split.test_length < 1:
            errs.append("val_length and test_length must be >= 1")
        if max(self.forecast.horizons, default=0) > self.split.test_length:
            errs.append(
                f"longest horizon ({max(self.forecast.horizons)}) exceeds test_length "
                f"({self.split.test_length}); the target would run past the series end"
            )
        if max(self.forecast.horizons, default=0) > self.split.val_length:
            errs.append(
                f"longest horizon ({max(self.forecast.horizons)}) exceeds val_length "
                f"({self.split.val_length}); correction acceptance could not be "
                "validated without touching test"
            )
        if not 0.0 <= self.agent.target_correction_rate <= 1.0:
            errs.append("target_correction_rate must be in [0, 1]")
        if self.agent.risk_threshold is not None and not (
            0.0 <= self.agent.risk_threshold <= 1.0
        ):
            errs.append("risk_threshold must be in [0, 1] or null")
        if not 0.0 < self.eval.alpha < 1.0:
            errs.append("eval.alpha must be in (0, 1)")
        if self.eval.bootstrap_unit not in {"series", "instance"}:
            errs.append("eval.bootstrap_unit must be 'series' or 'instance'")
        if not self.data.include_synthetic and not self.data.include_real:
            errs.append("at least one of include_synthetic / include_real must be true")
        sel = self.forecast.base_selection
        if sel != "backtest":
            if not sel.startswith("fixed:"):
                errs.append(
                    f"base_selection must be 'backtest' or 'fixed:<model>', got {sel!r}"
                )
            elif sel.split(":", 1)[1] not in self.forecast.models:
                errs.append(
                    f"base_selection {sel!r} names a model that is not in "
                    f"forecast.models"
                )
        for level in self.forecast.coverage_levels:
            if not 0.0 < level < 1.0:
                errs.append(f"coverage level {level} must be in (0, 1)")

        # An LLM arm with no budget and no cache will fail late and waste a long run;
        # catch it at config-parse time instead.
        llm_arms = {"baselineB_llm_single", "always_correct", "random_correct", "rgc"}
        if set(self.agent.arms) & llm_arms:
            if self.llm.max_calls <= 0 and not self.llm.cache_only:
                errs.append(
                    "LLM arms are enabled but llm.max_calls is 0 and cache_only is "
                    "false; set cache_only: true to replay from cache, or raise "
                    "max_calls"
                )

        if errs:
            raise ConfigError(
                "invalid experiment configuration:\n  - " + "\n  - ".join(errs)
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Coerce YAML scalars to the dataclass's declared type."""
    origin = get_origin(annotation)

    # Optional[X] / X | None
    if origin is not None and type(None) in get_args(annotation):
        if value is None:
            return None
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return _coerce(value, inner[0], path) if len(inner) == 1 else value

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_t,) = get_args(annotation) or (Any,)
        return [_coerce(v, item_t, f"{path}[{i}]") for i, v in enumerate(value)]

    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation is int and isinstance(value, bool):
        raise ConfigError(f"{path}: expected int, got bool")
    return value


def _build(cls: type, data: Mapping, path: str = ""):
    """Instantiate a dataclass from a mapping, rejecting unknown keys.

    Note: this module uses ``from __future__ import annotations``, so ``Field.type`` is
    a *string*, not a type object. Resolving the real annotations with
    :func:`get_type_hints` is therefore mandatory -- reading ``f.type`` directly would
    make every ``is_dataclass`` check silently false and quietly skip nested configs.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"unknown configuration key(s) at {path or '<root>'}: "
            f"{sorted(unknown)}. Known keys: {sorted(known)}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        sub_path = f"{path}.{name}" if path else name
        value = data[name]
        annotation = hints[name]
        if is_dataclass(annotation):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{sub_path}: expected a mapping")
            kwargs[name] = _build(annotation, value, sub_path)
        else:
            kwargs[name] = _coerce(value, annotation, sub_path)
    return cls(**kwargs)


def _resolve_path(spec: str | Path) -> Path:
    p = Path(spec)
    if not p.is_absolute():
        for candidate in (Path.cwd() / p, CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml"):
            if candidate.exists():
                return candidate.resolve()
    if not p.exists():
        raise ConfigError(f"config file not found: {spec}")
    return p.resolve()


def _load_raw(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load YAML, resolving a single-inheritance ``extends:`` chain."""
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular 'extends' chain involving {path}")
    _seen.add(path)

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    parent_spec = raw.pop("extends", None)
    if parent_spec is None:
        return raw
    parent = _load_raw(_resolve_path(parent_spec), _seen)
    return _deep_merge(parent, raw)


def _apply_override(tree: dict, dotted: str, raw_value: str) -> None:
    """Apply a single ``a.b.c=value`` CLI override in place."""
    node: Any = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {dotted!r}: {part!r} is not a mapping")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {dotted!r}: cannot parse {raw_value!r}") from exc
    node[parts[-1]] = value


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Load, merge, override and validate an experiment config.

    ``overrides`` are ``dotted.key=yaml_value`` strings from the command line, applied
    after inheritance so that a sweep can vary one setting without new files.
    """
    tree = _load_raw(_resolve_path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"malformed override {item!r}; expected key=value")
        key, _, value = item.partition("=")
        _apply_override(tree, key.strip(), value.strip())

    cfg = _build(ExperimentConfig, tree)
    cfg.validate()
    return cfg


def config_hash(cfg: ExperimentConfig, length: int = 12) -> str:
    """Stable short hash of a config, used in run ids and cache keys."""
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
