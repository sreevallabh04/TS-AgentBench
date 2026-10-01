"""Run manifests.

A manifest is the provenance record for one experiment run: the config, its hash, the
code version, the environment, and timing. Results in the paper are traceable to a
manifest, which is what makes "we can reproduce this" a checkable statement rather
than an assertion.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import ExperimentConfig, config_hash
from core.paths import PROJECT_ROOT, run_dir

__all__ = ["RunManifest", "make_run_id", "capture_environment"]

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "xgboost",
    "torch",
    "ruptures",
    "google.genai",
)


def _git_info() -> dict[str, Any]:
    """Best-effort git provenance. Absent git, record that explicitly."""

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None

    commit = _run("git", "rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _run("git", "status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the recorded commit does not fully describe the code that
        # ran. Surfacing it prevents a false reproducibility claim.
        "dirty": bool(status),
    }


def capture_environment() -> dict[str, Any]:
    """Snapshot interpreter, platform and key package versions."""
    import importlib

    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = getattr(
                importlib.import_module(name), "__version__", "unknown"
            )
        except Exception:
            packages[name] = "not installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "packages": packages,
        "git": _git_info(),
    }


def make_run_id(cfg: ExperimentConfig) -> str:
    """Human-scannable, collision-resistant run id."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{cfg.name}_{stamp}_{config_hash(cfg, 8)}"


@dataclass
class RunManifest:
    run_id: str
    config_name: str
    config_hash: str
    config: dict[str, Any]
    environment: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    error: str | None = None
    counters: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(cls, cfg: ExperimentConfig, run_id: str | None = None) -> "RunManifest":
        return cls(
            run_id=run_id or make_run_id(cfg),
            config_name=cfg.name,
            config_hash=config_hash(cfg),
            config=cfg.to_dict(),
            environment=capture_environment(),
        )

    @property
    def path(self) -> Path:
        return run_dir(self.run_id) / "manifest.json"

    def bump(self, key: str, amount: float = 1.0) -> None:
        """Increment a run counter (LLM calls, tool calls, cache hits, ...)."""
        self.counters[key] = self.counters.get(key, 0.0) + amount

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.finished_at = time.time()
        self.status = status
        self.error = error
        self.save()

    @property
    def duration_s(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.started_at

    def save(self) -> Path:
        payload = asdict(self)
        payload["duration_s"] = self.duration_s
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return self.path

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
