"""Canonical project paths.

Everything that writes to disk resolves its location through this module, so that
relocating the project or redirecting outputs is a one-line change rather than a
search-and-replace across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# core/paths.py -> core/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    """Allow any top-level directory to be redirected via an environment variable.

    Used by the test suite to sandbox writes, and by anyone whose data lives on a
    different volume from the code.
    """
    raw = os.environ.get(var)
    return Path(raw).resolve() if raw else default


DATA_DIR: Path = _env_path("TSAB_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
LLM_CACHE_DIR: Path = CACHE_DIR / "llm"
TENSOR_CACHE_DIR: Path = CACHE_DIR / "tensor"

RESULTS_DIR: Path = _env_path("TSAB_RESULTS_DIR", PROJECT_ROOT / "results")
RUNS_DIR: Path = RESULTS_DIR / "runs"

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
PAPER_DIR: Path = PROJECT_ROOT / "paper"
FIGURES_DIR: Path = PAPER_DIR / "figures"
TABLES_DIR: Path = PAPER_DIR / "tables"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

_ALL_DIRS = (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    CACHE_DIR,
    LLM_CACHE_DIR,
    TENSOR_CACHE_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)


def ensure_dirs() -> None:
    """Create every standard directory. Idempotent; safe to call repeatedly."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    """Directory holding all artifacts for a single experiment run."""
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
