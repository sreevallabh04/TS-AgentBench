"""Deterministic seeding.

Reproducibility claim in the paper: rerunning with an identical config and seed
reproduces identical metrics. That claim is only as strong as this module, so every
stochastic dependency is seeded here in one place.

A subtlety worth stating: seeding a global RNG makes a *sequential* program
reproducible, but not a parallel one, because worker scheduling order is
nondeterministic. Anything that runs under multiprocessing must therefore derive its
randomness from :func:`derive_seed` keyed by a stable identifier (e.g. the series id),
never from the global RNG.
"""

from __future__ import annotations

import hashlib
import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np

__all__ = ["set_global_seed", "derive_seed", "seeded_rng", "temporary_seed"]

_MAX_UINT32 = 2**32


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every global RNG the project can reach.

    Parameters
    ----------
    seed:
        Base seed.
    deterministic_torch:
        Force torch into deterministic kernel selection. Costs a little speed and is
        worth it here, since we are making a reproducibility claim in print.
    """
    seed = int(seed) % _MAX_UINT32
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional at import time so the benchmark stays usable without it
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def derive_seed(base_seed: int, *keys: object) -> int:
    """Derive a stable child seed from a base seed and arbitrary keys.

    Deterministic across processes and across Python invocations -- unlike
    :func:`hash`, which is randomised per process. This is what makes parallel
    execution reproducible: each work item seeds itself from its own identity rather
    than from a shared, order-dependent stream.

    >>> derive_seed(0, "series_7", 3) == derive_seed(0, "series_7", 3)
    True
    """
    payload = "|".join([str(base_seed), *(str(k) for k in keys)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def seeded_rng(base_seed: int, *keys: object) -> np.random.Generator:
    """A :class:`numpy.random.Generator` whose stream is fixed by ``keys``."""
    return np.random.default_rng(derive_seed(base_seed, *keys))


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Seed globals inside the block, then restore the previous RNG states.

    Used where a deterministic sub-computation must not disturb the surrounding
    stream (otherwise adding a diagnostic silently changes every later draw).
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except ImportError:  # pragma: no cover
        torch_state = None

    try:
        set_global_seed(seed, deterministic_torch=False)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            import torch

            torch.random.set_rng_state(torch_state)
