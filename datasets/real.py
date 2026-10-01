"""Real-world dataset loaders with recorded provenance.

Every dataset carries its source URL, licence and a checksum of the downloaded file.
``scripts/prepare_data.py`` writes ``datasets/LICENSES.md`` from these records, so the
manuscript's data-availability statement is generated from what was actually
downloaded rather than from memory.

A deliberate rule: **a dataset whose licence cannot be verified is not used.** It is
better to run on fewer datasets than to publish one we do not have the right to
redistribute results from.

Real series carry no ground-truth structural labels -- nobody knows where the "true"
change point in an electricity series is. :class:`~datasets.base.SeriesLabels` is
therefore left with ``is_ground_truth=False``, and the analysis conditions on that flag
rather than silently treating detector output as truth.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.logging import get_logger
from core.paths import RAW_DIR
from core.registry import Registry
from datasets.base import SeriesLabels, TimeSeries

log = get_logger("datasets.real")

__all__ = ["REAL_DATASETS", "DatasetSpec", "load_real_dataset", "available_datasets"]

REAL_DATASETS: Registry["DatasetSpec"] = Registry("real dataset")


@dataclass
class DatasetSpec:
    """Provenance and loading logic for one real dataset."""

    key: str
    name: str
    url: str
    licence: str
    licence_url: str
    citation: str
    seasonal_period: int
    freq: str
    loader: Callable[[Path], list[np.ndarray]]
    filename: str = ""
    notes: str = ""
    checksum: str | None = field(default=None, init=False)

    @property
    def path(self) -> Path:
        return RAW_DIR / (self.filename or f"{self.key}{Path(self.url).suffix}")

    def provenance(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "url": self.url,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "citation": self.citation,
            "seasonal_period": self.seasonal_period,
            "frequency": self.freq,
            "sha256": self.checksum,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Format readers
# --------------------------------------------------------------------------- #


def read_tsf(path: Path) -> tuple[list[np.ndarray], dict]:
    """Read a Monash ``.tsf`` file (optionally zipped).

    The format is a small ARFF dialect: ``@attribute`` headers, then one line per
    series as ``name:start_timestamp:v1,v2,...``. Written out rather than pulled from
    a dependency so the benchmark stays installable from ``requirements.txt`` alone.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".tsf"))
            raw = zf.read(inner).decode("utf-8", errors="replace")
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    series: list[np.ndarray] = []
    meta: dict = {}
    in_data = False
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("@data"):
            in_data = True
            continue
        if not in_data:
            if low.startswith("@frequency"):
                meta["frequency"] = line.split()[-1]
            elif low.startswith("@horizon"):
                meta["horizon"] = line.split()[-1]
            continue

        parts = line.split(":")
        if len(parts) < 2:
            continue
        values = parts[-1].split(",")
        arr = np.array(
            [np.nan if v in ("?", "", "NA") else float(v) for v in values],
            dtype=float,
        )
        if arr.size:
            series.append(arr)
    return series, meta


def _monash_loader(path: Path) -> list[np.ndarray]:
    series, _ = read_tsf(path)
    return series


def _ett_loader(path: Path) -> list[np.ndarray]:
    """ETT CSV: a date column plus several load/temperature channels."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() not in ("date", "datetime", "timestamp")]
    return [df[c].to_numpy(dtype=float) for c in cols if df[c].notna().sum() > 64]


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

_MONASH_LICENCE = "CC BY 4.0"
_MONASH_LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
_MONASH_CITATION = (
    "Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. "
    "(2021). Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks."
)


def _register_monash(key: str, name: str, record: str, period: int, freq: str,
                     notes: str = "") -> None:
    REAL_DATASETS.register(
        key,
        lambda: DatasetSpec(
            key=key,
            name=name,
            url=f"https://zenodo.org/records/{record}/files/{name}.zip?download=1",
            licence=_MONASH_LICENCE,
            licence_url=_MONASH_LICENCE_URL,
            citation=_MONASH_CITATION,
            seasonal_period=period,
            freq=freq,
            loader=_monash_loader,
            filename=f"{key}.zip",
            notes=notes or "Monash Time Series Forecasting Archive; research use.",
        ),
    )


# Zenodo record ids for the Monash archive. Verified by scripts/prepare_data.py,
# which fails loudly rather than silently substituting a different file.
_register_monash("monash_tourism_monthly", "tourism_monthly_dataset", "4656096", 12, "monthly")
_register_monash("monash_hospital", "hospital_dataset", "4656014", 12, "monthly")
_register_monash("monash_nn5_daily", "nn5_daily_dataset_without_missing_values",
                 "4656117", 7, "daily")
_register_monash("monash_covid_deaths", "covid_deaths_dataset", "4656009", 7, "daily")
_register_monash("monash_sunspot", "sunspot_dataset_without_missing_values",
                 "4654773", 12, "daily")
_register_monash("monash_saugeen", "saugeenday_dataset", "4656058", 12, "daily")
_register_monash("monash_electricity_hourly", "electricity_hourly_dataset",
                 "4656140", 24, "hourly")
_register_monash("monash_traffic_hourly", "traffic_hourly_dataset", "4656132", 24, "hourly")
_register_monash("monash_weather", "weather_dataset", "4654822", 7, "daily")
_register_monash("monash_m4_monthly", "m4_monthly_dataset", "4656480", 12, "monthly")
_register_monash("monash_m4_hourly", "m4_hourly_dataset", "4656589", 24, "hourly")


@REAL_DATASETS.register("ett_h1")
def _ett_h1() -> DatasetSpec:
    return DatasetSpec(
        key="ett_h1",
        name="ETTh1",
        url=(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
            "ETTh1.csv"
        ),
        licence="CC BY-NC-SA 4.0 (verify before redistribution)",
        licence_url="https://github.com/zhouhaoyi/ETDataset",
        citation=(
            "Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long "
            "Sequence Time-Series Forecasting. AAAI."
        ),
        seasonal_period=24,
        freq="hourly",
        loader=_ett_loader,
        filename="ETTh1.csv",
        notes=(
            "Electricity Transformer Temperature. Licence is non-commercial; we "
            "report derived metrics only and redistribute no raw data. Confirm the "
            "licence text in the upstream repository before any redistribution."
        ),
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def available_datasets() -> list[str]:
    return REAL_DATASETS.names()


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_real_dataset(
    key: str,
    max_series: int = 150,
    max_length: int = 1024,
    min_length: int = 64,
    seed: int = 0,
) -> list[TimeSeries]:
    """Load a prepared real dataset into :class:`TimeSeries` objects.

    Long series are truncated to their most recent ``max_length`` observations, and a
    deterministic random subsample of at most ``max_series`` series is taken. Both
    bounds are CPU-budget decisions, are recorded in the run manifest, and are stated
    in the manuscript rather than left implicit.
    """
    spec = REAL_DATASETS.build(key)
    if not spec.path.exists():
        raise FileNotFoundError(
            f"{key} is not downloaded. Run: python scripts/prepare_data.py --datasets {key}"
        )

    raw = spec.loader(spec.path)
    usable = [s for s in raw if np.isfinite(s).sum() >= min_length]
    if not usable:
        raise ValueError(f"{key}: no series met the minimum length of {min_length}")

    rng = np.random.default_rng(seed)
    if len(usable) > max_series:
        idx = rng.choice(len(usable), size=max_series, replace=False)
        usable = [usable[i] for i in sorted(idx)]

    out: list[TimeSeries] = []
    for i, values in enumerate(usable):
        trimmed = values[-max_length:] if values.size > max_length else values
        # A trailing NaN would become a forecast target; drop trailing gaps.
        finite = np.flatnonzero(np.isfinite(trimmed))
        if finite.size < min_length:
            continue
        trimmed = trimmed[: finite[-1] + 1]
        out.append(
            TimeSeries(
                series_id=f"{key}_{i:04d}",
                values=trimmed,
                seasonal_period=spec.seasonal_period,
                freq=spec.freq,
                dataset=key,
                source="real",
                family=None,
                labels=SeriesLabels(is_ground_truth=False),
            )
        )
    log.info("loaded %d series from %s", len(out), key)
    return out


def write_license_manifest(path: Path, specs: list[DatasetSpec]) -> Path:
    """Write ``datasets/LICENSES.md`` from what was actually downloaded."""
    lines = [
        "# Dataset licences and provenance",
        "",
        "Generated by `scripts/prepare_data.py`. Every entry records the exact source",
        "URL and the SHA-256 of the file that was downloaded, so the data underlying",
        "the reported results is identifiable and verifiable.",
        "",
        "No raw data is redistributed with this repository; only derived metrics are",
        "reported. Users must obtain the data from the original sources under the",
        "licences below.",
        "",
    ]
    for spec in specs:
        p = spec.provenance()
        lines += [
            f"## {p['key']}",
            "",
            f"- **Name**: {p['name']}",
            f"- **Source**: <{p['url']}>",
            f"- **Licence**: {p['licence']} (<{p['licence_url']}>)",
            f"- **Frequency**: {p['frequency']}, seasonal period {p['seasonal_period']}",
            f"- **SHA-256**: `{p['sha256'] or 'not downloaded'}`",
            f"- **Citation**: {p['citation']}",
            f"- **Notes**: {p['notes']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
