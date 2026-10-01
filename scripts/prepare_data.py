#!/usr/bin/env python
"""Download and verify the real datasets, then write the licence manifest.

    python scripts/prepare_data.py --list
    python scripts/prepare_data.py --datasets monash_tourism_monthly ett_h1
    python scripts/prepare_data.py --all

Each file is downloaded once, hashed, and test-loaded. The manifest written to
``datasets/LICENSES.md`` records the URL, licence and SHA-256 of what was actually
fetched, so the paper's data-availability statement describes the real artifacts
rather than an intention.

A download that fails is reported and skipped; it is never silently replaced by
synthetic data. A run that quietly substituted synthetic series for a missing real
dataset would misreport its own evidence base.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging import get_logger, setup_logging  # noqa: E402
from core.paths import PROJECT_ROOT, RAW_DIR, ensure_dirs  # noqa: E402
from datasets.real import (  # noqa: E402
    REAL_DATASETS, checksum_file, load_real_dataset, write_license_manifest,
)

log = get_logger("prepare_data")


def download(url: str, destination: Path, timeout: int = 120) -> bool:
    """Stream a URL to disk. Returns True on success."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(destination)
        return True
    except Exception as exc:  # noqa: BLE001 - network failures are expected
        log.error("failed to download %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    setup_logging()
    ensure_dirs()

    if args.list:
        print("Available real datasets:\n")
        for key in REAL_DATASETS.names():
            spec = REAL_DATASETS.build(key)
            status = "downloaded" if spec.path.exists() else "not downloaded"
            print(f"  {key:28s} {spec.licence:20s} [{status}]")
        return 0

    keys = REAL_DATASETS.names() if args.all else (args.datasets or [])
    if not keys:
        parser.error("specify --datasets, --all or --list")

    prepared, failed = [], []
    for key in keys:
        if key not in REAL_DATASETS:
            log.error("unknown dataset %r; use --list", key)
            failed.append(key)
            continue

        spec = REAL_DATASETS.build(key)
        if spec.path.exists() and not args.force:
            log.info("%s already present", key)
        else:
            log.info("downloading %s", key)
            if not download(spec.url, spec.path):
                failed.append(key)
                continue

        spec.checksum = checksum_file(spec.path)
        try:
            series = load_real_dataset(key, max_series=5, max_length=512)
            if not series:
                raise ValueError("loader produced no usable series")
            log.info(
                "%s verified: %d series sampled, sha256 %s...",
                key, len(series), spec.checksum[:12],
            )
            prepared.append(spec)
        except Exception as exc:  # noqa: BLE001
            log.error("%s downloaded but failed to load: %s", key, exc)
            failed.append(key)

    if prepared:
        manifest = write_license_manifest(
            PROJECT_ROOT / "datasets" / "LICENSES.md", prepared
        )
        log.info("licence manifest written to %s", manifest)

    print(f"\nPrepared {len(prepared)}/{len(keys)} datasets into {RAW_DIR}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print(
            "\nThese datasets will be skipped by experiments. Do not report results "
            "as covering them."
        )
    return 1 if failed and not prepared else 0


if __name__ == "__main__":
    raise SystemExit(main())
