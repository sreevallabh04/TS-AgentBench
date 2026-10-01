#!/usr/bin/env bash
# Waits for the running main experiment to write its scores, then completes the study:
# ablations -> robustness -> paper figures/tables. Launched in the background so the
# study finishes unattended.
set -u
PY="$1"
cd "$(dirname "$0")/.."
echo "[chain] waiting for the main experiment to finish ..."
for _ in $(seq 1 2000); do
  if ls results/runs/main_*/scores.parquet >/dev/null 2>&1; then break; fi
  sleep 30
done
if ! ls results/runs/main_*/scores.parquet >/dev/null 2>&1; then
  echo "[chain] main experiment did not produce scores; stopping."; exit 1
fi
echo "[chain] main experiment complete."
echo "[chain] === ablations ==="
"$PY" scripts/run_ablations.py --config configs/ablation.yaml 2>&1 | grep -vE "FutureWarning|adf_stat"
echo "[chain] === robustness ==="
"$PY" scripts/run_robustness.py --config configs/robustness.yaml 2>&1 | grep -vE "FutureWarning|adf_stat"
echo "[chain] === paper assets ==="
"$PY" scripts/make_paper_assets.py --run latest 2>&1 | grep -vE "FutureWarning|adf_stat"
echo "[chain] STUDY COMPLETE"
