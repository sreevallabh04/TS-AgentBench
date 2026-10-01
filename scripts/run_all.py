#!/usr/bin/env python
"""Run the complete study end to end.

    python scripts/run_all.py                 # full study (hours)
    python scripts/run_all.py --quick         # smoke-scale rehearsal (minutes)
    python scripts/run_all.py --skip-robustness

Stages, in order:

1. main experiment        -> results/runs/<run_id>/
2. ablation study         -> results/ablations.parquet
3. robustness sweep       -> results/robustness.parquet
4. paper figures & tables -> paper/figures, paper/tables, paper/generated
5. citation verification  -> results/citation_check.txt

Each stage is skipped if its output already exists, so an interrupted study resumes
instead of restarting. The expensive counterfactual tensor is content-hash cached
independently, so even a re-run of stage 1 reuses it.

Stage failures are reported and do not silently produce a partial paper: the summary at
the end states exactly which stages succeeded, so a half-finished study is never
mistaken for a complete one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import PROJECT_ROOT, RESULTS_DIR, RUNS_DIR  # noqa: E402


def run(label: str, command: list[str]) -> tuple[bool, float]:
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, *command], cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    status = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n-- {label}: {status} in {elapsed / 60:.1f} min", flush=True)
    return rc == 0, elapsed


def has_completed_run() -> bool:
    return RUNS_DIR.exists() and any(
        (p / "scores.parquet").exists() for p in RUNS_DIR.iterdir() if p.is_dir()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use the smoke config throughout (minutes, not hours)")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-llm", action="store_true",
                        help="omit the LLM arms (no API key, or exhausted quota)")
    parser.add_argument("--llm-series", type=int, default=None,
                        help="cap the LLM arms at this many series; sized to the "
                             "provider's daily quota (see ANALYSIS_PLAN.md D8)")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages even when their output exists")
    args = parser.parse_args(argv)

    main_config = "configs/smoke.yaml" if args.quick else "configs/experiment.yaml"
    ablation_config = "configs/smoke.yaml" if args.quick else "configs/ablation.yaml"
    results: dict[str, str] = {}

    # --- 1. main experiment ---
    if args.skip_experiment or (has_completed_run() and not args.force):
        results["experiment"] = "skipped (existing run found)"
    else:
        ok, _ = run("1/5  main experiment", ["run_experiment.py", "--config", main_config])
        results["experiment"] = "ok" if ok else "FAILED"
        if not ok:
            print("\nThe main experiment failed; later stages have nothing to work "
                  "from. Stopping.", file=sys.stderr)
            return 1

    # --- 2. ablations ---
    ablation_out = RESULTS_DIR / "ablations.parquet"
    if args.skip_ablations or (ablation_out.exists() and not args.force):
        results["ablations"] = "skipped (output exists)"
    else:
        ok, _ = run("2/5  ablation study",
                    ["scripts/run_ablations.py", "--config", ablation_config])
        results["ablations"] = "ok" if ok else "FAILED"

    # --- 3. robustness ---
    robustness_out = RESULTS_DIR / "robustness.parquet"
    if args.skip_robustness or (robustness_out.exists() and not args.force):
        results["robustness"] = "skipped"
    else:
        cmd = ["scripts/run_robustness.py", "--config", "configs/robustness.yaml"]
        if args.quick:
            cmd += ["--series-per-family", "2", "--severities", "0.0", "1.0"]
        ok, _ = run("3/5  robustness sweep", cmd)
        results["robustness"] = "ok" if ok else "FAILED"

    # --- 4. headline analysis: the dissociation table and series-level inference ---
    ok, _ = run("4/8  headline results (series-level)",
                ["scripts/headline_results.py", "--run", "latest"])
    results["headline"] = "ok" if ok else "FAILED"

    # --- 5. regret decomposition and triage ---
    ok, _ = run("5/8  regret decomposition",
                ["scripts/regret_decomposition.py", "--run", "latest"])
    results["decomposition"] = "ok" if ok else "FAILED"

    ok, _ = run("5/8  triage analysis",
                ["scripts/triage_analysis.py", "--run", "latest"])
    results["triage"] = "ok" if ok else "FAILED"

    # --- 6. LLM arms ---
    #
    # Needs GEMINI_API_KEY and, on a free tier, more daily quota than one day allows
    # for the full series set (see ANALYSIS_PLAN.md D8). Responses are cached, so a
    # repeated invocation resumes rather than restarts, and a failure here is reported
    # rather than passed over: the manuscript critiques LLM-driven correction and must
    # not be built without these arms.
    if args.skip_llm:
        results["llm_arms"] = "skipped (--skip-llm)"
    else:
        cmd = ["scripts/llm_arms.py", "--run", "latest"]
        if args.quick:
            cmd += ["--n", "20"]
        elif args.llm_series:
            cmd += ["--n", str(args.llm_series)]
        ok, _ = run("6/8  LLM correction arms", cmd)
        results["llm_arms"] = "ok" if ok else "FAILED"

    # --- 7. paper assets ---
    ok, _ = run("7/8  paper figures and tables",
                ["scripts/make_paper_assets.py", "--run", "latest"])
    results["paper_assets"] = "ok" if ok else "FAILED"

    # --- 8. citations and manuscript checks ---
    ok, _ = run("8/8  citation verification", ["scripts/verify_citations.py"])
    results["citations"] = "ok" if ok else "FAILED"

    ok, _ = run("8/8  manuscript static checks", ["scripts/verify_manuscript.py"])
    results["manuscript"] = "ok" if ok else "FAILED"

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for stage, status in results.items():
        print(f"  {stage:16s} {status}")

    failed = [s for s, v in results.items() if v == "FAILED"]
    if failed:
        print(f"\nFailed stages: {', '.join(failed)}. Do not report this as a complete "
              f"study until they pass.")
        return 1

    print("\nAll stages complete.")
    print("  results  -> results/runs/")
    print("  paper    -> paper/figures, paper/tables, paper/generated")
    print("  dashboard-> streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
