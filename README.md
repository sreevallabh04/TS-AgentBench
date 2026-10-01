# TS-AgentBench

**Knowing When to Re-Forecast: Reliability-Gated Self-Correction for Time-Series
Forecasting Agents**

A benchmark and experimental framework for a question the agentic forecasting
literature currently assumes rather than tests: **does self-correction actually help,
and is it triggered correctly?**

---

## The problem

A growing family of systems gives time-series forecasting agents a self-correction
stage — inspect the forecast, critique it, revise. The benefit of that stage has never
been demonstrated, and with existing tooling it could not be, for three reasons:

1. **No ground truth for the decision.** Benchmarks record the error of the forecast an
   agent produced, not what the error *would* have been under a different corrective
   action. So "the agent corrected and got error *x*" can never become "correcting was
   the right call."
2. **Compute confounding.** A correcting agent runs more models and spends more tokens.
   When it wins, the win is jointly attributable to better decisions and a larger
   budget, and essentially no published comparison separates the two.
3. **Agents judging themselves.** Reflection is typically verbal self-critique — the
   mechanism shown to be unreliable for reasoning tasks
   ([Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)).

## What this provides

**A benchmark with counterfactual ground truth.** Every corrective action from a typed
action space is executed on every instance and scored. Each instance therefore carries
an oracle action, a `should_correct` label, and a computable regret for *any* policy.
Because outcomes are precomputed, evaluating a new correction policy is a table lookup —
no model refitting — which makes exhaustive ablation affordable and results exactly
reproducible.

**Reliability-gated correction (RGC).** A policy that replaces verbal self-critique with
measurement: a calibrated estimator predicts whether any repair will help, a typed
diagnosis restricts candidates to matched repairs, and a candidate is adopted only if it
demonstrably improves accuracy on held-out backtest origins the agent already has.

**A compute-matched evaluation.** Including a rate-matched random-correction control,
per-regime analysis on data whose failure modes are known by construction, calibration
and selective-prediction reporting, and a quantification of the selection bias in
conventionally reported oracle headroom.

See [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) for the gap analysis and an explicit
statement of what we do **not** claim — the agentic forecasting space is crowded, and
several obvious claims are already taken.

---

## Quick start

```bash
# 1. Install (Python 3.12, CPU-only is fine)
python -m pip install -r requirements.txt

# 2. Prove the pipeline end to end (~1 minute, no network, no API key)
python run_experiment.py --config configs/smoke.yaml

# 3. Fetch real datasets with licence + checksum provenance
python scripts/prepare_data.py --all

# 4. Run the main experiment (~3-4 hours on 8 CPU cores)
python run_experiment.py --config configs/experiment.yaml

# 5. Ablations and robustness
python scripts/run_ablations.py  --config configs/ablation.yaml
python scripts/run_robustness.py --config configs/robustness.yaml

# 6. Analyses that produce the paper's findings
python scripts/headline_results.py     --run latest   # dissociation, series-level
python scripts/regret_decomposition.py --run latest   # why diagnose-then-repair loses
python scripts/triage_analysis.py      --run latest   # the deployable positive result
python scripts/llm_arms.py             --run latest   # LLM correction arms (needs a key)
python scripts/critique_discrimination.py --run latest   # does self-critique discriminate?
python scripts/action_choice.py        --run latest   # and is the chosen repair any good?
python scripts/confidence_calibration.py --run latest # H5: stated vs calibrated confidence
python scripts/robustness_summary.py                  # robustness sweep + H2 (Holm)
python scripts/regret_decomposition_realizable.py --run <main run>  # D12: validation-selected
python scripts/static_policy_check.py --run <main run>              # was the ensemble picked on test?
python scripts/fill_authors.py --write                # author fields from arXiv/Crossref, never memory

# 6b. The base-pipeline confound (deviation D11): rerun with a fixed ETS base and
#     compare within-run differences. ~5 h; the comparison verifies single-factor
#     design mechanically before it reports anything.
python run_experiment.py --config configs/fixed_base.yaml
python scripts/fixed_base_comparison.py

# 7. Regenerate every figure and table in the paper, then check and build it
python scripts/make_paper_assets.py --run latest
python scripts/verify_citations.py     # resolves every reference against arXiv/Crossref
python scripts/verify_manuscript.py    # undefined macros, refs, cites, unescaped specials
bash scripts/build_paper.sh            # LaTeX -> BibTeX -> LaTeX -> LaTeX, strict

# 8. Inspect interactively
streamlit run dashboard/app.py
```

Or run the whole study in one command (it resumes if interrupted, and reports exactly
which stages succeeded):

```bash
python scripts/run_all.py                      # full study (hours)
python scripts/run_all.py --quick              # smoke-scale rehearsal (minutes)
python scripts/run_all.py --llm-series 150     # size the LLM arms to your API quota
python scripts/run_all.py --skip-llm           # no key, or quota exhausted
```

### The LLM arms and API quota

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without a key those arms are
skipped with a warning and the non-LLM results are unaffected — but the manuscript
critiques LLM-driven correction, so a study reported as complete must include them.

Budget honestly. A pass over every series needs roughly 1,340 calls, and the Gemini free
tier allows **500 requests per day per model** (15/minute for `flash-lite`, 5/minute for
`flash`), so the full set does not fit in a day on it. Either enable billing — at
Flash-tier prices the whole run costs a few cents — or use `--llm-series` to fit the
budget, accepting wider intervals. Responses are cached under a hash of model, prompt and
parameters, so an interrupted run resumes rather than restarts, and published results
replay with no key and no network.

---

## How the pieces fit

```
datasets/    synthetic generators (ground-truth labels) + real loaders (licences, checksums)
forecasting/ 14 models behind one probabilistic interface
tools/       8 history-only diagnostics, each with a declared cost
benchmark/   typed action space -> counterfactual tensor -> oracle labels
agents/      reliability estimator, failure diagnosis, correction policies, LLM arms
evaluation/  metrics and the pre-registered statistical tests
experiments/ the runner
```

The central object is the **counterfactual tensor**: for every instance, the outcome of
every corrective action. Everything downstream — policies, ablations, robustness — is a
replay over that table.

---

## Scientific safeguards

These are not incidental; they are the reason the results are worth reading.

**Leakage is tested, not assumed.** `tests/test_leakage.py` walks the import graph and
fails the build if agent code can reach the oracle module; corrupts every post-origin
observation and asserts no diagnostic or action changes its output; verifies validation
targets never touch test observations; and checks the reliability estimator refuses to
be fitted on test data. **If these fail, no result from this repository should be
reported.**

**The analysis was pre-registered.** [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md)
fixed every hypothesis, metric, test and threshold before any experiment ran. Verdicts
are reported as found, including unsupported ones.

**The paper cannot disagree with the data.** `paper/` contains no hardcoded numbers.
Tables are `\input` fragments, figures are generated PDFs, and inline numbers are LaTeX
macros — all written by `scripts/make_paper_assets.py` from `results/`.

**Citations are machine-verified.** `scripts/verify_citations.py` resolves every arXiv
ID and DOI against the public APIs and fails on any that does not match its recorded
title.

**Two oracles are reported.** The unrestricted oracle takes a minimum over ten noisy
outcomes and is therefore inflated by the winner's curse. A *realizable* oracle, chosen
on validation and applied to test, is reported alongside it. The gap is the selection
bias, and it is substantial — quoting only the former overstates achievable headroom.

**Bootstrap resamples series, not instances.** Instances from one series share a model
fit and a regime; resampling instances would shrink every interval and manufacture
significance.

---

## Reproducibility

Each run writes a self-contained directory under `results/runs/<run_id>/` containing the
manifest (config, config hash, git commit and dirty flag, package versions), the
counterfactual tensor, per-policy decisions and scores, the statistical comparison, and
a structured event log. Reruns with the same config and seed reproduce identical
metrics.

LLM responses are cached under a hash of model, prompt and parameters, so published
results regenerate without an API key (`llm.cache_only: true`) and without network
access.

Seeds vary the *policy*, the bootstrap and subsampling — not the tensor. Model-internal
randomness is fixed when the tensor is built, because rebuilding it per seed would
multiply cost fivefold for a source of variation the study is not about.

---

## Data and licences

No raw data is redistributed. `scripts/prepare_data.py` downloads from original sources
and records URL, licence, SHA-256 and access date in `datasets/LICENSES.md`. Monash
archive datasets are CC BY 4.0; ETT is used under its upstream licence and only derived
metrics are reported. **Any dataset whose licence could not be verified was dropped
rather than assumed.**

Synthetic data is not shipped as files — it is fully determined by generator, seed and
index, and regenerates deterministically.

---

## Limitations

Stated here as well as in the paper, because they bound what the results support.

- **CPU-only.** Neural forecasters are small and trained on a subsample. We make no
  claim about the best available forecasters, only about correction policies applied to
  a fair, identically-treated pool.
- **The action space is finite and hand-designed.** Headroom estimates are relative to
  these ten actions, not absolute.
- **Ground-truth failure modes exist only for synthetic data.** Causal claims rest on
  synthetic series; real data supplies external validity.
- **LLM results apply to one model class** (a Gemini Flash-tier model at temperature 0),
  chosen under an API budget.
- **Validation origins may not represent the test regime.** Where a structural break
  falls between the windows, RGC degrades toward accepting — the right failure mode, but
  a real bound.

---

## Security

`.env` is gitignored and must never be committed. Rotate any API key that has been
pasted into a chat window, a terminal log or a shared document.

---

## Repository layout

```
TS-AgentBench/
├── agents/         reliability estimator, diagnosis, policies, LLM arms
├── benchmark/      action space, counterfactual tensor, oracle labels
├── configs/        YAML experiment configs (base / smoke / experiment / ablation / robustness)
├── core/           config, seeding, registry, logging, provenance
├── dashboard/      Streamlit research dashboard
├── datasets/       synthetic generators, real loaders, perturbations
├── docs/           pre-registered analysis plan
├── evaluation/     metrics and statistics
├── experiments/    experiment runner
├── forecasting/    forecasting models
├── paper/          manuscript (LaTeX) + auto-generated figures and tables
├── results/        run artifacts
├── scripts/        data prep, ablations, robustness, paper assets, citation check
├── tests/          unit tests + the leakage suite
├── tools/          diagnostic tools
├── LITERATURE_REVIEW.md
├── requirements.txt
└── run_experiment.py
```

## Current status

The full study is complete: 83 tests pass including the full leakage suite, all 82
citations resolve against arXiv/Crossref with zero mismatches, and the main experiment
(578 series, 3,284 test instances, 9 forecasters x 10 corrective actions) has run to
completion with every manuscript number traced back to `results/` via
`scripts/make_paper_assets.py`.

**Headline findings**, in order of how much they reshape the paper's contribution:

1. **Self-correction's benefit is probabilistic, not point-accurate.** Base forecasts
   are over-confident (nominal 80% intervals get ~65% empirical coverage); corrections
   repair that miscalibration far more than they move the point forecast.
2. **A trivial static policy — always apply a backtest-weighted ensemble — dominates
   every adaptive policy we built**, on both metrics (mean CRPS −12.9% vs. our best
   adaptive gate's −4.0%; mean MASE −4.2% vs. 0%, both p < 1e-7). We did not suppress
   or reframe this after finding it; it's the paper's second headline result.
3. **We show *why*, using ground-truth diagnosis labels**: even given the *true*
   failure mode, applying its textbook repair is worse than doing nothing. The loss
   decomposes into (a) committing to one canonical repair instead of validating among
   a failure mode's candidates, and (b) the hand-designed repair list excluding actions
   that often help more — each large enough alone to erase the achievable improvement.
   This is a mechanistic explanation for the negative result in (2), not just another
   number.
4. **The same risk estimate is a well-calibrated triage signal** (AUROC 0.79):
   reviewing the riskiest 10% of forecasts catches a third of all large errors at 66.5%
   precision — a positive, deployable finding independent of whether automated
   correction works.

Reproduce the full pipeline (resumes if interrupted, reports exactly which stage
failed):

```bash
python scripts/run_all.py
python scripts/regret_decomposition.py --run latest   # finding 3
python scripts/triage_analysis.py --run latest         # finding 4
```

See `paper/sections/03_results.tex` (Sections 5.3–5.4) and `docs/ANALYSIS_PLAN.md`
(deviations D1–D6) for the full, honestly-labelled account of what was pre-registered
versus discovered after the first test-set evaluation.

## Citation

A manuscript is in preparation; see [`paper/`](paper/). Please cite the repository until
it appears.
