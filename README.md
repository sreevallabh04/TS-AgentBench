# TS-AgentBench

Does letting a forecasting agent "fix" its own forecasts actually help?

Short version: no. Longer version: no, and the number everyone uses to argue yes is mostly
a statistical mirage. Longest version: there's a 37-page paper in `paper/`.

This repo is the benchmark, the code, every result file, and the manuscript source. Every
number in the paper is generated from a results file by a script. I typed none of them by
hand, which is good, because the first time I did the analysis I got the headline wrong
and the paper's own methods section had to tell me so.

## What this is

A lot of recent "agentic forecasting" systems bolt on a self-correction stage: look at the
forecast, diagnose what's wrong, apply a repair. Nobody had checked whether that stage
helps, because checking requires knowing what *would* have happened under the repairs you
didn't pick.

So we built the thing that knows. For every forecasting instance (578 series, ~9,750
instances), we run **all ten** corrective actions and store the outcome of each. That gives
you, per instance, an oracle action, a ground-truth "should you have corrected?" label, and
regret for any policy you like. New policy? Replay it against the tensor. No refitting.

## What we found (the parts that survived)

**The oracle headroom is almost entirely winner's curse.** Pick the best repair per
instance *on the test outcome* and correction looks like a 31% MASE improvement. Pick it on
earlier origins and score it on later ones and you get +4.5%. Worse than doing nothing.
Somewhere between 98% and 114% of the apparent gain was selection. (Yes, over 100%. The
realizable version is negative.) An earlier draft of this paper reported the inflated
decomposition as its main result. The paper's own section on measurement pathologies
describes the exact mistake. We caught it late, after a reviewer pointed at it. Section 3.6
is the corrected version; the analysis plan (`docs/ANALYSIS_PLAN.md`, deviation D12) has the
confession.

**The one thing that works is embarrassingly boring.** Apply a static ensemble to every
forecast, always, no thinking. CRPS drops 13%, it holds on real data alone, and it's what
validation picks when asked for one static action. Choosing a repair per series does worse
than choosing once. Bates and Granger said this in 1969. We've measured it again, at the
level of repair selection, and it's still true.

**Giving the decision to an LLM makes things worse.** Asked whether a forecast needs
revising, the model says yes 70% of the time. Its stated probability that a forecast is
unreliable averages 0.54; the true rate of large errors is 0.22. It is not lazy. It is
pessimistic about its own work, like a grad student, and it acts on it. Asking it to reflect
first changes its judgement by about 0.01 AUROC. Showing it worked examples made the
critique gate *worse*. A small gradient-boosted model on the same evidence decides better at
the same intervention rate. We ran three model families; the big one (578 series) resolves
this, the two small ones point the same way without resolving it.

**None of this is because our base forecaster was already too good.** We reran everything
with a plain fixed ETS base instead of the backtest-selected one. Same conclusions. The
pre-registration for that comparison (D11) was written before the run finished, with both
possible readings and what each would mean, so we couldn't pick the convenient one after.

**The consolation prize:** the risk estimator that fails to drive good corrections is a
perfectly decent triage signal (AUROC 0.79). Review the riskiest 10% of forecasts and you
catch a third of the big misses. That one you can actually ship.

## What we got wrong along the way, since it's all in the git history anyway

- Reported instance-level significance on heavy-tailed errors (D7). Several "results" died
  when we moved to series-level bootstrap.
- Reported the selection-inflated decomposition as the headline (D12). See above.
- Left `reasoning_effort` out of the LLM cache key, so two runs at different effort could
  have read each other's answers (D10). Caught before it affected anything.
- Had the paper's `--run latest` resolve to an experimental condition instead of the main
  run, which quietly regenerated every headline table from the wrong experiment. A
  duplicate-macro guard caught it. By luck.
- Wrote "verified" on three bib entries I'd typed from memory. Fixed the note, then actually
  verified them.

There are fourteen numbered deviations in the analysis plan. Each one says what changed,
why, and what it cost. If you only read one file in this repo, read that one.

## Reproducing it

You need Python 3.12. Everything else installs from `requirements.txt`.

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # downloads the real datasets, checks hashes
python run_experiment.py --config configs/experiment.yaml   # ~5 h on a laptop; builds the tensor
```

Or skip the five hours: the two runs the paper reports are checked in under
`results/runs/`, and so is the LLM response cache (`data/cache/llm/`, 2,188 cached
replies). Every LLM arm in the paper replays from cache. **You do not need an API key** to
reproduce any number in the paper.

Then the analyses, each a few minutes:

```bash
python scripts/headline_results.py                 --run main_20260918-233847_3fa46848
python scripts/regret_decomposition_realizable.py  --run main_20260918-233847_3fa46848
python scripts/static_policy_check.py              --run main_20260918-233847_3fa46848
python scripts/llm_arms.py --cache-only            --run main_20260918-233847_3fa46848
python scripts/critique_discrimination.py          --run main_20260918-233847_3fa46848
python scripts/confidence_calibration.py           --run main_20260918-233847_3fa46848
python scripts/action_choice.py                    --run main_20260918-233847_3fa46848
python scripts/robustness_summary.py
python scripts/run_ablations.py --run main_20260918-233847_3fa46848
python scripts/fixed_base_comparison.py
python scripts/triage_analysis.py                  --run main_20260918-233847_3fa46848
```

And the paper:

```bash
python scripts/make_paper_assets.py --run main_20260918-233847_3fa46848   # tables, figures, macros
python scripts/verify_citations.py     # every reference resolved against arXiv / Crossref
python scripts/verify_manuscript.py    # undefined macros, dangling refs, unescaped specials
bash scripts/build_paper.sh main       # pdflatex x3 + bibtex, fails on any unresolved ref
```

`build_paper.sh main_preprint` gives you the version with a byline. The plain `main` build
is anonymous for review.

Tests: `pytest tests/` (122 of them, including a leakage suite that fails the build if
test-time information can reach a training-time object).

## The one rule

`paper/` contains no hand-typed numbers. Every figure, table and inline statistic is a
macro written by `make_paper_assets.py` from a results file. If you change a result, the
paper changes. If you try to type a number into the paper, the verifier yells at you. This
is the only reason the paper survived three rounds of review where the headline moved.

## Layout

```
agents/         correction policies: the arithmetic gate, the LLM arms, diagnosis
benchmark/      the counterfactual tensor and the oracle labels
datasets/       synthetic generators (with ground-truth failure modes) and real loaders
evaluation/     metrics; series-level inference lives in series_level.py
scripts/        every analysis in the paper, one script each
paper/          TMLR manuscript. sections/*.tex is prose; tables/ and generated/ are built
docs/           ANALYSIS_PLAN.md: the pre-registration and all fourteen deviations
results/        result files and the two released runs
tests/          122 tests
```

## Citing

There's a paper under review. Until it lands, cite the repository. If it gets rejected
I'll put the reviews in `docs/` too, because at this point that seems in keeping.

## Licence

Code is MIT. The real datasets are not redistributed; `scripts/prepare_data.py` fetches them
from source and records licence and checksum for each in `datasets/LICENSES.md`.
