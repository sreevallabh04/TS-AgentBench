# Pre-Registered Analysis Plan

**Frozen:** 2026-09-18, before any experiment was run and before any test-set result was
observed.

This document exists so that the statistical analysis cannot be chosen after seeing the
results. Every hypothesis, metric, test, threshold and stopping rule below was fixed in
advance. Deviations are permitted but must be recorded in §9 with a reason, and reported in
the manuscript as post-hoc.

---

## 1. Primary research question

> Can a forecasting agent predict its own failure accurately enough to decide when
> re-forecasting is worth the compute — and does self-correction help once compute is
> controlled for?

## 2. Units of analysis

- **Instance**: one (series, forecast origin, horizon) triple. Primary unit.
- **Series**: one time series. Used for clustering / paired tests.
- **Stratum**: a labelled condition (e.g. `level_shift`, `anomaly_contaminated`,
  `stationary_clean`). Ground truth on synthetic data; derived on real data.

## 3. Arms

| Arm | Description | Corrects? | Uses LLM? |
|---|---|---|---|
| `baselineA_automl` | Classical automated pipeline; model chosen by validation error | No | No |
| `baselineB_llm_single` | LLM agent, single pass, no correction | No | Yes |
| `always_correct` | Self-Refine style; corrects every instance | Always | Yes |
| `random_correct` | Corrects at the same *rate* as RGC, chosen at random | Random | Yes |
| `threshold_single` | Corrects when one signal (backtest error) exceeds a validation-fit threshold | Threshold | No |
| `rgc` (**proposed**) | Calibrated risk gate + failure-mode-conditioned repair + backtest-validated acceptance | Adaptive | Yes |
| `oracle_correct` | Chooses the best action per instance using test outcomes | Oracle | No |

`oracle_correct` is an **upper bound only**. It uses test-set information by construction and
is never presented as a competing method.

`random_correct` is **rate-matched to `rgc`**: its correction probability is set to the
empirical correction rate of `rgc` on the same split. This is what makes H6 a fair test.

## 4. Metrics

**Primary outcome:** MASE, aggregated as the mean over instances, seasonal-naive denominator
computed on the training portion only.

**Secondary forecasting:** MAE, RMSE, sMAPE, WAPE, CRPS.

**Calibration / selective prediction:** PICP at nominal 80% and 95%, mean interval width,
ECE (15 equal-mass bins), risk-coverage AURC.

**Decision quality (requires the counterfactual tensor):**
- correction-decision precision / recall / F1 against `should_correct`
- **oracle regret** = MASE(chosen action) - MASE(oracle action), mean over instances
- **harmful-correction rate** = P(MASE after correction > MASE before correction | corrected)
- tool-selection F1 against the tool-relevance label
- failure-mode diagnosis accuracy / macro-F1 (synthetic only, where labels are ground truth)

**Cost:** LLM calls, prompt+completion tokens, tool calls, wall-clock seconds per instance.

## 5. Hypotheses and decision rules

Each hypothesis has one **confirmatory** test fixed in advance. Everything else is
exploratory and will be labelled as such.

| ID | Hypothesis | Confirmatory test | Rejection criterion |
|---|---|---|---|
| **H1** | RGC beats `baselineB_llm_single` on MASE | Wilcoxon signed-rank over series on per-series mean MASE, two-sided | Holm-adjusted p < 0.05 **and** Cliff's delta favouring RGC |
| **H2** | Gains concentrate in shift/anomaly/regime strata | Difference-in-differences: (RGC - baselineB) in perturbed strata vs clean stratum, bootstrap CI over series | 95% bootstrap CI on the difference excludes 0 |
| **H3** | Failure is predictable pre-hoc | AUROC of risk estimator for `large_error` on the test split, vs a features-only control (FFORMPP-style) | AUROC > 0.65 **and** DeLong-style bootstrap CI vs control excludes 0 |
| **H4** | Adaptive tool use matches exhaustive use at lower cost | TOST equivalence on MASE, margin = 0.05 MASE; plus paired test on tool calls | Equivalence established **and** tool calls strictly lower |
| **H5** | RGC confidence beats LLM-verbalized confidence | Paired bootstrap on ECE and on AURC | 95% CI on the ECE difference excludes 0 |
| **H6** | Gains are not explained by compute alone | RGC vs `random_correct` (rate-matched) on MASE | Holm-adjusted p < 0.05 favouring RGC |
| **H7** | Quantitative gating beats verbal self-critique at deciding | Correction-decision F1 and oracle regret, RGC vs `always_correct` and vs an LLM-critique gate | 95% bootstrap CI on the regret difference excludes 0 |

**H6 is the gatekeeper.** If H6 fails, H1 will be reported as *not attributable to the
correction policy*, and the paper's headline claim becomes the negative result. This is
stated now, in advance, so that outcome cannot be quietly reframed later.

## 6. Multiplicity control

Seven confirmatory hypotheses form one family. **Holm-Bonferroni** across the seven primary
p-values. Per-stratum, per-horizon and per-dataset breakdowns are **exploratory** and
reported with unadjusted p-values, explicitly labelled as such.

## 7. Statistical procedures

- **Paired comparisons across series:** Wilcoxon signed-rank on per-series mean metric.
- **Forecast-accuracy pairs:** Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample
  correction, computed per series then aggregated.
- **Uncertainty intervals:** non-parametric bootstrap, 10,000 resamples, **resampled at the
  series level** (not the instance level) to respect within-series dependence.
- **Effect size:** Cliff's delta, with the usual thresholds (0.147 / 0.33 / 0.474).
- **Seeds:** 5 seeds for every stochastic component. Seed variation is reported as a
  dispersion band, never pooled as if independent.

## 8. Data handling rules (leakage)

1. Splits are strictly temporal: `train < val < test`. No shuffling anywhere.
2. Scalers, feature statistics and seasonal-naive denominators are fit on **train only**.
3. The risk estimator, all thresholds (tau, delta, abstention cutoff) and any calibration map
   are fit on **validation only**.
4. The counterfactual tensor's oracle labels derive from test outcomes and are
   **evaluation-only**. They live behind `benchmark.oracle`, which `agents/` must not import.
   `tests/test_leakage.py` enforces this by static import analysis and fails the build
   otherwise.
5. No result is inspected on the test split until the full pipeline has passed the leakage
   suite.

## 9. Deviations from this plan

Any deviation is appended here with date, description and reason, and surfaces in the
manuscript. Everything below was decided **after** the first test-set evaluation, and is
therefore post-hoc. The manuscript labels it as such.

### D1 — Reporting the mean alongside outlier-robust summaries (2026-09-19)

**Deviation.** §4 pre-registered mean MASE as the primary outcome. We now report mean,
**geometric mean**, median and 10% trimmed mean together, and treat the geometric mean
as the headline.

**Reason.** MASE on this benchmark is extremely heavy-tailed: median 0.61, mean 1.01,
standard deviation 4.39, maximum 187. The worst 1% of instances carry 19% of total MASE
mass, so the arithmetic mean is a statement about a few dozen instances rather than
about typical behaviour. This was not anticipated when the plan was written.

**Why it is not outcome-shopping.** The change does not rescue the hypothesis. Under the
pre-registered mean, no correction policy beats never correcting; under every robust
summary, the same is true of every policy except a calibrated one whose advantage is not
significant. All summaries are reported for all arms so readers can see the whole
picture.

### D2 — Reporting bug in the comparison table (2026-09-19)

**Deviation.** `compare_policies` paired a mean-based effect size with a rank-based
Wilcoxon p-value. On heavy-tailed data these disagree in sign, and the first results
table accordingly showed one arm as "+0.42% worse" next to "significant: True" from a
test that was detecting an improvement. Fixed: mean and rank effects are now reported
separately, with a `direction_conflict` flag when they disagree.

**Reason.** A defect, not a design choice. Recorded because it changed how the first
result read.

### D3 — Gain measured as a log ratio (2026-09-19)

**Deviation.** The relative gain `(accept - action) / accept` is replaced by the log
accuracy ratio `log(accept / action)` wherever an action's value is estimated.

**Reason.** The relative form divides by an error that is sometimes near zero, producing
gains in the hundreds. On that scale the nested-validation correlation between observed
and realised gain is **0.006** with residual standard deviations above 1000 — no usable
signal. On the log scale the identical data gives **0.43** with residual standard
deviations below 0.7, and stable per-action structure. This was a measurement defect
that was concealing real signal, and the log ratio is the standard scale for error
ratios.

### D4 — A calibrated correction policy added post-hoc (2026-09-19)

**Deviation.** `rgc_calibrated` was designed after the first test evaluation, using a
per-action calibration and a risk-averse lower bound. Its two hyperparameters (margin,
`risk_z`) were selected on a held-out **validation** origin, never on test, and it was
evaluated on test exactly once.

**Reason.** The pre-registered `rgc` gate used a 2% acceptance margin when the
break-even margin implied by the data is far higher, and it treated all actions as
equally trustworthy when transfer is strongly action-specific. Those are defects in the
method, and fixing them is legitimate; reporting the fixed version as if it had been
pre-registered would not be.

**Status.** Its improvement over never correcting is **not statistically significant**
(Wilcoxon p = 0.57). It is reported as an exploratory arm.

### D5 — Fixed-base-model condition added (2026-09-19)

**Deviation.** A second experimental condition fixes the base forecaster to ETS instead
of selecting it per instance by backtest.

**Reason.** Backtest selection over a nine-model pool is itself a strong adaptive
procedure, so the pre-registered setup gives the never-correct arm an unusually strong
starting forecast and leaves correction little to repair. This confounds "self-correction
does not work" with "self-correction had nothing to fix". The added condition separates
them, and is closer to a deployed single-model pipeline.

**Status: specified but not completed.** Building a second counterfactual tensor is the
single most expensive operation in the study (~10k instances x 10 actions x 9 models).
Three attempts were made; two were killed by worker memory exhaustion at `n_jobs=-1`, and
a third at `n_jobs=3` did not finish within the available budget. The configuration is
released and runs with one command, but **no fixed-base result is reported**, and the
manuscript's limitations say so explicitly rather than implying the condition was run.
This is the largest outstanding gap in the study after the LLM-arm sample size.

### D6 — Always-ensemble added as a first-class arm (2026-09-19)

**Deviation.** A trivial static policy, "apply the backtest-weighted ensemble to every
instance," was added as an explicit control after `rgc_calibrated` was already built,
because it was not obviously a comparison worth making a priori.

**Finding.** It dominates every adaptive policy we built on both metrics: mean CRPS
-12.93% vs -4.01% for the calibrated gate (p=8.4e-23 vs never, p=6.1e-06 vs the gate);
mean MASE -4.16% vs -0.11% (p=2.2e-08 vs never; the gate is not significant vs never).
The adaptive machinery -- diagnosis, calibration, risk-averse thresholding -- does not
earn its complexity in this benchmark.

**Reason for reporting as the headline rather than a footnote.** This is a stronger,
more falsifiable, and more useful finding than "an adaptive gate beats never-correcting,"
and it is the honest result of the investigation. It reframes the paper's contribution
from "we built an adaptive policy that works" to "we built a benchmark that shows the
field's adaptive-correction premise has not yet been shown to beat a strong static
default," which is a legitimate and, we believe, more publishable claim.

### D7 — Confirmatory inference corrected to the series level (2026-09-19)

**Deviation.** This is a *correction of a departure from the plan*, not a new choice.
§7 fixes the bootstrap unit as the series. The first round of headline results was
nevertheless reported with **instance-level** Wilcoxon p-values, because
`scripts/headline_results.py` compared policies on the instance-indexed outcome table
directly. That is anti-conservative: the 3,284 test instances come from 578 series and
share, within a series, a generating process, a base model, a scaling denominator and
most of their history.

**Effect of the correction.** It changes conclusions, which is why it is recorded
prominently rather than as a footnote.

| Comparison | instance-level | series-level (95% CI) | verdict |
|---|---|---|---|
| `always_ensemble` vs never, MASE | −4.16%, p=2.2e-08 | −4.69%, [−11.75, +0.60] | **no longer significant** |
| `rgc_calibrated` vs never, MASE | −0.11%, p=0.57 | −0.38%, [−0.97, +0.04] | not significant (unchanged) |
| `always_ensemble` vs never, CRPS | −12.93%, p=8.4e-23 | −13.07%, [−22.85, −7.27] | significant (unchanged) |
| `rgc_calibrated` vs never, CRPS | −4.01%, p=8.9e-13 | −4.04%, [−14.47, +2.87] | **no longer significant** |
| `always_correct` vs never, MASE | +8.80%, p=0.00056 | +8.13%, [+2.73, +15.99] | significantly worse (unchanged) |

Two claims in the first draft were therefore overclaims and have been removed: that the
static ensemble improves point accuracy, and that the calibrated gate improves
probabilistic accuracy. The surviving claim is narrower and cleaner: **the only reliable
improvement in this benchmark comes from an unconditional static ensemble, on the
probabilistic metric only; nothing reliably improves point accuracy, and several
policies reliably harm it.**

**Note on rank-versus-magnitude.** The signed-rank test and the bootstrap interval
disagree for two comparisons (`always_ensemble` on MASE, win rate 60.0%, p=4.2e-08, CI
crossing zero; `rgc_calibrated` on CRPS, win rate 61.0%, p=5.4e-07, CI crossing zero).
Both statements are true: the policy wins on more series than it loses, but the sizes of
those wins do not add up to a mean improvement distinguishable from zero. This is the
same heavy-tail phenomenon recorded in D2, now at the series level. Both statistics are
reported for every comparison; neither is suppressed in favour of the other. Inference
now runs through `evaluation/series_level.py` for every comparison in the manuscript.

### D8 — LLM arms executed (2026-09-19)

**Deviation.** The arms in §3 that require a language model (`baselineB_llm_single`,
`llm_critique_gate`, `rgc_llm_router`) were skipped in the first full run because no API
key was configured; the run log records the skip. They have now been executed against the
same counterfactual tensor via `scripts/llm_arms.py`.

**Why this mattered more than a missing arm usually would.** The manuscript's argument is
about architectures that delegate the correction decision to a language model. Running
zero language models while critiquing that pattern was the largest gap in the evidence,
and no amount of care elsewhere substitutes for it. With the arms run, **H7 is evaluated
against its pre-registered comparison** (quantitative gate vs. LLM critique gate) rather
than the weaker substitute used while they were missing.

**A second model family was attempted and did not complete.** A replication on
`qwen3.8-27b` (Groq) was started because a single model failing to beat never-correcting
invites the objection that the model was weak. That provider meters *tokens* per day
rather than requests, and the budget was exhausted with the replication short of the
coverage needed to support an interval; 140 of 140 `llm_single` and 116 of 140
`llm_critique_gate` decisions were obtained. **No replication result is reported.** The
manuscript states this as an unmet limitation in §
ef{sec:llmarms} rather than reporting
a partial arm, and the single-family scope is stated wherever the LLM findings appear.
Completing it requires one command and more quota.

**Sampling, and the quotas that set it.** Free API tiers turned out to be far tighter
than their documented per-minute limits suggest: 500 requests per day per *project* per
model on one provider, and a rolling ~200k tokens per day on the other. The primary
family therefore runs on one instance from each of the 578 series — matching the main
analysis's bootstrap unit count, though with one origin per series rather than all of
them — and the replication on 140 series. Every non-LLM arm is re-scored within each
family's own subsample, so comparisons are like-for-like within a family and must not be
read across families.

**A discrepancy this creates, recorded rather than smoothed.** On the primary family's
one-origin-per-series subsample, `always_ensemble` improves MASE significantly
(−5.73%, CI [−9.54, −1.99]); on the full test split it does not (−4.69%, CI
[−11.75, +0.60]). The subsample draws fewer of the extreme origins that widen the full
interval. **The full analysis is primary** and the manuscript states the conservative
claim; the difference is flagged in §
ef{sec:llmarms} rather than resolved in whichever
direction suits.

**Integrity guard.** A failed provider call falls back to `accept`, which in aggregate
would convert exhausted quota into an apparent decision not to correct, flattering the
arms under test. `scripts/llm_arms.py` aborts rather than reporting any arm whose
decisions were degraded above 2%. This fired twice in practice — once on a genuine quota
exhaustion and once on a bug in our own key-rotation code — and in both cases prevented a
results file that would have understated how often the models choose to correct. 52 of
2,650 decisions (2.0%) on the primary family came from calls that failed after retries;
no single arm exceeded the threshold.

### D9 — Third model family added, with its falsification criterion fixed in advance (2026-09-19, 23:05 local)

**Deviation.** After two families (`gemini-3.5-flash-lite`, `qwen3.8-27b`) both showed
LLM correction arms significantly worse than never correcting, a third family was added:
`openai/gpt-oss-120b`, an open-weight **reasoning** model run at `reasoning_effort=medium`.

**Why this is the dangerous kind of addition.** Adding an arm after two confirmations is
exactly where pre-registration discipline fails quietly: the temptation is to run until
the picture is pleasing and stop. This entry is written and timestamped **before the run
completes and before any of its numbers are seen**, so the criterion cannot be chosen to
fit the outcome.

**What this arm is for.** The two families run so far are non-reasoning models answering
in a single zero-shot turn. The open objection is that deliberation, not model identity,
is what the earlier arms lacked — that a model which reasons explicitly before answering
would restrain its intervention rate. `reasoning_effort=low` would not test this, so the
arm runs at `medium` with a completion budget large enough for the trace *and* the JSON.

**Stated in advance, the disposition thesis is falsified by this arm if:**

1. **(primary)** the critique's discrimination gap over the uncritiqued arm --- the same
   model, same instances, no reflection --- exceeds **+0.05 AUROC**. The two families so
   far give +0.010 and +0.008. This is the primary falsifier because Finding 4 is a
   *within-model* claim: it says adding reflection does not change what the model decides,
   not that all models decide alike. A within-model delta is the only quantity that bears
   on it directly.
2. **(primary)** `llm_critique_gate` does **not** degrade MASE relative to never
   correcting (point estimate at or below zero).
3. **(secondary, weak)** the intervention rate falls below **35%**. Recorded for
   completeness, but it bears on the thesis only weakly: rates already span 44--70% across
   the two families without disturbing the within-model result, so a third family at a low
   rate would not by itself contradict anything. It is listed so that a reader can see we
   considered it and judged it uninformative, rather than discovering later that it moved
   and we said nothing.

Any of these outcomes is reported as a qualification of Findings 3 and 4, not omitted,
and the abstract's claim is narrowed to non-reasoning models accordingly.

**Secondary quantity, and a caution about what it can show.** We compute **harm per
intervention** (degradation relative to never correcting, divided by intervention rate)
for every arm and family, from the one-origin design only, never mixed with the headline
analysis. The observation is that it matches closely across families within an arm type
(+12.19 vs +11.72 for single-pass; +10.73 vs +10.50 for the critique gate) while
intervention rates differ by 26 percentage points.

**This is largely a corollary of the discrimination result, not independent evidence for
it.** If selection is at chance --- and AUROC 0.532 is close to chance --- then harm per
intervention is just the mean harm of the chosen action mix over an effectively random
sample of instances, so cross-family agreement reduces to the two families having similar
action priors. That is a real but much smaller claim than "the reasoner does not affect
whether intervening is wise", and the manuscript states it at the smaller size. It is
reported as descriptive support for the discrimination finding, never as a second,
independent confirmation of it.

**Guard added at the same time.** `_parse` falls back to `accept` on an unparseable
reply, and on a reasoning model an unparseable reply usually means the completion budget
was consumed by the chain of thought. That path is not a provider failure and the
integrity guard did not previously catch it, so any truncation would have biased the
intervention rate downward --- the one quantity these findings rest on.
`scripts/llm_arms.py` now counts provider failures and unparsed replies together against
the same 2% ceiling and reports them separately.

### D9 (verdict) — Third family completed; none of the pre-registered falsifiers fired (2026-09-19, 23:30 local)

**Model substituted, and why.** D9 named `openai/gpt-oss-120b`. That model exhausted its
200,000-token daily cap partway through the run (199,857 of 200,000 used, 124 rejected
requests logged) and its window would not reset in time. The arm was re-run on
`openai/gpt-oss-20b`, the same open-weight reasoning family at the **same**
`reasoning_effort=medium` and the same completion budget. This is a weaker model on
scale, and the substitution is disclosed rather than presented as the planned arm. The
falsification criteria were not touched; they had already been fixed in writing.

**Run integrity.** 60 series, one origin each, 120 calls. Provider failures 0.0%,
unparsed replies 0.0% — the truncation path that motivated the guard did not trigger.

**Against the criteria fixed in advance:**

| # | Pre-registered falsifier | Observed | Fired? |
|---|---|---|---|
| 1 (primary) | critique's AUROC gap over the uncritiqued arm exceeds **+0.05** | **−0.097** (0.493 with critique, 0.590 without) | **No** |
| 2 (primary) | `llm_critique_gate` does not degrade MASE (point estimate $\le 0$) | **+5.62%** [−3.77, +15.32] | **No** |
| 3 (secondary) | intervention rate falls below **35%** | **78%** (single-pass 90%) | **No** |

None fired. The primary within-model result replicates in a third family and in the
strongest form yet seen: asking this model to critique itself did not merely fail to
improve its discrimination, it **reduced** it, from 0.590 to 0.493. The matched
arithmetic comparator scored 0.774 on the identical 60 instances and the identical label.

**One result runs the other way, and it sharpens rather than weakens the thesis.**
On action choice — *given* the decision to correct, which repair — this model is
markedly better than the two non-reasoning families. It recovers **+40.0%**
[−0.8, +67.1] of the coin-to-oracle headroom on the single-pass arm and +32.0%
[−5.8, +60.4] with critique, against roughly zero (−8.9% and −5.2%) for
`flash-lite`. Neither interval quite excludes zero at n=60, so this is reported as
directional rather than established.

The consequence is a cleaner statement of where the failure lives. Deliberation helps
with *which repair to apply* and does not help with *whether to apply one*; because the
arm still intervenes on 90% of instances, the net effect on accuracy remains negative.
The binding constraint is the gate, not the router. This was not predicted in advance and
is reported as an observation from the third family, not as a tested hypothesis.

**Caution on the reference arms.** These 60 series are a different draw from the
578-series subsample, and the non-LLM arms confirm it: `always_ensemble` is +5.04%
[−9.06, +24.71] here against −5.73% on the larger draw. The two intervals are compatible,
but no level from this family should be compared to a level from another. Only
*within-family, within-denominator* contrasts are read.

### D10 — Few-shot arm, matched zero-shot control, and a cache-key defect found in our own instrument (2026-09-19, 23:35 local)

**Deviation 1: a few-shot arm.** The LLM arms are zero-shot while the arithmetic gate
they are measured against is fitted on the validation split. A null result from a
zero-shot model is therefore ambiguous between "cannot read the evidence" and "was never
shown what a good decision looks like". Eight worked examples, drawn from **validation
only** and labelled by the counterfactual tensor, are now prepended to every prompt in a
separate arm. The mix follows the validation base rate rather than being balanced — a
balanced prefix would itself hint at how often to intervene, and intervention rate is the
quantity under study — and the positive examples are spread round-robin across distinct
oracle actions so the prefix does not quietly teach one repair. The example set is fixed
across instances, so the only thing differing between two prompts is the case being asked
about.

**It is run on the same 60 series as the third family**, alongside a **matched zero-shot
control on those identical series**, so the few-shot contrast is within-model on the same
instances and the two robustness results share one denominator.

**Deviation 2: a defect in the response cache, found and fixed.** `reasoning_effort` was
not part of the content-addressed cache key. Two runs of the same reasoning model at
different efforts would therefore have collided, and a medium-effort run could have been
served a low-effort reply with nothing in its output to show it. No published number is
affected — the only reasoning model whose entries survived is `gpt-oss-20b`, which was
run exclusively at medium effort, and its 120 entries were migrated to effort-aware keys.
The 264 `gpt-oss-120b` entries, written across two abandoned runs at two different
efforts, could not have their provenance recovered and were deleted rather than left
readable under an unknown setting. The field is added to the key only when set, so the
1,800-odd responses cached before reasoning models were introduced keep their original
keys. A regression test now asserts all three properties.

We record this because the paper argues that agent evaluations fail on measurement
hygiene, and this is a measurement-hygiene failure in our own instrument. It was caught
by asking what the cache key contained before trusting a cached read, which is the same
discipline the manuscript recommends to others.

### D11 — Fixed-base condition: both readings fixed in advance (2026-09-23, before the run completed)

**The confound.** Every result in this paper is conditional on a base forecast that was
itself chosen per instance by rolling-origin backtest over a nine-model pool. That is a
strong adaptive procedure in its own right. "Correction does not help" and "correction had
nothing left to fix" are therefore not separated anywhere in the main experiment, and the
separation is the single largest threat to the paper's claims. We name it in the
Limitations and we are now testing it.

**The condition.** `configs/fixed_base.yaml` fixes the base forecaster to ETS --- the model
the backtest selector picks most often, and a realistic default for a deployed pipeline ---
and changes nothing else. The resolved configurations of the two runs differ in exactly
three keys: `forecast.base_selection` (`backtest` to `fixed:ets`), `n_jobs` (`-1` to `6`,
a memory-bound computational setting that cannot affect any computed value), and the run
name and description. Seeds, splits, origins, data, action space, diagnostics and every
gate parameter are identical, and the MASE denominator is a function of the training
history and seasonal period alone, with no dependence on the base model. The comparison is
single-factor by construction and the manifests are released so that this can be checked
rather than taken on trust.

**Why this entry exists.** The result is not yet known. A paper whose central argument is
about selecting on outcomes should not decide, after seeing a number, which of two stories
that number tells. Both readings are therefore fixed here, in writing, with the
consequences each carries:

**Reading A --- correction helps on a fixed ETS base.** The confound is real and material.
The abstract's claims are then scoped, in the abstract itself and not only in the
Limitations, to pipelines whose base forecast is already adaptively selected. The
Discussion gains a paragraph arguing that the strength of the base pipeline is the
operative variable determining whether a correction stage pays, which is a more useful
finding for practitioners than the unqualified negative: it tells them when to bother. The
regret decomposition is unaffected either way, because it holds diagnosis at ground truth
and varies only repair selection.

**Reading B --- correction does not help on a fixed ETS base either.** The confound is
discharged. The largest limitation in Section~\ref{sec:limitations} is then replaced by a
strengthened claim: the result does not depend on the base forecast having been adaptively
chosen, and it holds for the conventional single-model pipeline that most deployed systems
actually run.

**Reading C --- the run does not complete.** Reported as not completed, in the Limitations,
in those words. The condition is not quietly dropped and the limitation is not softened.

**What is not permitted.** Choosing between A and B after the fact on any basis other than
the pre-registered comparison; reporting the condition only if it is favourable; or
reading a difference whose series-level interval includes zero as support for either
reading. The comparison is the series-level paired difference of each policy against never
correcting, computed inside each run separately and then set beside the other, exactly as
Table~\ref{tab:headline} is computed. Levels are not compared across runs, because the two
have different base forecasts and therefore different references.

### D11 (verdict) — Reading B: the base-pipeline confound is discharged (2026-09-24)

The fixed-base run completed (`fixed_base_20260923-164259_8a8949c5`, 578 series, 9,751
instances). `scripts/fixed_base_comparison.py`, written before the run finished, verified
the single-factor property mechanically before reporting: identical instance sets, and
every history-derived diagnostic identical between runs to within 1e-9.

| Policy | Adaptive (backtest) base | Fixed ETS base |
|---|---|---|
| always_correct | +8.13% [+2.73, +15.99] * | +10.59% [+1.21, +27.56] * |
| random_correct | +2.72% [+2.06, +3.46] * | +3.59% [+2.88, +4.40] * |
| threshold_single | +5.48% [+1.46, +10.88] * | +3.95% [+0.53, +9.29] * |
| rgc | +0.14% [−1.84, +3.03] | −0.49% [−1.64, +0.67] |

Each column is the series-level difference from never correcting *within its own run*;
levels are not compared across runs. Under a conventional single-model ETS base — the
pipeline most deployed systems actually run — no policy significantly improves on never
correcting, and unconditional correction remains significantly harmful. The hypothesis
that correction only failed because the adaptive base left nothing to fix is **rejected**.

Per the pre-registration, Reading B applies: the largest limitation in the Limitations
section is replaced by a strengthened claim, and the abstract is not scoped to adaptive
bases. No level from the fixed-base run is quoted beside a level from the main run.

**One arm excluded, and why.** The fixed-base `scores.parquet` contains an `llm_single`
policy although the run log records the LLM arm as skipped for lack of a client. It is
not present in the adaptive run, so it has no counterpart to compare against, and its
interval (−2.56% [−27.38, +14.44]) is consistent with a degenerate arm. It is excluded
from the comparison and from the manuscript pending the check recorded below.

**Resolution of the excluded arm, and a guard gap it exposed.** The fixed-base run's
`llm_single` arm made 747 live calls on 800 instances, and 747 of the replies were
**unparseable**: the model returned coherent prose instead of the JSON the prompt asks
for, so `_parse` fell back to `accept` on 93% of instances. Its 5% intervention rate is
therefore a parsing artefact, not a decision, and the arm is excluded from every table and
sentence. This is the same failure path that deviation D9 guarded against in
`scripts/llm_arms.py` (provider failures and unparsed replies counted against a 2%
ceiling, run aborted on breach). That guard lives in `llm_arms.py` and **not** in
`run_experiment.py`, so the experiment runner will score a degraded LLM arm without
complaint. The LLM results in the manuscript all come from `llm_arms.py` and are
unaffected; the gap is recorded here so the runner is not trusted for LLM arms until the
same ceiling is applied there.

### D12 — The regret decomposition was selection-inflated; replaced by a realizable one (2026-09-24)

**What was wrong.** A reviewer pointed out that the regret decomposition commits the
selection-inflated-oracle pathology this paper's own measurement-pathologies section warns
about. They were right. `scripts/regret_decomposition.py` compared the minimum over ten
actions against the minimum over two or three matched candidates, and a canonical fix
against that matched minimum, with **every minimum taken over test outcomes**. The
expected minimum of k noisy draws falls as k grows even when no action carries real
signal, so both named losses were inflated by order statistics alone. The table also
reported instance-level means with no intervals, the error deviation D7 corrected
everywhere else.

**What was done.** `scripts/regret_decomposition_realizable.py` makes every choice on
validation origins and scores it on held-out test origins, using the same rule as the
realizable oracle (per series and horizon, the action with the lowest mean validation
error). Every stage and loss term carries a series-level bootstrap interval.

**Result, MASE, 420 synthetic series with ground-truth diagnosis:**

| Quantity | Test-selected (as first reported) | Validation-selected |
|---|---|---|
| achievable headroom vs accept | −31.24% [−33.46, −29.10] | **+4.45% [+0.16, +9.95]** |
| first-choice loss | +17.55 pp | −2.90 pp [−16.62, +5.49] |
| restriction loss | +16.12 pp | +0.88 pp [−5.13, +9.89] |
| canonical fix vs accept | — | +2.43% [−0.50, +5.33] |

CRPS: realizable headroom −0.80% [−4.24, +2.79]; canonical fix +3.59% [+0.03, +7.26].

Between 98% and 114% of the reported headroom was winner's curse. There is no realizable
headroom for per-series repair selection on this benchmark, so the claim that the two
losses "erase the achievable headroom" is withdrawn: there was nothing realizable to
erase, and neither loss is distinguishable from zero once selection is removed.

**What survives, and replaces it.** (i) Under ground-truth diagnosis the canonical repair
does not improve on accepting, and on CRPS it is significantly worse. (ii) Even choosing
the best repair per series on validation does not beat accepting; on MASE it is
significantly worse. (iii) The oracle headroom that motivates correction is almost
entirely a selection artefact. (iv) The one realizable improvement is a single static
ensemble on CRPS, and that is exactly the action validation selects when choosing one
static policy globally, so it is not a post-hoc pick on test.

**Also withdrawn.** The robustness claim that "the oracle still recovers 32.9% at maximum
severity" used the same test-selected oracle. The perturbation sweep has no validation
origins, so a realizable version cannot be computed; the claim is removed rather than
repaired. The H2 difference-in-differences is unaffected, since it compares policies and
uses no oracle.

### D13 — Was `always_ensemble` a post-hoc pick? Checked on validation (2026-09-24)

**Objection.** `always_ensemble` was not among the pre-registered arms. Declaring the best
of ten static actions the winner after seeing test results would be selection on
outcomes.

**Check.** `scripts/static_policy_check.py` chooses the single static action with the
lowest mean error on validation origins, blind to test, and applies it everywhere.

| Metric | Validation chooses | Its test effect vs never | `always_ensemble` on test |
|---|---|---|---|
| CRPS | **ensemble** | −13.07% [−22.85, −7.27] | same row |
| MASE | refit_post_changepoint | **+4.31%** [+2.34, +7.17] (harmful) | −4.69% [−11.75, +0.60] (ns) |

**Reading.** On CRPS the ensemble is validation's own choice, so the result is realizable
and not a post-hoc pick. On MASE validation picks a different action that turns out
harmful, and the ensemble's own MASE interval includes zero; the paper accordingly claims
the ensemble result on CRPS only. The same script reports the CRPS effect on real series
alone (−17.29% [−39.71, −4.96], 158 series) and synthetic alone (−10.27% [−13.49, −6.99],
420 series), so it does not depend on generator design.

### D14 — Ablation table given series-level intervals; two claims withdrawn (2026-09-24)

The ablation table reported single-run instance-level means with no interval, the error
D7 corrected elsewhere. `scripts/run_ablations.py` now keeps per-instance scores
(`results/ablations_instances.parquet`) and reports each ablation against the full system
with a paired bootstrap over series. It also accepts `--run` to replay a completed run's
tensor instead of rebuilding it.

| Ablation | Instance-level Δ (old) | Series-level Δ [95% CI] |
|---|---|---|
| no_backtesting | +2.68% | **+4.41% [+0.92, +7.41]** |
| no_uncertainty | −1.45% | −1.36% [−3.97, +0.10] |
| no_changepoint | −1.02% | −0.98% [−3.52, +0.42] |
| no_correction_loop | −0.42% | −0.13% [−3.02, +1.84] |
| all others | within ±0.3% | include zero |

Only backtesting's removal is distinguishable from zero. The earlier reading that several
components were "mildly harmful" rested on point estimates whose intervals include zero
and is withdrawn. Removing the correction loop entirely is indistinguishable from the full
system, which restates the headline result as an ablation.
