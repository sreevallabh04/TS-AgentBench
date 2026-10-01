# Literature Review

**Scope.** Recent work on time-series foundation models, LLMs for time-series forecasting,
agentic time-series reasoning, self-correction in AI agents, uncertainty estimation,
selective prediction, adaptive forecasting, time-series benchmarks, and AI agents for
analytical tasks.

**Compiled:** 2026-09-18. **Re-check before submission** — this literature is moving fast
(see §11).

## Verification protocol

Every entry is tagged:

| Tag | Meaning |
|-----|---------|
| `[V]` | Title and identifier **verified during this review** by retrieving the source page. |
| `[K]` | Cited from model knowledge; identifier **not yet machine-verified**. |

`scripts/verify_citations.py` queries the arXiv/Crossref APIs for every identifier in this
file and in `paper/references.bib`, and fails if a title does not match the recorded one.
**No `[K]` entry may appear in the submitted manuscript until it has passed that check.**
This is the project's structural defence against fabricated citations.

---

## 1. Agentic time-series forecasting and reasoning

The central finding of this review. As of 2026 this area is **crowded**, which constrains
what we may claim.

| Paper | Year | ID | Contribution | Limitations | Relevance |
|---|---|---|---|---|---|
| Position: Beyond Model-Centric Prediction — Agentic Time Series Forecasting `[V]` | 2026 | [arXiv:2602.01776](https://arxiv.org/abs/2602.01776) | Argues forecasting should be reframed from single-pass prediction to an agentic workflow (perception, planning, action, reflection, memory). Outlines workflow-based, agentic-RL and hybrid paradigms. | A position paper: no benchmark, no empirical validation of the reflection stage it advocates. | **Blocks a novelty claim.** "Forecasting should be agentic" is already argued in print. We must not claim it. Its open problems motivate us. |
| TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis `[V]` | 2025 | [arXiv:2510.01538](https://arxiv.org/abs/2510.01538) | Self-described *first* LLM-driven agentic framework for general TS forecasting. Four agents: Curator (LLM-guided diagnostics + tools), Planner (model pre-selection), Forecaster (fitting, validation, ensembling), Reporter. Reports 10.4% error reduction vs statistical and 38.2% vs LLM baselines over 8 benchmarks. | Correction/adaptation is embedded in agent prompts; **no reported criterion for when adaptation fires**. No calibration or abstention metrics. Compute not matched against baselines. | **Closest prior art for the agent.** We must position against it, and must not claim to be first. |
| Nexus: An Agentic Framework for Time Series Forecasting `[V]` | 2026 | [arXiv:2605.14389](https://arxiv.org/abs/2605.14389) | Decomposes forecasting into macro/micro temporal fluctuation isolation plus contextual (news/event) integration; emits reasoning traces. Evaluated beyond LLM knowledge cutoffs (Zillow, equities). | Focus is contextual/textual reasoning, not reliability. No failure-detection, uncertainty or abstention machinery reported in the abstract. | Establishes that multi-stage agentic decomposition is solved. Our differentiator is the *decision to re-forecast*, not the decomposition. |
| CastFlow: Learning Role-Specialized Agentic Workflows for Time Series Forecasting `[V]` | 2026 | [arXiv:2604.27840](https://arxiv.org/abs/2604.27840) | Learns role-specialized agentic workflows with planning → action → forecasting → reflection, enabling iterative forecast refinement. Explicitly builds on Self-Refine and Reflexion. | Abstract does not disclose the refinement **trigger**; no ablation of correction policies (always/never/random/threshold) documented. | Direct evidence for our gap: iterative refinement is adopted without published evidence that the trigger is principled. |
| TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning `[V]` | 2026 | [arXiv:2605.10038](https://arxiv.org/abs/2605.10038) | Agent with exploratory execution learning for TS analysis. | Not examined in depth; reliability gating not advertised. | Contemporaneous agentic system; cite in related work. |
| Bridging the Last Mile of Time Series Forecasting with LLM Agents `[V]` | 2026 | [arXiv:2606.02497](https://arxiv.org/abs/2606.02497) | Action-centric agent converting contextual reasoning into constrained, evidence-backed **edits** over a shared forecast workspace; long-horizon decomposition; memory-bank post-hoc reflection. | Whether edits are triggered by quantitative reliability signals or LLM judgement is **not disclosed** in the abstract. No calibration/abstention evaluation stated. | Closest in spirit to a typed corrective-action space. Must be read in full and contrasted carefully before we claim C1/C2. **Action item.** |
| FLAIRR-TS: Forecasting LLM-Agents with Iterative Refinement and Retrieval `[V]` | 2025 | [arXiv:2508.19279](https://arxiv.org/abs/2508.19279) | Test-time prompting with iterative refinement and retrieval for TS forecasting. | Refinement is unconditional test-time prompting; no reliability gate. | Instance of the unexamined premise. |
| TS-Agent: Understanding and Reasoning Over Raw Time Series via Iterative Insight Gathering `[V]` | 2025 | [arXiv:2510.07432](https://arxiv.org/abs/2510.07432) | Delegates statistical extraction to TS tools, uses the LLM only to gather evidence and synthesize — explicitly limiting the LLM to what it is good at. | Targets understanding/reasoning benchmarks, not forecast reliability or correction. | Strong precedent for our design principle (LLM routes, tools measure). **Name collision risk with `TS-AgentBench`** — see §11. |
| KairosAgent: Agentic Time Series Forecasting with Fused Semantic Reasoning `[V]` | 2026 | [arXiv:2605.30002](https://arxiv.org/abs/2605.30002) | Agentic forecasting fusing semantic reasoning. | Not examined in depth. | Contemporaneous; cite. |
| From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection `[V]` | 2024 | [arXiv:2409.17515](https://arxiv.org/abs/2409.17515) | LLM agents refine selection logic by comparing forecast errors to events; iterative self-evaluation surfaces overlooked news. | Reflection is over *news selection*, not forecast reliability; error feedback uses realized outcomes. | Early example of error-feedback-driven refinement. Note its feedback is retrospective, ours is pre-hoc. |
| AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection `[V]` | 2025 | [arXiv:2505.12594](https://arxiv.org/abs/2505.12594) | Multi-agent pipeline for anomaly detection. | Anomaly detection only. | Tool-orchestration precedent. |
| A Survey of Reasoning and Agentic Systems in Time Series with LLMs `[V]` | 2026 | [TMLR; repo](https://github.com/blacksnail789521/Time-Series-Reasoning-Survey) | Taxonomy by reasoning topology (direct / linear chain / branch-structured) and objective (analysis / explanation / causal-decision / generation). | **The taxonomy contains no uncertainty category, and lists no uncertainty papers.** | **Primary evidence for gap 3.** A survey of the field finding no uncertainty work is a citable justification for our contribution. |

### Assessment

Multi-agent decomposition, tool use, reasoning traces and iterative refinement are **solved
and published**. What is consistently absent is any *evidence* that the refinement stage is
correctly triggered. Several papers do not even disclose the trigger.

---

## 2. Self-correction in AI agents

| Paper | Year | ID | Contribution | Limitations | Relevance |
|---|---|---|---|---|---|
| Large Language Models Cannot Self-Correct Reasoning Yet `[V]` | 2024 | [arXiv:2310.01798](https://arxiv.org/abs/2310.01798), ICLR'24 | Shows *intrinsic* self-correction (no external feedback) fails to improve and often **degrades** reasoning accuracy. | Reasoning tasks, not forecasting. | **Theoretical anchor for C2.** Predicts that verbal self-critique in TS agents should fail. We test this in a domain where an external signal (backtest error) is unusually cheap and well-defined. |
| When Can LLMs Actually Correct Their Own Mistakes? `[V]` | 2024 | [TACL, doi:10.1162/tacl_a_00713](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/) | Critical survey establishing the narrow conditions under which self-correction succeeds: gains come from outside the model — a tool-interactive critic, execution or environment signal. | Survey; no TS. | Motivates grounding the critic in backtesting rather than prompting. |
| Self-Refine: Iterative Refinement with Self-Feedback `[K]` | 2023 | arXiv:2303.17651 | Iterative self-feedback refinement. | Unconditional refinement; no gate. | **Implemented as our "always-correct" arm.** |
| Reflexion: Language Agents with Verbal Reinforcement Learning `[K]` | 2023 | arXiv:2303.11366 | Verbal RL with episodic memory for self-correction. | Needs task-level reward signal. | Cited by CastFlow as the basis for TS reflection; the lineage we test. |
| ReAct: Synergizing Reasoning and Acting in Language Models `[K]` | 2023 | arXiv:2210.03629 | Interleaved reasoning and tool acting. | No reliability estimation. | Baseline agent scaffold. |

### Assessment

The negative result is established for reasoning. **Nobody has tested whether it transfers
to time-series forecasting** — a domain where, unusually, a cheap and principled external
signal exists (rolling-origin backtest error). That makes forecasting the *right* domain in
which to ask whether grounded correction escapes the Huang et al. result. This is the
intellectual core of the paper.

---

## 3. Forecast failure / error prediction (the signal we gate on)

| Paper | Year | ID | Contribution | Limitations | Relevance |
|---|---|---|---|---|---|
| Mast: interpretable stress testing via meta-learning for forecasting model robustness evaluation `[V]` (preprint: Meta-learning and Data Augmentation for Stress Testing Forecasting Models) | 2025 | [Springer ML, doi:10.1007/s10994-025-06881-3](https://link.springer.com/article/10.1007/s10994-025-06881-3); cf. [arXiv:2406.17008](https://arxiv.org/abs/2406.17008) | Binary probabilistic classifier predicting forecasting-model **stress** from TS features, using model outputs (error scores, prediction intervals). Defines three stress manifestations: large errors, high uncertainty, and **hubris** (high error + low uncertainty). Validated on 6 datasets / 97,829 series; adds augmentation to improve stress classification. | Purely **offline stress testing and explanation**. Never used to control an agent; no corrective action is taken. | **Direct ancestor of our risk estimator.** Our novelty is closing it into a control loop. The "hubris" construct maps exactly onto our miscalibration analysis (H5) — adopt the terminology and cite. |
| FFORMPP: Feature-based Forecast Model Performance Prediction `[V]` | 2019 | [arXiv:1908.11500](https://arxiv.org/abs/1908.11500) | Models forecast error as a function of TS features via Bayesian multivariate surface regression; uses minimum predicted error to select a model/combination. | Selection only, at fit time. No notion of re-forecasting after observing diagnostics; no agent. | Establishes that error is predictable from features — the premise of H3. Our H3 control condition ("features only") is essentially FFORMPP, so we must beat it to claim anything. |
| FFORMA: Feature-based Forecast Model Averaging `[V]` | 2020 | [IJF; PDF](https://robjhyndman.com/papers/fforma.pdf) | Meta-learner selecting combination weights from series features; 2nd in M4. | Static weighting; no reconsideration. | Strong non-LLM baseline for model selection (Baseline A). |
| Evaluation-free Time-series Forecasting Model Selection via Meta-learning `[V]` | 2025 | [ACM TKDD, doi:10.1145/3715149](https://dl.acm.org/doi/10.1145/3715149) | Model selection without running candidate models. | Selection only. | Relevant to H4 (cost reduction). |

### Assessment

Predicting forecast failure is **solved as an offline diagnostic**. The step nobody has
taken is using a calibrated failure prediction as the **gate on agentic corrective action**.
That step is contribution C2, and its ancestry is explicit and citable.

---

## 4. Time-series benchmarks

| Paper | Year | ID | Contribution | Limitations | Relevance |
|---|---|---|---|---|---|
| TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks `[V]` | 2026 | [arXiv:2602.13272](https://arxiv.org/abs/2602.13272); KDD'26 [doi:10.1145/3770855.3817485](https://dl.acm.org/doi/10.1145/3770855.3817485) | LLM-agent benchmark over 4 domains (retail, health, energy, physical) with a 4-tier taxonomy: historical structure interpretation, context-free forecasting, contextual reasoning, event-conditioned prediction. Finds forecasting accuracy does **not** imply robust contextual reasoning. Public data + leaderboard. | Evaluates *reasoning quality under context*, not the agent's control decisions. No counterfactual action outcomes, no correction-decision ground truth, no calibration. | **Closest benchmark competitor.** We must state precisely how TS-AgentBench differs: it records what *would* have happened under alternative actions. |
| TimeSage-MT: A Multi-Turn Benchmark for Evaluating Agentic Time Series Reasoning `[V]` | 2026 | [arXiv:2606.01498](https://arxiv.org/abs/2606.01498) | 240 multi-turn agentic TS reasoning tasks / 2,680 turns across 8 domains, from real data with verifiable answers. Explicitly exposes failures in memory, **uncertainty handling** and decision making. | Multi-turn dialogue evaluation; uncertainty is probed qualitatively, not scored by calibration metrics. No counterfactual action outcomes. | Nearest work on "agents handle uncertainty badly". Cite as corroborating our motivation; differentiate on quantitative calibration. |
| TimeSage-EV: A Live Benchmark for Agentic Time Series Analysis in Evolving Environments `[V]` | 2026 | [arXiv:2608.14270](https://arxiv.org/abs/2608.14270) | Live benchmark for agentic TS analysis in evolving environments. | Very recent; live-eval focus. | Contemporaneous; check before submission. |
| TimeSeriesGym: A Scalable Benchmark for (Time Series) Machine Learning Engineering Agents `[V]` | 2025 | [arXiv:2505.13291](https://arxiv.org/abs/2505.13291) | Benchmark for TS **ML-engineering** agents: data handling, repo comprehension, code translation; hybrid numeric + LLM-judge evaluation. | Engineering-workflow evaluation, not forecast reliability. Abstract does not cover abstention or self-correction. | Different axis; cite to delimit scope. |
| AION: Next-Generation Tasks and Practical Harness for Time Series `[V]` | 2026 | [arXiv:2605.25045](https://arxiv.org/abs/2605.25045) | Task/workspace/validation-interface formalism plus a 6-component harness (agents, skills, rules, memory, evaluation, protocols). Design principles include **reliability mechanisms** (post-experiment analysis, layered review). | Harness and protocol contribution; reliability is procedural (review steps), not a calibrated quantitative gate. | Nearest on the word "reliability". Differentiate explicitly: procedural review vs calibrated risk estimation. **Read in full.** |
| MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering `[V]` | 2025 | [arXiv:2503.16858](https://arxiv.org/abs/2503.16858) | Multimodal TS benchmark for temporal reasoning. | Reasoning, not control. | Related work. |
| Towards Verifiable Agentic Data Science: Solving Irregular TSQA Via Tool-Grounded Reasoning `[V]` | 2026 | [arXiv:2606.15107](https://arxiv.org/abs/2606.15107) | 1,700 questions / 10 task types / 13 domains for irregular TS QA; scores answer correctness **and tool-grounded reasoning behaviour**. | QA over irregular series; no forecasting-correction decisions. | Precedent for scoring tool use, which informs our tool-selection metric (H4). |
| GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation `[V]` | 2024 | [arXiv:2410.10393](https://arxiv.org/abs/2410.10393) | 28 datasets, 144k series, 177M points across domain/frequency/variate/horizon; train/val/test splits plus a non-leaking pretraining corpus; 17 baselines. | Pure forecast accuracy. No agentic or decision-level evaluation. | Gold standard for split hygiene and baseline breadth. We follow its protocol conventions and cite it as our evaluation ancestor. |
| Monash Time Series Forecasting Archive `[V]` | 2021 | [arXiv:2105.06643](https://arxiv.org/abs/2105.06643); [forecastingdata.org](https://forecastingdata.org/) | 30+ datasets; CC-BY-4.0; research use. | Accuracy only. | **Primary real-data source.** License recorded in `datasets/LICENSES.md`. |

### Assessment

There are many TS-agent benchmarks. **None records counterfactual outcomes of alternative
corrective actions**, so none can score the correction decision itself. That is the precise,
narrow and defensible sense in which TS-AgentBench is new — and it is the only benchmark
claim we will make.

---

## 5. Uncertainty estimation and selective prediction

| Paper | Year | ID | Contribution | Limitations | Relevance |
|---|---|---|---|---|---|
| Conformal Prediction for TS Forecasting with Change Points (CPTC) `[V]` | 2025 | NeurIPS'25; [repo](https://github.com/Rose-STL-Lab/CPTC) | Couples a state-prediction model with online CP for non-stationary series with sudden shifts. | Standalone UQ; not connected to agent control. | Our conformal recalibration action (A8) and changepoint-aware intervals. |
| Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series `[V]` | 2024 | [arXiv:2402.05203](https://arxiv.org/abs/2402.05203) | Calibrates multi-horizon prediction intervals via dynamic programming. | UQ only. | Multi-horizon interval construction. |
| Optimization-based Online Conformal Prediction for Multi-step Forecasting `[V]` | 2025 | [arXiv:2508.13362](https://arxiv.org/abs/2508.13362) | Online CP for multi-step horizons. | UQ only. | Interval machinery. |
| Improved Online Conformal Prediction via Strongly Adaptive Online Learning `[V]` | 2023 | [arXiv:2302.07869](https://arxiv.org/abs/2302.07869) | Strongly adaptive online CP. | UQ only. | Adaptive interval baseline. |
| A Gentle Introduction to Conformal Time Series Forecasting `[V]` | 2025 | [arXiv:2511.13608](https://arxiv.org/abs/2511.13608) | Tutorial/overview. | — | Reference for methods section. |
| Drift-Aware Spectral Conformal Prediction for Non-Exchangeable Streaming Data `[V]` | 2026 | [arXiv:2606.15953](https://arxiv.org/abs/2606.15953) | CP under drift without exchangeability. | UQ only. | Drift-robust intervals. |
| Adaptive Conformal Inference Under Distribution Shift `[K]` | 2021 | arXiv:2106.00170 | Online interval adaptation under shift. | UQ only. | Foundational for A8. |
| Conformal PID Control for Time Series Prediction `[K]` | 2023 | arXiv:2307.16895 | Control-theoretic view of conformal calibration. | Calibrates intervals, not actions. | **Conceptually adjacent to our thesis** (control loop over forecasts); differentiate: they control interval width, we control whether to re-forecast. |
| Selective Classification for Deep Neural Networks `[K]` | 2017 | arXiv:1705.08500 | Risk–coverage framework for abstention. | Classification. | Our risk–coverage/AURC analysis (H5). |
| On Calibration of Modern Neural Networks `[K]` | 2017 | arXiv:1706.04599 | ECE; temperature scaling. | Classification. | ECE methodology. |
| AgentAbstain: Do LLM Agents Know When Not to Act? `[V]` | 2026 | [arXiv:2607.10059](https://arxiv.org/abs/2607.10059) | First systematic evaluation of **agentic abstention**: calibrated ability of tool-using agents to recognise when not to act; should-act vs should-abstain variants. | Generic tool agents; not forecasting; no notion of corrective re-planning. | **Strongest conceptual sibling.** Same question ("should the agent act?"), different domain and no counterfactual outcome data. Must cite and differentiate prominently. |
| I-CALM: Incentivizing Confidence-Aware Abstention `[V]` | 2026 | [arXiv:2604.03904](https://arxiv.org/abs/2604.03904) | Prompt-based abstention incentives for black-box models. | Prompt-level; no quantitative gate. | Contrast for our LLM-verbal-confidence arm. |

### Assessment

UQ for time series is mature; agentic abstention is emerging for generic agents. The
intersection — **calibrated uncertainty driving a forecasting agent's control decisions** —
is empty. The TMLR'26 survey (§1) independently confirms this by having no uncertainty
category at all.

---

## 6. Time-series foundation models and LLMs for forecasting

All `[K]`; used as baselines rather than as claims. To be machine-verified before citation.

| Paper | Year | ID | Role in this project |
|---|---|---|---|
| Chronos: Learning the Language of Time Series | 2024 | arXiv:2403.07815 | **Foundation-model baseline** (Chronos-Bolt-small, CPU-feasible). |
| A decoder-only foundation model for time-series forecasting | 2024 | arXiv:2310.10688 | Alternative TSFM; compute permitting. |
| Moirai / Unified Training of Universal TS Forecasting Transformers | 2024 | arXiv:2402.02592 | Alternative TSFM (multivariate-native). |
| Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting | 2023 | arXiv:2310.08278 | Probabilistic TSFM; optional. |
| MOMENT: A Family of Open Time-series Foundation Models | 2024 | arXiv:2402.03885 | General TS foundation model; optional. |
| LLMTime: Large Language Models Are Zero-Shot Time Series Forecasters | 2023 | arXiv:2310.07820 | **Direct-numeric LLM forecasting**; informs our LLM-agent arms and contamination concerns. |
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | 2024 | arXiv:2310.01728 | LLM-reprogramming line; related work. |
| A Time Series is Worth 64 Words: Long-term Forecasting with Transformers | 2023 | arXiv:2211.14730 | **Transformer forecasting baseline** (patched, CPU-tractable). |
| Are Transformers Effective for Time Series Forecasting? | 2023 | arXiv:2205.13504 | Critical baseline; guards against over-claiming deep models. |
| Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting | 2021 | arXiv:2012.07436 | Source of the **ETT** dataset. |
| The M4 Competition: 100,000 series and 61 methods | 2020 | IJF | M4 data and evaluation conventions (MASE, sMAPE). |

---

## 7. Statistical methodology

All `[K]`; standard references.

| Method | Reference | Use |
|---|---|---|
| Diebold-Mariano test | Diebold & Mariano 1995, *JBES* | Pairwise forecast-accuracy comparison. |
| Harvey-Leybourne-Newbold correction | 1997, *IJF* | Small-sample DM correction. |
| Wilcoxon signed-rank | Wilcoxon 1945 | Non-parametric paired comparison across series. |
| Holm-Bonferroni | Holm 1979, *Scand. J. Statist.* | Multiplicity control across hypotheses/strata. |
| Cliff's delta | Cliff 1993 | Non-parametric effect size. |
| MASE | Hyndman & Koehler 2006, *IJF* | Scale-free accuracy; primary metric. |
| CRPS | Gneiting & Raftery 2007, *JASA* | Probabilistic scoring. |

---

## 8. What is already solved

1. Multi-agent decomposition of forecasting into diagnose -> select -> forecast -> report.
2. Tool-augmented TS reasoning where the LLM orchestrates and statistical tools measure.
3. Large-scale, leakage-controlled forecast-accuracy benchmarking (GIFT-Eval, Monash).
4. Benchmarking LLM agents on TS reasoning, QA, multi-turn dialogue and ML engineering.
5. Predicting forecast error/stress from series features, offline (FFORMPP, MAST).
6. Conformal UQ for non-stationary series with drift and change points.
7. Abstention evaluation for generic tool-using agents (AgentAbstain).
8. The negative result that intrinsic self-correction fails without external feedback.

## 9. What is missing

1. **The correction decision is never evaluated.** No benchmark records counterfactual
   outcomes of alternative corrective actions, so "should the agent have corrected?" has no
   ground truth and no paper can measure it.
2. **Failure prediction is never closed into a control loop.** MAST/FFORMPP stop at
   diagnosis.
3. **Uncertainty is absent from agentic TS systems.** Confirmed by a TMLR'26 survey whose
   taxonomy has no uncertainty category. No TS agent reports ECE or risk-coverage.
4. **Comparisons are compute-confounded.** Correcting agents spend strictly more compute
   than non-correcting ones; no published TS-agent comparison controls for this.
5. **Repair is not failure-mode-conditioned.** Agents "reflect" generically rather than
   diagnosing a failure mode and applying a matched repair from a typed action space.

## 10. Our precise research gap

> A fast-growing 2025-26 literature adds self-correction to time-series forecasting agents.
> No published work tests whether the correction step is triggered correctly, whether its
> benefit survives compute matching, or whether the agent's confidence is calibrated —
> because no benchmark records what would have happened under a different action.

**Research question.** Can a forecasting agent predict its own failure accurately enough to
decide when re-forecasting is worth the compute — and does self-correction help once compute
is controlled for?

**Post-hoc update (2026-09-19).** The empirical answer turned out to be more specific than
the question anticipated, and the contribution below reflects what the completed
experiments actually showed rather than what was hypothesised going in. See
`docs/ANALYSIS_PLAN.md` deviations D1–D6 for the full record of what changed after the
first test-set evaluation and why.

### Claims we will and will not make

| | |
|---|---|
| NOT claimed | First agentic TS framework (TimeSeriesScientist, Nexus, CastFlow precede us). |
| NOT claimed | First TS agent benchmark (TemporalBench, TimeSage-MT, TimeSeriesGym, AION precede us). |
| NOT claimed | First to argue forecasting should be agentic (position paper 2602.01776 precedes us). |
| NOT claimed | First to predict forecast failure from features (FFORMPP, MAST precede us). |
| NOT claimed | First agentic abstention evaluation (AgentAbstain precedes us). |
| NOT claimed | That per-instance adaptive correction beats a static baseline — our own experiments show the opposite, and we report that rather than the hypothesis we started with. |
| **C1** | First TS benchmark with **counterfactual ground truth for the correction decision** — every corrective action pre-executed and scored per instance, yielding an oracle action, a `should_correct` label and computable regret for any policy. |
| **C2** | A correction policy gated on **calibrated quantitative failure risk**, with **failure-mode-conditioned** repair and **backtest-validated** acceptance — extending offline stress prediction (MAST/FFORMPP) into agent control. Empirically, this policy avoids being actively harmful but is dominated by a trivial static ensemble; we report this honestly rather than tuning the comparison to favour C2. |
| **C3** | The first **compute-matched audit** of whether, when and why self-correction helps TS forecasting agents, reporting calibration and selective prediction. The audit's finding is negative for adaptive gating and positive for probabilistic-accuracy repair via a static default. |
| **C4 (new)** | The first **quantitative attribution of agentic correction failure to two separable architectural causes**, using ground-truth failure-mode labels to hold diagnosis fixed at "perfect" and isolate (a) the loss from a fixed canonical diagnosis-to-repair mapping versus validated selection among candidates, and (b) the loss from the candidate list itself excluding the best available repair. We are not aware of prior work that runs this experiment on any agentic time-series system, because it requires exactly the counterfactual-per-action outcome data C1 provides. |
| **C5 (new)** | A demonstration that the same failure-risk estimate that does not support good automated correction is a **well-calibrated selective-prediction / triage signal**, reported with precision/recall/lift at multiple review budgets — a positive, deployable finding independent of whether automated correction works. |

C3 returned a negative result for adaptive gating specifically: on this benchmark, no
adaptive policy we built beats a trivial static ensemble. This is reported as found,
per the pre-registration commitment, rather than reframed after the fact. C4 is what
makes the negative result more than a benchmark leaderboard entry: it identifies *why*
diagnosis-conditioned correction underperforms, in terms specific enough to suggest what
would need to change (validated candidate selection; a learned rather than hand-designed
repair vocabulary) rather than only that it currently does not work.

## 11. Open risks to novelty

1. **Read-in-full before submission**: *Bridging the Last Mile* (2606.02497) — typed edit
   actions are conceptually near C1; *AION* (2605.25045) — "reliability mechanisms";
   *CastFlow* (2604.27840) — refinement trigger; *TimeSage-MT* (2606.01498) — uncertainty
   handling. If any performs calibrated gating or counterfactual action scoring, C1/C2 must
   be narrowed.
2. **Name collision**: `TS-Agent` (2510.07432) exists. `TS-AgentBench` appears unused, but
   reconsider the name before submission.
3. **Fast-moving field**: re-run this review immediately before submission and record the
   date. A near-duplicate appearing mid-project is a live risk.
4. **`[K]` entries are unverified** and must pass `scripts/verify_citations.py` before
   entering the manuscript.
