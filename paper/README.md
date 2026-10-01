# Manuscript

Target: **Transactions on Machine Learning Research (TMLR)**.

TMLR judges a submission on two questions: are the claims supported by accurate,
convincing and clear evidence, and would some individuals in TMLR's audience be
interested in the findings. There is explicitly no novelty or significance bar and no
page limit. That fits a measurement paper whose headline result is partly negative and
whose value is in the rigour of the measurement rather than in a new state of the art —
better than a conference, where the same work competes on perceived impact, and better
than a journal whose reviewers may not be the audience for the claim.

The previous Frontiers-targeted preamble is kept at `main_frontiers.tex`. It includes
the same `sections/*.tex` files, so both variants stay in sync automatically; only the
preamble and back matter differ.

## Build

```bash
# 1. Regenerate every number from the latest completed run
python scripts/make_paper_assets.py --run latest

# 2. Check what a compile would catch, without needing LaTeX
python scripts/verify_manuscript.py --strict

# 3. Compile
cd paper && latexmk -pdf main.tex
```

Style files (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`) are vendored here from
<https://github.com/JmlrOrg/tmlr-style-file>. Options:

| Option | Use |
|---|---|
| `\usepackage{tmlr}` | anonymous, for submission — **current setting** |
| `\usepackage[preprint]{tmlr}` | named, for arXiv |
| `\usepackage[accepted]{tmlr}` | camera-ready |

Two build targets share every section file and differ only in the byline:

| Command | Output | Use |
|---|---|---|
| `bash scripts/build_paper.sh main` | `main.pdf` (anonymous) | **TMLR submission**; double-blind is required |
| `bash scripts/build_paper.sh main_preprint` | `main_preprint.pdf` (named byline) | arXiv / sharing; never submit to TMLR |

## Integrity rule

**This manuscript contains no hardcoded experimental numbers.**

| What | Where it comes from |
|---|---|
| Tables | `tables/*.tex`, `\input` fragments |
| Figures | `figures/*.pdf` |
| Inline numbers in prose | `generated/macros.tex` |
| Machine-readable summary | `generated/results_summary.json` |

All four are written by `scripts/make_paper_assets.py` from `results/`. Rerunning the
experiment re-renders the paper; the text cannot silently drift from the data. Never
type an experimental number into a `.tex` file — add a macro instead.
`scripts/verify_manuscript.py` fails if the prose uses a macro nothing generates.

**Inference is series-level.** Every claim of a difference rests on
`evaluation/series_level.py`, which collapses each policy to one value per series and
bootstraps over series. Instance-level statistics appear in the repo and are labelled
descriptive; they must never be quoted to assert a difference. See deviation D7 in
`docs/ANALYSIS_PLAN.md` for what went wrong the first time and what it changed.

## Structure

| File | Contents |
|---|---|
| `sections/00_abstract.tex` | Abstract |
| `sections/01_introduction.tex` | Introduction, related work, gap, contributions, hypotheses |
| `sections/02_methods.tex` | Benchmark design, method, data, metrics, analysis plan |
| `sections/03_results.tex` | Results, decomposition, triage, ablations, robustness |
| `sections/04_discussion.tex` | Interpretation, relation to prior work, limitations |
| `sections/05_backmatter.tex` | Hypothesis table, broader impact, reproducibility, data availability |

TMLR **requires** a Broader Impact Statement; it is in the back matter. Author
contributions and acknowledgements are camera-ready only, since submissions are
anonymous.

## Before submission

- [ ] **Resolve every `\todo{}`.** They render in red and are deliberately hard to
      miss. None may remain. `scripts/verify_manuscript.py` counts them.
- [ ] **Keep the submission anonymous.** `\usepackage{tmlr}` handles the title block,
      but check the text for self-identifying references and anonymise the repository
      URL in the data-availability statement for review.
- [ ] **Verify citations**: `python scripts/verify_citations.py --strict`. Must report
      zero unresolved and zero mismatches.
- [ ] **Re-run the literature review.** This area moves fast; confirm no near-duplicate
      has appeared and record the date in `LITERATURE_REVIEW.md`.
- [ ] **Complete the LLM arms**, or state plainly in the limitations that they are
      absent. The paper's argument is about LLM-driven correction; running none of it
      is the one gap a reviewer is most entitled to object to. See `ANALYSIS_PLAN.md`
      D8 for the quota constraint and what it cost.
- [ ] **Archive the code and benchmark** (e.g. Zenodo) and insert the DOI into the data
      availability statement.
- [ ] **Check figure resolution** (300 dpi minimum; PDFs are vector and satisfy this).
- [ ] **Reconsider the name.** `TS-Agent` (arXiv:2510.07432) already exists and the
      `*Bench` namespace is crowded (TemporalBench, TSAIA, IRTS-ToolBench), so
      `TS-AgentBench` invites confusion. The benchmark's distinguishing property is that
      it stores counterfactual outcomes for corrective actions; a name built on that
      would be both clearer and safer.

## If TMLR is not the right home

- **International Journal of Forecasting** — the natural venue for the
  forecast-combination framing, with reviewers who know that literature well. Expect
  them to push hardest on the classical prior, which the introduction already concedes.
- **Frontiers in Artificial Intelligence** — `main_frontiers.tex` is ready; note the
  ~1,950 CHF APC and the mandatory generative-AI disclosure.
