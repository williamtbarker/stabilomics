# Stabilomics

[![CI](https://github.com/williamtbarker/stabilomics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/williamtbarker/stabilomics/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/williamtbarker/stabilomics)](https://github.com/williamtbarker/stabilomics/releases) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Stabilomics is a deterministic Python CLI and library for asking a sharper question than “which
features did my model select?”:

> Which features survive repeated perturbation of a heavy-tailed scientific dataset?

It repeatedly fits median-regression LASSO (LAD-LASSO) on seeded subsamples, keeps covariates
unpenalized, and reports per-feature selection frequency, coefficient direction consistency, and
pairwise Jaccard stability of the selected sets. It is designed for high-dimensional genomics and
other scientific tables where outliers can make a single sparse fit look more decisive than it is.

Stabilomics is an independent, operational adaptation of the robust stability-selection strategy
described by Yang, Lu, and Wu (2026). It does not copy their R code, reproduce their case studies, or
claim numerical identity with their implementation.

## Why this is useful

Ordinary LASSO uses squared residual loss and can react strongly to heavy-tailed outcomes or gross
outliers. LAD-LASSO replaces squared loss with absolute loss. Stability selection then repeats that
fit across subsamples and prioritizes features that recur. The source paper reports improved feature
reproducibility from this combination in controlled simulations and two genomic case studies.

The authors provide a compact R example. Existing Python projects provide general stability
selection or alternative robust sparse estimators, but a small installable CLI combining exact
linear-program LAD-LASSO, free covariates, deterministic subsampling, workflow gates, and auditable
JSON/CSV/Markdown output was not obvious in the reviewed ecosystem.

## Installation

Stabilomics requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Five-minute quickstart

Generate a deterministic synthetic table with correlated features, Student-t noise, and 8% gross
outcome contamination:

```bash
stabilomics simulate \
  --output heavy_tailed.csv \
  --samples 160 \
  --features 24 \
  --seed 7
```

Then run robust stability selection:

```bash
stabilomics fit \
  --input heavy_tailed.csv \
  --outcome phenotype \
  --id-column sample_id \
  --covariate age \
  --feature-prefix gene_ \
  --alpha 0.15 \
  --subsamples 40 \
  --fraction 0.70 \
  --threshold 0.80 \
  --seed 42 \
  --output-dir results
```

Expected console output for the bundled fixture:

```text
PASS: 3 consensus features; median pairwise Jaccard 0.800
wrote results
```

The result directory contains:

- `features.csv`: every feature ranked by selection frequency;
- `report.json`: deterministic parameters, input hash, diagnostics, results, and limitations;
- `report.md`: a compact human-review report.

The three planted signals are `gene_001`, `gene_002`, and `gene_003`. Recovering them in the toy
fixture validates controlled behavior, not biological validity.

## Input contract

Input is a rectangular CSV or TSV table. Gzip compression is detected by magic bytes. Every modeled
value must be finite and numeric. Missing values are rejected rather than silently imputed or
dropped.

```text
sample_id,phenotype,age,gene_001,gene_002,...
sample_001,2.41,55.2,0.18,-0.90,...
```

Specify one outcome, zero or more repeated `--covariate` options, and optionally an ID column.
Covariates and the intercept are fitted but not penalized. By default, every remaining column is a
candidate feature; `--feature-prefix` narrows the feature set.

Constant or near-constant outcome, feature, or covariate columns are rejected because robust
standardization is undefined for them. Categorical covariates must be encoded explicitly before use.

## Method

Each feature, numeric covariate, and the outcome is median-centered and divided by its interquartile
range using the full input table. This makes the fixed penalty comparable across differently scaled
columns while limiting sensitivity to extreme values.

For each seeded subsample, Stabilomics solves

```text
mean(|y - Xβ - Cγ - intercept|) + α ||β||₁
```

as a linear program using SciPy's HiGHS backend. `β` contains penalized candidate features and `γ`
contains unpenalized covariates. A feature is selected when `|β|` exceeds
`--coefficient-tolerance`.

Across fits, the tool reports:

- selection frequency and consensus membership at `--threshold`;
- median standardized coefficient among selecting fits and among all fits;
- sign consistency among nonzero coefficients;
- selected-feature count per fit;
- all pairwise Jaccard similarities between selected sets.

If both compared selections are empty, Jaccard is defined as 1.0. Random row selections are sorted
before fitting, and no wall-clock timestamps enter outputs, so identical inputs and parameters yield
byte-identical reports on the same numerical stack.

### Choosing parameters

The defaults are starting points, not universal biological thresholds. The paper used 80 subsamples,
a 0.70 subsample fraction, and a 0.90 selection-frequency threshold in its simulations. This package
defaults to 40 and 0.80 for a faster exploratory run. For consequential analysis, rerun an alpha and
threshold sensitivity grid and look for conclusions that persist.

`alpha` is expressed for the mean absolute-error objective above; it is not numerically identical to
every R or Python LASSO convention. Larger values select fewer features.

## Workflow gates

Optional gates make instability visible in CI without pretending to define scientific truth:

```bash
stabilomics fit \
  --input heavy_tailed.csv \
  --outcome phenotype \
  --id-column sample_id \
  --covariate age \
  --feature-prefix gene_ \
  --alpha 0.15 \
  --subsamples 40 \
  --threshold 0.80 \
  --min-stable-features 1 \
  --min-median-jaccard 0.70 \
  --output-dir results
```

A failed gate still writes all reports and exits with code 2. Invalid input or solver failure exits
with code 1.

## Python API

```python
from stabilomics import StabilityConfig, stability_select

result = stability_select(
    features,
    outcome,
    covariates=covariates,
    config=StabilityConfig(alpha=0.15, subsamples=80, threshold=0.9, seed=42),
)
print(result.selection_frequency)
```

Arrays are NumPy-compatible and must contain finite floating-point values.

## Verification

```bash
python -m pip install -e '.[dev]'
make verify
```

The gate checks Ruff formatting and linting, strict mypy, branch-aware pytest coverage, the paired
clean/contaminated scientific fixture, wheel/source distribution construction, and package metadata
with Twine. CI runs it on Python 3.10 and 3.12 on Linux and macOS.

## Controlled synthetic benchmark

Across three independent runs, Stabilomics recovered exactly the three planted features in matched
clean and 8%-contaminated heavy-tailed datasets: precision 1.000, recall 1.000, and median pairwise
selection-set Jaccard 1.000. These are controlled simulation results, not evidence of biological or
causal validity. [Method, timings, and limitations](docs/BENCHMARK_MACOS_2026-09-03.md).

## Scientific limitations

- Stable selection does not establish causality, clinical utility, or biological validity.
- Correlated predictors may substitute for one another, depressing individual selection frequency.
- Full-table robust scaling is unsupervised, but nested preprocessing is still required when the
  selected features feed a predictive evaluation.
- Repeated subsamples overlap, so selection frequencies and Jaccard summaries are descriptive; they
  are not independent-trial confidence intervals.
- This release supports continuous outcomes and numeric covariates only.
- It does not implement the paper's complete simulation suite, TCGA/eQTL analyses, or every tuning
  convention in the authors' R example.
- Linear programming scales less favorably than coordinate descent. Benchmark your own dimensions.

Do not use the output alone for diagnosis, treatment, biomarker qualification, or other clinical or
public-health decisions.

## Sources and relationship to prior work

- Yang G, Lu X, Wu C. “Robust prioritization of genomic features with stability selection.”
  *Bioinformatics* 42(7), 2026. [DOI: 10.1093/bioinformatics/btag398](https://doi.org/10.1093/bioinformatics/btag398)
- Meinshausen N, Bühlmann P. “Stability selection.” *Journal of the Royal Statistical Society:
  Series B* 72(4), 2010. [DOI: 10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x)
- Virtanen P, et al. “SciPy 1.0: fundamental algorithms for scientific computing in Python.”
  *Nature Methods* 17, 2020. [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

The source paper's repository is [cenwu/RSS](https://github.com/cenwu/RSS). It contains a short R
example and did not display a software license during this project's review. No source code or data
were copied from it. Stabilomics is original MIT-licensed Python code based on the published method
description and established statistical ideas.

## License

MIT. See `LICENSE`.
