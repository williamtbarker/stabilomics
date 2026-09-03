# Project selection report

## Decision

Build **Stabilomics**, an independent Python operationalization of robust LAD-LASSO stability
selection for high-dimensional scientific tables.

The immediate prompt was Yang, Lu, and Wu's Bioinformatics article, published 17 June 2026 and
corrected/typeset 10 July 2026. It argues that heavy-tailed disease outcomes and contamination can
make high-dimensional feature selection unstable, and combines LAD-LASSO with repeated subsampling.

## Candidate comparison

Scores use a 1–5 scale. “Defensible” means the claims can be bounded and tested in one release.

| Candidate | Useful | Distinct | Defensible | Feasible | Validatable | Portfolio | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust LAD-LASSO stability-selection CLI | 5 | 4 | 5 | 4 | 5 | 5 | **28** |
| Cross-split biological-sequence leakage auditor | 5 | 3 | 4 | 4 | 4 | 5 | 25 |
| Spatial-omics batch-biased ranking auditor | 4 | 4 | 3 | 3 | 3 | 4 | 21 |

The sequence-leakage idea was rejected because DataSAIL already offers a sophisticated,
leakage-reducing splitter, and homology-aware split/audit tools have substantial prior art. A narrow
auditor might still be useful, but its approximate-similarity claims would require a larger benchmark.

The spatial-omics idea was motivated by BatchSVG (Bioinformatics, July 2026) but was rejected for this
release because method-specific assumptions and realistic validation data would exceed a narrow,
fully defensible build.

## Prior art and contribution boundary

Reviewed prior art included the paper authors' `cenwu/RSS` R example, the archived
`scikit-learn-contrib/stability-selection` project, Python `ipss`, `c-lasso`, and the broader robust
sparse-regression ecosystem. These establish that neither stability selection nor robust sparse
regression is new.

Stabilomics adds a small, inspectable combination:

- exact LP formulation of LAD-LASSO through SciPy HiGHS;
- intercept and numeric covariates excluded from the L1 penalty;
- deterministic subsampling and stable machine-readable output;
- selection-frequency, coefficient-direction, and pairwise set-stability evidence together;
- strict scientific-table validation and optional CI gates;
- a reproducible synthetic heavy-tail/contamination fixture.

This is an adaptation and engineering contribution, not a claimed invention of the statistical
method and not an exact replication of the source paper.

## Data and license check

Only generated synthetic data are included. No TCGA, eQTL, patient, controlled-access, or source-paper
data are redistributed. The implementation was written independently from the paper description.
The source repository did not visibly specify a license, so no code was copied. NumPy and SciPy are
BSD-licensed projects; their binary distributions can include third-party runtime components under
compatible terms. This repository vendors none of them.

## Strongest reason not to publish

The current scientific validation is controlled and synthetic. It demonstrates deterministic
behavior, solver invariants, recovery of planted signals, and workflow correctness, but does not
benchmark against the authors' R implementation or reproduce the paper's real-data results. Publish
only if that explicitly bounded “independent operational adaptation” is a portfolio asset you are
comfortable defending.
