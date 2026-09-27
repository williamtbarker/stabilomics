# Method provenance and scope

## Scientific motivation

**Stabilomics** is an independent Python operationalization of robust LAD-LASSO stability
selection for high-dimensional scientific tables.

The method is motivated by Yang, Lu, and Wu's Bioinformatics article, published 17 June 2026 and
corrected/typeset 10 July 2026. It argues that heavy-tailed disease outcomes and contamination can
make high-dimensional feature selection unstable, and combines LAD-LASSO with repeated subsampling.

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

## Validation boundary

The current scientific validation is controlled and synthetic. It demonstrates deterministic
behavior, solver invariants, recovery of planted signals, and workflow correctness, but does not
benchmark against the authors' R implementation or reproduce the paper's real-data results.
These results support the implementation's controlled behavior, not real-data scientific validity.
