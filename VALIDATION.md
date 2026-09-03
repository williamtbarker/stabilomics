# Validation record

## Scope

The validation target is deliberately limited: demonstrate that the implementation recovers a known
sparse signal reproducibly from a controlled correlated dataset before and after heavy-tailed noise
and gross outcome contamination. This does not reproduce the source paper or validate biomarkers.

`scripts/validate_science.py` generates paired 160-row, 24-feature datasets with identical predictors,
age covariate, and Student-t noise. The contaminated outcome receives 8% additional signed deviations
of magnitude 8–14. Both datasets contain the same three planted coefficients.

Parameters are alpha 0.15, 40 subsamples, 0.70 sampling fraction, 0.80 consensus threshold, and seed
42.

## Recorded result

Run on Python 3.12.13 with NumPy 2.2.6 and SciPy 1.18.1:

```text
clean consensus: gene_001, gene_002, gene_003
contaminated consensus: gene_001, gene_002, gene_003
contaminated signal-frequency floor: 0.950
clean median pairwise Jaccard: 1.000
contaminated median pairwise Jaccard: 0.800
scientific fixture validation: PASS
```

The validation script fails unless both consensus sets equal the planted feature set and every
planted feature is selected in at least 90% of contaminated-data subsamples.

## What this establishes

- deterministic recovery of the planted sparse signal for this fixture;
- persistence of that consensus after controlled outcome contamination;
- correct aggregation of selection frequencies and pairwise selection-set stability;
- an executable regression target for future solver or preprocessing changes.

## What this does not establish

- superiority to another estimator across a representative benchmark;
- equivalence to the authors' R implementation;
- calibrated false-discovery control;
- performance on a real genomic cohort;
- biological, diagnostic, or clinical validity.
