# Controlled synthetic validation benchmark — 2026-09-03

This benchmark tests reproducible recovery of planted variables. It does not establish biological
validity, biomarker utility, causal relevance, or performance on a real cohort.

## Environment and source

- Host: Apple M2 Max (`Mac14,6`), 12 logical CPUs, 96 GiB RAM, macOS-14.7.6-arm64-arm-64bit, Python 3.12.7, Rust/Cargo 1.98.1
- Stabilomics commit: `de1313b52d006a8a39528aa474d8d40eaa19de93`
- Working tree: clean
- Replicates: three independent complete harness invocations

## Results

The package's deterministic simulator plants `gene_001`, `gene_002`, and `gene_003`. Every one of
the nine measured fits selected exactly those three variables: planted-feature precision 1.000,
recall 1.000, and median pairwise selection-set Jaccard 1.000.

Times and peak RSS are medians with observed three-run ranges in brackets.

| Samples × candidate features | Subsamples | Gross outcome contamination | Wall time, s | Peak RSS, MiB | Precision / recall | Median Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| 160 × 24 | 40 | 0% | 0.463 [0.460–0.473] | 70.8 [70.0–78.5] | 1.000 / 1.000 | 1.000 |
| 160 × 24 | 40 | 8% | 0.468 [0.453–0.472] | 70.2 [70.0–70.4] | 1.000 / 1.000 | 1.000 |
| 250 × 50 | 20 | 8% | 0.500 [0.491–0.508] | 75.2 [75.1–76.1] | 1.000 / 1.000 | 1.000 |

## Interpretation

The strongest result is invariance of the selected set when 8% gross outcome contamination is
introduced into the matched 160 × 24 case. The 250 × 50 case uses only 20 subsamples, so its runtime
must not be used as a direct feature-scaling comparison with the 40-subsample cases.

## Limitations

- The data are simulated from the package's own documented generator.
- Exact signal recovery in these fixtures does not imply recovery in observational genomic data.
- The alpha and stability threshold were fixed rather than tuned across a sensitivity grid.
- Correlated alternatives, weak effects, missingness, categorical covariates, and real batch effects
  are not represented.
- The current matrix does not isolate computational scaling because the widest case uses fewer
  subsamples.

## Reproduction

```bash
python3 -u benchmark.py run --profile standard --label stabilomics-r1 --tool stabilomics
```

Run three complete invocations and retain the generated input hashes and full reports.
