# Development notes

## Design choices

- Python was selected because NumPy/SciPy provide an auditable scientific-computing interface and the
  HiGHS linear-program solver on both Linux and macOS. Recent READY portfolio releases already meet
  the rolling Rust requirement.
- LAD-LASSO is represented as one linear program with positive/negative feature and residual
  variables. Covariates remain unrestricted and unpenalized.
- Robust scaling uses median and IQR. Constant columns are errors instead of being silently removed.
- Random subsamples come from a local NumPy generator with an explicit seed. Outputs omit timestamps
  and absolute paths to stay reproducible and portable.
- Reports emphasize stability evidence and limitations rather than presenting a single feature list
  as ground truth.

## Releasing

1. Create a fresh virtual environment on a supported Python version.
2. Install `.[dev]` and run `make verify`.
3. Run the README quickstart and compare the generated report with the documented expectation.
4. Review `git diff --check`, the license, citation metadata, and version numbers.
5. Tag only after human review.
