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

## AI-assisted development disclosure

This project was designed and implemented with substantial assistance from OpenAI Codex under human
direction. The system researched candidate problems, drafted code and documentation, generated
synthetic fixtures, and ran automated checks. Will Barker is expected to review the statistical
assumptions, source relationship, code, tests, and public claims before publication. AI assistance
does not constitute scientific validation, and no claim in this repository should be accepted solely
because a test passes.

## Releasing

1. Create a fresh virtual environment on a supported Python version.
2. Install `.[dev]` and run `make verify`.
3. Run the README quickstart and compare the generated report with the documented expectation.
4. Review `git diff --check`, the license, citation metadata, and version numbers.
5. Tag only after human review.
