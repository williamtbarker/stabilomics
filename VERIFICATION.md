# Verification record

Release candidate: Stabilomics 0.1.0

## Source-tree gate

Executed on Linux under both CPython 3.10.20 and CPython 3.12.13:

```bash
python -m pip install -e '.[dev]'
make verify
```

Results on both interpreters:

- Ruff format check: pass (23 files)
- Ruff lint: pass
- strict mypy: pass (9 source files)
- pytest: pass (39 tests)
- branch-aware coverage: 93.81%
- paired synthetic scientific validation: pass
- source distribution and universal wheel build: pass
- Twine metadata checks: pass
- generated CSV newline regression and Git whitespace check: pass

The scientific fixture recovered exactly `gene_001`, `gene_002`, and `gene_003` before and after 8%
gross outcome contamination. The contaminated-data minimum selection frequency among planted signals
was 0.950; median pairwise selection-set Jaccard was 0.800.

## Clean wheel install

A new CPython 3.12 virtual environment installed only the built wheel and its declared runtime
dependencies. The installed `stabilomics` command generated the fixture and completed the documented
quickstart twice. Recursive diff of both result directories was empty, confirming byte-identical
JSON, CSV, and Markdown output for identical inputs and parameters on the same numerical stack.

## Dependency audit

`pip-audit --local` reported no known vulnerabilities in the resolved Python 3.12 verification
environment. The unpublished local `stabilomics` distribution was necessarily skipped because it is
not on PyPI. Runtime dependency licenses were inspected: NumPy and SciPy are BSD-licensed projects;
binary distributions may include compatible third-party runtime components.

## Platform boundary

Linux execution was performed locally. The repository configures the identical gate for Python 3.10
and 3.12 on `ubuntu-latest` and `macos-latest`, but those GitHub-hosted jobs cannot run until a human
publishes the repository. The macOS commands in the handoff are therefore an important final check.
