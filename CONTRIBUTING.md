# Contributing

Bug reports and narrowly scoped pull requests are welcome. Please include a minimal synthetic example
that contains no protected or confidential data.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make verify
```

Changes to statistical behavior must document the objective, scaling convention, assumptions, and a
deterministic validation case. Do not weaken input validation or clinical-use limitations merely to
accept a dataset.
