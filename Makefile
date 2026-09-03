.PHONY: format lint typecheck test science package verify

format:
	python -m ruff format --check .

lint:
	python -m ruff check .

typecheck:
	python -m mypy src

test:
	python -m pytest

science:
	python scripts/validate_science.py

package:
	python -m build
	python -m twine check dist/*

verify: format lint typecheck test science package
