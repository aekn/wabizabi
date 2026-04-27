.PHONY: sync fmt lint typecheck test check

sync:
	uv sync --all-packages

fmt:
	uv run --all-packages ruff format .

lint:
	uv run --all-packages ruff check . --fix

typecheck:
	uv run --all-packages pyright

test:
	uv run --all-packages pytest

check: lint typecheck test
