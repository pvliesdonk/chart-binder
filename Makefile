.PHONY: all install lint test

all: install lint test

install:
	uv sync --all-extras

lint:
	uv run python devtools/lint.py

test:
	uv run pytest
