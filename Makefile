.PHONY: all install lint test

all: install lint test

install:
	uv sync

lint:
	uv run python devtools/lint.py

test:
	uv run pytest
