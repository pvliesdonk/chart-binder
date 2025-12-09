# Contributing to Chart-Binder

Thank you for your interest in contributing to Chart-Binder!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/pvliesdonk/chart-binder.git
cd chart-binder

# Install dependencies (including dev dependencies)
uv sync --all-extras

# Run linter
uv run python devtools/lint.py

# Run tests
uv run pytest
```

See [docs/installation.md](docs/installation.md) for detailed setup instructions.

## Code Standards

- **Python**: Target 3.11-3.13
- **Package manager**: Always use `uv` (never `pip` or `python` directly)
- **Linting**: Ruff (zero warnings/errors required)
- **Type checking**: mypy strict mode
- **Testing**: pytest

### Commit Convention

Use conventional commits:

```
feat: implement artist normalization pipeline
fix: handle Nordic characters in diacritics
refactor: simplify guest extraction logic
test: add QA pack test for radio edit tags
docs: update normalization ruleset
```

## Quality Gates

Before submitting a PR, ensure:

```bash
# Run all checks
make all

# Or individually:
make lint
make test
```

## Documentation

- Specifications live in `docs/spec/`
- User-facing docs live in `docs/`
- QA packs and core specs live in `docs/appendix/`

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

---

## Publishing Releases

This project publishes to [PyPI](https://pypi.org/) via GitHub Actions using the
[uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning/) plugin.

### How to Publish

1. **Ensure tests pass**: Check the Actions tab for green CI.

2. **Create a release** on GitHub:
   - Go to Releases > Create new release
   - Create a new tag with semantic versioning (e.g., `v0.1.0`)
   - Fill in release notes
   - Submit

3. **Verify**: Watch the publish workflow in Actions tab. On success, the package
   appears at `https://pypi.org/project/chart-binder`.

### First-Time Setup (Maintainers)

1. Get a [PyPI account](https://pypi.org/)
2. Register the repo as a trusted publisher at [PyPI publishing settings](https://pypi.org/manage/account/publishing/)
3. Enter: project name, repo owner (`pvliesdonk`), repo name (`chart-binder`), workflow (`publish.yml`)

---

## Getting Help

- Check [GitHub Issues](https://github.com/pvliesdonk/chart-binder/issues)
- Read the [specification](docs/spec/overview.md)
