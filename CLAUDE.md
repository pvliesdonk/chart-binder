# Agent Instructions for chart-binder

This document provides guidance for AI agents working on the chart-binder project.

## Project Overview

**chart-binder** is a music chart data pipeline that normalizes, deduplicates, and enriches chart entries using MusicBrainz, Discogs, and Spotify APIs. The project follows a specification-driven approach with comprehensive documentation.

## Core Principles

### 1. Assistant Conduct

- **Be direct and factual**: Avoid unnecessary encouragement or fluff
- **Think step-by-step**: Show your reasoning process
- **Provide expert opinions**: Give technical recommendations without hedging
- **Reference line numbers**: When discussing code, use `file_path:line_number` format

### 2. Documentation as Source of Truth

- **The `docs/` directory is canonical**: All implementation must follow specs
- **Read specs before implementing**: Always check `docs/spec/overview.md` and relevant appendix files
- **Follow the roadmap**: Epic structure is defined in `docs/spec/roadmap.md`
- **Test against QA packs**: Appendix QA packs provide test cases with expected outputs

### 3. Development Workflow

- **Conventional commits**: Use `feat:`, `fix:`, `refactor:`, `test:`, `docs:` prefixes
- **Epic-based branches**: Branch format is `claude/epic-{number}-{description}-{session-id}`
- **Atomic commits**: One logical change per commit
- **Quality gates**: All code must pass lint and tests before push

## Python Development Standards

### Tools & Versions

- **Python**: Target 3.11-3.13
- **Package manager**: ALWAYS use `uv` (never `pip` or `python` directly)
- **Linting**: Ruff (zero warnings/errors required)
- **Type checking**: mypy strict mode
- **Testing**: pytest

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your values
```

**Required variables:**
- `MUSICBRAINZ_USER_AGENT` - Required by MusicBrainz API rate limiting policy
- `ACOUSTID_API_KEY` - For audio fingerprint lookups

**Optional - External APIs:**
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` - Spotify enrichment
- `DISCOGS_TOKEN` - Discogs lookups
- `OPENAI_API_KEY` - LLM adjudication (Phase D)
- `GOOGLE_API_KEY` - Google services

**Optional - Local MusicBrainz mirror:**
- `MUSICBRAINZ_DB_HOST` - PostgreSQL host (e.g., `192.168.50.212`)
- `MUSICBRAINZ_DB_PORT` - PostgreSQL port (default: `5432`)
- `MUSICBRAINZ_WEB_URL` - Web interface URL

**Optional - Local services:**
- `OLLAMA_HOST` - Local LLM endpoint
- `SEARXNG_URL` - Search engine
- `BEETS_CONFIG` - Path to beets config for playlist generation
- `MUSIC_LIBRARY` - Path to local music files

**Optional - Observability:**
- `LANGSMITH_*` - LangSmith tracing configuration

### Running Commands

```bash
# Install dependencies (including dev dependencies)
uv sync --all-extras

# Run linter
uv run python devtools/lint.py

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_normalization_qa_pack.py

# Run all quality checks
make all
```

### Code Style

**Imports**
- Use absolute imports (not relative)
- Group: stdlib, third-party, local
- Modern type syntax: `str | None` instead of `Optional[str]`
- Built-in generics: `dict[str, Any]`, `list[str]`

**Type Annotations**
- Full type annotations throughout
- Use `from __future__ import annotations` at top of file
- Use `typing.Any` sparingly

**Code Organization**
- All source code in `src/chart_binder/`
- All tests in `tests/`
- Inline tests below `## Tests` comments for simple modules
- Use bare `assert` statements (no pytest fixtures for inline tests)

**String Formatting**
- Use `textwrap.dedent` for multi-line strings
- Prefer f-strings for interpolation

**Comments & Docstrings**
- Comments explain *why*, not *what*
- Concise docstrings for public APIs only
- No docstrings for obvious functions

**Database Conventions**
- Use SQLite with explicit foreign key constraints
- Connection helper: `PRAGMA foreign_keys = ON`
- Upsert pattern for all write operations
- Use `time.time()` for timestamps (REAL type)
- Field naming: `created_at`, `fetched_at` (not `updated_at`)

### Testing Approach

**Test Organization**
- Simple modules: Inline tests below `## Tests` comment
- Complex features: Dedicated test files in `tests/`
- QA packs: Comprehensive test suites from spec appendices

**Test Style**
- Use bare `assert` statements
- No unnecessary fixtures or setup
- Test one concept per function
- Clear test names: `test_qa_radio_edit()` not `test_1()`
- Prefer `raise AssertionError("message")` over `assert False, "message"`

**Test Coverage**
- All QA pack cases must pass
- Edge cases from spec must be covered
- Idempotence checks for normalization functions

## Project Structure

```
chart-binder/
├── docs/                    # Canonical specifications
│   ├── spec/               # Developer/architecture docs
│   │   ├── overview.md     # Master specification
│   │   └── roadmap.md      # Epic-based roadmap
│   └── appendix/           # Detailed specs and QA packs
├── src/chart_binder/       # Source code
│   ├── config.py           # Configuration management
│   ├── http_cache.py       # HTTP caching layer
│   ├── musicgraph.py       # MusicBrainz schema
│   ├── charts_db.py        # Charts and alias registry
│   └── normalize.py        # Normalization pipeline
├── tests/                  # Test suite
│   └── test_normalization_qa_pack.py
└── devtools/               # Development utilities
    └── lint.py             # Linting wrapper
```

## Key Subsystems

### Normalization (Epic 3)

The normalization subsystem provides deterministic text processing for matching:

**Entry Points**
- `Normalizer.normalize_artist(artist: str) -> NormalizedResult`
- `Normalizer.normalize_title(title: str) -> NormalizedResult`

**Key Features**
- Article stripping (with exceptions: "The The", "De Dijk")
- Diacritics handling (NFD normalization + transliteration)
- Guest extraction (feat., ft., featuring, with)
- Edition tag extraction (live, remix, remaster, radio edit, etc.)
- Artist separator normalization (& → •, + → •)
- Whitespace and punctuation canonicalization

**Testing**
- All 30 QA pack tests must pass
- Check idempotence for all normalization operations
- Verify diacritics signature captures all non-ASCII

### Database Schema

**musicgraph.sqlite** - MusicBrainz entities
- `artist`: Core artist data with MBID
- `recording`: Track/song recordings
- `release_group`: Album/EP/Single groupings
- `release`: Specific releases (formats, regions)
- `recording_release`: Many-to-many relationships

**charts.sqlite** - Chart data and aliases
- `chart`: Chart metadata
- `alias_norm`: Normalization exception registry

**decisions.sqlite** - Matching decisions (future)

### Alias Registry

The alias_norm table stores normalization exceptions:

```python
db.upsert_alias(
    alias_id="alias-1",
    type="artist",           # or "title"
    raw="The Beatles",
    normalized="beatles",
    ruleset_version="norm-v1"
)
```

## Git Workflow

### Branch Strategy

**Branch naming**: `claude/epic-{number}-{description}-{session-id}`

Example: `claude/epic-3-normalization-013pzWAhc8duj7f1wXhJ1zkh`

**Important**: The session ID suffix must match your current session for push authentication.

### Commit Convention

Use conventional commits:

```
feat: implement artist normalization pipeline
fix: handle Nordic characters in diacritics
refactor: simplify guest extraction logic
test: add QA pack test for radio edit tags
docs: update normalization ruleset
```

**Commit scope** (optional): `feat(norm):`, `fix(cache):`, etc.

### Development Cycle

1. **Read specs**: Check `docs/spec/overview.md` and relevant appendix
2. **Create branch**: Follow naming convention
3. **Implement**: Write code following standards
4. **Test**: Ensure all tests pass (`make test`)
5. **Lint**: Ensure clean linting (`make lint`)
6. **Commit**: Use conventional commits
7. **Push**: Push to feature branch
8. **PR**: Create pull request with comprehensive description

### Pull Request Format

```markdown
## Summary
- [High-level description of changes]
- [What epic/milestone this addresses]

## Changes
- [Specific change 1]
- [Specific change 2]

## Test Coverage
- [QA pack results: X/Y passing]
- [New tests added]

## Implementation Notes
- [Technical decisions]
- [Spec deviations if any]
```

### Merge Strategy

**Goal**: Minimize commits and keep history clean.

**For simple changes (1-2 commits):**

- Merge directly to main without a PR
- Use `git merge <branch> --no-gpg-sign` then push

**For complex changes (3+ commits or needs review):**

1. Try to create PR via `gh pr create` with inline body
2. If that fails, create a PR description file (`EPIC{N}_PR.md`)
3. Remove the PR file before merging to keep repo clean

**Direct merge example:**

```bash
git checkout main && git pull
git merge claude/feature-branch --no-gpg-sign
git push origin main
git branch -d claude/feature-branch  # cleanup local
```

**PR creation example:**

```bash
gh pr create --title "feat: add feature X" --body "$(cat <<'EOF'
## Summary
- Brief description

## Changes
- Change 1
- Change 2
EOF
)"
```

## Common Tasks

### Adding a New Normalization Rule

1. Check if rule is in `docs/appendix/normalization.md`
2. Update `Normalizer` class in `src/chart_binder/normalize.py`
3. Add inline test demonstrating the rule
4. Verify QA pack tests still pass
5. Update diacritics or exception lists as needed

### Extending Database Schema

1. Check spec in `docs/spec/overview.md`
2. Add fields to schema in appropriate DB class
3. Update upsert methods
4. Add test for new functionality
5. Consider adding index for query performance
6. Follow additive-only migrations (no destructive changes)

### Adding a New API Client

1. Implement using `HTTPCache` for rate limiting
2. Store raw responses in cache
3. Parse into domain entities
4. Use schema_meta table to track versions
5. Add comprehensive error handling
6. Test with real API (use VCR for CI)

## Migration Strategy

**Current approach**: Additive migrations only

- Use `schema_meta` table for version tracking
- Store migration SQL in `migrations/` directory when needed
- Never drop columns or tables
- Add new columns with sensible defaults
- Rename by adding new column + deprecating old

**Future**: Consider Alembic for more complex migrations

## Anti-Patterns to Avoid

❌ **Don't**:
- Edit database schemas destructively
- Skip reading specs before implementing
- Use `pip` or `python` directly (use `uv`)
- Add docstrings to obvious functions
- Create trivial test cases
- Use relative imports
- Hardcode configuration values
- Skip type annotations
- Push code with lint errors

✅ **Do**:
- Read specs first
- Follow epic structure from roadmap
- Use conventional commits
- Write inline tests for simple modules
- Add comprehensive QA pack tests
- Use type annotations throughout
- Handle errors gracefully
- Document non-obvious decisions

## Resources

- **Specification**: `docs/spec/overview.md`
- **Roadmap**: `docs/spec/roadmap.md`
- **Normalization Rules**: `docs/appendix/normalization.md`
- **QA Pack**: `docs/appendix/normalization-qa.md`
- **Database Schema**: `docs/database-schema.md`
- **Environment Config**: `.env.example`

## Quick Reference

```bash
# Full quality check
make all

# Individual checks
make install
make lint
make test

# Run specific test
uv run pytest tests/test_normalization_qa_pack.py -v

# Type check only
uv run mypy src/

# Format check
uv run ruff check src/
```

When in doubt, check the specs first. The documentation is comprehensive and should answer most questions.
