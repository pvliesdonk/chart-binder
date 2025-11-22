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
- **Read specs before implementing**: Always check `docs/spec.md` and relevant appendix files
- **Follow the roadmap**: Epic structure is defined in `docs/roadmap.md`
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

### Running Commands

```bash
# Install dependencies
uv sync

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
│   ├── spec.md             # Master specification
│   ├── roadmap.md          # Epic-based roadmap
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

1. **Read specs**: Check `docs/spec.md` and relevant appendix
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

### Pull Request Description Workflow

**IMPORTANT**: To make PR descriptions easy to access and copy, follow this workflow:

**When creating a PR:**

1. **Create PR description file** at the root of the repo:
   - Filename: `EPIC{N}_PR.md` (e.g., `EPIC4_PR.md`)
   - Content: Full PR description in markdown format
   - Commit and push this file with your branch

2. **Provide GitHub links** to the user:
   - PR creation link: `https://github.com/{owner}/{repo}/compare/main...{branch}?expand=1`
   - Direct raw link for copying: `https://raw.githubusercontent.com/{owner}/{repo}/{branch}/EPIC{N}_PR.md`
   - Alternative: View file at `https://github.com/{owner}/{repo}/blob/{branch}/EPIC{N}_PR.md` and click "Raw" button

3. **Benefits**:
   - No need to copy/paste from Claude Code interface
   - Description viewable directly on GitHub
   - Easy to access via "Raw" button
   - Version controlled with the branch

**Before merging the PR:**

1. **Remove the PR description file**:
   ```bash
   git rm EPIC{N}_PR.md
   git commit -m "docs: remove PR description file before merge"
   git push
   ```

2. **Why remove it before merge**:
   - PR description is preserved in the GitHub PR itself
   - Keeps repo clean (file doesn't end up in main)
   - Avoids stale documentation files
   - No longer needed once PR is created

3. **When to remove**:
   - Can be removed right after PR creation (once description is transferred to GitHub)
   - Or as part of addressing review comments
   - Must be removed before merge to keep main clean

**Example workflow:**

```bash
# 1. Creating PR
echo "PR content..." > EPIC4_PR.md
git add EPIC4_PR.md
git commit -m "docs: add Epic 4 PR description"
git push
# Now create the PR on GitHub using the file

# 2. Before merge (choose one approach):

# Option A: Remove immediately after PR creation
git rm EPIC4_PR.md
git commit -m "docs: remove PR description file"
git push

# Option B: Remove when addressing review comments
git rm EPIC4_PR.md
# Make your fixes
git add src/...
git commit -m "fix: address PR review comments"
git push

# Either way, ensure it's removed before merge
```

**Note**: This pattern solves the copy-paste difficulty in web-based Claude Code interface while keeping the repo clean long-term.

## Common Tasks

### Adding a New Normalization Rule

1. Check if rule is in `docs/appendix/normalization_ruleset_v1.md`
2. Update `Normalizer` class in `src/chart_binder/normalize.py`
3. Add inline test demonstrating the rule
4. Verify QA pack tests still pass
5. Update diacritics or exception lists as needed

### Extending Database Schema

1. Check spec in `docs/spec.md`
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

- **Specification**: `docs/spec.md`
- **Roadmap**: `docs/roadmap.md`
- **Normalization Rules**: `docs/appendix/normalization_ruleset_v1.md`
- **QA Pack**: `docs/appendix/normalization_qa_pack_v1.md`
- **MusicGraph Schema**: `docs/appendix/musicgraph_schema.md`
- **Chart Schema**: `docs/appendix/chart_schema.md`

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
