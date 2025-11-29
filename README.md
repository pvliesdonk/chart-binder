# Chart-Binder

**Charts-aware audio tagger**: Pick the most canonical release, link MusicBrainz/Discogs/Spotify IDs, and embed compact chart history into your audio files.

Chart-Binder solves the problem of determining the "canonical" version of a song when you have multiple release options (singles, albums, compilations, remasters). It uses chart data and intelligent heuristics to pick the best representative release and maintains traceable decisions.

## Features

- **Canonical Release Group (CRG) Selection**: Automatically picks the most representative release group for a track using chart data and heuristics
- **Representative Release (RR) Selection**: Chooses the best release within a group based on country/label preferences and official status
- **Chart Data Integration**: Ingest and link chart data to build historical context for songs
- **Text Normalization**: Robust artist/title normalization with locale-aware rules (NL/EN focus) and edition tag extraction
- **Audio Tag Management**: Read and write ID3/Vorbis/MP4 tags with backup of original values
- **Multi-Source Data**: Integrate data from MusicBrainz, Discogs, Spotify, Wikidata, and AcoustID
- **HTTP Caching**: Persistent caching with configurable TTL for API responses
- **LLM Adjudication**: Optional AI-powered decision-making for ambiguous cases (Ollama/OpenAI)
- **Human Review Queue**: Handle edge cases that need manual review
- **Decision Drift Detection**: Track and review decisions that may need updating
- **Beets Plugin**: Integrate with the beets music library manager

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/pvliesdonk/chart-binder.git
cd chart-binder

# Install dependencies with uv (includes beets plugin support)
uv sync --all-extras

# Or install with pip
pip install -e .
```

### Development Setup

```bash
# Install all dependencies including dev
make install

# Run linting and type checking
make lint

# Run tests
make test

# Run all checks
make
```

## Quick Start

### 1. Scan Audio Files

Read existing metadata from your audio files:

```bash
# Scan a single file
canon scan path/to/song.mp3

# Scan a directory recursively
canon scan /path/to/music/library/

# Output as JSON
canon scan --output json path/to/song.mp3
```

### 2. Make Canonicalization Decisions

Resolve the best release group and release for each track:

```bash
# Make decisions for files
canon decide path/to/song.mp3

# Show detailed rationale
canon decide --explain path/to/song.mp3
```

### 3. Write Tags

Apply canonical tags to your audio files:

```bash
# Preview changes (dry run)
canon write --dry-run path/to/song.mp3

# Apply changes
canon write --apply path/to/song.mp3
```

### 4. Work with Charts

Import and link chart data:

```bash
# Ingest chart data from JSON
canon charts ingest nl_top40 2024-W01 chart_data.json

# Link chart entries to work keys
canon charts link nl_top40 2024-W01

# View coverage report
canon coverage chart nl_top40 2024-W01

# Export chart blob for a work
canon charts export "Artist Name // Song Title"
```

## CLI Reference

The main command is `canon` with several subcommands:

| Command | Description |
|---------|-------------|
| `canon scan` | Scan audio files and discover metadata |
| `canon decide` | Make canonicalization decisions |
| `canon write` | Write canonical tags to files |
| `canon cache status` | Show cache statistics |
| `canon cache purge` | Clear caches |
| `canon charts ingest` | Import chart data |
| `canon charts link` | Link chart entries |
| `canon charts export` | Export chart blob |
| `canon coverage chart` | Show coverage report |
| `canon coverage missing` | Show unlinked entries |
| `canon drift review` | Review drifted decisions |
| `canon llm status` | Show LLM configuration |
| `canon llm adjudicate` | Run LLM adjudication |
| `canon review list` | List pending reviews |
| `canon review accept` | Accept a review |
| `canon review skip` | Skip a review |

### Global Options

```bash
canon --config path/to/config.toml  # Custom config file
canon --offline                     # Run without network requests
canon --frozen                      # Use only cached data
canon --refresh                     # Force refresh cached data
canon --output json                 # Output as JSON
```

## Configuration

Create a `config.toml` file:

```toml
[http_cache]
directory = ".cache/http"
ttl_seconds = 86400
enabled = true

[database]
music_graph_path = "musicgraph.sqlite"
charts_path = "charts.sqlite"
decisions_path = "decisions.sqlite"

[live_sources]
# Rate limits
musicbrainz_rate_limit = 1.0  # requests per second
discogs_rate_limit = 25       # requests per minute
acoustid_rate_limit = 3.0     # requests per second

# Cache TTLs (seconds)
cache_ttl_musicbrainz = 3600
cache_ttl_discogs = 86400
cache_ttl_spotify = 7200

[llm]
enabled = false
provider = "ollama"  # or "openai"
model_id = "llama3.2"
auto_accept_threshold = 0.85
review_threshold = 0.60
```

### Environment Variables

Configuration can also be set via environment variables:

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_OFFLINE_MODE` | Run in offline mode |
| `CHART_BINDER_HTTP_CACHE_DIRECTORY` | Cache directory path |
| `CHART_BINDER_HTTP_CACHE_TTL_SECONDS` | Cache TTL |
| `ACOUSTID_API_KEY` | AcoustID API key |
| `DISCOGS_TOKEN` | Discogs API token |
| `SPOTIFY_CLIENT_ID` | Spotify client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify client secret |
| `CHART_BINDER_LLM_ENABLED` | Enable LLM adjudication |
| `CHART_BINDER_LLM_PROVIDER` | LLM provider (ollama/openai) |
| `CHART_BINDER_LLM_MODEL_ID` | Model identifier |

## Python API

```python
from chart_binder.tagging import verify, write_tags, TagSet
from chart_binder.resolver import Resolver, ConfigSnapshot
from chart_binder.normalizer import normalize

# Read tags from an audio file
tagset = verify("path/to/song.mp3")
print(f"Title: {tagset.title}")
print(f"Artist: {tagset.artist}")
print(f"MB Release Group: {tagset.ids.mb_release_group_id}")

# Normalize artist/title text
result = normalize("Artist Name feat. Guest")
print(f"Core: {result.core}")
print(f"Tags: {result.tags}")

# Create resolver and make decisions
config = ConfigSnapshot(lead_window_days=90)
resolver = Resolver(config)
decision = resolver.resolve(evidence_bundle)
print(f"CRG: {decision.release_group_mbid}")
print(f"Rationale: {decision.crg_rationale}")
```

## Documentation

- [Quick Start Guide](docs/quickstart.md) - Get started quickly
- [CLI Reference](docs/cli-reference.md) - Complete command documentation
- [Configuration Guide](docs/configuration.md) - All configuration options
- [API Reference](docs/api-reference.md) - Python API documentation
- [Beets Plugin](docs/beets-plugin.md) - Using with beets
- [Specification](docs/spec.md) - Technical specification
- [Roadmap](docs/roadmap.md) - Development roadmap

## Architecture

Chart-Binder uses a multi-stage pipeline:

1. **Normalization**: Clean and normalize artist/title text
2. **Evidence Collection**: Gather data from multiple sources
3. **CRG Selection**: Choose canonical release group using chart data
4. **RR Selection**: Pick representative release within group
5. **Tag Writing**: Embed decision trace and IDs into audio files

See [docs/spec.md](docs/spec.md) for detailed architecture documentation.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with conventional commit messages
4. Run `make` to ensure lint and tests pass
5. Submit a pull request

### Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(norm): add radio edit extraction
fix(tags): handle missing year gracefully
docs: update CLI reference
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MusicBrainz](https://musicbrainz.org/) for their comprehensive music database
- [Discogs](https://www.discogs.com/) for release and label data
- [AcoustID](https://acoustid.org/) for audio fingerprinting
- [beets](https://beets.io/) for inspiration and plugin integration
