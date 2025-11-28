# Configuration Guide

Chart-Binder can be configured through a TOML configuration file and/or environment variables.

## Configuration File

Create a `config.toml` file in your working directory or specify it with `--config`:

```bash
canon --config /path/to/config.toml scan ...
```

### Complete Configuration Example

```toml
# Chart-Binder Configuration

[http_cache]
# Directory for HTTP cache storage
directory = ".cache/http"

# Cache entry time-to-live in seconds (default: 86400 = 24 hours)
ttl_seconds = 86400

# Enable/disable HTTP caching (default: true)
enabled = true

[database]
# Path to MusicGraph SQLite database
music_graph_path = "musicgraph.sqlite"

# Path to Charts SQLite database
charts_path = "charts.sqlite"

# Path to Decisions SQLite database
decisions_path = "decisions.sqlite"

[live_sources]
# API credentials (can also use environment variables)
acoustid_api_key = ""
discogs_token = ""
spotify_client_id = ""
spotify_client_secret = ""

# Rate limits
musicbrainz_rate_limit = 1.0   # requests per second
acoustid_rate_limit = 3.0      # requests per second
discogs_rate_limit = 25        # requests per minute

# Source-specific cache TTLs (seconds)
cache_ttl_musicbrainz = 3600   # 1 hour
cache_ttl_discogs = 86400      # 24 hours
cache_ttl_spotify = 7200       # 2 hours
cache_ttl_wikidata = 604800    # 7 days
cache_ttl_acoustid = 86400     # 24 hours

[llm]
# Enable LLM adjudication for ambiguous cases
enabled = false

# Provider: "ollama" (local) or "openai"
provider = "ollama"

# Model identifier
# Ollama: llama3.2, mistral, phi3, codellama, etc.
# OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
model_id = "llama3.2"

# API configuration
api_key_env = "OPENAI_API_KEY"  # Environment variable for API key
ollama_base_url = "http://localhost:11434"
openai_base_url = "https://api.openai.com/v1"

# Request settings
timeout_s = 30.0       # Request timeout in seconds
max_tokens = 1024      # Maximum response tokens
temperature = 0.0      # Generation temperature (0.0-2.0)

# Confidence thresholds
auto_accept_threshold = 0.85   # Auto-accept if confidence >= this
review_threshold = 0.60        # Send to review if confidence >= this

# Prompt versioning (for A/B testing)
prompt_template_version = "v1"

# Review queue database
review_queue_path = "review_queue.sqlite"
```

## Configuration Sections

### HTTP Cache

Controls caching of API responses.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `directory` | Path | `.cache/http` | Cache storage directory |
| `ttl_seconds` | int | 86400 | Default cache TTL |
| `enabled` | bool | true | Enable caching |

### Database

Paths to SQLite databases.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `music_graph_path` | Path | `musicgraph.sqlite` | Entity relationship database |
| `charts_path` | Path | `charts.sqlite` | Chart data database |
| `decisions_path` | Path | `decisions.sqlite` | Decision history database |

### Live Sources

External API configuration.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `acoustid_api_key` | string | (none) | AcoustID API key |
| `discogs_token` | string | (none) | Discogs personal access token |
| `spotify_client_id` | string | (none) | Spotify app client ID |
| `spotify_client_secret` | string | (none) | Spotify app client secret |
| `musicbrainz_rate_limit` | float | 1.0 | MB requests per second |
| `acoustid_rate_limit` | float | 3.0 | AcoustID requests per second |
| `discogs_rate_limit` | int | 25 | Discogs requests per minute |
| `cache_ttl_musicbrainz` | int | 3600 | MB cache TTL |
| `cache_ttl_discogs` | int | 86400 | Discogs cache TTL |
| `cache_ttl_spotify` | int | 7200 | Spotify cache TTL |
| `cache_ttl_wikidata` | int | 604800 | Wikidata cache TTL |
| `cache_ttl_acoustid` | int | 86400 | AcoustID cache TTL |

### LLM Configuration

Settings for AI-powered adjudication.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | false | Enable LLM adjudication |
| `provider` | string | `ollama` | Provider: `ollama` or `openai` |
| `model_id` | string | `llama3.2` | Model identifier |
| `api_key_env` | string | `OPENAI_API_KEY` | Env var for API key |
| `ollama_base_url` | string | `http://localhost:11434` | Ollama API URL |
| `openai_base_url` | string | `https://api.openai.com/v1` | OpenAI API URL |
| `timeout_s` | float | 30.0 | Request timeout |
| `max_tokens` | int | 1024 | Max response tokens |
| `temperature` | float | 0.0 | Generation temperature |
| `auto_accept_threshold` | float | 0.85 | Auto-accept confidence |
| `review_threshold` | float | 0.60 | Review queue threshold |
| `prompt_template_version` | string | `v1` | Prompt template version |
| `review_queue_path` | Path | `review_queue.sqlite` | Review queue database |

## Environment Variables

All configuration options can be overridden via environment variables. Environment variables take precedence over config file values.

### General

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_OFFLINE_MODE` | Set to `true` for offline mode |

### HTTP Cache

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_HTTP_CACHE_DIRECTORY` | Cache directory path |
| `CHART_BINDER_HTTP_CACHE_TTL_SECONDS` | Cache TTL |
| `CHART_BINDER_HTTP_CACHE_ENABLED` | Enable/disable cache |

### Database

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_DATABASE_MUSIC_GRAPH_PATH` | MusicGraph database path |
| `CHART_BINDER_DATABASE_CHARTS_PATH` | Charts database path |
| `CHART_BINDER_DATABASE_DECISIONS_PATH` | Decisions database path |

### API Credentials

| Variable | Description |
|----------|-------------|
| `ACOUSTID_API_KEY` | AcoustID API key |
| `DISCOGS_TOKEN` | Discogs personal access token |
| `SPOTIFY_CLIENT_ID` | Spotify client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify client secret |

### Rate Limits

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_LIVE_SOURCES_MUSICBRAINZ_RATE_LIMIT` | MB rate limit |
| `CHART_BINDER_LIVE_SOURCES_ACOUSTID_RATE_LIMIT` | AcoustID rate limit |
| `CHART_BINDER_LIVE_SOURCES_DISCOGS_RATE_LIMIT` | Discogs rate limit |

### LLM Settings

| Variable | Description |
|----------|-------------|
| `CHART_BINDER_LLM_ENABLED` | Enable LLM (`true`/`false`) |
| `CHART_BINDER_LLM_PROVIDER` | Provider (`ollama`/`openai`) |
| `CHART_BINDER_LLM_MODEL_ID` | Model identifier |
| `CHART_BINDER_LLM_API_KEY_ENV` | API key env var name |
| `CHART_BINDER_LLM_OLLAMA_BASE_URL` | Ollama base URL |
| `CHART_BINDER_LLM_TIMEOUT_S` | Request timeout |
| `CHART_BINDER_LLM_MAX_TOKENS` | Max tokens |
| `CHART_BINDER_LLM_AUTO_ACCEPT_THRESHOLD` | Auto-accept threshold |
| `CHART_BINDER_LLM_REVIEW_THRESHOLD` | Review threshold |

## Common Configuration Scenarios

### Minimal Configuration

For basic usage with defaults:

```toml
# config.toml - minimal
[http_cache]
directory = ".cache"
```

### Development Setup

```toml
# config.toml - development
[http_cache]
directory = ".cache/http"
ttl_seconds = 3600  # Shorter TTL for development

[database]
music_graph_path = "dev_musicgraph.sqlite"
charts_path = "dev_charts.sqlite"
decisions_path = "dev_decisions.sqlite"

[llm]
enabled = true
provider = "ollama"
model_id = "llama3.2"
```

### Production Setup with LLM

```toml
# config.toml - production with OpenAI
[http_cache]
directory = "/var/cache/chart-binder"
ttl_seconds = 86400

[database]
music_graph_path = "/var/lib/chart-binder/musicgraph.sqlite"
charts_path = "/var/lib/chart-binder/charts.sqlite"
decisions_path = "/var/lib/chart-binder/decisions.sqlite"

[live_sources]
musicbrainz_rate_limit = 0.5  # Conservative rate limiting

[llm]
enabled = true
provider = "openai"
model_id = "gpt-4o-mini"
auto_accept_threshold = 0.90
review_threshold = 0.70
timeout_s = 60.0
```

### Offline-First Setup

```toml
# config.toml - offline focused
[http_cache]
directory = ".cache/http"
ttl_seconds = 604800  # 7 days
enabled = true
```

Then run with:

```bash
# Build cache online
canon scan /path/to/music/

# Work offline later
canon --frozen decide /path/to/music/
```

## API Keys Setup

### AcoustID

1. Register at https://acoustid.org/
2. Create an application at https://acoustid.org/my-applications
3. Set `ACOUSTID_API_KEY` or add to config

### Discogs

1. Create account at https://www.discogs.com/
2. Go to Settings → Developer → Generate Personal Access Token
3. Set `DISCOGS_TOKEN` or add to config

### Spotify

1. Create app at https://developer.spotify.com/dashboard
2. Copy Client ID and Client Secret
3. Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`

### OpenAI (for LLM)

1. Get API key from https://platform.openai.com/api-keys
2. Set `OPENAI_API_KEY` environment variable

## Ollama Setup

For local LLM adjudication with Ollama:

1. Install Ollama: https://ollama.com/
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Start Ollama (usually runs automatically)
4. Configure:
   ```toml
   [llm]
   enabled = true
   provider = "ollama"
   model_id = "llama3.2"
   ollama_base_url = "http://localhost:11434"
   ```

### Recommended Models

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2` | 3B | Good balance of speed and quality |
| `llama3.2:7b` | 7B | Better quality, slower |
| `mistral` | 7B | Fast, good for simple cases |
| `codellama` | 7B | Good at structured output |

## Validation

Check your configuration:

```bash
# Show current LLM status
canon llm status

# Show cache status
canon cache status

# Test with a simple scan
canon scan --help
```

## Configuration Precedence

1. Command-line options (highest priority)
2. Environment variables
3. Configuration file
4. Default values (lowest priority)

Example:

```bash
# This uses offline mode even if config says otherwise
canon --offline scan /path/to/music/
```
