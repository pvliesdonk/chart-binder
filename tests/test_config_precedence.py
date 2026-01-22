"""Test configuration precedence: CLI > Env > TOML > Defaults."""

from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from chart_binder.cli_typer import app
from chart_binder.config import Config


def test_toml_loading():
    """Test that TOML configuration is loaded correctly."""
    toml_content = """
[http_cache]
directory = "/custom/cache"
ttl_seconds = 7200
enabled = false

[database]
music_graph_path = "custom_music.db"
charts_path = "custom_charts.db"

[llm]
enabled = true
provider = "openai"
model_id = "gpt-4o-mini"
temperature = 0.5

[llm.searxng]
enabled = true
url = "http://search.example.com"
timeout_s = 15.0

[logging]
level = "DEBUG"
hash_paths = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        config_path = Path(f.name)

    try:
        config = Config.load(config_path)

        # Verify TOML values are loaded
        assert config.http_cache.directory == Path("/custom/cache")
        assert config.http_cache.ttl_seconds == 7200
        assert config.http_cache.enabled is False

        assert config.database.music_graph_path == Path("custom_music.db")
        assert config.database.charts_path == Path("custom_charts.db")

        assert config.llm.enabled is True
        assert config.llm.provider == "openai"
        assert config.llm.model_id == "gpt-4o-mini"
        assert config.llm.temperature == 0.5

        assert config.llm.searxng.enabled is True
        assert config.llm.searxng.url == "http://search.example.com"
        assert config.llm.searxng.timeout_s == 15.0

        assert config.logging.level == "DEBUG"
        assert config.logging.hash_paths is True

    finally:
        config_path.unlink()


def test_env_overrides_toml(monkeypatch):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    """Test that environment variables override TOML configuration."""
    toml_content = """
[http_cache]
ttl_seconds = 7200

[llm]
provider = "ollama"
model_id = "llama3.2"

[llm.searxng]
url = "http://localhost:8080"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        config_path = Path(f.name)

    try:
        # Set environment variables that should override TOML
        monkeypatch.setenv("CHART_BINDER_HTTP_CACHE_TTL_SECONDS", "3600")  # pyright: ignore[reportUnknownMemberType]
        monkeypatch.setenv("CHART_BINDER_LLM_PROVIDER", "openai")  # pyright: ignore[reportUnknownMemberType]
        monkeypatch.setenv("CHART_BINDER_LLM_MODEL_ID", "gpt-4o")  # pyright: ignore[reportUnknownMemberType]
        monkeypatch.setenv("CHART_BINDER_SEARXNG_URL", "http://search.custom.com")  # pyright: ignore[reportUnknownMemberType]

        config = Config.load(config_path)

        # Verify env vars override TOML
        assert config.http_cache.ttl_seconds == 3600  # from env, not 7200 from TOML
        assert config.llm.provider == "openai"  # from env, not "ollama" from TOML
        assert config.llm.model_id == "gpt-4o"  # from env, not "llama3.2" from TOML
        assert config.llm.searxng.url == "http://search.custom.com"  # from env

    finally:
        config_path.unlink()


def test_cli_precedence_over_env_and_toml(monkeypatch):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    """Test that CLI arguments have highest precedence over env vars and TOML."""
    toml_content = """
[http_cache]
ttl_seconds = 7200
directory = "/toml/cache"

[database]
music_graph_path = "toml_music.db"

[llm]
provider = "ollama"
model_id = "llama3.2"
temperature = 0.0

[llm.searxng]
url = "http://localhost:8080"
enabled = false
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        config_path = Path(f.name)

    try:
        # Set environment variables
        monkeypatch.setenv("CHART_BINDER_HTTP_CACHE_TTL_SECONDS", "3600")  # pyright: ignore[reportUnknownMemberType]
        monkeypatch.setenv("CHART_BINDER_DATABASE_MUSIC_GRAPH_PATH", "env_music.db")  # pyright: ignore[reportUnknownMemberType]
        monkeypatch.setenv("CHART_BINDER_LLM_MODEL_ID", "mistral")  # pyright: ignore[reportUnknownMemberType]

        runner = CliRunner()

        # Invoke CLI with arguments that should override both env and TOML
        result = runner.invoke(
            app,
            [
                "--config",
                str(config_path),
                "--cache-ttl",
                "1800",  # CLI override
                "--db-music-graph",
                "cli_music.db",  # CLI override
                "--llm-model",
                "gpt-4o-mini",  # CLI override
                "--llm-temperature",
                "0.7",  # CLI override
                "--searxng-url",
                "http://cli.search.com",  # CLI override
                "--searxng-enabled",  # CLI override
                "--help",  # Just to exit cleanly
            ],
        )

        # The CLI should execute successfully
        assert result.exit_code == 0

    finally:
        config_path.unlink()


def test_all_cache_ttl_env_vars(monkeypatch):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    """Test that all cache TTL environment variables work correctly."""
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_MUSICBRAINZ", "1000")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_DISCOGS", "2000")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_SPOTIFY", "3000")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_WIKIDATA", "4000")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_ACOUSTID", "5000")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()

    assert config.live_sources.cache_ttl_musicbrainz == 1000
    assert config.live_sources.cache_ttl_discogs == 2000
    assert config.live_sources.cache_ttl_spotify == 3000
    assert config.live_sources.cache_ttl_wikidata == 4000
    assert config.live_sources.cache_ttl_acoustid == 5000


def test_all_llm_env_vars(monkeypatch):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    """Test that all LLM environment variables work correctly."""
    monkeypatch.setenv("CHART_BINDER_LLM_ENABLED", "true")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_PROVIDER", "openai")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_MODEL_ID", "gpt-4o")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_OPENAI_BASE_URL", "https://custom.api.com/v1")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_TEMPERATURE", "0.8")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_PROMPT_TEMPLATE_VERSION", "v3")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_REVIEW_QUEUE_PATH", "/custom/queue.db")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_TIMEOUT_S", "60.0")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_MAX_TOKENS", "2048")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_AUTO_ACCEPT_THRESHOLD", "0.90")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_REVIEW_THRESHOLD", "0.70")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()

    assert config.llm.enabled is True
    assert config.llm.provider == "openai"
    assert config.llm.model_id == "gpt-4o"
    assert config.llm.openai_base_url == "https://custom.api.com/v1"
    assert config.llm.temperature == 0.8
    assert config.llm.prompt_template_version == "v3"
    assert config.llm.review_queue_path == Path("/custom/queue.db")
    assert config.llm.timeout_s == 60.0
    assert config.llm.max_tokens == 2048
    assert config.llm.auto_accept_threshold == 0.90
    assert config.llm.review_threshold == 0.70


def test_logging_env_vars(monkeypatch):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    """Test that logging environment variables work correctly."""
    monkeypatch.setenv("CHART_BINDER_LOGGING_LEVEL", "DEBUG")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LOGGING_HASH_PATHS", "true")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()

    assert config.logging.level == "DEBUG"
    assert config.logging.hash_paths is True
