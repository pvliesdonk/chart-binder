"""Pytest configuration and shared fixtures for chart-binder tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# =============================================================================
# Fixture Paths
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CASSETTES_DIR = FIXTURES_DIR / "cassettes"


def load_fixture(scraper_type: str, fixture_name: str) -> str:
    """Load a fixture file as text."""
    fixture_path = CASSETTES_DIR / scraper_type / fixture_name
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    return fixture_path.read_text(encoding="utf-8")


# =============================================================================
# HTTP Cache Fixtures
# =============================================================================


@pytest.fixture
def tmp_cache(tmp_path):
    """Provide a temporary HttpCache for tests."""
    from chart_binder.http_cache import HttpCache

    return HttpCache(tmp_path / "cache")


# =============================================================================
# Scraper Fixtures
# =============================================================================


@pytest.fixture
def top40_scraper(tmp_cache):
    """Provide a Top40Scraper instance with temporary cache."""
    from chart_binder.scrapers import Top40Scraper

    return Top40Scraper(tmp_cache)


@pytest.fixture
def top40jaar_scraper(tmp_cache):
    """Provide a Top40JaarScraper instance with temporary cache."""
    from chart_binder.scrapers import Top40JaarScraper

    return Top40JaarScraper(tmp_cache)


@pytest.fixture
def top2000_scraper(tmp_cache):
    """Provide a Top2000Scraper instance with temporary cache."""
    from chart_binder.scrapers import Top2000Scraper

    return Top2000Scraper(tmp_cache)


@pytest.fixture
def zwaarste_scraper(tmp_cache):
    """Provide a ZwaarsteScraper instance with temporary cache."""
    from chart_binder.scrapers import ZwaarsteScraper

    return ZwaarsteScraper(tmp_cache)


# =============================================================================
# Fixture Loading Helpers
# =============================================================================


@pytest.fixture
def top40_fixture():
    """Load Top40 HTML fixture."""

    def _load(fixture_name: str) -> str:
        return load_fixture("top40", fixture_name)

    return _load


@pytest.fixture
def top40jaar_fixture():
    """Load Top40Jaar HTML fixture."""

    def _load(fixture_name: str) -> str:
        return load_fixture("top40_jaar", fixture_name)

    return _load


@pytest.fixture
def top2000_fixture():
    """Load Top2000 JSON fixture."""

    def _load(fixture_name: str) -> str:
        return load_fixture("top2000", fixture_name)

    return _load


@pytest.fixture
def zwaarste_fixture():
    """Load Zwaarste HTML fixture."""

    def _load(fixture_name: str) -> str:
        return load_fixture("zwaarste", fixture_name)

    return _load
