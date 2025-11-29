from __future__ import annotations

"""
Chart scrapers for ETL pipeline.

Provides robust scraping infrastructure for chart data from various sources.
"""

__all__ = [
    "ChartScraper",
    "ScrapeResult",
    "EXPECTED_ENTRY_COUNTS",
    "Top40Scraper",
    "Top40JaarScraper",
    "Top2000Scraper",
    "ZwaarsteScraper",
    "SCRAPER_REGISTRY",
]

from chart_binder.scrapers.base import EXPECTED_ENTRY_COUNTS, ChartScraper, ScrapeResult
from chart_binder.scrapers.top40 import Top40Scraper
from chart_binder.scrapers.top40_jaar import Top40JaarScraper
from chart_binder.scrapers.top2000 import Top2000Scraper
from chart_binder.scrapers.zwaarste import ZwaarsteScraper

# Registry mapping CLI chart types to scraper classes and their DB IDs
SCRAPER_REGISTRY: dict[str, tuple[type[ChartScraper], str]] = {
    "t40": (Top40Scraper, "nl_top40"),
    "t40jaar": (Top40JaarScraper, "nl_top40_jaar"),
    "top2000": (Top2000Scraper, "nl_top2000"),
    "zwaarste": (ZwaarsteScraper, "nl_538_zwaarste"),
}
