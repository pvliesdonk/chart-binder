__all__ = (
    "main",
    "Config",
    "HttpCache",
    "MusicGraphDB",
    "DecisionsDB",
    "DriftDetector",
    "DriftResult",
    "StaleDecisionSummary",
    "UnifiedFetcher",
    "MusicBrainzClient",
    "AcoustIDClient",
    "DiscogsClient",
    "SpotifyClient",
    "WikidataClient",
    # Tagging (Epic 7)
    "TagSet",
    "TagWriter",
    "ID3TagWriter",
    "VorbisTagWriter",
    "MP4TagWriter",
    "WriteReport",
    "CanonicalIDs",
    "CompactFields",
    "ReleaseType",
    "assemble_tags",
    "write_tags",
    "verify",
    "get_writer_for_file",
    # Charts ETL (Epic 8)
    "ChartsDB",
    "ChartsETL",
    "ChartEntry",
    "ChartLink",
    "CoverageReport",
    "EntryUnit",
    "LinkMethod",
    # Chart Scrapers (Epic 8)
    "ChartScraper",
    "Top40Scraper",
    "Top2000Scraper",
    "ZwaarsteScraper",
    # Charts Export (Epic 9)
    "ChartsBlob",
    "ChartScore",
    "ChartsExporter",
    # Resolver (Epic 5 + 11)
    "Resolver",
    "CanonicalDecision",
    "DecisionTrace",
    "CRGRationale",
    "RRRationale",
    "ConfigSnapshot",
    "explain",
    # LLM Adjudication (Epic 13)
    "LLMAdjudicator",
    "AdjudicationResult",
    "AdjudicationOutcome",
    "LLMProvider",
    "ProviderRegistry",
    "OllamaProvider",
    "OpenAIProvider",
    "SearchTool",
    "SearchResult",
    "ReviewQueue",
    "ReviewItem",
    "ReviewAction",
    # Rate Limiting & Batch Processing (Epic 14)
    "TokenBucket",
    "RateLimiterRegistry",
    "get_rate_limiter_registry",
    "rate_limit",
    "BatchConfig",
    "BatchResult",
    "BatchProcessor",
    "batch_iter",
    "process_batch",
    "collect_audio_files",
    # PII-safe Logging (Epic 14)
    "hash_path",
    "relativize_path",
    "safe_path",
    "redact_dict",
    "sanitize_message",
    "SafeLogFormatter",
    "configure_safe_logging",
)

from chart_binder.acoustid import AcoustIDClient
from chart_binder.batch import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
    batch_iter,
    collect_audio_files,
    process_batch,
)
from chart_binder.charts_db import (
    ChartEntry,
    ChartLink,
    ChartsDB,
    ChartsETL,
    CoverageReport,
    EntryUnit,
    LinkMethod,
)
from chart_binder.charts_export import ChartsBlob, ChartScore, ChartsExporter
from chart_binder.cli_typer import main
from chart_binder.config import Config
from chart_binder.decisions_db import DecisionsDB
from chart_binder.discogs import DiscogsClient
from chart_binder.drift import DriftDetector, DriftResult, StaleDecisionSummary
from chart_binder.fetcher import UnifiedFetcher
from chart_binder.http_cache import HttpCache
from chart_binder.llm import (
    AdjudicationOutcome,
    AdjudicationResult,
    LLMAdjudicator,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderRegistry,
    ReviewAction,
    ReviewItem,
    ReviewQueue,
    SearchResult,
    SearchTool,
)
from chart_binder.musicbrainz import MusicBrainzClient
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.rate_limiter import (
    RateLimiterRegistry,
    TokenBucket,
    get_rate_limiter_registry,
    rate_limit,
)
from chart_binder.resolver import (
    CanonicalDecision,
    ConfigSnapshot,
    CRGRationale,
    DecisionTrace,
    Resolver,
    RRRationale,
)
from chart_binder.safe_logging import (
    SafeLogFormatter,
    configure_safe_logging,
    hash_path,
    redact_dict,
    relativize_path,
    safe_path,
    sanitize_message,
)
from chart_binder.scrapers import (
    ChartScraper,
    Top40Scraper,
    Top2000Scraper,
    ZwaarsteScraper,
)
from chart_binder.spotify import SpotifyClient
from chart_binder.tagging import (
    CanonicalIDs,
    CompactFields,
    ID3TagWriter,
    MP4TagWriter,
    ReleaseType,
    TagSet,
    TagWriter,
    VorbisTagWriter,
    WriteReport,
    assemble_tags,
    get_writer_for_file,
    verify,
    write_tags,
)
from chart_binder.wikidata import WikidataClient


def explain(target: CanonicalDecision) -> str:
    """
    Generate human-readable trace for a decision.

    This is the public API function for explainability (Epic 11).

    Args:
        target: CanonicalDecision to explain

    Returns:
        Human-readable explanation string
    """
    return target.decision_trace.to_human_readable()
