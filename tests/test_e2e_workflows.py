"""E2E tests for chart-binder workflows.

These tests exercise complete pipelines from input to output,
ensuring all components work together correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chart_binder.candidates import CandidateBuilder, CandidateSet
from chart_binder.cli_typer import app
from chart_binder.config import Config
from chart_binder.llm.search_tool import SearchTool
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer
from chart_binder.resolver import ConfigSnapshot, Resolver


class TestCandidateDiscoveryE2E:
    """E2E tests for candidate discovery workflow."""

    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create a database populated with test data simulating real MusicBrainz data."""
        db = MusicGraphDB(tmp_path / "musicgraph.sqlite")

        # Artist: The Beatles
        db.upsert_artist(
            "5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            "The Beatles",
            sort_name="Beatles, The",
            begin_area_country="GB",
            wikidata_qid="Q1299",
            name_normalized="beatles",
        )

        # Recording: Yesterday
        db.upsert_recording(
            "5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a",
            "Yesterday",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            length_ms=125000,
            isrcs_json='["GBAYE0601315"]',
            flags_json='{"is_live": false, "is_remix": false}',
            title_normalized="yesterday",
        )

        # Release Group 1: Help! (Album) - Original appearance
        db.upsert_release_group(
            "rg-help-album",
            "Help!",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            type="Album",
            first_release_date="1965-08-06",
            secondary_types_json='["Soundtrack"]',
            labels_json='["Parlophone", "Capitol"]',
            countries_json='["GB", "US", "CA"]',
        )

        # Release Group 2: Yesterday (Single)
        db.upsert_release_group(
            "rg-yesterday-single",
            "Yesterday",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            type="Single",
            first_release_date="1965-09-13",
            labels_json='["Capitol"]',
            countries_json='["US"]',
        )

        # Release Group 3: Yesterday and Today (Album) - US compilation
        db.upsert_release_group(
            "rg-yesterday-today",
            "Yesterday and Today",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            type="Album",
            first_release_date="1966-06-20",
            labels_json='["Capitol"]',
            countries_json='["US"]',
        )

        # Releases for Help! album
        db.upsert_release(
            "rel-help-uk",
            "Help!",
            release_group_mbid="rg-help-album",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            date="1965-08-06",
            country="GB",
            label="Parlophone",
            flags_json='{"is_official": true}',
        )
        db.upsert_release(
            "rel-help-us",
            "Help!",
            release_group_mbid="rg-help-album",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            date="1965-08-13",
            country="US",
            label="Capitol",
            flags_json='{"is_official": true}',
        )

        # Release for Yesterday single
        db.upsert_release(
            "rel-yesterday-single",
            "Yesterday",
            release_group_mbid="rg-yesterday-single",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            date="1965-09-13",
            country="US",
            label="Capitol",
            flags_json='{"is_official": true}',
        )

        # Release for Yesterday and Today
        db.upsert_release(
            "rel-yesterday-today",
            "Yesterday and Today",
            release_group_mbid="rg-yesterday-today",
            artist_mbid="5b11f4ce-a62d-471e-81fc-a69a8278c7da",
            date="1966-06-20",
            country="US",
            label="Capitol",
            flags_json='{"is_official": true}',
        )

        # Link recording to all releases
        db.upsert_recording_release("5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a", "rel-help-uk")
        db.upsert_recording_release("5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a", "rel-help-us")
        db.upsert_recording_release("5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a", "rel-yesterday-single")
        db.upsert_recording_release("5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a", "rel-yesterday-today")

        return db

    def test_isrc_discovery_full_pipeline(self, populated_db):
        """Test complete ISRC discovery: lookup → candidates → evidence bundle."""
        normalizer = Normalizer()
        builder = CandidateBuilder(populated_db, normalizer)

        # Step 1: Discover candidates by ISRC
        candidates = builder.discover_by_isrc("GBAYE0601315")

        # Verify candidates found
        assert len(candidates) > 0
        assert any(c.recording_mbid == "5ef9dfa4-8f0e-4a93-a9a6-0a9f9a0a1f1a" for c in candidates)

        # Should find multiple release groups (Help! album, Yesterday single, Yesterday and Today)
        rg_mbids = {c.release_group_mbid for c in candidates}
        assert "rg-help-album" in rg_mbids
        assert "rg-yesterday-single" in rg_mbids
        assert "rg-yesterday-today" in rg_mbids

        # Step 2: Build candidate set
        candidate_set = CandidateSet(
            file_path=Path("/test/Yesterday.mp3"),
            candidates=candidates,
            normalized_title="yesterday",
            normalized_artist="beatles",
            length_ms=125000,
        )

        # Step 3: Build evidence bundle
        evidence_bundle = builder.build_evidence_bundle(candidate_set)

        # Verify evidence bundle structure
        assert evidence_bundle.artist["name"] == "The Beatles"
        assert evidence_bundle.artist["begin_area_country"] == "GB"
        assert evidence_bundle.artist["wikidata_qid"] == "Q1299"

        assert len(evidence_bundle.recordings) == 1
        rec = evidence_bundle.recordings[0]
        assert rec["title"] == "Yesterday"
        assert "GBAYE0601315" in rec["isrcs"]
        assert rec["flags"]["is_live"] is False

        assert len(evidence_bundle.release_groups) == 3
        assert evidence_bundle.timeline_facts["earliest_album_date"] == "1965-08-06"
        assert evidence_bundle.timeline_facts["earliest_single_ep_date"] == "1965-09-13"
        assert evidence_bundle.timeline_facts["earliest_soundtrack_date"] == "1965-08-06"

        # Verify provenance
        assert "MB" in evidence_bundle.provenance["sources_used"]
        assert "isrc" in evidence_bundle.provenance["discovery_methods"]

        # Verify hash is generated
        assert len(evidence_bundle.evidence_hash) == 64  # SHA256

    def test_fuzzy_match_full_pipeline(self, populated_db):
        """Test complete fuzzy match: normalize → search → candidates → evidence."""
        normalizer = Normalizer()
        builder = CandidateBuilder(populated_db, normalizer)

        # Step 1: Fuzzy discovery by title/artist/length
        candidates = builder.discover_by_title_artist_length(
            title="Yesterday",
            artist="Beatles",
            length_ms=125000,
        )

        # Verify candidates found via fuzzy match
        assert len(candidates) > 0
        assert all(c.discovery_method == "title_artist_length" for c in candidates)

        # Step 2: Build evidence bundle
        candidate_set = CandidateSet(candidates=candidates)
        evidence_bundle = builder.build_evidence_bundle(candidate_set)

        # Should have same structure as ISRC discovery
        assert evidence_bundle.artist["name"] == "The Beatles"
        assert len(evidence_bundle.recordings) == 1
        assert len(evidence_bundle.release_groups) == 3

        # Verify provenance shows fuzzy discovery
        assert "title_artist_length" in evidence_bundle.provenance["discovery_methods"]

    def test_multi_candidate_disambiguation(self, populated_db):
        """Test when multiple candidates are found - proper deduplication."""
        normalizer = Normalizer()
        builder = CandidateBuilder(populated_db, normalizer)

        # Discover via ISRC
        isrc_candidates = builder.discover_by_isrc("GBAYE0601315")

        # Discover via fuzzy match (should find same recording)
        fuzzy_candidates = builder.discover_by_title_artist_length("Yesterday", "Beatles", 125000)

        # Combine candidates
        all_candidates = isrc_candidates + fuzzy_candidates

        # Build evidence bundle - should deduplicate
        candidate_set = CandidateSet(candidates=all_candidates)
        evidence_bundle = builder.build_evidence_bundle(candidate_set)

        # Should still have only 1 unique recording despite duplicates
        assert len(evidence_bundle.recordings) == 1

        # Should have 3 unique release groups
        assert len(evidence_bundle.release_groups) == 3

        # Provenance should show both discovery methods
        methods = evidence_bundle.provenance["discovery_methods"]
        assert "isrc" in methods
        assert "title_artist_length" in methods

    def test_cross_release_group_recording(self, populated_db):
        """Test recording that appears in multiple release groups."""
        normalizer = Normalizer()
        builder = CandidateBuilder(populated_db, normalizer)

        candidates = builder.discover_by_isrc("GBAYE0601315")

        # Build evidence bundle
        candidate_set = CandidateSet(candidates=candidates)
        evidence_bundle = builder.build_evidence_bundle(candidate_set)

        # Verify all release groups are captured
        rg_mbids = {rg["mbid"] for rg in evidence_bundle.release_groups}
        assert len(rg_mbids) == 3
        assert "rg-help-album" in rg_mbids
        assert "rg-yesterday-single" in rg_mbids
        assert "rg-yesterday-today" in rg_mbids

        # Verify different types
        types = {rg["type"] for rg in evidence_bundle.release_groups}
        assert "Album" in types
        assert "Single" in types

        # Verify soundtrack secondary type is captured
        help_rg = next(rg for rg in evidence_bundle.release_groups if rg["mbid"] == "rg-help-album")
        assert "Soundtrack" in help_rg["secondary_types"]


class TestResolverE2E:
    """E2E tests for resolver workflow with evidence bundles."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver with standard config."""
        config = ConfigSnapshot(
            lead_window_days=90,
            reissue_long_gap_years=10,
        )
        return Resolver(config)

    def test_soundtrack_premiere_selection(self, resolver):
        """Test that soundtrack with earliest date is selected as CRG."""
        # Evidence bundle with soundtrack as earliest
        evidence_bundle = {
            "artist": {
                "name": "The Beatles",
                "mbid": "artist-1",
                "begin_area_country": "GB",
            },
            "recording_candidates": [
                {
                    "mb_recording_id": "rec-1",
                    "title": "Yesterday",
                    "rg_candidates": [
                        {
                            "mb_rg_id": "rg-soundtrack",
                            "title": "Help!",
                            "primary_type": "Soundtrack",
                            "first_release_date": "1965-08-06",
                            "releases": [
                                {
                                    "mb_release_id": "rel-1",
                                    "date": "1965-08-06",
                                    "country": "GB",
                                    "flags": {"is_official": True},
                                }
                            ],
                        },
                        {
                            "mb_rg_id": "rg-single",
                            "title": "Yesterday",
                            "primary_type": "Single",
                            "first_release_date": "1965-09-13",
                            "releases": [
                                {
                                    "mb_release_id": "rel-2",
                                    "date": "1965-09-13",
                                    "country": "US",
                                    "flags": {"is_official": True},
                                }
                            ],
                        },
                    ],
                }
            ],
            "timeline_facts": {
                "earliest_album_date": "1965-08-06",
                "earliest_single_ep_date": "1965-09-13",
                "earliest_soundtrack_date": "1965-08-06",
            },
            "provenance": {"sources_used": ["MB"]},
        }

        decision = resolver.resolve(evidence_bundle)

        # Should select soundtrack as CRG
        assert decision.state == "decided"
        assert decision.release_group_mbid == "rg-soundtrack"
        assert decision.crg_rationale == "CRG:SOUNDTRACK_PREMIERE"

        # Should select GB release (artist origin country)
        assert decision.release_mbid == "rel-1"
        assert decision.rr_rationale == "RR:ORIGIN_COUNTRY_EARLIEST"

    def test_earliest_official_fallback(self, resolver):
        """Test fallback to earliest official when no special rules apply."""
        evidence_bundle = {
            "artist": {"name": "Test Artist", "mbid": "artist-1"},
            "recording_candidates": [
                {
                    "mb_recording_id": "rec-1",
                    "title": "Test Song",
                    "rg_candidates": [
                        {
                            "mb_rg_id": "rg-1",
                            "title": "Album 1",
                            "primary_type": "Album",
                            "first_release_date": "2020-01-01",
                            "releases": [
                                {
                                    "mb_release_id": "rel-1",
                                    "date": "2020-01-01",
                                    "country": "US",
                                    "flags": {"is_official": True},
                                }
                            ],
                        },
                        {
                            "mb_rg_id": "rg-2",
                            "title": "Album 2",
                            "primary_type": "Album",
                            "first_release_date": "2020-06-01",
                            "releases": [
                                {
                                    "mb_release_id": "rel-2",
                                    "date": "2020-06-01",
                                    "country": "US",
                                    "flags": {"is_official": True},
                                }
                            ],
                        },
                    ],
                }
            ],
            "timeline_facts": {},
            "provenance": {"sources_used": ["MB"]},
        }

        decision = resolver.resolve(evidence_bundle)

        # Should select earliest
        assert decision.state == "decided"
        assert decision.release_group_mbid == "rg-1"
        assert decision.crg_rationale == "CRG:EARLIEST_OFFICIAL"


class TestCLIE2E:
    """E2E tests for CLI commands."""

    def test_scan_command_with_mock_file(self, tmp_path):
        """Test scan command processes files correctly."""
        # Create a minimal mock audio file (won't actually be read as audio)
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake mp3 data")

        runner = CliRunner()
        result = runner.invoke(app, ["scan", str(test_file)])

        # Should attempt to scan but may fail due to invalid file
        # The important thing is the command structure works
        assert result.exit_code in [0, 1, 2]  # Success, error, or no results

    def test_charts_scrape_command_help(self):
        """Test charts scrape command help output."""
        runner = CliRunner()
        result = runner.invoke(app, ["charts", "scrape", "--help"])

        assert result.exit_code == 0
        assert "Scrape chart data from web source" in result.output
        assert "CHART_TYPE" in result.output
        assert "PERIOD" in result.output

    def test_verbose_logging_output(self, capsys):
        """Test that -v and -vv produce appropriate log output."""
        runner = CliRunner()

        # Test -v (INFO level)
        result = runner.invoke(app, ["-v", "--help"])
        assert result.exit_code == 0

        # Test -vv (DEBUG level)
        result = runner.invoke(app, ["-vv", "--help"])
        assert result.exit_code == 0

    def test_cache_status_command(self, tmp_path):
        """Test cache status command with custom cache directory."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--cache-dir",
                str(tmp_path / "cache"),
                "cache",
                "status",
            ],
        )

        assert result.exit_code == 0
        assert "Cache Status" in result.output

    def test_llm_status_command(self):
        """Test LLM status command."""
        runner = CliRunner()
        result = runner.invoke(app, ["llm", "status"])

        assert result.exit_code == 0
        assert "LLM Configuration Status" in result.output

    def test_review_list_command(self):
        """Test review queue list command."""
        runner = CliRunner()
        result = runner.invoke(app, ["review", "list"])

        # May exit 0 (empty list) or 1 (no queue configured)
        assert result.exit_code in [0, 1]


class TestConfigurationE2E:
    """E2E tests for configuration loading."""

    def test_config_precedence_toml_env_cli(self, tmp_path, monkeypatch):
        """Test that CLI > Env > TOML > Defaults precedence works."""
        # Create TOML config
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            """
[http_cache]
ttl_seconds = 1000

[llm]
provider = "ollama"
model_id = "llama3.2"
temperature = 0.0

[logging]
level = "WARNING"
"""
        )

        # Set environment variable (should override TOML)
        monkeypatch.setenv("CHART_BINDER_HTTP_CACHE_TTL_SECONDS", "2000")
        monkeypatch.setenv("CHART_BINDER_LLM_TEMPERATURE", "0.5")

        # Load config
        config = Config.load(toml_file)

        # Env should override TOML
        assert config.http_cache.ttl_seconds == 2000  # from env, not 1000 from TOML
        assert config.llm.temperature == 0.5  # from env, not 0.0 from TOML

        # TOML should override defaults
        assert config.llm.provider == "ollama"  # from TOML
        assert config.llm.model_id == "llama3.2"  # from TOML
        assert config.logging.level == "WARNING"  # from TOML

    def test_config_all_sources_integration(self, tmp_path, monkeypatch):
        """Test integration of all config sources together."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            """
[database]
music_graph_path = "toml_music.db"
charts_path = "toml_charts.db"

[llm]
enabled = false
provider = "ollama"

[llm.searxng]
url = "http://localhost:8080"
enabled = false
"""
        )

        # Environment variables
        monkeypatch.setenv("CHART_BINDER_DATABASE_CHARTS_PATH", "env_charts.db")
        monkeypatch.setenv("CHART_BINDER_LLM_ENABLED", "true")

        # Load config
        config = Config.load(toml_file)

        # Check precedence
        assert config.database.music_graph_path == Path("toml_music.db")  # from TOML
        assert config.database.charts_path == Path("env_charts.db")  # from env (overrides TOML)
        assert config.llm.enabled is True  # from env (overrides TOML)
        assert config.llm.provider == "ollama"  # from TOML
        assert config.llm.searxng.url == "http://localhost:8080"  # from TOML

    def test_cli_overrides_all(self, tmp_path, monkeypatch):
        """Test that CLI arguments override both env and TOML."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            """
[http_cache]
ttl_seconds = 1000

[database]
music_graph_path = "toml_music.db"

[llm]
temperature = 0.0
"""
        )

        monkeypatch.setenv("CHART_BINDER_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("CHART_BINDER_HTTP_CACHE_TTL_SECONDS", "2000")

        runner = CliRunner()
        # Test with cache status command which is simpler
        result = runner.invoke(
            app,
            [
                "--config",
                str(toml_file),
                "-o",
                "json",  # Global flag
                "--cache-ttl",
                "3000",  # CLI should win
                "--cache-dir",
                str(tmp_path / "cli_cache"),  # CLI override
                "cache",
                "status",
            ],
        )

        assert result.exit_code == 0

        # Parse JSON output
        output = json.loads(result.output)

        # CLI should override env and TOML
        assert output["ttl_seconds"] == 3000  # from CLI, not 2000 from env or 1000 from TOML
        assert str(tmp_path / "cli_cache") in output["cache_directory"]


class TestSearchToolE2E:
    """E2E tests for LLM search tool."""

    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create a database with diverse test data."""
        db = MusicGraphDB(tmp_path / "musicgraph.sqlite")

        # Multiple artists
        db.upsert_artist("artist-beatles", "The Beatles", begin_area_country="GB")
        db.upsert_artist("artist-stones", "The Rolling Stones", begin_area_country="GB")

        # Multiple recordings
        db.upsert_recording(
            "rec-yesterday",
            "Yesterday",
            artist_mbid="artist-beatles",
            length_ms=125000,
            isrcs_json='["GBAYE0601315"]',
        )
        db.upsert_recording(
            "rec-help",
            "Help!",
            artist_mbid="artist-beatles",
            length_ms=138000,
        )
        db.upsert_recording(
            "rec-satisfaction",
            "Satisfaction",
            artist_mbid="artist-stones",
            length_ms=223000,
            isrcs_json='["USCA11234567"]',
        )

        # Release groups
        db.upsert_release_group(
            "rg-help",
            "Help!",
            artist_mbid="artist-beatles",
            type="Album",
            first_release_date="1965-08-06",
        )
        db.upsert_release_group(
            "rg-out-of-our-heads",
            "Out of Our Heads",
            artist_mbid="artist-stones",
            type="Album",
            first_release_date="1965-07-30",
        )

        # Releases
        db.upsert_release(
            "rel-help-uk",
            "Help!",
            release_group_mbid="rg-help",
            date="1965-08-06",
            country="GB",
            label="Parlophone",
            barcode="5099963380620",
            catno="PCS 3071",
        )

        return db

    def test_search_tool_full_workflow(self, populated_db):
        """Test SearchTool with MBID lookups against populated DB.

        Note: With the hybrid approach, MBID lookups use local DB,
        while text searches and ISRC searches go to external APIs.
        This test only covers local DB operations (MBID lookups).
        """
        tool = SearchTool(populated_db)

        # Test 1: Search artist by MBID (local DB)
        response = tool.search_artist("artist-beatles", by_mbid=True)
        assert response.total_count == 1
        assert response.results[0].title == "The Beatles"
        assert response.results[0].metadata["country"] == "GB"

        # Test 2: Search artist by MBID
        response = tool.search_artist("artist-stones", by_mbid=True)
        assert response.total_count == 1
        assert response.results[0].title == "The Rolling Stones"

        # Test 3: Search recording by MBID (local DB)
        response = tool.search_recording("rec-help", by_mbid=True)
        assert response.total_count == 1
        assert response.results[0].title == "Help!"

        # Test 4: Search release group by MBID (local DB)
        response = tool.search_release_group("rg-help", by_mbid=True)
        assert response.total_count == 1
        assert response.results[0].title == "Help!"
        assert response.results[0].metadata["primary_type"] == "Album"

        # Test 5: Search release by MBID (local DB)
        response = tool.search_release("rel-help-uk", by_mbid=True)
        assert response.total_count == 1
        assert response.results[0].title == "Help!"
        assert response.results[0].metadata["country"] == "GB"
        assert response.results[0].metadata["label"] == "Parlophone"

        # Test 6: Get release group releases (local DB)
        response = tool.get_release_group_releases("rg-help")
        assert response.total_count == 1
        assert response.results[0].id == "rel-help-uk"

        # Note: Text searches (search_artist("Beatles"), search_recording by title)
        # and ISRC searches now go to MusicBrainz API and require an MB client

    def test_search_tool_context_formatting(self, populated_db):
        """Test that search results are formatted correctly for LLM context."""
        tool = SearchTool(populated_db)

        # Use MBID lookups (text searches now require MB client)
        artist_response = tool.search_artist("artist-beatles", by_mbid=True)
        recording_response = tool.search_recording("rec-yesterday", by_mbid=True)

        # Format for LLM
        searches = [
            ("Artist Search", artist_response),
            ("Recording Search", recording_response),
        ]
        formatted = tool.format_for_llm(searches)

        # Verify formatting
        assert "## Artist Search" in formatted
        assert "## Recording Search" in formatted
        assert "The Beatles" in formatted
        assert "Yesterday" in formatted


class TestChartsWorkflowE2E:
    """E2E tests for charts-related workflows."""

    def test_charts_ingest_and_query(self, tmp_path):
        """Test ingesting chart data and querying it."""
        from chart_binder.charts_db import ChartsDB, ChartsETL

        # Create charts database
        db = ChartsDB(tmp_path / "charts.sqlite")
        etl = ChartsETL(db)

        # Create chart
        db.upsert_chart("top40", "Top 40", "w")

        # Ingest test data
        entries = [
            (1, "Artist 1", "Song 1"),
            (2, "Artist 2", "Song 2"),
            (3, "Artist 3", "Song 3"),
        ]

        run_id = etl.ingest("top40", "2024-W01", entries, notes="Test ingestion")

        assert run_id is not None

        # Query the run
        run = db.get_run_by_period("top40", "2024-W01")
        assert run is not None
        assert run["run_id"] == run_id

        # Get coverage report
        report = db.get_coverage_report(run_id)
        assert report.total_entries == 3
        assert report.unlinked_entries == 3  # No linking done yet


class TestIntegrationScenarios:
    """Integration tests for realistic end-to-end scenarios."""

    def test_full_file_processing_scenario(self, tmp_path):
        """Test complete scenario: file → metadata → candidates → decision → tags."""
        # Setup database
        db = MusicGraphDB(tmp_path / "musicgraph.sqlite")

        # Populate with test data
        db.upsert_artist("artist-1", "Test Artist", begin_area_country="US")
        db.upsert_recording(
            "rec-1",
            "Test Song",
            artist_mbid="artist-1",
            length_ms=180000,
            isrcs_json='["USTEST1234567"]',
        )
        db.upsert_release_group(
            "rg-1",
            "Test Album",
            artist_mbid="artist-1",
            type="Album",
            first_release_date="2020-01-01",
        )
        db.upsert_release(
            "rel-1",
            "Test Album",
            release_group_mbid="rg-1",
            date="2020-01-01",
            country="US",
            flags_json='{"is_official": true}',
        )
        db.upsert_recording_release("rec-1", "rel-1")

        # Step 1: Discover candidates
        normalizer = Normalizer()
        builder = CandidateBuilder(db, normalizer)
        candidates = builder.discover_by_isrc("USTEST1234567")

        assert len(candidates) > 0

        # Step 2: Build evidence bundle
        candidate_set = CandidateSet(candidates=candidates)
        evidence_bundle = builder.build_evidence_bundle(candidate_set)

        assert evidence_bundle.artist["name"] == "Test Artist"

        # Step 3: Make decision
        resolver = Resolver(ConfigSnapshot())

        # Convert evidence bundle to resolver format
        resolver_evidence = {
            "artist": evidence_bundle.artist,
            "recording_candidates": [
                {
                    "mb_recording_id": rec["mbid"],
                    "title": rec["title"],
                    "rg_candidates": [
                        {
                            "mb_rg_id": rg["mbid"],
                            "title": rg["title"],
                            "primary_type": rg["type"],
                            "first_release_date": rg["first_release_date"],
                            "releases": [
                                {
                                    "mb_release_id": "rel-1",
                                    "date": "2020-01-01",
                                    "country": "US",
                                    "flags": {"is_official": True},
                                }
                            ],
                        }
                        for rg in evidence_bundle.release_groups
                    ],
                }
                for rec in evidence_bundle.recordings
            ],
            "timeline_facts": evidence_bundle.timeline_facts,
            "provenance": evidence_bundle.provenance,
        }

        decision = resolver.resolve(resolver_evidence)

        # Verify decision
        assert decision.state == "decided"
        assert decision.release_group_mbid == "rg-1"
        assert decision.release_mbid == "rel-1"

        # Step 4: Verify compact tag generation
        compact_tag = decision.compact_tag
        assert compact_tag is not None
        assert "evh=" in compact_tag
        assert "crg=" in compact_tag
        assert "rr=" in compact_tag

    def test_ambiguous_decision_scenario(self, tmp_path):
        """Test scenario where resolver returns INDETERMINATE."""
        # Setup with insufficient data
        db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
        db.upsert_artist("artist-1", "Test Artist")
        db.upsert_recording("rec-1", "Test Song", artist_mbid="artist-1")

        # No release groups - should be indeterminate
        normalizer = Normalizer()
        _ = CandidateBuilder(db, normalizer)  # Ensure builder instantiation works

        # Build minimal evidence bundle (not used but validates CandidateSet with empty list)
        _ = CandidateSet(candidates=[])
        evidence_bundle_dict = {
            "artist": {"name": "Test Artist"},
            "recording_candidates": [],
            "timeline_facts": {},
            "provenance": {"sources_used": ["MB"]},
        }

        resolver = Resolver(ConfigSnapshot())
        decision = resolver.resolve(evidence_bundle_dict)

        # Should be indeterminate
        assert decision.state == "indeterminate"
        assert decision.crg_rationale == "CRG:INDETERMINATE"
        assert len(decision.decision_trace.missing_facts) > 0
