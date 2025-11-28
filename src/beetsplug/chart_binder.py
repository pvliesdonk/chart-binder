"""Chart-Binder Beets Plugin Adapter (Epic 12).

Thin adapter over chart-binder library for integration with Beets import flow.
Provides modes for advisory, authoritative, and augment-only canonicalization.

See: docs/appendix/beets_plugin_protocol_spec_v1.md
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Handle optional beets dependency
_beets_available = False
_beets_print: Callable[..., None] | None = None
_BeetsPlugin: type | None = None
_Subcommand: type | None = None
_decargs: Callable[..., Any] | None = None

try:
    from beets.plugins import BeetsPlugin as _BeetsPluginImport  # pyright: ignore
    from beets.ui import Subcommand as _SubcommandImport  # pyright: ignore
    from beets.ui import decargs as _decargsImport  # pyright: ignore
    from beets.ui import print_ as _beets_print_import  # pyright: ignore

    _beets_available = True
    _beets_print = _beets_print_import
    _BeetsPlugin = _BeetsPluginImport
    _Subcommand = _SubcommandImport
    _decargs = _decargsImport
except ImportError:
    pass

from chart_binder.charts_db import ChartsDB
from chart_binder.charts_export import ChartsExporter
from chart_binder.decisions_db import DecisionsDB
from chart_binder.resolver import (
    ConfigSnapshot,
    DecisionState,
    Resolver,
)
from chart_binder.tagging import (
    CanonicalIDs,
    CompactFields,
    TagSet,
    verify,
    write_tags,
)

log = logging.getLogger("beets.chart_binder")


def _print_msg(msg: str) -> None:
    """Print message using beets print_ if available, otherwise use standard print."""
    if _beets_print is not None:
        _beets_print(msg)
    else:
        print(msg)


class CanonMode(StrEnum):
    """Canonicalization modes for the plugin."""

    ADVISORY = "advisory"
    AUTHORITATIVE = "authoritative"
    AUGMENT = "augment"


@dataclass
class DecisionSummary:
    """Summary of a canonicalization decision for UI display."""

    file_path: Path
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    release_group_mbid: str | None = None
    release_mbid: str | None = None
    crg_rationale: str | None = None
    rr_rationale: str | None = None
    state: DecisionState = DecisionState.INDETERMINATE
    compact_trace: str | None = None
    charts_blob: str | None = None

    # Comparison fields for side-by-side display
    beets_album: str | None = None
    beets_date: str | None = None
    beets_label: str | None = None
    delta_days: int | None = None


@dataclass
class ImportSummary:
    """Summary of an import operation."""

    total_items: int = 0
    canonized: int = 0
    augmented: int = 0
    skipped: int = 0
    indeterminate: int = 0
    errors: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    def print_summary(self) -> None:
        """Print summary to console."""
        _print_msg("Chart-Binder Import Summary:")
        _print_msg(f"  Total items: {self.total_items}")
        _print_msg(f"  Canonized: {self.canonized}")
        _print_msg(f"  Augmented: {self.augmented}")
        _print_msg(f"  Skipped: {self.skipped}")
        _print_msg(f"  Indeterminate: {self.indeterminate}")
        if self.errors:
            _print_msg(f"  Errors: {len(self.errors)}")
            for error in self.errors[:10]:
                _print_msg(f"    - {error}")
        _print_msg(f"  Cache: {self.cache_hits} hits, {self.cache_misses} misses")


class ChartBinderPluginBase:
    """
    Base class for Chart-Binder Beets Plugin.

    This class contains all the logic and can be used for testing
    without requiring beets to be installed.
    """

    def __init__(self) -> None:
        self._resolver: Resolver | None = None
        self._charts_db: ChartsDB | None = None
        self._decisions_db: DecisionsDB | None = None
        self._charts_exporter: ChartsExporter | None = None
        self._import_summary: ImportSummary | None = None
        self._config: dict[str, Any] = {
            "mode": "advisory",
            "explain": False,
            "dry_run": False,
            "offline": False,
            "accept_threshold": 0.85,
            "lead_window_days": 90,
            "reissue_long_gap_years": 10,
            "charts_enabled": True,
            "charts_db_path": None,
            "decisions_db_path": None,
        }

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set config value."""
        self._config[key] = value

    @property
    def mode(self) -> CanonMode:
        """Get current canonicalization mode."""
        mode_str = self.get_config("mode", "advisory")
        try:
            return CanonMode(mode_str)
        except ValueError:
            return CanonMode.ADVISORY

    @property
    def resolver(self) -> Resolver:
        """Lazy-initialize resolver."""
        if self._resolver is None:
            config = ConfigSnapshot(
                lead_window_days=self.get_config("lead_window_days", 90),
                reissue_long_gap_years=self.get_config("reissue_long_gap_years", 10),
            )
            self._resolver = Resolver(config)
        return self._resolver

    @property
    def charts_db(self) -> ChartsDB | None:
        """Lazy-initialize charts DB."""
        if self._charts_db is None and self.get_config("charts_enabled", True):
            db_path = self.get_config("charts_db_path")
            if db_path:
                self._charts_db = ChartsDB(Path(db_path))
        return self._charts_db

    @property
    def charts_exporter(self) -> ChartsExporter | None:
        """Lazy-initialize charts exporter."""
        if self._charts_exporter is None and self.charts_db is not None:
            self._charts_exporter = ChartsExporter(self.charts_db)
        return self._charts_exporter

    @property
    def decisions_db(self) -> DecisionsDB | None:
        """Lazy-initialize decisions DB."""
        if self._decisions_db is None:
            db_path = self.get_config("decisions_db_path")
            if db_path:
                self._decisions_db = DecisionsDB(Path(db_path))
        return self._decisions_db

    def on_import_begin(self, session: Any) -> None:
        """Handle import_begin event."""
        log.debug("Chart-Binder: Import begin")
        self._import_summary = ImportSummary()
        log.info(f"Chart-Binder mode: {self.mode.value}")
        if self.get_config("explain", False):
            log.info("Chart-Binder explain mode enabled")
        if self.get_config("dry_run", False):
            log.info("Chart-Binder dry-run mode enabled")

    def on_import_task_files(self, session: Any, task: Any) -> None:
        """Handle import_task_files event."""
        if not hasattr(task, "items"):
            return

        for item in task.items:
            try:
                file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
                tagset = verify(file_path)
                item._chart_binder_tagset = tagset
                log.debug(f"Chart-Binder: Read tags from {file_path}")
            except Exception as e:
                log.warning(f"Chart-Binder: Error reading {item.path}: {e}", exc_info=True)
                if self._import_summary:
                    self._import_summary.errors.append(str(e))

    def on_import_task_choice(self, session: Any, task: Any) -> None:
        """Handle import_task_choice event."""
        if not hasattr(task, "items"):
            return

        for item in task.items:
            try:
                decision_summary = self._make_decision(item)
                item._chart_binder_decision = decision_summary
                if self.mode == CanonMode.ADVISORY and self.get_config("explain", False):
                    self._show_side_by_side(item, decision_summary, task)
            except Exception as e:
                log.warning(f"Chart-Binder: Error deciding {item.path}: {e}", exc_info=True)
                if self._import_summary:
                    self._import_summary.errors.append(str(e))

    def on_import_task_apply(self, session: Any, task: Any) -> None:
        """Handle import_task_apply event."""
        if not hasattr(task, "items"):
            return

        dry_run = self.get_config("dry_run", False)
        summary = self._import_summary

        for item in task.items:
            decision_summary = getattr(item, "_chart_binder_decision", None)
            if not decision_summary:
                if summary:
                    summary.skipped += 1
                continue

            if summary:
                summary.total_items += 1

            try:
                file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
                tagset = self._build_tagset_from_decision(decision_summary)

                if decision_summary.state == DecisionState.INDETERMINATE:
                    if summary:
                        summary.indeterminate += 1
                    if self.mode == CanonMode.AUGMENT:
                        write_tags(file_path, tagset, authoritative=False, dry_run=dry_run)
                        if summary:
                            summary.augmented += 1
                    continue

                if self.mode == CanonMode.AUTHORITATIVE:
                    write_tags(file_path, tagset, authoritative=True, dry_run=dry_run)
                    if summary:
                        summary.canonized += 1
                elif self.mode == CanonMode.AUGMENT:
                    write_tags(file_path, tagset, authoritative=False, dry_run=dry_run)
                    if summary:
                        summary.augmented += 1
                else:
                    accepted = getattr(item, "_chart_binder_accepted", False)
                    if accepted:
                        write_tags(file_path, tagset, authoritative=True, dry_run=dry_run)
                        if summary:
                            summary.canonized += 1
                    else:
                        write_tags(file_path, tagset, authoritative=False, dry_run=dry_run)
                        if summary:
                            summary.augmented += 1

                self._update_item_fields(item, decision_summary, tagset)

            except Exception as e:
                log.warning(f"Chart-Binder: Error writing {item.path}: {e}", exc_info=True)
                if summary:
                    summary.errors.append(str(e))
                    summary.skipped += 1

    def on_import_end(self, session: Any) -> None:
        """Handle import_end event."""
        log.debug("Chart-Binder: Import end")
        if self._import_summary:
            self._import_summary.print_summary()

    def _make_decision(self, item: Any) -> DecisionSummary:
        """Make canonicalization decision for an item."""
        file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
        tagset = getattr(item, "_chart_binder_tagset", None)

        if not tagset:
            tagset = verify(file_path)

        evidence_bundle = self._build_evidence_bundle(item, tagset)
        decision = self.resolver.resolve(evidence_bundle)

        charts_blob = None
        if self.charts_exporter and decision.state == DecisionState.DECIDED:
            work_key = self._compute_work_key(item, tagset)
            if work_key:
                try:
                    blob = self.charts_exporter.export_for_work(work_key)
                    charts_blob = blob.to_json()
                except Exception:
                    log.warning(
                        f"Chart-Binder: Error exporting charts for work key '{work_key}'",
                        exc_info=True,
                    )

        return DecisionSummary(
            file_path=file_path,
            title=tagset.title,
            artist=tagset.artist,
            album=tagset.album,
            release_group_mbid=decision.release_group_mbid,
            release_mbid=decision.release_mbid,
            crg_rationale=decision.crg_rationale.value if decision.crg_rationale else None,
            rr_rationale=decision.rr_rationale.value if decision.rr_rationale else None,
            state=decision.state,
            compact_trace=decision.compact_tag,
            charts_blob=charts_blob,
            beets_album=getattr(item, "album", None),
            beets_date=str(getattr(item, "year", "")) if hasattr(item, "year") else None,
            beets_label=getattr(item, "label", None),
        )

    def _build_evidence_bundle(self, item: Any, tagset: TagSet) -> dict[str, Any]:
        """Build evidence bundle from Beets item and tagset."""
        evidence_bundle: dict[str, Any] = {
            "artifact": {
                "file_path": str(item.path),
                "duration_ms": int(getattr(item, "length", 0) * 1000),
            },
            "artist": {
                "name": tagset.artist or getattr(item, "artist", "Unknown"),
                "mb_artist_id": getattr(item, "mb_artistid", None),
            },
            "recording_candidates": [],
            "timeline_facts": {},
            "provenance": {
                "sources_used": ["beets", "local_tags"],
            },
        }

        if tagset.ids.mb_release_group_id:
            evidence_bundle["recording_candidates"] = [
                {
                    "mb_recording_id": tagset.ids.mb_recording_id
                    or getattr(item, "mb_trackid", None),
                    "title": tagset.title or getattr(item, "title", None),
                    "rg_candidates": [
                        {
                            "mb_rg_id": tagset.ids.mb_release_group_id,
                            "title": tagset.album or getattr(item, "album", None),
                            "primary_type": "Album",
                            "first_release_date": tagset.original_year
                            or str(getattr(item, "year", "")),
                            "releases": [
                                {
                                    "mb_release_id": tagset.ids.mb_release_id
                                    or getattr(item, "mb_albumid", None),
                                    "date": tagset.original_year or str(getattr(item, "year", "")),
                                    "country": tagset.country,
                                    "label": tagset.label or getattr(item, "label", None),
                                    "title": tagset.album or getattr(item, "album", None),
                                    "flags": {"is_official": True},
                                }
                            ],
                        }
                    ],
                }
            ]

        return evidence_bundle

    def _build_tagset_from_decision(self, decision_summary: DecisionSummary) -> TagSet:
        """Build TagSet from DecisionSummary."""
        return TagSet(
            title=decision_summary.title,
            artist=decision_summary.artist,
            album=decision_summary.album,
            ids=CanonicalIDs(
                mb_release_group_id=decision_summary.release_group_mbid,
                mb_release_id=decision_summary.release_mbid,
            ),
            compact=CompactFields(
                decision_trace=decision_summary.compact_trace,
                charts_blob=decision_summary.charts_blob,
                ruleset_version="canon-1.0",
            ),
        )

    def _compute_work_key(self, item: Any, tagset: TagSet) -> str | None:
        """Compute work key for CHARTS lookup."""
        artist = tagset.artist or getattr(item, "artist", None)
        title = tagset.title or getattr(item, "title", None)

        if not artist or not title:
            return None

        def normalize(s: str) -> str:
            s = unicodedata.normalize("NFC", s)
            return s.lower().strip()

        return f"{normalize(artist)} // {normalize(title)}"

    def _show_side_by_side(self, item: Any, decision: DecisionSummary, task: Any) -> None:
        """Show side-by-side comparison in advisory mode."""
        _print_msg("\n--- Chart-Binder Decision ---")
        _print_msg(f"File: {decision.file_path}")
        _print_msg("")

        if hasattr(task, "match") and task.match:
            _print_msg("Beets candidate:")
            _print_msg(f"  Album: {decision.beets_album}")
            _print_msg(f"  Date: {decision.beets_date}")
            _print_msg(f"  Label: {decision.beets_label}")
            _print_msg("")

        _print_msg("Chart-Binder decision:")
        _print_msg(f"  State: {decision.state.value}")
        if decision.release_group_mbid:
            _print_msg(f"  CRG: {decision.release_group_mbid}")
            _print_msg(f"       ({decision.crg_rationale})")
        if decision.release_mbid:
            _print_msg(f"  RR: {decision.release_mbid}")
            _print_msg(f"       ({decision.rr_rationale})")
        if decision.compact_trace:
            _print_msg(f"  Trace: {decision.compact_trace}")
        _print_msg("")

    def _update_item_fields(self, item: Any, decision: DecisionSummary, tagset: TagSet) -> None:
        """Update Beets item with canonical fields."""
        if decision.release_group_mbid:
            item.mb_releasegroupid = decision.release_group_mbid
        if decision.release_mbid:
            item.mb_albumid = decision.release_mbid

        if hasattr(item, "update"):
            item.update(
                {
                    "canon_state": decision.state.value,
                    "canon_crg_rationale": decision.crg_rationale,
                    "canon_rr_rationale": decision.rr_rationale,
                    "canon_trace": decision.compact_trace,
                }
            )
            if decision.charts_blob:
                item.update({"charts": decision.charts_blob})

    def explain_items(self, lib: Any, query: list[str], albums: bool) -> None:
        """Show decision trace for matched items."""
        if albums:
            items = lib.albums(query)
        else:
            items = lib.items(query)

        for item in items:
            try:
                file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
                tagset = verify(file_path)

                _print_msg(f"\n{file_path}")
                _print_msg(f"  Title: {tagset.title}")
                _print_msg(f"  Artist: {tagset.artist}")
                _print_msg(f"  Album: {tagset.album}")

                if tagset.compact.decision_trace:
                    _print_msg(f"  Trace: {tagset.compact.decision_trace}")
                if tagset.ids.mb_release_group_id:
                    _print_msg(f"  MB RG: {tagset.ids.mb_release_group_id}")
                if tagset.ids.mb_release_id:
                    _print_msg(f"  MB Release: {tagset.ids.mb_release_id}")
                if tagset.compact.charts_blob:
                    _print_msg(f"  CHARTS: {tagset.compact.charts_blob[:50]}...")

            except Exception as e:
                _print_msg(f"Error: {e}")

    def pin_items(self, lib: Any, query: list[str], albums: bool, pin: bool) -> None:
        """Pin or unpin decisions for matched items."""
        if self.decisions_db is None:
            _print_msg("Error: decisions database not configured")
            return

        if albums:
            items = lib.albums(query)
        else:
            items = lib.items(query)

        count = 0
        for item in items:
            try:
                file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
                file_id = self._compute_file_id(file_path)
                decision = self.decisions_db.get_decision(file_id)

                if decision:
                    self.decisions_db.set_pinned(decision["decision_id"], pin)
                    count += 1
                    action = "Pinned" if pin else "Unpinned"
                    _print_msg(f"{action}: {file_path}")

            except Exception as e:
                _print_msg(f"Error: {e}")

        _print_msg(f"\n{'Pinned' if pin else 'Unpinned'} {count} items")

    def override_items(
        self,
        lib: Any,
        query: list[str],
        albums: bool,
        rg_mbid: str | None,
        release_mbid: str | None,
    ) -> None:
        """Override canonicalization for matched items."""
        if not rg_mbid:
            _print_msg("Error: --rg is required")
            return

        if self.decisions_db is None:
            _print_msg("Error: decisions database not configured")
            return

        if albums:
            items = lib.albums(query)
        else:
            items = lib.items(query)

        count = 0
        for item in items:
            try:
                file_path = Path(item.path.decode() if isinstance(item.path, bytes) else item.path)
                self.decisions_db.create_override_rule(
                    scope="file",
                    scope_id=str(file_path),
                    directive=f"prefer_rg={rg_mbid}"
                    + (f",prefer_release={release_mbid}" if release_mbid else ""),
                    created_by="beets_plugin",
                )
                count += 1
                _print_msg(f"Override created: {file_path} -> {rg_mbid}")

            except Exception as e:
                _print_msg(f"Error: {e}")

        _print_msg(f"\nCreated {count} override rules")

    def _compute_file_id(self, file_path: Path) -> str:
        """Compute stable file ID for decisions DB."""
        stat = file_path.stat()
        key = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]


# Only define the plugin class if beets is available
if _beets_available and _BeetsPlugin is not None:

    class ChartBinderPlugin(_BeetsPlugin):  # pyright: ignore[reportGeneralTypeIssues]
        """
        Chart-Binder Beets Plugin.

        Provides canonicalization of audio files during Beets import,
        with optional CHARTS blob attachment and ID linking.
        """

        def __init__(self) -> None:
            super().__init__()

            # Default configuration
            self.config.add(
                {
                    "mode": "advisory",
                    "explain": False,
                    "dry_run": False,
                    "offline": False,
                    "accept_threshold": 0.85,
                    "lead_window_days": 90,
                    "reissue_long_gap_years": 10,
                    "charts_enabled": True,
                    "charts_db_path": None,
                    "decisions_db_path": None,
                }
            )

            # Delegate to base class
            self._base = ChartBinderPluginBase()

            # Register listeners
            self.register_listener("import_begin", self._base.on_import_begin)
            self.register_listener("import_task_files", self._base.on_import_task_files)
            self.register_listener("import_task_choice", self._base.on_import_task_choice)
            self.register_listener("import_task_apply", self._base.on_import_task_apply)
            self.register_listener("import_end", self._base.on_import_end)

            # Sync config
            self._sync_config()

        def _sync_config(self) -> None:
            """Sync beets config to base class."""
            for key in self._base._config:
                self._base.set_config(key, self.config[key].get())

        def commands(self) -> list[Any]:
            """Register CLI subcommands."""
            if _Subcommand is None or _decargs is None:
                return []

            commands = [
                self._make_canon_explain_cmd(),
                self._make_canon_pin_cmd(),
                self._make_canon_unpin_cmd(),
                self._make_canon_override_cmd(),
            ]
            return [c for c in commands if c is not None]

        def _make_canon_explain_cmd(self) -> Any:
            """Create the canon-explain command."""
            if _Subcommand is None or _decargs is None:
                return None
            decargs_fn = _decargs  # Local reference to help type checker
            cmd = _Subcommand("canon-explain", help="Show decision trace for items")
            cmd.parser.add_option("-a", "--album", action="store_true", help="Match albums instead")

            def func(lib: Any, opts: Any, args: Any) -> None:
                self._base.explain_items(lib, decargs_fn(args), opts.album)

            cmd.func = func
            return cmd

        def _make_canon_pin_cmd(self) -> Any:
            """Create the canon-pin command."""
            if _Subcommand is None or _decargs is None:
                return None
            decargs_fn = _decargs  # Local reference to help type checker
            cmd = _Subcommand("canon-pin", help="Pin current decision for items")
            cmd.parser.add_option("-a", "--album", action="store_true", help="Match albums instead")

            def func(lib: Any, opts: Any, args: Any) -> None:
                self._base.pin_items(lib, decargs_fn(args), opts.album, pin=True)

            cmd.func = func
            return cmd

        def _make_canon_unpin_cmd(self) -> Any:
            """Create the canon-unpin command."""
            if _Subcommand is None or _decargs is None:
                return None
            decargs_fn = _decargs  # Local reference to help type checker
            cmd = _Subcommand("canon-unpin", help="Unpin decision for items")
            cmd.parser.add_option("-a", "--album", action="store_true", help="Match albums instead")

            def func(lib: Any, opts: Any, args: Any) -> None:
                self._base.pin_items(lib, decargs_fn(args), opts.album, pin=False)

            cmd.func = func
            return cmd

        def _make_canon_override_cmd(self) -> Any:
            """Create the canon-override command."""
            if _Subcommand is None or _decargs is None:
                return None
            decargs_fn = _decargs  # Local reference to help type checker
            cmd = _Subcommand("canon-override", help="Override canonicalization for items")
            cmd.parser.add_option("--rg", dest="rg_mbid", help="Target release group MBID")
            cmd.parser.add_option("--release", dest="release_mbid", help="Target release MBID")
            cmd.parser.add_option("-a", "--album", action="store_true", help="Match albums instead")

            def func(lib: Any, opts: Any, args: Any) -> None:
                self._base.override_items(
                    lib, decargs_fn(args), opts.album, opts.rg_mbid, opts.release_mbid
                )

            cmd.func = func
            return cmd
