"""Canonical Release Group and Representative Release resolver.

Implements the deterministic rule table for selecting:
- Canonical Release Group (CRG): The authoritative release group for a recording
- Representative Release (RR): The specific release within CRG to use for metadata

See: docs/appendix/canonicalization_rule_table_v1.md
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class CRGRationale(StrEnum):
    """Rationale codes for Canonical Release Group selection."""

    SOUNDTRACK_PREMIERE = "CRG:SOUNDTRACK_PREMIERE"
    ALBUM_LEAD_WINDOW = "CRG:ALBUM_LEAD_WINDOW"
    SINGLE_TRUE_PREMIERE = "CRG:SINGLE_TRUE_PREMIERE"
    LIVE_ONLY_PREMIERE = "CRG:LIVE_ONLY_PREMIERE"
    INTENT_MATCH = "CRG:INTENT_MATCH"
    COMPILATION_EXCLUDED = "CRG:COMPILATION_EXCLUDED"
    COMPILATION_AS_PREMIERE = "CRG:COMPILATION_AS_PREMIERE"
    EARLIEST_OFFICIAL = "CRG:EARLIEST_OFFICIAL"
    LLM_ADJUDICATION = "CRG:LLM_ADJUDICATION"
    INDETERMINATE = "CRG:INDETERMINATE"


class RRRationale(StrEnum):
    """Rationale codes for Representative Release selection."""

    ORIGIN_COUNTRY_EARLIEST = "RR:ORIGIN_COUNTRY_EARLIEST"
    WORLD_EARLIEST = "RR:WORLD_EARLIEST"
    REISSUE_FILTER_APPLIED = "RR:REISSUE_FILTER_APPLIED"
    LLM_ADJUDICATION = "RR:LLM_ADJUDICATION"
    INDETERMINATE = "RR:INDETERMINATE"


class DecisionState(StrEnum):
    """State of canonicalization decision."""

    DECIDED = "decided"
    INDETERMINATE = "indeterminate"


@dataclass
class ConfigSnapshot:
    """Configuration knobs snapshot for decision reproducibility."""

    lead_window_days: int = 90
    reissue_long_gap_years: int = 10
    reissue_terms: list[str] = field(
        default_factory=lambda: [
            "remaster",
            "remastered",
            "reissue",
            "re-issue",
            "anniversary",
            "deluxe",
            "expanded",
        ]
    )
    label_authority_order: list[str] = field(default_factory=list)


@dataclass
class DecisionTrace:
    """Full structured decision trace for audit and debugging."""

    ruleset_version: str = "1.0"
    evidence_hash: str = ""
    artist_origin_country: str | None = None
    considered_candidates: list[dict[str, Any]] = field(default_factory=list)
    crg_selection: dict[str, Any] = field(default_factory=dict)
    rr_selection: dict[str, Any] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    config_snapshot: ConfigSnapshot = field(default_factory=ConfigSnapshot)

    def to_compact_tag(self) -> str:
        """
        Generate compact tag string for embedding.

        Format: evh=9b1f2c...;crg=SINGLE_TRUE_PREMIERE;rr=ORIGIN_COUNTRY_EARLIEST;src=mb,dc;cfg=lw90,rg10
        """
        # Extract sources used (mb, dc, sp, wd)
        source_map = {
            "MB": "mb",
            "Discogs": "dc",
            "Spotify": "sp",
            "Wikidata": "wd",
        }
        sources = set()
        for candidate in self.considered_candidates:
            for src in candidate.get("sources", []):
                if src in source_map:
                    sources.add(source_map[src])

        sources_str = ",".join(sorted(sources))

        # Extract rationale codes
        crg_code = self.crg_selection.get("rule", "").replace("CRG:", "")
        rr_code = self.rr_selection.get("rule", "").replace("RR:", "")

        # Config snapshot
        cfg = self.config_snapshot
        cfg_str = f"lw{cfg.lead_window_days},rg{cfg.reissue_long_gap_years}"

        # Evidence hash (truncate to first 8 chars for readability)
        evh_short = self.evidence_hash[:8] if self.evidence_hash else "unknown"

        return f"evh={evh_short};crg={crg_code};rr={rr_code};src={sources_str};cfg={cfg_str}"

    def to_human_readable(self) -> str:
        """
        Generate human-readable explanation of the decision.

        Used for CLI --explain output and audit trails.
        """
        lines = []
        lines.append("Decision Trace")
        lines.append("=" * 50)

        lines.append(f"Ruleset Version: {self.ruleset_version}")
        lines.append(f"Evidence Hash: {self.evidence_hash[:16]}...")

        if self.artist_origin_country:
            lines.append(f"Artist Origin Country: {self.artist_origin_country}")

        # CRG selection explanation
        lines.append("\nCRG Selection:")
        if self.crg_selection:
            rule = self.crg_selection.get("rule", "unknown")
            lines.append(f"  Rule Applied: {rule}")

            if "first_release_date" in self.crg_selection:
                lines.append(f"  First Release Date: {self.crg_selection['first_release_date']}")
            if "delta_days" in self.crg_selection:
                lines.append(f"  Lead Window Delta: {self.crg_selection['delta_days']} days")
            if "tie_breaker" in self.crg_selection:
                lines.append(f"  Tie-breaker: {self.crg_selection['tie_breaker']}")

        # RR selection explanation
        lines.append("\nRR Selection:")
        if self.rr_selection:
            rule = self.rr_selection.get("rule", "unknown")
            lines.append(f"  Rule Applied: {rule}")

            release = self.rr_selection.get("release", {})
            if release:
                if release.get("date"):
                    lines.append(f"  Release Date: {release['date']}")
                if release.get("country"):
                    lines.append(f"  Country: {release['country']}")
                if release.get("label"):
                    lines.append(f"  Label: {release['label']}")

        # Candidates considered
        if self.considered_candidates:
            lines.append(f"\nCandidates Considered: {len(self.considered_candidates)}")
            for i, cand in enumerate(self.considered_candidates[:5]):  # Show first 5
                rg_id = cand.get("rg")
                rg_display = rg_id[:8] if isinstance(rg_id, str) and rg_id else "?"
                lines.append(f"  {i + 1}. {cand.get('type', '?')} RG {rg_display}...")
                if cand.get("first_date"):
                    lines.append(f"      First date: {cand['first_date']}")

        # Missing facts
        if self.missing_facts:
            lines.append("\nMissing Facts:")
            for fact in self.missing_facts:
                lines.append(f"  - {fact}")

        # Config
        lines.append("\nConfig:")
        lines.append(f"  Lead Window: {self.config_snapshot.lead_window_days} days")
        lines.append(f"  Reissue Gap: {self.config_snapshot.reissue_long_gap_years} years")

        return "\n".join(lines)


@dataclass
class CanonicalDecision:
    """Result of canonicalization process."""

    state: DecisionState
    release_group_mbid: str | None = None
    release_mbid: str | None = None
    crg_rationale: CRGRationale | None = None
    rr_rationale: RRRationale | None = None
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    compact_tag: str = ""


class Resolver:
    """
    Canonical Release Group and Representative Release resolver.

    Implements the 7-rule CRG selection algorithm and RR selection within CRG.
    """

    def __init__(self, config: ConfigSnapshot | None = None):
        self.config = config or ConfigSnapshot()

    def resolve(self, evidence_bundle: dict[str, Any]) -> CanonicalDecision:
        """
        Resolve canonical release group and representative release.

        Args:
            evidence_bundle: Evidence bundle v1 structure

        Returns:
            CanonicalDecision with CRG/RR mbids and rationale codes
        """
        trace = DecisionTrace(config_snapshot=self.config)

        # Compute evidence hash
        trace.evidence_hash = self._hash_evidence(evidence_bundle)

        # Extract artist origin country
        artist = evidence_bundle.get("artist", {})
        trace.artist_origin_country = artist.get("begin_area_country") or artist.get(
            "wikidata_country"
        )

        # Build considered candidates list for trace
        trace.considered_candidates = self._build_candidate_list(evidence_bundle)

        # CRG Selection (7 rules)
        crg_result = self._select_crg(evidence_bundle, trace)

        if crg_result["state"] == DecisionState.INDETERMINATE:
            trace.crg_selection = crg_result
            trace.missing_facts = crg_result.get("missing_facts", [])
            decision = CanonicalDecision(
                state=DecisionState.INDETERMINATE,
                crg_rationale=CRGRationale.INDETERMINATE,
                decision_trace=trace,
            )
            decision.compact_tag = trace.to_compact_tag()
            return decision

        # CRG was selected, now select RR within it
        crg_mbid = crg_result["crg_mbid"]
        crg_rationale = crg_result["rationale"]
        trace.crg_selection = {
            "rule": crg_rationale,
            **{k: v for k, v in crg_result.items() if k not in ["crg_mbid", "rationale", "state"]},
        }

        # RR Selection within CRG
        rr_result = self._select_rr(evidence_bundle, crg_mbid, trace.artist_origin_country, trace)

        if rr_result["state"] == DecisionState.INDETERMINATE:
            trace.rr_selection = rr_result
            trace.missing_facts.extend(rr_result.get("missing_facts", []))
            decision = CanonicalDecision(
                state=DecisionState.INDETERMINATE,
                release_group_mbid=crg_mbid,
                crg_rationale=crg_rationale,
                rr_rationale=RRRationale.INDETERMINATE,
                decision_trace=trace,
            )
            decision.compact_tag = trace.to_compact_tag()
            return decision

        # Both CRG and RR decided
        rr_mbid = rr_result["rr_mbid"]
        rr_rationale = rr_result["rationale"]
        trace.rr_selection = {
            "rule": rr_rationale,
            **{k: v for k, v in rr_result.items() if k not in ["rr_mbid", "rationale", "state"]},
        }

        decision = CanonicalDecision(
            state=DecisionState.DECIDED,
            release_group_mbid=crg_mbid,
            release_mbid=rr_mbid,
            crg_rationale=crg_rationale,
            rr_rationale=rr_rationale,
            decision_trace=trace,
        )
        decision.compact_tag = trace.to_compact_tag()
        return decision

    def _select_crg(self, evidence_bundle: dict[str, Any], trace: DecisionTrace) -> dict[str, Any]:
        """
        Select Canonical Release Group using 7-rule algorithm.

        Returns dict with: state, crg_mbid, rationale, and rule-specific metadata
        """
        recording_candidates = evidence_bundle.get("recording_candidates", [])

        if not recording_candidates:
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": ["no_recording_candidates"],
            }

        # Flatten all RG candidates from all recordings
        all_rg_candidates = []
        for rec in recording_candidates:
            for rg in rec.get("rg_candidates", []):
                all_rg_candidates.append(
                    {
                        "recording": rec,
                        "rg": rg,
                        "rg_mbid": rg["mb_rg_id"],
                        "primary_type": rg.get("primary_type"),
                        "secondary_types": rg.get("secondary_types", []),
                        "first_release_date": rg.get("first_release_date"),
                    }
                )

        if not all_rg_candidates:
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": ["no_release_group_candidates"],
            }

        # Rule 1: Soundtrack Origin
        result = self._rule_soundtrack_origin(all_rg_candidates, trace)
        if result:
            return result

        # Rule 2: Album vs Promo-Single (Lead-Window)
        result = self._rule_album_lead_window(all_rg_candidates, evidence_bundle)
        if result:
            return result

        # Rule 3: Live Origin
        result = self._rule_live_origin(all_rg_candidates)
        if result:
            return result

        # Rule 4: Remix/DJ-mix/Other Intent
        result = self._rule_intent_match(all_rg_candidates, recording_candidates)
        if result:
            return result

        # Rule 5: Compilation Exclusion
        # (This is more of a filter than a direct selection)
        all_rg_candidates = self._filter_compilation_exclusion(all_rg_candidates)

        # Rule 6: Fallback by Earliest Official Date
        result = self._rule_earliest_official(all_rg_candidates, trace)
        if result:
            return result

        # Rule 7: Ambiguous / Insufficient
        return {
            "state": DecisionState.INDETERMINATE,
            "rationale": CRGRationale.INDETERMINATE,
            "missing_facts": ["insufficient_date_evidence_or_tie"],
        }

    def _rule_soundtrack_origin(
        self, candidates: list[dict[str, Any]], trace: DecisionTrace
    ) -> dict[str, Any] | None:
        """
        Rule 1: Soundtrack Origin.

        Select Soundtrack RG if its first_release_date ≤ min of all other types.
        """
        soundtrack_candidates = [
            c for c in candidates if c["primary_type"] == "Soundtrack" and c["first_release_date"]
        ]

        if not soundtrack_candidates:
            return None

        # Find earliest date among all candidates
        all_dates = [c["first_release_date"] for c in candidates if c["first_release_date"]]
        if not all_dates:
            return None

        earliest_date = min(all_dates)

        # Check if any soundtrack has this earliest date
        earliest_soundtracks = [
            c for c in soundtrack_candidates if c["first_release_date"] == earliest_date
        ]

        if not earliest_soundtracks:
            return None

        # Tie-breaker: label authority → artist origin country presence
        # TODO: Implement label authority and country tie-breakers
        # If still tied, defer to LLM adjudicator

        # If multiple soundtracks with same earliest date, return INDETERMINATE
        if len(earliest_soundtracks) > 1:
            return None  # Let later rules handle it, or fall through to INDETERMINATE

        selected = earliest_soundtracks[0]

        return {
            "state": DecisionState.DECIDED,
            "crg_mbid": selected["rg_mbid"],
            "rationale": CRGRationale.SOUNDTRACK_PREMIERE,
            "first_release_date": selected["first_release_date"],
        }

    def _rule_album_lead_window(
        self, candidates: list[dict[str, Any]], evidence_bundle: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Rule 2: Album vs Promo-Single (Lead-Window).

        Two sub-rules:
        - 2A: Album came first within lead window → CRG:ALBUM_LEAD_WINDOW
        - 2B: Single came first outside window → CRG:SINGLE_TRUE_PREMIERE
        """
        timeline_facts = evidence_bundle.get("timeline_facts", {})
        earliest_album_date = timeline_facts.get("earliest_album_date")
        earliest_single_ep_date = timeline_facts.get("earliest_single_ep_date")

        if not earliest_album_date and not earliest_single_ep_date:
            return None

        # Parse dates (assume ISO format YYYY-MM-DD)
        album_date = self._parse_date(earliest_album_date) if earliest_album_date else None
        single_date = self._parse_date(earliest_single_ep_date) if earliest_single_ep_date else None

        # Rule 2A: Album-first within window
        if album_date and single_date:
            # delta_days > 0 means album came AFTER single (lead single scenario)
            delta_days = (album_date - single_date).days

            # Lead Window Rule: If single is within 90 days before album, prefer album
            # This catches promo/lead singles that typically precede album releases
            if 0 < delta_days <= self.config.lead_window_days:
                # Select Album RG
                album_candidates = [
                    c
                    for c in candidates
                    if c["primary_type"] == "Album" and c["first_release_date"]
                ]
                if album_candidates:
                    earliest_album = min(album_candidates, key=lambda c: c["first_release_date"])
                    # Check for ties
                    earliest_date = earliest_album["first_release_date"]
                    tied = [c for c in album_candidates if c["first_release_date"] == earliest_date]
                    if len(tied) > 1:
                        return None  # Let later rules or INDETERMINATE handle it
                    return {
                        "state": DecisionState.DECIDED,
                        "crg_mbid": earliest_album["rg_mbid"],
                        "rationale": CRGRationale.ALBUM_LEAD_WINDOW,
                        "lead_window_days": self.config.lead_window_days,
                        "delta_days": delta_days,
                    }

        # Rule 2B: Single truly first
        if single_date:
            if not album_date or (album_date - single_date).days > self.config.lead_window_days:
                # Select Single/EP RG
                single_ep_candidates = [
                    c
                    for c in candidates
                    if c["primary_type"] in ["Single", "EP"] and c["first_release_date"]
                ]
                if single_ep_candidates:
                    earliest_single = min(
                        single_ep_candidates, key=lambda c: c["first_release_date"]
                    )
                    # Check for ties
                    earliest_date = earliest_single["first_release_date"]
                    tied = [
                        c for c in single_ep_candidates if c["first_release_date"] == earliest_date
                    ]
                    if len(tied) > 1:
                        return None  # Let later rules or INDETERMINATE handle it
                    return {
                        "state": DecisionState.DECIDED,
                        "crg_mbid": earliest_single["rg_mbid"],
                        "rationale": CRGRationale.SINGLE_TRUE_PREMIERE,
                        "first_release_date": earliest_single["first_release_date"],
                    }

        return None

    def _rule_live_origin(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Rule 3: Live Origin.

        Select Live RG if all non-Live candidates are absent or verified later.
        """
        live_candidates = [
            c for c in candidates if c["primary_type"] == "Live" and c["first_release_date"]
        ]
        non_live_candidates = [
            c for c in candidates if c["primary_type"] != "Live" and c["first_release_date"]
        ]

        if not live_candidates:
            return None

        # Check if Live is earliest
        if not non_live_candidates:
            # Only Live candidates exist
            earliest_live = min(live_candidates, key=lambda c: c["first_release_date"])
            # Check for ties
            earliest_date = earliest_live["first_release_date"]
            tied = [c for c in live_candidates if c["first_release_date"] == earliest_date]
            if len(tied) > 1:
                return None  # Let later rules or INDETERMINATE handle it
            return {
                "state": DecisionState.DECIDED,
                "crg_mbid": earliest_live["rg_mbid"],
                "rationale": CRGRationale.LIVE_ONLY_PREMIERE,
                "first_release_date": earliest_live["first_release_date"],
            }

        # Check if Live is earlier than all non-Live
        earliest_live_date = min(c["first_release_date"] for c in live_candidates)
        earliest_non_live_date = min(c["first_release_date"] for c in non_live_candidates)

        if earliest_live_date < earliest_non_live_date:
            earliest_live = min(live_candidates, key=lambda c: c["first_release_date"])
            # Check for ties
            tied = [c for c in live_candidates if c["first_release_date"] == earliest_live_date]
            if len(tied) > 1:
                return None  # Let later rules or INDETERMINATE handle it
            return {
                "state": DecisionState.DECIDED,
                "crg_mbid": earliest_live["rg_mbid"],
                "rationale": CRGRationale.LIVE_ONLY_PREMIERE,
                "first_release_date": earliest_live["first_release_date"],
            }

        return None

    def _rule_intent_match(
        self, candidates: list[dict[str, Any]], recording_candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Rule 4: Remix / DJ-mix / Other Intent.

        Select earliest RG matching recording intent (remix, radio edit, extended mix, etc.).
        """
        # Check if recording is a remix/alt version
        # TODO: Implement intent detection from recording flags and title normalization
        is_remix_or_alt = False  # Placeholder

        if not is_remix_or_alt:
            return None

        # Find earliest Single/EP or specialized RG (Remix EP, DJ Mix)
        intent_candidates = [
            c
            for c in candidates
            if c["primary_type"] in ["Single", "EP"]
            or "Remix" in c.get("secondary_types", [])
            or "DJ-mix" in c.get("secondary_types", [])
        ]

        if not intent_candidates:
            return None

        intent_candidates_with_dates = [c for c in intent_candidates if c["first_release_date"]]
        if not intent_candidates_with_dates:
            return None

        earliest_intent = min(intent_candidates_with_dates, key=lambda c: c["first_release_date"])

        return {
            "state": DecisionState.DECIDED,
            "crg_mbid": earliest_intent["rg_mbid"],
            "rationale": CRGRationale.INTENT_MATCH,
            "first_release_date": earliest_intent["first_release_date"],
        }

    def _filter_compilation_exclusion(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Rule 5: Compilation Exclusion.

        Filter out compilations - they are never canonical release groups.
        A recording's canonical home is always the original single/album, not
        a "Greatest Hits" or other compilation.
        """
        non_compilation = [
            c for c in candidates if "Compilation" not in c.get("secondary_types", [])
        ]

        # If all candidates are compilations, fall through to INDETERMINATE
        # (this would be unusual but possible for obscure recordings)
        if not non_compilation:
            return candidates

        return non_compilation

    def _rule_earliest_official(
        self, candidates: list[dict[str, Any]], trace: DecisionTrace
    ) -> dict[str, Any] | None:
        """
        Rule 6: Fallback by Earliest Official Date.

        Select RG with earliest confirmed official first_release_date.
        Tie-breakers: artist origin country → label authority → country precedence
        """
        candidates_with_dates = [c for c in candidates if c["first_release_date"]]

        if not candidates_with_dates:
            return None

        # Find earliest date
        earliest_date = min(c["first_release_date"] for c in candidates_with_dates)
        earliest_candidates = [
            c for c in candidates_with_dates if c["first_release_date"] == earliest_date
        ]

        if len(earliest_candidates) == 1:
            selected = earliest_candidates[0]
            return {
                "state": DecisionState.DECIDED,
                "crg_mbid": selected["rg_mbid"],
                "rationale": CRGRationale.EARLIEST_OFFICIAL,
                "first_release_date": selected["first_release_date"],
            }

        # Tie-breakers
        # TODO: Implement tie-breakers:
        # 1. Presence of artist origin country in RG's releases
        # 2. Label authority
        # 3. Country precedence
        # If still tied, defer to LLM adjudicator

        # If multiple candidates with same earliest date, return INDETERMINATE
        # to allow LLM adjudicator to break the tie
        if len(earliest_candidates) > 1:
            return {
                "state": DecisionState.INDETERMINATE,
                "rationale": CRGRationale.INDETERMINATE,
                "missing_facts": [
                    f"tie_between_{len(earliest_candidates)}_release_groups_with_same_earliest_date"
                ],
                "earliest_date": earliest_date,
                "tied_candidates": [c["rg_mbid"] for c in earliest_candidates],
            }

        selected = earliest_candidates[0]

        return {
            "state": DecisionState.DECIDED,
            "crg_mbid": selected["rg_mbid"],
            "rationale": CRGRationale.EARLIEST_OFFICIAL,
            "first_release_date": selected["first_release_date"],
        }

    def _select_rr(
        self,
        evidence_bundle: dict[str, Any],
        crg_mbid: str,
        artist_origin_country: str | None,
        trace: DecisionTrace,
    ) -> dict[str, Any]:
        """
        Select Representative Release within CRG.

        5-step algorithm:
        1. Filter official
        2. Artist origin country preference
        3. Else earliest worldwide
        4. Reissue/remaster guard
        5. Tie-breakers
        """
        # Find the CRG in evidence bundle
        crg_data = None
        for rec in evidence_bundle.get("recording_candidates", []):
            for rg in rec.get("rg_candidates", []):
                if rg["mb_rg_id"] == crg_mbid:
                    crg_data = rg
                    break
            if crg_data:
                break

        if not crg_data:
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": ["crg_not_found_in_evidence"],
            }

        releases = crg_data.get("releases", [])

        if not releases:
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": ["no_releases_in_crg"],
            }

        # Step 1: Filter official
        official_releases = [r for r in releases if r.get("flags", {}).get("is_official")]
        if not official_releases:
            # If only promos exist, keep best-evidenced promo
            promo_releases = [r for r in releases if r.get("flags", {}).get("is_promo")]
            if promo_releases:
                official_releases = promo_releases
            else:
                # No official or promo releases
                return {
                    "state": DecisionState.INDETERMINATE,
                    "missing_facts": ["no_official_releases"],
                }

        # Step 2: Artist origin country preference
        if artist_origin_country:
            origin_releases = [
                r for r in official_releases if r.get("country") == artist_origin_country
            ]
            if origin_releases:
                # Select earliest among origin country releases
                origin_with_dates = [r for r in origin_releases if r.get("date")]
                if origin_with_dates:
                    earliest_origin = min(origin_with_dates, key=lambda r: r["date"])
                    # Check for ties
                    earliest_date = earliest_origin["date"]
                    tied = [r for r in origin_with_dates if r["date"] == earliest_date]
                    if len(tied) > 1:
                        # Multiple releases on same date - return INDETERMINATE
                        return {
                            "state": DecisionState.INDETERMINATE,
                            "missing_facts": [
                                f"tie_between_{len(tied)}_origin_country_releases_on_same_date"
                            ],
                        }
                    return {
                        "state": DecisionState.DECIDED,
                        "rr_mbid": earliest_origin["mb_release_id"],
                        "rationale": RRRationale.ORIGIN_COUNTRY_EARLIEST,
                        "release": earliest_origin,
                    }

        # Step 3: Earliest worldwide
        releases_with_dates = [r for r in official_releases if r.get("date")]
        if not releases_with_dates:
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": ["no_release_dates"],
            }

        earliest_release = min(releases_with_dates, key=lambda r: r["date"])

        # Check for ties
        earliest_date = earliest_release["date"]
        tied = [r for r in releases_with_dates if r["date"] == earliest_date]
        if len(tied) > 1:
            # Multiple releases on same date - return INDETERMINATE
            # TODO: Implement label authority, format, catalog number tie-breakers
            return {
                "state": DecisionState.INDETERMINATE,
                "missing_facts": [f"tie_between_{len(tied)}_worldwide_releases_on_same_date"],
            }

        # Step 4: Reissue/remaster guard
        # TODO: Implement reissue detection and guard logic
        # For now, accept earliest_release as-is

        return {
            "state": DecisionState.DECIDED,
            "rr_mbid": earliest_release["mb_release_id"],
            "rationale": RRRationale.WORLD_EARLIEST,
            "release": earliest_release,
        }

    def _build_candidate_list(self, evidence_bundle: dict[str, Any]) -> list[dict[str, Any]]:
        """Build considered candidates list for decision trace."""
        candidates = []
        for rec in evidence_bundle.get("recording_candidates", []):
            for rg in rec.get("rg_candidates", []):
                candidates.append(
                    {
                        "rg": rg["mb_rg_id"],
                        "type": rg.get("primary_type"),
                        "first_date": rg.get("first_release_date"),
                        "sources": evidence_bundle.get("provenance", {}).get("sources_used", []),
                    }
                )
        return candidates

    def _hash_evidence(self, evidence_bundle: dict[str, Any]) -> str:
        """
        Hash evidence bundle deterministically.

        Removes timestamps and cache ages, then computes SHA256.
        """
        # Deep copy to avoid mutating the original bundle
        canonical = copy.deepcopy(evidence_bundle)

        # Remove volatile fields from provenance if present
        if "provenance" in canonical:
            canonical["provenance"].pop("fetched_at_utc", None)
            canonical["provenance"].pop("cache_age_s", None)

        # Serialize to canonical JSON
        json_bytes = json.dumps(
            canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")

        return hashlib.sha256(json_bytes).hexdigest()

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse ISO date string (YYYY-MM-DD) to datetime.

        Args:
            date_str: Date string in format YYYY, YYYY-MM, or YYYY-MM-DD

        Returns:
            datetime object (partial dates padded with Jan 1 for missing components)

        Raises:
            ValueError: If date_str is not a valid ISO date format
        """
        try:
            # Handle partial dates (YYYY-MM or YYYY)
            parts = date_str.split("-")
            if len(parts) == 1:
                # Year only
                return datetime(int(parts[0]), 1, 1)
            elif len(parts) == 2:
                # Year-month
                return datetime(int(parts[0]), int(parts[1]), 1)
            else:
                # Full date
                return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid date format: '{date_str}'. Expected YYYY, YYYY-MM, or YYYY-MM-DD."
            ) from e
