#!/usr/bin/env python3
"""Debug script to see if resolver filtering is actually working."""

import sys
sys.path.insert(0, "src")

from chart_binder.resolver import Resolver, ConfigSnapshot

# Create test candidates with compilations
test_candidates = [
    {
        "rg_mbid": "rg1",
        "primary_type": "Album",
        "secondary_types": [],
        "first_release_date": "1974-11-01",
    },
    {
        "rg_mbid": "rg2",
        "primary_type": "Album",
        "secondary_types": ["Compilation"],
        "first_release_date": "1981",
    },
    {
        "rg_mbid": "rg3",
        "primary_type": "Album",
        "secondary_types": ["Compilation"],
        "first_release_date": "1981",
    },
    {
        "rg_mbid": "rg4",
        "primary_type": "Single",
        "secondary_types": [],
        "first_release_date": "1974-10",
    },
]

print(f"Input: {len(test_candidates)} candidates")
for c in test_candidates:
    print(f"  - {c['rg_mbid']}: {c['primary_type']} {c.get('secondary_types', [])} {c.get('first_release_date')}")

resolver = Resolver(ConfigSnapshot())

# Test compilation filtering
filtered = resolver._filter_compilation_exclusion(test_candidates)
print(f"\nAfter compilation filtering: {len(filtered)} candidates")
for c in filtered:
    print(f"  - {c['rg_mbid']}: {c['primary_type']} {c.get('secondary_types', [])} {c.get('first_release_date')}")

# Test late re-release filtering
filtered2 = resolver._filter_late_rereleases(filtered)
print(f"\nAfter late re-release filtering: {len(filtered2)} candidates")
for c in filtered2:
    print(f"  - {c['rg_mbid']}: {c['primary_type']} {c.get('secondary_types', [])} {c.get('first_release_date')}")
