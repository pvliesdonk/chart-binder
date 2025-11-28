#!/usr/bin/env python3
"""
Pre-flight verification script for chart scrapers.

Validates assumptions about chart data sources against actual internet responses.
Run before finalizing scraper implementations to confirm URL patterns and CSS selectors.
"""

from __future__ import annotations

import sys
from textwrap import dedent

import httpx


def check_top2000_api() -> bool:
    """Verify Top 2000 API URL patterns for different years."""
    print("\n" + "=" * 60)
    print("TOP 2000 API PATTERN VERIFICATION")
    print("=" * 60)

    client = httpx.Client(timeout=30.0, follow_redirects=True)
    success = True

    for year in [2023, 2020, 2024]:
        url_new = f"https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-{year}-12-25"
        url_old = f"https://www.nporadio2.nl/api/charts/top-2000-van-{year}-12-25"

        print(f"\n--- Year {year} ---")

        try:
            resp_new = client.get(url_new)
            print(f"  New pattern: {resp_new.status_code}")
            if resp_new.status_code == 200:
                data = resp_new.json()
                print(f"  Response type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:5]}")
                elif isinstance(data, list) and len(data) > 0:
                    print(
                        f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}"
                    )
        except Exception as e:
            print(f"  New pattern error: {e}")

        try:
            resp_old = client.get(url_old)
            print(f"  Old pattern: {resp_old.status_code}")
            if resp_old.status_code == 200:
                data = resp_old.json()
                print(f"  Response type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:5]}")
        except Exception as e:
            print(f"  Old pattern error: {e}")

    client.close()
    return success


def check_top40_split() -> bool:
    """Verify Top 40 separator patterns for legacy data."""
    print("\n" + "=" * 60)
    print("TOP 40 SPLIT ENTRY VERIFICATION (Beatles 1967-W07)")
    print("=" * 60)

    client = httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "chart-binder/1.0"},
    )

    url = "https://www.top40.nl/top40/1967/week-07"
    print(f"\nFetching: {url}")

    try:
        resp = client.get(url)
        print(f"Status: {resp.status_code}")

        if resp.status_code == 200:
            html = resp.text

            import re

            strawberry_matches = list(re.finditer(r"strawberry.{0,100}field", html, re.IGNORECASE))
            print(f"\n'Strawberry Fields' matches: {len(strawberry_matches)}")
            for i, m in enumerate(strawberry_matches[:3]):
                ctx_start = max(0, m.start() - 50)
                ctx_end = min(len(html), m.end() + 50)
                context = html[ctx_start:ctx_end].replace("\n", " ")
                print(f"  Match {i + 1}: ...{context}...")

            penny_lane_matches = list(re.finditer(r"penny.{0,20}lane", html, re.IGNORECASE))
            print(f"\n'Penny Lane' matches: {len(penny_lane_matches)}")

            artist_divs = list(
                re.finditer(r'class="[^"]*artist[^"]*"[^>]*>([^<]+)', html, re.IGNORECASE)
            )
            print(f"\nArtist divs found: {len(artist_divs)}")
            for i, m in enumerate(artist_divs[:5]):
                print(f"  Artist {i + 1}: {m.group(1).strip()[:50]}")

            title_divs = list(
                re.finditer(r'class="[^"]*title[^"]*"[^>]*>([^<]+)', html, re.IGNORECASE)
            )
            print(f"\nTitle divs found: {len(title_divs)}")
            for i, m in enumerate(title_divs[:5]):
                print(f"  Title {i + 1}: {m.group(1).strip()[:50]}")

        else:
            print(f"Failed to fetch: {resp.status_code}")

    except Exception as e:
        print(f"Error: {e}")
        return False

    client.close()
    return True


def check_zwaarste_lijst() -> bool:
    """Verify De Zwaarste Lijst page structure."""
    print("\n" + "=" * 60)
    print("DE ZWAARSTE LIJST 2024 VERIFICATION")
    print("=" * 60)

    client = httpx.Client(timeout=30.0, follow_redirects=True)

    url = "https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst"
    print(f"\nFetching: {url}")

    try:
        resp = client.get(url)
        print(f"Status: {resp.status_code}")

        if resp.status_code == 200:
            html = resp.text

            import re

            ol_matches = list(re.finditer(r"<ol[^>]*>", html, re.IGNORECASE))
            print(f"\n<ol> tags found: {len(ol_matches)}")

            li_matches = list(
                re.finditer(r"<li[^>]*>(.{0,200}?)</li>", html, re.IGNORECASE | re.DOTALL)
            )
            print(f"<li> items found: {len(li_matches)}")
            for i, m in enumerate(li_matches[:3]):
                content = re.sub(r"<[^>]+>", "", m.group(1)).strip()[:80]
                print(f"  Item {i + 1}: {content}")

            table_matches = list(re.finditer(r"<table[^>]*>", html, re.IGNORECASE))
            print(f"\n<table> tags found: {len(table_matches)}")

            p_matches = list(re.finditer(r"<p[^>]*>(\d+\.?\s+.{0,150}?)</p>", html, re.IGNORECASE))
            print(f"\nNumbered <p> tags found: {len(p_matches)}")
            for i, m in enumerate(p_matches[:3]):
                content = re.sub(r"<[^>]+>", "", m.group(1)).strip()[:80]
                print(f"  Item {i + 1}: {content}")

        elif resp.status_code == 404:
            print("Page not found - URL may have changed")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    client.close()
    return True


def main() -> int:
    """Run all pre-flight checks."""
    print(
        dedent("""
        ╔════════════════════════════════════════════════════════════╗
        ║         CHART SCRAPERS PRE-FLIGHT VERIFICATION             ║
        ╚════════════════════════════════════════════════════════════╝
    """).strip()
    )

    results = []

    try:
        results.append(("Top 2000 API", check_top2000_api()))
    except Exception as e:
        print(f"\nTop 2000 check failed: {e}")
        results.append(("Top 2000 API", False))

    try:
        results.append(("Top 40 Split", check_top40_split()))
    except Exception as e:
        print(f"\nTop 40 check failed: {e}")
        results.append(("Top 40 Split", False))

    try:
        results.append(("De Zwaarste Lijst", check_zwaarste_lijst()))
    except Exception as e:
        print(f"\nDe Zwaarste Lijst check failed: {e}")
        results.append(("De Zwaarste Lijst", False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✔" if passed else "✘"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
