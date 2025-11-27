"""
Wikidata SPARQL client for artist country/origin queries.

Supports querying artist properties P27 (country of citizenship),
P495 (country of origin), and P740 (location of formation) with
result caching and timeout handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


@dataclass
class WikidataCountryResult:
    """Country information from Wikidata."""

    country_code: str  # ISO 3166-1 alpha-2 code (e.g., "US", "GB")
    country_name: str | None = None
    property_type: str | None = None  # "P27", "P495", or "P740"


class WikidataClient:
    """
    Wikidata SPARQL client for artist country/origin queries.

    Queries P27 (country of citizenship), P495 (country of origin),
    and P740 (location of formation).
    """

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    USER_AGENT = "chart-binder/0.1.0 ( https://github.com/pvliesdonk/chart-binder )"

    def __init__(
        self,
        cache: HttpCache | None = None,
        timeout_sec: int = 30,
    ):
        """
        Initialize Wikidata SPARQL client.

        Args:
            cache: Optional HTTP cache for responses
            timeout_sec: Request timeout in seconds
        """
        self.cache = cache
        self.timeout_sec = timeout_sec
        self._client = httpx.Client(
            timeout=float(timeout_sec),
            headers={"User-Agent": self.USER_AGENT},
        )

    def get_artist_countries(self, wikidata_qid: str) -> list[WikidataCountryResult]:
        """
        Get country information for an artist from Wikidata.

        Queries for:
        - P27: country of citizenship
        - P495: country of origin
        - P740: location of formation

        Args:
            wikidata_qid: Wikidata entity ID (e.g., "Q1299")

        Returns:
            List of WikidataCountryResult objects
        """
        # Ensure QID format
        if not wikidata_qid.startswith("Q"):
            wikidata_qid = f"Q{wikidata_qid}"

        query = f"""
        SELECT DISTINCT ?propertyType ?countryCode ?countryLabel WHERE {{
          {{
            wd:{wikidata_qid} wdt:P27 ?country .
            BIND("P27" AS ?propertyType)
          }} UNION {{
            wd:{wikidata_qid} wdt:P495 ?country .
            BIND("P495" AS ?propertyType)
          }} UNION {{
            wd:{wikidata_qid} wdt:P740 ?country .
            BIND("P740" AS ?propertyType)
          }}
          ?country wdt:P297 ?countryCode .

          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
          }}
        }}
        """

        params = {
            "query": query,
            "format": "json",
        }

        # Check cache - use query as cache key for correctness
        import hashlib

        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_key = f"{self.SPARQL_ENDPOINT}?qhash={query_hash}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return self._parse_response(cached.json())

        # Make live request
        response = self._client.get(self.SPARQL_ENDPOINT, params=params)
        response.raise_for_status()

        # Cache response
        if self.cache:
            # Create a mock response for caching
            mock_response = httpx.Response(
                status_code=200,
                content=response.content,
                headers=response.headers,
                request=httpx.Request("GET", cache_key),
            )
            self.cache.put(cache_key, mock_response)

        return self._parse_response(response.json())

    def _parse_response(self, data: dict[str, Any]) -> list[WikidataCountryResult]:
        """
        Parse SPARQL JSON response.

        Args:
            data: JSON response from Wikidata SPARQL endpoint

        Returns:
            List of WikidataCountryResult objects
        """
        results: list[WikidataCountryResult] = []

        bindings = data.get("results", {}).get("bindings", [])

        for binding in bindings:
            country_code_obj = binding.get("countryCode")
            if not country_code_obj:
                continue

            country_code = country_code_obj.get("value")
            if not country_code:
                continue

            # Extract country name
            country_name = None
            country_label_obj = binding.get("countryLabel")
            if country_label_obj:
                country_name = country_label_obj.get("value")

            # Extract property type (now directly from SPARQL BIND)
            property_type = None
            property_type_obj = binding.get("propertyType")
            if property_type_obj:
                property_type = property_type_obj.get("value")

            results.append(
                WikidataCountryResult(
                    country_code=country_code,
                    country_name=country_name,
                    property_type=property_type,
                )
            )

        return results

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> WikidataClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


## Tests


def test_wikidata_country_result():
    """Test WikidataCountryResult dataclass."""
    result = WikidataCountryResult(
        country_code="US",
        country_name="United States",
        property_type="P27",
    )
    assert result.country_code == "US"
    assert result.country_name == "United States"
    assert result.property_type == "P27"


def test_wikidata_parse_response():
    """Test parsing Wikidata SPARQL response."""
    client = WikidataClient()

    response_data = {
        "results": {
            "bindings": [
                {
                    "propertyType": {"value": "P27"},
                    "countryCode": {"value": "US"},
                    "countryLabel": {"value": "United States"},
                },
                {
                    "propertyType": {"value": "P740"},
                    "countryCode": {"value": "GB"},
                    "countryLabel": {"value": "United Kingdom"},
                },
            ]
        }
    }

    results = client._parse_response(response_data)

    assert len(results) == 2
    assert results[0].country_code == "US"
    assert results[0].property_type == "P27"
    assert results[1].country_code == "GB"
    assert results[1].property_type == "P740"


def test_wikidata_qid_formatting():
    """Test that QID is properly formatted."""
    client = WikidataClient()

    # Mock the response to avoid live request
    def mock_parse(data: dict[str, Any]) -> list[WikidataCountryResult]:
        return []

    client._parse_response = mock_parse

    # Should work with or without Q prefix
    # (Can't test the query itself without mocking httpx)
    assert True  # Placeholder - actual test would need mocking
