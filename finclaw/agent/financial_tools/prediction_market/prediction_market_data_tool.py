"""Prediction market data fetcher — Polymarket + Kalshi APIs.

No API keys required. Both platforms offer free public endpoints.

Polymarket Gamma API: market metadata, events, categories.
Polymarket CLOB API:  prices, order book, historical timeseries.
Kalshi API:           markets, events, trades, candlesticks.
"""

from __future__ import annotations

import asyncio
import difflib
import time
from typing import Any

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
_POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
_KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

_POLYMARKET_RATE_LIMIT = 300  # req/min — no documented public limit; conservative buffer
_KALSHI_RATE_LIMIT = 300      # req/min — no documented public limit; conservative buffer
_DEFAULT_HTTP_TIMEOUT = 15.0

_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry

# Minimum SequenceMatcher ratio to consider two market titles a cross-platform
# match.  0.55 balances precision (avoids "Fed Rate Hike" ≈ "Fed Rate Cut"
# false positives at 0.4) with recall for legitimately similar titles.
_COMPARE_MATCH_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Module-level rate-limit tracking (guarded by asyncio.Lock)
# ---------------------------------------------------------------------------

_polymarket_calls: list[float] = []
_kalshi_calls: list[float] = []
_rate_lock = asyncio.Lock()


async def _check_rate_limit(log: list[float], limit: int) -> float | None:
    """Return seconds to wait if at limit, else None.  Thread/async-safe."""
    async with _rate_lock:
        cutoff = time.time() - 60
        while log and log[0] < cutoff:
            log.pop(0)
        if len(log) >= limit:
            wait = 60 - (time.time() - log[0])
            return max(0.1, wait)
        log.append(time.time())
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(x: Any) -> float:
    """Safely convert to float, default 0.0."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _extract_poly_category(tags: Any) -> str:
    """Extract the primary visible category label from Polymarket tags list."""
    if not isinstance(tags, list):
        return ""
    # Prefer tags with forceShow=True, otherwise take the first with a label
    for t in tags:
        if isinstance(t, dict) and t.get("forceShow"):
            return t.get("label") or ""
    for t in tags:
        if isinstance(t, dict) and not t.get("forceHide"):
            return t.get("label") or ""
    return ""


def _parse_json_list(val: Any) -> list:
    """Parse a value that might be a JSON-encoded string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            import json
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ---------------------------------------------------------------------------
# PredictionMarketDataTool
# ---------------------------------------------------------------------------

class PredictionMarketDataTool:
    """Fetch prediction market data from Polymarket and Kalshi.

    This is NOT a Tool subclass — it is used internally by
    PredictionMarketRouter and never exposed to the LLM directly.
    """

    def __init__(self, *, timeout: float = _DEFAULT_HTTP_TIMEOUT) -> None:
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True,
            )
        return self._client

    # ── Polymarket Gamma API ──────────────────────────────────────────

    async def poly_trending(self, limit: int = 20) -> dict[str, Any]:
        """Get top active Polymarket events by 24h volume."""
        params = {
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": str(max(limit * 2, 50)),
        }
        data = await self._poly_gamma_get("/events", params=params)
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_events(data, top_n=limit)

    async def poly_search(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search Polymarket markets by keyword.

        Uses /markets instead of /events: individual market records are ~2 KB
        each (no nested sub-markets) vs ~25 KB for event records.  We fetch a
        larger pool for client-side keyword filtering without blowing up memory.
        """
        if not query:
            return {"error": "query is required for search"}
        data = await self._poly_gamma_get(
            "/markets",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": "200",
            },
        )
        if isinstance(data, dict) and "error" in data:
            return data
        return _filter_poly_markets(data, query, limit)

    async def poly_event_detail(self, event_id: str) -> dict[str, Any]:
        """Get detailed info on a specific Polymarket event."""
        if not event_id:
            return {"error": "event_id is required"}
        # Path param works for numeric IDs; some slugs trigger 422 — fall back
        # to the query-param form which the Gamma API always accepts.
        data = await self._poly_gamma_get(f"/events/{event_id}")
        if isinstance(data, dict) and "error" in data:
            data = await self._poly_gamma_get(
                "/events",
                params={"slug": event_id, "limit": "1"},
            )
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_event_detail(data)

    async def poly_market_detail(self, market_id: str) -> dict[str, Any]:
        """Get detailed info on a specific Polymarket market (condition)."""
        if not market_id:
            return {"error": "market_id is required"}
        data = await self._poly_gamma_get(f"/markets/{market_id}")
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_market_detail(data)

    async def poly_categories(self) -> dict[str, Any]:
        """List Polymarket tags/categories."""
        data = await self._poly_gamma_get("/tags")
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_tags(data)

    async def poly_events_by_tag(
        self, tag_id: str, limit: int = 20,
    ) -> dict[str, Any]:
        """Get active events filtered by tag/category."""
        params = {
            "tag_id": tag_id,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": str(limit),
        }
        data = await self._poly_gamma_get("/events", params=params)
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_events(data, top_n=limit)

    async def poly_price_history(
        self,
        token_id: str,
        interval: str = "1m",
        fidelity: int = 60,
    ) -> dict[str, Any]:
        """Get price history for a Polymarket token (sub-market).

        Args:
            token_id: clobTokenId from poly_event_detail output.
            interval:  time range — '1d', '1w', '1m', '6m', '1y', 'max'.
            fidelity:  resolution in minutes (60 = hourly, 1440 = daily).
        """
        if not token_id:
            return {"error": "token_id is required"}
        if interval not in {"1d", "1w", "1m", "6m", "1y", "max"}:
            interval = "1m"
        data = await self._clob_get(
            "/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": str(fidelity)},
        )
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_poly_price_history(data, token_id, interval)

    # ── Kalshi API ────────────────────────────────────────────────────

    async def kalshi_trending(self, limit: int = 20) -> dict[str, Any]:
        """Get top active Kalshi events by aggregated 24h volume.

        Uses /events (not /markets) to avoid being flooded by zero-volume
        MVE combo markets.  Aggregates volume_24h across each event's nested
        markets and sorts client-side.
        """
        params = {
            "status": "open",
            "with_nested_markets": "true",
            "limit": "200",
        }
        data = await self._kalshi_get("/events", params=params)
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_kalshi_events_trending(data, top_n=limit)

    async def kalshi_search(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search Kalshi events by keyword (client-side filter).

        Fetches the same event pool as kalshi_trending (sorted by aggregated
        volume_24h) so high-volume events like "Fed Chair" are always in scope,
        regardless of creation date.
        """
        if not query:
            return {"error": "query is required for search"}
        params = {
            "status": "open",
            "with_nested_markets": "true",
            "limit": "200",
        }
        data = await self._kalshi_get("/events", params=params)
        if isinstance(data, dict) and "error" in data:
            return data
        # Sort events by aggregated 24h volume (same as trending) before
        # keyword filtering — ensures popular markets are never missed because
        # the API returned them outside the first N records.
        events_raw = data.get("events", []) if isinstance(data, dict) else []
        events_raw.sort(
            key=lambda e: sum(
                _num(m.get("volume_24h") or 0) for m in e.get("markets", [])
            ),
            reverse=True,
        )
        return _filter_kalshi_events({"events": events_raw}, query, limit)

    async def kalshi_market_detail(self, ticker: str) -> dict[str, Any]:
        """Get detailed info on a specific Kalshi market."""
        if not ticker:
            return {"error": "ticker is required"}
        data = await self._kalshi_get(f"/markets/{ticker}")
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_kalshi_market_detail(data)

    async def kalshi_event_detail(self, event_ticker: str) -> dict[str, Any]:
        """Get detailed info on a Kalshi event with nested markets."""
        if not event_ticker:
            return {"error": "event_ticker is required"}
        data = await self._kalshi_get(
            f"/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_kalshi_event_detail(data)

    async def kalshi_series(self) -> dict[str, Any]:
        """List Kalshi series (categories)."""
        data = await self._kalshi_get("/series")
        if isinstance(data, dict) and "error" in data:
            return data
        return _normalize_kalshi_series(data)

    # ── Combined convenience methods ──────────────────────────────────

    async def get_trending(self, limit: int = 20) -> dict[str, Any]:
        """Get trending markets from both platforms."""
        poly_result, kalshi_result = await asyncio.gather(
            self.poly_trending(limit),
            self.kalshi_trending(limit),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        if isinstance(poly_result, dict):
            result["polymarket"] = poly_result
        else:
            result["polymarket_error"] = str(poly_result)
        if isinstance(kalshi_result, dict):
            result["kalshi"] = kalshi_result
        else:
            result["kalshi_error"] = str(kalshi_result)
        return result

    async def search_markets(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search both platforms."""
        if not query:
            return {"error": "query is required for search"}
        poly_result, kalshi_result = await asyncio.gather(
            self.poly_search(query, limit),
            self.kalshi_search(query, limit),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        if isinstance(poly_result, dict):
            result["polymarket"] = poly_result
        else:
            result["polymarket_error"] = str(poly_result)
        if isinstance(kalshi_result, dict):
            result["kalshi"] = kalshi_result
        else:
            result["kalshi_error"] = str(kalshi_result)
        return result

    async def compare_markets(self, query: str) -> dict[str, Any]:
        """Cross-platform comparison: search both, fuzzy-match by title."""
        if not query:
            return {"error": "query is required for compare"}
        poly_result, kalshi_result = await asyncio.gather(
            self.poly_search(query, 10),
            self.kalshi_search(query, 10),
            return_exceptions=True,
        )
        return _build_comparison(poly_result, kalshi_result, query)

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity to both APIs."""
        poly_ok, kalshi_ok = False, False
        poly_err, kalshi_err = "", ""

        try:
            client = await self._get_client()
            r = await client.get(
                f"{_POLYMARKET_GAMMA_BASE}/events",
                params={"active": "true", "limit": "1"},
            )
            poly_ok = r.is_success
        except Exception as e:
            poly_err = str(e)

        try:
            client = await self._get_client()
            r = await client.get(
                f"{_KALSHI_BASE}/markets",
                params={"limit": "1"},
            )
            kalshi_ok = r.is_success
        except Exception as e:
            kalshi_err = str(e)

        return {
            "polymarket": {"reachable": poly_ok, "error": poly_err or None},
            "kalshi": {"reachable": kalshi_ok, "error": kalshi_err or None},
        }

    # ── Internal HTTP helpers ─────────────────────────────────────────

    async def _request_with_retry(
        self,
        base_url: str,
        path: str,
        params: dict[str, str] | None,
        rate_log: list[float],
        rate_limit: int,
        label: str,
    ) -> dict[str, Any] | list:
        """GET with rate limiting and exponential-backoff retry."""
        wait = await _check_rate_limit(rate_log, rate_limit)
        if wait is not None:
            return {"error": f"{label} rate limit, retry in {wait:.0f}s", "retry_after": wait}

        last_err: str = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                r = await client.get(f"{base_url}{path}", params=params)
                if not r.is_success:
                    return {"error": f"{label} HTTP {r.status_code}"}
                return r.json()
            except httpx.TimeoutException:
                last_err = f"{label} request timed out"
            except httpx.RequestError as e:
                last_err = f"{label} request failed: {e}"
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.debug(f"{label} retry {attempt + 1}/{_MAX_RETRIES} after {delay:.1f}s")
                await asyncio.sleep(delay)
        return {"error": last_err}

    async def _poly_gamma_get(
        self, path: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list:
        """GET request to Polymarket Gamma API with rate limiting + retry."""
        return await self._request_with_retry(
            _POLYMARKET_GAMMA_BASE, path, params,
            _polymarket_calls, _POLYMARKET_RATE_LIMIT, "Polymarket Gamma",
        )

    async def _clob_get(
        self, path: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list:
        """GET request to Polymarket CLOB API (shares Polymarket rate-limit bucket)."""
        return await self._request_with_retry(
            _POLYMARKET_CLOB_BASE, path, params,
            _polymarket_calls, _POLYMARKET_RATE_LIMIT, "Polymarket CLOB",
        )

    async def _kalshi_get(
        self, path: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list:
        """GET request to Kalshi API with rate limiting + retry."""
        return await self._request_with_retry(
            _KALSHI_BASE, path, params,
            _kalshi_calls, _KALSHI_RATE_LIMIT, "Kalshi",
        )


# ---------------------------------------------------------------------------
# Response normalizers
# ---------------------------------------------------------------------------

_MAX_SUB_MARKETS_INLINE = 3  # keep trending payloads compact


def _normalize_poly_events(data: list | dict, top_n: int = 25) -> dict[str, Any]:
    """Normalize Polymarket events response."""
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data.get("data") or data.get("events") or data.get("results") or []
    else:
        events = []

    events = sorted(
        events,
        key=lambda e: _num(e.get("volume24hr") or e.get("volume24h") or e.get("volume") or 0),
        reverse=True,
    )[:top_n]

    normalized = []
    for e in events:
        slug = e.get("slug", "")
        markets = e.get("markets", [])
        market_count = len(markets)
        outcomes: list[dict] = []
        sub_markets: list[dict] = []

        if market_count == 1:
            # Binary market — show Yes/No outcomes directly
            m = markets[0]
            outcome_names = _parse_json_list(m.get("outcomes"))
            outcome_prices = _parse_json_list(m.get("outcomePrices"))
            for name, price in zip(outcome_names, outcome_prices):
                outcomes.append({"outcome": name, "probability": _num(price)})
        elif market_count > 1:
            # Multi-outcome event (e.g. date-bracketed, named candidates).
            # Summarise each sub-market so callers are not misled by a single
            # near-zero Yes from the first bracket.
            seen_questions: set[str] = set()
            for m in markets:
                question = (m.get("question") or "").strip()
                # Deduplicate by question text (API sometimes repeats entries)
                if question in seen_questions:
                    continue
                seen_questions.add(question)

                outcome_names = _parse_json_list(m.get("outcomes"))
                outcome_prices = _parse_json_list(m.get("outcomePrices"))
                yes_prob: float | None = None
                for name, price in zip(outcome_names, outcome_prices):
                    if (name or "").lower() == "yes":
                        yes_prob = _num(price)
                        break
                if yes_prob is None and outcome_prices:
                    yes_prob = _num(outcome_prices[0])

                # Flag expired / suspended sub-markets instead of silently
                # dropping them so callers know data exists.
                if yes_prob is None or yes_prob == 0.0:
                    logger.debug(f"Sub-market likely expired/resolved: {question!r}")
                    sub_markets.append({
                        "question": question,
                        "yes_probability": yes_prob or 0.0,
                        "status": "expired",
                    })
                    continue

                sub_markets.append({
                    "question": question,
                    "yes_probability": yes_prob,
                    "status": "active",
                })

        entry: dict = {
            "title": e.get("title") or e.get("question") or "",
            "slug": slug,
            "url": f"https://polymarket.com/event/{slug}" if slug else "",
            "volume_24h": _num(e.get("volume24hr") or e.get("volume24h") or e.get("volume") or 0),
            "liquidity": _num(e.get("liquidity") or 0),
            "category": _extract_poly_category(e.get("tags")),
            "end_date": e.get("endDate") or "",
            "market_count": market_count,
            "platform": "polymarket",
        }
        if market_count > 1:
            entry["market_type"] = "multi_outcome"
            # Keep only top active sub-markets by probability to stay compact.
            active = [s for s in sub_markets if s.get("status") == "active"]
            active.sort(key=lambda s: s.get("yes_probability", 0), reverse=True)
            shown = active[:_MAX_SUB_MARKETS_INLINE]
            entry["sub_markets"] = shown
            omitted = len(sub_markets) - len(shown)
            if omitted > 0:
                entry["sub_markets_omitted"] = omitted
        else:
            entry["market_type"] = "binary"
            entry["outcomes"] = outcomes
        normalized.append(entry)

    return {"markets": normalized, "total": len(normalized), "source": "polymarket"}


def _words(query: str) -> list[str]:
    """Split query into lowercase words for AND-style keyword matching."""
    return [w for w in query.lower().split() if len(w) > 2]


def _text_matches(text: str, words: list[str]) -> bool:
    """True if all query words appear in text (case-insensitive)."""
    low = text.lower()
    return all(w in low for w in words)


def _filter_poly_markets(data: list | dict, query: str, limit: int) -> dict[str, Any]:
    """Filter Polymarket /markets response by keyword (word-based AND match).

    /markets records are flat (~2 KB each) so we can afford a larger pool.
    Returns a slim listing for discovery; use market_detail for full outcomes.
    """
    if isinstance(data, list):
        markets = data
    elif isinstance(data, dict):
        markets = data.get("data") or data.get("markets") or data.get("results") or []
    else:
        markets = []

    words = _words(query)
    matched = []
    for m in markets:
        question = (m.get("question") or m.get("title") or "").lower()
        desc = (m.get("description") or "").lower()
        group = (m.get("groupItemTitle") or "").lower()
        if _text_matches(question + " " + desc + " " + group, words):
            matched.append(m)

    # Sort by 24h volume, then cap
    matched.sort(
        key=lambda m: _num(m.get("volume24hr") or m.get("volume24h") or m.get("volume") or 0),
        reverse=True,
    )
    matched = matched[:limit]

    normalized = []
    for m in matched:
        slug = m.get("slug") or m.get("marketSlug") or ""
        outcome_names = _parse_json_list(m.get("outcomes"))
        outcome_prices = _parse_json_list(m.get("outcomePrices"))
        # clobTokenIds may be a JSON-encoded string (events API) or already a list
        clob_token_ids = _parse_json_list(m.get("clobTokenIds"))
        outcomes = [
            {"outcome": n, "probability": _num(p)}
            for n, p in zip(outcome_names, outcome_prices)
        ]
        normalized.append({
            "question": m.get("question") or m.get("title") or "",
            "slug": slug,
            "url": f"https://polymarket.com/market/{slug}" if slug else "",
            "group_title": m.get("groupItemTitle") or "",
            "outcomes": outcomes,
            # Preserve token IDs — needed for price history calls
            "clob_token_ids": clob_token_ids,
            "volume_24h": _num(m.get("volume24hr") or m.get("volume24h") or 0),
            "platform": "polymarket",
        })

    return {"markets": normalized, "total": len(normalized), "source": "polymarket"}


def _filter_poly_events(data: list | dict, query: str, limit: int) -> dict[str, Any]:
    """Filter Polymarket /events response by keyword (used by poly_trending path).

    Returns a slim listing (no sub-market breakdowns) since search is for
    discovery — callers use market_detail to drill into a specific event.
    """
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data.get("data") or data.get("events") or data.get("results") or []
    else:
        events = []

    words = _words(query)
    matched = []
    for e in events:
        title = (e.get("title") or e.get("question") or "").lower()
        desc = (e.get("description") or "").lower()
        if _text_matches(title + " " + desc, words):
            matched.append(e)

    return _normalize_poly_events_slim(matched, top_n=limit)


def _normalize_poly_events_slim(data: list, top_n: int = 20) -> dict[str, Any]:
    """Slim normalizer for search results — event metadata only, no sub-markets.

    Keeps payloads compact for discovery; use market_detail for full outcomes.
    """
    events = sorted(
        data,
        key=lambda e: _num(e.get("volume24hr") or e.get("volume24h") or e.get("volume") or 0),
        reverse=True,
    )[:top_n]

    normalized = []
    for e in events:
        slug = e.get("slug", "")
        market_count = len(e.get("markets", []))
        normalized.append({
            "title": e.get("title") or e.get("question") or "",
            "slug": slug,
            "url": f"https://polymarket.com/event/{slug}" if slug else "",
            "volume_24h": _num(e.get("volume24hr") or e.get("volume24h") or e.get("volume") or 0),
            "market_count": market_count,
            "market_type": "multi_outcome" if market_count > 1 else "binary",
            "platform": "polymarket",
        })

    return {"markets": normalized, "total": len(normalized), "source": "polymarket"}


def _normalize_poly_event_detail(data: dict | list) -> dict[str, Any]:
    """Normalize a single Polymarket event detail."""
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return {"error": "Unexpected response format"}

    slug = data.get("slug", "")
    markets = data.get("markets", [])
    market_details = []
    for m in markets:
        # Gamma API returns outcomes/outcomePrices/clobTokenIds as JSON strings
        outcome_names = _parse_json_list(m.get("outcomes"))
        outcome_prices = _parse_json_list(m.get("outcomePrices"))
        clob_token_ids = _parse_json_list(m.get("clobTokenIds"))
        outcomes = []
        for name, price in zip(outcome_names, outcome_prices):
            outcomes.append({"outcome": name, "probability": _num(price)})
        market_details.append({
            "question": m.get("question") or "",
            "condition_id": m.get("conditionId") or m.get("condition_id") or "",
            "outcomes": outcomes,
            "volume_24h": _num(m.get("volume24hr") or 0),
            "liquidity": _num(m.get("liquidity") or 0),
            # Token IDs needed for price history (index 0 = Yes, index 1 = No)
            "clob_token_ids": clob_token_ids,
        })

    return {
        "title": data.get("title") or "",
        "description": (data.get("description") or "")[:500],
        "slug": slug,
        "url": f"https://polymarket.com/event/{slug}" if slug else "",
        "volume_24h": _num(data.get("volume24hr") or 0),
        "liquidity": _num(data.get("liquidity") or 0),
        "category": _extract_poly_category(data.get("tags")),
        "start_date": data.get("startDate") or "",
        "end_date": data.get("endDate") or "",
        "markets": market_details,
        "platform": "polymarket",
    }


def _normalize_poly_market_detail(data: dict) -> dict[str, Any]:
    """Normalize a single Polymarket market (condition) detail."""
    if not isinstance(data, dict):
        return {"error": "Unexpected response format"}

    outcome_names = _parse_json_list(data.get("outcomes"))
    outcome_prices = _parse_json_list(data.get("outcomePrices"))
    outcomes = []
    for name, price in zip(outcome_names, outcome_prices):
        outcomes.append({"outcome": name, "probability": _num(price)})

    return {
        "question": data.get("question") or "",
        "condition_id": data.get("conditionId") or data.get("condition_id") or "",
        "clob_token_ids": data.get("clobTokenIds") or [],
        "outcomes": outcomes,
        "volume_24h": _num(data.get("volume24hr") or 0),
        "liquidity": _num(data.get("liquidity") or 0),
        "end_date": data.get("endDate") or "",
        "platform": "polymarket",
    }


_MAX_INLINE_POINTS = 30  # enough to show trend shape without blowing up context


def _normalize_poly_price_history(
    data: dict, token_id: str, interval: str,
) -> dict[str, Any]:
    """Normalize Polymarket CLOB /prices-history response.

    Returns pre-computed summary stats so the synthesis LLM never needs to
    crunch hundreds of raw data points.  The ``points`` array is downsampled
    to at most ``_MAX_INLINE_POINTS`` evenly-spaced entries; the full
    timeseries is preserved in the workspace cache file.
    """
    import datetime

    history_raw = data.get("history", []) if isinstance(data, dict) else []
    all_points: list[dict[str, Any]] = []
    for pt in history_raw:
        if not isinstance(pt, dict):
            continue
        t = pt.get("t")
        p = pt.get("p")
        if t is None or p is None:
            continue
        try:
            unix = float(t)
            # Polymarket sometimes returns milliseconds; normalise to seconds
            if unix > 1e10:
                unix /= 1000
            ts = datetime.datetime.utcfromtimestamp(unix).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OSError):
            ts = str(t)
        all_points.append({"timestamp": ts, "probability": round(_num(p), 4)})

    total = len(all_points)

    # ── Summary statistics ────────────────────────────────────────────
    summary: dict[str, Any] = {}
    if all_points:
        probs = [pt["probability"] for pt in all_points]
        min_idx = probs.index(min(probs))
        max_idx = probs.index(max(probs))
        start_p, end_p = probs[0], probs[-1]
        change = round(end_p - start_p, 4)
        change_pct = round((change / start_p) * 100, 1) if start_p else 0.0
        summary = {
            "start": {"timestamp": all_points[0]["timestamp"], "probability": start_p},
            "end": {"timestamp": all_points[-1]["timestamp"], "probability": end_p},
            "min": {"timestamp": all_points[min_idx]["timestamp"], "probability": probs[min_idx]},
            "max": {"timestamp": all_points[max_idx]["timestamp"], "probability": probs[max_idx]},
            "change": change,
            "change_pct": change_pct,
        }

    # ── Downsample for inline context (keep first, last, evenly spaced) ──
    if total <= _MAX_INLINE_POINTS:
        sampled = all_points
    else:
        step = (total - 1) / (_MAX_INLINE_POINTS - 1)
        indices = {0, total - 1}
        indices |= {round(i * step) for i in range(_MAX_INLINE_POINTS)}
        sampled = [all_points[i] for i in sorted(indices)]

    return {
        "token_id": token_id,
        "interval": interval,
        "summary": summary,
        "points": sampled,
        "total_points": total,
        "downsampled": total > _MAX_INLINE_POINTS,
        "platform": "polymarket",
    }


def _normalize_poly_tags(data: list | dict) -> dict[str, Any]:
    """Normalize Polymarket tags/categories."""
    tags = data if isinstance(data, list) else []
    normalized = []
    for t in tags:
        if isinstance(t, dict):
            normalized.append({
                "id": t.get("id") or t.get("slug") or "",
                "label": t.get("label") or t.get("name") or "",
            })
        elif isinstance(t, str):
            normalized.append({"id": t, "label": t})
    return {"categories": normalized, "total": len(normalized)}


# ── Kalshi normalizers ────────────────────────────────────────────────

def _normalize_kalshi_events_trending(data: dict, top_n: int = 20) -> dict[str, Any]:
    """Normalize Kalshi /events response into a trending list.

    Aggregates volume_24h across each event's nested markets, sorts by that
    total descending, and returns the top N events with their top sub-market.
    """
    events_raw = data.get("events", []) if isinstance(data, dict) else []

    scored: list[tuple[float, dict]] = []
    for e in events_raw:
        nested = e.get("markets", [])
        total_vol_24h = sum(_num(m.get("volume_24h") or 0) for m in nested)
        total_vol = sum(_num(m.get("volume") or 0) for m in nested)
        total_oi = sum(_num(m.get("open_interest") or 0) for m in nested)

        # Pick the frontrunner (highest yes probability) as the representative.
        # Using volume would pick the most-traded candidate, which on Kalshi
        # bulk responses often differs from the individual-event fetch due to
        # stale/partial volume data in the aggregated response.
        def _yes_price(m: dict) -> float:
            p = _num(m.get("yes_ask_dollars") or m.get("last_price_dollars") or 0)
            if p == 0:
                raw = _num(m.get("yes_ask") or m.get("last_price") or 0)
                p = raw / 100.0 if raw != 0 else 0.0
            return p

        best = max(nested, key=_yes_price) if nested else {}
        yes_price = _yes_price(best)

        scored.append((total_vol_24h, {
            "title": e.get("title") or "",
            "event_ticker": e.get("event_ticker") or e.get("ticker") or "",
            "category": e.get("category") or "",
            "url": f"https://kalshi.com/events/{e.get('event_ticker', '')}",
            "top_market_title": best.get("title") or "",
            "top_market_ticker": best.get("ticker") or "",
            "top_market_probability": round(yes_price, 4),
            "volume_24h": total_vol_24h,
            "volume": total_vol,
            "open_interest": total_oi,
            "market_count": len(nested),
            "platform": "kalshi",
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    normalized = [entry for _, entry in scored[:top_n]]
    return {"markets": normalized, "total": len(normalized), "source": "kalshi"}


def _normalize_kalshi_markets(data: dict, top_n: int = 20) -> dict[str, Any]:
    """Normalize Kalshi markets response, sorted by volume."""
    markets_raw = data.get("markets", []) if isinstance(data, dict) else []

    def _vol(m: dict) -> float:
        return _num(m.get("volume") or m.get("volume_24h") or 0)

    markets_raw = sorted(markets_raw, key=_vol, reverse=True)

    normalized = []
    for m in markets_raw[:top_n]:
        # Prefer _dollars fields (already 0-1 scale); fall back to raw
        # cent-scale integer fields (0-100) which always need / 100.
        yes_price = _num(m.get("yes_ask_dollars") or m.get("last_price_dollars") or 0)
        if yes_price == 0:
            raw = _num(m.get("yes_ask") or m.get("last_price") or 0)
            yes_price = raw / 100.0 if raw != 0 else 0.0

        normalized.append({
            "title": m.get("title") or "",
            "ticker": m.get("ticker") or "",
            "event_ticker": m.get("event_ticker") or "",
            "url": f"https://kalshi.com/markets/{m.get('ticker', '')}",
            "yes_probability": round(yes_price, 4),
            "volume": _num(m.get("volume") or 0),
            "volume_24h": _num(m.get("volume_24h") or 0),
            "open_interest": _num(m.get("open_interest") or 0),
            "status": m.get("status") or "",
            "close_time": m.get("close_time") or m.get("expiration_time") or "",
            "platform": "kalshi",
        })

    return {"markets": normalized, "total": len(normalized), "source": "kalshi"}


_MAX_KALSHI_SUB_MARKETS = 10  # cap nested markets per event in search results


def _filter_kalshi_events(
    data: dict, query: str, limit: int,
) -> dict[str, Any]:
    """Filter Kalshi events by keyword match.

    Returns matched events with their top sub-markets (capped to keep payload
    compact).  Sub-markets are sorted by volume descending.
    """
    events = data.get("events", []) if isinstance(data, dict) else []
    matched_markets: list[dict] = []

    words = _words(query)
    for e in events:
        title = (e.get("title") or "").lower()
        sub_title = (e.get("sub_title") or "").lower()
        category = (e.get("category") or "").lower()
        if _text_matches(title + " " + sub_title + " " + category, words):
            nested = e.get("markets", [])
            if nested:
                # Sort by volume desc and cap to avoid payload blowup
                nested_sorted = sorted(
                    nested,
                    key=lambda m: _num(m.get("volume") or m.get("volume_24h") or 0),
                    reverse=True,
                )
                for m in nested_sorted[:_MAX_KALSHI_SUB_MARKETS]:
                    m.setdefault("event_title", e.get("title", ""))
                    matched_markets.append(m)
            else:
                matched_markets.append({
                    "title": e.get("title") or "",
                    "ticker": e.get("ticker") or "",
                    "event_ticker": e.get("event_ticker") or "",
                    "status": "open",
                })

    return _normalize_kalshi_markets(
        {"markets": matched_markets}, top_n=limit,
    )


def _normalize_kalshi_market_detail(data: dict) -> dict[str, Any]:
    """Normalize a single Kalshi market detail."""
    m = data.get("market", data) if isinstance(data, dict) else {}
    if not isinstance(m, dict):
        return {"error": "Unexpected response format"}

    # Prefer _dollars fields (already 0-1 scale)
    yes_price = _num(m.get("yes_ask_dollars") or m.get("last_price_dollars") or 0)
    if yes_price == 0:
        raw = _num(m.get("yes_ask") or m.get("last_price") or 0)
        yes_price = raw / 100.0 if raw > 1 else raw

    return {
        "title": m.get("title") or "",
        "ticker": m.get("ticker") or "",
        "event_ticker": m.get("event_ticker") or "",
        "subtitle": m.get("subtitle") or "",
        "url": f"https://kalshi.com/markets/{m.get('ticker', '')}",
        "yes_probability": round(yes_price, 4),
        "yes_ask": _num(m.get("yes_ask_dollars") or 0),
        "yes_bid": _num(m.get("yes_bid_dollars") or 0),
        "no_ask": _num(m.get("no_ask_dollars") or 0),
        "no_bid": _num(m.get("no_bid_dollars") or 0),
        "volume": _num(m.get("volume") or 0),
        "volume_24h": _num(m.get("volume_24h") or 0),
        "open_interest": _num(m.get("open_interest") or 0),
        "status": m.get("status") or "",
        "close_time": m.get("close_time") or m.get("expiration_time") or "",
        "result": m.get("result") or "",
        "platform": "kalshi",
    }


def _normalize_kalshi_event_detail(data: dict) -> dict[str, Any]:
    """Normalize a Kalshi event with nested markets."""
    e = data.get("event", data) if isinstance(data, dict) else {}
    if not isinstance(e, dict):
        return {"error": "Unexpected response format"}

    nested = e.get("markets", [])
    markets = []
    for m in nested:
        yes_price = _num(m.get("yes_ask_dollars") or m.get("last_price_dollars") or 0)
        if yes_price == 0:
            raw = _num(m.get("yes_ask") or m.get("last_price") or 0)
            yes_price = raw / 100.0 if raw > 1 else raw
        markets.append({
            "title": m.get("title") or "",
            "ticker": m.get("ticker") or "",
            "yes_probability": round(yes_price, 4),
            "volume": _num(m.get("volume") or 0),
            "status": m.get("status") or "",
        })

    return {
        "title": e.get("title") or "",
        "event_ticker": e.get("event_ticker") or e.get("ticker") or "",
        "category": e.get("category") or "",
        "sub_title": e.get("sub_title") or "",
        "markets": markets,
        "platform": "kalshi",
    }


def _normalize_kalshi_series(data: dict) -> dict[str, Any]:
    """Normalize Kalshi series (categories)."""
    series_list = data.get("series", []) if isinstance(data, dict) else []
    normalized = []
    for s in series_list:
        if isinstance(s, dict):
            normalized.append({
                "ticker": s.get("ticker") or "",
                "title": s.get("title") or "",
                "category": s.get("category") or "",
            })
    return {"categories": normalized, "total": len(normalized)}


# ── Cross-platform comparison ─────────────────────────────────────────

def _build_comparison(
    poly_result: dict | Exception,
    kalshi_result: dict | Exception,
    query: str,
) -> dict[str, Any]:
    """Build cross-platform comparison using fuzzy title matching."""
    result: dict[str, Any] = {"query": query, "matches": []}

    poly_markets: list[dict] = []
    kalshi_markets: list[dict] = []

    if isinstance(poly_result, dict) and "markets" in poly_result:
        poly_markets = poly_result["markets"]
    elif isinstance(poly_result, dict) and "error" in poly_result:
        result["polymarket_error"] = poly_result["error"]
    elif isinstance(poly_result, Exception):
        result["polymarket_error"] = str(poly_result)

    if isinstance(kalshi_result, dict) and "markets" in kalshi_result:
        kalshi_markets = kalshi_result["markets"]
    elif isinstance(kalshi_result, dict) and "error" in kalshi_result:
        result["kalshi_error"] = kalshi_result["error"]
    elif isinstance(kalshi_result, Exception):
        result["kalshi_error"] = str(kalshi_result)

    matched_kalshi_titles: set[str] = set()

    for pm in poly_markets:
        pm_title = (pm.get("title") or "").lower()
        best_match = None
        best_ratio = 0.0
        for km in kalshi_markets:
            km_title = (km.get("title") or "").lower()
            ratio = difflib.SequenceMatcher(None, pm_title, km_title).ratio()
            if ratio > best_ratio and ratio > _COMPARE_MATCH_THRESHOLD:
                best_ratio = ratio
                best_match = km

        if best_match:
            matched_kalshi_titles.add(best_match.get("title", ""))
            # Extract Polymarket probability.
            # For multi-outcome events use the highest sub-market Yes probability
            # as the representative value (avoids showing 0.0 from one bracket).
            poly_prob = None
            if pm.get("market_type") == "multi_outcome":
                sub_probs = [
                    sm.get("yes_probability") or 0.0
                    for sm in (pm.get("sub_markets") or [])
                    if sm.get("yes_probability") is not None
                ]
                if sub_probs:
                    poly_prob = max(sub_probs)
            else:
                outcomes = pm.get("outcomes", [])
                if outcomes:
                    for o in outcomes:
                        if (o.get("outcome") or "").lower() == "yes":
                            poly_prob = o.get("probability")
                            break
                    if poly_prob is None:
                        poly_prob = outcomes[0].get("probability")

            result["matches"].append({
                "topic": pm.get("title", ""),
                "match_confidence": round(best_ratio, 2),
                "polymarket": {
                    "title": pm.get("title"),
                    "probability": poly_prob,
                    "volume_24h": pm.get("volume_24h"),
                    "url": pm.get("url"),
                },
                "kalshi": {
                    "title": best_match.get("title"),
                    "yes_probability": best_match.get("yes_probability"),
                    "volume": best_match.get("volume"),
                    "url": best_match.get("url"),
                },
            })

    result["polymarket_only"] = [
        m for m in poly_markets
        if m.get("title") not in {
            match["polymarket"]["title"] for match in result["matches"]
        }
    ]
    result["kalshi_only"] = [
        m for m in kalshi_markets
        if m.get("title") not in matched_kalshi_titles
    ]

    return result
