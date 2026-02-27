"""Prediction Market Tool — Polymarket + Kalshi command dispatch.

Supported commands:
  trending        — top markets by volume across both platforms
  search          — search markets by keyword
  market_detail   — detailed info on a specific market (outcomes/odds)
  history         — price history for a token (requires token_id)
  market_history  — composite: market_detail + history in one call (by slug)
  top_mover       — composite: trending → pick #1 → market_detail + history
  compare         — cross-platform odds comparison for equivalent events
  categories      — list market categories / filter by category
  status          — API health check
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool

# Lazy-initialized singleton (guarded by asyncio.Lock for async safety)
_data: Any = None
_data_lock = asyncio.Lock()


async def _get_data() -> Any:
    """Lazy-init PredictionMarketDataTool singleton (async-safe)."""
    global _data
    if _data is None:
        async with _data_lock:
            if _data is None:
                from finclaw.agent.financial_tools.prediction_market.prediction_market_data_tool import (
                    PredictionMarketDataTool,
                )
                _data = PredictionMarketDataTool()
    return _data


class PredictionMarketTool(Tool):
    """Query prediction markets (Polymarket + Kalshi)."""

    @property
    def name(self) -> str:
        return "prediction_market_query"

    @property
    def description(self) -> str:
        return (
            "Query prediction markets on Polymarket and Kalshi. "
            "Commands: trending (top markets by volume), search (find markets by keyword), "
            "market_detail (specific market outcomes/current odds), "
            "compare (cross-platform odds comparison), categories (list/filter), "
            "status (API health check)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "trending", "search", "market_detail",
                        "history", "market_history", "top_mover",
                        "compare", "categories", "status",
                    ],
                    "description": (
                        "trending: top markets by 24h volume across both platforms; "
                        "search: find markets by keyword (use for discovery, NOT for known slugs); "
                        "market_detail: current outcomes/odds for a specific market by slug or ticker; "
                        "history: price history for a token_id (low-level, prefer market_history); "
                        "market_history: get market detail + price history in one call — "
                        "pass market_id (slug/ticker), returns detail + history combined; "
                        "top_mover: find the #1 trending market and return its detail + history; "
                        "compare: cross-platform odds comparison; "
                        "categories: list categories or filter; "
                        "status: API health check."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Search term or topic for search/compare commands.",
                },
                "market_id": {
                    "type": "string",
                    "description": (
                        "Market/event identifier. For Polymarket: event slug or ID. "
                        "For Kalshi: market ticker. Prefix with 'poly:' or 'kalshi:' "
                        "to specify platform (e.g. 'poly:fed-decision-in-october', "
                        "'kalshi:KXHIGHNY-26FEB28'). Without prefix, tries both."
                    ),
                },
                "token_id": {
                    "type": "string",
                    "description": (
                        "Polymarket CLOB token ID for the history command. "
                        "Obtained from clob_token_ids in a market_detail response. "
                        "Each binary sub-market has two tokens: index 0 = Yes, index 1 = No."
                    ),
                },
                "interval": {
                    "type": "string",
                    "enum": ["1d", "1w", "1m", "6m", "1y", "max"],
                    "description": "Time range for history command. Default: 1m (one month).",
                },
                "fidelity": {
                    "type": "integer",
                    "description": (
                        "Resolution in minutes for history command "
                        "(60 = hourly, 1440 = daily). Default: 60."
                    ),
                    "minimum": 1,
                },
                "platform": {
                    "type": "string",
                    "enum": ["polymarket", "kalshi", "both"],
                    "description": "Target platform. Default: both.",
                },
                "category": {
                    "type": "string",
                    "description": "Category/tag ID for filtering (categories command).",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        logger.info(f"prediction_market:{command}")

        data = await _get_data()

        try:
            if command == "trending":
                result = await self._trending(data, kwargs)
            elif command == "search":
                result = await self._search(data, kwargs)
            elif command == "market_detail":
                result = await self._market_detail(data, kwargs)
            elif command == "history":
                result = await self._history(data, kwargs)
            elif command == "market_history":
                result = await self._market_history(data, kwargs)
            elif command == "top_mover":
                result = await self._top_mover(data, kwargs)
            elif command == "compare":
                result = await self._compare(data, kwargs)
            elif command == "categories":
                result = await self._categories(data, kwargs)
            elif command == "status":
                result = await data.health_check()
            else:
                result = {"error": f"Unknown command: {command!r}"}
        except Exception as exc:
            logger.warning(f"prediction_market error ({command}): {exc}")
            result = {"error": str(exc)}

        return json.dumps(result, ensure_ascii=False, default=str)

    # ── Command implementations ───────────────────────────────────────

    async def _trending(self, data: Any, kwargs: dict) -> dict:
        platform = kwargs.get("platform", "both")

        if platform == "polymarket":
            return await data.poly_trending(5)
        elif platform == "kalshi":
            return await data.kalshi_trending(5)
        else:
            return await data.get_trending(5)

    async def _search(self, data: Any, kwargs: dict) -> dict:
        query = kwargs.get("query", "")
        if not query:
            return {"error": "query is required for search command"}
        platform = kwargs.get("platform", "both")

        if platform == "polymarket":
            return await data.poly_search(query, 5)
        elif platform == "kalshi":
            return await data.kalshi_search(query, 5)
        else:
            return await data.search_markets(query, 5)

    async def _market_detail(self, data: Any, kwargs: dict) -> dict:
        market_id = kwargs.get("market_id", "")
        if not market_id:
            return {"error": "market_id is required for market_detail command"}

        # Parse platform prefix
        if market_id.startswith("poly:"):
            return await data.poly_event_detail(market_id[5:])
        elif market_id.startswith("kalshi:"):
            return await self._kalshi_detail_with_fallback(data, market_id[7:])
        else:
            platform = kwargs.get("platform", "both")
            if platform == "polymarket":
                return await data.poly_event_detail(market_id)
            elif platform == "kalshi":
                return await self._kalshi_detail_with_fallback(data, market_id)
            else:
                # Try Polymarket first (slug-based), then Kalshi (ticker-based)
                result = await data.poly_event_detail(market_id)
                if isinstance(result, dict) and "error" in result:
                    result = await self._kalshi_detail_with_fallback(data, market_id)
                return result

    @staticmethod
    async def _kalshi_detail_with_fallback(data: Any, ticker: str) -> dict:
        """Try /markets/{ticker} first; if 404, try /events/{ticker}."""
        result = await data.kalshi_market_detail(ticker)
        if isinstance(result, dict) and "error" in result:
            result = await data.kalshi_event_detail(ticker)
        return result

    async def _history(self, data: Any, kwargs: dict) -> dict:
        token_id = kwargs.get("token_id", "")
        if not token_id:
            return {
                "error": (
                    "token_id is required for history command. "
                    "Use market_detail first to get clob_token_ids for the sub-market "
                    "(index 0 = Yes token, index 1 = No token)."
                )
            }
        interval = kwargs.get("interval", "1m")
        fidelity = int(kwargs.get("fidelity", 60))
        return await data.poly_price_history(token_id, interval, fidelity)

    async def _market_history(self, data: Any, kwargs: dict) -> dict:
        """Composite: market_detail + history for the first sub-market in one call.

        For Polymarket "ladder" events (e.g. "US strikes Iran by Feb 28, 2026"),
        the per-rung event detail often lacks clob_token_ids because the rung is
        a sub-market of the parent event rather than a standalone event.  When
        that happens we fall back to a keyword search via the /markets endpoint
        (poly_search), which returns flat market records that do carry token IDs.
        """
        market_id = kwargs.get("market_id", "")
        if not market_id:
            return {"error": "market_id is required for market_history command"}

        detail = await self._market_detail(data, kwargs)
        if isinstance(detail, dict) and "error" in detail:
            return detail

        # ── Primary: extract token from event detail ─────────────────────────
        token_id = self._extract_first_token(detail)

        # ── Fallback: search via /markets endpoint for ladder-rung markets ───
        if not token_id:
            search_query = self._derive_search_query(detail, market_id)
            logger.debug(
                f"market_history: no token in event detail, searching '{search_query}'"
            )
            search_result = await data.poly_search(search_query, limit=10)
            token_id = self._extract_token_from_markets(search_result)

        if not token_id:
            return {
                "detail": detail,
                "history_error": (
                    "No clob_token_ids found in market detail or search results. "
                    "Try fetching the parent ladder event or a different platform."
                ),
            }

        interval = kwargs.get("interval", "1w")
        fidelity = int(kwargs.get("fidelity", 60))
        history = await data.poly_price_history(token_id, interval, fidelity)

        return {"detail": detail, "history": history}

    @staticmethod
    def _derive_search_query(detail: dict, market_id: str) -> str:
        """Derive a keyword search query from market detail or fall back to slug."""
        title = detail.get("title", "")
        if title:
            return title
        # Convert slug to words: "us-strikes-iran-by-feb-28" → "us strikes iran feb 28"
        return market_id.removeprefix("poly:").replace("-", " ")

    @staticmethod
    def _extract_token_from_markets(search_result: dict) -> str:
        """Extract the first Yes clob_token_id from a poly_search result."""
        for m in search_result.get("markets", []):
            ids = m.get("clob_token_ids", [])
            if ids:
                return ids[0]  # index 0 = Yes token
        return ""

    async def _top_mover(self, data: Any, kwargs: dict) -> dict:
        """Composite: trending → pick #1 by volume → market_detail + history."""
        interval = kwargs.get("interval", "1w")
        fidelity = int(kwargs.get("fidelity", 60))

        # Step 1: get trending (Polymarket only — Kalshi has no CLOB token history)
        trending = await data.poly_trending(limit=5)
        if isinstance(trending, dict) and "error" in trending:
            return trending

        markets = trending.get("markets", [])
        if not markets:
            return {"error": "No trending markets found"}

        top = markets[0]
        slug = top.get("slug", "")
        if not slug:
            return {"trending_top": top, "history_error": "Top market has no slug"}

        # Step 2: get detail for the top market
        detail = await data.poly_event_detail(slug)
        if isinstance(detail, dict) and "error" in detail:
            return {"trending_top": top, "detail_error": detail["error"]}

        # Step 3: get history for the first sub-market Yes token
        token_id = self._extract_first_token(detail)
        if not token_id:
            return {
                "trending_top": top,
                "detail": detail,
                "history_error": "No clob_token_ids found — history unavailable.",
            }

        history = await data.poly_price_history(token_id, interval, fidelity)
        return {"trending_top": top, "detail": detail, "history": history}

    @staticmethod
    def _extract_first_token(detail: dict) -> str:
        """Extract the first Yes clob_token_id from a market_detail response."""
        for m in detail.get("markets", []):
            ids = m.get("clob_token_ids", [])
            if ids:
                return ids[0]  # index 0 = Yes token
        return ""

    async def _compare(self, data: Any, kwargs: dict) -> dict:
        query = kwargs.get("query", "")
        if not query:
            return {"error": "query is required for compare command"}
        return await data.compare_markets(query)

    async def _categories(self, data: Any, kwargs: dict) -> dict:
        platform = kwargs.get("platform", "both")
        category = kwargs.get("category", "")

        if category and platform in ("polymarket", "both"):
            return await data.poly_events_by_tag(category, 5)

        if platform == "polymarket":
            return await data.poly_categories()
        elif platform == "kalshi":
            return await data.kalshi_series()
        else:
            poly_cats, kalshi_cats = await asyncio.gather(
                data.poly_categories(),
                data.kalshi_series(),
                return_exceptions=True,
            )
            result: dict[str, Any] = {}
            if isinstance(poly_cats, dict):
                result["polymarket"] = poly_cats
            else:
                result["polymarket_error"] = str(poly_cats)
            if isinstance(kalshi_cats, dict):
                result["kalshi"] = kalshi_cats
            else:
                result["kalshi_error"] = str(kalshi_cats)
            return result
