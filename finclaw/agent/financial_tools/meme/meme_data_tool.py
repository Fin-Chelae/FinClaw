"""Meme coin data fetcher — DexScreener + CoinGecko APIs.

No API keys required for basic usage. CoinGecko Pro key is optional
for higher rate limits (configure via tools.memeMonitor.coingeckoApiKey
in ~/.finclaw/config.json or COINGECKO_API_KEY env var).

DexScreener: 300 req/min, no key needed.
CoinGecko:   30 req/min free tier, no key needed.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEXSCREENER_BASE = "https://api.dexscreener.com"
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
_POLYMARKET_API = "https://gamma-api.polymarket.com/events"

_DEX_RATE_LIMIT = 300       # per minute
_CG_RATE_LIMIT_FREE = 30    # per minute
_CG_RATE_LIMIT_PRO = 500    # per minute
_HTTP_TIMEOUT = 15.0

# ---------------------------------------------------------------------------
# Module-level rate-limit tracking
# ---------------------------------------------------------------------------

_dexscreener_calls: list[float] = []
_coingecko_calls: list[float] = []


def _prune_rate_log(log: list[float]) -> None:
    """Remove entries older than 60 seconds."""
    cutoff = time.time() - 60
    while log and log[0] < cutoff:
        log.pop(0)


def _check_rate_limit(log: list[float], limit: int) -> float | None:
    """Return seconds to wait if at limit, else None."""
    _prune_rate_log(log)
    if len(log) >= limit:
        wait = 60 - (time.time() - log[0])
        return max(0.1, wait)
    return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_coingecko_api_key() -> str:
    """Resolve CoinGecko API key: config.json first, then env var."""
    try:
        from finclaw.config.loader import load_config
        key = load_config().tools.meme_monitor.coingecko_api_key
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("COINGECKO_API_KEY", "")


# ---------------------------------------------------------------------------
# MemeDataTool
# ---------------------------------------------------------------------------

class MemeDataTool:
    """Fetch meme coin data from DexScreener and CoinGecko.

    This is NOT a Tool subclass — it is used internally by MemeRouter
    and never exposed to the LLM directly.
    """

    def __init__(self) -> None:
        self._cg_key = _get_coingecko_api_key()
        self._cg_base = _COINGECKO_PRO_BASE if self._cg_key else _COINGECKO_BASE
        self._cg_limit = _CG_RATE_LIMIT_PRO if self._cg_key else _CG_RATE_LIMIT_FREE
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT, follow_redirects=True,
            )
        return self._client

    # ── DexScreener ──────────────────────────────────────────────────

    async def dex_search(self, query: str) -> dict[str, Any]:
        """Search DexScreener for token pairs by name/symbol."""
        data = await self._dex_get(f"/latest/dex/search", params={"q": query})
        return _normalize_dex_pairs(data)

    async def dex_trending(self) -> dict[str, Any]:
        """Get boosted (trending) tokens from DexScreener."""
        latest, top = await asyncio.gather(
            self._dex_get("/token-boosts/latest/v1"),
            self._dex_get("/token-boosts/top/v1"),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        if isinstance(latest, Exception):
            result["latest_boosts_error"] = str(latest)
        elif isinstance(latest, dict) and "error" in latest:
            result["latest_boosts_error"] = latest["error"]
        else:
            result["latest_boosts"] = _normalize_boost_tokens(latest)
        if isinstance(top, Exception):
            result["top_boosts_error"] = str(top)
        elif isinstance(top, dict) and "error" in top:
            result["top_boosts_error"] = top["error"]
        else:
            result["top_boosts"] = _normalize_boost_tokens(top)
        return result

    async def dex_token_pairs(
        self, chain_id: str, token_address: str,
    ) -> dict[str, Any]:
        """Get all pairs for a specific token on a chain."""
        data = await self._dex_get(
            f"/token-pairs/v1/{chain_id}/{token_address}",
        )
        return _normalize_dex_pairs(data)

    # ── CoinGecko ────────────────────────────────────────────────────

    async def cg_search(self, query: str) -> dict[str, Any]:
        """Search CoinGecko for coins by name/symbol."""
        data = await self._cg_get("/search", params={"query": query})
        coins = data.get("coins", [])
        return {
            "coins": [
                {
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "symbol": c.get("symbol", ""),
                    "market_cap_rank": c.get("market_cap_rank"),
                    "thumb": c.get("thumb", ""),
                }
                for c in coins[:20]
            ],
        }

    async def cg_trending(self) -> dict[str, Any]:
        """Get trending coins from CoinGecko (top 7)."""
        data = await self._cg_get("/search/trending")
        coins_raw = data.get("coins", [])
        return {
            "coins": [
                {
                    "id": c.get("item", {}).get("id", ""),
                    "name": c.get("item", {}).get("name", ""),
                    "symbol": c.get("item", {}).get("symbol", ""),
                    "market_cap_rank": c.get("item", {}).get("market_cap_rank"),
                    "score": c.get("item", {}).get("score"),
                }
                for c in coins_raw
            ],
        }

    async def cg_price(self, coin_ids: list[str]) -> dict[str, Any]:
        """Get prices for CoinGecko coin IDs (max 50)."""
        if not coin_ids:
            return {"error": "coin_ids is empty"}
        ids_str = ",".join(coin_ids[:50])
        return await self._cg_get(
            "/simple/price",
            params={
                "ids": ids_str,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
        )

    async def cg_coin_info(self, coin_id: str) -> dict[str, Any]:
        """Get detailed coin info from CoinGecko."""
        data = await self._cg_get(
            f"/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "true",
                "developer_data": "false",
            },
        )
        if "error" in data:
            return data
        # Extract key fields to keep response concise
        md = data.get("market_data", {})
        return {
            "id": data.get("id", ""),
            "symbol": data.get("symbol", ""),
            "name": data.get("name", ""),
            "description": (data.get("description", {}).get("en", "") or "")[:500],
            "image": data.get("image", {}).get("small", ""),
            "market_data": {
                "current_price_usd": md.get("current_price", {}).get("usd"),
                "market_cap_usd": md.get("market_cap", {}).get("usd"),
                "market_cap_rank": md.get("market_cap_rank"),
                "total_volume_usd": md.get("total_volume", {}).get("usd"),
                "price_change_24h_pct": md.get("price_change_percentage_24h"),
                "price_change_7d_pct": md.get("price_change_percentage_7d"),
                "price_change_30d_pct": md.get("price_change_percentage_30d"),
                "ath_usd": md.get("ath", {}).get("usd"),
                "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
                "fully_diluted_valuation_usd": md.get("fully_diluted_valuation", {}).get("usd"),
                "circulating_supply": md.get("circulating_supply"),
                "total_supply": md.get("total_supply"),
                "max_supply": md.get("max_supply"),
            },
            "community": {
                "twitter_followers": data.get("community_data", {}).get("twitter_followers"),
                "reddit_subscribers": data.get("community_data", {}).get("reddit_subscribers"),
            },
            "links": {
                "homepage": (data.get("links", {}).get("homepage", [None]) or [None])[0],
                "twitter": data.get("links", {}).get("twitter_screen_name", ""),
                "subreddit": data.get("links", {}).get("subreddit_url", ""),
            },
            "contract_address": data.get("contract_address", ""),
            "platforms": data.get("platforms", {}),
            "categories": data.get("categories", []),
        }

    # ── Combined convenience methods ─────────────────────────────────

    async def search_token(self, query: str) -> dict[str, Any]:
        """Search both DexScreener and CoinGecko, merge results."""
        dex_result, cg_result = await asyncio.gather(
            self.dex_search(query),
            self.cg_search(query),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        if isinstance(dex_result, dict):
            result["dexscreener"] = dex_result
        else:
            result["dexscreener_error"] = str(dex_result)
        if isinstance(cg_result, dict):
            result["coingecko"] = cg_result
        else:
            result["coingecko_error"] = str(cg_result)
        return result

    async def get_trending(self) -> dict[str, Any]:
        """Get trending from both DexScreener and CoinGecko."""
        dex_result, cg_result = await asyncio.gather(
            self.dex_trending(),
            self.cg_trending(),
            return_exceptions=True,
        )
        result: dict[str, Any] = {}
        if isinstance(dex_result, dict):
            result["dexscreener"] = dex_result
        else:
            result["dexscreener_error"] = str(dex_result)
        if isinstance(cg_result, dict):
            result["coingecko"] = cg_result
        else:
            result["coingecko_error"] = str(cg_result)
        return result

    async def polymarket_trending(self, top_n: int = 20) -> dict[str, Any]:
        """Get top Polymarket prediction markets ranked by 24h volume."""
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(max(top_n * 2, 50)),
            "offset": "0",
        }
        try:
            client = await self._get_client()
            r = await client.get(_POLYMARKET_API, params=params, timeout=25.0)
            if not r.is_success:
                return {"error": f"Polymarket API HTTP {r.status_code}"}
            data = r.json()
        except httpx.TimeoutException:
            return {"error": "Polymarket request timed out"}
        except httpx.RequestError as e:
            return {"error": f"Polymarket request failed: {e}"}

        # Normalise: API may return a list or a dict with a data/markets key
        if isinstance(data, list):
            markets_raw = data
        elif isinstance(data, dict):
            markets_raw = (
                data.get("data") or data.get("markets") or data.get("results") or []
            )
        else:
            return {"error": f"Unexpected Polymarket response type: {type(data)}"}

        def _num(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        markets_raw = sorted(
            markets_raw,
            key=lambda m: _num(m.get("volume24hr") or m.get("volume24h") or m.get("volume") or 0),
            reverse=True,
        )[:top_n]

        markets = []
        for m in markets_raw:
            slug = m.get("slug")
            markets.append({
                "question": m.get("question") or m.get("title") or "",
                "url": f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com",
                "volume_24h": _num(m.get("volume24hr") or m.get("volume24h") or m.get("volume") or 0),
                "liquidity": _num(m.get("liquidity") or 0),
                "category": m.get("category") or "",
                "end_date": m.get("endDate") or "",
            })

        return {"markets": markets, "total": len(markets), "source": "polymarket"}

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity to both APIs."""
        dex_ok, cg_ok = False, False
        dex_err, cg_err = "", ""

        try:
            client = await self._get_client()
            r = await client.get(f"{_DEXSCREENER_BASE}/latest/dex/search", params={"q": "test"})
            dex_ok = r.is_success
        except Exception as e:
            dex_err = str(e)

        try:
            headers = {}
            if self._cg_key:
                headers["x-cg-pro-api-key"] = self._cg_key
            client = await self._get_client()
            r = await client.get(f"{self._cg_base}/ping", headers=headers)
            cg_ok = r.is_success
        except Exception as e:
            cg_err = str(e)

        return {
            "dexscreener": {"reachable": dex_ok, "error": dex_err or None},
            "coingecko": {
                "reachable": cg_ok,
                "error": cg_err or None,
                "has_api_key": bool(self._cg_key),
                "rate_limit": self._cg_limit,
            },
        }

    # ── Internal HTTP helpers ────────────────────────────────────────

    async def _dex_get(
        self, path: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list:
        """GET request to DexScreener with rate limiting."""
        wait = _check_rate_limit(_dexscreener_calls, _DEX_RATE_LIMIT)
        if wait is not None:
            return {"error": f"DexScreener rate limit reached, retry in {wait:.0f}s", "retry_after": wait}
        _dexscreener_calls.append(time.time())

        try:
            client = await self._get_client()
            r = await client.get(f"{_DEXSCREENER_BASE}{path}", params=params)
            if not r.is_success:
                return {"error": f"DexScreener HTTP {r.status_code}"}
            return r.json()
        except httpx.TimeoutException:
            return {"error": "DexScreener request timed out"}
        except httpx.RequestError as e:
            return {"error": f"DexScreener request failed: {e}"}

    async def _cg_get(
        self, path: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """GET request to CoinGecko with rate limiting and optional API key."""
        wait = _check_rate_limit(_coingecko_calls, self._cg_limit)
        if wait is not None:
            return {"error": f"CoinGecko rate limit reached, retry in {wait:.0f}s", "retry_after": wait}
        _coingecko_calls.append(time.time())

        headers: dict[str, str] = {}
        if self._cg_key:
            headers["x-cg-pro-api-key"] = self._cg_key

        try:
            client = await self._get_client()
            r = await client.get(
                f"{self._cg_base}{path}", params=params, headers=headers,
            )
            if not r.is_success:
                return {"error": f"CoinGecko HTTP {r.status_code}"}
            return r.json()
        except httpx.TimeoutException:
            return {"error": "CoinGecko request timed out"}
        except httpx.RequestError as e:
            return {"error": f"CoinGecko request failed: {e}"}


# ---------------------------------------------------------------------------
# Response normalizers
# ---------------------------------------------------------------------------

def _normalize_boost_tokens(data: list | dict) -> list[dict[str, Any]]:
    """Normalize DexScreener boost/trending response (returns a JSON array)."""
    tokens = data if isinstance(data, list) else []
    return [
        {
            "chain": t.get("chainId", ""),
            "token_address": t.get("tokenAddress", ""),
            "url": t.get("url", ""),
            "description": (t.get("description") or "")[:200],
            "total_amount": t.get("totalAmount"),
        }
        for t in tokens[:25]
    ]


def _normalize_dex_pairs(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize DexScreener pair response to consistent format."""
    if "error" in data:
        return data
    pairs_raw = data.get("pairs") or []
    pairs = []
    for p in pairs_raw[:25]:  # cap at 25 results
        base = p.get("baseToken", {})
        quote = p.get("quoteToken", {})
        vol = p.get("volume", {})
        chg = p.get("priceChange", {})
        liq = p.get("liquidity", {})
        pairs.append({
            "chain": p.get("chainId", ""),
            "dex": p.get("dexId", ""),
            "pair_address": p.get("pairAddress", ""),
            "url": p.get("url", ""),
            "base_token": {
                "address": base.get("address", ""),
                "symbol": base.get("symbol", ""),
                "name": base.get("name", ""),
            },
            "quote_token": quote.get("symbol", ""),
            "price_usd": p.get("priceUsd"),
            "volume_24h": vol.get("h24"),
            "volume_1h": vol.get("h1"),
            "price_change_5m": chg.get("m5"),
            "price_change_1h": chg.get("h1"),
            "price_change_24h": chg.get("h24"),
            "liquidity_usd": liq.get("usd"),
            "fdv": p.get("fdv"),
            "market_cap": p.get("marketCap"),
            "pair_created_at": p.get("pairCreatedAt"),
        })
    return {"pairs": pairs, "total": len(pairs_raw)}
