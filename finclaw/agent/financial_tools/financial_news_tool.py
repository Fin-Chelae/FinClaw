"""Financial News Tool — fetch and summarize financial news headlines.

Pulls from Bloomberg Markets, MarketWatch, and Google News RSS feeds.
Returns headlines with title, summary, source, link, and publish time.
No meme scoring — pure financial news retrieval.

Registered as tool named ``financial_news`` in the agent loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from loguru import logger

from finclaw.agent.tools.base import Tool

_FEEDS: dict[str, dict[str, str]] = {
    "bloomberg": {
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "label": "Bloomberg Markets",
    },
    "marketwatch": {
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "label": "MarketWatch",
    },
    "google_finance": {
        "url": "https://news.google.com/rss/search?q=financial+market&hl=en&gl=US&ceid=US:en",
        "label": "Google News: Financial Markets",
    },
    "google_fed": {
        "url": "https://news.google.com/rss/search?q=Federal+Reserve+interest+rate&hl=en&gl=US&ceid=US:en",
        "label": "Google News: Federal Reserve",
    },
    "google_crypto_news": {
        "url": "https://news.google.com/rss/search?q=cryptocurrency+bitcoin&hl=en&gl=US&ceid=US:en",
        "label": "Google News: Crypto",
    },
}

_DEFAULT_SOURCES = "bloomberg,marketwatch,google_finance"


def _parse_time(entry: dict) -> datetime:
    for field in ("published", "updated"):
        val = entry.get(field)
        if val:
            try:
                dt = parsedate_to_datetime(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


async def _fetch_feed(url: str, label: str, limit: int) -> list[dict]:
    try:
        import feedparser
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            parsed = feedparser.parse(r.text)
    except Exception as e:
        logger.warning(f"financial_news: failed to fetch {label}: {e}")
        return []

    items = []
    for entry in parsed.entries[:limit]:
        title = entry.get("title", "").strip()
        if not title:
            continue
        summary = entry.get("summary", "") or entry.get("description", "")
        # Strip HTML tags from summary
        import re
        summary = re.sub(r"<[^>]+>", " ", summary).strip()
        summary = " ".join(summary.split())[:300]

        items.append({
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "source": label,
            "published": _parse_time(entry).strftime("%Y-%m-%d %H:%M UTC"),
        })
    return items


class FinancialNewsTool(Tool):
    """Fetch financial news headlines from Bloomberg, MarketWatch, and Google News."""

    @property
    def name(self) -> str:
        return "financial_news"

    @property
    def description(self) -> str:
        return (
            "Fetch latest financial news headlines from Bloomberg, MarketWatch, "
            "and Google News. Use 'headlines' to get top stories. "
            "Use 'search' to filter news by keyword (e.g. 'Fed', 'tariff', 'crypto'). "
            "Useful for macro context, market-moving events, and financial background research."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["headlines", "search"],
                    "description": (
                        "headlines: get latest top stories; "
                        "search: filter headlines by keyword."
                    ),
                },
                "sources": {
                    "type": "string",
                    "description": (
                        f"Comma-separated source slugs to query. "
                        f"Available: {', '.join(_FEEDS)}. "
                        f"Default: {_DEFAULT_SOURCES}."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword to filter headlines (used with 'search' command).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max headlines per source. Default: 10.",
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "headlines")
        limit = int(kwargs.get("limit", 10))
        keyword = kwargs.get("keyword", "").lower().strip()
        source_slugs = [
            s.strip()
            for s in kwargs.get("sources", _DEFAULT_SOURCES).split(",")
            if s.strip()
        ]

        logger.info(f"financial_news:{command} sources={source_slugs} keyword={keyword!r}")

        # Resolve sources
        feeds_to_fetch = {
            slug: _FEEDS[slug]
            for slug in source_slugs
            if slug in _FEEDS
        }
        if not feeds_to_fetch:
            return json.dumps({"error": f"No valid sources. Available: {list(_FEEDS)}"})

        # Fetch all feeds concurrently
        import asyncio
        tasks = [
            _fetch_feed(info["url"], info["label"], limit)
            for info in feeds_to_fetch.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: list[dict] = []
        for items in results:
            if isinstance(items, list):
                all_items.extend(items)

        # Filter by keyword if search command
        if command == "search" and keyword:
            all_items = [
                item for item in all_items
                if keyword in item["title"].lower() or keyword in item["summary"].lower()
            ]

        # Sort by published time (newest first)
        all_items.sort(key=lambda x: x["published"], reverse=True)

        return json.dumps({
            "command": command,
            "keyword": keyword or None,
            "sources_queried": list(feeds_to_fetch.keys()),
            "total": len(all_items),
            "items": all_items,
        }, ensure_ascii=False)
