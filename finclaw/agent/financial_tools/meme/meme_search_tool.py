"""Meme Coin Search Tool — social media search and meme word extraction.

Searches X/Twitter accounts and RSS feeds for trending content, extracts the
most "meme-worthy" word/phrase using LiteLLM, and presents candidates for user
approval. Does NOT auto-launch anything.

Supported commands (command parameter):
  check_tweets    - fetch latest tweets, extract meme words, return candidates
  analyze_tweet   - analyze specific tweet text for meme potential
  start_monitor   - configure accounts to monitor (default: elonmusk, realDonaldTrump)
  stop_monitor    - clear monitoring configuration
  status          - show current monitoring state and rate limit info

Requires: twikit>=2.0.0 (pip install twikit)
LLM calls use litellm directly (already a project dependency).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx
import litellm
import json_repair
from loguru import logger

from finclaw.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Module-level state (persists across tool invocations within a process)
# ---------------------------------------------------------------------------

# Accounts being monitored: {screen_name: user_id_or_empty}
_monitored_accounts: dict[str, str] = {}

# Seen tweet IDs to avoid re-processing: {tweet_id: timestamp_first_seen}
_seen_tweet_ids: dict[str, float] = {}

_MAX_SEEN_TWEETS = 5000

# Last check timestamp per account
_last_check: dict[str, float] = {}

# Whether monitoring is "active" (conceptual flag for status reporting)
_monitor_active: bool = False

# Default accounts to monitor
_DEFAULT_ACCOUNTS = ["elonmusk", "realDonaldTrump", "WhiteHouse"]

# Rate limit tracking for twikit
_rate_limit_reset: float = 0.0

# Twitter cookie-based auth.
# Priority: config.json (tools.memeMonitor.twitterCookies) → TWITTER_COOKIES env var (file path).
# Without either, falls back to GuestClient (currently broken).
_TWITTER_COOKIES_PATH: str | None = os.environ.get("TWITTER_COOKIES")

# Reusable authenticated twikit Client (initialized on first use)
_twitter_client: Any = None

# LLM model used for meme word extraction — set by MemeRouter on init via set_extraction_model()
_extraction_model: str = ""


def set_extraction_model(model: str) -> None:
    """Set the LLM model for meme extraction (called by MemeRouter on initialisation)."""
    global _extraction_model
    _extraction_model = model


def _load_meme_monitor_config() -> "MemeMonitorConfig | None":
    """Load meme monitor config from ~/.finclaw/config.json (returns None on failure)."""
    try:
        from finclaw.config.loader import load_config
        return load_config().tools.meme_monitor
    except Exception:
        return None

# ---------------------------------------------------------------------------
# RSS feed state
# ---------------------------------------------------------------------------

# RSSHub base URL: config.json → env var → default
def _resolve_rsshub_base() -> str:
    """Resolve RSSHub base URL: config.json first, then RSSHUB_BASE_URL env var."""
    cfg = _load_meme_monitor_config()
    if cfg and cfg.rsshub_base_url:
        return cfg.rsshub_base_url.rstrip("/")
    return os.environ.get("RSSHUB_BASE_URL", "https://rsshub.app")

_RSSHUB_BASE = _resolve_rsshub_base()

# Fallback RSSHub mirrors to try when the primary instance blocks (403/5xx).
_RSSHUB_MIRRORS: list[str] = [
    url.strip()
    for url in os.environ.get("RSSHUB_MIRRORS", "").split(",")
    if url.strip()
] or [
    "https://rsshub.rssforever.com",
    "https://rsshub.moeyy.cn",
]

# Default RSS feeds.
# Feeds with "url" use the full URL directly (native RSS).
# Feeds with "path" use _RSSHUB_BASE + path (RSSHub proxy).
# Feeds with "fallback_urls" list alternative URLs tried on 403.
_DEFAULT_RSS_FEEDS: dict[str, dict[str, str | list[str]]] = {
    "reddit_wsb": {
        "url": "https://www.reddit.com/r/wallstreetbets/.rss",
        "fallback_urls": [
            "https://old.reddit.com/r/wallstreetbets/.rss",
            "https://www.reddit.com/r/wallstreetbets/hot.json",
        ],
        "label": "Reddit r/wallstreetbets",
    },
    "reddit_crypto": {
        "url": "https://www.reddit.com/r/CryptoCurrency/.rss",
        "fallback_urls": [
            "https://old.reddit.com/r/CryptoCurrency/.rss",
        ],
        "label": "Reddit r/CryptoCurrency",
    },
    "reddit_memecoins": {
        "url": "https://www.reddit.com/r/memecoins/.rss",
        "fallback_urls": [
            "https://old.reddit.com/r/memecoins/.rss",
        ],
        "label": "Reddit r/memecoins",
    },
    "trump_truth": {
        "path": "/truthsocial/user/realDonaldTrump",
        "label": "Trump Truth Social (RSSHub)",
    },
    "tiktok_memecoin": {
        "path": "/tiktok/hashtag/memecoin",
        "label": "TikTok #memecoin (RSSHub)",
    },
    "twitter_elonmusk": {
        "path": "/twitter/user/elonmusk",
        "label": "Twitter @elonmusk (RSSHub)",
    },
    "twitter_trump": {
        "path": "/twitter/user/realDonaldTrump",
        "label": "Twitter @realDonaldTrump (RSSHub)",
    },
    "twitter_WhiteHouse": {
        "path": "/twitter/user/WhiteHouse",
        "label": "Twitter @WhiteHouse (RSSHub)",
    },
    # ── Google News (crypto/meme focused) ────────────────────────────
    "google_news_bitcoin": {
        "url": "https://news.google.com/rss/search?q=bitcoin&hl=en",
        "label": "Google News: Bitcoin",
    },
    "google_news_memecoin": {
        "url": "https://news.google.com/rss/search?q=meme+coin&hl=en",
        "label": "Google News: Meme Coin",
    },
    # ── More Reddit ───────────────────────────────────────────────────
    "reddit_investing": {
        "url": "https://www.reddit.com/r/investing/hot/.rss",
        "fallback_urls": [
            "https://old.reddit.com/r/investing/hot/.rss",
        ],
        "label": "Reddit r/investing",
    },
}

# Active RSS feeds: {slug: {path, label}}
_rss_feeds: dict[str, dict[str, str]] = {}

# Seen RSS item IDs (GUID or link): {id: timestamp_first_seen}
_seen_rss_ids: dict[str, float] = {}

_MAX_SEEN_RSS = 5000

# Last RSS check timestamp per feed slug
_last_rss_check: dict[str, float] = {}

# Max characters of RSS item content to send to LLM
_RSS_TEXT_TRUNCATE = 1000

# ---------------------------------------------------------------------------
# LLM prompt for meme word extraction
# ---------------------------------------------------------------------------

_MEME_EXTRACTION_PROMPT = """\
You are a meme coin analyst. Given a social media post, extract the single most \
"meme-worthy" word or short phrase that could become a viral meme coin ticker.

Rules:
1. Pick the most unusual, catchy, or culturally resonant word/phrase (1-3 words max).
2. Ignore common stopwords, URLs, @mentions, and generic filler.
3. If the post references a specific product, animal, slang term, or neologism, prefer that.
4. If the post is mundane with no meme potential, say so honestly.
5. Suggest a plausible ticker symbol (3-5 uppercase letters, no spaces).
6. Rate the meme potential on a scale of 1-10.
7. Rate your confidence in this extraction on a scale of 1-10.
8. Classify the meme into one of the VIRAL CATEGORIES below.

VIRAL CATEGORIES (ranked by typical viral potential):

1. CURRENT_EVENT (score 8-10) — Reactions to breaking news, political gaffes, celebrity \
moments, or trending events. The fastest-moving category. A politician says a funny word, \
a CEO tweets something absurd, a news headline goes viral. Speed matters most here — the \
first coin to tokenize the moment wins. Examples: $HAWK (Hawk Tuah), $LUIGI (Luigi Mangione).

2. ANIMAL (score 6-9) — Animals, especially dogs, frogs, cats, penguins, with an absurd \
modifier or unusual twist. The OG meme coin formula. Plain animal names are played out — \
the modifier is what matters. Examples: dogwifhat (WIF), BONK, PEPE, PENGU, FLOKI.

3. CELEBRITY_REF (score 6-9) — Direct or indirect references to public figures, \
influencers, or their catchphrases. The person does NOT need to endorse it. Works best \
when tied to a specific viral moment, not just a generic name. Examples: $TRUMP, $MELANIA.

4. INTERNET_SLANG (score 5-8) — New slang, neologisms, or phrases born from viral \
internet moments. Short shelf life but explosive initial pumps. Must be genuinely NEW — \
if it's been around for more than a few weeks, it's already stale. Examples: RIZZ, \
SKIBIDI, SIGMA, NPC.

5. ABSURDIST (score 5-8) — Deliberately stupid, nonsensical, or provocatively dumb names. \
The joke IS the name. Works through shock value and shareability. The dumber it sounds, \
the more people talk about it. Examples: FARTCOIN, BUTTCOIN, $NOTHING.

6. AI_TECH (score 5-8) — Coins riding the AI/tech narrative, especially if an AI agent \
"created" or promoted the coin. Meta-narrative plays. Examples: GOAT (AI-agent-promoted), \
TURBO (GPT-4 concept).

7. POP_CULTURE (score 4-7) — References to movies, games, anime, music, but twisted or \
tied to a specific moment. Plain character names are weak — a specific scene, quote, or \
cultural moment is stronger. Examples: references to viral movie scenes, game events.

8. FOOD_OBJECT (score 4-7) — Random everyday objects turned into tokens. Only works when \
tied to a specific viral moment. Without context, a random object name is weak. Examples: \
SUSHI, BANANA (when tied to viral art piece).

9. POLITICAL (score 4-7) — Coins tied to political movements, elections, policy outrage, \
or geopolitical events. Spikes around elections and controversy. Can be polarizing which \
limits audience but drives passion.

10. GENERIC_CRYPTO (score 1-3) — Recycled crypto culture terms. These are NOT novel. \
HODL, DEGEN, FOMO, WAGMI, NGMI, GM, GN, MOON, LAMBO, APE, DIAMOND HANDS, PAPER HANDS, \
REKT, PUMP, DUMP, BULL, BEAR, WHALE, RUG, YOLO, TO THE MOON, LFG, BASED. If the ONLY \
meme-worthy content falls here, set meme_score to 2.

SCORING GUIDANCE:
- The goal is to find FRESH meme words from TODAY's viral moments — not recycled slang.
- CURRENT_EVENT + speed = highest value. A mediocre word tied to a breaking event (score 8) \
beats a clever word with no cultural moment behind it (score 5).
- Absurdist names punch above their weight — shareability drives virality.
- Compound concepts score higher: animal + modifier > plain animal, event + slang > plain slang.
- If the post contains MULTIPLE potential meme words, pick the one with the highest \
category ranking AND most novelty.

Respond with ONLY valid JSON (no markdown fences, no explanation):
{
  "meme_word": "the extracted word or phrase",
  "ticker_suggestion": "TICKER",
  "category": "CURRENT_EVENT",
  "meme_score": 7,
  "confidence": 8,
  "origin": "One-sentence summary of what the post is about and why this word emerged from it. Example: 'Elon Musk tweeted about his dog wearing a cowboy hat, sparking a wave of cowboy dog memes'",
  "reasoning": "brief explanation — mention category, freshness, and why this word has viral potential"
}

If the post has no meme potential at all, respond:
{
  "meme_word": null,
  "ticker_suggestion": null,
  "category": null,
  "meme_score": 0,
  "confidence": 9,
  "origin": null,
  "reasoning": "explanation of why this post has no meme potential"
}
"""


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def _prune_seen_tweets() -> None:
    """Remove oldest seen tweet IDs if exceeding _MAX_SEEN_TWEETS."""
    if len(_seen_tweet_ids) > _MAX_SEEN_TWEETS:
        sorted_ids = sorted(_seen_tweet_ids.items(), key=lambda x: x[1])
        to_remove = len(_seen_tweet_ids) - _MAX_SEEN_TWEETS
        for tweet_id, _ in sorted_ids[:to_remove]:
            del _seen_tweet_ids[tweet_id]


def _prune_seen_rss() -> None:
    """Remove oldest seen RSS item IDs if exceeding _MAX_SEEN_RSS."""
    if len(_seen_rss_ids) > _MAX_SEEN_RSS:
        sorted_ids = sorted(_seen_rss_ids.items(), key=lambda x: x[1])
        to_remove = len(_seen_rss_ids) - _MAX_SEEN_RSS
        for item_id, _ in sorted_ids[:to_remove]:
            del _seen_rss_ids[item_id]


def _parse_rss_xml(xml_bytes: bytes, feed_slug: str) -> list[dict]:
    """Parse RSS 2.0 or Atom XML into a list of item dicts.

    Supports both RSS 2.0 (<item>) and Atom (<entry>) formats.
    Filters out already-seen items via _seen_rss_ids.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.error(f"RSS XML parse error for feed '{feed_slug}': {e}")
        return []

    # Detect Atom vs RSS 2.0
    atom_ns = "{http://www.w3.org/2005/Atom}"
    is_atom = root.tag == f"{atom_ns}feed" or root.tag == "feed"

    if is_atom:
        return _parse_atom_entries(root, feed_slug, atom_ns)
    return _parse_rss_items(root, feed_slug)


def _parse_rss_items(root: ET.Element, feed_slug: str) -> list[dict]:
    """Parse RSS 2.0 <item> elements."""
    items: list[dict] = []
    channel = root.find("channel")
    if channel is None:
        channel = root

    for item_elem in channel.findall("item"):
        guid = (
            (item_elem.findtext("guid") or "").strip()
            or (item_elem.findtext("link") or "").strip()
        )
        if not guid or guid in _seen_rss_ids:
            continue

        _seen_rss_ids[guid] = time.time()

        title = (item_elem.findtext("title") or "").strip()
        link = (item_elem.findtext("link") or "").strip()
        description = (item_elem.findtext("description") or "").strip()
        pub_date = (item_elem.findtext("pubDate") or "").strip()

        text = _build_item_text(title, description)
        items.append({
            "guid": guid,
            "title": title,
            "link": link,
            "text": text,
            "pub_date": pub_date,
            "feed_slug": feed_slug,
        })

    return items


def _parse_atom_entries(root: ET.Element, feed_slug: str, ns: str) -> list[dict]:
    """Parse Atom <entry> elements (e.g. Reddit native RSS)."""
    items: list[dict] = []

    for entry in root.findall(f"{ns}entry"):
        # Atom uses <id> for GUID
        guid = (entry.findtext(f"{ns}id") or "").strip()
        if not guid:
            # Fallback to <link href="...">
            link_elem = entry.find(f"{ns}link")
            guid = (link_elem.get("href", "") if link_elem is not None else "").strip()
        if not guid or guid in _seen_rss_ids:
            continue

        _seen_rss_ids[guid] = time.time()

        title = (entry.findtext(f"{ns}title") or "").strip()

        link_elem = entry.find(f"{ns}link")
        link = (link_elem.get("href", "") if link_elem is not None else "").strip()

        # Atom uses <content> or <summary> instead of <description>
        content = (
            (entry.findtext(f"{ns}content") or "").strip()
            or (entry.findtext(f"{ns}summary") or "").strip()
        )
        pub_date = (
            (entry.findtext(f"{ns}updated") or "").strip()
            or (entry.findtext(f"{ns}published") or "").strip()
        )

        text = _build_item_text(title, content)
        items.append({
            "guid": guid,
            "title": title,
            "link": link,
            "text": text,
            "pub_date": pub_date,
            "feed_slug": feed_slug,
        })

    return items


def _build_item_text(title: str, description: str) -> str:
    """Build LLM-ready text from title + description, stripping HTML."""
    text = title
    if description:
        clean_desc = re.sub(r"<[^>]+>", " ", description)
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
        text = f"{title}\n\n{clean_desc}"

    if len(text) > _RSS_TEXT_TRUNCATE:
        text = text[:_RSS_TEXT_TRUNCATE] + "..."
    return text


def _resolve_feed_url(feed_info: dict[str, str]) -> str:
    """Build primary feed URL from config: full 'url' key or _RSSHUB_BASE + 'path'."""
    if "url" in feed_info:
        return feed_info["url"]
    return _RSSHUB_BASE.rstrip("/") + feed_info["path"]


def _resolve_fallback_urls(feed_info: dict) -> list[str]:
    """Build ordered list of fallback URLs to try when the primary URL fails.

    For feeds with explicit 'fallback_urls', those come first.
    For RSSHub-path feeds, mirrors are appended automatically.
    """
    urls: list[str] = []

    # Explicit fallback URLs (e.g. old.reddit.com, .json endpoint)
    for u in feed_info.get("fallback_urls", []):
        if u not in urls:
            urls.append(u)

    # For RSSHub-path feeds, try known mirrors
    if "path" in feed_info:
        for mirror in _RSSHUB_MIRRORS:
            mirror_url = mirror.rstrip("/") + feed_info["path"]
            if mirror_url not in urls:
                urls.append(mirror_url)

    return urls


def _parse_reddit_json(json_bytes: bytes, feed_slug: str) -> list[dict]:
    """Parse Reddit's JSON listing (hot.json) into item dicts.

    Fallback parser when .rss returns 403 (e.g. NSFW subreddits like WSB).
    """
    try:
        data = json.loads(json_bytes)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Reddit JSON parse error for '{feed_slug}': {e}")
        return []

    items: list[dict] = []
    children = data.get("data", {}).get("children", [])

    for child in children:
        post = child.get("data", {})
        guid = post.get("name", "") or post.get("id", "")
        if not guid or guid in _seen_rss_ids:
            continue

        _seen_rss_ids[guid] = time.time()

        title = (post.get("title") or "").strip()
        selftext = (post.get("selftext") or "").strip()
        link = post.get("url") or f"https://reddit.com{post.get('permalink', '')}"

        text = _build_item_text(title, selftext)
        items.append({
            "guid": guid,
            "title": title,
            "link": link,
            "text": text,
            "pub_date": str(post.get("created_utc", "")),
            "feed_slug": feed_slug,
        })

    return items


async def _try_fetch_feed(
    client: httpx.AsyncClient,
    url: str,
    slug: str,
    label: str,
) -> tuple[list[dict] | None, int | None]:
    """Attempt to fetch and parse a single feed URL.

    Returns (items, None) on success, or (None, status_code) on HTTP failure.
    Raises on network/timeout errors (caller handles).
    """
    response = await client.get(url)
    if not response.is_success:
        return None, response.status_code

    # Reddit JSON endpoint returns application/json
    content_type = response.headers.get("content-type", "")
    if url.endswith(".json") or "application/json" in content_type:
        return _parse_reddit_json(response.content, slug), None

    return _parse_rss_xml(response.content, slug), None


async def _fetch_single_feed(
    client: httpx.AsyncClient,
    slug: str,
    feed_info: dict,
    max_per_feed: int,
) -> list[dict]:
    """Fetch one RSS feed, trying fallback URLs on failure. Never raises."""
    label = feed_info.get("label", slug)
    primary_url = _resolve_feed_url(feed_info)
    fallback_urls = _resolve_fallback_urls(feed_info)
    urls_to_try = [primary_url] + [u for u in fallback_urls if u != primary_url]

    last_status: int | None = None
    for url in urls_to_try:
        try:
            feed_items, status = await _try_fetch_feed(client, url, slug, label)
            if feed_items is not None:
                items = []
                for item in feed_items[:max_per_feed]:
                    item["source_label"] = label
                    items.append(item)
                _last_rss_check[slug] = time.time()
                if url != primary_url:
                    logger.info(f"RSS feed '{label}' succeeded via fallback: {url}")
                return items
            last_status = status
            logger.debug(f"RSS feed '{label}' returned HTTP {status} from {url}, trying next...")
        except httpx.TimeoutException:
            logger.debug(f"RSS feed '{label}' timed out at {url}, trying next...")
            last_status = None
        except httpx.RequestError as e:
            logger.debug(f"RSS feed '{label}' request error at {url}: {e}, trying next...")
            last_status = None

    msg = (
        f"RSS feed '{label}' failed on all URLs (last HTTP {last_status})"
        if last_status
        else f"RSS feed '{label}' failed on all URLs (timeout/network error)"
    )
    if "path" in feed_info:
        msg += (
            ". This is an RSSHub feed — set tools.memeMonitor.rsshubBaseUrl "
            "in ~/.finclaw/config.json to a self-hosted instance "
            "(docker run -p 1200:1200 diygod/rsshub) for reliable access."
        )
    logger.warning(msg)
    return [{"error": msg}]


async def _fetch_rss_items(
    feeds: dict[str, dict[str, str]] | None = None,
    max_per_feed: int = 10,
) -> list[dict]:
    """Fetch RSS/Atom items from all configured feeds concurrently.

    All feeds are fetched in parallel — a slow or timing-out RSSHub feed
    does not block native RSS feeds. Per-feed timeout is 8s for RSSHub
    paths (public instances are flaky) and 15s for native RSS URLs.
    """
    active_feeds = feeds or _rss_feeds or dict(_DEFAULT_RSS_FEEDS)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }

    # Use a shorter timeout for RSSHub feeds (public instances are unreliable)
    # and a normal timeout for native RSS/Atom URLs.
    # We create one client per timeout class to keep things simple.
    rsshub_timeout = httpx.Timeout(8.0)
    native_timeout = httpx.Timeout(15.0)

    async with (
        httpx.AsyncClient(timeout=rsshub_timeout, follow_redirects=True, headers=headers) as rsshub_client,
        httpx.AsyncClient(timeout=native_timeout, follow_redirects=True, headers=headers) as native_client,
    ):
        tasks = [
            _fetch_single_feed(
                rsshub_client if "path" in feed_info else native_client,
                slug,
                feed_info,
                max_per_feed,
            )
            for slug, feed_info in active_feeds.items()
        ]
        results = await asyncio.gather(*tasks)

    new_items = [item for feed_items in results for item in feed_items]
    _prune_seen_rss()
    return new_items


def _resolve_twitter_cookies_path() -> str | None:
    """Resolve Twitter cookies path: config.json → TWITTER_COOKIES env var.

    If config.json has twitterCookies dict, writes it to ~/.finclaw/twitter_cookies.json
    and returns that path. Falls back to TWITTER_COOKIES env var (raw file path).
    """
    cfg = _load_meme_monitor_config()
    if cfg and cfg.twitter_cookies and any(v for v in cfg.twitter_cookies.values()):
        # Config has non-empty cookie values — write to a known location for twikit
        from pathlib import Path
        cookies_file = Path.home() / ".finclaw" / "twitter_cookies.json"
        cookies_file.parent.mkdir(parents=True, exist_ok=True)
        cookies_file.write_text(json.dumps(cfg.twitter_cookies, indent=2))
        cookies_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600 — owner only
        return str(cookies_file)

    return _TWITTER_COOKIES_PATH


async def _get_authenticated_client() -> Any | None:
    """Get or create an authenticated twikit Client using saved cookies.

    Cookie source priority:
    1. config.json → tools.memeMonitor.twitterCookies (written to ~/.finclaw/twitter_cookies.json)
    2. TWITTER_COOKIES env var (path to a cookies JSON file)

    Returns None if no cookies are configured or they're invalid.
    The client is cached in _twitter_client for reuse across calls.
    """
    global _twitter_client

    if _twitter_client is not None:
        return _twitter_client

    cookies_path = _resolve_twitter_cookies_path()
    if not cookies_path:
        return None

    try:
        from twikit import Client
    except ImportError:
        return None

    try:
        client = Client(language="en-US")
        client.load_cookies(cookies_path)
        _twitter_client = client
        logger.info(f"Loaded Twitter cookies from {cookies_path}")
        return client
    except Exception as e:
        logger.warning(f"Failed to load Twitter cookies from {cookies_path}: {e}")
        return None


def _extract_tweet_data(tweet: Any, screen_name: str) -> dict:
    """Extract standardized tweet dict from a twikit Tweet object."""
    tweet_id = str(tweet.id)

    text = ""
    if hasattr(tweet, "full_text") and tweet.full_text:
        text = tweet.full_text
    elif hasattr(tweet, "text") and tweet.text:
        text = tweet.text

    return {
        "tweet_id": tweet_id,
        "screen_name": (
            tweet.user.screen_name
            if hasattr(tweet, "user") and tweet.user
            else screen_name
        ),
        "text": text,
        "created_at": (
            tweet.created_at_datetime.isoformat()
            if hasattr(tweet, "created_at_datetime") and tweet.created_at_datetime
            else getattr(tweet, "created_at", None)
        ),
        "favorite_count": getattr(tweet, "favorite_count", None),
        "retweet_count": getattr(tweet, "retweet_count", None),
        "view_count": getattr(tweet, "view_count", None),
    }


async def _fetch_tweets_authenticated(
    client: Any,
    screen_names: list[str],
    max_per_account: int = 10,
) -> list[dict]:
    """Fetch tweets using an authenticated twikit Client (cookie-based)."""
    try:
        from twikit import TooManyRequests
    except ImportError:
        return [{"error": "twikit not installed"}]

    global _rate_limit_reset
    new_tweets: list[dict] = []

    for screen_name in screen_names:
        try:
            user = await client.get_user_by_screen_name(screen_name)
            _monitored_accounts[screen_name] = str(user.id)

            tweets = await client.get_user_tweets(
                user.id, tweet_type="Tweets", count=max_per_account,
            )

            for tweet in tweets:
                tweet_id = str(tweet.id)
                if tweet_id in _seen_tweet_ids:
                    continue
                _seen_tweet_ids[tweet_id] = time.time()
                new_tweets.append(_extract_tweet_data(tweet, screen_name))

            _last_check[screen_name] = time.time()

        except TooManyRequests as e:
            _rate_limit_reset = (
                e.rate_limit_reset
                if hasattr(e, "rate_limit_reset") and e.rate_limit_reset
                else time.time() + 900
            )
            reset_str = datetime.fromtimestamp(
                _rate_limit_reset, tz=timezone.utc
            ).isoformat()
            logger.warning(f"Twitter rate limit hit for @{screen_name}. Reset at {reset_str}")
            new_tweets.append({
                "error": f"Rate limited for @{screen_name}. Retry after {reset_str}"
            })
        except Exception as e:
            logger.error(f"Error fetching @{screen_name} (authenticated): {e}")
            new_tweets.append({"error": f"Failed to fetch @{screen_name}: {e}"})

    _prune_seen_tweets()
    return new_tweets


async def _fetch_tweets_guest(
    screen_names: list[str],
    max_per_account: int = 10,
) -> list[dict]:
    """Fetch tweets using twikit GuestClient (unauthenticated, currently broken)."""
    global _rate_limit_reset

    try:
        from twikit.guest import GuestClient
        from twikit import TooManyRequests
    except ImportError:
        return [{"error": "twikit not installed. Run: pip install twikit>=2.0.0"}]

    now = time.time()
    if now < _rate_limit_reset:
        wait = int(_rate_limit_reset - now)
        return [{"error": f"Rate limited. Retry after {wait}s."}]

    client = GuestClient()
    try:
        await client.activate()
    except Exception as e:
        logger.warning(f"twikit guest client activation failed: {e}")
        try:
            await client.close()
        except Exception:
            pass
        return [{
            "error": (
                f"Twitter/X guest API unavailable: {e}. "
                "Set tools.memeMonitor.twitterCookies in ~/.finclaw/config.json "
                "for authenticated access. See docs for setup."
            ),
        }]

    new_tweets: list[dict] = []

    try:
        for screen_name in screen_names:
            try:
                user = await client.get_user_by_screen_name(screen_name)
                _monitored_accounts[screen_name] = str(user.id)

                tweets = await client.get_user_tweets(user.id, count=max_per_account)

                for tweet in tweets:
                    tweet_id = str(tweet.id)
                    if tweet_id in _seen_tweet_ids:
                        continue
                    _seen_tweet_ids[tweet_id] = time.time()
                    new_tweets.append(_extract_tweet_data(tweet, screen_name))

                _last_check[screen_name] = time.time()

            except TooManyRequests as e:
                _rate_limit_reset = (
                    e.rate_limit_reset
                    if hasattr(e, "rate_limit_reset") and e.rate_limit_reset
                    else time.time() + 900
                )
                reset_str = datetime.fromtimestamp(
                    _rate_limit_reset, tz=timezone.utc
                ).isoformat()
                logger.warning(f"Twitter rate limit hit for @{screen_name}. Reset at {reset_str}")
                new_tweets.append({
                    "error": f"Rate limited for @{screen_name}. Retry after {reset_str}"
                })
            except Exception as e:
                logger.error(f"Error fetching tweets for @{screen_name}: {e}")
                new_tweets.append({"error": f"Failed to fetch @{screen_name}: {e}"})
    finally:
        try:
            await client.close()
        except Exception:
            pass

    _prune_seen_tweets()
    return new_tweets


async def _fetch_tweets(
    screen_names: list[str],
    max_per_account: int = 10,
) -> list[dict]:
    """Fetch tweets: authenticated client (cookies) → guest client (fallback).

    Strategy:
    1. If TWITTER_COOKIES is set, use authenticated Client (reliable).
    2. Otherwise, fall back to GuestClient (currently broken on Twitter's side).
    """
    # Try authenticated client first
    auth_client = await _get_authenticated_client()
    if auth_client is not None:
        result = await _fetch_tweets_authenticated(
            auth_client, screen_names, max_per_account,
        )
        valid = [t for t in result if "error" not in t]
        if valid:
            return result
        # Auth failed (expired cookies?) — clear cached client and try guest
        global _twitter_client
        _twitter_client = None
        logger.warning("Authenticated Twitter client returned no tweets, trying guest...")

    # Fall back to guest client
    return await _fetch_tweets_guest(screen_names, max_per_account)


def _get_extraction_model() -> str:
    """Return the LLM model for meme extraction (set by MemeRouter from agents.defaults.inner_model or model)."""
    return _extraction_model


async def _extract_meme_word(tweet_text: str, screen_name: str = "") -> dict:
    """Use LiteLLM to extract the most meme-worthy word from a tweet."""
    try:
        user_message = (
            f"Tweet from @{screen_name}:\n\n{tweet_text}"
            if screen_name
            else f"Tweet:\n\n{tweet_text}"
        )

        response = await litellm.acompletion(
            model=_get_extraction_model(),
            messages=[
                {"role": "system", "content": _MEME_EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0.3,
        )

        text = response.choices[0].message.content or ""
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json_repair.loads(text)
        if not isinstance(result, dict):
            return {"error": f"LLM returned non-dict: {text[:200]}"}
        return result

    except Exception as e:
        logger.error(f"Meme extraction failed: {e}")
        return {"error": f"LLM extraction failed: {e}"}


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class MemeSearchTool(Tool):
    """Search social media for meme-worthy content."""

    @property
    def name(self) -> str:
        return "meme_search"

    @property
    def description(self) -> str:
        return (
            "Search X/Twitter accounts and RSS feeds (Reddit, Truth Social, TikTok) "
            "for meme-worthy content. "
            "Use check_all to fetch from ALL sources (tweets + RSS) simultaneously. "
            "Use check_tweets for Twitter only, check_rss for RSS feeds only. "
            "Use analyze_tweet to analyze specific text for meme potential. "
            "Use start_monitor/stop_monitor to configure which accounts/feeds to watch. "
            "Use status to see current monitoring state. "
            "IMPORTANT: This tool only identifies candidates — it does NOT auto-launch. "
            "Always present candidates to the user for approval before taking any action."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "check_tweets",
                        "check_rss",
                        "check_all",
                        "analyze_tweet",
                        "start_monitor",
                        "stop_monitor",
                        "status",
                    ],
                    "description": (
                        "check_tweets: fetch latest tweets and extract meme words; "
                        "check_rss: fetch latest RSS feed items and extract meme words; "
                        "check_all: fetch tweets + RSS in parallel, merge and rank candidates; "
                        "analyze_tweet: analyze specific tweet text for meme potential; "
                        "start_monitor: configure accounts and feeds to monitor; "
                        "stop_monitor: clear monitoring config; "
                        "status: show monitoring state."
                    ),
                },
                "accounts": {
                    "type": "string",
                    "description": (
                        "Comma-separated Twitter screen names (without @) for "
                        "start_monitor or check_tweets. "
                        "Default: 'elonmusk,realDonaldTrump,WhiteHouse'."
                    ),
                },
                "tweet_text": {
                    "type": "string",
                    "description": (
                        "The tweet text to analyze. Required for analyze_tweet."
                    ),
                },
                "screen_name": {
                    "type": "string",
                    "description": (
                        "Screen name of the tweet author (for context in analyze_tweet)."
                    ),
                },
                "max_tweets": {
                    "type": "integer",
                    "description": "Max tweets per account for check_tweets. Default: 10.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "min_meme_score": {
                    "type": "integer",
                    "description": (
                        "Minimum meme score (1-10) to include in results. "
                        "Default: 3."
                    ),
                    "minimum": 1,
                    "maximum": 10,
                },
                "feeds": {
                    "type": "string",
                    "description": (
                        "Comma-separated feed slugs for check_rss/check_all "
                        "(e.g. 'reddit_wsb,trump_truth'). "
                        "Default: all configured feeds."
                    ),
                },
                "max_items": {
                    "type": "integer",
                    "description": "Max items per RSS feed. Default: 10.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "rsshub_base": {
                    "type": "string",
                    "description": (
                        "Override RSSHub base URL "
                        "(e.g. 'http://localhost:1200' for self-hosted). "
                        "Default: RSSHUB_BASE_URL env var or https://rsshub.app."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        logger.info(f"meme_monitor:{command}")

        if command == "check_tweets":
            result = await self._check_tweets(kwargs)
        elif command == "check_rss":
            result = await self._check_rss(kwargs)
        elif command == "check_all":
            result = await self._check_all(kwargs)
        elif command == "analyze_tweet":
            result = await self._analyze_tweet(kwargs)
        elif command == "start_monitor":
            result = self._start_monitor(kwargs)
        elif command == "stop_monitor":
            result = self._stop_monitor()
        elif command == "status":
            result = self._status()
        else:
            result = {"error": f"Unknown command: {command!r}"}

        return json.dumps(result, ensure_ascii=False, default=str)

    # -----------------------------------------------------------------------
    # Command implementations
    # -----------------------------------------------------------------------

    async def _check_tweets(self, kwargs: dict) -> dict:
        """Fetch latest tweets via RSS (RSSHub) first, twikit as fallback.

        Strategy:
        1. Try RSSHub Twitter feeds (twitter_elonmusk, twitter_trump, etc.)
           — these use the mirror fallback system automatically.
        2. If RSS returns zero items, fall back to twikit GuestClient.
        3. Both sources produce the same candidate format.
        """
        accounts_str = kwargs.get("accounts", "")
        if accounts_str:
            accounts = [a.strip().lstrip("@") for a in accounts_str.split(",") if a.strip()]
        elif _monitored_accounts:
            accounts = list(_monitored_accounts.keys())
        else:
            accounts = list(_DEFAULT_ACCOUNTS)

        max_tweets = int(kwargs.get("max_tweets", 10))
        min_score = int(kwargs.get("min_meme_score", 3))

        # --- Phase 1: Try RSS-based Twitter feeds (RSSHub mirrors) ---
        twitter_feeds = {}
        all_feeds = _rss_feeds or dict(_DEFAULT_RSS_FEEDS)
        for account in accounts:
            slug = f"twitter_{account}"
            if slug in all_feeds:
                twitter_feeds[slug] = all_feeds[slug]
            else:
                # Dynamically build feed for accounts not in defaults
                twitter_feeds[slug] = {
                    "path": f"/twitter/user/{account}",
                    "label": f"Twitter @{account} (RSSHub)",
                }

        rss_items = await _fetch_rss_items(feeds=twitter_feeds, max_per_feed=max_tweets)
        rss_errors = [it for it in rss_items if "error" in it]
        valid_rss = [it for it in rss_items if "error" not in it]

        # Convert RSS items to tweet-like dicts for uniform processing
        tweets: list[dict] = []
        for item in valid_rss:
            # Extract screen_name from feed slug or label
            slug = item.get("feed_slug", "")
            screen_name = slug.replace("twitter_", "") if slug.startswith("twitter_") else ""
            tweets.append({
                "tweet_id": item.get("guid", ""),
                "screen_name": screen_name or item.get("source_label", ""),
                "text": item.get("text", ""),
                "created_at": item.get("pub_date"),
                "favorite_count": None,
                "retweet_count": None,
                "view_count": None,
                "source": "rss",
            })

        source_used = "rss" if tweets else None
        errors: list[dict] = list(rss_errors)

        # --- Phase 2: If RSS yielded nothing, try twikit ---
        if not tweets:
            logger.info("Twitter RSS feeds returned no items, trying twikit...")
            twikit_results = await _fetch_tweets(accounts, max_per_account=max_tweets)
            twikit_errors = [t for t in twikit_results if "error" in t]
            valid_twikit = [t for t in twikit_results if "error" not in t]

            for t in valid_twikit:
                t["source"] = "twikit"
            tweets.extend(valid_twikit)
            errors.extend(twikit_errors)
            if valid_twikit:
                source_used = "twikit"

        if not tweets:
            return {
                "candidates": [],
                "tweets_fetched": 0,
                "errors": errors or None,
                "note": "No new tweets found via RSS or twikit. Already-seen items are skipped.",
            }

        # Parallel LLM extraction for all tweets
        extractions = await asyncio.gather(*(
            _extract_meme_word(tweet["text"], tweet.get("screen_name", ""))
            for tweet in tweets
        ))

        candidates = []
        for tweet, extraction in zip(tweets, extractions):
            meme_score = extraction.get("meme_score", 0) or 0
            if meme_score >= min_score:
                candidates.append({
                    "tweet_id": tweet["tweet_id"],
                    "screen_name": tweet["screen_name"],
                    "tweet_text": tweet["text"][:280],
                    "created_at": tweet["created_at"],
                    "engagement": {
                        "likes": tweet.get("favorite_count"),
                        "retweets": tweet.get("retweet_count"),
                        "views": tweet.get("view_count"),
                    },
                    "meme_word": extraction.get("meme_word"),
                    "ticker_suggestion": extraction.get("ticker_suggestion"),
                    "category": extraction.get("category"),
                    "origin": extraction.get("origin"),
                    "meme_score": meme_score,
                    "confidence": extraction.get("confidence"),
                    "reasoning": extraction.get("reasoning"),
                    "source": tweet.get("source", "unknown"),
                })

        candidates.sort(key=lambda c: c.get("meme_score", 0), reverse=True)

        return {
            "candidates": candidates,
            "tweets_fetched": len(tweets),
            "tweets_filtered": len(tweets) - len(candidates),
            "min_meme_score_filter": min_score,
            "accounts_checked": accounts,
            "tweet_source": source_used,
            "errors": errors or None,
            "note": (
                "These are candidates only. Present to user for approval "
                "before taking any action. Higher meme_score = higher viral potential."
            ),
        }

    async def _analyze_tweet(self, kwargs: dict) -> dict:
        """Analyze a specific tweet text for meme potential."""
        tweet_text = kwargs.get("tweet_text", "")
        if not tweet_text:
            return {"error": "tweet_text is required for analyze_tweet"}

        screen_name = kwargs.get("screen_name", "")
        extraction = await _extract_meme_word(tweet_text, screen_name)

        return {
            "tweet_text": tweet_text[:280],
            "screen_name": screen_name or None,
            "analysis": extraction,
            "note": "This is analysis only. Present to user for approval before any action.",
        }

    async def _check_rss(self, kwargs: dict, _exclude_twitter: bool = False) -> dict:
        """Fetch latest RSS feed items and extract meme words.

        When _exclude_twitter=True (used by _check_all), twitter_* feeds are
        skipped since _check_tweets handles them separately.
        """
        global _RSSHUB_BASE

        rsshub_base = kwargs.get("rsshub_base", "")
        if rsshub_base:
            _RSSHUB_BASE = rsshub_base.rstrip("/")

        feeds_str = kwargs.get("feeds", "")
        if feeds_str:
            slugs = [s.strip() for s in feeds_str.split(",") if s.strip()]
            active = _rss_feeds or dict(_DEFAULT_RSS_FEEDS)
            feeds = {s: active[s] for s in slugs if s in active}
            if not feeds:
                return {
                    "error": f"No known feeds matching: {feeds_str}. "
                    f"Available: {', '.join(active.keys())}",
                }
        else:
            feeds = dict(_rss_feeds or _DEFAULT_RSS_FEEDS)

        # When called from check_all, skip twitter_* feeds to avoid duplication
        if _exclude_twitter and feeds:
            feeds = {k: v for k, v in feeds.items() if not k.startswith("twitter_")}

        max_items = int(kwargs.get("max_items", 10))
        min_score = int(kwargs.get("min_meme_score", 3))

        items = await _fetch_rss_items(feeds=feeds, max_per_feed=max_items)

        errors = [it for it in items if "error" in it]
        valid_items = [it for it in items if "error" not in it]

        if not valid_items:
            return {
                "candidates": [],
                "items_fetched": 0,
                "errors": errors or None,
                "note": "No new RSS items found. Already-seen items are skipped.",
            }

        # Parallel LLM extraction for all items
        extractions = await asyncio.gather(*(
            _extract_meme_word(
                item["text"],
                item.get("source_label", item.get("feed_slug", "")),
            )
            for item in valid_items
        ))

        candidates = []
        for item, extraction in zip(valid_items, extractions):
            meme_score = extraction.get("meme_score", 0) or 0
            if meme_score >= min_score:
                candidates.append({
                    "source": "rss",
                    "feed": item.get("source_label", item.get("feed_slug")),
                    "guid": item["guid"],
                    "title": item["title"],
                    "link": item.get("link"),
                    "item_text": item["text"][:500],
                    "pub_date": item.get("pub_date"),
                    "meme_word": extraction.get("meme_word"),
                    "ticker_suggestion": extraction.get("ticker_suggestion"),
                    "category": extraction.get("category"),
                    "origin": extraction.get("origin"),
                    "meme_score": meme_score,
                    "confidence": extraction.get("confidence"),
                    "reasoning": extraction.get("reasoning"),
                })

        candidates.sort(key=lambda c: c.get("meme_score", 0), reverse=True)

        return {
            "candidates": candidates,
            "items_fetched": len(valid_items),
            "items_filtered": len(valid_items) - len(candidates),
            "min_meme_score_filter": min_score,
            "feeds_checked": list((feeds or _rss_feeds or _DEFAULT_RSS_FEEDS).keys()),
            "errors": errors or None,
            "note": (
                "These are candidates only. Present to user for approval "
                "before taking any action. Higher meme_score = higher viral potential."
            ),
        }

    async def _check_all(self, kwargs: dict) -> dict:
        """Fetch tweets + RSS in parallel, merge and rank all candidates.

        _check_tweets handles twitter_* feeds internally (via RSSHub + twikit).
        _check_rss skips twitter_* feeds to avoid duplication.
        """
        tweet_result, rss_result = await asyncio.gather(
            self._check_tweets(kwargs),
            self._check_rss(kwargs, _exclude_twitter=True),
            return_exceptions=True,
        )

        if isinstance(tweet_result, Exception):
            logger.error(f"check_tweets failed in check_all: {tweet_result}")
            tweet_result = {"candidates": [], "errors": [{"error": str(tweet_result)}]}
        if isinstance(rss_result, Exception):
            logger.error(f"check_rss failed in check_all: {rss_result}")
            rss_result = {"candidates": [], "errors": [{"error": str(rss_result)}]}

        # Tag tweet candidates with source
        for c in tweet_result.get("candidates", []):
            c.setdefault("source", "twitter")

        # Merge candidates sorted by meme_score
        all_candidates = (
            tweet_result.get("candidates", []) + rss_result.get("candidates", [])
        )
        all_candidates.sort(key=lambda c: c.get("meme_score", 0), reverse=True)

        # Merge errors
        all_errors = []
        if tweet_result.get("errors"):
            all_errors.extend(tweet_result["errors"])
        if rss_result.get("errors"):
            all_errors.extend(rss_result["errors"])

        return {
            "candidates": all_candidates,
            "tweets_fetched": tweet_result.get("tweets_fetched", 0),
            "rss_items_fetched": rss_result.get("items_fetched", 0),
            "total_filtered": (
                tweet_result.get("tweets_filtered", 0)
                + rss_result.get("items_filtered", 0)
            ),
            "min_meme_score_filter": int(kwargs.get("min_meme_score", 3)),
            "twitter_accounts_checked": tweet_result.get("accounts_checked", []),
            "rss_feeds_checked": rss_result.get("feeds_checked", []),
            "errors": all_errors or None,
            "note": (
                "Combined Twitter + RSS candidates, ranked by meme_score. "
                "Present to user for approval before taking any action."
            ),
        }

    def _start_monitor(self, kwargs: dict) -> dict:
        """Configure accounts to monitor."""
        global _monitor_active

        accounts_str = kwargs.get("accounts", "")
        if accounts_str:
            accounts = [a.strip().lstrip("@") for a in accounts_str.split(",") if a.strip()]
        else:
            accounts = list(_DEFAULT_ACCOUNTS)

        for account in accounts:
            if account not in _monitored_accounts:
                _monitored_accounts[account] = ""

        _monitor_active = True

        # Initialize RSS feeds if not already configured
        if not _rss_feeds:
            _rss_feeds.update(_DEFAULT_RSS_FEEDS)

        return {
            "status": "monitoring_configured",
            "accounts": list(_monitored_accounts.keys()),
            "rss_feeds": {k: v["label"] for k, v in _rss_feeds.items()},
            "note": (
                "Monitoring configured. Use check_all to fetch and analyze "
                "latest tweets and RSS feeds simultaneously."
            ),
        }

    def _stop_monitor(self) -> dict:
        """Stop monitoring and clear configuration."""
        global _monitor_active

        cleared_accounts = list(_monitored_accounts.keys())
        cleared_feeds = list(_rss_feeds.keys())
        _monitored_accounts.clear()
        _last_check.clear()
        _rss_feeds.clear()
        _last_rss_check.clear()
        _monitor_active = False

        return {
            "status": "monitoring_stopped",
            "cleared_accounts": cleared_accounts,
            "cleared_feeds": cleared_feeds,
            "seen_tweets_retained": len(_seen_tweet_ids),
            "seen_rss_retained": len(_seen_rss_ids),
        }

    def _status(self) -> dict:
        """Return current monitoring state."""
        now = time.time()

        account_status = {}
        for name in _monitored_accounts:
            last = _last_check.get(name)
            account_status[name] = {
                "user_id": _monitored_accounts[name] or "not_yet_resolved",
                "last_checked": (
                    datetime.fromtimestamp(last, tz=timezone.utc).isoformat()
                    if last else "never"
                ),
                "seconds_since_check": int(now - last) if last else None,
            }

        rate_limit_info = None
        if _rate_limit_reset > now:
            rate_limit_info = {
                "rate_limited": True,
                "reset_at": datetime.fromtimestamp(
                    _rate_limit_reset, tz=timezone.utc
                ).isoformat(),
                "seconds_remaining": int(_rate_limit_reset - now),
            }

        feed_status = {}
        for slug, info in (_rss_feeds or _DEFAULT_RSS_FEEDS).items():
            last = _last_rss_check.get(slug)
            feed_status[slug] = {
                "label": info["label"],
                "url": _resolve_feed_url(info),
                "last_checked": (
                    datetime.fromtimestamp(last, tz=timezone.utc).isoformat()
                    if last else "never"
                ),
                "seconds_since_check": int(now - last) if last else None,
            }

        return {
            "monitor_active": _monitor_active,
            "twitter_auth": {
                "cookies_configured": bool(_resolve_twitter_cookies_path()),
                "cookies_source": (
                    "config.json (tools.memeMonitor.twitterCookies)"
                    if (
                        (cfg := _load_meme_monitor_config())
                        and cfg.twitter_cookies
                        and any(v for v in cfg.twitter_cookies.values())
                    )
                    else (
                        f"env TWITTER_COOKIES={_TWITTER_COOKIES_PATH}"
                        if _TWITTER_COOKIES_PATH
                        else "not set — add twitterCookies to tools.memeMonitor in ~/.finclaw/config.json"
                    )
                ),
                "client_active": _twitter_client is not None,
            },
            "accounts": account_status,
            "seen_tweets_count": len(_seen_tweet_ids),
            "max_seen_tweets": _MAX_SEEN_TWEETS,
            "rate_limit": rate_limit_info,
            "rss_feeds": feed_status,
            "seen_rss_count": len(_seen_rss_ids),
            "max_seen_rss": _MAX_SEEN_RSS,
            "rsshub_base": _RSSHUB_BASE,
            "rsshub_mirrors": _RSSHUB_MIRRORS,
        }
