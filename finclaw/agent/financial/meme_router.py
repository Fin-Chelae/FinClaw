"""Meme Router — LLM sub-agent for meme coin operations (Dexter pattern).

The public ``MemeRouter`` is an LLM sub-agent: it accepts a natural-language query,
runs an inner LLM with ``_MemeDispatch`` as the sole inner tool, and returns a
synthesised meme coin analysis.

``_MemeDispatch`` (private) contains the original command-based dispatch logic and
is exposed to the inner LLM as "meme_dispatch".  The inner LLM orchestrates multi-step
operations (e.g., find trending coins, then scan social media for each).

Routes to:
  - MemeDataTool (DexScreener + CoinGecko) for market data
  - MemeSearchTool (Twitter + RSS) for social scanning
  - MemeCreateTool for token creation (pump.fun / four.meme)
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool

# Lazy-initialized backends (created on first use, not at import time)
_search: Any = None
_data: Any = None
_create: Any = None


def _get_search() -> Any:
    """Lazy-init MemeSearchTool singleton."""
    global _search
    if _search is None:
        from finclaw.agent.financial_tools.meme.meme_search_tool import MemeSearchTool
        _search = MemeSearchTool()
    return _search


def _get_data() -> Any:
    """Lazy-init MemeDataTool singleton."""
    global _data
    if _data is None:
        from finclaw.agent.financial_tools.meme.meme_data_tool import MemeDataTool
        _data = MemeDataTool()
    return _data


def _get_create() -> Any:
    """Lazy-init MemeCreateTool singleton."""
    global _create
    if _create is None:
        from finclaw.agent.financial_tools.meme.meme_create_tool import MemeCreateTool
        _create = MemeCreateTool()
    return _create


class _MemeDispatch(Tool):
    """Inner tool: command-based meme coin dispatch (used by MemeRouter sub-agent).

    Exposed to the inner LLM as ``meme_dispatch``.
    """

    @property
    def name(self) -> str:
        return "meme_dispatch"

    @property
    def description(self) -> str:
        return (
            "Execute a meme coin operation. "
            "Commands: search (find tokens), price (market data), trending, info (coin details), "
            "launch_scan (scan all social sources), check_tweets, check_rss, analyze_text, "
            "polymarket (prediction market signals), check_env, create (deploy token), status. "
            "Supports pump.fun (Solana) and four.meme (BSC) for token creation. "
            "For create operations: call check_env first, then create. "
            "The create command returns a confirmation token — present it to the user. "
            "Use confirm/cancel with the token once the user responds."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "search", "price", "trending", "info",
                        "launch_scan", "check_tweets", "check_rss",
                        "analyze_text",
                        "start_monitor", "stop_monitor", "status",
                        "polymarket",
                        "check_env", "create", "confirm", "cancel",
                    ],
                    "description": (
                        "search: find tokens by name/symbol; "
                        "price: get price/market data; "
                        "trending: trending meme coins; "
                        "info: detailed coin info; "
                        "launch_scan: scan all social sources for launch ideas; "
                        "check_tweets: Twitter scanning only; "
                        "check_rss: RSS feeds only; "
                        "analyze_text: analyze text for meme potential; "
                        "polymarket: top prediction markets by 24h volume (meme catalyst signals); "
                        "start_monitor/stop_monitor: configure monitoring; "
                        "status: show all status; "
                        "check_env: verify wallet credentials are set before creating a token; "
                        "create: deploy a new memecoin on pump.fun (Solana) or four.meme (BSC). "
                        "NOTE: if the user provides a token name/symbol, skip 'search' and go directly to check_env → create."
                    ),
                },
                "confirmation_token": {
                    "type": "string",
                    "description": "Confirmation token for confirm/cancel commands.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Token name, symbol, or search term. "
                        "Required for search, price, info."
                    ),
                },
                "coin_id": {
                    "type": "string",
                    "description": (
                        "CoinGecko coin ID for 'info' command "
                        "(e.g. 'dogecoin', 'pepe'). "
                        "If not provided, 'query' is used to search first."
                    ),
                },
                "chain": {
                    "type": "string",
                    "description": (
                        "Blockchain for DexScreener lookups "
                        "(e.g. 'solana', 'ethereum', 'bsc'). "
                        "Default: search across all chains."
                    ),
                },
                "token_address": {
                    "type": "string",
                    "description": (
                        "Token contract address for direct lookup. "
                        "Used with 'chain' for DexScreener pair data."
                    ),
                },
                # Pass-through params for social scanning commands
                "accounts": {
                    "type": "string",
                    "description": (
                        "Comma-separated Twitter screen names for "
                        "check_tweets/start_monitor."
                    ),
                },
                "tweet_text": {
                    "type": "string",
                    "description": "Text to analyze for meme potential (analyze_text).",
                },
                "screen_name": {
                    "type": "string",
                    "description": "Author context for analyze_text.",
                },
                "max_tweets": {
                    "type": "integer",
                    "description": "Max tweets per account. Default: 10.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "min_meme_score": {
                    "type": "integer",
                    "description": "Minimum meme score filter (1-10). Default: 3.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "feeds": {
                    "type": "string",
                    "description": "Comma-separated RSS feed slugs for check_rss.",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Max items per RSS feed. Default: 10.",
                    "minimum": 1,
                    "maximum": 50,
                },
                # Token creation params (for 'create' command)
                "platform": {
                    "type": "string",
                    "enum": ["pump.fun", "four.meme"],
                    "description": "Target platform for 'create'. pump.fun = Solana, four.meme = BSC. Required for 'create'.",
                },
                "token_name": {
                    "type": "string",
                    "description": "Token name for 'create'. When this is provided, do NOT call 'search' first.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Token ticker symbol. Required for 'create'.",
                },
                "token_description": {
                    "type": "string",
                    "description": "Short token description. Required for 'create'.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to logo image (PNG/JPG/GIF). Required for 'create'.",
                },
                "buy_amount": {
                    "type": "number",
                    "description": "Initial buy in SOL (pump.fun only). Default: 0.01.",
                },
                "slippage_bps": {
                    "type": "integer",
                    "description": "Slippage in basis points (pump.fun only). Default: 10.",
                },
                "priority_fee": {
                    "type": "number",
                    "description": "Priority fee in SOL (pump.fun only). Default: 0.0005.",
                },
                "label": {
                    "type": "string",
                    "enum": ["Meme", "AI", "Defi", "Games", "Infra", "De-Sci", "Social", "Depin", "Charity", "Others"],
                    "description": "Token category label (four.meme only). Default: 'Meme'.",
                },
                "presale_bnb": {
                    "type": "number",
                    "description": "Creator presale amount in BNB (four.meme only). Default: 0.",
                },
                "twitter": {
                    "type": "string",
                    "description": "Twitter/X URL for token (optional, for 'create').",
                },
                "telegram": {
                    "type": "string",
                    "description": "Telegram URL for token (optional, for 'create').",
                },
                "website": {
                    "type": "string",
                    "description": "Website URL for token (optional, for 'create').",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        logger.info(f"meme:{command}")

        result: dict[str, Any] = {"command": command}

        try:
            if command == "search":
                result.update(await self._run_search(kwargs))
            elif command == "price":
                result.update(await self._run_price(kwargs))
            elif command == "trending":
                result.update(await self._run_trending(kwargs))
            elif command == "info":
                result.update(await self._run_info(kwargs))
            elif command == "launch_scan":
                raw = await _get_search().execute(
                    command="check_all", **_passthrough(kwargs),
                )
                result.update(json.loads(raw))
            elif command == "check_tweets":
                raw = await _get_search().execute(
                    command="check_tweets", **_passthrough(kwargs),
                )
                result.update(json.loads(raw))
            elif command == "check_rss":
                raw = await _get_search().execute(
                    command="check_rss", **_passthrough(kwargs),
                )
                result.update(json.loads(raw))
            elif command == "analyze_text":
                raw = await _get_search().execute(
                    command="analyze_tweet",
                    tweet_text=kwargs.get("tweet_text", ""),
                    screen_name=kwargs.get("screen_name", ""),
                )
                result.update(json.loads(raw))
            elif command == "start_monitor":
                raw = await _get_search().execute(
                    command="start_monitor", **_passthrough(kwargs),
                )
                result.update(json.loads(raw))
            elif command == "stop_monitor":
                raw = await _get_search().execute(command="stop_monitor")
                result.update(json.loads(raw))
            elif command == "polymarket":
                result.update(await _get_data().polymarket_trending(
                    top_n=int(kwargs.get("max_items", 20))
                ))
            elif command == "status":
                result.update(await self._run_status())
            elif command == "check_env":
                raw = await _get_create().execute(
                    command="check_env",
                    platform=kwargs.get("platform"),
                )
                result.update(json.loads(raw))
            elif command == "create":
                raw = await _get_create().execute(**kwargs)
                result.update(json.loads(raw))
            elif command in ("confirm", "cancel"):
                raw = await _get_create().execute(**kwargs)
                result.update(json.loads(raw))
            else:
                result["error"] = f"Unknown command: {command!r}"
        except Exception as exc:
            logger.warning(f"meme error ({command}): {exc}")
            result["error"] = str(exc)

        return json.dumps(result, ensure_ascii=False, default=str)

    # ── Data commands ────────────────────────────────────────────────

    async def _run_search(self, kwargs: dict) -> dict:
        query = kwargs.get("query", "")
        if not query:
            return {"error": "query is required for search command"}
        return await _get_data().search_token(query)

    async def _run_price(self, kwargs: dict) -> dict:
        chain = kwargs.get("chain", "")
        token_address = kwargs.get("token_address", "")

        if token_address and chain:
            return await _get_data().dex_token_pairs(chain, token_address)

        query = kwargs.get("query", "")
        if not query:
            return {"error": "query or (chain + token_address) required for price"}
        # Search DexScreener — returns pairs with price data
        return await _get_data().dex_search(query)

    async def _run_trending(self, kwargs: dict) -> dict:
        return await _get_data().get_trending()

    async def _run_info(self, kwargs: dict) -> dict:
        coin_id = kwargs.get("coin_id", "")
        if not coin_id:
            query = kwargs.get("query", "")
            if not query:
                return {"error": "coin_id or query is required for info"}
            # Search CoinGecko to resolve the coin_id
            search = await _get_data().cg_search(query)
            coins = search.get("coins", [])
            if not coins:
                return {"error": f"No coins found for '{query}' on CoinGecko"}
            coin_id = coins[0].get("id", "")
        return await _get_data().cg_coin_info(coin_id)

    async def _run_status(self) -> dict:
        monitor_raw = await _get_search().execute(command="status")
        monitor_status = json.loads(monitor_raw)
        api_health = await _get_data().health_check()
        return {
            "monitor": monitor_status,
            "api_health": api_health,
        }


def _passthrough(kwargs: dict) -> dict:
    """Extract monitor-relevant kwargs, excluding router-specific params."""
    exclude = {"command", "query", "coin_id", "chain", "token_address"}
    return {k: v for k, v in kwargs.items() if k not in exclude}


# ---------------------------------------------------------------------------
# Public sub-agent wrapper (Dexter pattern)
# ---------------------------------------------------------------------------

class MemeRouter(LLMRouterTool):
    """Meme coin sub-agent: market data + social scanning.

    Accepts a natural-language query; the inner LLM orchestrates multi-step
    operations via ``meme_dispatch`` (e.g., find trending coins, then scan
    social media for each candidate).
    """

    name = "meme"
    description = (
        "Meme coin data, social scanning, and token deployment. "
        "Use to find tokens, get prices, see trending coins, scan social media "
        "for viral content and launch candidates, check Polymarket signals, "
        "and deploy new meme coins on pump.fun (Solana) or four.meme (BSC)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language query. Examples: "
                    "'What meme coins are trending right now?', "
                    "'Find PEPE token price and market data', "
                    "'Scan social media for new meme coin launch ideas', "
                    "'Check Polymarket for meme catalyst signals', "
                    "'Create token: name=\\'Moon Cat\\', symbol=\\'MCAT\\', "
                    "description=\\'A lunar cat\\', image_path=\\'/path/to/img.png\\', platform=\\'pump.fun\\'' or platform=\\'four.meme\\'."
                ),
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a meme coin market analyst with access to on-chain and social data.\n\n"
        "Use meme_dispatch to execute operations. Available commands:\n"
        "  search        — find tokens by name/symbol (DexScreener + CoinGecko)\n"
        "  price         — get price, volume, liquidity, market cap\n"
        "  trending      — currently trending meme coins\n"
        "  info          — detailed coin metadata (use coin_id for CoinGecko lookup)\n"
        "  launch_scan   — scan all social sources (Twitter + RSS) for launch candidates\n"
        "  check_tweets  — Twitter scanning only\n"
        "  check_rss     — RSS feeds only\n"
        "  analyze_text  — evaluate a specific text for meme potential\n"
        "  polymarket    — top Polymarket prediction markets (meme catalyst signals)\n"
        "  check_env     — verify wallet credentials are configured (pass platform to check specific chain)\n"
        "  create        — deploy a new memecoin on pump.fun (Solana) or four.meme (BSC)\n"
        "  status        — show monitor and API health status\n\n"
        "For a comprehensive meme analysis, call trending + launch_scan in parallel.\n"
        "For token creation: call check_env, then create. The create command returns a "
        "pending confirmation with a token — present the details to the user. "
        "Use confirm/cancel with the confirmation_token once the user responds.\n"
        "For launch_scan results, present candidates for the user to choose from.\n"
        "Summarise the key findings in 2-4 sentences. Raw data is preserved separately."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from finclaw.agent.financial_tools.meme import meme_search_tool
        meme_search_tool.set_extraction_model(self._model)

    def _build_inner_tools(self) -> list[Tool]:
        return [_MemeDispatch()]

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"meme (inner-LLM): {query[:120]}")

        # Fast-path for token creation — bypass inner LLM to avoid the
        # confirmation loop.  The outer agent (skill workflow) has already
        # collected all fields and confirmed with the user.
        create_params = self._parse_create_query(query)
        if create_params:
            return await self._fast_create(create_params)

        return await self._run_inner_agent(query)

    # ── Fast-path create helpers ─────────────────────────────────────

    @staticmethod
    def _parse_create_query(query: str) -> dict[str, str] | None:
        """Extract create parameters from a ``Create token: ...`` query.

        Returns a dict of params if this is a confirmed create query,
        None otherwise (falls through to the normal inner-LLM path).
        """
        if not re.match(r"(?i)create\s+token\s*:", query):
            return None

        params: dict[str, str] = {}
        for match in re.finditer(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", query):
            params[match.group(1)] = match.group(2)

        required = {"name", "symbol", "description", "image_path"}
        if not required.issubset(params.keys()):
            return None

        params.setdefault("platform", "pump.fun")
        return params

    async def _fast_create(self, params: dict[str, str]) -> str:
        """Fast-path create — bypasses inner LLM but still goes through
        the confirmation flow in MemeCreateTool.  Returns a pending
        confirmation that the outer agent must present to the user."""
        dispatch = _MemeDispatch()
        platform = params.get("platform", "pump.fun")

        # 1. Environment check
        env_raw = await dispatch.execute(command="check_env", platform=platform)
        env_result = json.loads(env_raw)
        if not env_result.get("ready", False):
            return json.dumps({
                "summary": (
                    f"Cannot create token: environment not ready. "
                    f"Missing: {env_result.get('missing', env_result.get('platforms', []))}"
                ),
                "data": env_result,
            })

        # 2. Initiate create (returns pending_confirmation, NOT a direct deploy)
        create_raw = await dispatch.execute(command="create", **params)
        create_result = json.loads(create_raw)

        # The result will have status=pending_confirmation with a token.
        # Return it so the outer agent presents it to the user for approval.
        if create_result.get("status") == "pending_confirmation":
            details = create_result.get("details", {})
            summary = (
                f"Token deployment prepared — awaiting confirmation.\n"
                f"Name: {details.get('name')}, Symbol: {details.get('symbol')}, "
                f"Platform: {details.get('platform')}\n"
                f"Confirmation token: {create_result.get('confirmation_token')}\n"
                f"Reply with confirmation to proceed, or cancel to abort. Expires in 5 minutes."
            )
            return json.dumps(
                {"summary": summary, "data": create_result},
                ensure_ascii=False,
                default=str,
            )

        # Fallback for unexpected responses
        return json.dumps(
            {"summary": json.dumps(create_result), "data": create_result},
            ensure_ascii=False,
            default=str,
        )
