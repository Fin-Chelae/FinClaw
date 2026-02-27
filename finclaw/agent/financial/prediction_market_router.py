"""Prediction Market Router — LLM sub-agent for prediction market queries (Dexter pattern).

The public ``PredictionMarketRouter`` accepts a natural-language query, runs an
inner LLM with ``PredictionMarketTool`` as the sole inner tool, and returns
a synthesised prediction market analysis.

Routes to:
  - PredictionMarketDataTool (Polymarket + Kalshi) for market data
  - PredictionMarketTool for command-based dispatch
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool


class PredictionMarketRouter(LLMRouterTool):
    """Prediction market sub-agent: Polymarket + Kalshi data and comparison.

    Accepts a natural-language query; the inner LLM orchestrates multi-step
    operations via ``prediction_market_query``.
    """

    name = "prediction_market"
    description = (
        "Prediction market data from Polymarket and Kalshi. "
        "Use to find trending markets, search by topic, compare odds across platforms, "
        "get historical probability data, and filter by category."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language query. Examples: "
                    "'What are the hottest prediction markets right now?', "
                    "'Compare Polymarket and Kalshi odds on the next Fed rate decision', "
                    "'Show me election prediction markets', "
                    "'What is the probability of a recession by 2027?', "
                    "'Show historical probability for the Trump election market'."
                ),
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a prediction market analyst with access to Polymarket and Kalshi data.\n\n"
        "Use prediction_market_query to execute operations.\n\n"
        "COMMAND SELECTION RULES:\n"
        "1. Known event slug or ticker → market_detail (current outcomes/odds).\n"
        "   NEVER use search for a known slug — it does keyword matching and will miss.\n"
        "2. Discovery (no known slug) → search or trending.\n"
        "3. Cross-platform comparison → compare with a topic query.\n"
        "4. Price history for a known slug → market_history (detail + history in ONE call).\n"
        "5. 'What's hot?' + 'how did odds change?' → top_mover (trending + detail + history).\n\n"
        "SELF-DISCOVERY WORKFLOW (history without a known slug):\n"
        "When the user asks about probability history / change over time for a market\n"
        "you don't have a slug for, use TWO turns:\n"
        "  Turn 1 → search(query='<topic>') to find the market and its slug.\n"
        "  Turn 2 → market_history(market_id='<slug from top result>', interval='1m')\n"
        "           to get current odds + 1-month probability history.\n"
        "NEVER tell the user you need a URL, link, or slug. Always search first.\n\n"
        "LADDER-RUNG FALLBACK (market_history returns 'No clob_token_ids found'):\n"
        "Polymarket ladder markets (e.g. 'US strikes Iran by Feb 28') are sub-markets\n"
        "of a parent event. The rung's event detail often lacks token IDs, but\n"
        "market_history will automatically retry via search. If it still fails:\n"
        "  Turn A → search(query='<event title keywords>') — the /markets endpoint\n"
        "            returns flat records that carry clob_token_ids.\n"
        "  Turn B → history(token_id='<clob_token_ids[0] from search result>',\n"
        "                    interval='1m') to fetch the 1-month price series.\n"
        "  Identify the matching market by comparing the question text to the user's rung.\n\n"
        "SUMMARISATION RULES:\n"
        "- Probabilities are 0-1 (1.00 = 100%). Raw data is preserved in a cache file.\n"
        "- For history responses: use the pre-computed 'summary' object (start, end,\n"
        "  min, max, change, change_pct). Do NOT enumerate individual data points.\n"
        "- Keep your synthesis to 2-4 sentences covering the key finding and trend.\n"
        "- Use plain language. Refer to bets by their title (e.g. 'US strikes Iran by…').\n"
        "  Use internal jargon very sparingly (like 'ladder parent market', 'rung', 'sub-market'),\n"
        "  Just say the name of the bet."
    )

    def _build_inner_tools(self) -> list[Tool]:
        from finclaw.agent.financial_tools.prediction_market.prediction_market_tool import (
            PredictionMarketTool,
        )
        return [PredictionMarketTool()]

    @staticmethod
    def _extract_data_excerpt(payload: Any, max_items: int = 5) -> str:
        """Override: format market data as human-readable text, not raw JSON.

        When synthesis returns empty, the main agent needs plain language it
        can quote directly — not a JSON blob it has to parse.

        Collects balanced results from each platform so both Polymarket and
        Kalshi appear in the output.
        """
        def _extract_flat(obj: Any) -> list[dict]:
            """Recursively collect market/history dicts from a single platform result."""
            found: list[dict] = []
            if isinstance(obj, list):
                for item in obj:
                    found.extend(_extract_flat(item))
            elif isinstance(obj, dict):
                for key in ("markets", "results", "data"):
                    items = obj.get(key)
                    if isinstance(items, list):
                        found.extend(items)
                if "summary" in obj and isinstance(obj["summary"], dict):
                    found.append(obj)
            return found

        def _collect_balanced(obj: Any) -> list[dict]:
            """Collect markets balanced across platforms (up to ceil(max_items/2) each)."""
            poly: list[dict] = []
            kalshi: list[dict] = []

            if isinstance(obj, dict):
                # Combined result with nested platform keys
                p = obj.get("polymarket")
                if isinstance(p, dict):
                    poly.extend(_extract_flat(p))
                k = obj.get("kalshi")
                if isinstance(k, dict):
                    kalshi.extend(_extract_flat(k))
                # Single-platform result (e.g. poly_trending with source="polymarket")
                if not poly and not kalshi:
                    source = obj.get("source", "")
                    items = _extract_flat(obj)
                    if source == "kalshi":
                        kalshi.extend(items)
                    else:
                        poly.extend(items)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        source = item.get("source", "")
                        items = _extract_flat(item)
                        if source == "kalshi":
                            kalshi.extend(items)
                        else:
                            poly.extend(items)

            half = (max_items + 1) // 2
            return poly[:half] + kalshi[:half]

        markets = _collect_balanced(payload)[:max_items]
        if not markets:
            return ""

        lines: list[str] = []
        for m in markets:
            # Prediction market result (question/title + outcomes or probability)
            if "question" in m or "title" in m:
                label = m.get("question") or m.get("title") or ""
                platform = m.get("platform", "")
                url = m.get("url", "")
                if not url and m.get("slug"):
                    url = f"https://polymarket.com/market/{m['slug']}"
                url_str = f"  {url}" if url else ""
                outcomes = m.get("outcomes", [])
                if outcomes:
                    odds = ", ".join(
                        f"{o.get('outcome','?')} {round(o.get('probability', 0)*100,1)}%"
                        for o in outcomes[:4]
                    )
                    lines.append(f"- {label} [{platform}]: {odds}{url_str}")
                else:
                    prob = m.get("yes_probability") or m.get("top_market_probability")
                    prob_str = f"{round(prob*100,1)}%" if prob else ""
                    vol = m.get("volume_24h", 0)
                    lines.append(
                        f"- {label} [{platform}]: {prob_str}"
                        + (f"  (24h vol: {vol:,.0f})" if vol else "")
                        + url_str
                    )
            # History object
            elif "summary" in m and isinstance(m.get("summary"), dict):
                s = m["summary"]
                start = s.get("start", {}).get("probability", "?")
                end = s.get("end", {}).get("probability", "?")
                change = s.get("change_pct", "?")
                lines.append(
                    f"Probability history: started at {start}, now {end} "
                    f"(change: {change}%)"
                )

        if not lines:
            # Last resort: return compact JSON so something is surfaced
            try:
                return json.dumps(markets, ensure_ascii=False, default=str)
            except Exception:
                return ""

        return "\n".join(lines)

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"prediction_market (inner-LLM): {query[:120]}")
        return await self._run_inner_agent(query)
