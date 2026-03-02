"""Financial router tools — LLM sub-agent pattern (Dexter-inspired).

Each router exposes a single natural-language ``query`` parameter to the main LLM.
Internally it runs a focused inner LLM that has access to specialized data tools,
calls them in parallel, and returns the combined result.

Data source status:
  ✅ yfinance   - US/global quotes, historical data, fundamentals
  ✅ akshare    - Chinese A-share quotes, historical, financials, sectors, indices
  ✅ sec_edgar  - SEC EDGAR 10-Q/10-K filings (daily list, ticker filings, full parse)
"""

from pathlib import Path
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool
from finclaw.providers.base import LLMProvider

# Keywords that indicate an earnings / filing query.
_FILING_KEYWORDS: frozenset[str] = frozenset({
    "10-q", "10-k", "earnings", "annual report",
    "financial result", "财报", "filing", "md&a", "10q", "10k",
    "quarterly result", "earnings release", "季报", "年报", "业绩",
})


def _is_filing_query(query: str) -> bool:
    """Return True if the query appears to be about earnings/filings."""
    low = query.lower()
    return any(kw in low for kw in _FILING_KEYWORDS)


# Appended to the tool result so the OUTER LLM (Claude) receives explicit
# format instructions and reliably produces a structured table in its reply.
_EARNINGS_FORMAT_HINT = """
---
[FORMAT INSTRUCTION — fill in the table below using the data above]

| Metric           | Actual | YoY Δ | QoQ Δ |
|------------------|--------|-------|-------|
| Revenue          |        |       |       |
| Gross Profit     |        |       |       |
| Gross Margin     |        | (pp)  | (pp)  |
| Operating Income |        |       |       |
| Operating Margin |        | (pp)  | (pp)  |
| Net Income       |        |       |       |
| EPS (diluted)    |        |       |       |

**EPS vs Consensus:** [actual] vs [estimate] → Beat / Miss by [surprise %]
**Analyst Price Target (mean):** $XX  |  **Ratings:** strongBuy=X  hold=X  sell=X
"""


class FinancialMetricsRouter(LLMRouterTool):
    """Query company financial metrics via an inner LLM sub-agent."""

    name = "financial_metrics"
    description = (
        "Fetch structured financial metrics and ratios for a company. "
        "Supports US stocks (AAPL, NVDA) and Chinese A-shares (600519, 000651). "
        "Use for: historical income statements, balance sheets, cash flow statements, "
        "valuation ratios (P/E, P/B), analyst estimates, insider trades, segment data. "
        "Do NOT use for earnings analysis or quarterly results review — "
        "use financial_search instead, which also fetches the actual SEC filing text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of the financial data to fetch. "
                    "Examples: 'Get AAPL income statement for last 3 years', "
                    "'Fetch 600519 quarterly balance sheet and key ratios', "
                    "'Compare NVDA and AMD cash flow statements'."
                ),
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a financial data specialist. Fetch the requested metrics precisely.\n\n"
        "Tool selection:\n"
        "  yfinance_tool  → US / global stocks (AAPL, NVDA, MSFT, …)\n"
        "  akshare_tool   → Chinese A-shares (600519, 000651, …)\n\n"
        "yfinance_tool commands: info, quote, historical, financials, financial_ratios, "
        "batch_quotes, analyst_estimates\n"
        "akshare_tool  commands: quote, historical, info, financials, news, search, "
        "sector_performance, index_quotes\n\n"
        "For earnings / quarterly results queries: call both financials AND analyst_estimates "
        "in parallel — analyst_estimates returns EPS consensus, beat/miss history, "
        "analyst ratings, and price targets.\n\n"
        "Fetch data comprehensively. Call multiple tools in parallel when several "
        "metrics or tickers are requested. "
        "Summarise the key findings in 2-4 sentences. Raw data is preserved separately."
    )

    def _build_inner_tools(self) -> list[Tool]:
        from finclaw.agent.financial_tools import YFinanceTool, AKShareTool
        return [YFinanceTool(), AKShareTool()]

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"financial_metrics (inner-LLM): {query[:120]}")
        return await self._run_inner_agent(query)


class FinancialSearchRouter(LLMRouterTool):
    """Search for financial data, market info, and SEC filings via an inner LLM sub-agent."""

    name = "financial_search"
    description = (
        "Search for financial data, market info, news, and SEC filings. "
        "Supports US stocks, Chinese A-shares, FX pairs, and SEC EDGAR. "
        "Use for: real-time quotes, historical OHLCV, company facts, stock news, "
        "sector rankings, index quotes, and 10-K/10-Q filings. "
        "IMPORTANT: Use this tool for any earnings analysis, quarterly results review, "
        "财报分析, or 季报/年报 request — it fetches the actual SEC filing text (MD&A, "
        "risk factors, guidance) alongside analyst consensus data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language query. Examples: "
                    "'Get AAPL current price and latest news', "
                    "'Search for 茅台 A-share price and company info', "
                    "'Fetch Apple latest 10-K annual report from SEC', "
                    "'Show trending A-share sectors today', "
                    "'AAPL vs MSFT 30-day historical price data'."
                ),
            },
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a financial search specialist. Retrieve all requested data accurately.\n\n"
        "Available tools:\n"
        "  yfinance_tool   → US/global: quote, historical, info, batch_quotes, financials, "
        "financial_ratios, analyst_estimates\n"
        "  akshare_tool    → A-shares:  quote, historical, info, news, search, sector_performance, "
        "index_quotes, financials\n"
        "  sec_edgar_tool  → SEC EDGAR: ticker_filings, fetch_and_parse, daily_parsed\n"
        "  web_search      → general web search (fallback for US stock news if configured)\n\n"
        "Rules:\n"
        "  - For A-share stocks use numeric code: '600519', NOT '600519.SS'\n"
        "  - For SEC filings: call ticker_filings first to list available filings, "
        "then select the filing with the MOST RECENT date (sort by date descending, pick index 0). "
        "Never fetch a filing that is more than 6 months old when a newer one exists.\n"
        "  - For any earnings/quarterly results/财报 request: call three tools in parallel:\n"
        "      (1) sec_edgar ticker_filings → fetch_and_parse  (filing text: MD&A, guidance)\n"
        "      (2) yfinance financials  — returns both income_statement (annual) AND "
        "quarterly_income_statement (last 4-5 quarters). Use quarterly_income_statement "
        "to fill YoY Δ (current quarter vs same quarter last year) and QoQ Δ "
        "(current quarter vs previous quarter) in the output table.\n"
        "      (3) yfinance analyst_estimates (EPS consensus, beat/miss, ratings, price target)\n"
        "  - Fetch all requested data in parallel when multiple pieces are needed\n\n"
        "For earnings / quarterly results queries, your response MUST start with "
        "the filled-in table below — do not write a prose summary first. "
        "Pull exact numbers from quarterly_income_statement for Actual/YoY/QoQ. "
        "Then add 2-3 sentences of key findings.\n\n"
        "| Metric           | Actual | YoY Δ | QoQ Δ |\n"
        "|------------------|--------|-------|-------|\n"
        "| Revenue          |        |       |       |\n"
        "| Gross Profit     |        |       |       |\n"
        "| Gross Margin     |        | (pp)  | (pp)  |\n"
        "| Operating Income |        |       |       |\n"
        "| Operating Margin |        | (pp)  | (pp)  |\n"
        "| Net Income       |        |       |       |\n"
        "| EPS (diluted)    |        |       |       |\n\n"
        "**EPS vs Consensus:** [actual] vs [estimate] → Beat/Miss\n"
        "**Analyst Price Target (mean):** $XX  |  "
        "**Ratings:** strongBuy=X  hold=X  sell=X\n\n"
        "For all other queries: summarise in 2-4 sentences. Raw data is preserved separately."
    )

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        inner_model: str = "",
        inner_provider: LLMProvider | None = None,
        search_tool: Tool | None = None,
        workspace: Path | None = None,
    ) -> None:
        super().__init__(provider, model, inner_model, inner_provider, workspace)
        self._search_tool = search_tool

    def _build_inner_tools(self) -> list[Tool]:
        from finclaw.agent.financial_tools import YFinanceTool, AKShareTool, SecEdgarTool
        tools: list[Tool] = [YFinanceTool(), AKShareTool(), SecEdgarTool()]
        if self._search_tool is not None:
            tools.append(self._search_tool)
        return tools

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"financial_search (inner-LLM): {query[:120]}")
        result = await self._run_inner_agent(query)
        if _is_filing_query(query):
            return result + _EARNINGS_FORMAT_HINT
        return result
