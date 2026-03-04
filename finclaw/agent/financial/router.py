"""Financial router tools — LLM sub-agent pattern (Dexter-inspired).

Each router exposes a single natural-language ``query`` parameter to the main LLM.
Internally it runs a focused inner LLM that has access to specialized data tools,
calls them in parallel, and returns the combined result.

Data source status:
  ✅ yfinance          - US/global quotes, historical data, fundamentals
  ✅ akshare           - Chinese A-share quotes, historical, financials, sectors, indices
  ✅ sec_edgar         - SEC EDGAR 10-Q/10-K filings (daily list, ticker filings, full parse)
  ✅ earnings_calendar - Earnings dates, EPS surprises, consensus estimates, revisions
"""

from pathlib import Path
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool
from finclaw.providers.base import LLMProvider


class FinancialMetricsRouter(LLMRouterTool):
    """Query company financial metrics via an inner LLM sub-agent."""

    name = "financial_metrics"
    description = (
        "Query company financial metrics, fundamental data, and earnings intelligence. "
        "Supports US stocks (AAPL, NVDA) and Chinese A-shares (600519, 000651). "
        "Use for income statements, balance sheets, cash flow, key ratios, "
        "analyst estimates, insider trades, and segment data. "
        "Also use for ALL earnings-related queries: next earnings date, EPS beat/miss history, "
        "forward EPS/revenue consensus estimates, and analyst estimate revisions. "
        "Describe what you need in plain language."
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
        "  yfinance_tool       → US / global stocks (AAPL, NVDA, MSFT, …)\n"
        "  akshare_tool        → Chinese A-shares (600519, 000651, …)\n"
        "  earnings_calendar   → Earnings dates, EPS surprises, consensus estimates, revisions\n\n"
        "yfinance_tool commands: info, quote, historical, financials, financial_ratios, batch_quotes\n"
        "akshare_tool  commands: quote, historical, info, financials, news, search, "
        "sector_performance, index_quotes\n"
        "earnings_calendar commands: calendar, upcoming, surprise, consensus, revisions\n\n"
        "Use earnings_calendar for any query about: earnings dates, when a company reports, "
        "EPS beat/miss history, analyst EPS/revenue estimates, or estimate revisions.\n\n"
        "Fetch data comprehensively. Call multiple tools in parallel when several "
        "metrics or tickers are requested. "
        "Summarise the key findings in 2-4 sentences. Raw data is preserved separately."
    )

    def _build_inner_tools(self) -> list[Tool]:
        from finclaw.agent.financial_tools import YFinanceTool, AKShareTool, EarningsCalendarTool
        return [YFinanceTool(), AKShareTool(), EarningsCalendarTool()]

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"financial_metrics (inner-LLM): {query[:120]}")
        return await self._run_inner_agent(query)


class FinancialSearchRouter(LLMRouterTool):
    """Search for financial data, market info, and SEC filings via an inner LLM sub-agent."""

    name = "financial_search"
    description = (
        "Search for financial data, company information, market data, news, and SEC filings. "
        "Supports US stocks, Chinese A-shares, FX pairs, and SEC EDGAR. "
        "Use for real-time quotes, historical OHLCV, company facts, stock news, "
        "sector rankings, index quotes, and 10-K/10-Q filings. "
        "IMPORTANT: For any 10-K, 10-Q, SEC filing, annual report, risk factors, or MD&A "
        "request, this tool fetches filing text directly from SEC EDGAR."
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
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a financial search specialist. Retrieve all requested data accurately.\n\n"
        "Available tools:\n"
        "  yfinance_tool       → US/global: quote, historical, info, batch_quotes, financials, financial_ratios\n"
        "  akshare_tool        → A-shares:  quote, historical, info, news, search, sector_performance, "
        "index_quotes, financials\n"
        "  sec_edgar_tool      → SEC EDGAR: ticker_filings, fetch_and_parse, daily_parsed\n"
        "  web_search          → general web search (fallback for US stock news if configured)\n\n"
        "Rules:\n"
        "  - For A-share stocks use numeric code: '600519', NOT '600519.SS'\n"
        "  - For SEC filings: call ticker_filings to list, then fetch_and_parse for the latest\n"
        "  - For any 10-K/10-Q/filing/MD&A/risk-factors request, always use sec_edgar_tool\n"
        "  - Fetch all requested data in parallel when multiple pieces are needed\n"
        "Summarise the key findings in 2-4 sentences. Raw data is preserved separately."
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
        return await self._run_inner_agent(query)
