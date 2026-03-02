"""YFinance Tool - stock quotes and fundamental data fetcher.

Ported from FinceptTerminal/yfinance_data.py and wrapped as a finclaw Tool.
Synchronous yfinance calls are offloaded via asyncio.to_thread for non-blocking execution.

Supported commands (command parameter):
  quote             - real-time quote for a single symbol
  batch_quotes      - batch quotes for multiple symbols (comma-separated)
  historical        - historical OHLCV data
  historical_price  - closing price for a specific date
  info              - full company info and valuation metrics
  financials        - financial statements (income, balance sheet, cash flow)
  company_profile   - company profile in FMP-compatible format
  financial_ratios  - financial ratios for peer comparison
  multiple_profiles - batch company profiles
  multiple_ratios   - batch financial ratios
  search            - symbol lookup by keyword
  resolve_symbol    - auto-resolve exchange suffix (.NS / .BO)
  analyst_estimates - analyst consensus estimates, price targets, EPS trend,
                      beat/miss history, and recommendations
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote as url_quote

import httpx
import pandas as pd
import yfinance as yf
from loguru import logger

from finclaw.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Pure function layer (sync) - ported from FinceptTerminal/yfinance_data.py
# ---------------------------------------------------------------------------

def _get_quote(symbol: str) -> dict:
    """Fetch real-time quote for a single symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d")

        if hist.empty:
            return {"error": "No data available", "symbol": symbol}

        current_price = hist["Close"].iloc[-1]
        previous_close = info.get("previousClose", current_price)
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close else 0

        return {
            "symbol": symbol,
            "price": round(float(current_price), 2),
            "change": round(float(change), 2),
            "change_percent": round(float(change_percent), 2),
            "volume": int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None,
            "high": round(float(hist["High"].iloc[-1]), 2) if not hist["High"].empty else None,
            "low": round(float(hist["Low"].iloc[-1]), 2) if not hist["Low"].empty else None,
            "open": round(float(hist["Open"].iloc[-1]), 2) if not hist["Open"].empty else None,
            "previous_close": round(float(previous_close), 2),
            "timestamp": int(datetime.now().timestamp()),
            "exchange": info.get("exchange", ""),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_batch_quotes(symbols: list[str]) -> list:
    """Fetch quotes for multiple symbols in a single HTTP request."""
    try:
        data = yf.download(
            symbols, period="2d", group_by="ticker",
            progress=False, threads=True, auto_adjust=True
        )
        if data is None or data.empty:
            return []

        results = []
        for symbol in symbols:
            try:
                hist = data if len(symbols) == 1 else (
                    data[symbol] if symbol in data.columns.get_level_values(0) else None
                )
                if hist is None or hist.empty or hist.dropna(how="all").empty:
                    continue

                hist = hist.dropna(how="all")
                current_price = float(hist["Close"].iloc[-1])
                previous_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close else 0

                results.append({
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "volume": int(hist["Volume"].iloc[-1]) if not pd.isna(hist["Volume"].iloc[-1]) else 0,
                    "high": round(float(hist["High"].iloc[-1]), 2) if not pd.isna(hist["High"].iloc[-1]) else None,
                    "low": round(float(hist["Low"].iloc[-1]), 2) if not pd.isna(hist["Low"].iloc[-1]) else None,
                    "open": round(float(hist["Open"].iloc[-1]), 2) if not pd.isna(hist["Open"].iloc[-1]) else None,
                    "previous_close": round(previous_close, 2),
                    "timestamp": int(datetime.now().timestamp()),
                    "exchange": "",
                })
            except Exception:
                continue

        return results
    except Exception:
        # Batch fetch failed, fall back to individual requests (preserve all results)
        return [_get_quote(s) for s in symbols]


def _get_historical(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> list | dict:
    """Fetch historical OHLCV data. Valid intervals: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date, interval=interval)

        if hist.empty:
            return []

        return [
            {
                "symbol": symbol,
                "timestamp": int(index.timestamp()),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "adj_close": round(float(row["Close"]), 2),
            }
            for index, row in hist.iterrows()
        ]
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_historical_price(symbol: str, target_date: str) -> dict:
    """Fetch closing price for a specific date, or the nearest trading day."""
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        start = target - timedelta(days=5)
        end = target + timedelta(days=1)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
        )

        if hist.empty:
            return {"found": False, "error": "No data for this date range", "symbol": symbol}

        closest_date = closest_price = None
        for index, row in hist.iterrows():
            idx_date = index.to_pydatetime().replace(tzinfo=None)
            if idx_date.date() <= target.date():
                closest_date = idx_date
                closest_price = round(float(row["Close"]), 2)

        if closest_price is None:
            first_row = hist.iloc[0]
            closest_date = hist.index[0].to_pydatetime()
            closest_price = round(float(first_row["Close"]), 2)

        return {
            "found": True,
            "symbol": symbol,
            "price": closest_price,
            "date": closest_date.strftime("%Y-%m-%d"),
            "requested_date": target_date,
        }
    except Exception as e:
        return {"found": False, "error": str(e), "symbol": symbol}


def _get_info(symbol: str) -> dict:
    """Fetch comprehensive company information and valuation metrics."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        book_value = info.get("bookValue")
        shares_outstanding = info.get("sharesOutstanding")

        return {
            "symbol": symbol,
            "company_name": info.get("longName", info.get("shortName", "N/A")),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "average_volume": info.get("averageVolume"),
            "description": info.get("longBusinessSummary", "N/A"),
            "website": info.get("website", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "employees": info.get("fullTimeEmployees"),
            "current_price": info.get("currentPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_mean_price": info.get("targetMeanPrice"),
            "recommendation_mean": info.get("recommendationMean"),
            "recommendation_key": info.get("recommendationKey"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "total_revenue": info.get("totalRevenue"),
            "revenue_per_share": info.get("revenuePerShare"),
            "return_on_assets": info.get("returnOnAssets"),
            "return_on_equity": info.get("returnOnEquity"),
            "gross_profits": info.get("grossProfits"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "profit_margins": info.get("profitMargins"),
            "book_value": book_value,
            "price_to_book": info.get("priceToBook"),
            "enterprise_value": info.get("enterpriseValue"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "shares_outstanding": shares_outstanding,
            "float_shares": info.get("floatShares"),
            "held_percent_insiders": info.get("heldPercentInsiders"),
            "held_percent_institutions": info.get("heldPercentInstitutions"),
            "short_ratio": info.get("shortRatio"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "peg_ratio": info.get("trailingPegRatio"),
            "total_assets": info.get("totalAssets"),
            "total_liabilities": info.get("totalLiab", info.get("totalLiabilities")),
            "book_value_total": (
                book_value * shares_outstanding
                if book_value and shares_outstanding
                else None
            ),
            "timestamp": int(datetime.now().timestamp()),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_financials(symbol: str) -> dict:
    """Fetch income statement, balance sheet, and cash flow statement."""
    try:
        ticker = yf.Ticker(symbol)

        def df_to_dict(df: pd.DataFrame) -> dict:
            if df is None or df.empty:
                return {}
            result = {}
            for col in df.columns:
                result[str(col)] = {
                    str(idx): (
                        float(df.loc[idx, col])
                        if isinstance(df.loc[idx, col], (int, float))
                        else str(df.loc[idx, col])
                    )
                    for idx in df.index
                    if pd.notna(df.loc[idx, col])
                }
            return result

        return {
            "symbol": symbol,
            "income_statement": df_to_dict(ticker.financials),
            "quarterly_income_statement": df_to_dict(ticker.quarterly_financials),
            "balance_sheet": df_to_dict(ticker.balance_sheet),
            "cash_flow": df_to_dict(ticker.cashflow),
            "timestamp": int(datetime.now().timestamp()),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_company_profile(symbol: str) -> dict:
    """Fetch company profile in FMP-compatible format, suitable for peer comparison."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or "symbol" not in info:
            return {"error": f"No data found for symbol: {symbol}"}

        return {
            "symbol": info.get("symbol", symbol),
            "companyName": info.get("longName", info.get("shortName", "")),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "website": info.get("website", ""),
            "description": info.get("longBusinessSummary", ""),
            "exchange": info.get("exchange", ""),
            "country": info.get("country", ""),
            "city": info.get("city", ""),
            "address": info.get("address1", ""),
            "phone": info.get("phone", ""),
            "marketCap": info.get("marketCap", 0),
            "employees": info.get("fullTimeEmployees", 0),
            "currency": info.get("currency", "USD"),
            "beta": info.get("beta", 0),
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "changes": info.get("regularMarketChangePercent", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_financial_ratios(symbol: str) -> dict:
    """Fetch financial ratios for peer comparison."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or "symbol" not in info:
            return {"error": f"No data found for symbol: {symbol}"}

        free_cashflow = info.get("freeCashflow", 0)
        shares_outstanding = info.get("sharesOutstanding", 1)
        fcf_per_share = free_cashflow / shares_outstanding if shares_outstanding else 0

        return {
            "symbol": symbol,
            "peRatio": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "priceToSales": info.get("priceToSalesTrailing12Months"),
            "pegRatio": info.get("trailingPegRatio"),
            "debtToEquity": info.get("debtToEquity"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "profitMargin": info.get("profitMargins"),
            "operatingMargin": info.get("operatingMargins"),
            "grossMargin": info.get("grossMargins"),
            "currentRatio": info.get("currentRatio"),
            "quickRatio": info.get("quickRatio"),
            "dividendYield": info.get("dividendYield"),
            "revenuePerShare": info.get("revenuePerShare"),
            "bookValuePerShare": info.get("bookValue"),
            "freeCashFlowPerShare": fcf_per_share if shares_outstanding else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _search_symbols(query: str, limit: int = 20) -> dict:
    """Search for stock symbols by keyword using the Yahoo Finance search API."""
    try:
        results = []
        query_str = query.strip()
        if not query_str:
            return {"results": [], "query": query, "count": 0}

        url = (
            f"https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={url_quote(query_str)}&quotesCount={limit}&newsCount=0"
            f"&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query"
        )

        try:
            with httpx.Client(timeout=5, headers={"User-Agent": "Mozilla/5.0"}) as client:
                response = client.get(url)
                data = response.json()
                for q in data.get("quotes", []):
                    symbol = q.get("symbol", "")
                    if not symbol:
                        continue
                    results.append({
                        "symbol": symbol,
                        "name": q.get("longname") or q.get("shortname", ""),
                        "exchange": q.get("exchange", ""),
                        "type": q.get("quoteType", "EQUITY"),
                        "currency": q.get("currency", "USD"),
                        "sector": "",
                        "industry": q.get("industry", ""),
                    })
        except Exception:
            # Fallback: validate directly via yfinance Ticker
            for suffix in ["", ".NS", ".BO"]:
                candidate = query_str.upper() + suffix
                try:
                    ticker = yf.Ticker(candidate)
                    info = ticker.info
                    if info.get("longName") or info.get("shortName"):
                        results.append({
                            "symbol": candidate,
                            "name": info.get("longName", info.get("shortName", "")),
                            "exchange": info.get("exchange", ""),
                            "type": info.get("quoteType", "EQUITY"),
                            "currency": info.get("currency", "USD"),
                            "sector": info.get("sector", ""),
                            "industry": info.get("industry", ""),
                        })
                        break
                except Exception:
                    continue

        return {"results": results, "query": query, "count": len(results)}
    except Exception as e:
        return {"error": str(e), "query": query, "results": []}


def _get_analyst_estimates(symbol: str) -> dict:
    """Fetch analyst consensus estimates, price targets, EPS trend, and recommendations.

    Returns a dict with these keys (each is a nested dict or None if unavailable):
      recommendations_summary  - buy/hold/sell counts from covering analysts
      upgrades_downgrades      - most recent 20 rating changes (firm, grade, action)
      analyst_price_targets    - price target distribution (current, low, high, mean, median)
      earnings_estimate        - forward EPS consensus by period (0q, +1q, 0y, +1y)
      revenue_estimate         - forward revenue consensus by period
      earnings_history         - historical EPS actual vs estimate (beat/miss data)
      eps_trend                - how EPS estimates have been revised over time
      eps_revisions            - number of upward/downward revisions per period
      growth_estimates         - analyst long-term growth forecasts
    """
    try:
        t = yf.Ticker(symbol)

        def _fetch(getter_name: str, prop_name: str, limit: int | None = None) -> Any:
            """Try getter method then property; convert DataFrame to JSON-serializable dict."""
            for name in (getter_name, prop_name):
                try:
                    attr = getattr(t, name)
                    val = attr() if callable(attr) else attr
                    if val is None:
                        continue
                    if isinstance(val, pd.DataFrame):
                        if val.empty:
                            continue
                        val = val.copy()
                        if limit:
                            val = val.head(limit)
                        # Stringify index to handle DatetimeIndex / period strings
                        val.index = val.index.astype(str)
                        return val.to_dict()
                    return val  # dict, scalar, etc.
                except Exception:
                    continue
            return None

        return {
            "symbol": symbol,
            "recommendations_summary": _fetch("get_recommendations_summary", "recommendations_summary"),
            "upgrades_downgrades": _fetch("get_upgrades_downgrades", "upgrades_downgrades", limit=20),
            "analyst_price_targets": _fetch("get_analyst_price_targets", "analyst_price_targets"),
            "earnings_estimate": _fetch("get_earnings_estimate", "earnings_estimate"),
            "revenue_estimate": _fetch("get_revenue_estimate", "revenue_estimate"),
            "earnings_history": _fetch("get_earnings_history", "earnings_history"),
            "eps_trend": _fetch("get_eps_trend", "eps_trend"),
            "eps_revisions": _fetch("get_eps_revisions", "eps_revisions"),
            "growth_estimates": _fetch("get_growth_estimates", "growth_estimates"),
            "timestamp": int(datetime.now().timestamp()),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _resolve_symbol(symbol: str) -> dict:
    """Resolve a stock symbol, auto-appending .NS or .BO exchange suffix if needed."""
    symbol = symbol.strip().upper()
    if not symbol:
        return {"resolved_symbol": symbol, "original_symbol": symbol, "exchange": "", "found": False}

    # Already has a suffix, verify as-is
    if "." in symbol or symbol.startswith("^") or "-" in symbol or "=" in symbol:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            exchange = ticker.info.get("exchange", "") if hasattr(ticker, "info") else ""
            if not hist.empty:
                return {"resolved_symbol": symbol, "original_symbol": symbol, "exchange": exchange, "found": True}
        except Exception:
            pass
        return {"resolved_symbol": symbol, "original_symbol": symbol, "exchange": "", "found": False}

    # Try bare symbol first, then .NS and .BO suffixes
    for suffix in ["", ".NS", ".BO"]:
        candidate = symbol + suffix
        try:
            ticker = yf.Ticker(candidate)
            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty and len(hist) >= 1:
                exchange = ""
                try:
                    exchange = ticker.info.get("exchange", "")
                except Exception:
                    pass
                return {
                    "resolved_symbol": candidate,
                    "original_symbol": symbol,
                    "exchange": exchange,
                    "found": True,
                }
        except Exception:
            continue

    return {"resolved_symbol": symbol, "original_symbol": symbol, "exchange": "", "found": False}


# ---------------------------------------------------------------------------
# Async Tool wrapper
# ---------------------------------------------------------------------------

class YFinanceTool(Tool):
    """YFinance data tool - provides stock quotes, historical data, and fundamentals.

    All synchronous yfinance calls are offloaded via asyncio.to_thread for non-blocking execution.
    """

    name = "yfinance"
    description = (
        "Fetch stock market data via Yahoo Finance. Supports real-time quotes, "
        "historical OHLCV data, company information, financial statements, "
        "financial ratios, symbol search, and symbol resolution."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "quote",
                    "batch_quotes",
                    "historical",
                    "historical_price",
                    "info",
                    "financials",
                    "company_profile",
                    "financial_ratios",
                    "multiple_profiles",
                    "multiple_ratios",
                    "search",
                    "resolve_symbol",
                    "analyst_estimates",
                ],
                "description": (
                    "Action to perform. "
                    "quote/batch_quotes: real-time price; "
                    "historical/historical_price: OHLCV history; "
                    "info: full company metrics; "
                    "financials: income/balance/cashflow statements; "
                    "company_profile/financial_ratios: peer comparison data; "
                    "multiple_profiles/multiple_ratios: batch peer data; "
                    "search: symbol lookup by keyword; "
                    "resolve_symbol: auto-resolve exchange suffix; "
                    "analyst_estimates: consensus EPS/revenue estimates, beat/miss history, "
                    "price targets, EPS revisions, and buy/hold/sell recommendations."
                ),
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Stock ticker (e.g. 'AAPL', 'RELIANCE.NS'). "
                    "Required for: quote, historical, historical_price, info, "
                    "financials, company_profile, financial_ratios, resolve_symbol, "
                    "analyst_estimates."
                ),
            },
            "symbols": {
                "type": "string",
                "description": (
                    "Comma-separated tickers (e.g. 'AAPL,MSFT,GOOG'). "
                    "Required for: batch_quotes, multiple_profiles, multiple_ratios."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search keyword. Required for: search.",
            },
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD. Required for: historical.",
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD. Required for: historical.",
            },
            "target_date": {
                "type": "string",
                "description": "Target date YYYY-MM-DD. Required for: historical_price.",
            },
            "interval": {
                "type": "string",
                "enum": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                "description": "OHLCV interval for historical. Default: 1d.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for search. Default: 20.",
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        symbol = kwargs.get("symbol") or kwargs.get("symbols") or kwargs.get("query", "")
        logger.info(f"yfinance:{command}  target={symbol!r}")

        if command == "quote":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for quote"})
            result = await asyncio.to_thread(_get_quote, symbol)

        elif command == "batch_quotes":
            symbols_str = kwargs.get("symbols", "")
            if not symbols_str:
                return json.dumps({"error": "symbols is required for batch_quotes"})
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            result = await asyncio.to_thread(_get_batch_quotes, symbols)

        elif command == "historical":
            symbol = kwargs.get("symbol", "")
            start_date = kwargs.get("start_date", "")
            end_date = kwargs.get("end_date", "")
            if not (symbol and start_date and end_date):
                return json.dumps({"error": "symbol, start_date, end_date are required for historical"})
            interval = kwargs.get("interval", "1d")
            result = await asyncio.to_thread(_get_historical, symbol, start_date, end_date, interval)

        elif command == "historical_price":
            symbol = kwargs.get("symbol", "")
            target_date = kwargs.get("target_date", "")
            if not (symbol and target_date):
                return json.dumps({"error": "symbol and target_date are required for historical_price"})
            result = await asyncio.to_thread(_get_historical_price, symbol, target_date)

        elif command == "info":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for info"})
            result = await asyncio.to_thread(_get_info, symbol)

        elif command == "financials":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for financials"})
            result = await asyncio.to_thread(_get_financials, symbol)

        elif command == "company_profile":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for company_profile"})
            result = await asyncio.to_thread(_get_company_profile, symbol)

        elif command == "financial_ratios":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for financial_ratios"})
            result = await asyncio.to_thread(_get_financial_ratios, symbol)

        elif command == "multiple_profiles":
            symbols_str = kwargs.get("symbols", "")
            if not symbols_str:
                return json.dumps({"error": "symbols is required for multiple_profiles"})
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            result = await asyncio.to_thread(
                lambda: [p for s in symbols if "error" not in (p := _get_company_profile(s))]
            )

        elif command == "multiple_ratios":
            symbols_str = kwargs.get("symbols", "")
            if not symbols_str:
                return json.dumps({"error": "symbols is required for multiple_ratios"})
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            result = await asyncio.to_thread(
                lambda: [r for s in symbols if "error" not in (r := _get_financial_ratios(s))]
            )

        elif command == "search":
            query = kwargs.get("query", "")
            if not query:
                return json.dumps({"error": "query is required for search"})
            limit = kwargs.get("limit", 20)
            result = await asyncio.to_thread(_search_symbols, query, limit)

        elif command == "resolve_symbol":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for resolve_symbol"})
            result = await asyncio.to_thread(_resolve_symbol, symbol)

        elif command == "analyst_estimates":
            symbol = kwargs.get("symbol", "")
            if not symbol:
                return json.dumps({"error": "symbol is required for analyst_estimates"})
            result = await asyncio.to_thread(_get_analyst_estimates, symbol)

        else:
            result = {"error": f"Unknown command: {command!r}"}

        return json.dumps(result, ensure_ascii=False)
