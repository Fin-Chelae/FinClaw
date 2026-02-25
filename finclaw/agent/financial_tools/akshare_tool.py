"""AKShare Tool - Chinese A-share stock market data fetcher.

Provides A-share market data via the akshare library, wrapped as a finclaw Tool.
Synchronous akshare calls are offloaded via asyncio.to_thread for non-blocking execution.

Supported commands (command parameter):
  quote              - real-time quote for a single A-share stock
  historical         - historical K-line (OHLCV) data
  info               - individual stock basic information
  financials         - financial report abstract
  search             - search stocks by keyword (name/code)
  sector_performance - industry sector performance ranking
  index_quotes       - major index quotes (SSE, SZSE, etc.)
  news               - recent stock news from East Money
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from finclaw.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Rate limiting & cache
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0
_RATE_LIMIT_SECONDS = 1.0

# In-memory cache for full-market spot data (expensive to fetch)
_spot_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_SPOT_CACHE_TTL = 600  # 10 minutes


def _rate_limit() -> None:
    """Ensure at least _RATE_LIMIT_SECONDS between akshare API calls."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


# ---------------------------------------------------------------------------
# Column name mapping: Chinese → English
# ---------------------------------------------------------------------------

_SPOT_COLUMN_MAP = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "change_percent",
    "涨跌额": "change",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "最高": "high",
    "最低": "low",
    "今开": "open",
    "昨收": "previous_close",
    "量比": "volume_ratio",
    "换手率": "turnover_rate",
    "市盈率-动态": "pe_ratio",
    "市净率": "pb_ratio",
    "总市值": "market_cap",
    "流通市值": "float_market_cap",
    "60日涨跌幅": "change_60d",
    "年初至今涨跌幅": "change_ytd",
}

_HIST_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "change_percent",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

_INFO_COLUMN_MAP = {
    "item": "item",
    "value": "value",
}

_SECTOR_COLUMN_MAP = {
    "板块名称": "sector_name",
    "板块代码": "sector_code",
    "最新价": "price",
    "涨跌幅": "change_percent",
    "涨跌额": "change",
    "总市值": "market_cap",
    "换手率": "turnover_rate",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股票": "leading_stock",
    "领涨涨跌幅": "leading_change_percent",
}

_INDEX_COLUMN_MAP = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌额": "change",
    "涨跌幅": "change_percent",
    "昨收": "previous_close",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
}


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns using a mapping, keeping unmapped columns as-is."""
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


# ---------------------------------------------------------------------------
# Code identification helpers
# ---------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    """Normalize A-share stock code to 6-digit format.

    Strips common prefixes like 'sh', 'sz', 'SH', 'SZ' and dots.
    """
    code = code.strip().upper()
    for prefix in ("SH", "SZ", "SH.", "SZ.", "SS."):
        if code.startswith(prefix):
            code = code[len(prefix):]
    # Remove .SH / .SZ / .SS suffixes
    for suffix in (".SH", ".SZ", ".SS"):
        if code.endswith(suffix):
            code = code[: -len(suffix)]
    return code.strip()


def _is_etf(code: str) -> bool:
    """Check if a code is likely an ETF (51xxxx, 15xxxx, 159xxx)."""
    return code.startswith(("51", "15", "159"))


def _is_hk_connect(code: str) -> bool:
    """Check if a code might be a Hong Kong stock (5-digit code)."""
    return len(code) == 5 and code.isdigit()


# ---------------------------------------------------------------------------
# Full-market spot data (cached)
# ---------------------------------------------------------------------------

def _get_spot_data() -> pd.DataFrame:
    """Fetch or return cached full A-share market spot data."""
    import akshare as ak

    now = time.time()
    if _spot_cache["data"] is not None and (now - _spot_cache["timestamp"]) < _SPOT_CACHE_TTL:
        return _spot_cache["data"]

    _rate_limit()
    df = ak.stock_zh_a_spot_em()
    df = _rename_columns(df, _SPOT_COLUMN_MAP)
    _spot_cache["data"] = df
    _spot_cache["timestamp"] = now
    return df


# ---------------------------------------------------------------------------
# Pure function layer (sync)
# ---------------------------------------------------------------------------

def _get_quote(code: str) -> dict:
    """Fetch real-time quote for a single A-share stock from cached spot data."""
    try:
        code = _normalize_code(code)
        df = _get_spot_data()
        row = df[df["code"] == code]
        if row.empty:
            return {"error": f"Stock {code} not found in A-share market", "code": code}
        row = row.iloc[0]
        result = {}
        for col in row.index:
            val = row[col]
            if pd.isna(val):
                result[col] = None
            elif isinstance(val, (int, float)):
                result[col] = round(float(val), 4)
            else:
                result[col] = str(val)
        result["timestamp"] = int(datetime.now().timestamp())
        return result
    except Exception as e:
        return {"error": str(e), "code": code}


def _get_historical(code: str, start_date: str, end_date: str) -> list | dict:
    """Fetch historical K-line data for an A-share stock."""
    import akshare as ak

    try:
        code = _normalize_code(code)
        _rate_limit()
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return {"error": f"No historical data for {code}", "code": code}

        df = _rename_columns(df, _HIST_COLUMN_MAP)
        # Convert date column to string if present
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)

        records = df.to_dict(orient="records")
        # Round numeric values
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float):
                    rec[k] = round(v, 4)
        return records
    except Exception as e:
        return {"error": str(e), "code": code}


def _get_info(code: str) -> dict:
    """Fetch individual stock basic information."""
    import akshare as ak

    try:
        code = _normalize_code(code)
        _rate_limit()
        df = ak.stock_individual_info_em(symbol=code)
        if df is None or df.empty:
            return {"error": f"No info for {code}", "code": code}

        df = _rename_columns(df, _INFO_COLUMN_MAP)
        # Convert to key-value dict
        info = {"code": code}
        for _, row in df.iterrows():
            key = str(row.get("item", "")).strip()
            val = row.get("value", "")
            if key:
                info[key] = str(val) if not isinstance(val, (int, float)) else val
        info["timestamp"] = int(datetime.now().timestamp())
        return info
    except Exception as e:
        return {"error": str(e), "code": code}


def _get_financials(code: str) -> dict:
    """Fetch financial report abstract from Tonghuashun."""
    import akshare as ak

    try:
        code = _normalize_code(code)
        _rate_limit()
        df = ak.stock_financial_abstract_ths(symbol=code)
        if df is None or df.empty:
            return {"error": f"No financial data for {code}", "code": code}

        # Rename Chinese columns to English where possible
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and pd.notna(v):
                    rec[k] = round(v, 4)
                elif pd.isna(v):
                    rec[k] = None
                else:
                    rec[k] = str(v)
        return {"code": code, "financials": records, "timestamp": int(datetime.now().timestamp())}
    except Exception as e:
        return {"error": str(e), "code": code}


def _search_stocks(query: str) -> dict:
    """Search A-share stocks by keyword (name or code)."""
    try:
        df = _get_spot_data()
        query_lower = query.strip().lower()

        # Match against code or name
        mask = df["code"].str.contains(query_lower, case=False, na=False) | df[
            "name"
        ].str.contains(query, case=False, na=False)
        results = df[mask].head(20)

        if results.empty:
            return {"results": [], "query": query, "count": 0}

        records = []
        for _, row in results.iterrows():
            records.append(
                {
                    "code": str(row.get("code", "")),
                    "name": str(row.get("name", "")),
                    "price": round(float(row["price"]), 2) if pd.notna(row.get("price")) else None,
                    "change_percent": (
                        round(float(row["change_percent"]), 2)
                        if pd.notna(row.get("change_percent"))
                        else None
                    ),
                    "market_cap": (
                        round(float(row["market_cap"]), 0)
                        if pd.notna(row.get("market_cap"))
                        else None
                    ),
                }
            )
        return {"results": records, "query": query, "count": len(records)}
    except Exception as e:
        return {"error": str(e), "query": query, "results": []}


def _get_sector_performance() -> dict:
    """Fetch industry sector performance ranking."""
    import akshare as ak

    try:
        _rate_limit()
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return {"error": "No sector data available"}

        df = _rename_columns(df, _SECTOR_COLUMN_MAP)
        records = df.head(30).to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and pd.notna(v):
                    rec[k] = round(v, 4)
                elif pd.isna(v):
                    rec[k] = None
                else:
                    rec[k] = str(v)
        return {"sectors": records, "count": len(records), "timestamp": int(datetime.now().timestamp())}
    except Exception as e:
        return {"error": str(e)}


_NEWS_COLUMN_MAP = {
    "关键词": "keyword",
    "新闻标题": "title",
    "新闻内容": "content",
    "发布时间": "publish_time",
    "文章来源": "source",
    "新闻链接": "url",
}


def _get_news(code: str) -> dict:
    """Fetch recent news for an A-share stock via East Money (stock_news_em)."""
    import akshare as ak

    try:
        code = _normalize_code(code)
        _rate_limit()
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return {"error": f"No news for {code}", "code": code, "news": []}

        df = _rename_columns(df, _NEWS_COLUMN_MAP)
        records = df.head(20).to_dict(orient="records")
        for rec in records:
            # Truncate content to 200 chars to control token usage
            if "content" in rec and isinstance(rec["content"], str) and len(rec["content"]) > 200:
                rec["content"] = rec["content"][:200] + "..."
            for k, v in rec.items():
                if pd.isna(v):
                    rec[k] = None
                elif not isinstance(v, (int, float)):
                    rec[k] = str(v)
        return {"code": code, "news": records, "count": len(records), "timestamp": int(datetime.now().timestamp())}
    except Exception as e:
        return {"error": str(e), "code": code, "news": []}


def _get_index_quotes() -> dict:
    """Fetch major index quotes (SSE, SZSE, etc.)."""
    import akshare as ak

    try:
        _rate_limit()
        df = ak.stock_zh_index_spot_sina()
        if df is None or df.empty:
            return {"error": "No index data available"}

        df = _rename_columns(df, _INDEX_COLUMN_MAP)

        # Filter to well-known indices
        known_indices = {
            "sh000001",  # SSE Composite
            "sh000300",  # CSI 300
            "sh000016",  # SSE 50
            "sh000905",  # CSI 500
            "sz399001",  # SZSE Component
            "sz399006",  # ChiNext
            "sz399005",  # SME
            "sh000688",  # STAR 50
        }
        if "code" in df.columns:
            filtered = df[df["code"].isin(known_indices)]
            if not filtered.empty:
                df = filtered

        records = df.head(20).to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and pd.notna(v):
                    rec[k] = round(v, 4)
                elif pd.isna(v):
                    rec[k] = None
                else:
                    rec[k] = str(v)
        return {"indices": records, "count": len(records), "timestamp": int(datetime.now().timestamp())}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Async Tool wrapper
# ---------------------------------------------------------------------------

class AKShareTool(Tool):
    """AKShare data tool - provides Chinese A-share stock market data.

    All synchronous akshare calls are offloaded via asyncio.to_thread for non-blocking execution.
    Real-time spot data is cached for 10 minutes to avoid redundant full-market fetches.
    """

    name = "akshare"
    description = (
        "Fetch Chinese A-share stock market data via AKShare. Supports real-time quotes, "
        "historical K-line data, company information, financial reports, stock search, "
        "industry sector performance, and major index quotes. Use 6-digit stock codes "
        "(e.g. 600519 for Kweichow Moutai, 000651 for Gree Electric)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "quote",
                    "historical",
                    "info",
                    "financials",
                    "search",
                    "sector_performance",
                    "index_quotes",
                    "news",
                ],
                "description": (
                    "Action to perform. "
                    "quote: real-time A-share quote; "
                    "historical: daily K-line OHLCV history; "
                    "info: individual stock basic information; "
                    "financials: financial report abstract; "
                    "search: search stocks by name/code keyword; "
                    "sector_performance: industry sector ranking; "
                    "index_quotes: major index (SSE, SZSE) quotes; "
                    "news: recent stock news from East Money."
                ),
            },
            "code": {
                "type": "string",
                "description": (
                    "A-share stock code (e.g. '600519', '000651'). "
                    "Required for: quote, historical, info, financials."
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
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        target = kwargs.get("code") or kwargs.get("query", "")
        logger.info(f"akshare:{command}  target={target!r}")

        if command == "quote":
            code = kwargs.get("code", "")
            if not code:
                return json.dumps({"error": "code is required for quote"})
            result = await asyncio.to_thread(_get_quote, code)

        elif command == "historical":
            code = kwargs.get("code", "")
            start_date = kwargs.get("start_date", "")
            end_date = kwargs.get("end_date", "")
            if not code:
                return json.dumps({"error": "code is required for historical"})
            if not start_date or not end_date:
                # Default to last 30 days
                end_date = end_date or datetime.now().strftime("%Y-%m-%d")
                start_date = start_date or (datetime.now() - timedelta(days=30)).strftime(
                    "%Y-%m-%d"
                )
            result = await asyncio.to_thread(_get_historical, code, start_date, end_date)

        elif command == "info":
            code = kwargs.get("code", "")
            if not code:
                return json.dumps({"error": "code is required for info"})
            result = await asyncio.to_thread(_get_info, code)

        elif command == "financials":
            code = kwargs.get("code", "")
            if not code:
                return json.dumps({"error": "code is required for financials"})
            result = await asyncio.to_thread(_get_financials, code)

        elif command == "search":
            query = kwargs.get("query", "")
            if not query:
                return json.dumps({"error": "query is required for search"})
            result = await asyncio.to_thread(_search_stocks, query)

        elif command == "sector_performance":
            result = await asyncio.to_thread(_get_sector_performance)

        elif command == "index_quotes":
            result = await asyncio.to_thread(_get_index_quotes)

        elif command == "news":
            code = kwargs.get("code", "")
            if not code:
                return json.dumps({"error": "code is required for news"})
            result = await asyncio.to_thread(_get_news, code)

        else:
            result = {"error": f"Unknown command: {command!r}"}

        return json.dumps(result, ensure_ascii=False)
