"""EarningsCalendarTool — earnings dates, surprises, and analyst estimates.

Commands:
  calendar   - next earnings date + EPS/revenue estimate ranges for one ticker
  upcoming   - earnings calendar across multiple tickers for a date window
  surprise   - historical EPS actual vs estimate for last N quarters (beat/miss)
  consensus  - forward analyst EPS/revenue estimates by quarter and year
  revisions  - EPS estimate revision counts and trend over 7/30/60/90 days
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from finclaw.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_str(val: Any) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return None


def _df_to_nested(df: pd.DataFrame | None) -> dict:
    """Convert a DataFrame to {col: {index: value}} with safe float values."""
    if df is None or df.empty:
        return {}
    result: dict = {}
    for col in df.columns:
        result[str(col)] = {
            str(idx): _safe_float(df.loc[idx, col])
            for idx in df.index
            if pd.notna(df.loc[idx, col])
        }
    return result


def _pick(nested: dict, period: str, field: str) -> float | None:
    return nested.get(period, {}).get(field)


# ---------------------------------------------------------------------------
# Pure function layer (sync) — offloaded via asyncio.to_thread
# ---------------------------------------------------------------------------

def _get_calendar(symbol: str) -> dict:
    """Next earnings date and analyst estimate ranges for one ticker."""
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        info = ticker.info

        if not cal:
            return {"error": "No earnings calendar available", "symbol": symbol}

        # Earnings Date may be a list or scalar depending on yfinance version
        dates_raw = cal.get("Earnings Date", [])
        if not isinstance(dates_raw, list):
            dates_raw = [dates_raw]
        date_strs = [_ts_to_str(d) for d in dates_raw if d is not None]

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", info.get("shortName", "")),
            "next_earnings_date": date_strs[0] if date_strs else None,
            "all_earnings_dates": date_strs,
            "eps_estimate_avg": _safe_float(cal.get("Earnings Average")),
            "eps_estimate_low": _safe_float(cal.get("Earnings Low")),
            "eps_estimate_high": _safe_float(cal.get("Earnings High")),
            "revenue_estimate_avg": _safe_int(cal.get("Revenue Average")),
            "revenue_estimate_low": _safe_int(cal.get("Revenue Low")),
            "revenue_estimate_high": _safe_int(cal.get("Revenue High")),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_upcoming(symbols: list[str], days_ahead: int = 7) -> dict:
    """Filter a list of symbols to those reporting within days_ahead."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    results = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if not cal:
                continue

            dates_raw = cal.get("Earnings Date", [])
            if not isinstance(dates_raw, list):
                dates_raw = [dates_raw]

            for d in dates_raw:
                if d is None:
                    continue
                if hasattr(d, "tzinfo") and d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                elif not hasattr(d, "tzinfo"):
                    continue

                if now <= d <= cutoff:
                    info = ticker.info
                    results.append({
                        "symbol": symbol.upper(),
                        "company_name": info.get("longName", info.get("shortName", "")),
                        "earnings_date": _ts_to_str(d),
                        "eps_estimate_avg": _safe_float(cal.get("Earnings Average")),
                        "revenue_estimate_avg": _safe_int(cal.get("Revenue Average")),
                        "sector": info.get("sector", ""),
                        "market_cap": info.get("marketCap"),
                    })
                    break  # one entry per symbol
        except Exception:
            continue

    results.sort(key=lambda x: x.get("earnings_date") or "")
    return {
        "window_days": days_ahead,
        "from": now.date().isoformat(),
        "to": cutoff.date().isoformat(),
        "count": len(results),
        "earnings": results,
    }


def _get_surprise(symbol: str, limit: int = 8) -> dict:
    """Historical EPS actual vs estimate for last N reported quarters."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        df = ticker.earnings_dates

        if df is None or df.empty:
            return {"error": "No earnings history available", "symbol": symbol}

        reported = df[df["Reported EPS"].notna()].head(limit)
        records = []

        for date, row in reported.iterrows():
            eps_est = _safe_float(row.get("EPS Estimate"))
            eps_act = _safe_float(row.get("Reported EPS"))
            surprise_pct = _safe_float(row.get("Surprise(%)"))

            if eps_act is not None and eps_est is not None:
                beat = eps_act > eps_est
            else:
                beat = None

            records.append({
                "date": _ts_to_str(date),
                "eps_estimate": eps_est,
                "eps_actual": eps_act,
                "eps_surprise_pct": surprise_pct,
                "beat": beat,
            })

        beat_count = sum(1 for r in records if r.get("beat") is True)
        miss_count = sum(1 for r in records if r.get("beat") is False)

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", info.get("shortName", "")),
            "quarters_shown": len(records),
            "beat_count": beat_count,
            "miss_count": miss_count,
            "beat_rate": round(beat_count / len(records), 3) if records else None,
            "history": records,
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_consensus(symbol: str) -> dict:
    """Forward analyst EPS and revenue consensus estimates."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        eps_est = _df_to_nested(ticker.earnings_estimate)
        rev_est = _df_to_nested(ticker.revenue_estimate)

        def _eps_period(period: str) -> dict:
            return {
                "period": period,
                "avg": _pick(eps_est, period, "avg"),
                "low": _pick(eps_est, period, "low"),
                "high": _pick(eps_est, period, "high"),
                "year_ago_eps": _pick(eps_est, period, "yearAgoEps"),
                "growth": _pick(eps_est, period, "growth"),
                "num_analysts": _pick(eps_est, period, "numberOfAnalysts"),
            }

        def _rev_period(period: str) -> dict:
            return {
                "period": period,
                "avg": _pick(rev_est, period, "avg"),
                "low": _pick(rev_est, period, "low"),
                "high": _pick(rev_est, period, "high"),
                "growth": _pick(rev_est, period, "growth"),
                "num_analysts": _pick(rev_est, period, "numberOfAnalysts"),
            }

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", info.get("shortName", "")),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "target_price_mean": _safe_float(info.get("targetMeanPrice")),
            "target_price_low": _safe_float(info.get("targetLowPrice")),
            "target_price_high": _safe_float(info.get("targetHighPrice")),
            "eps_estimates": {
                "current_quarter": _eps_period("0q"),
                "next_quarter": _eps_period("+1q"),
                "current_year": _eps_period("0y"),
                "next_year": _eps_period("+1y"),
            },
            "revenue_estimates": {
                "current_quarter": _rev_period("0q"),
                "next_quarter": _rev_period("+1q"),
                "current_year": _rev_period("0y"),
                "next_year": _rev_period("+1y"),
            },
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _get_revisions(symbol: str) -> dict:
    """EPS estimate revision counts and trend over 7/30/60/90 days."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        revisions = _df_to_nested(ticker.eps_revisions)
        trend = _df_to_nested(ticker.eps_trend)

        def _rev_period(period: str) -> dict:
            return {
                "period": period,
                "up_last_7_days": _pick(revisions, period, "upLast7days"),
                "up_last_30_days": _pick(revisions, period, "upLast30days"),
                "down_last_7_days": _pick(revisions, period, "downLast7days"),
                "down_last_30_days": _pick(revisions, period, "downLast30days"),
            }

        def _trend_period(period: str) -> dict:
            return {
                "period": period,
                "current": _pick(trend, period, "current"),
                "7_days_ago": _pick(trend, period, "7daysAgo"),
                "30_days_ago": _pick(trend, period, "30daysAgo"),
                "60_days_ago": _pick(trend, period, "60daysAgo"),
                "90_days_ago": _pick(trend, period, "90daysAgo"),
            }

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", info.get("shortName", "")),
            "eps_revisions": {
                "current_quarter": _rev_period("0q"),
                "next_quarter": _rev_period("+1q"),
                "current_year": _rev_period("0y"),
                "next_year": _rev_period("+1y"),
            },
            "eps_trend": {
                "current_quarter": _trend_period("0q"),
                "next_quarter": _trend_period("+1q"),
                "current_year": _trend_period("0y"),
                "next_year": _trend_period("+1y"),
            },
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ---------------------------------------------------------------------------
# Async Tool wrapper
# ---------------------------------------------------------------------------

class EarningsCalendarTool(Tool):
    """Earnings calendar, EPS surprise history, and analyst estimate tracking.

    All synchronous yfinance calls are offloaded via asyncio.to_thread.
    """

    name = "earnings_calendar"
    description = (
        "Track earnings dates, EPS/revenue surprises, and analyst estimates for US/global stocks. "
        "Use for: next earnings date, historical beat/miss record, forward EPS consensus by "
        "quarter and year, and analyst estimate revision trends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["calendar", "upcoming", "surprise", "consensus", "revisions"],
                "description": (
                    "calendar: next earnings date + estimate ranges for one ticker; "
                    "upcoming: earnings across multiple tickers in a date window; "
                    "surprise: historical EPS actual vs estimate (beat/miss history); "
                    "consensus: forward analyst EPS/revenue estimates by quarter and year; "
                    "revisions: EPS estimate revision counts and trend over 7/30/60/90 days."
                ),
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Single ticker symbol (e.g. 'AAPL', 'NVDA'). "
                    "Required for: calendar, surprise, consensus, revisions."
                ),
            },
            "symbols": {
                "type": "string",
                "description": (
                    "Comma-separated tickers (e.g. 'AAPL,MSFT,NVDA,TSLA'). "
                    "Required for: upcoming. Max 30 symbols recommended."
                ),
            },
            "days_ahead": {
                "type": "integer",
                "description": "Days ahead to scan for upcoming earnings. Default: 7.",
                "minimum": 1,
                "maximum": 90,
            },
            "limit": {
                "type": "integer",
                "description": "Past quarters to return for surprise command. Default: 8.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        symbol = kwargs.get("symbol", "")
        logger.info(f"earnings_calendar:{command}  target={symbol or kwargs.get('symbols', '')!r}")

        if command == "calendar":
            if not symbol:
                return json.dumps({"error": "symbol is required for calendar"})
            result = await asyncio.to_thread(_get_calendar, symbol.upper())

        elif command == "upcoming":
            symbols_str = kwargs.get("symbols", "")
            if not symbols_str:
                return json.dumps({"error": "symbols is required for upcoming"})
            symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
            days_ahead = int(kwargs.get("days_ahead", 7))
            result = await asyncio.to_thread(_get_upcoming, symbols, days_ahead)

        elif command == "surprise":
            if not symbol:
                return json.dumps({"error": "symbol is required for surprise"})
            limit = int(kwargs.get("limit", 8))
            result = await asyncio.to_thread(_get_surprise, symbol.upper(), limit)

        elif command == "consensus":
            if not symbol:
                return json.dumps({"error": "symbol is required for consensus"})
            result = await asyncio.to_thread(_get_consensus, symbol.upper())

        elif command == "revisions":
            if not symbol:
                return json.dumps({"error": "symbol is required for revisions"})
            result = await asyncio.to_thread(_get_revisions, symbol.upper())

        else:
            result = {"error": f"Unknown command: {command!r}"}

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return json.dumps(result, ensure_ascii=False, default=str)
