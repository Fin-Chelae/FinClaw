"""Economics Data Tool — macro data fetcher via FRED API and yfinance.

Dual data source strategy:
  - FRED (Federal Reserve Economic Data): official US macro series
    (GDP, CPI, UNRATE, FEDFUNDS, M2, treasury yields, etc.)
  - yfinance: FX rates, market-based yields, commodities

Supported commands (command parameter):
  fred_series    - fetch one or more FRED time series
  fx_rates       - FX spot rates via yfinance (e.g. EURUSD=X)
  yields         - treasury yields via yfinance (^TNX, ^IRX, ^TYX)
  indicators     - composite of multiple FRED series in one call
  commodity      - commodity prices via yfinance (CL=F, GC=F, NG=F, etc.)
  calendar       - economic release calendar (what macro data releases today/this week)

FRED API key is optional.  Configure via tools.financial.fredApiKey in
~/.finclaw/config.json, or set FRED_API_KEY env var as fallback.
If neither is set, FRED-based commands return an appropriate error;
yfinance commands still work.
"""

import asyncio
import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import yfinance as yf
from loguru import logger

from finclaw.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# FRED client (sync) — thin wrapper around the public REST API
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred"
_FRED_NO_KEY_ERROR = (
    "FRED API key not configured. Set tools.financial.fredApiKey in "
    "~/.finclaw/config.json, or export FRED_API_KEY env var. "
    "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
)


def _get_fred_api_key() -> str:
    """Resolve FRED API key: config.json first, then FRED_API_KEY env var."""
    try:
        from finclaw.config.loader import load_config
        key = load_config().tools.financial.fred_api_key
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("FRED_API_KEY", "")

# Well-known FRED series for composite indicator requests
_INDICATOR_SERIES: dict[str, str] = {
    "gdp": "GDP",
    "gdp_growth": "A191RL1Q225SBEA",
    "cpi": "CPIAUCSL",              # index level — use cpi_yoy for inflation rate
    "cpi_yoy": "CPIAUCSL",          # auto-computes YoY % change
    "core_cpi": "CPILFESL",         # index level — use core_cpi_yoy for rate
    "core_cpi_yoy": "CPILFESL",     # auto-computes YoY % change
    "inflation": "T10YIE",           # 10-year breakeven inflation rate
    "unemployment": "UNRATE",
    "fed_funds": "FEDFUNDS",
    "m2": "M2SL",
    "industrial_production": "INDPRO",
    "retail_sales": "RSXFS",
    "housing_starts": "HOUST",
    "consumer_sentiment": "UMCSENT",
    "leading_index": "USSLIND",      # Conference Board Leading Index
    "yield_2y": "DGS2",
    "yield_10y": "DGS10",
    "yield_30y": "DGS30",
    "yield_spread": "T10Y2Y",        # 10y-2y term spread
    "credit_spread": "BAMLH0A0HYM2", # HY OAS spread
    "vix": "VIXCLS",
}

# Indicator keys that return index levels; _fetch_fred_indicators will
# auto-compute YoY % change when these keys are requested with the _yoy suffix.
_YOY_KEYS: set[str] = {"cpi_yoy", "core_cpi_yoy"}


def _csv_list(s: str) -> list[str]:
    """Split a comma-separated string into a stripped, non-empty list."""
    return [item.strip() for item in s.split(",") if item.strip()]


def _fetch_fred_series(
    series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
    limit: int = 100,
) -> dict:
    """Fetch observations and metadata for a single FRED series.

    Uses one HTTP connection for both API calls (observations + metadata).
    Metadata failure is non-fatal; observations are returned regardless.
    """
    api_key = _get_fred_api_key()
    if not api_key:
        return {"error": _FRED_NO_KEY_ERROR}

    common = {"api_key": api_key, "file_type": "json"}
    obs_params = {"series_id": series_id, "sort_order": "desc", "limit": limit, **common}
    if observation_start:
        obs_params["observation_start"] = observation_start
    if observation_end:
        obs_params["observation_end"] = observation_end

    try:
        with httpx.Client(timeout=15) as client:
            obs_resp = client.get(f"{_FRED_BASE}/series/observations", params=obs_params)
            if not obs_resp.is_success:
                return {"error": f"FRED API error {obs_resp.status_code}: {obs_resp.text[:200]}"}
            raw = obs_resp.json()

            # Metadata is best-effort; its failure should not suppress observations.
            try:
                meta_resp = client.get(
                    f"{_FRED_BASE}/series",
                    params={"series_id": series_id, **common},
                )
                meta_raw = meta_resp.json() if meta_resp.is_success else {}
            except Exception:
                meta_raw = {}

    except httpx.RequestError as exc:
        return {"error": f"Network error fetching FRED data: {exc}"}
    except Exception as exc:
        return {"error": f"FRED request failed: {exc}"}

    meta: dict[str, Any] = {}
    if "seriess" in meta_raw and meta_raw["seriess"]:
        s = meta_raw["seriess"][0]
        meta = {
            "id": s.get("id"),
            "title": s.get("title"),
            "units": s.get("units_short", s.get("units")),
            "frequency": s.get("frequency_short", s.get("frequency")),
            "seasonal_adjustment": s.get("seasonal_adjustment_short"),
            "last_updated": s.get("last_updated"),
        }

    observations = [
        {"date": obs["date"], "value": obs["value"]}
        for obs in raw.get("observations", [])
        if obs.get("value") not in (".", None, "")
    ]

    return {
        "series_id": series_id,
        "metadata": meta,
        "observations": observations,
        "count": len(observations),
    }


def _compute_yoy_rate(observations: list[dict]) -> float | None:
    """Compute YoY % change from FRED observations (desc order by date).

    Finds the latest value and the value ~12 months prior, then returns
    ((latest / prior) - 1) * 100.  Returns None if insufficient data.
    """
    if len(observations) < 2:
        return None

    # observations are in descending date order from FRED
    try:
        latest_val = float(observations[0]["value"])
        latest_date = observations[0]["date"]  # YYYY-MM-DD
    except (KeyError, ValueError, TypeError):
        return None

    # Walk backwards to find the observation closest to 12 months prior
    target_year = int(latest_date[:4]) - 1
    target_month_day = latest_date[4:]  # e.g. "-02-01"
    target_date = f"{target_year}{target_month_day}"

    best_obs = None
    best_gap = 999
    for obs in observations[1:]:
        try:
            obs_date = obs["date"]
            # Compute approximate gap in months from the target date
            gap = abs(
                (int(obs_date[:4]) - target_year) * 12
                + int(obs_date[5:7]) - int(latest_date[5:7])
            )
            if gap < best_gap:
                best_gap = gap
                best_obs = obs
        except (KeyError, ValueError, TypeError):
            continue

    if best_obs is None or best_gap > 3:
        return None

    try:
        prior_val = float(best_obs["value"])
        if prior_val == 0:
            return None
        return round((latest_val / prior_val - 1.0) * 100.0, 2)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _fetch_fred_indicators(indicator_keys: list[str], lookback_years: int = 1) -> dict:
    """Fetch multiple FRED indicators and return latest values as a flat dict.

    Keys ending in ``_yoy`` (e.g. ``cpi_yoy``, ``core_cpi_yoy``) automatically
    compute Year-over-Year percentage change from the underlying index-level
    FRED series, so the ``latest_value`` returned is an inflation **rate** (%)
    rather than a raw index level.
    """
    start = (datetime.now() - timedelta(days=365 * max(lookback_years, 2))).strftime("%Y-%m-%d")
    results: dict[str, Any] = {}

    for key in indicator_keys:
        lkey = key.lower()
        series_id = _INDICATOR_SERIES.get(lkey)
        if not series_id:
            results[key] = {
                "error": f"Unknown indicator '{key}'. Known: {sorted(_INDICATOR_SERIES)}"
            }
            continue

        needs_yoy = lkey in _YOY_KEYS

        # For YoY computation we need at least 13+ months of data
        limit = 24 if needs_yoy else 12
        data = _fetch_fred_series(series_id, observation_start=start, limit=limit)
        if "error" in data:
            results[key] = data
            continue

        meta = data.get("metadata", {})
        obs = data.get("observations", [])

        entry: dict[str, Any] = {
            "series_id": series_id,
            "title": meta.get("title", key),
            "units": meta.get("units"),
            "frequency": meta.get("frequency"),
            "latest_date": obs[0]["date"] if obs else None,
            "history": obs[:12],
        }

        if needs_yoy:
            yoy = _compute_yoy_rate(obs)
            entry["latest_value"] = yoy
            entry["units"] = "% YoY"
            entry["index_level"] = obs[0]["value"] if obs else None
            entry["note"] = "YoY % change computed from index-level series"
        else:
            entry["latest_value"] = obs[0]["value"] if obs else None

        results[key] = entry

    return results


# ---------------------------------------------------------------------------
# FRED economic release calendar
# ---------------------------------------------------------------------------

# Curated set of market-moving FRED releases (from ~400 total).
# Maps release_id → {name, category, series[]}.
_MARKET_MOVING_RELEASES: dict[int, dict[str, Any]] = {
    10:  {"name": "Consumer Price Index (CPI)", "category": "inflation",
          "series": ["CPIAUCSL", "CPILFESL"]},
    11:  {"name": "Producer Price Index (PPI)", "category": "inflation",
          "series": ["PPIACO"]},
    13:  {"name": "Industrial Production & Capacity Utilization", "category": "production",
          "series": ["INDPRO", "TCU"]},
    14:  {"name": "Consumer Sentiment (UMich)", "category": "sentiment",
          "series": ["UMCSENT"]},
    21:  {"name": "H.15 Selected Interest Rates", "category": "monetary_policy",
          "series": ["FEDFUNDS"]},
    22:  {"name": "H.6 Money Stock Measures", "category": "monetary",
          "series": ["M2SL"]},
    46:  {"name": "Gross Domestic Product (GDP)", "category": "growth",
          "series": ["GDP", "A191RL1Q225SBEA"]},
    50:  {"name": "Employment Situation", "category": "employment",
          "series": ["UNRATE", "PAYEMS"]},
    53:  {"name": "U.S. Import/Export Price Indexes", "category": "trade",
          "series": ["IR"]},
    54:  {"name": "Manufacturers' Shipments, Inventories & Orders", "category": "production",
          "series": ["DGORDER"]},
    86:  {"name": "Personal Income & Outlays (incl. PCE)", "category": "consumption",
          "series": ["PCE", "PCEPILFE"]},
    101: {"name": "Job Openings & Labor Turnover (JOLTS)", "category": "employment",
          "series": ["JTSJOL"]},
    175: {"name": "S&P/Case-Shiller Home Price Indices", "category": "housing",
          "series": ["CSUSHPINSA"]},
    323: {"name": "New Residential Construction", "category": "housing",
          "series": ["HOUST", "PERMIT"]},
}


def _fetch_release_calendar(
    start_date: str | None = None,
    end_date: str | None = None,
    include_values: bool = True,
) -> dict:
    """Fetch FRED economic release calendar, filtered to market-moving releases.

    Calls ``fred/releases/dates`` to get scheduled releases, filters to the
    curated ``_MARKET_MOVING_RELEASES`` set, and optionally fetches the latest
    and prior values for each release's key series.
    """
    api_key = _get_fred_api_key()
    if not api_key:
        return {"error": _FRED_NO_KEY_ERROR}

    today = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = today
    if not end_date:
        end_date = start_date

    common = {"api_key": api_key, "file_type": "json"}

    try:
        with httpx.Client(timeout=25) as client:
            # 1. Get all releases in the date range
            resp = client.get(
                f"{_FRED_BASE}/releases/dates",
                params={
                    "realtime_start": start_date,
                    "realtime_end": end_date,
                    "include_release_dates_with_no_data": "true",
                    **common,
                },
            )
            if not resp.is_success:
                return {"error": f"FRED releases/dates API error {resp.status_code}: {resp.text[:200]}"}

            all_releases = resp.json().get("release_dates", [])

            # 2. Filter to market-moving releases
            calendar: list[dict[str, Any]] = []
            series_to_fetch: set[str] = set()

            for rel in all_releases:
                rid = rel.get("release_id")
                if rid not in _MARKET_MOVING_RELEASES:
                    continue
                info = _MARKET_MOVING_RELEASES[rid]
                entry: dict[str, Any] = {
                    "release_id": rid,
                    "release_name": info["name"],
                    "category": info["category"],
                    "date": rel.get("date", ""),
                    "key_series": info["series"],
                }
                calendar.append(entry)
                series_to_fetch.update(info["series"])

            # 3. Optionally fetch latest + prior values for each key series
            series_values: dict[str, dict] = {}
            if include_values and series_to_fetch:
                for sid in series_to_fetch:
                    try:
                        obs_resp = client.get(
                            f"{_FRED_BASE}/series/observations",
                            params={
                                "series_id": sid,
                                "sort_order": "desc",
                                "limit": 2,
                                **common,
                            },
                        )
                        if obs_resp.is_success:
                            obs = [
                                {"date": o["date"], "value": o["value"]}
                                for o in obs_resp.json().get("observations", [])
                                if o.get("value") not in (".", None, "")
                            ]
                            series_values[sid] = {
                                "latest": obs[0] if obs else None,
                                "prior": obs[1] if len(obs) > 1 else None,
                            }
                    except Exception:
                        series_values[sid] = {"error": "fetch failed"}

            # 4. Attach values to calendar entries
            for entry in calendar:
                entry["data"] = {}
                for sid in entry["key_series"]:
                    if sid in series_values:
                        entry["data"][sid] = series_values[sid]

    except httpx.RequestError as exc:
        return {"error": f"Network error: {exc}"}
    except Exception as exc:
        return {"error": f"Calendar fetch failed: {exc}"}

    return {
        "calendar": calendar,
        "date_range": {"start": start_date, "end": end_date},
        "total_releases_scanned": len(all_releases),
        "total_market_moving": len(calendar),
        "note": (
            "Filtered to market-moving releases only. "
            "Use fred_series for full historical data on any series."
        ),
    }


# ---------------------------------------------------------------------------
# yfinance helpers (sync)
# ---------------------------------------------------------------------------

# FX tickers: Yahoo Finance convention is BASE+QUOTE=X (e.g. EURUSD=X)
_FX_ALIASES: dict[str, str] = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "JPYUSD=X",
    "CNY": "CNYUSD=X",
    "AUD": "AUDUSD=X",
    "CAD": "CADUSD=X",
    "CHF": "CHFUSD=X",
    "MXN": "MXNUSD=X",
    "BRL": "BRLUSD=X",
    "INR": "INRUSD=X",
    "KRW": "KRWUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
}

_YIELD_ALIASES: dict[str, str] = {
    "3m": "^IRX",
    "2y": "^TWO",
    "5y": "^FVX",
    "10y": "^TNX",
    "30y": "^TYX",
    "vix": "^VIX",
}

_COMMODITY_ALIASES: dict[str, str] = {
    "oil": "CL=F",
    "crude": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "nat_gas": "NG=F",
    "natural_gas": "NG=F",
    "wheat": "ZW=F",
    "corn": "ZC=F",
    "soybeans": "ZS=F",
}


def _yf_latest(symbol: str) -> dict:
    """Fetch the latest closing price for a yfinance symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return {"error": f"No data for {symbol}"}
        return {
            "symbol": symbol,
            "price": round(float(hist["Close"].iloc[-1]), 6),
            "date": hist.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as exc:
        return {"error": str(exc), "symbol": symbol}


def _yf_fetch_many(
    items: list[str],
    resolve: Callable[[str], tuple[str, str]],
    output_key: str,
) -> dict:
    """Fetch the latest yfinance price for each item using a resolver function.

    resolve(item) -> (canonical_key, yfinance_ticker)
    """
    results = {}
    for item in items:
        canonical, ticker = resolve(item)
        results[canonical] = _yf_latest(ticker)
    return {output_key: results, "timestamp": datetime.now(timezone.utc).isoformat()}


def _fetch_fx_rates(pairs: list[str]) -> dict:
    """Fetch FX spot rates for requested currency pairs."""
    def resolve(pair: str) -> tuple[str, str]:
        c = pair.upper().strip()
        return c, _FX_ALIASES.get(c, c if "=" in c else c + "=X")
    return _yf_fetch_many(pairs, resolve, "fx_rates")


def _fetch_yields(maturities: list[str]) -> dict:
    """Fetch treasury yields and VIX from yfinance."""
    def resolve(mat: str) -> tuple[str, str]:
        c = mat.lower().strip()
        return c, _YIELD_ALIASES.get(c, f"^{c.upper()}")
    return _yf_fetch_many(maturities, resolve, "yields")


def _fetch_commodities(names: list[str]) -> dict:
    """Fetch commodity prices from yfinance."""
    def resolve(name: str) -> tuple[str, str]:
        c = name.lower().strip()
        return c, _COMMODITY_ALIASES.get(c, name.upper())
    return _yf_fetch_many(names, resolve, "commodities")


# ---------------------------------------------------------------------------
# Tool wrapper (async)
# ---------------------------------------------------------------------------

class EconomicsDataTool(Tool):
    """Fetch macroeconomic data from FRED and yfinance.

    FRED provides official US macro series (GDP, CPI, unemployment, fed funds rate, M2, etc.).
    yfinance provides FX spot rates, treasury yields, and commodity prices.

    Configure FRED key via tools.financial.fredApiKey in ~/.finclaw/config.json
    or FRED_API_KEY env var (free key at fred.stlouisfed.org).
    yfinance commands work without any API key.
    """

    name = "economics_data"
    description = (
        "Fetch macroeconomic data and release calendars. FRED provides GDP, CPI, inflation, "
        "unemployment, fed funds rate, M2 money supply, and other official US macro series. "
        "yfinance provides FX exchange rates, treasury yield curve, VIX, and commodity prices. "
        "Use command='indicators' for a multi-series dashboard in one call. "
        "Use command='calendar' to see what macro data is being released today/this week. "
        "IMPORTANT: For any question about what data is released/scheduled/coming out, "
        "use command='calendar' — do NOT use economics_analysis for that."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["fred_series", "fx_rates", "yields", "indicators", "commodity", "calendar"],
                "description": (
                    "Data fetch type. "
                    "fred_series: single FRED time series by ID; "
                    "fx_rates: FX spot rates (EUR, GBP, JPY, CNY, etc.); "
                    "yields: treasury yield curve + VIX (3m, 2y, 5y, 10y, 30y, vix); "
                    "indicators: multi-series macro dashboard from FRED; "
                    "commodity: commodity prices (oil, gold, copper, wheat, etc.); "
                    "calendar: economic release calendar — what macro data is being released "
                    "today/this week, with latest + prior values for each release."
                ),
            },
            "series_id": {
                "type": "string",
                "description": (
                    "FRED series ID (e.g. 'GDP', 'CPIAUCSL', 'FEDFUNDS', 'UNRATE', 'M2SL'). "
                    "Required for command='fred_series'."
                ),
            },
            "pairs": {
                "type": "string",
                "description": (
                    "Comma-separated currency pairs or codes for command='fx_rates'. "
                    "Examples: 'EUR,GBP,JPY' or 'EURUSD,GBPUSD'."
                ),
            },
            "maturities": {
                "type": "string",
                "description": (
                    "Comma-separated yield maturities for command='yields'. "
                    "Valid values: 3m, 2y, 5y, 10y, 30y, vix. "
                    "Default: '3m,2y,5y,10y,30y'."
                ),
            },
            "indicators": {
                "type": "string",
                "description": (
                    "Comma-separated macro indicator keys for command='indicators'. "
                    "Supported: gdp, gdp_growth, cpi (index level), cpi_yoy (YoY inflation rate %), "
                    "core_cpi (index level), core_cpi_yoy (YoY core inflation rate %), "
                    "inflation (10Y breakeven), unemployment, fed_funds, m2, industrial_production, "
                    "retail_sales, housing_starts, consumer_sentiment, leading_index, "
                    "yield_2y, yield_10y, yield_spread, credit_spread, vix. "
                    "IMPORTANT: Use cpi_yoy (not cpi) for inflation rate analysis — "
                    "cpi returns raw index level (~326), cpi_yoy returns YoY % (~2.8%). "
                    "Default: 'gdp_growth,cpi_yoy,unemployment,fed_funds,yield_spread'."
                ),
            },
            "commodities": {
                "type": "string",
                "description": (
                    "Comma-separated commodity names for command='commodity'. "
                    "Supported: oil, brent, gold, silver, copper, nat_gas, wheat, corn, soybeans. "
                    "Default: 'oil,gold,copper'."
                ),
            },
            "start_date": {
                "type": "string",
                "description": (
                    "Start date YYYY-MM-DD. For fred_series/indicators: observation start date. "
                    "For calendar: first date of release range (defaults to today). Optional."
                ),
            },
            "end_date": {
                "type": "string",
                "description": (
                    "End date YYYY-MM-DD. For fred_series/indicators: observation end date. "
                    "For calendar: last date of release range (defaults to start_date). Optional."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max number of observations to return for fred_series. Default: 100.",
                "minimum": 1,
                "maximum": 1000,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        logger.info(f"economics_data:{command}")

        if command == "fred_series":
            series_id = kwargs.get("series_id", "")
            if not series_id:
                return json.dumps({"error": "series_id is required for fred_series"})
            result = await asyncio.to_thread(
                _fetch_fred_series,
                series_id,
                kwargs.get("start_date"),
                kwargs.get("end_date"),
                int(kwargs.get("limit", 100)),
            )

        elif command == "indicators":
            keys = _csv_list(kwargs.get("indicators", "gdp_growth,cpi_yoy,unemployment,fed_funds,yield_spread"))
            result = await asyncio.to_thread(_fetch_fred_indicators, keys, 2)

        elif command == "fx_rates":
            pairs = _csv_list(kwargs.get("pairs", "EUR,GBP,JPY"))
            result = await asyncio.to_thread(_fetch_fx_rates, pairs)

        elif command == "yields":
            maturities = _csv_list(kwargs.get("maturities", "3m,2y,5y,10y,30y"))
            result = await asyncio.to_thread(_fetch_yields, maturities)

        elif command == "commodity":
            commodities = _csv_list(kwargs.get("commodities", "oil,gold,copper"))
            result = await asyncio.to_thread(_fetch_commodities, commodities)

        elif command == "calendar":
            result = await asyncio.to_thread(
                _fetch_release_calendar,
                kwargs.get("start_date"),
                kwargs.get("end_date"),
            )

        else:
            result = {"error": f"Unknown command: {command!r}"}

        return json.dumps(result, ensure_ascii=False, default=str)
