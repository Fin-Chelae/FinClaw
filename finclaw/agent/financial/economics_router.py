"""Economics Router — CFA-level macroeconomics analysis models (LLM sub-agent pattern).

The public ``EconomicsRouter`` is an LLM sub-agent: it accepts a natural-language
query, runs an inner LLM with ``_EconomicsDispatch`` as the sole inner tool, and
returns a synthesised macroeconomic analysis.  The inner LLM decides which model(s)
to run and can call them in parallel.

``_EconomicsDispatch`` (private) contains the original deterministic dispatch logic
and is exposed to the inner LLM as "run_economics_model".

Models supported:
  currency_analysis      — spot/forward FX, carry trade, triangular arbitrage
  exchange_calculations  — cross-rates, forward points, CIP
  growth_analysis        — GDP decomposition, potential GDP, growth factor comparison
  market_cycles          — business cycle phase detection, sector cyclicality
  policy_analysis        — monetary policy tools, policy effectiveness assessment
  capital_flows          — BOP analysis, FX market structure
  trade_analysis         — trade benefits/costs, trade restrictions, trading blocs
  statistical_analysis   — ADF, ARIMA, correlation, time-series analysis
  forecasting            — exponential smoothing, simple forecasting methods
  scenario_analysis      — Monte Carlo simulation, stress testing
"""

import json
import statistics
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool
from finclaw.agent.financial_tools import EconomicsDataTool

# Analytics library — analysis
from finclaw.analytics.economics.analysis.analytics_engine import (
    StatisticalAnalyzer, ForecastingEngine, ScenarioAnalyzer,
)
# Analytics library — FX
from finclaw.analytics.economics.fx.currency_analysis import (
    CarryTradeAnalyzer,
)
from finclaw.analytics.economics.fx.exchange_calculations import (
    CrossRateCalculator, ForwardCalculator,
)
# Analytics library — macro
from finclaw.analytics.economics.macro.growth_analysis import GrowthAnalyzer
from finclaw.analytics.economics.macro.market_cycles import BusinessCycleAnalyzer
from finclaw.analytics.economics.macro.policy_analysis import MonetaryPolicyAnalyzer
from finclaw.analytics.economics.macro.capital_flows import CapitalFlowAnalyzer, FXMarketAnalyzer
from finclaw.analytics.economics.macro.trade_geopolitics import TradeAnalyzer, GeopoliticalRiskAnalyzer

# Singleton data client
_economics_data = EconomicsDataTool()


async def _fetch_json(command: str, **kwargs: Any) -> dict:
    """Execute an EconomicsDataTool command and return the parsed JSON result."""
    return json.loads(await _economics_data.execute(command=command, **kwargs))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_dec(value: Any) -> Decimal:
    return Decimal(str(_safe_float(value, 0.0)))


def _extract_obs_values(fred_data: dict) -> list[float]:
    """Extract numeric observation values from a FRED series result (chronological order)."""
    obs = fred_data.get("observations", [])
    values = []
    for o in reversed(obs):   # FRED returns desc; reverse to chronological
        try:
            values.append(float(o["value"]))
        except (KeyError, ValueError, TypeError):
            pass
    return values


def _latest_indicator(indicator_data: dict, key: str) -> float | None:
    entry = indicator_data.get(key, {})
    v = entry.get("latest_value") if isinstance(entry, dict) else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class _EconomicsDispatch(Tool):
    """Inner tool: deterministic macroeconomics dispatch (used by EconomicsRouter sub-agent).

    Fetches FRED + market data automatically and runs the requested analytics model.
    Exposed to the inner LLM as ``run_economics_model``.
    """

    name = "run_economics_model"
    description = (
        "Run a single macroeconomics analysis model. "
        "Fetches FRED and market data automatically. "
        "Models: currency_analysis, exchange_calculations, growth_analysis, market_cycles, "
        "policy_analysis, capital_flows, trade_analysis, statistical_analysis, "
        "forecasting, scenario_analysis."
    )
    parameters = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": [
                    "currency_analysis",
                    "exchange_calculations",
                    "growth_analysis",
                    "market_cycles",
                    "policy_analysis",
                    "capital_flows",
                    "trade_analysis",
                    "statistical_analysis",
                    "forecasting",
                    "scenario_analysis",
                ],
                "description": (
                    "Analysis model to run. "
                    "currency_analysis: spot/forward FX, carry trade, triangular arbitrage; "
                    "exchange_calculations: cross-rates, forward points, CIP; "
                    "growth_analysis: potential GDP, growth factor decomposition; "
                    "market_cycles: business cycle phase detection, sector cyclicality; "
                    "policy_analysis: monetary policy tools, policy effectiveness; "
                    "capital_flows: BOP analysis, FX market structure; "
                    "trade_analysis: trade benefits/costs, trade restrictions; "
                    "statistical_analysis: ADF test, ARIMA, correlation analysis; "
                    "forecasting: exponential smoothing, naive/drift/average forecasts; "
                    "scenario_analysis: Monte Carlo simulation, stress testing."
                ),
            },
            "country": {
                "type": "string",
                "description": "Country code or name for country-specific analysis (e.g., 'US', 'EUR', 'CHN'). Optional.",
            },
            "currency_pair": {
                "type": "string",
                "description": "FX pair for currency/FX models (e.g., 'EURUSD', 'USDJPY'). Optional.",
            },
            "indicators": {
                "type": "string",
                "description": (
                    "Comma-separated FRED indicator keys to fetch for analysis. "
                    "Supported: gdp_growth, cpi, unemployment, fed_funds, yield_spread, m2, vix, etc. "
                    "Default varies by model."
                ),
            },
            "fred_series": {
                "type": "string",
                "description": (
                    "Comma-separated FRED series IDs for statistical_analysis or forecasting "
                    "(e.g., 'GDP,CPIAUCSL,UNRATE'). Optional."
                ),
            },
            "data_input": {
                "type": "object",
                "description": (
                    "Explicit data dict for offline / custom analysis "
                    "(bypasses FRED/yfinance fetch). Keys depend on model."
                ),
            },
            "forecast_periods": {
                "type": "integer",
                "description": "Number of periods to forecast (forecasting model). Default: 8.",
                "minimum": 1,
                "maximum": 60,
            },
            "simulations": {
                "type": "integer",
                "description": "Number of Monte Carlo simulations (scenario_analysis). Default: 1000.",
                "minimum": 100,
                "maximum": 50000,
            },
        },
        "required": ["model"],
    }

    async def execute(self, **kwargs: Any) -> str:
        model = kwargs.get("model", "market_cycles")
        logger.info(f"economics_analysis  model={model}")

        result: dict[str, Any] = {"model": model}

        try:
            if model == "currency_analysis":
                result.update(await self._run_currency_analysis(kwargs))
            elif model == "exchange_calculations":
                result.update(await self._run_exchange_calculations(kwargs))
            elif model == "growth_analysis":
                result.update(await self._run_growth_analysis(kwargs))
            elif model == "market_cycles":
                result.update(await self._run_market_cycles(kwargs))
            elif model == "policy_analysis":
                result.update(await self._run_policy_analysis(kwargs))
            elif model == "capital_flows":
                result.update(await self._run_capital_flows(kwargs))
            elif model == "trade_analysis":
                result.update(await self._run_trade_analysis(kwargs))
            elif model == "statistical_analysis":
                result.update(await self._run_statistical_analysis(kwargs))
            elif model == "forecasting":
                result.update(await self._run_forecasting(kwargs))
            elif model == "scenario_analysis":
                result.update(await self._run_scenario_analysis(kwargs))
            else:
                result["error"] = f"Unknown model: {model!r}"

        except Exception as exc:
            logger.warning(f"economics_analysis error ({model}): {exc}")
            result["error"] = str(exc)
            result["note"] = (
                "Model calculation failed. Check that required data is available or "
                "supply explicit values via the data_input parameter."
            )

        return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    async def _run_currency_analysis(self, kwargs: dict) -> dict:
        """Spot/forward FX, carry trade."""
        pair = kwargs.get("currency_pair", "EURUSD")
        data_input = kwargs.get("data_input", {})

        if data_input:
            fx_rates = data_input.get("fx_rates", {})
            yields = data_input.get("yields", {})
        else:
            fx_json = await _fetch_json("fx_rates", pairs=pair)
            yields_json = await _fetch_json("yields", maturities="3m,10y")
            fx_rates = fx_json.get("fx_rates", {})
            yields = yields_json.get("yields", {})

        pair_key = pair.upper().replace("/", "")
        spot_entry = fx_rates.get(pair_key, {})
        spot_rate = _safe_float(spot_entry.get("price"), 1.10)

        rate_3m = _safe_float(yields.get("3m", {}).get("price"), 5.0) / 100
        rate_10y = _safe_float(yields.get("10y", {}).get("price"), 4.0) / 100

        result: dict[str, Any] = {
            "currency_pair": pair_key,
            "spot_rate": spot_rate,
            "data_source": "yfinance" if not data_input else "manual",
        }

        # Carry trade: fund at short rate, invest at long rate
        try:
            carry = CarryTradeAnalyzer()
            carry_result = carry.calculate_carry_trade_return(
                funding_currency_rate=_to_dec(rate_3m),
                target_currency_rate=_to_dec(rate_10y),
                exchange_rate_change=_to_dec(0.0),   # assume unchanged spot
                time_period=_to_dec(1.0),
            )
            result["carry_trade"] = carry_result
        except Exception as e:
            result["carry_trade_error"] = str(e)

        # Forward premium/discount (requires a forward rate — approximate via CIP)
        try:
            fwd_calc = ForwardCalculator()
            # Forward = Spot * (1 + domestic) / (1 + foreign); use rate diff as premium
            rate_diff_pct = (rate_10y - rate_3m) * 100
            fwd_result = fwd_calc.calculate_forward_rate_percentage(
                spot_rate=_to_dec(spot_rate),
                premium_discount_percent=_to_dec(rate_diff_pct),
                time_to_maturity=_to_dec(1.0),
            )
            result["forward_calculation"] = fwd_result
        except Exception as e:
            result["forward_calculation_error"] = str(e)

        return result

    async def _run_exchange_calculations(self, kwargs: dict) -> dict:
        """Cross-rates, forward points, CIP."""
        data_input = kwargs.get("data_input", {})

        if data_input:
            base_usd = _safe_float(data_input.get("base_usd", 1.10))
            quote_usd = _safe_float(data_input.get("quote_usd", 1.27))
            premium_pct = _safe_float(data_input.get("premium_pct", 0.5))
            tenor = _safe_float(data_input.get("tenor_years", 1.0))
            pair = data_input.get("pair", "EURGBP")
        else:
            fx_json = await _fetch_json("fx_rates", pairs="EUR,GBP")
            base_usd = _safe_float(fx_json.get("fx_rates", {}).get("EUR", {}).get("price"), 1.10)
            quote_usd = _safe_float(fx_json.get("fx_rates", {}).get("GBP", {}).get("price"), 1.27)
            yields_json = await _fetch_json("yields", maturities="10y")
            rate_10y = _safe_float(yields_json.get("yields", {}).get("10y", {}).get("price"), 4.0)
            premium_pct = rate_10y * 0.1   # rough forward premium proxy
            tenor = 1.0
            pair = "EURGBP"

        cross_calc = CrossRateCalculator()
        fwd_calc = ForwardCalculator()

        result: dict[str, Any] = {
            "base_usd": round(base_usd, 6),
            "quote_usd": round(quote_usd, 6),
            "pair": pair,
        }

        try:
            # CrossRateCalculator expects base_quote_rates dict and currency_pair string
            cross_result = cross_calc.calculate_cross_rate(
                base_quote_rates={"USD_BASE": _to_dec(base_usd), "USD_QUOTE": _to_dec(quote_usd)},
                currency_pair=pair,
            )
            result["cross_rate"] = cross_result
        except Exception as e:
            # Fallback: compute directly
            result["cross_rate"] = {"implied": round(base_usd / quote_usd, 6), "error": str(e)}

        try:
            fwd_result = fwd_calc.calculate_forward_rate_percentage(
                spot_rate=_to_dec(base_usd),
                premium_discount_percent=_to_dec(premium_pct),
                time_to_maturity=_to_dec(tenor),
            )
            result["forward_rate"] = fwd_result
        except Exception as e:
            result["forward_rate_error"] = str(e)

        return result

    async def _run_growth_analysis(self, kwargs: dict) -> dict:
        """GDP growth factor comparison and potential GDP forecast."""
        data_input = kwargs.get("data_input", {})
        country = kwargs.get("country", "US")

        if data_input:
            economic_data = data_input
        else:
            ind_json = await _fetch_json(
                "indicators",
                indicators="gdp_growth,unemployment",
            )
            if "error" in ind_json:
                return {"warning": ind_json["error"], "note": "Supply data via data_input for offline use."}

            gdp_growth = _latest_indicator(ind_json, "gdp_growth")
            unemployment = _latest_indicator(ind_json, "unemployment")
            economic_data = {
                "gdp_growth": gdp_growth,
                "unemployment": unemployment,
            }

        analyzer = GrowthAnalyzer()
        result: dict[str, Any] = {"country": country, "economic_snapshot": economic_data}

        # compare_growth_factors expects specific field names for the scoring model.
        # Map FRED data to reasonable proxies for a US developed-economy assessment.
        gdp_growth_val = _safe_float(economic_data.get("gdp_growth"), 2.5)
        analyzer_data = {
            "gdp_per_capita": data_input.get("gdp_per_capita", 65000),  # US approx
            "rd_spending_percent_gdp": data_input.get("rd_spending_percent_gdp", 3.5),  # US ~3.5%
            "education_index": data_input.get("education_index", 0.9),  # US ~0.9
            "infrastructure_quality": data_input.get("infrastructure_quality", 80),
            "population_growth_rate": data_input.get("population_growth_rate", 0.5),  # US ~0.5%
            "old_age_dependency_ratio": data_input.get("old_age_dependency_ratio", 28),  # US ~28%
        }

        try:
            growth_factors = analyzer.compare_growth_factors(
                country_type="developed",
                economic_data=analyzer_data,
            )
            result["growth_factors"] = growth_factors
        except Exception as e:
            result["growth_factors_error"] = str(e)

        # forecast_potential_gdp uses growth accounting: g_Y = g_A + α*g_K + (1-α)*g_L
        # Supply reasonable US defaults when not provided via data_input.
        try:
            gdp_g = _safe_float(economic_data.get("gdp_growth"), 2.5)
            forecast_data = data_input if data_input else {
                "labor_force_growth": [0.5, 0.6, 0.4, 0.5, 0.5],  # US recent ~0.5%
                "productivity_growth": [1.2, 1.5, 1.0, 1.3, 1.4],  # US ~1.2-1.5%
                "capital_growth": [2.0, 2.5, 1.8, 2.2, 2.0],  # US ~2%
            }
            forecast_assumptions = data_input.get("forecast_assumptions", {}) if data_input else {
                "periods": 5,
                "labor_growth_rate": 0.5,
                "productivity_growth_rate": 1.3,
                "capital_growth_rate": 2.0,
                "capital_share": 0.3,
            }
            gdp_forecast = analyzer.forecast_potential_gdp(
                historical_data=forecast_data,
                forecast_assumptions=forecast_assumptions,
            )
            result["potential_gdp_forecast"] = gdp_forecast
        except Exception as e:
            result["potential_gdp_forecast_error"] = str(e)

        return result

    async def _run_market_cycles(self, kwargs: dict) -> dict:
        """Business cycle phase detection with comprehensive market context."""
        data_input = kwargs.get("data_input", {})

        if data_input:
            indicator_data = data_input
            market_data: dict[str, Any] = {}
        else:
            # Fetch FRED indicators (two batches to stay within limits)
            ind_json = await _fetch_json(
                "indicators",
                indicators=(
                    "gdp_growth,unemployment,yield_spread,cpi_yoy,core_cpi_yoy,"
                    "fed_funds,consumer_sentiment,leading_index,"
                    "credit_spread,vix,yield_2y,yield_10y"
                ),
            )
            if "error" in ind_json:
                return {"warning": ind_json["error"], "note": "Supply indicator data via data_input for offline use."}

            indicator_data = {
                "gdp_growth": _latest_indicator(ind_json, "gdp_growth"),
                "unemployment": _latest_indicator(ind_json, "unemployment"),
                "yield_spread": _latest_indicator(ind_json, "yield_spread"),
                "cpi_inflation": _latest_indicator(ind_json, "cpi_yoy"),
                "core_cpi_inflation": _latest_indicator(ind_json, "core_cpi_yoy"),
                "fed_funds": _latest_indicator(ind_json, "fed_funds"),
                "consumer_sentiment": _latest_indicator(ind_json, "consumer_sentiment"),
                "leading_index": _latest_indicator(ind_json, "leading_index"),
                "credit_spread_hy_oas": _latest_indicator(ind_json, "credit_spread"),
                "vix": _latest_indicator(ind_json, "vix"),
                "yield_2y": _latest_indicator(ind_json, "yield_2y"),
                "yield_10y": _latest_indicator(ind_json, "yield_10y"),
            }

            # Fetch market-based data (commodities, USD) via yfinance
            try:
                commodity_json = await _fetch_json("commodity", commodities="oil,gold")
                market_data = {
                    k: v.get("price")
                    for k, v in commodity_json.get("commodities", {}).items()
                    if isinstance(v, dict) and "price" in v
                }
            except Exception:
                market_data = {}

        # Remove None values
        clean_data = {k: v for k, v in indicator_data.items() if v is not None}

        analyzer = BusinessCycleAnalyzer()
        result: dict[str, Any] = {"indicators_snapshot": clean_data}

        if market_data:
            result["market_snapshot"] = market_data

        # Map router field names → analyzer expected field names
        analyzer_input = {
            "gdp_growth_rate": clean_data.get("gdp_growth", 0),
            "unemployment_rate": clean_data.get("unemployment", 0),
            "inflation_rate": clean_data.get("cpi_inflation", 0),
            "interest_rate": clean_data.get("fed_funds", 0),
            "consumer_confidence": clean_data.get("consumer_sentiment", 0),
        }

        try:
            phase = analyzer.detect_cycle_phase(economic_indicators=analyzer_input)
            result["business_cycle_phase"] = phase
        except Exception as e:
            result["business_cycle_phase_error"] = str(e)

        # --- Derived signals ---

        # Yield curve
        spread = clean_data.get("yield_spread")
        if spread is not None:
            result["yield_curve"] = {
                "spread_10y_2y_pct": spread,
                "signal": (
                    "inverted (recession signal)" if spread < 0
                    else "flat (caution)" if spread < 0.5
                    else "normal (expansion)"
                ),
                "yield_2y_pct": clean_data.get("yield_2y"),
                "yield_10y_pct": clean_data.get("yield_10y"),
            }

        # Real yields (10Y nominal − headline CPI YoY)
        y10 = clean_data.get("yield_10y")
        cpi = clean_data.get("cpi_inflation")
        if y10 is not None and cpi is not None:
            real_10y = round(y10 - cpi, 2)
            result["real_yield_10y_pct"] = real_10y
            result["real_yield_signal"] = (
                "strongly positive (tight conditions)" if real_10y > 2.0
                else "positive (mildly tight)" if real_10y > 0.5
                else "near zero (neutral)" if real_10y > -0.5
                else "negative (easy conditions)"
            )

        # Credit conditions
        hy_spread = clean_data.get("credit_spread_hy_oas")
        if hy_spread is not None:
            result["credit_conditions"] = {
                "hy_oas_spread_pct": hy_spread,
                "signal": (
                    "distressed (>6%)" if hy_spread > 6.0
                    else "tight credit (>4.5%)" if hy_spread > 4.5
                    else "normal (2.5-4.5%)" if hy_spread > 2.5
                    else "easy credit (<2.5%)"
                ),
            }

        # VIX context
        vix = clean_data.get("vix")
        if vix is not None:
            result["volatility"] = {
                "vix": vix,
                "signal": (
                    "extreme fear (>30)" if vix > 30
                    else "elevated (20-30)" if vix > 20
                    else "normal (12-20)" if vix > 12
                    else "complacent (<12)"
                ),
            }

        # Consumer sentiment context (UMich: long-run avg ~85, recent low ~50, boom >100)
        sentiment = clean_data.get("consumer_sentiment")
        if sentiment is not None:
            result["consumer_sentiment_context"] = {
                "value": sentiment,
                "series": "UMich Consumer Sentiment (UMCSENT)",
                "long_run_avg": 85,
                "signal": (
                    "very pessimistic (<60, well below average)" if sentiment < 60
                    else "below average (60-80)" if sentiment < 80
                    else "average (80-95)" if sentiment < 95
                    else "optimistic (>95)"
                ),
            }

        return result

    async def _run_policy_analysis(self, kwargs: dict) -> dict:
        """Monetary policy tools and effectiveness assessment."""
        data_input = kwargs.get("data_input", {})

        if data_input:
            policy_data = data_input
        else:
            ind_json = await _fetch_json(
                "indicators",
                indicators="cpi_yoy,core_cpi_yoy,unemployment,fed_funds,yield_spread,gdp_growth,yield_10y,inflation",
            )
            if "error" in ind_json:
                return {"warning": ind_json["error"], "note": "Supply indicator data via data_input for offline use."}

            policy_data = {
                "current_rate": _latest_indicator(ind_json, "fed_funds"),
                "headline_cpi_yoy": _latest_indicator(ind_json, "cpi_yoy"),
                "core_cpi_yoy": _latest_indicator(ind_json, "core_cpi_yoy"),
                "breakeven_10y": _latest_indicator(ind_json, "inflation"),
                "unemployment": _latest_indicator(ind_json, "unemployment"),
                "gdp_growth": _latest_indicator(ind_json, "gdp_growth"),
                "yield_spread": _latest_indicator(ind_json, "yield_spread"),
                "yield_10y": _latest_indicator(ind_json, "yield_10y"),
            }
            # Keep "inflation" key pointing to headline for analyzer compat
            policy_data["inflation"] = policy_data["headline_cpi_yoy"]

        analyzer = MonetaryPolicyAnalyzer()
        result: dict[str, Any] = {"policy_snapshot": policy_data}

        try:
            tools = analyzer.analyze_monetary_tools(policy_data)
            result["monetary_tools_analysis"] = tools
        except Exception as e:
            result["monetary_tools_error"] = str(e)

        try:
            effectiveness = analyzer.assess_policy_effectiveness(
                effectiveness_data=policy_data
            )
            result["policy_effectiveness"] = effectiveness
        except Exception as e:
            result["policy_effectiveness_error"] = str(e)

        # --- Taylor Rule (dual: headline CPI + core CPI) ---
        # Taylor (1993): r = r* + π + 0.5*(π − π*) + 0.5*(y − y*)
        # Using Okun's law proxy: output gap ≈ −2*(u − u*), so 0.5*(y−y*) ≈ 0.5*(4−u)*(-2)*(-1) = 0.5*(4−u)
        # Simplified: r = 2 + π + 0.5*(π − 2) + 0.5*(4 − u)
        headline = policy_data.get("headline_cpi_yoy") or policy_data.get("inflation")
        core = policy_data.get("core_cpi_yoy")
        unemployment = policy_data.get("unemployment")
        current_rate = policy_data.get("current_rate")
        real_yield_10y = None
        if policy_data.get("yield_10y") is not None and headline is not None:
            real_yield_10y = round(policy_data["yield_10y"] - headline, 2)

        def _taylor(pi: float, u: float) -> float:
            return 2.0 + pi + 0.5 * (pi - 2.0) + 0.5 * (4.0 - u)

        taylor_rules: dict[str, Any] = {}
        if headline is not None and unemployment is not None:
            taylor_headline = _taylor(headline, unemployment)
            taylor_rules["headline_cpi"] = {
                "implied_rate_pct": round(taylor_headline, 2),
                "inflation_used_pct": round(headline, 2),
            }
        if core is not None and unemployment is not None:
            taylor_core = _taylor(core, unemployment)
            taylor_rules["core_cpi"] = {
                "implied_rate_pct": round(taylor_core, 2),
                "inflation_used_pct": round(core, 2),
            }

        if taylor_rules:
            result["taylor_rule"] = {
                "variants": taylor_rules,
                "assumptions": {
                    "neutral_real_rate_pct": 2.0,
                    "inflation_target_pct": 2.0,
                    "natural_unemployment_pct": 4.0,
                    "note": (
                        "Taylor rule is one benchmark among many. Result depends heavily "
                        "on assumed r*, inflation measure (headline vs core vs PCE), and "
                        "output-gap proxy. Treat as directional, not definitive."
                    ),
                },
                "inputs": {
                    "headline_cpi_yoy_pct": round(headline, 2) if headline else None,
                    "core_cpi_yoy_pct": round(core, 2) if core else None,
                    "unemployment_pct": round(unemployment, 2) if unemployment else None,
                    "fed_funds_pct": round(current_rate, 2) if current_rate is not None else None,
                    "real_yield_10y_pct": real_yield_10y,
                },
            }

            # Stance: average of headline/core Taylor rules vs current rate
            taylor_rates = [v["implied_rate_pct"] for v in taylor_rules.values()]
            avg_taylor = sum(taylor_rates) / len(taylor_rates)
            result["taylor_rule_implied_rate_pct"] = round(avg_taylor, 2)

            if current_rate is not None:
                diff = current_rate - avg_taylor
                if diff > 1.0:
                    stance = "restrictive"
                elif diff > 0.25:
                    stance = "mildly restrictive"
                elif diff < -1.0:
                    stance = "accommodative"
                elif diff < -0.25:
                    stance = "mildly accommodative"
                else:
                    stance = "roughly neutral"
                result["policy_stance"] = stance
                result["policy_stance_detail"] = (
                    f"Fed funds {current_rate:.2f}% vs avg Taylor {avg_taylor:.2f}% "
                    f"(headline={taylor_rules.get('headline_cpi', {}).get('implied_rate_pct', 'N/A')}%, "
                    f"core={taylor_rules.get('core_cpi', {}).get('implied_rate_pct', 'N/A')}%). "
                    f"Gap: {diff:+.2f}pp."
                )

        return result

    async def _run_capital_flows(self, kwargs: dict) -> dict:
        """BOP analysis, FX market structure."""
        data_input = kwargs.get("data_input", {})

        if data_input:
            market_data = data_input
        else:
            fx_json = await _fetch_json("fx_rates", pairs="EUR,GBP,JPY,CNY")
            if "error" in fx_json:
                return {"warning": fx_json["error"]}
            fx_rates = fx_json.get("fx_rates", {})
            market_data = {"fx_rates": {k: v.get("price") for k, v in fx_rates.items() if "price" in v}}

        fx_analyzer = FXMarketAnalyzer()
        cap_analyzer = CapitalFlowAnalyzer()
        result: dict[str, Any] = {"market_data_snapshot": market_data}

        try:
            fx_structure = fx_analyzer.analyze_fx_market_structure(market_data=market_data)
            result["fx_market_structure"] = fx_structure
        except Exception as e:
            result["fx_market_structure_error"] = str(e)

        try:
            bop = cap_analyzer.analyze_balance_of_payments_impact(market_data)
            result["bop_analysis"] = bop
        except Exception as e:
            result["bop_analysis_error"] = str(e)

        return result

    async def _run_trade_analysis(self, kwargs: dict) -> dict:
        """Trade benefits/costs, trade restrictions, geopolitical risk."""
        data_input = kwargs.get("data_input", {})
        country = kwargs.get("country", "US")

        if data_input:
            trade_data = data_input
        else:
            # Use trade balance from FRED
            fred_json = await _fetch_json("fred_series", series_id="BOPGSTB", limit=12)
            values = _extract_obs_values(fred_json) if "error" not in fred_json else []
            trade_data = {
                "country": country,
                "trade_balance_series": values,
                "latest_trade_balance": values[-1] if values else None,
                "title": fred_json.get("metadata", {}).get("title", "US Trade Balance"),
                "units": fred_json.get("metadata", {}).get("units", "Millions of Dollars"),
            }

        analyzer = TradeAnalyzer()
        geo_analyzer = GeopoliticalRiskAnalyzer()
        result: dict[str, Any] = {
            "country": country,
            "trade_data_points": len(trade_data.get("trade_balance_series", [])),
            "latest_trade_balance": trade_data.get("latest_trade_balance"),
        }

        try:
            benefits = analyzer.analyze_trade_benefits_costs(trade_data=trade_data)
            result["trade_benefits_costs"] = benefits
        except Exception as e:
            result["trade_benefits_costs_error"] = str(e)

        try:
            geo_risk = geo_analyzer.assess_geopolitical_risk(trade_data)
            result["geopolitical_risk"] = geo_risk
        except Exception as e:
            result["geopolitical_risk_error"] = str(e)

        return result

    async def _run_statistical_analysis(self, kwargs: dict) -> dict:
        """ADF stationarity test, ARIMA, correlation analysis."""
        data_input = kwargs.get("data_input", {})
        fred_series_str = kwargs.get("fred_series", "")

        if data_input:
            series_dict = {k: v for k, v in data_input.items() if isinstance(v, list)}
        elif fred_series_str:
            ids = [s.strip() for s in fred_series_str.split(",") if s.strip()][:4]
            limit = 100
        else:
            ids = ["A191RL1Q225SBEA", "UNRATE"]
            limit = 80

        if not data_input:
            series_dict = {}
            for sid in ids:
                fj = await _fetch_json("fred_series", series_id=sid, limit=limit)
                if "error" not in fj:
                    series_dict[sid] = _extract_obs_values(fj)

        if not series_dict:
            return {
                "error": "No data available for statistical analysis. "
                         "Supply series via fred_series parameter or data_input."
            }

        analyzer = StatisticalAnalyzer()
        result: dict[str, Any] = {
            "series_analyzed": list(series_dict.keys()),
            "series_lengths": {k: len(v) for k, v in series_dict.items()},
        }

        # Time-series analysis (includes ADF stationarity) for each series
        ts_results = {}
        for name, values in series_dict.items():
            if len(values) >= 10:
                try:
                    ts = analyzer.time_series_analysis(data=pd.Series(values))
                    ts_results[name] = ts
                except Exception as e:
                    ts_results[name] = {"error": str(e)}
        if ts_results:
            result["time_series_analysis"] = ts_results

        # ARIMA forecast on first series with enough data
        first_name, first_values = next(iter(series_dict.items()))
        if len(first_values) >= 20:
            try:
                arima = analyzer.arima_forecast(
                    data=pd.Series(first_values),
                    forecast_periods=4,
                )
                result["arima_forecast"] = {"series": first_name, **arima}
            except Exception as e:
                result["arima_forecast_error"] = str(e)

        # Correlation if multiple series
        if len(series_dict) >= 2:
            try:
                series_list = list(series_dict.values())
                min_len = min(len(s) for s in series_list)
                df = pd.DataFrame(
                    {k: v[-min_len:] for k, v in series_dict.items()}
                )
                corr = analyzer.correlation_analysis(data=df)
                result["correlation_analysis"] = corr
            except Exception as e:
                result["correlation_error"] = str(e)

        return result

    async def _run_forecasting(self, kwargs: dict) -> dict:
        """Simple forecasting: naive, drift, average, exponential smoothing."""
        data_input = kwargs.get("data_input", {})
        fred_series_str = kwargs.get("fred_series", "CPIAUCSL")
        forecast_periods = int(kwargs.get("forecast_periods", 8))

        if data_input:
            values = data_input.get("values", [])
            series_name = data_input.get("name", "custom")
        else:
            sid = fred_series_str.split(",")[0].strip()
            fj = await _fetch_json("fred_series", series_id=sid, limit=120)
            if "error" in fj:
                return {"warning": fj["error"], "note": "Supply values via data_input for offline use."}
            values = _extract_obs_values(fj)
            series_name = fj.get("metadata", {}).get("title", sid)

        if len(values) < 12:
            return {"error": f"Insufficient data for forecasting (need ≥12, got {len(values)})"}

        engine = ForecastingEngine()
        result: dict[str, Any] = {
            "series": series_name,
            "history_points": len(values),
            "forecast_periods": forecast_periods,
        }

        try:
            forecasts = engine.simple_forecasting_methods(
                data=pd.Series(values),
                forecast_periods=forecast_periods,
                methods=["naive", "mean", "linear_trend", "exponential_smoothing"],
            )
            result["forecasts"] = forecasts
        except Exception as e:
            result["forecasting_error"] = str(e)

        return result

    async def _run_scenario_analysis(self, kwargs: dict) -> dict:
        """Monte Carlo simulation."""
        data_input = kwargs.get("data_input", {})
        simulations = int(kwargs.get("simulations", 1000))

        if data_input:
            base_value = _safe_float(data_input.get("base_value", 2.5))
            volatility = _safe_float(data_input.get("volatility", 1.5))
            drift = _safe_float(data_input.get("drift", 0.0))
            time_periods = int(data_input.get("time_periods", 4))
            series_name = data_input.get("name", "custom")
        else:
            fj = await _fetch_json("fred_series", series_id="A191RL1Q225SBEA", limit=60)
            values = _extract_obs_values(fj) if "error" not in fj else []
            if len(values) >= 2:
                base_value = statistics.mean(values[-20:]) if len(values) >= 20 else statistics.mean(values)
                volatility = statistics.stdev(values) if len(values) > 1 else 1.5
            else:
                base_value, volatility = 2.5, 1.5   # historical US GDP growth approx
            drift = 0.0
            time_periods = 4
            series_name = "US Real GDP Growth Rate (%)"

        analyzer = ScenarioAnalyzer()
        result: dict[str, Any] = {
            "series": series_name,
            "simulations": simulations,
            "base_value": base_value,
            "volatility": round(volatility, 4),
        }

        try:
            mc = analyzer.monte_carlo_simulation(
                base_value=base_value,
                volatility=volatility,
                drift=drift,
                time_periods=time_periods,
                num_simulations=simulations,
            )
            result["monte_carlo"] = mc
        except Exception as e:
            result["monte_carlo_error"] = str(e)

        return result


# ---------------------------------------------------------------------------
# Public sub-agent wrapper (Dexter pattern)
# ---------------------------------------------------------------------------

class EconomicsRouter(LLMRouterTool):
    """CFA-level macroeconomics sub-agent.

    Accepts a natural-language query; the inner LLM decides which model(s) to run
    via ``run_economics_model`` and can call them in parallel for a comprehensive view.
    """

    name = "economics_analysis"
    description = (
        "Run macroeconomic analysis: FX & currency analysis, GDP growth decomposition, "
        "business cycle phase detection, monetary policy analysis (Taylor Rule), "
        "capital flows & BOP, trade analysis, statistical tests (ADF, ARIMA), "
        "time-series forecasting, and Monte Carlo scenario analysis. "
        "Fetches FRED and market data automatically. "
        "NOT for raw data lookups or current prices — use economics_data for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language query. Examples: "
                    "'What is the current US business cycle phase?', "
                    "'Analyse EUR/USD carry trade opportunity', "
                    "'Run GDP growth analysis for the US', "
                    "'What does the Taylor Rule imply for the current Fed stance?', "
                    "'Comprehensive macro overview: business cycles + policy + trade'."
                ),
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a CFA-level macroeconomics analyst.\n\n"
        "Use run_economics_model to run analysis models. FRED and market data is fetched automatically.\n\n"
        "Model quick-reference:\n"
        "  market_cycles      → business cycle phase (GDP, unemployment, CPI, VIX, yield curve)\n"
        "  policy_analysis    → monetary policy tools, Taylor Rule, policy stance\n"
        "  growth_analysis    → GDP decomposition, potential GDP forecast\n"
        "  currency_analysis  → FX carry trade, forward rates (needs currency_pair)\n"
        "  exchange_calculations → cross-rates, CIP (needs currency_pair)\n"
        "  capital_flows      → BOP analysis, FX market structure\n"
        "  trade_analysis     → trade benefits/costs, geopolitical risk\n"
        "  statistical_analysis → ADF stationarity, ARIMA, correlation (needs fred_series)\n"
        "  forecasting        → exponential smoothing, trend forecasting (needs fred_series)\n"
        "  scenario_analysis  → Monte Carlo simulation\n\n"
        "For a broad macro overview, run market_cycles + policy_analysis in parallel.\n"
        "Summarise the key macro findings in 2-4 sentences. Raw data is preserved separately."
    )

    def _build_inner_tools(self) -> list[Tool]:
        return [_EconomicsDispatch()]

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"economics_analysis (inner-LLM): {query[:120]}")
        return await self._run_inner_agent(query)
