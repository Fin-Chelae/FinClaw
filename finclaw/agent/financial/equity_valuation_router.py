"""Equity Valuation Router — CFA-level quantitative valuation models (LLM sub-agent pattern).

The public ``EquityValuationRouter`` is an LLM sub-agent: it accepts a natural-language
query, runs an inner LLM with ``_EquityValuationDispatch`` as the sole inner tool, and
returns a synthesised analysis.  The inner LLM decides which model(s) to run and can
call them in parallel for a comprehensive view.

``_EquityValuationDispatch`` (private) contains the original deterministic dispatch logic
and is exposed to the inner LLM as "run_valuation_model".

Models supported:
  dcf_fcff          — Free Cash Flow to Firm (WACC discounting)
  dcf_fcfe          — Free Cash Flow to Equity
  ddm_gordon        — Gordon (Constant) Growth Model
  ddm_two_stage     — Two-Stage DDM
  ddm_h_model       — H-Model DDM
  ddm_three_stage   — Three-Stage DDM
  multiples         — P/E, P/B, P/S, EV/EBITDA from yfinance data
  residual_income   — Multi-stage Residual Income Model
  eva               — Economic Value Added Model
  fundamental_ratios — Comprehensive ratio analysis
"""

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.agent.tools.llm_router import LLMRouterTool
from finclaw.agent.financial_tools import YFinanceTool

# Analytics library
from finclaw.analytics.equity.valuation.dcf_models import FCFFModel, FCFEModel
from finclaw.analytics.equity.valuation.dividend_models import (
    GordonGrowthModel, TwoStageDDM, HModelDDM, ThreeStageDDM,
)
from finclaw.analytics.equity.valuation.residual_income import (
    ResidualIncomeModel, EconomicValueAddedModel,
)
from finclaw.analytics.equity.analysis.fundamental_analysis import (
    ProfitabilityRatios, LiquidityRatios, SolvencyRatios,
    EfficiencyRatios, DuPontAnalysis,
)

# Singleton yfinance client
_yfinance = YFinanceTool()

# Default macro assumptions (user can override via router parameters)
_RISK_FREE_RATE = 0.045      # ~10-yr Treasury yield
_EQUITY_RISK_PREMIUM = 0.055
_DEFAULT_REQUIRED_RETURN = 0.10
_DEFAULT_TERMINAL_GROWTH = 0.025
_DEFAULT_FORECAST_YEARS = 5
_RECOMMENDATION_THRESHOLD = 0.15  # ±15% upside for BUY/SELL signal


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _recommend(upside: float) -> str:
    """Translate an upside/downside fraction into a BUY/HOLD/SELL signal."""
    if upside > _RECOMMENDATION_THRESHOLD:
        return "BUY"
    if upside < -_RECOMMENDATION_THRESHOLD:
        return "SELL"
    return "HOLD"


def _project_cash_flows(base_cf: float, growth_rate: float, years: int) -> list[float]:
    """Project cash flows forward using constant growth."""
    flows = []
    cf = base_cf
    for _ in range(years):
        cf *= (1 + growth_rate)
        flows.append(cf)
    return flows


def _extract_multiples(info: dict) -> dict:
    """Extract valuation multiples directly from yfinance info."""
    pe = _safe_float(info.get("pe_ratio"))
    return {
        "pe_ratio": pe,
        "forward_pe": _safe_float(info.get("forward_pe")),
        "pb_ratio": _safe_float(info.get("price_to_book")),
        "ps_ratio": _safe_float(info.get("price_to_sales_trailing_12months")),
        "ev_ebitda": _safe_float(info.get("enterprise_to_ebitda")),
        "ev_revenue": _safe_float(info.get("enterprise_to_revenue")),
        "peg_ratio": _safe_float(info.get("peg_ratio")),
        "dividend_yield": _safe_float(info.get("dividend_yield")),
        "earnings_yield": 1.0 / pe if pe > 0 else 0.0,
        "current_price": _safe_float(info.get("current_price")),
        "market_cap": _safe_float(info.get("market_cap")),
        "enterprise_value": _safe_float(info.get("enterprise_value")),
    }


def _extract_fundamental_inputs(info: dict) -> dict:
    """Extract fundamental ratio inputs from yfinance info."""
    return {
        # Profitability
        "gross_margin": _safe_float(info.get("gross_margins")),
        "operating_margin": _safe_float(info.get("operating_margins")),
        "net_margin": _safe_float(info.get("profit_margins")),
        "ebitda_margin": _safe_float(info.get("ebitda_margins")),
        "roe": _safe_float(info.get("return_on_equity")),
        "roa": _safe_float(info.get("return_on_assets")),
        # Growth
        "revenue_growth": _safe_float(info.get("revenue_growth")),
        "earnings_growth": _safe_float(info.get("earnings_growth")),
        # Balance sheet / cash flow
        "total_debt": _safe_float(info.get("total_debt")),
        "total_cash": _safe_float(info.get("total_cash")),
        "total_revenue": _safe_float(info.get("total_revenue")),
        "operating_cashflow": _safe_float(info.get("operating_cashflow")),
        "free_cashflow": _safe_float(info.get("free_cashflow")),
        "book_value": _safe_float(info.get("book_value")),
        "shares_outstanding": _safe_float(info.get("shares_outstanding")),
        "market_cap": _safe_float(info.get("market_cap")),
        "enterprise_value": _safe_float(info.get("enterprise_value")),
    }


@dataclass
class _ValuationCtx:
    """Carries all pre-computed inputs so _run_* methods stay parameter-light."""
    ticker: str
    model: str
    current_price: float
    shares: float
    roe: float
    earnings_growth: float
    free_cashflow: float
    operating_cashflow: float
    total_cash: float
    total_debt: float
    book_value_total: float
    d0: float
    required_return: float
    wacc: float
    terminal_growth: float
    forecast_years: int
    high_growth_rate: float
    high_growth_years: int


class _EquityValuationDispatch(Tool):
    """Inner tool: deterministic equity valuation dispatch (used by EquityValuationRouter sub-agent).

    Fetches financial data automatically from yfinance and runs the requested model.
    Exposed to the inner LLM as ``run_valuation_model``.
    """

    name = "run_valuation_model"
    description = (
        "Run a single CFA-level equity valuation model for a stock. "
        "Fetches yfinance data automatically. Available models: "
        "dcf_fcff, dcf_fcfe, ddm_gordon, ddm_two_stage, ddm_h_model, ddm_three_stage, "
        "multiples, residual_income, eva, fundamental_ratios."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', '600519.SH')",
            },
            "model": {
                "type": "string",
                "enum": [
                    "dcf_fcff",
                    "dcf_fcfe",
                    "ddm_gordon",
                    "ddm_two_stage",
                    "ddm_h_model",
                    "ddm_three_stage",
                    "multiples",
                    "residual_income",
                    "eva",
                    "fundamental_ratios",
                ],
                "description": (
                    "Valuation model to run. "
                    "dcf_fcff/dcf_fcfe: free cash flow discounting; "
                    "ddm_*: dividend discount models; "
                    "multiples: P/E, P/B, EV/EBITDA from market data; "
                    "residual_income: RI model; eva: Economic Value Added; "
                    "fundamental_ratios: profitability/liquidity/solvency ratios."
                ),
            },
            "wacc": {
                "type": "number",
                "description": (
                    "Weighted Average Cost of Capital as decimal (e.g., 0.10 for 10%). "
                    "Used for FCFF and EVA models. Default: CAPM estimate from beta."
                ),
            },
            "required_return": {
                "type": "number",
                "description": (
                    "Required return on equity as decimal (e.g., 0.12 for 12%). "
                    "Used for FCFE, DDM, and RI models. Default: CAPM estimate from beta."
                ),
            },
            "terminal_growth_rate": {
                "type": "number",
                "description": "Long-term terminal growth rate as decimal (default: 0.025).",
            },
            "high_growth_rate": {
                "type": "number",
                "description": (
                    "High-growth phase rate for two-stage/three-stage models "
                    "(default: earnings_growth from yfinance)."
                ),
            },
            "high_growth_years": {
                "type": "integer",
                "description": "Number of high-growth years for two-stage model (default: 5).",
                "minimum": 1,
                "maximum": 20,
            },
            "forecast_years": {
                "type": "integer",
                "description": "Number of years to project cash flows (default: 5).",
                "minimum": 1,
                "maximum": 15,
            },
        },
        "required": ["ticker", "model"],
    }

    async def execute(self, **kwargs: Any) -> str:
        ticker = kwargs.get("ticker", "UNKNOWN")
        model = kwargs.get("model", "multiples")
        logger.info(f"equity_valuation  ticker={ticker}  model={model}")

        raw_info = await _yfinance.execute(command="info", symbol=ticker)
        info = json.loads(raw_info)

        if "error" in info:
            return json.dumps({"error": info["error"], "ticker": ticker, "model": model})

        current_price = _safe_float(info.get("current_price"))
        if current_price == 0 and _safe_float(info.get("market_cap")) == 0:
            return json.dumps({
                "error": f"No data found for ticker '{ticker}'. Please verify the symbol.",
                "ticker": ticker, "model": model,
            }, ensure_ascii=False)

        # Shared derived values used across models
        shares = _safe_float(info.get("shares_outstanding"))
        beta = _safe_float(info.get("beta"), 1.0)
        roe = _safe_float(info.get("return_on_equity"))
        earnings_growth = _safe_float(info.get("earnings_growth"), 0.08)
        free_cashflow = _safe_float(info.get("free_cashflow"))
        operating_cashflow = _safe_float(info.get("operating_cashflow"))
        total_cash = _safe_float(info.get("total_cash"))
        total_debt = _safe_float(info.get("total_debt"))
        book_value_per_share = _safe_float(info.get("book_value"))
        book_value_total = _safe_float(info.get("book_value_total"), book_value_per_share * shares)
        dividend_yield = _safe_float(info.get("dividend_yield"))
        d0 = dividend_yield * current_price if dividend_yield and current_price else 0.0

        capm = _RISK_FREE_RATE + beta * _EQUITY_RISK_PREMIUM
        required_return = _safe_float(kwargs.get("required_return"), capm)
        wacc = _safe_float(kwargs.get("wacc"), required_return * 0.85)  # simple debt-weighted approx
        terminal_growth = _safe_float(kwargs.get("terminal_growth_rate"), _DEFAULT_TERMINAL_GROWTH)
        forecast_years = int(kwargs.get("forecast_years", _DEFAULT_FORECAST_YEARS))
        high_growth_rate = _safe_float(kwargs.get("high_growth_rate"), max(earnings_growth, 0.05))
        high_growth_years = int(kwargs.get("high_growth_years", 5))

        ctx = _ValuationCtx(
            ticker=ticker, model=model, current_price=current_price, shares=shares,
            roe=roe, earnings_growth=earnings_growth,
            free_cashflow=free_cashflow, operating_cashflow=operating_cashflow,
            total_cash=total_cash, total_debt=total_debt,
            book_value_total=book_value_total, d0=d0,
            required_return=required_return, wacc=wacc,
            terminal_growth=terminal_growth, forecast_years=forecast_years,
            high_growth_rate=high_growth_rate, high_growth_years=high_growth_years,
        )

        result: dict[str, Any] = {
            "ticker": ticker,
            "model": model,
            "current_price": current_price,
            "assumptions": {
                "wacc": round(wacc, 4),
                "required_return": round(required_return, 4),
                "terminal_growth_rate": terminal_growth,
                "forecast_years": forecast_years,
            },
        }

        try:
            if model == "multiples":
                result["valuation"] = _extract_multiples(info)
            elif model == "fundamental_ratios":
                result["valuation"] = self._run_fundamental_ratios(info, book_value_total)
            elif model in ("dcf_fcff", "dcf_fcfe"):
                result.update(self._run_dcf(ctx))
            elif model in ("ddm_gordon", "ddm_two_stage", "ddm_h_model", "ddm_three_stage"):
                ddm_result = self._run_ddm(ctx)
                if "error" in ddm_result:
                    return json.dumps(ddm_result, ensure_ascii=False)
                result.update(ddm_result)
            elif model == "residual_income":
                ri_result = self._run_residual_income(ctx)
                if "error" in ri_result:
                    return json.dumps(ri_result, ensure_ascii=False)
                result.update(ri_result)
            elif model == "eva":
                result.update(self._run_eva(ctx, info))
            else:
                result["error"] = f"Unknown model: {model}"

        except Exception as exc:
            logger.warning(f"equity_valuation error: {exc}")
            result["error"] = str(exc)
            result["note"] = (
                "Model calculation failed — this often means insufficient data is available "
                "from yfinance for the selected model. Try 'multiples' or 'fundamental_ratios' instead."
            )

        return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    @staticmethod
    def _run_fundamental_ratios(info: dict, book_value_total: float) -> dict:
        valuation = _extract_fundamental_inputs(info)
        net_margin = _safe_float(info.get("profit_margins"))
        total_revenue = _safe_float(info.get("total_revenue"))
        total_assets = _safe_float(info.get("total_assets"))
        asset_turnover = total_revenue / total_assets if total_assets else 0
        equity_multiplier = total_assets / book_value_total if book_value_total else 0
        valuation["dupont_roe"] = round(net_margin * asset_turnover * equity_multiplier, 4)
        valuation["asset_turnover"] = round(asset_turnover, 4)
        valuation["equity_multiplier"] = round(equity_multiplier, 4)
        return valuation

    @staticmethod
    def _run_dcf(ctx: "_ValuationCtx") -> dict:
        base_cf = ctx.free_cashflow or ctx.operating_cashflow * 0.7
        result: dict[str, Any] = {}
        if base_cf <= 0:
            result["warning"] = (
                "Free cash flow is non-positive — DCF may be unreliable. "
                "Consider using multiples or residual income model."
            )
            base_cf = abs(base_cf) if base_cf != 0 else 1e6

        growth = ctx.earnings_growth if ctx.earnings_growth > 0 else 0.05
        projections = _project_cash_flows(base_cf, growth, ctx.forecast_years)

        if ctx.model == "dcf_fcff":
            val = FCFFModel().calculate(
                fcff_projections=projections, wacc=ctx.wacc,
                shares_outstanding=ctx.shares, terminal_growth=ctx.terminal_growth,
                cash=ctx.total_cash, total_debt=ctx.total_debt,
                current_price=ctx.current_price,
            )
        else:
            val = FCFEModel().calculate(
                fcfe_projections=projections, required_return=ctx.required_return,
                shares_outstanding=ctx.shares, terminal_growth=ctx.terminal_growth,
                current_price=ctx.current_price,
            )

        result["valuation"] = {
            "intrinsic_value_per_share": round(val.intrinsic_value, 2),
            "current_price": round(val.current_price, 2),
            "upside_downside_pct": round(val.upside_downside * 100, 1),
            "recommendation": val.recommendation,
            "confidence": val.confidence_level,
            "projected_cash_flows": [round(c, 0) for c in projections],
            "enterprise_value": round(val.calculation_details.get("enterprise_value", 0), 0)
            if val.calculation_details else None,
            "terminal_value": round(val.calculation_details.get("terminal_value", 0), 0)
            if val.calculation_details else None,
        }
        return result

    @staticmethod
    def _run_ddm(ctx: "_ValuationCtx") -> dict:
        if ctx.d0 <= 0:
            return {
                "error": (
                    f"{ctx.ticker} does not pay dividends — "
                    f"{ctx.model} requires a dividend-paying stock."
                ),
                "suggestion": "Try 'dcf_fcff' or 'multiples' instead.",
                "ticker": ctx.ticker, "model": ctx.model,
            }

        if ctx.model == "ddm_gordon":
            val = GordonGrowthModel().calculate_intrinsic_value(
                dividend=ctx.d0, growth_rate=ctx.terminal_growth,
                required_return=ctx.required_return, is_d0=True,
            )
            upside = (val.intrinsic_value - ctx.current_price) / ctx.current_price if ctx.current_price else 0
            return {"valuation": {
                "intrinsic_value": round(val.intrinsic_value, 2),
                "current_price": ctx.current_price,
                "upside_downside_pct": round(upside * 100, 1),
                "recommendation": _recommend(upside),
                "d0": round(ctx.d0, 4),
                "d1": round(val.assumptions.get("d1", 0), 4),
                "assumptions": val.assumptions,
            }}

        if ctx.model == "ddm_two_stage":
            val = TwoStageDDM().calculate_intrinsic_value(
                d0=ctx.d0, high_growth_rate=ctx.high_growth_rate,
                stable_growth_rate=ctx.terminal_growth, required_return=ctx.required_return,
                high_growth_years=ctx.high_growth_years,
            )
        elif ctx.model == "ddm_h_model":
            val = HModelDDM().calculate_intrinsic_value(
                d0=ctx.d0, short_term_growth=ctx.high_growth_rate,
                long_term_growth=ctx.terminal_growth, required_return=ctx.required_return,
                high_growth_period=ctx.high_growth_years * 2,
            )
        else:  # ddm_three_stage
            val = ThreeStageDDM().calculate_intrinsic_value(
                d0=ctx.d0, growth_stage1=ctx.high_growth_rate,
                growth_stage3=ctx.terminal_growth, required_return=ctx.required_return,
                years_stage1=ctx.high_growth_years,
                years_stage2=max(ctx.high_growth_years // 2, 2),
            )

        upside = (val.intrinsic_value - ctx.current_price) / ctx.current_price if ctx.current_price else 0
        return {"valuation": {
            "intrinsic_value": round(val.intrinsic_value, 2),
            "current_price": ctx.current_price,
            "upside_downside_pct": round(upside * 100, 1),
            "recommendation": _recommend(upside),
            "assumptions": val.assumptions,
        }}

    @staticmethod
    def _run_residual_income(ctx: "_ValuationCtx") -> dict:
        if ctx.book_value_total <= 0 or ctx.shares <= 0:
            return {
                "error": "Insufficient book value data for Residual Income model.",
                "ticker": ctx.ticker,
            }

        ri_model = ResidualIncomeModel()
        projected_roes = [max(ctx.roe, ctx.required_return * 0.5)] * ctx.forecast_years
        projected_ris = []
        bv = ctx.book_value_total

        for proj_roe in projected_roes:
            net_income = bv * proj_roe
            projected_ris.append(ri_model.calculate_residual_income(net_income, bv, ctx.required_return))
            bv *= (1 + proj_roe * 0.6)   # assume 40% payout ratio

        val = ri_model.calculate(
            current_book_value=ctx.book_value_total, projected_ris=projected_ris,
            required_return=ctx.required_return, shares_outstanding=ctx.shares,
            terminal_ri=projected_ris[-1] if projected_ris else 0,
            terminal_growth=ctx.terminal_growth, current_price=ctx.current_price,
        )
        return {"valuation": {
            "intrinsic_value_per_share": round(val.intrinsic_value, 2),
            "current_price": round(val.current_price, 2),
            "upside_downside_pct": round(val.upside_downside * 100, 1),
            "recommendation": val.recommendation,
            "book_value_per_share": round(ctx.book_value_total / ctx.shares, 2) if ctx.shares else 0,
            "roe": round(ctx.roe, 4),
            "justified_pb": round(
                1 + (ctx.roe - ctx.required_return) / (ctx.required_return - ctx.terminal_growth), 3
            ) if ctx.required_return > ctx.terminal_growth else None,
        }}

    @staticmethod
    def _run_eva(ctx: "_ValuationCtx", info: dict) -> dict:
        market_cap = _safe_float(info.get("market_cap"))
        invested_capital = market_cap + ctx.total_debt - ctx.total_cash
        # Approximate NOPAT: use 21% US statutory rate when effective rate unavailable
        nopat = ctx.operating_cashflow * 0.79 if ctx.operating_cashflow else 0

        eva_model = EconomicValueAddedModel()
        eva_value = eva_model.calculate_eva(nopat, invested_capital, ctx.wacc)
        projected_evas = [eva_value * (0.5 ** y) for y in range(ctx.forecast_years)]
        firm_val = eva_model.eva_valuation(
            current_invested_capital=invested_capital, projected_evas=projected_evas,
            wacc=ctx.wacc, terminal_eva=projected_evas[-1],
            terminal_growth=ctx.terminal_growth,
        )
        firm_value_per_share = (
            (firm_val["total_firm_value"] - ctx.total_debt + ctx.total_cash) / ctx.shares
            if ctx.shares else 0
        )
        upside = (firm_value_per_share - ctx.current_price) / ctx.current_price if ctx.current_price else 0
        return {"valuation": {
            "current_eva": round(eva_value, 0),
            "nopat": round(nopat, 0),
            "invested_capital": round(invested_capital, 0),
            "total_firm_value": round(firm_val["total_firm_value"], 0),
            "equity_value_per_share": round(firm_value_per_share, 2),
            "current_price": ctx.current_price,
            "upside_downside_pct": round(upside * 100, 1),
            "recommendation": _recommend(upside),
            "interpretation": (
                "Positive EVA indicates value creation above cost of capital."
                if eva_value > 0 else
                "Negative EVA indicates value destruction — returns below cost of capital."
            ),
        }}


# ---------------------------------------------------------------------------
# Public sub-agent wrapper (Dexter pattern)
# ---------------------------------------------------------------------------

class EquityValuationRouter(LLMRouterTool):
    """CFA-level equity valuation sub-agent.

    Accepts a natural-language query; the inner LLM decides which model(s) to run
    via ``run_valuation_model`` and can call them in parallel for a comprehensive view.
    """

    name = "equity_valuation"
    description = (
        "Run CFA-level equity valuation models: DCF (FCFF/FCFE), "
        "Dividend Discount Models (Gordon, Two-Stage, H-Model, Three-Stage), "
        "Multiples analysis, Residual Income, EVA, and Fundamental Ratio analysis. "
        "Fetches financial data automatically. Describe the analysis you need in plain language."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of the valuation to run. "
                    "Examples: 'Run DCF FCFF analysis on AAPL with 10% WACC', "
                    "'Comprehensive valuation of MSFT including multiples and fundamentals', "
                    "'DDM Gordon Growth for JNJ', "
                    "'Compare AAPL across DCF, multiples, and fundamental ratios'."
                ),
            }
        },
        "required": ["query"],
    }

    _inner_system_prompt = (
        "You are a CFA-level equity valuation specialist.\n\n"
        "Use run_valuation_model to execute valuation models. Financial data is fetched automatically.\n\n"
        "Model selection guide:\n"
        "  Tech / growth stocks  → dcf_fcff + multiples + fundamental_ratios\n"
        "  Dividend-paying stocks → also add ddm_gordon or ddm_two_stage\n"
        "  Quick overview        → multiples + fundamental_ratios\n"
        "  Comprehensive         → run all applicable models and compare\n\n"
        "Call multiple models in parallel for comprehensive analysis.\n"
        "Summarise the key valuation findings in 2-4 sentences. Raw data is preserved separately."
    )

    def _build_inner_tools(self) -> list[Tool]:
        return [_EquityValuationDispatch()]

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        logger.info(f"equity_valuation (inner-LLM): {query[:120]}")
        return await self._run_inner_agent(query)
