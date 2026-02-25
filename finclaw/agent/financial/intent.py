"""Financial intent detection using rule-based matching."""

import re
from dataclasses import dataclass, field


@dataclass
class FinancialIntent:
    """Result of financial intent detection."""

    is_financial: bool = False
    intent_type: str = "unknown"  # price_query / financial_analysis / market_search / macro_analysis / meme / unknown
    tickers: list[str] = field(default_factory=list)
    confidence: float = 0.0


class FinancialIntentDetector:
    """Rule-driven financial intent detector.

    Covers ~90% of cases via keyword + ticker pattern matching.
    When uncertain, does NOT classify as financial (saves tokens).
    """

    TICKER_PATTERN = re.compile(
        r"\$([A-Z]{1,5})\b"                    # $AAPL
        r"|(?<!\w)([A-Z]{2,5})(?=\s|$|[,.])"   # standalone AAPL, TSLA
        r"|(\d{6}\.(SH|SZ|SS))"                 # A-shares: 600519.SH
        r"|([A-Z]{2,10}[-/](USD|USDT|BTC))",    # crypto: BTC-USD
        re.MULTILINE,
    )

    # Common English words that look like tickers but aren't
    TICKER_BLACKLIST = {
        "THE", "FOR", "AND", "NOT", "ARE", "BUT", "HAS", "WAS",
        "CAN", "HIS", "HER", "ALL", "ITS", "YOU", "WHO", "HOW",
        "NEW", "OLD", "NOW", "MAY", "USE", "SET", "GET", "PUT",
        "SAY", "ASK", "RUN", "LET", "TRY", "ADD", "END", "TOP",
        "TWO", "ONE", "CEO", "CFO", "CTO", "PDF", "API", "URL",
        "USD", "EUR", "GBP", "JPY", "CNY",
    }

    FINANCIAL_KEYWORDS = {
        # price / market
        "stock", "share", "price", "quote", "market cap", "volume",
        "ticker", "trading", "rally", "dip", "bull", "bear",
        # fundamentals
        "revenue", "earnings", "profit", "margin", "EPS",
        "P/E", "PE ratio", "ROE", "ROA", "EBITDA",
        "balance sheet", "income statement", "cash flow",
        "dividend", "yield", "book value", "debt",
        # equity analysis
        "valuation", "fundamental", "technical", "DCF",
        "analyst", "estimate", "forecast", "guidance",
        "SEC filing", "10-K", "10-Q", "8-K", "annual report",
        # macro / economics
        "GDP", "inflation", "CPI", "PCE", "deflation",
        "unemployment", "employment", "labor market", "payrolls",
        "monetary policy", "fiscal policy", "central bank", "fed", "federal reserve",
        "interest rate", "fed funds", "FOMC", "quantitative easing", "QE",
        "Taylor rule", "yield curve", "treasury", "bond",
        "business cycle", "recession", "expansion", "contraction",
        "FRED", "macro", "macroeconomic", "macroeconomics",
        "exchange rate", "FX", "forex", "currency", "carry trade",
        "PPP", "purchasing power parity", "triangular arbitrage",
        "balance of payments", "BOP", "current account", "trade balance",
        "comparative advantage", "trade war", "tariff", "sanctions",
        "geopolitical risk", "supply chain",
        "M2", "money supply", "credit cycle", "credit spread",
        "leading indicator", "coincident indicator", "lagging indicator",
        "industrial production", "retail sales", "housing starts",
        "consumer sentiment", "purchasing managers", "PMI",
        "ARIMA", "cointegration", "stationarity", "ADF test",
        "Monte Carlo", "stress test", "scenario analysis",
        # Chinese macro (GDP/CPI already in English section above)
        "通胀", "通货膨胀", "利率", "货币政策",
        "财政政策", "汇率", "外汇", "贸易", "贸易战",
        "经济周期", "衰退", "美联储", "央行", "宏观经济",
        # meme coin / crypto culture (detected as financial AND routed via MEME_KEYWORDS)
        "meme coin", "memecoin", "meme token", "degen", "solana meme",
        "dexscreener", "dex screener", "coingecko", "coin gecko",
        "degen coin", "degen token", "shitcoin",
        # launch-intent keywords (also in MEME_KEYWORDS below)
        "pump.fun", "four.meme", "fourmeme", "first minter",
        "launch a coin", "launch coin", "launch a token", "launch token",
        "launch a meme", "launch meme",
        "mint a coin", "mint coin", "mint a token", "mint token",
        "create a coin", "create coin", "create a token", "create token",
        "trending tweet", "viral tweet", "meme potential", "meme word",
        "ticker suggestion", "meme idea",
        "token name", "token names", "coin name", "coin names",
        "coin idea", "coin ideas", "token idea", "token ideas",
        # RSS / multi-source meme monitoring
        "rss feed", "rsshub", "reddit wsb", "wallstreetbets",
        "truth social", "tiktok memecoin", "tiktok meme",
        "check all sources", "trending reddit", "trending tiktok",
    }

    MACRO_KEYWORDS = {
        "GDP", "inflation", "CPI", "PCE", "unemployment", "employment",
        "monetary policy", "fiscal policy", "central bank", "fed", "federal reserve",
        "interest rate", "fed funds", "FOMC", "QE", "Taylor rule",
        "yield curve", "business cycle", "recession", "macro", "macroeconomic",
        "exchange rate", "FX", "forex", "currency", "carry trade", "PPP",
        "balance of payments", "BOP", "trade balance", "tariff", "geopolitical",
        "M2", "money supply", "credit cycle", "FRED", "leading indicator",
        "industrial production", "PMI", "ARIMA", "cointegration", "Monte Carlo",
        "economic calendar", "data release", "released today", "scheduled",
        "coming out", "NFP", "nonfarm", "non-farm", "PPI", "JOLTS",
        "通胀", "利率", "货币政策", "汇率", "宏观经济", "经济周期",
    }

    # Triggers the meme router — covers ALL meme coin queries (data + social + launch).
    MEME_KEYWORDS = {
        # general meme coin terms (price, info, trending all route here)
        "meme coin", "memecoin", "meme token", "degen",
        "meme coin price", "memecoin price", "meme token price",
        "dexscreener", "dex screener", "coingecko", "coin gecko",
        "meme coin trending", "trending meme",
        "solana meme", "sol meme", "base meme",
        "degen coin", "degen token", "shitcoin",
        # explicit launch / create intent (various phrasings)
        "pump.fun", "four.meme", "fourmeme", "first minter",
        "launch a coin", "launch coin", "launch a token", "launch token",
        "launch a meme", "launch meme",
        "mint a coin", "mint coin", "mint a token", "mint token",
        "create a coin", "create coin", "create a token", "create token",
        # name / idea generation
        "token name", "token names", "coin name", "coin names",
        "coin idea", "coin ideas", "token idea", "token ideas",
        "meme idea", "meme ideas", "ticker suggestion",
        # social media scanning for launch candidates
        "trending tweet", "viral tweet", "meme potential", "meme word",
        "check all sources", "trending reddit", "trending tiktok",
        # RSS / multi-source monitoring
        "rss feed", "rsshub", "reddit wsb", "wallstreetbets",
        "truth social", "tiktok memecoin", "tiktok meme",
    }

    PRICE_KEYWORDS = {
        "price", "quote", "how much", "current",
        "股价", "价格", "多少钱", "现在", "实时", "行情",
    }

    ANALYSIS_KEYWORDS = {
        "analyze", "analysis", "report", "filing", "10-K", "10-Q",
        "earnings", "financial statement", "valuation", "DCF",
        "分析", "财报", "年报", "季报", "估值",
    }

    def detect(self, query: str) -> FinancialIntent:
        """Detect financial intent from user query."""
        query_lower = query.lower()
        intent = FinancialIntent()

        tickers = self._extract_tickers(query)
        matched = [kw for kw in self.FINANCIAL_KEYWORDS if kw.lower() in query_lower]

        if not tickers and not matched:
            return intent

        intent.is_financial = True
        intent.tickers = tickers
        intent.confidence = min(1.0, len(tickers) * 0.4 + len(matched) * 0.2)

        if any(kw.lower() in query_lower for kw in self.MEME_KEYWORDS):
            intent.intent_type = "meme"
        elif any(kw.lower() in query_lower for kw in self.MACRO_KEYWORDS):
            intent.intent_type = "macro_analysis"
        elif any(kw.lower() in query_lower for kw in self.PRICE_KEYWORDS):
            intent.intent_type = "price_query"
        elif any(kw.lower() in query_lower for kw in self.ANALYSIS_KEYWORDS):
            intent.intent_type = "financial_analysis"
        elif tickers:
            intent.intent_type = "market_search"

        return intent

    def _extract_tickers(self, query: str) -> list[str]:
        """Extract stock/crypto ticker symbols from query."""
        tickers = []
        for match in self.TICKER_PATTERN.finditer(query):
            raw = match.group(0).lstrip("$").upper().strip(".,")
            if len(raw) >= 2 and raw not in self.TICKER_BLACKLIST:
                tickers.append(raw)
        return list(dict.fromkeys(tickers))  # deduplicate, preserve order
