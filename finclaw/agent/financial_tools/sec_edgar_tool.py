"""SEC EDGAR Tool — fetch and parse 10-Q/10-K filings from SEC EDGAR.

Commands:
  daily_filings   - List all 10-Q/10-K filings for a given date
  fetch_and_parse - Fetch and parse key financial facts for a specific filing URL
  ticker_filings  - Get recent filings for a specific ticker (via EDGAR submissions API)
  daily_parsed    - Full pipeline: detect → filter by market cap → download → parse key facts
"""

from __future__ import annotations

import json
import os
import re
import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from finclaw.agent.tools.base import Tool

USER_AGENT = os.environ.get("SEC_USER_AGENT", "FinClaw research@finclaw.ai")
_SLEEP_SEC = 0.12
_FORMS_CANDIDATE = {"10-Q", "10-K"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _sec_get(url: str):
    import requests
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    }
    time.sleep(_SLEEP_SEC)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EarningsHit:
    form: str
    cik: str
    company_name: str
    filing_date: str
    filing_url: str


# ---------------------------------------------------------------------------
# SEC daily index helpers
# ---------------------------------------------------------------------------

def _quarter_of_date(d: dt.date) -> int:
    return (d.month - 1) // 3 + 1


def _idx_url(d: dt.date) -> str:
    yyyy = d.year
    qtr = _quarter_of_date(d)
    yyyymmdd = d.strftime("%Y%m%d")
    return f"https://www.sec.gov/Archives/edgar/daily-index/{yyyy}/QTR{qtr}/form.{yyyymmdd}.idx"


def _parse_form_idx(text: str) -> list[EarningsHit]:
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Form Type"):
            start = i + 2
            break

    hits: list[EarningsHit] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        form = parts[0].strip()
        if form not in _FORMS_CANDIDATE:
            continue
        filename = parts[-1].strip()
        date_filed_raw = parts[-2].strip()
        if re.fullmatch(r"\d{8}", date_filed_raw):
            date_filed = f"{date_filed_raw[:4]}-{date_filed_raw[4:6]}-{date_filed_raw[6:]}"
        else:
            date_filed = date_filed_raw
        cik = parts[-3].strip()
        company_name = " ".join(parts[1:-3]).strip()
        filing_url = f"https://www.sec.gov/Archives/{filename}"
        hits.append(EarningsHit(form=form, cik=cik, company_name=company_name,
                                filing_date=date_filed, filing_url=filing_url))
    return hits


def _detect_daily_filings(date_: dt.date) -> list[EarningsHit]:
    import requests
    url = _idx_url(date_)
    try:
        resp = _sec_get(url)
        return _parse_form_idx(resp.text)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (403, 404):
            return []
        raise


# ---------------------------------------------------------------------------
# CIK ↔ ticker helpers (uses EDGAR company_tickers.json — downloaded on demand)
# ---------------------------------------------------------------------------

_TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
_tickers_cache: dict | None = None


def _load_tickers_json() -> dict:
    global _tickers_cache
    if _tickers_cache is not None:
        return _tickers_cache
    resp = _sec_get(_TICKERS_JSON_URL)
    _tickers_cache = resp.json()
    return _tickers_cache


def _cik_to_ticker(cik: str) -> str | None:
    data = _load_tickers_json()
    cik_norm = str(int(cik))
    for rec in data.values():
        if str(rec.get("cik_str")) == cik_norm:
            return rec.get("ticker")
    return None


def _ticker_to_cik(ticker: str) -> str | None:
    data = _load_tickers_json()
    ticker_upper = ticker.upper()
    for rec in data.values():
        if (rec.get("ticker") or "").upper() == ticker_upper:
            return str(rec["cik_str"]).zfill(10)
    return None


# ---------------------------------------------------------------------------
# Market cap helpers (via yfinance, already a dependency)
# ---------------------------------------------------------------------------

def _fetch_market_caps(tickers: list[str]) -> dict[str, int | None]:
    import yfinance as yf
    result: dict[str, int | None] = {}
    if not tickers:
        return result
    symbols = " ".join(tickers)
    tk = yf.Tickers(symbols)
    for t in tickers:
        try:
            v = tk.tickers[t].info.get("marketCap")
            result[t] = int(v) if isinstance(v, (int, float)) and v > 0 else None
        except Exception:
            result[t] = None
        time.sleep(0.05)
    return result


# ---------------------------------------------------------------------------
# Filing download + iXBRL parse helpers
# ---------------------------------------------------------------------------

def _parse_cik_acc(filing_url: str) -> tuple[str, str] | None:
    m = re.search(r"/edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})\.txt$", filing_url)
    if not m:
        return None
    return m.group(1), m.group(2).replace("-", "")


def _filing_dir_url(filing_url: str) -> str | None:
    parsed = _parse_cik_acc(filing_url)
    if not parsed:
        return None
    cik_num, acc = parsed
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}/"


def _pick_primary_html(index_json: dict) -> str | None:
    items = index_json.get("directory", {}).get("item", [])
    htmls = []
    for it in items:
        name = str(it.get("name", ""))
        size = int(it.get("size", 0) or 0)
        if name.upper().endswith((".HTM", ".HTML")) and "INDEX" not in name.upper():
            htmls.append((size, name))
    if not htmls:
        return None
    htmls.sort(reverse=True)
    return htmls[0][1]


def _download_filing_html(filing_url: str) -> str | None:
    # Case 1: direct .htm/.html URL (e.g. from ticker_filings primaryDocument)
    url_lower = filing_url.lower()
    if url_lower.endswith((".htm", ".html")):
        try:
            return _sec_get(filing_url).text
        except Exception:
            return None

    # Case 2: .txt index URL — resolve directory and pick the largest HTML file
    dir_url = _filing_dir_url(filing_url)
    if not dir_url:
        # Last-resort: try fetching the URL directly
        try:
            return _sec_get(filing_url).text
        except Exception:
            return None
    try:
        idx = _sec_get(dir_url + "index.json").json()
        fname = _pick_primary_html(idx)
        if not fname:
            return None
        return _sec_get(dir_url + fname).text
    except Exception:
        try:
            return _sec_get(filing_url).text
        except Exception:
            return None


# iXBRL concept lists
_REVENUE_CONCEPTS = [
    "us-gaap:Revenues",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenuesNetOfInterestExpense",
    "ifrs-full:Revenue",
]
_GROSS_PROFIT_CONCEPTS = ["us-gaap:GrossProfit", "ifrs-full:GrossProfit"]
_OPERATING_INCOME_CONCEPTS = [
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    "ifrs-full:OperatingProfitLoss",
]
_NETINCOME_CONCEPTS = [
    "us-gaap:NetIncomeLoss",
    "us-gaap:NetIncomeLossAttributableToParent",
    "us-gaap:ProfitLoss",
    "ifrs-full:ProfitLoss",
]
_EPS_BASIC_CONCEPTS = ["us-gaap:EarningsPerShareBasic", "ifrs-full:BasicEarningsLossPerShare"]
_EPS_DILUTED_CONCEPTS = ["us-gaap:EarningsPerShareDiluted", "ifrs-full:DilutedEarningsLossPerShare"]
_CASH_CONCEPTS = [
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "ifrs-full:CashAndCashEquivalents",
]
_ASSETS_CONCEPTS = ["us-gaap:Assets", "ifrs-full:Assets"]
_LIAB_CONCEPTS = ["us-gaap:Liabilities", "ifrs-full:Liabilities"]
_EQUITY_CONCEPTS = [
    "us-gaap:StockholdersEquity",
    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "ifrs-full:Equity",
]
_OCF_CONCEPTS = [
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "ifrs-full:CashFlowsFromUsedInOperatingActivities",
]
_CAPEX_CONCEPTS = [
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsToAcquireProductiveAssets",
    "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
]
_TICKER_CONCEPTS = ["dei:TradingSymbol"]
_COMPANY_CONCEPTS = ["dei:EntityRegistrantName"]
_FORM_CONCEPTS = ["dei:DocumentType"]
_PERIOD_CONCEPTS = ["dei:DocumentPeriodEndDate"]
_SHARES_CONCEPTS = ["dei:EntityCommonStockSharesOutstanding"]


def _parse_ix_numeric(tag) -> float | None:
    raw = tag.get_text(" ", strip=True)
    if not raw:
        return None
    raw = raw.replace("—", "").replace("–", "").strip()
    neg = raw.startswith("(") and raw.endswith(")")
    s = re.sub(r"[\$,]", "", raw.strip("()")).strip()
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if neg:
        v = -v
    scale = tag.get("scale")
    if scale is not None:
        try:
            v *= 10 ** int(scale)
        except Exception:
            pass
    return v


def _find_ix_value(soup, concepts: list[str], numeric: bool):
    for nm in concepts:
        tags = soup.find_all(
            lambda t, _nm=nm: (
                hasattr(t, "name") and t.name
                and t.name.lower().endswith(("nonfraction", "nonnumeric"))
                and t.get("name") == _nm
            )
        )
        for tag in tags:
            v = _parse_ix_numeric(tag) if numeric else re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
            if v is None or v == "":
                continue
            return v
    return None


def _parse_html_key_facts(html: str) -> dict[str, Any]:
    from bs4 import BeautifulSoup
    is_ixbrl = html.lstrip().lower()[:50].startswith("<?xml") or "ix:" in html[:2000].lower()
    soup = BeautifulSoup(html, "lxml-xml" if is_ixbrl else "lxml")

    return {
        "is_ixbrl": is_ixbrl,
        "ticker": _find_ix_value(soup, _TICKER_CONCEPTS, numeric=False),
        "company_name": _find_ix_value(soup, _COMPANY_CONCEPTS, numeric=False),
        "form_type": _find_ix_value(soup, _FORM_CONCEPTS, numeric=False),
        "period_end": _find_ix_value(soup, _PERIOD_CONCEPTS, numeric=False),
        "shares_outstanding": _find_ix_value(soup, _SHARES_CONCEPTS, numeric=True),
        "revenue": _find_ix_value(soup, _REVENUE_CONCEPTS, numeric=True),
        "gross_profit": _find_ix_value(soup, _GROSS_PROFIT_CONCEPTS, numeric=True),
        "operating_income": _find_ix_value(soup, _OPERATING_INCOME_CONCEPTS, numeric=True),
        "net_income": _find_ix_value(soup, _NETINCOME_CONCEPTS, numeric=True),
        "eps_basic": _find_ix_value(soup, _EPS_BASIC_CONCEPTS, numeric=True),
        "eps_diluted": _find_ix_value(soup, _EPS_DILUTED_CONCEPTS, numeric=True),
        "cash_and_equivalents": _find_ix_value(soup, _CASH_CONCEPTS, numeric=True),
        "total_assets": _find_ix_value(soup, _ASSETS_CONCEPTS, numeric=True),
        "total_liabilities": _find_ix_value(soup, _LIAB_CONCEPTS, numeric=True),
        "equity": _find_ix_value(soup, _EQUITY_CONCEPTS, numeric=True),
        "operating_cash_flow": _find_ix_value(soup, _OCF_CONCEPTS, numeric=True),
        "capex": _find_ix_value(soup, _CAPEX_CONCEPTS, numeric=True),
    }


# ---------------------------------------------------------------------------
# Text section extraction (MD&A, Risk Factors, etc.)
# ---------------------------------------------------------------------------

# Sections to extract: (name, pattern). Deduplicated by name; first in list wins.
# 10-K patterns listed before their 10-Q equivalents so annual filings take priority.
_EXTRACT_SECTIONS: list[tuple[str, re.Pattern]] = [
    ("business_overview", re.compile(r"item\s+1[\.\s]*business\b", re.I)),
    ("risk_factors",      re.compile(r"item\s+1a[\.\s]*risk\s+factors", re.I)),
    ("mda",               re.compile(r"item\s+7[\.\s]*management.{0,20}discussion", re.I)),
    ("market_risk",       re.compile(r"item\s+7a[\.\s]*quantitative", re.I)),
    # 10-Q variants (only used if the 10-K pattern above found nothing)
    ("mda",               re.compile(r"item\s+2[\.\s]*management.{0,20}discussion", re.I)),
    ("market_risk",       re.compile(r"item\s+3[\.\s]*quantitative", re.I)),
]

# Generic item-heading detector used ONLY for section boundary detection.
# ^ + MULTILINE ensures we only match headings at the start of a line,
# not in-text references like "see Item 7 for details".
_ANY_ITEM_HEADING = re.compile(r"^item\s+\d{1,2}[a-c]?[\.\s]", re.I | re.MULTILINE)

# Max chars per section. 15 K captures full Risk Factors and most MD&A sections.
_SECTION_MAX_CHARS = 15_000


def _clean_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    # Remove script/style noise
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines and trim lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _extract_text_sections(html: str) -> dict[str, str]:
    """
    Extract key narrative sections from a 10-K/10-Q filing.

    Three improvements over the naive slice approach:
    1. Uses the LAST occurrence of each item heading — earlier occurrences
       are table-of-contents entries; the last one is the actual section body.
    2. Detects the next item heading to find the true section end boundary,
       so we capture the full section rather than an arbitrary char slice.
    3. Returns up to _SECTION_MAX_CHARS (15 K) with sentence-boundary truncation.
    """
    text = _clean_text(html)
    found: dict[str, str] = {}

    for section_name, pattern in _EXTRACT_SECTIONS:
        if section_name in found:
            continue

        candidates = list(pattern.finditer(text))
        if not candidates:
            continue

        # Last match = actual section content; earlier matches are TOC entries.
        m = candidates[-1]
        heading_end = m.end()

        # Skip the rest of the heading line so content starts on the next line.
        nl = text.find("\n", heading_end)
        content_start = (nl + 1) if 0 <= nl - heading_end <= 200 else heading_end

        # Find the next item heading to establish the section end boundary.
        next_heading = _ANY_ITEM_HEADING.search(text, heading_end + 10)
        section_end = next_heading.start() if next_heading else len(text)

        content_end = min(content_start + _SECTION_MAX_CHARS, section_end)
        content = text[content_start:content_end].strip()

        # Truncate at last sentence boundary when we hit the char cap.
        if content_end - content_start >= _SECTION_MAX_CHARS:
            last_period = content.rfind(". ")
            if last_period > _SECTION_MAX_CHARS // 2:
                content = content[: last_period + 1]

        if content:
            found[section_name] = content

    return found


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class SecEdgarTool(Tool):
    """Fetch and parse SEC EDGAR 10-Q/10-K filings.

    Commands
    --------
    daily_filings   : List all 10-Q/10-K hits for a date (no download).
    fetch_and_parse : Download + parse key financial facts from a single filing URL.
    ticker_filings  : List recent 10-Q/10-K filings for a ticker (EDGAR submissions API).
    daily_parsed    : Full pipeline — detect filings, filter top-N by market cap, parse facts.
    """

    name = "sec_edgar"
    description = (
        "Fetch SEC EDGAR 10-Q/10-K filings. "
        "Use 'daily_filings' to list filings for a date; "
        "'ticker_filings' to get recent filings for a specific ticker; "
        "'fetch_and_parse' to download and extract key financial facts from a filing URL; "
        "'daily_parsed' to run the full pipeline (detect → filter by market cap → parse)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["daily_filings", "fetch_and_parse", "ticker_filings", "daily_parsed"],
                "description": "Operation to perform.",
            },
            "date": {
                "type": "string",
                "description": "Filing date in YYYY-MM-DD format. Defaults to yesterday.",
            },
            "ticker": {
                "type": "string",
                "description": "Stock ticker (e.g. 'AAPL'). Used by ticker_filings.",
            },
            "filing_url": {
                "type": "string",
                "description": "Direct filing URL from EDGAR Archives. Used by fetch_and_parse.",
            },
            "top_n": {
                "type": "integer",
                "description": "Max companies to return in daily_parsed (0 = no limit). Default: 10.",
                "minimum": 0,
                "maximum": 100,
            },
            "min_mcap": {
                "type": "integer",
                "description": "Minimum market cap filter in USD for daily_parsed (e.g. 1000000000 for $1B).",
            },
            "include_text": {
                "type": "boolean",
                "description": (
                    "If true, fetch_and_parse also returns key narrative sections: "
                    "MD&A, Risk Factors, Market Risk (up to 4000 chars each). "
                    "Use this for qualitative analysis of management commentary and risks."
                ),
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command")
        try:
            if command == "daily_filings":
                return await self._daily_filings(kwargs)
            elif command == "fetch_and_parse":
                return await self._fetch_and_parse(kwargs)
            elif command == "ticker_filings":
                return await self._ticker_filings(kwargs)
            elif command == "daily_parsed":
                return await self._daily_parsed(kwargs)
            else:
                return json.dumps({"error": f"Unknown command: {command!r}"})
        except Exception as exc:
            logger.warning(f"sec_edgar command={command} error: {exc}")
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------

    async def _daily_filings(self, kwargs: dict) -> str:
        date_ = _resolve_date(kwargs.get("date"))
        logger.info(f"sec_edgar daily_filings date={date_}")
        hits = _detect_daily_filings(date_)
        result = {
            "date": str(date_),
            "total_filings": len(hits),
            "filings": [
                {
                    "form": h.form,
                    "cik": h.cik,
                    "company_name": h.company_name,
                    "filing_date": h.filing_date,
                    "filing_url": h.filing_url,
                }
                for h in hits
            ],
        }
        return json.dumps(result, ensure_ascii=False)

    async def _fetch_and_parse(self, kwargs: dict) -> str:
        filing_url = kwargs.get("filing_url", "")
        include_text = kwargs.get("include_text", False)
        if not filing_url:
            return json.dumps({"error": "filing_url is required for fetch_and_parse"})
        logger.info(f"sec_edgar fetch_and_parse url={filing_url[:80]} include_text={include_text}")
        html = _download_filing_html(filing_url)
        if not html:
            return json.dumps({"error": "Failed to download filing HTML", "filing_url": filing_url})
        facts = _parse_html_key_facts(html)
        facts["filing_url"] = filing_url
        if include_text:
            sections = _extract_text_sections(html)
            if sections:
                facts["text_sections"] = sections
                logger.info(f"sec_edgar extracted text sections: {list(sections.keys())}")
            else:
                facts["text_sections"] = {}
                logger.info("sec_edgar no text sections found")
        return json.dumps(facts, ensure_ascii=False, default=str)

    async def _ticker_filings(self, kwargs: dict) -> str:
        ticker = (kwargs.get("ticker") or "").upper()
        if not ticker:
            return json.dumps({"error": "ticker is required for ticker_filings"})
        logger.info(f"sec_edgar ticker_filings ticker={ticker}")

        cik = _ticker_to_cik(ticker)
        if not cik:
            return json.dumps({"error": f"CIK not found for ticker {ticker!r}"})

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            data = _sec_get(url).json()
        except Exception as e:
            return json.dumps({"error": f"EDGAR submissions fetch failed: {e}"})

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for form, date_str, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form not in _FORMS_CANDIDATE:
                continue
            acc_no_dashes = acc.replace("-", "")
            cik_int = str(int(cik))
            url_txt = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{doc}"
            filings.append({"form": form, "filing_date": date_str, "filing_url": url_txt, "accession": acc})
            if len(filings) >= 10:
                break

        return json.dumps({
            "ticker": ticker,
            "cik": cik,
            "company_name": data.get("name", ""),
            "filings": filings,
        }, ensure_ascii=False)

    async def _daily_parsed(self, kwargs: dict) -> str:
        date_ = _resolve_date(kwargs.get("date"))
        top_n = int(kwargs.get("top_n", 10))
        min_mcap = kwargs.get("min_mcap")
        logger.info(f"sec_edgar daily_parsed date={date_} top_n={top_n} min_mcap={min_mcap}")

        # 1. detect filings
        hits = _detect_daily_filings(date_)
        if not hits:
            is_weekend = date_.weekday() >= 5
            note = "weekend — SEC daily index typically empty" if is_weekend else "no 10-Q/10-K filings found"
            return json.dumps({"date": str(date_), "note": note, "items": []}, ensure_ascii=False)

        # 2. map CIK → ticker
        cik_to_ticker: dict[str, str] = {}
        for h in hits:
            t = _cik_to_ticker(h.cik)
            if t:
                cik_to_ticker[h.cik] = t.upper()

        tickers = sorted(set(cik_to_ticker.values()))

        # 3. filter by market cap / top-N
        if min_mcap is not None or top_n > 0:
            mcaps = _fetch_market_caps(tickers)
            # apply min_mcap
            if min_mcap is not None:
                tickers = [t for t in tickers if (mcaps.get(t) or 0) >= int(min_mcap)]
            # apply top_n
            if top_n > 0:
                tickers = sorted(tickers, key=lambda t: mcaps.get(t) or 0, reverse=True)[:top_n]

        selected = set(tickers)
        selected_hits = [h for h in hits if cik_to_ticker.get(h.cik) in selected]

        if not selected_hits:
            return json.dumps({
                "date": str(date_),
                "note": "no filings matched the selected tickers",
                "items": [],
            }, ensure_ascii=False)

        # 4. download + parse
        items = []
        for h in selected_hits:
            ticker = cik_to_ticker.get(h.cik, "")
            html = _download_filing_html(h.filing_url)
            if not html:
                items.append({"ticker": ticker, "form": h.form, "filing_url": h.filing_url, "error": "download failed"})
                continue
            facts = _parse_html_key_facts(html)
            items.append({
                "ticker": ticker,
                "form": h.form,
                "filing_url": h.filing_url,
                **facts,
            })

        return json.dumps({
            "date": str(date_),
            "total_selected": len(selected_hits),
            "items": items,
        }, ensure_ascii=False, default=str)


def _resolve_date(date_str: str | None) -> dt.date:
    if date_str:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    return dt.date.today() - dt.timedelta(days=1)
