---
name: earnings
description: Look up earnings dates, EPS beat/miss history, analyst consensus estimates, and estimate revisions for US/global stocks.
---

# Earnings Calendar & Surprise Tracking

Use this skill when the user asks about:
- When a company reports earnings ("when does AAPL report?")
- Whether a company beat or missed estimates ("did NVDA beat last quarter?")
- Upcoming earnings this week/month for a watchlist
- Forward EPS or revenue consensus estimates
- Whether analysts have been raising or cutting estimates

---

## Command reference

| User intent | Command | Key params |
|---|---|---|
| Next earnings date | `calendar` | `symbol` |
| Upcoming earnings for a list | `upcoming` | `symbols`, `days_ahead` |
| Beat/miss history | `surprise` | `symbol`, `limit` |
| Forward EPS/revenue estimates | `consensus` | `symbol` |
| Estimate revision trend | `revisions` | `symbol` |

---

## Workflow: "When does X report?"

Call `earnings_calendar` with `command="calendar"` and the ticker:

```
earnings_calendar(command="calendar", symbol="AAPL")
```

Report: next date, EPS estimate range (low/avg/high), revenue estimate range.

---

## Workflow: "Did X beat last quarter?" / "Show me NVDA's earnings history"

Call `surprise` to get the last 8 reported quarters:

```
earnings_calendar(command="surprise", symbol="NVDA", limit=8)
```

Format the response as a table: date | EPS est | EPS actual | surprise % | beat/miss.
Summarise the beat rate (e.g. "beat in 7 of the last 8 quarters").

---

## Workflow: "What are analysts expecting for MSFT this quarter?"

Call `consensus` for forward estimates:

```
earnings_calendar(command="consensus", symbol="MSFT")
```

Report: current quarter EPS avg/low/high, year-over-year growth, revenue estimate, analyst count and recommendation.

---

## Workflow: "Have analysts been cutting estimates for META?"

Call `revisions` to see upgrade/downgrade counts over recent periods:

```
earnings_calendar(command="revisions", symbol="META")
```

Interpret the result:
- More `up_last_30_days` than `down_last_30_days` → analysts are bullish / raising numbers
- More `down_last_30_days` → negative revision trend, worth flagging
- Also check `eps_trend` to show the actual EPS estimate drift (e.g. current vs 90 days ago)

---

## Workflow: "What earnings are coming up this week?"

The user must provide a list of symbols (FinClaw cannot scan the whole market).
Ask for their watchlist if not given, then call `upcoming`:

```
earnings_calendar(command="upcoming", symbols="AAPL,MSFT,NVDA,AMZN,TSLA,META,GOOGL", days_ahead=7)
```

Group results by date. For each company show: ticker, date, EPS estimate.

---

## Notes

- **Data source**: Yahoo Finance via yfinance. Coverage is best for US/global large caps.
- **`upcoming` is slow for large lists** (one API call per ticker). Keep to ≤ 30 symbols.
- **`surprise` only shows reported quarters** — future dates are excluded automatically.
- If a ticker returns `"error": "No earnings calendar available"`, it may be a non-US stock, ETF, or the data is simply not in Yahoo Finance for that ticker.
