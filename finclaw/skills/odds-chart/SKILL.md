---
name: odds-chart
description: Plot probability-over-time chart for a Polymarket prediction market using cached history data.
---

# Prediction Market Chart

Use this skill when the user asks to **chart, plot, graph, or visualize** prediction market odds or probability over time.

Only Polymarket supports price history (CLOB timeseries). Kalshi does not expose historical candlestick data publicly.

---

## Step 1 — Get the history data

If you don't already have market history, call:

```
prediction_market(query="Get market history for <event name or slug>")
```

The synthesis output will end with a line like:
```
(Full raw data: /path/to/workspace/cache/prediction_market_20260227_143012_123456.json)
```

Copy that path exactly — you'll need it in Step 2.

---

## Step 2 — Read the cache file

```
read_file("/path/to/cache/prediction_market_....json")
```

Navigate to the points array based on which command was used:

| Command used | Path to points |
|---|---|
| `market_history` | `result["history"]["points"]` |
| `top_mover` | `result["history"]["points"]` |
| `history` (standalone) | `result["points"]` |

Each point has the shape: `{"timestamp": "2026-02-01T12:00:00Z", "probability": 0.6523}`

Also extract:
- **Title**: `result["detail"]["title"]` (for market_history/top_mover) or the user's query text (for standalone history)
- **Interval**: `result["history"]["interval"]` or `result["interval"]` (e.g. `"1w"`)

---

## Step 3 — Ensure the charts directory exists

```
exec("mkdir -p {workspace}/charts")
```

Replace `{workspace}` with the actual workspace path from the system prompt.

---

## Step 4 — Write the plotting script

Write the script below to `{workspace}/charts/plot_pm.py`, substituting the three placeholders:

```python
import json
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE_FILE = "{CACHE_FILE_PATH}"
OUTPUT_FILE = "{OUTPUT_PATH}"
TITLE = "{MARKET_TITLE}"

with open(CACHE_FILE) as f:
    raw = json.load(f)

# Locate the points array (handle market_history / top_mover / standalone history)
hist = raw.get("history") or raw
points = hist.get("points", [])
if not points:
    print("No data points found in cache file"); sys.exit(1)


def _parse_ts(val) -> datetime:
    """Handle ISO strings ('2026-02-01T12:00:00Z') and numeric Unix timestamps
    (int, float, or numeric strings in seconds or milliseconds)."""
    if isinstance(val, (int, float)):
        unix = val / 1000 if val > 1e10 else val
        return datetime.fromtimestamp(unix, tz=timezone.utc)
    s = str(val).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        unix = float(s)
        unix = unix / 1000 if unix > 1e10 else unix
        return datetime.fromtimestamp(unix, tz=timezone.utc)
    except (ValueError, OSError) as exc:
        print(f"Cannot parse timestamp {val!r}: {exc}"); sys.exit(1)


timestamps = [_parse_ts(p["timestamp"]) for p in points]
probs = [round(float(p["probability"]) * 100, 2) for p in points]

# ── Chart ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

ax.plot(timestamps, probs, color="#00d4aa", linewidth=1.8, zorder=3)
ax.fill_between(timestamps, probs, alpha=0.12, color="#00d4aa")

# Annotate start and end probability
ax.annotate(f"{probs[0]:.1f}%", xy=(timestamps[0], probs[0]),
            xytext=(6, 4), textcoords="offset points",
            color="#00d4aa", fontsize=9)
ax.annotate(f"{probs[-1]:.1f}%", xy=(timestamps[-1], probs[-1]),
            xytext=(-30, 4), textcoords="offset points",
            color="#00d4aa", fontsize=9)

ax.set_ylim(max(0, min(probs) - 5), min(100, max(probs) + 5))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate(rotation=30, ha="right")

ax.set_title(TITLE, color="white", fontsize=13, pad=12, wrap=True)
ax.set_ylabel("Yes Probability", color="#aaaaaa", fontsize=10)
ax.tick_params(colors="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2a2a")
ax.grid(axis="y", color="#1e1e1e", linewidth=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Chart saved: {OUTPUT_FILE}")
```

**Substitutions:**
- `{CACHE_FILE_PATH}` → the exact cache file path from Step 2
- `{OUTPUT_PATH}` → `{workspace}/charts/pm_{slug}_{YYYYMMDD}.png` (use the event slug or a short title slug and today's date)
- `{MARKET_TITLE}` → the event title string

---

## Step 5 — Execute the script

```
exec("python3 {workspace}/charts/plot_pm.py")
```

If it exits with `"Chart saved: ..."`, proceed to Step 6. If it errors, check that the cache file path is correct and the points array is non-empty.

---

## Step 6 — Report the result

Tell the user:
1. The chart was saved to `{output_path}` (give the full path so they can open it)
2. A one-sentence trend summary, e.g.:
   > "Probability climbed from 34% to 61% over the past week, with the sharpest move on Feb 24."

Use the `summary` object from the cache file (`start.probability`, `end.probability`, `change_pct`) to fill in those numbers — no need to read individual points.

---

## Notes

- **Downsampled data**: if `downsampled: true` in the cache, the 30 evenly-spaced points are sufficient for a clean chart. Full resolution isn't needed.
- **Multi-sub-market events**: `market_history` always charts the first sub-market's Yes token. If the user wants a different sub-market, they need the specific `token_id` (from `market_detail → clob_token_ids[0]`) and the standalone `history` command.
- **Kalshi**: has no public price history endpoint — tell the user only Polymarket charts are supported.
