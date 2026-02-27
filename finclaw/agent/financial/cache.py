"""Financial data cache management (financial_data/ + index.json).

Stores raw API responses and analysis results locally to avoid
re-fetching and re-analyzing the same data.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# TTL in seconds per task type
TTL_SECONDS: dict[str, int] = {
    "price_query": 0,              # never cache (real-time)
    "financial_analysis": 604800,  # 7 days
    "earnings_data": 2592000,      # 30 days
    "market_search": 86400,        # 1 day
    "prediction_market": 300,      # 5 minutes (market odds change rapidly)
}


class FinancialDataCache:
    """Manages financial_data/ directory and index.json.

    Directory layout:
        financial_data/
        ├── index.json
        ├── raw/{TICKER}/{YYYYMMDD}_{type}.json
        └── analysis/{TICKER}/{YYYYMMDD}_{TICKER}_{topic}_analysis.json
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.index_path = cache_dir / "index.json"
        self.raw_dir = cache_dir / "raw"
        self.analysis_dir = cache_dir / "analysis"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"entries": []}
        return {"entries": []}

    def _save_index(self, index: dict) -> None:
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def lookup(
        self,
        tickers: list[str] | None = None,
        task_type: str | None = None,
    ) -> list[dict]:
        """Find matching, non-expired cache entries.

        Returns:
            List of matching index entries.
        """
        index = self._load_index()
        now = datetime.now(timezone.utc)
        results = []

        for entry in index.get("entries", []):
            if self._is_expired(entry.get("expires_at"), now):
                continue
            if tickers and entry.get("ticker") not in [t.upper() for t in tickers]:
                continue
            if task_type and entry.get("task_type") != task_type:
                continue
            results.append(entry)

        return results

    @staticmethod
    def _is_expired(expires_at: str | None, now: datetime) -> bool:
        """Safely evaluate expiration timestamp."""
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return exp < now
        except ValueError:
            # Invalid timestamp should not block normal behavior.
            return False

    def save_raw(self, ticker: str, data_type: str, data: Any) -> str:
        """Save raw data to raw/{TICKER}/.

        Returns:
            Relative file path from cache_dir.
        """
        ticker_dir = self.raw_dir / ticker.upper()
        ticker_dir.mkdir(exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{date_str}_{data_type}.json"
        file_path = ticker_dir / filename

        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return f"raw/{ticker.upper()}/{filename}"

    def save_analysis(self, ticker: str, topic: str, analysis: Any) -> str:
        """Save analysis result to analysis/{TICKER}/.

        Returns:
            Relative file path from cache_dir.
        """
        ticker_dir = self.analysis_dir / ticker.upper()
        ticker_dir.mkdir(exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{date_str}_{ticker.upper()}_{topic}_analysis.json"
        file_path = ticker_dir / filename

        file_path.write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return f"analysis/{ticker.upper()}/{filename}"

    def add_index_entry(
        self,
        ticker: str,
        task_type: str,
        query: str,
        summary: str,
        raw_files: list[str] | None = None,
        analysis_file: str | None = None,
        period: str | None = None,
    ) -> dict | None:
        """Add a cache index entry and return it."""
        ttl = TTL_SECONDS.get(task_type, 604800)
        if ttl == 0:
            return None

        index = self._load_index()
        now = datetime.now(timezone.utc)

        entry = {
            "id": f"{ticker.lower()}_{task_type}_{now.strftime('%Y%m%d%H%M')}",
            "query_hash": hashlib.md5(query.encode()).hexdigest(),
            "ticker": ticker.upper(),
            "task_type": task_type,
            "period": period,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
            "raw_files": raw_files or [],
            "analysis_file": analysis_file,
            "summary": summary[:200],
            "history_ref": f"{now.strftime('%Y-%m-%d')} | {ticker.upper()} | {task_type}",
        }

        index["entries"].append(entry)
        self._save_index(index)
        return entry
