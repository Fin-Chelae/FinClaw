"""Financial history management (FINANCIAL_HISTORY.md).

Separate from the main HISTORY.md - this file tracks financial-specific
queries and analyses with structured metadata for later retrieval.
"""

from datetime import datetime
from pathlib import Path
from typing import Any


class FinancialHistoryManager:
    """Manages FINANCIAL_HISTORY.md read/write and search."""

    def __init__(self, history_path: Path):
        self.path = history_path

    def read_all(self) -> str:
        """Read entire financial history."""
        if self.path.exists():
            return self.path.read_text(encoding="utf-8")
        return ""

    def write_all(self, content: str) -> None:
        """Overwrite financial history (used by consolidation)."""
        self.path.write_text(content, encoding="utf-8")

    async def add_entry(
        self,
        query: str,
        response: str,
        intent: Any,
        tools_used: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a financial history entry.

        Uses two detail levels:
        - price_query: brief record (query + result summary)
        - earnings_calendar / financial_analysis / others: detailed record (query + tools + key findings)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        tickers = ", ".join(intent.tickers) if intent.tickers else "N/A"
        intent_label = intent.intent_type.replace("_", " ").title()

        if intent.intent_type == "price_query":
            entry = (
                f"### {now} | {tickers} | {intent_label}\n"
                f"- **Query**: {query[:200]}\n"
                f"- **Result**: {response[:300]}\n"
                f"---\n\n"
            )
        else:
            tools_str = ", ".join(tools_used) if tools_used else "none"
            metadata = metadata or {}
            period = metadata.get("period", "N/A")
            raw_files = metadata.get("raw_files") or []
            analysis_file = metadata.get("analysis_file")
            index_id = metadata.get("index_id")
            history_ref = metadata.get("history_ref")

            artifact_lines: list[str] = []
            if period and period != "N/A":
                artifact_lines.append(f"- **Period**: {period}")
            if raw_files:
                artifact_lines.append(f"- **Raw Files**: {', '.join(raw_files)}")
            if analysis_file:
                artifact_lines.append(f"- **Analysis File**: {analysis_file}")
            if index_id:
                artifact_lines.append(f"- **Cache Index ID**: {index_id}")
            if history_ref:
                artifact_lines.append(f"- **History Ref**: {history_ref}")
            artifact_block = "\n".join(artifact_lines)
            if artifact_block:
                artifact_block += "\n"

            entry = (
                f"### {now} | {tickers} | {intent_label}\n"
                f"- **Query**: {query[:200]}\n"
                f"- **Tickers**: {tickers}\n"
                f"- **Tools Used**: {tools_str}\n"
                f"- **Key Findings**: {response[:1000]}\n"
                f"{artifact_block}"
                f"---\n\n"
            )

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)

    def search(
        self,
        tickers: list[str] | None = None,
        intent_type: str | None = None,
    ) -> list[dict[str, str]]:
        """Search financial history by ticker and/or intent type.

        Uses keyword matching (grep-style). No semantic search needed.
        Returns up to 5 most recent matching entries.
        """
        content = self.read_all()
        if not content:
            return []

        results = []
        entries = content.split("---")

        for entry in entries:
            entry = entry.strip()
            if not entry or entry.startswith("# "):
                continue

            match = True
            if tickers:
                match = any(t.upper() in entry.upper() for t in tickers)
            if intent_type and match:
                label = intent_type.replace("_", " ").lower()
                match = label in entry.lower()

            if match:
                lines = entry.strip().split("\n")
                title = lines[0].lstrip("# ") if lines else entry[:100]
                results.append({
                    "title": title,
                    "content": entry[:500],
                })

        return results[-5:]
