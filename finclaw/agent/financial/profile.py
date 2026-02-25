"""Finance profile management (FINANCE_PROFILE.md)."""

from datetime import datetime
from pathlib import Path
import re


class FinanceProfileManager:
    """Manages the user's investment profile stored in FINANCE_PROFILE.md.

    The profile is created on first interaction via LLM-driven conversation
    (not hardcoded CLI forms), making it compatible with all channels.
    """

    def __init__(self, workspace: Path):
        self.profile_path = workspace / "FINANCE_PROFILE.md"

    def exists(self) -> bool:
        """Check whether profile file exists."""
        return self.profile_path.exists()

    def is_complete(self) -> bool:
        """Check whether profile contains required sections and non-draft values."""
        if not self.profile_path.exists():
            return False
        content = self.profile_path.read_text(encoding="utf-8").strip()
        if len(content) < 80:
            return False

        required_sections = [
            "## Markets",
            "## Style",
            "## Interests",
            "## Change Log",
        ]
        if not all(section in content for section in required_sections):
            return False

        incomplete_markers = [
            "Not specified yet",
            "Not specified",
            "(draft)",
            "TBD",
        ]
        return not any(marker in content for marker in incomplete_markers)

    def read(self) -> str:
        """Read profile content."""
        if self.profile_path.exists():
            return self.profile_path.read_text(encoding="utf-8")
        return ""

    def write(self, content: str) -> None:
        """Write profile content."""
        self.profile_path.write_text(content, encoding="utf-8")

    def create_from_answers(self, answers: dict) -> str:
        """Create a profile from structured answers.

        Args:
            answers: Dict with keys: markets, risk, horizon, approach, sectors, tickers.

        Returns:
            The generated markdown content.
        """
        now = datetime.now().strftime("%Y-%m-%d")
        content = f"""# Finance Profile
Last updated: {now}

## Markets
- Primary: {answers.get('markets', 'Not specified')}

## Style
- Risk: {answers.get('risk', 'Balanced')}
- Horizon: {answers.get('horizon', 'Medium-term')}
- Approach: {answers.get('approach', 'Fundamental + Technical')}

## Interests
- Sectors: {answers.get('sectors', 'Not specified')}
- Tickers: {answers.get('tickers', 'Not specified')}

## Change Log
- {now}: Initial profile created
"""
        self.write(content)
        return content

    def bootstrap_from_message(self, message: str, tickers: list[str] | None = None) -> str:
        """Create a minimal profile from the first financial message.

        This prevents repeated onboarding loops if the model does not create
        FINANCE_PROFILE.md on its own.
        """
        message_lower = message.lower()
        tickers = tickers or []

        markets: list[str] = []
        if any(k in message_lower for k in ["美股", "us stock", "nasdaq", "nyse"]) or any(
            re.fullmatch(r"[A-Z]{1,5}", t or "") for t in tickers
        ):
            markets.append("US Stocks")
        if any(k in message_lower for k in ["a股", ".sh", ".sz", ".ss"]) or any(
            ".SH" in t or ".SZ" in t or ".SS" in t for t in tickers
        ):
            markets.append("A-shares")
        if any(k in message_lower for k in ["crypto", "加密", "btc", "eth"]) or any(
            any(c in (t or "") for c in ["-USD", "-USDT", "BTC", "ETH"]) for t in tickers
        ):
            markets.append("Crypto")
        if not markets:
            markets = ["Not specified yet"]

        answers = {
            "markets": ", ".join(dict.fromkeys(markets)),
            "risk": "Balanced (draft)",
            "horizon": "Medium-term (draft)",
            "approach": "Fundamental + Technical (draft)",
            "sectors": "Not specified yet",
            "tickers": ", ".join(tickers) if tickers else "Not specified yet",
        }
        return self.create_from_answers(answers)
