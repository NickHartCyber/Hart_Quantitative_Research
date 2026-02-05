"""RSS news placeholder until the real feed integration is added."""
from __future__ import annotations

from typing import Any


def get_company_news(ticker: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """
    Placeholder for RSS-based company news.
    Returns an empty list until an RSS backend is wired in.
    """
    _ = (ticker, limit)
    return []
