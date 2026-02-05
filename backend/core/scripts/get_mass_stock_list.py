"""Ticker list utilities with safe fallbacks."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from loguru import logger


def normalize_and_filter(tickers: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in tickers or []:
        if raw is None:
            continue
        token = str(raw).strip().upper()
        token = re.sub(r"[^A-Z0-9\.\-]", "", token)
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return cleaned


def write_ticker_file(path: Path, tickers: Iterable[str], *, header: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header:
        lines.append(f"# {header}")
    lines.extend(normalize_and_filter(tickers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_seed() -> list[str]:
    raw = os.getenv("HART_QUANT_DEFAULT_TICKERS", "SPY,QQQ,DIA,IWM")
    return normalize_and_filter(re.split(r"[,\s]+", raw))


def fetch_sp500(*, prefer: str | None = None) -> list[str]:
    """
    Return an S&P 500 ticker list when available.

    This fallback returns a small ETF seed list to keep the pipeline running.
    """
    logger.warning("Using fallback S&P 500 list (seed tickers).")
    return _default_seed()


def fetch_russell2000_iwm() -> list[str]:
    """
    Return a Russell 2000 ticker list when available.

    This fallback returns a small ETF seed list to keep the pipeline running.
    """
    logger.warning("Using fallback Russell 2000 list (seed tickers).")
    return _default_seed()


def build_mass_universe(
    *,
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    include_russell2000: bool = True,
    include_nasdaq_all: bool = False,
    include_russell3000: bool = False,
) -> list[str]:
    universe: list[str] = []
    if include_sp500:
        universe.extend(fetch_sp500())
    if include_nasdaq100:
        universe.extend(["QQQ"])
    if include_russell2000:
        universe.extend(fetch_russell2000_iwm())
    if include_nasdaq_all:
        universe.extend(["QQQ"])
    if include_russell3000:
        universe.extend(fetch_russell2000_iwm())
    return normalize_and_filter(universe)
