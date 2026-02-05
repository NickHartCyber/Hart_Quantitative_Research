# backend/api_prices_tiingo.py
from __future__ import annotations
import ast
import bisect
import datetime as dt
import json
import logging
import math
import os
import statistics
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Literal, List, Dict, Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.core.edgar_fundamentals import get_edgar_fundamentals
from backend.platform_apis.tiingo_api.tiingo_api import (
    get_daily_prices,
    get_fundamentals_metrics,
    get_fundamentals_statements,
    get_iex_prices,
    get_latest_prices,
)
from backend.platform_apis.rss_feed_api.rss_feed import get_company_news

log = logging.getLogger(__name__)
_TZ_NY = ZoneInfo("America/New_York")

router = APIRouter(prefix="/api", tags=["market"])

TF = Literal["1D", "5D", "1M", "6M", "1Y", "5Y", "10Y"]

_AI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_AI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
_AI_FUNDAMENTAL_KEYS = [
    "price",
    "market_cap",
    "pe_ratio",
    "forward_pe",
    "peg_ratio",
    "eps_ttm",
    "roe",
    "debt_to_equity",
    "dividend_yield",
    "profit_margin",
    "book_value",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "beta",
    "shares_outstanding",
    "sector",
    "currency",
    "revenue_ttm",
    "revenue_growth",
    "eps_growth",
    "gross_margin",
    "gross_margin_trend",
    "fcf_ttm",
    "fcf_margin",
    "fcf_yield",
    "net_debt",
    "net_debt_to_ebitda",
    "ps_ratio",
    "ev_to_ebitda",
    "dilution_rate",
]
_TRADING_INFO_KEYS = [
    "lastPrice",
    "previousClose",
    "open",
    "dayLow",
    "dayHigh",
    "yearLow",
    "yearHigh",
    "lastVolume",
    "tenDayAverageVolume",
    "threeMonthAverageVolume",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "marketCap",
    "shares",
    "currency",
    "exchange",
    "yearChange",
]

_INDEX_SPECS = [
    {
        "key": "spx",
        "name": "SPDR S&P 500 ETF (SPY)",
        "ticker": "SPY",
        "data_ticker": "SPY",
        "news_symbol": "SPY",
    },
    {
        "key": "dji",
        "name": "SPDR Dow Jones Industrial Average ETF (DIA)",
        "ticker": "DIA",
        "data_ticker": "DIA",
        "news_symbol": "DIA",
    },
    {
        "key": "ixic",
        "name": "Invesco QQQ Trust (QQQ)",
        "ticker": "QQQ",
        "data_ticker": "QQQ",
        "news_symbol": "QQQ",
    },
    {
        "key": "rut",
        "name": "iShares Russell 2000 ETF (IWM)",
        "ticker": "IWM",
        "data_ticker": "IWM",
        "news_symbol": "IWM",
    },
]

_FILES_DIR = (Path(__file__).resolve().parent / ".." / "files").resolve()
_MARKET_CACHE_DIR = _FILES_DIR / "market_cache"
_MARKET_INDICES_CACHE_PATH = _MARKET_CACHE_DIR / "indices.json"


def _read_market_indices_cache() -> dict[str, Any] | None:
    try:
        if not _MARKET_INDICES_CACHE_PATH.exists():
            return None
        with _MARKET_INDICES_CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "indices" not in payload:
            return None
        return payload
    except Exception as exc:
        log.warning("Failed to read market indices cache %s: %s", _MARKET_INDICES_CACHE_PATH, exc)
        return None


def _write_market_indices_cache(payload: dict[str, Any]) -> Path:
    _MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_payload = dict(payload)
    cached_payload["cached_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with _MARKET_INDICES_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cached_payload, handle, ensure_ascii=True)
    return _MARKET_INDICES_CACHE_PATH


def _build_market_indices_payload(
    analysis_tf: TF,
    news_limit: int,
    include_ai: bool,
    extended: bool,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for spec in _INDEX_SPECS:
        try:
            summaries.append(_build_index_summary(spec, analysis_tf, news_limit, include_ai, extended))
        except Exception as exc:
            log.exception("Index summary failed for %s: %s", spec["ticker"], exc)
            summaries.append(
                {
                    "key": spec.get("key") or spec["ticker"],
                    "name": spec["name"],
                    "ticker": spec["ticker"],
                    "error": str(exc),
                }
            )

    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_timeframe": analysis_tf,
        "include_ai": include_ai,
        "indices": summaries,
    }


def refresh_market_indices_cache(
    analysis_tf: TF = "6M",
    news_limit: int = 5,
    include_ai: bool = True,
    extended: bool = True,
) -> tuple[dict[str, Any], Path]:
    payload = _build_market_indices_payload(analysis_tf, news_limit, include_ai, extended)
    cache_path = _write_market_indices_cache(payload)
    return payload, cache_path


def _parse_tiingo_ts(value: Any) -> dt.datetime | None:
    if value in (None, "", False):
        return None
    try:
        if isinstance(value, (int, float)):
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).astimezone(_TZ_NY)
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_TZ_NY)
        return parsed.astimezone(_TZ_NY)
    except Exception:
        return None


def _normalize_tiingo_series(
    payload: Any,
    tf: TF,
    *,
    intraday: bool,
) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts_raw = row.get("date") or row.get("timestamp") or row.get("time") or row.get("datetime")
        ts = _parse_tiingo_ts(ts_raw)
        if ts is None:
            continue

        if intraday:
            label = ts.strftime("%H:%M") if tf == "1D" else ts.strftime("%a %H:%M")
        else:
            label = ts.strftime("%Y-%m-%d")

        close_val = _safe_number(
            row.get("close")
            if row.get("close") is not None
            else row.get("adjClose") if row.get("adjClose") is not None else row.get("price")
        )
        if close_val is None:
            continue

        open_val = _safe_number(
            row.get("open")
            if row.get("open") is not None
            else row.get("adjOpen") if row.get("adjOpen") is not None else close_val
        )
        high_val = _safe_number(
            row.get("high")
            if row.get("high") is not None
            else row.get("adjHigh") if row.get("adjHigh") is not None else close_val
        )
        low_val = _safe_number(
            row.get("low")
            if row.get("low") is not None
            else row.get("adjLow") if row.get("adjLow") is not None else close_val
        )
        volume_val = _safe_number(row.get("volume"))

        out.append(
            {
                "label": label,
                "price": close_val,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "ts": ts.isoformat(),
                "volume": volume_val,
            }
        )

    out.sort(key=lambda r: r.get("ts") or "")
    return _filter_latest_session(out) if intraday and tf == "1D" else out


def _tiingo_daily_range(tf: TF, now_ny: dt.datetime) -> tuple[str | None, str | None]:
    if tf == "5D":
        days = 10
    elif tf == "1M":
        days = 35
    elif tf == "6M":
        days = 190
    elif tf == "1Y":
        days = 370
    elif tf == "5Y":
        days = 5 * 366
    else:
        days = 10 * 366

    start = (now_ny - dt.timedelta(days=days)).date().isoformat()
    end = (now_ny + dt.timedelta(days=1)).date().isoformat()
    return start, end


def _tiingo_intraday_range(tf: TF, now_ny: dt.datetime) -> tuple[str | None, str | None, str]:
    if tf == "5D":
        start = (now_ny - dt.timedelta(days=7)).date().isoformat()
        resample = "5min"
    else:
        start = now_ny.date().isoformat()
        resample = "1min"
    end = (now_ny + dt.timedelta(days=1)).date().isoformat()
    return start, end, resample


def _get_prices_from_tiingo(ticker: str, tf: TF) -> List[Dict[str, Any]]:
    now_ny = dt.datetime.now(_TZ_NY)
    if tf == "1D":
        start, end, resample = _tiingo_intraday_range(tf, now_ny)
        payload = get_iex_prices(
            ticker,
            start_date=start,
            end_date=end,
            resample_freq=resample,
        )
        rows = _normalize_tiingo_series(payload, tf, intraday=True)
        if rows:
            return rows
        # Intraday can be empty off-hours; fall back to recent daily bars.
        start_d, end_d = _tiingo_daily_range("5D", now_ny)
        daily_payload = get_daily_prices(ticker, start_date=start_d, end_date=end_d)
        daily_rows = _normalize_tiingo_series(daily_payload, "5D", intraday=False)
        return daily_rows[-1:] if daily_rows else []

    start, end = _tiingo_daily_range(tf, now_ny)
    payload = get_daily_prices(ticker, start_date=start, end_date=end)
    rows = _normalize_tiingo_series(payload, tf, intraday=False)
    if tf == "5D" and len(rows) > 5:
        rows = rows[-5:]
    return rows



def _ms_to_dt(ms: int) -> dt.datetime:
    # API returns ms since epoch (UTC). Normalize to US/Eastern for labels.
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc).astimezone(_TZ_NY)


def _parse_ts(val: Any) -> dt.datetime | None:
    """Parse a timestamp value (iso string or epoch seconds) into an aware NY datetime."""
    if val in (None, "", False):
        return None
    try:
        if isinstance(val, (int, float)):
            return dt.datetime.fromtimestamp(float(val), tz=_TZ_NY)
        text = str(val).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_TZ_NY)
        return parsed.astimezone(_TZ_NY)
    except Exception:
        return None


def _latest_ts(rows: list[dict[str, Any]]) -> dt.datetime | None:
    """Return the most recent timestamp in a price series, if any."""
    newest = None
    for row in rows:
        ts = _parse_ts(row.get("ts"))
        if ts is None:
            continue
        if newest is None or ts > newest:
            newest = ts
    return newest


def _filter_latest_session(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only rows belonging to the most recent session date in the series.
    Helps avoid mixing prior-day candles when providers return multi-day windows.
    """
    latest = _latest_ts(rows)
    if latest is None:
        return rows
    latest_date = latest.date()
    filtered = [row for row in rows if (_parse_ts(row.get("ts")) or latest).date() == latest_date]
    return filtered or rows


def _intraday_series_stale(rows: list[dict[str, Any]], now_ny: dt.datetime) -> bool:
    """
    Decide if a 1D series is stale relative to the current NY time.
    - Consider data fresh if it is from today.
    - Allow previous-day after-hours data overnight before 4am ET.
    - Flag data as stale when the last timestamp lags >20m during trading hours.
    """
    latest = _latest_ts(rows)
    if latest is None:
        return True

    latest = latest.astimezone(_TZ_NY)
    today = now_ny.date()
    if latest.date() == today:
        if now_ny.hour >= 4 and now_ny.hour <= 20 and (now_ny - latest) > dt.timedelta(minutes=20):
            return True
        return False

    # Overnight window: keep previous day's after-hours until 4am ET
    if (today - latest.date()).days == 1 and now_ny.hour < 4:
        return False

    return True


def _get_prices_with_fallback(ticker: str, tf: TF, extended: bool) -> tuple[list[dict], str]:
    """
    Fetch prices from Tiingo only. Returns (rows, source).
    """
    try:
        rows = _get_prices_from_tiingo(ticker, tf)
        return rows, "tiingo"
    except Exception as exc:
        log.warning("Tiingo price fetch failed for %s %s: %s", ticker, tf, exc)
        return [], "tiingo"


def _is_missing(value: Any) -> bool:
    """Return True when a value is None/NaN-like (including pandas NA)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pick_first(container: Any, *keys: str) -> Any:
    """Grab the first non-missing value for any of the provided keys."""
    getter = container.get if hasattr(container, "get") else None
    for key in keys:
        val = getter(key) if getter else None
        if not _is_missing(val):
            return val
    return None


def _normalize_quotes_tiingo(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        return []

    results: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("ticker") or row.get("symbol") or row.get("tkr")
        symbol_text = str(symbol).strip().upper() if symbol else ""
        if not symbol_text:
            continue
        price = _safe_number(row.get("close") or row.get("adjClose") or row.get("last") or row.get("price"))
        if price is None:
            continue
        results.append(
            {
                "symbol": symbol_text,
                "description": row.get("name") or row.get("description") or symbol_text,
                "asset_type": row.get("assetType") or "EQUITY",
                "bid": None,
                "ask": None,
                "last": price,
                "mark": price,
                "close": price,
                "mid": price,
            }
        )
    return results


def _fill_missing(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Populate missing keys in target with values from source (non-destructive)."""
    for key, val in source.items():
        if _is_missing(val):
            continue
        if key not in target or _is_missing(target.get(key)):
            target[key] = val
    return target


def _latest_price_from_series(prices: dict[str, list[dict[str, Any]]]) -> float | None:
    """Pick a recent price from the assembled price series as a final fallback."""
    tf_pref = ["1D", "5D", "1M", "6M", "1Y", "5Y", "10Y"]
    for tf in tf_pref:
        series = prices.get(tf) or []
        if not series:
            continue
        last = series[-1]
        if not isinstance(last, dict):
            continue
        for key in ("close", "price"):
            val = last.get(key)
            if _is_missing(val):
                continue
            try:
                return float(val)
            except Exception:
                continue
    return None


def _get_fundamentals(
    ticker: str,
    *,
    price_hint: float | None = None,
    include_filings: bool = True,
    timeout: float | None = None,
) -> tuple[dict[str, Any], str]:
    """EDGAR-backed fundamentals wrapper (kept for local call sites)."""
    return get_edgar_fundamentals(
        ticker,
        price_hint=price_hint,
        include_filings=include_filings,
        timeout=timeout,
    )


def _parse_news_datetime(value) -> dt.datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        cleaned = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = dt.datetime.fromisoformat(cleaned)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        pass
    try:
        return dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc)
    except Exception:
        return None


def _normalize_news_items(raw_news, limit: int = 30) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(raw_news, pd.DataFrame):
        try:
            records = raw_news.reset_index().to_dict("records")
        except Exception:
            records = []
    elif isinstance(raw_news, list):
        records = [row for row in raw_news if isinstance(row, dict)]

    items: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for row in records:
        merged = dict(row)
        content_raw = row.get("content")
        if isinstance(content_raw, dict):
            merged.update(content_raw)
        elif isinstance(content_raw, str):
            trimmed = content_raw.strip()
            if trimmed:
                parsed = None
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(trimmed)
                        break
                    except Exception:
                        continue
                if isinstance(parsed, dict):
                    merged.update(parsed)

        provider = merged.get("provider")
        if isinstance(provider, dict):
            merged.setdefault("publisher", provider.get("displayName") or provider.get("name"))

        canonical = merged.get("canonicalUrl") or merged.get("canonical_url")
        if isinstance(canonical, dict):
            merged.setdefault("link", canonical.get("url"))
        clickthrough = merged.get("clickThroughUrl") or merged.get("click_through_url")
        if isinstance(clickthrough, dict):
            merged.setdefault("link", merged.get("link") or clickthrough.get("url"))

        title = str(merged.get("title") or "").strip()
        if not title:
            continue
        published_raw = (
            merged.get("published_at")
            or merged.get("providerPublishTime")
            or merged.get("publishTime")
            or merged.get("time")
            or merged.get("created")
            or merged.get("pubDate")
            or merged.get("displayTime")
        )
        published_dt = _parse_news_datetime(published_raw)
        published_at = published_dt.isoformat() if published_dt else None
        publisher_raw = merged.get("publisher") or merged.get("source")
        if isinstance(publisher_raw, dict):
            publisher_raw = publisher_raw.get("displayName") or publisher_raw.get("name")
        publisher = None if _is_missing(publisher_raw) else (str(publisher_raw).strip() or None)
        link_raw = merged.get("link") or merged.get("url")
        if isinstance(link_raw, dict):
            link_raw = link_raw.get("url")
        link = None if _is_missing(link_raw) else (str(link_raw).strip() or None)
        summary_raw = merged.get("summary") or merged.get("description")
        summary = None if _is_missing(summary_raw) else (str(summary_raw).strip() or None)
        related_raw = (
            merged.get("relatedTickers")
            or merged.get("related_tickers")
            or merged.get("related")
            or merged.get("tickerSymbols")
            or merged.get("symbols")
        )
        related = (
            [str(t).upper() for t in related_raw if t]
            if isinstance(related_raw, (list, tuple, set))
            else None
        )
        uuid = merged.get("uuid") or merged.get("id")
        key = (title, published_at, link, uuid)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "id": uuid,
                "title": title,
                "link": link,
                "publisher": publisher,
                "published_at": published_at,
                "summary": summary,
                "related_tickers": related,
            }
        )

    items.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return items[:limit]


def _get_news_with_fallback(ticker: str, limit: int = 30) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = get_company_news(ticker, limit=limit)
        items = _normalize_news_items(raw, limit=limit)
        return items, "rss_pending"
    except Exception as exc:
        log.warning("RSS news fetch failed for %s: %s", ticker, exc)
        return [], "rss_pending"


class TickerAiAnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker symbol, e.g., AAPL")
    timeframe: TF = Field("1D", description="Timeframe to analyze the chart context")
    prices: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional price series to summarize for the LLM. Backend fetches when omitted.",
    )
    fundamentals: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional fundamentals snapshot; backend will fetch if missing.",
    )
    news: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional news headlines; backend will fetch when omitted.",
    )
    extended: bool = Field(
        default=True,
        description="Include pre/post market in 1D fetches when the backend needs to load prices.",
    )


class TickerAiQuestionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker symbol, e.g., AAPL")
    question: str = Field(..., min_length=1, description="Question to ask about the ticker context.")
    timeframe: TF = Field("1D", description="Timeframe anchoring the price context.")
    prices: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional price series for context; backend fetches when omitted.",
    )
    fundamentals: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional fundamentals snapshot; backend will fetch if missing.",
    )
    news: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional news headlines; backend will fetch when omitted.",
    )
    extended: bool = Field(
        default=True,
        description="Include pre/post market in 1D fetches when the backend needs to load prices.",
    )


class TickerNewsAiRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker symbol, e.g., AAPL")
    news: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional news headlines; backend will fetch when omitted.",
    )
    limit: int = Field(
        default=15,
        ge=1,
        le=40,
        description="Maximum number of headlines to pass to the LLM for summarization.",
    )


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured; AI analysis is disabled.")
    return OpenAI(api_key=api_key, base_url=_AI_BASE_URL)


def _serialize_openai_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "dict"):
        method = getattr(usage, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                try:
                    return method(exclude_none=True)
                except Exception:
                    pass
    try:
        return dict(usage)
    except Exception:
        return str(usage)


def _safe_number(val: Any) -> float | None:
    try:
        num = float(val)
    except Exception:
        return None
    return num if math.isfinite(num) else None


def _thin_price_series(data: list[dict[str, Any]], max_points: int = 120) -> list[dict[str, Any]]:
    if not data:
        return []
    cleaned: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        close = _safe_number(row.get("close") if "close" in row else row.get("price"))
        if close is None:
            continue
        cleaned.append(
            {
                "label": row.get("label") or row.get("ts"),
                "ts": row.get("ts"),
                "close": close,
                "high": _safe_number(row.get("high")),
                "low": _safe_number(row.get("low")),
                "open": _safe_number(row.get("open")),
                "volume": _safe_number(row.get("volume")),
            }
        )
    if len(cleaned) <= max_points:
        return cleaned
    step = max(1, len(cleaned) // max_points)
    trimmed = [cleaned[i] for i in range(0, len(cleaned), step)]
    if trimmed[-1] is not cleaned[-1]:
        trimmed.append(cleaned[-1])
    return trimmed


def _price_stats(series: list[dict[str, Any]]) -> dict[str, Any]:
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []

    for row in series:
        if not isinstance(row, dict):
            continue
        close = _safe_number(row.get("close") if "close" in row else row.get("price"))
        high = _safe_number(row.get("high"))
        low = _safe_number(row.get("low"))
        if close is not None:
            closes.append(close)
        if high is not None:
            highs.append(high)
        if low is not None:
            lows.append(low)

    if not closes:
        return {}

    start = closes[0]
    end = closes[-1]
    change = end - start
    pct = (change / start * 100) if start else None
    stats = {
        "start": start,
        "end": end,
        "change": change,
        "pct_change": pct,
        "high": max(highs) if highs else max(closes),
        "low": min(lows) if lows else min(closes),
        "count": len(closes),
        "volatility": statistics.pstdev(closes) if len(closes) > 1 else 0.0,
    }
    if pct is not None:
        stats["direction"] = "up" if pct > 1 else "down" if pct < -1 else "flat"
    return stats


def _filter_fundamentals(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    filtered = {k: data.get(k) for k in _AI_FUNDAMENTAL_KEYS if k in data}
    if data.get("name"):
        filtered["name"] = data["name"]
    return filtered


def _filter_trading_info(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {k: data.get(k) for k in _TRADING_INFO_KEYS if k in data}


def _tiingo_latest_row(symbol: str) -> dict[str, Any]:
    try:
        payload = get_latest_prices([symbol])
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if not rows or not isinstance(rows[0], dict):
        return {}
    return rows[0]


def _tiingo_trading_info(symbol: str) -> dict[str, Any]:
    row = _tiingo_latest_row(symbol)
    if not row:
        return {}
    return {
        "lastPrice": row.get("last") or row.get("close") or row.get("adjClose"),
        "previousClose": row.get("prevClose") or row.get("previousClose"),
        "open": row.get("open"),
        "dayLow": row.get("low"),
        "dayHigh": row.get("high"),
        "lastVolume": row.get("volume"),
        "marketCap": row.get("marketCap"),
        "shares": row.get("shares"),
        "currency": row.get("currency"),
        "exchange": row.get("exchange"),
    }


def _tiingo_statement_data_map(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("statementData") or item.get("statement_data") or item.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out: dict[str, Any] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            key = row.get("dataCode") or row.get("code") or row.get("concept") or row.get("label")
            if not key:
                continue
            out[str(key)] = row.get("value") if "value" in row else row.get("val") or row.get("data")
        return out
    return {}


def _tiingo_statement_period(item: dict[str, Any]) -> str | None:
    raw = str(item.get("period") or item.get("fiscalPeriod") or item.get("periodType") or "").lower()
    if any(tag in raw for tag in ("q", "quarter")):
        return "quarterly"
    if any(tag in raw for tag in ("fy", "annual", "year")):
        return "annual"
    return None


def _tiingo_statement_date(item: dict[str, Any]) -> str | None:
    for key in ("date", "reportDate", "statementDate", "fiscalDate", "endDate"):
        val = item.get(key)
        if val:
            return str(val)[:10]
    return None


def _tiingo_statement_frame(symbol: str, statement_type: str, *, period: str | None = None) -> pd.DataFrame:
    try:
        raw = get_fundamentals_statements(symbol, statement_type=statement_type)
    except Exception:
        raw = []
    items = raw if isinstance(raw, list) else raw.get("data") if isinstance(raw, dict) else []
    if not isinstance(items, list):
        return pd.DataFrame()

    records: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if period:
            item_period = _tiingo_statement_period(item)
            if item_period and item_period != period:
                continue
        date_key = _tiingo_statement_date(item)
        if not date_key:
            continue
        data_map = _tiingo_statement_data_map(item)
        if not data_map:
            continue
        records[date_key] = data_map

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _tiingo_shares_history(symbol: str) -> pd.DataFrame:
    try:
        raw = get_fundamentals_metrics(symbol)
    except Exception:
        raw = []
    data = raw if isinstance(raw, list) else []
    frame = pd.DataFrame(data)
    if frame.empty:
        return pd.DataFrame()

    date_col = None
    for col in frame.columns:
        if "date" in col.lower() or "period" in col.lower() or "end" in col.lower():
            date_col = col
            break
    value_col = None
    for col in frame.columns:
        if col.lower() in {"sharesoutstanding", "shares_outstanding", "shares"}:
            value_col = col
            break
    if not date_col or not value_col:
        return pd.DataFrame()
    out = frame[[date_col, value_col]].dropna()
    if out.empty:
        return pd.DataFrame()
    out = out.rename(columns={value_col: "shares_outstanding"}).set_index(date_col)
    return out


def _df_to_split_payload(
    df: pd.DataFrame | None,
    *,
    row_limit: int | None = None,
    col_limit: int | None = None,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {"columns": [], "index": [], "data": []}
    frame = df.copy()
    if row_limit is not None:
        frame = frame.iloc[:row_limit, :]
    if col_limit is not None:
        frame = frame.iloc[:, :col_limit]
    payload = json.loads(frame.to_json(orient="split", date_format="iso"))
    payload["columns"] = [str(c) for c in payload.get("columns", [])]
    payload["index"] = [str(i) for i in payload.get("index", [])]
    return payload


def _df_to_records_payload(
    df: pd.DataFrame | None,
    *,
    row_limit: int | None = None,
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    frame = df.copy()
    if row_limit is not None:
        frame = frame.iloc[:row_limit, :]
    return json.loads(frame.reset_index().to_json(orient="records", date_format="iso"))


def _add_earnings_impacts(symbol: str, earnings_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if earnings_df is None or earnings_df.empty:
        return earnings_df
    df = earnings_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Earnings Date" in df.columns:
            df = df.set_index("Earnings Date")
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    if df.empty:
        return earnings_df

    start = (df.index.min() - pd.Timedelta(days=5)).date().isoformat()
    end = (df.index.max() + pd.Timedelta(days=5)).date().isoformat()
    try:
        payload = get_daily_prices(symbol, start_date=start, end_date=end)
    except Exception:
        payload = []

    history = pd.DataFrame(payload if isinstance(payload, list) else [])
    if history.empty:
        df["Earnings Move %"] = None
        df["Earnings Move $"] = None
        return df

    history_index = pd.to_datetime(history.get("date"), errors="coerce", utc=True)
    history = history.copy()
    history["trade_date"] = pd.Series(history_index).dt.date.values
    history = history.dropna(subset=["trade_date"])
    close_col = None
    for candidate in ("close", "adjClose", "adj_close", "adjclose"):
        if candidate in history.columns:
            close_col = candidate
            break
    if close_col is None:
        df["Earnings Move %"] = None
        df["Earnings Move $"] = None
        return df

    close_series = history.groupby("trade_date")[close_col].last()
    trading_dates = sorted(close_series.index)
    if not trading_dates:
        df["Earnings Move %"] = None
        df["Earnings Move $"] = None
        return df

    moves_pct = []
    moves_amt = []
    market_close = dt.time(16, 0)
    for ts in df.index:
        event_ts = ts
        try:
            if getattr(event_ts, "tzinfo", None) is None:
                event_ts = event_ts.replace(tzinfo=_TZ_NY)
            else:
                event_ts = event_ts.astimezone(_TZ_NY)
        except Exception:
            event_ts = ts

        event_date = event_ts.date()
        event_time = event_ts.time()
        pos = bisect.bisect_left(trading_dates, event_date)
        prev_date = trading_dates[pos - 1] if pos > 0 else None
        is_trading_day = pos < len(trading_dates) and trading_dates[pos] == event_date
        after_close = event_time >= market_close
        if is_trading_day:
            if after_close:
                post_date = trading_dates[pos + 1] if pos + 1 < len(trading_dates) else None
            else:
                post_date = trading_dates[pos]
        else:
            post_date = trading_dates[pos] if pos < len(trading_dates) else None
        if post_date is None and is_trading_day:
            post_date = trading_dates[pos]

        prev_close = close_series.get(prev_date) if prev_date else None
        post_close = close_series.get(post_date) if post_date else None
        if prev_close is None or post_close is None or not prev_close:
            moves_amt.append(None)
            moves_pct.append(None)
            continue

        move_amt = float(post_close) - float(prev_close)
        move_pct = (move_amt / float(prev_close)) * 100
        moves_amt.append(move_amt)
        moves_pct.append(move_pct)

    df["Earnings Move %"] = moves_pct
    df["Earnings Move $"] = moves_amt
    return df


def _sector_metric_snapshot(
    key: str,
    base_value: Any,
    peers: list[dict[str, Any]],
    *,
    invert: bool = False,
) -> dict[str, Any] | None:
    """Percentile snapshot for one metric against peer values."""
    value = _safe_number(base_value)
    if value is None:
        return None
    peer_values = []
    for peer in peers:
        num = _safe_number(peer.get(key))
        if num is not None:
            peer_values.append(num)

    percentile = None
    if peer_values:
        better_or_equal = sum(1 for v in peer_values if (value <= v if invert else value >= v))
        percentile = better_or_equal / len(peer_values)

    return {
        "value": value,
        "peer_median": statistics.median(peer_values) if peer_values else None,
        "percentile": round(percentile * 100, 1) if percentile is not None else None,
        "direction": "lower_better" if invert else "higher_better",
    }


def _collect_sector_peers(sector: str, exclude: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Placeholder for sector peer collection.
    (Will be re-enabled once a Tiingo/EDGAR-backed peer universe is wired in.)
    """
    _ = (sector, exclude, limit)
    return []


def _build_sector_comparison(ticker: str, fundamentals: dict[str, Any], peer_limit: int = 20) -> dict[str, Any]:
    """
    Build a lightweight sector comparison snapshot and score for a ticker.
    Peers are currently disabled until a Tiingo/EDGAR peer universe is added.
    """
    baseline = dict(fundamentals or {})
    sector = str(baseline.get("sector") or "").strip()

    peers = _collect_sector_peers(sector, ticker, limit=peer_limit) if sector else []
    metrics: dict[str, dict[str, Any]] = {}
    for key, invert in (("pe_ratio", True), ("forward_pe", True), ("profit_margin", False), ("roe", False)):
        snap = _sector_metric_snapshot(key, baseline.get(key), peers, invert=invert)
        if snap:
            metrics[key] = snap

    score_inputs = [snap["percentile"] for snap in metrics.values() if snap and snap.get("percentile") is not None]
    sector_score = round(sum(score_inputs) / len(score_inputs), 1) if score_inputs else None

    return {
        "ticker": ticker.upper(),
        "sector": sector or None,
        "sector_score": sector_score,
        "peer_count": len(peers),
        "peers": peers,
        "metrics": metrics,
    }


def _news_for_ai(news: list[dict[str, Any]] | None, limit: int = 10) -> list[dict[str, Any]]:
    items = _normalize_news_items(news or [], limit=limit)
    condensed: list[dict[str, Any]] = []
    for item in items:
        entry = {
            "title": item.get("title"),
            "publisher": item.get("publisher"),
            "published_at": item.get("published_at"),
        }
        if item.get("summary"):
            entry["summary"] = str(item["summary"])[:320]
        condensed.append(entry)
    return condensed


def _build_ai_payload(
    ticker: str,
    timeframe: TF,
    prices: list[dict[str, Any]],
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sampled_prices = _thin_price_series(prices, max_points=120)
    stats = _price_stats(prices or sampled_prices)
    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "price_stats": stats,
        "sampled_prices": sampled_prices,
        "fundamentals": _filter_fundamentals(fundamentals),
        "recent_news": _news_for_ai(news),
    }


def _run_ai_analysis(
    ticker: str,
    timeframe: TF,
    prices: list[dict[str, Any]],
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _build_ai_payload(ticker, timeframe, prices, fundamentals, news)
    sampled_prices = payload["sampled_prices"]
    if not sampled_prices:
        raise RuntimeError("Price history missing; cannot analyze chart.")

    client = _openai_client()
    model = _AI_MODEL_DEFAULT or "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an equity analyst for an active trading desk. "
                "Be concise, flag risks, and avoid overconfidence."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the structured price action summary, fundamentals, and recent news headlines below, "
                "write succinct insights. If news is empty, ignore that section. "
                "Return JSON with keys: "
                "chart_analysis (2-3 sentences on trend/levels), "
                "fundamental_analysis (2-3 sentences on quality/valuation), "
                "rating (buy/sell/hold), "
                "rating_reason (one sentence supporting the rating), "
                "confidence (low/medium/high). "
                "Only return JSON."
                f"\n\nData:\n{json.dumps(payload, default=str)}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.25,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {}

    chart_text = parsed.get("chart_analysis") or parsed.get("price_analysis") or content
    fundamental_text = (
        parsed.get("fundamental_analysis")
        or parsed.get("fundamentals_analysis")
        or parsed.get("fundamental_view")
    )
    rating = (parsed.get("rating") or parsed.get("call") or "").lower() or None
    rating_reason = parsed.get("rating_reason") or parsed.get("reasoning") or parsed.get("rationale")
    confidence = parsed.get("confidence")

    return {
        "chart_analysis": chart_text,
        "fundamental_analysis": fundamental_text,
        "rating": rating,
        "rating_reason": rating_reason,
        "confidence": confidence,
        "model": getattr(resp, "model", model),
        "usage": _serialize_openai_usage(getattr(resp, "usage", None)),
        "raw": None if parsed else content,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "price_points_sampled": len(sampled_prices),
    }


def _run_ai_question(
    ticker: str,
    timeframe: TF,
    prices: list[dict[str, Any]],
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    payload = _build_ai_payload(ticker, timeframe, prices, fundamentals, news)
    sampled_prices = payload.get("sampled_prices") or []
    if not sampled_prices and not payload.get("fundamentals"):
        raise RuntimeError("No price or fundamentals data available for context.")

    client = _openai_client()
    model = _AI_MODEL_DEFAULT or "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an equity analyst for an active trading desk. "
                "Answer concisely using only the provided context. "
                "If data is missing, say so briefly instead of guessing."
            ),
        },
        {
            "role": "user",
            "content": (
                "Context (JSON) includes prices, fundamentals, and recent news headlines (if any):\n"
                f"{json.dumps(payload, default=str)}\n\n"
                f"Question: {question.strip()}\n\n"
                "Answer in plain text (max ~120 words). Avoid fabricating numbers not present in the context."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    content = (resp.choices[0].message.content or "").strip()
    return {
        "answer": content,
        "model": getattr(resp, "model", model),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "usage": _serialize_openai_usage(getattr(resp, "usage", None)),
        "price_points_sampled": len(sampled_prices),
    }


def _run_ai_news_analysis(ticker: str, news: list[dict[str, Any]], limit: int = 15) -> dict[str, Any]:
    condensed = _news_for_ai(news, limit=limit)
    if not condensed:
        raise RuntimeError("No headlines available for AI analysis.")

    client = _openai_client()
    model = _AI_MODEL_DEFAULT or "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an equity desk analyst. "
                "Summarize news without fabricating specifics not present in the headlines."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the recent headlines for the ticker below, produce a concise read. "
                "Return JSON with keys: summary (2-3 sentences), sentiment (bullish/bearish/neutral), "
                "themes (array of short phrases), risks (array of short phrases). "
                "Avoid numbers unless present in the headlines. "
                f"\n\nTicker: {ticker.upper()}\nHeadlines:\n{json.dumps(condensed, default=str)}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.25,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {"summary": content}

    summary = parsed.get("summary") or parsed.get("headline_summary") or content
    sentiment = (parsed.get("sentiment") or parsed.get("tone") or "").lower() or None

    def _to_list(val: Any) -> list[str]:
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    themes = _to_list(parsed.get("themes") or parsed.get("drivers") or parsed.get("takeaways"))
    risks = _to_list(parsed.get("risks") or parsed.get("watchouts") or parsed.get("catalysts"))

    return {
        "summary": summary,
        "sentiment": sentiment,
        "themes": themes,
        "risks": risks,
        "model": getattr(resp, "model", model),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "usage": _serialize_openai_usage(getattr(resp, "usage", None)),
    }


def _rating_to_sentiment(rating: str | None) -> str:
    if not rating:
        return "neutral"
    r = rating.lower()
    if "sell" in r or "bear" in r:
        return "bearish"
    if "buy" in r or "bull" in r or "long" in r:
        return "bullish"
    return "neutral"


def _latest_price_entry(rows: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    for row in reversed(rows or []):
        price = _safe_number(row.get("close") if "close" in row else row.get("price"))
        if price is None:
            continue
        ts = row.get("ts") or row.get("label")
        return price, ts
    return None, None


def _thin_index_series(rows: list[dict[str, Any]], tf: TF) -> list[dict[str, Any]]:
    max_points = 320 if tf == "1D" else 220
    return _thin_price_series(rows, max_points=max_points)


def _build_index_summary(
    spec: dict[str, str],
    analysis_tf: TF,
    news_limit: int,
    include_ai: bool,
    extended: bool,
) -> dict[str, Any]:
    tf_list: list[TF] = ["1D", "5D", "1M", analysis_tf]
    prices: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    price_sources: dict[str, str | None] = {}
    price_source = None
    pricing_ticker = spec.get("data_ticker") or spec["ticker"]

    for tf in tf_list:
        if tf in prices:
            continue
        series, src = _get_prices_with_fallback(pricing_ticker, tf, extended)
        prices[tf] = _thin_index_series(series, tf)
        stats[tf] = _price_stats(series)
        price_sources[tf] = src
        price_source = price_source or src

    news_items, news_source = _get_news_with_fallback(spec.get("news_symbol") or spec["ticker"], limit=news_limit)

    ai = None
    ai_error = None
    if include_ai:
        try:
            ai_prices = prices.get(analysis_tf) or []
            ai = _run_ai_analysis(pricing_ticker, analysis_tf, ai_prices, {}, news_items)
        except RuntimeError as exc:
            ai_error = str(exc)
        except Exception as exc:
            log.exception("AI index analysis failed for %s: %s", spec["ticker"], exc)
            ai_error = f"AI analysis failed: {exc}"

    latest_price, latest_ts = _latest_price_entry(prices.get("1D") or prices.get(analysis_tf) or [])
    sentiment = _rating_to_sentiment(ai.get("rating") if ai else None)

    return {
        "key": spec.get("key") or spec["ticker"],
        "name": spec["name"],
        "ticker": spec["ticker"],
        "pricing_ticker": pricing_ticker,
        "latest": {"price": latest_price, "as_of": latest_ts},
        "prices": prices,
        "price_source": price_source,
        "price_sources": price_sources,
        "stats": stats,
        "news": news_items,
        "news_source": news_source,
        "ai": ai,
        "sentiment": sentiment,
        "ai_error": ai_error,
    }


@router.get("/market/indices")
def market_indices(
    analysis_tf: TF = Query("6M", description="Timeframe used for AI analysis and trend stats."),
    news_limit: int = Query(6, ge=1, le=25, description="Number of headlines to pull per index."),
    include_ai: bool = Query(True, description="Include OpenAI bull/bear commentary."),
    extended: bool = Query(True, description="Include pre/post market in 1D series when available."),
    cache_only: bool = Query(False, description="Return cached market indices only; skip live fetch."),
):
    """
    Aggregate market context for the four major US indices with price trends, headlines, and AI sentiment.
    """
    if cache_only:
        cached = _read_market_indices_cache()
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail="No cached market indices available. Run the cache refresh job.",
            )
        cached["cached"] = True
        cached["cache_file"] = _MARKET_INDICES_CACHE_PATH.name
        return cached

    return _build_market_indices_payload(analysis_tf, news_limit, include_ai, extended)


@router.post("/ticker/ai-analysis")
def ticker_ai_analysis(req: TickerAiAnalysisRequest):
    prices = req.prices or []
    price_source = "client" if req.prices else None
    if not prices:
        prices, price_source = _get_prices_with_fallback(req.ticker, req.timeframe, req.extended)

    fundamentals = req.fundamentals or {}
    fundamentals_source = "client" if req.fundamentals else None
    if not fundamentals:
        price_hint = _latest_price_from_series({req.timeframe: prices})
        fundamentals, fundamentals_source = _get_fundamentals(req.ticker, price_hint=price_hint)

    news_items = req.news
    news_source = "client" if req.news is not None else None
    if news_items is None:
        news_items, news_source = _get_news_with_fallback(req.ticker, limit=25)
    else:
        news_items = _normalize_news_items(news_items, limit=25)

    try:
        ai = _run_ai_analysis(req.ticker, req.timeframe, prices, fundamentals, news_items)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("AI analysis failed for %s: %s", req.ticker, exc)
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {exc}")

    return {
        "ticker": req.ticker.upper(),
        "timeframe": req.timeframe,
        "chart_analysis": ai.get("chart_analysis"),
        "fundamental_analysis": ai.get("fundamental_analysis"),
        "rating": ai.get("rating"),
        "rating_reason": ai.get("rating_reason"),
        "confidence": ai.get("confidence"),
        "price_points_sampled": ai.get("price_points_sampled"),
        "price_source": price_source,
        "fundamentals_source": fundamentals_source,
        "news_source": news_source,
        "model": ai.get("model") or _AI_MODEL_DEFAULT,
        "generated_at": ai.get("generated_at"),
        "usage": ai.get("usage"),
        "raw": ai.get("raw"),
    }


@router.post("/ticker/news-analysis")
def ticker_news_analysis(req: TickerNewsAiRequest):
    news_items = req.news
    news_source = "client" if req.news is not None else None
    if news_items is None:
        news_items, news_source = _get_news_with_fallback(req.ticker, limit=req.limit)
    else:
        news_items = _normalize_news_items(news_items, limit=req.limit)

    try:
        ai = _run_ai_news_analysis(req.ticker, news_items, limit=req.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("AI news analysis failed for %s: %s", req.ticker, exc)
        raise HTTPException(status_code=502, detail=f"AI news analysis failed: {exc}")

    return {
        "ticker": req.ticker.upper(),
        "summary": ai.get("summary"),
        "sentiment": ai.get("sentiment"),
        "themes": ai.get("themes"),
        "risks": ai.get("risks"),
        "model": ai.get("model") or _AI_MODEL_DEFAULT,
        "generated_at": ai.get("generated_at"),
        "news_source": news_source,
        "usage": ai.get("usage"),
    }


@router.post("/ticker/ai-question")
def ticker_ai_question(req: TickerAiQuestionRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    prices = req.prices or []
    price_source = "client" if req.prices else None
    if not prices:
        prices, price_source = _get_prices_with_fallback(req.ticker, req.timeframe, req.extended)

    fundamentals = req.fundamentals or {}
    fundamentals_source = "client" if req.fundamentals else None
    if not fundamentals:
        price_hint = _latest_price_from_series({req.timeframe: prices})
        fundamentals, fundamentals_source = _get_fundamentals(req.ticker, price_hint=price_hint)

    news_items = req.news
    news_source = "client" if req.news is not None else None
    if news_items is None:
        news_items, news_source = _get_news_with_fallback(req.ticker, limit=25)
    else:
        news_items = _normalize_news_items(news_items, limit=25)

    try:
        ai = _run_ai_question(req.ticker, req.timeframe, prices, fundamentals, news_items, question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("AI question failed for %s: %s", req.ticker, exc)
        raise HTTPException(status_code=502, detail=f"AI question failed: {exc}")

    return {
        "ticker": req.ticker.upper(),
        "timeframe": req.timeframe,
        "question": question,
        "answer": ai.get("answer"),
        "price_points_sampled": ai.get("price_points_sampled"),
        "price_source": price_source,
        "fundamentals_source": fundamentals_source,
        "news_source": news_source,
        "model": ai.get("model") or _AI_MODEL_DEFAULT,
        "generated_at": ai.get("generated_at"),
        "usage": ai.get("usage"),
    }


@router.get("/ticker/sector-comparison")
def ticker_sector_comparison(
    ticker: str = Query(..., min_length=1, description="Ticker symbol, e.g., AAPL"),
    max_peers: int = Query(
        20,
        ge=5,
        le=60,
        description="Maximum number of peers to sample from the S&P 500 list.",
    ),
):
    """
    Compare a ticker's valuation/quality metrics against same-sector peers.
    """
    fundamentals, fund_source = _get_fundamentals(ticker, price_hint=None, include_filings=False)
    comparison = _build_sector_comparison(ticker, fundamentals, peer_limit=max_peers)
    return {
        **comparison,
        "fundamentals_source": fund_source,
    }


@router.get("/prices")
def prices(
    ticker: str = Query(..., min_length=1),
    tf: TF = Query("1D"),
    extended: bool = Query(True, description="Include pre/post market in 1D if available"),
):
    """
    Examples:
      /api/prices?ticker=NVDA&tf=1D
      /api/prices?ticker=NVDA&tf=5D
      /api/prices?ticker=NVDA&tf=1M
      /api/prices?ticker=NVDA&tf=6M
      /api/prices?ticker=NVDA&tf=1Y
      /api/prices?ticker=NVDA&tf=5Y
      /api/prices?ticker=NVDA&tf=10Y
    """
    rows, _src = _get_prices_with_fallback(ticker, tf, extended)
    return JSONResponse(rows)


@router.get("/quotes")
def quotes(
    symbols: str = Query(..., min_length=1, description="Comma-separated symbols (equities or options)."),
    indicative: bool = Query(False, description="Include indicative ETF symbol quotes (e.g., $ABC.IV)."),
):
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Symbols cannot be empty.")
    try:
        data = get_latest_prices(symbol_list)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Tiingo quotes failed: {exc}")
    quotes_payload = _normalize_quotes_tiingo(data)
    requested = {s.upper() for s in symbol_list}
    available = {str(item.get("symbol", "")).upper() for item in quotes_payload if item.get("symbol")}
    missing = sorted(requested - available)
    return {"count": len(quotes_payload), "quotes": quotes_payload, "missing": missing}


@router.get("/ticker/overview")
def ticker_overview(
    ticker: str = Query(..., min_length=1, description="Ticker symbol, e.g., AAPL"),
    extended: bool = Query(True, description="Include pre/post market in 1D if available"),
):
    """
    Return multi-timeframe prices + fundamentals for a ticker.
    Fundamentals are pulled from SEC EDGAR (no fallback).
    """
    tf_order: list[TF] = ["1D", "5D", "1M", "6M", "1Y", "5Y", "10Y"]
    prices: dict[str, list[dict[str, Any]]] = {}
    price_source = None
    for tf in tf_order:
        rows, src = _get_prices_with_fallback(ticker, tf, extended)
        prices[tf] = rows
        price_source = price_source or src

    price_hint = _latest_price_from_series(prices)
    fundamentals, fund_source = _get_fundamentals(ticker, price_hint=price_hint)
    news_items, news_source = _get_news_with_fallback(ticker, limit=40)

    return {
        "ticker": ticker.upper(),
        "prices": prices,
        "price_source": price_source,
        "fundamentals": fundamentals,
        "fundamentals_source": fund_source,
        "news": news_items,
        "news_source": news_source,
    }


@router.get("/ticker/fundamentals-detail")
def ticker_fundamentals_detail(
    ticker: str = Query(..., min_length=1, description="Ticker symbol, e.g., AAPL"),
):
    """
    Return balance sheets, income statements, cashflow, and key dates for a ticker.
    Fundamentals snapshot comes from SEC EDGAR.
    """
    symbol = ticker.strip().upper()
    fundamentals, fund_source = _get_fundamentals(symbol, price_hint=None)
    trading_info = _filter_trading_info(_tiingo_trading_info(symbol))

    annual_balance = _df_to_split_payload(_tiingo_statement_frame(symbol, "balanceSheet", period="annual"))
    quarterly_balance = _df_to_split_payload(_tiingo_statement_frame(symbol, "balanceSheet", period="quarterly"))
    annual_income = _df_to_split_payload(_tiingo_statement_frame(symbol, "incomeStatement", period="annual"))
    quarterly_income = _df_to_split_payload(_tiingo_statement_frame(symbol, "incomeStatement", period="quarterly"))
    annual_cashflow = _df_to_split_payload(_tiingo_statement_frame(symbol, "cashFlow", period="annual"))
    quarterly_cashflow = _df_to_split_payload(_tiingo_statement_frame(symbol, "cashFlow", period="quarterly"))
    earnings_dates = []
    shares_history = _df_to_records_payload(_tiingo_shares_history(symbol), row_limit=16)

    return {
        "ticker": symbol,
        "fundamentals": _filter_fundamentals(fundamentals),
        "fundamentals_source": fund_source,
        "trading_info": trading_info,
        "balance_sheet": {
            "annual": annual_balance,
            "quarterly": quarterly_balance,
        },
        "income_statement": {
            "annual": annual_income,
            "quarterly": quarterly_income,
        },
        "cashflow": {
            "annual": annual_cashflow,
            "quarterly": quarterly_cashflow,
        },
        "earnings_dates": earnings_dates,
        "shares_history": shares_history,
    }
