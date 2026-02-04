# backend/core/holdings_analysis.py
"""
Lightweight portfolio helper that scores a set of holdings and suggests
whether to buy more, hold, or sell based on trend, momentum, and P&L.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import pandas as pd

from backend.core.yfinance_api import yf_service as yfs


@dataclass
class Holding:
    ticker: str
    shares: Optional[float] = None
    cost_basis: Optional[float] = None


def _to_float(val: Any) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else math.nan
    except Exception:
        return math.nan


def _clean_ticker(raw: str) -> str:
    return (raw or "").strip().upper()


def _close_series(df: Any) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    candidates = ["Close", "close", "Adj Close", "adjclose", "AdjClose", "Adj_Close"]
    for col in candidates:
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            try:
                ser.index = pd.to_datetime(df.index)
            except Exception:
                pass
            return ser.sort_index()
    try:
        ser = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        ser.index = pd.to_datetime(df.index)
        return ser.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _latest_price(info: dict, closes: pd.Series) -> float:
    for key in (
        "last_price",
        "lastPrice",
        "regularMarketPrice",
        "regular_market_price",
        "previousClose",
        "previous_close",
        "regularMarketPreviousClose",
    ):
        val = _to_float(info.get(key)) if isinstance(info, dict) else math.nan
        if math.isfinite(val) and val > 0:
            return val
    if closes is not None and not closes.empty:
        last = _to_float(closes.iloc[-1])
        if math.isfinite(last) and last > 0:
            return last
    return math.nan


def _moving_average(ser: pd.Series, window: int) -> float:
    if ser is None or ser.empty or window <= 0:
        return math.nan
    trimmed = ser.dropna()
    if len(trimmed) < max(5, window // 3):
        return math.nan
    return _to_float(trimmed.tail(window).mean())


def _window_return(ser: pd.Series, days: int) -> float:
    if ser is None or ser.empty or days <= 0:
        return math.nan
    trimmed = ser.dropna()
    if len(trimmed) <= days:
        return math.nan
    start = _to_float(trimmed.iloc[-days - 1])
    end = _to_float(trimmed.iloc[-1])
    if not math.isfinite(start) or start == 0 or not math.isfinite(end):
        return math.nan
    return (end - start) / start * 100.0


def _fifty_two_week_range(closes: pd.Series, info: dict) -> tuple[float, float]:
    low_keys = ("yearLow", "year_low", "fiftyTwoWeekLow", "fifty_two_week_low")
    high_keys = ("yearHigh", "year_high", "fiftyTwoWeekHigh", "fifty_two_week_high")
    low = math.nan
    high = math.nan
    if isinstance(info, dict):
        for key in low_keys:
            cand = _to_float(info.get(key))
            if math.isfinite(cand):
                low = cand
                break
        for key in high_keys:
            cand = _to_float(info.get(key))
            if math.isfinite(cand):
                high = cand
                break
    if (not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= 0) and closes is not None:
        lookback_start = closes.index.max() - timedelta(days=365) if isinstance(closes.index, pd.DatetimeIndex) else None
        last_year = closes if lookback_start is None else closes[closes.index >= lookback_start]
        if last_year is not None and not last_year.empty:
            low = _to_float(last_year.min()) if not math.isfinite(low) else low
            high = _to_float(last_year.max()) if not math.isfinite(high) else high
    if math.isfinite(low) and math.isfinite(high) and high > low:
        return low, high
    return math.nan, math.nan


def _position_in_range(price: float, low: float, high: float) -> float:
    if not all(math.isfinite(v) for v in (price, low, high)) or high <= low or price <= 0:
        return math.nan
    return max(0.0, min(1.0, (price - low) / (high - low)))


def _calc_recommendation(payload: dict) -> tuple[str, list[str], float]:
    """
    Return (recommendation, reasons, score [-1..1])
    """
    reasons: list[str] = []
    score = 0.0
    price = _to_float(payload.get("current_price"))
    change_pct = _to_float(payload.get("unrealized_pl_pct"))
    ret_1m = _to_float(payload.get("one_month_return"))
    ret_3m = _to_float(payload.get("three_month_return"))
    pos_52w = _to_float(payload.get("fifty_two_week_position"))
    sma_short = _to_float(payload.get("fifty_day_sma"))
    sma_long = _to_float(payload.get("two_hundred_day_sma"))

    trend_up = math.isfinite(sma_short) and math.isfinite(sma_long) and sma_short > sma_long * 1.01
    trend_down = math.isfinite(sma_short) and math.isfinite(sma_long) and sma_short < sma_long * 0.99
    strong_gain = math.isfinite(change_pct) and change_pct > 35
    deep_drawdown = math.isfinite(change_pct) and change_pct < -18
    momentum_positive = any(math.isfinite(v) and v > 0 for v in (ret_1m, ret_3m))
    momentum_negative = any(math.isfinite(v) and v < -5 for v in (ret_1m, ret_3m))

    if trend_up:
        score += 0.35
        reasons.append("Price above 200d trend")
    elif trend_down:
        score -= 0.35
        reasons.append("Below long-term trend")

    if math.isfinite(ret_3m):
        score += max(-0.35, min(0.35, ret_3m / 40))
    if math.isfinite(ret_1m):
        score += max(-0.25, min(0.25, ret_1m / 20))

    if math.isfinite(pos_52w):
        if pos_52w > 0.9:
            score -= 0.05
            reasons.append("Near 52w high")
        elif pos_52w < 0.25:
            score += 0.05
            reasons.append("Near 52w low")

    if math.isfinite(change_pct):
        if change_pct > 20:
            reasons.append("Already up meaningfully from cost")
            score -= 0.05
        if change_pct < -12:
            score -= 0.1 if trend_down else 0.0
            reasons.append("Down vs cost basis")

    score = max(-1.0, min(1.0, score))

    if not math.isfinite(price):
        return "hold", ["Price data unavailable"], score

    if strong_gain and (momentum_negative or pos_52w > 0.9):
        reasons.append("Locking gains after a strong run")
        return "sell", reasons, score
    if deep_drawdown and (trend_down or momentum_negative):
        reasons.append("Protecting capital during drawdown")
        return "sell", reasons, score
    if trend_up and momentum_positive and (not math.isfinite(change_pct) or change_pct < 20):
        reasons.append("Uptrend intact with positive momentum")
        return "buy_more", reasons, score
    if trend_up:
        reasons.append("Uptrend but extended")
        return "hold", reasons, score
    if momentum_negative and change_pct < -8:
        reasons.append("Weak momentum and losses vs cost basis")
        return "sell", reasons, score
    return "hold", reasons, score


def analyze_holding(holding: Holding, *, price_period: str = "1y") -> dict[str, Any]:
    ticker = _clean_ticker(holding.ticker)
    if not ticker:
        raise ValueError("Ticker is required for each holding.")

    info = yfs.get_company_info(ticker, full=False) or {}
    history = yfs.get_historical_stock_price(ticker, period=price_period, interval="1d")
    closes = _close_series(history)

    price = _latest_price(info, closes)
    prev_close = _to_float(info.get("previousClose") if isinstance(info, dict) else math.nan)
    if not math.isfinite(prev_close) and closes is not None and not closes.empty and len(closes) >= 2:
        prev_close = _to_float(closes.iloc[-2])

    sma_50 = _moving_average(closes, 50)
    sma_200 = _moving_average(closes, 200)
    ret_1m = _window_return(closes, 21)
    ret_3m = _window_return(closes, 63)
    low_52w, high_52w = _fifty_two_week_range(closes, info)
    pos_52w = _position_in_range(price, low_52w, high_52w)
    last_ts = closes.index.max() if isinstance(closes.index, pd.DatetimeIndex) and not closes.empty else None

    shares = _to_float(holding.shares) if holding.shares is not None else math.nan
    cost = _to_float(holding.cost_basis) if holding.cost_basis is not None else math.nan
    market_value = shares * price if math.isfinite(shares) and math.isfinite(price) else math.nan
    cost_value = shares * cost if math.isfinite(shares) and math.isfinite(cost) else math.nan
    unrealized_pl = market_value - cost_value if math.isfinite(market_value) and math.isfinite(cost_value) else math.nan
    unrealized_pl_pct = (
        (price - cost) / cost * 100.0 if math.isfinite(price) and math.isfinite(cost) and cost > 0 else math.nan
    )

    rec, reasons, score = _calc_recommendation(
        {
            "current_price": price,
            "unrealized_pl_pct": unrealized_pl_pct,
            "one_month_return": ret_1m,
            "three_month_return": ret_3m,
            "fifty_two_week_position": pos_52w,
            "fifty_day_sma": sma_50,
            "two_hundred_day_sma": sma_200,
        }
    )

    name = ""
    if isinstance(info, dict):
        name = info.get("shortName") or info.get("longName") or ""
    payload = {
        "ticker": ticker,
        "name": name,
        "shares": shares if math.isfinite(shares) else None,
        "cost_basis": cost if math.isfinite(cost) else None,
        "current_price": price if math.isfinite(price) else None,
        "previous_close": prev_close if math.isfinite(prev_close) else None,
        "market_value": market_value if math.isfinite(market_value) else None,
        "cost_value": cost_value if math.isfinite(cost_value) else None,
        "unrealized_pl": unrealized_pl if math.isfinite(unrealized_pl) else None,
        "unrealized_pl_pct": unrealized_pl_pct if math.isfinite(unrealized_pl_pct) else None,
        "one_month_return": ret_1m if math.isfinite(ret_1m) else None,
        "three_month_return": ret_3m if math.isfinite(ret_3m) else None,
        "fifty_day_sma": sma_50 if math.isfinite(sma_50) else None,
        "two_hundred_day_sma": sma_200 if math.isfinite(sma_200) else None,
        "fifty_two_week_low": low_52w if math.isfinite(low_52w) else None,
        "fifty_two_week_high": high_52w if math.isfinite(high_52w) else None,
        "fifty_two_week_position": pos_52w if math.isfinite(pos_52w) else None,
        "recommendation": rec,
        "score": score,
        "rationale": "; ".join(reasons) if reasons else "",
        "data_as_of": last_ts,
        "currency": info.get("currency") if isinstance(info, dict) else None,
    }
    return payload


def analyze_holdings(holdings: Iterable[dict | Holding], *, price_period: str = "1y") -> dict[str, Any]:
    normalized: list[Holding] = []
    for raw in holdings:
        if isinstance(raw, Holding):
            normalized.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        ticker = _clean_ticker(raw.get("ticker", ""))
        if not ticker:
            continue
        normalized.append(
            Holding(
                ticker=ticker,
                shares=raw.get("shares"),
                cost_basis=raw.get("cost_basis") or raw.get("cost") or raw.get("basis"),
            )
        )

    if not normalized:
        raise ValueError("At least one holding with a ticker is required.")

    analyses = [analyze_holding(h, price_period=price_period) for h in normalized]

    cost_values = [a["cost_value"] for a in analyses if a.get("cost_value") is not None]
    value_values = [a["market_value"] for a in analyses if a.get("market_value") is not None]
    total_cost = sum(cost_values) if cost_values else math.nan
    total_value = sum(value_values) if value_values else math.nan
    pl = total_value - total_cost if math.isfinite(total_cost) and math.isfinite(total_value) else math.nan
    pl_pct = (pl / total_cost * 100.0) if math.isfinite(pl) and math.isfinite(total_cost) and total_cost > 0 else math.nan

    summary = {
        "as_of": datetime.now(timezone.utc),
        "count": len(analyses),
        "total_cost": total_cost if math.isfinite(total_cost) else None,
        "total_value": total_value if math.isfinite(total_value) else None,
        "total_unrealized_pl": pl if math.isfinite(pl) else None,
        "total_unrealized_pl_pct": pl_pct if math.isfinite(pl_pct) else None,
    }

    return {"summary": summary, "holdings": analyses}
