# backend/core/horizon_buy_suggesters.py
"""
Horizon-specific buy suggestion scripts.

This module defines three opinionated screeners that share the same data
plumbing as ``daily_buy_suggester`` but tune the filters and sizing for
distinct holding windows:

- 1-2 day momentum (short-term breakout sizing reused from the daily suggester)
- 3-4 month swing (medium-term trend/momentum with wider stops/targets)
- 1+ year compounder (long-term trend and stability bias)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from loguru import logger
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.core.daily_buy_suggester import _read_default_tickers, compute_features, generate_daily_buy_suggestions, TiingoMarketDataProvider
from backend.core.suggester_cancel import CancelToken
from backend.core.edgar_fundamentals import get_edgar_fundamentals
from backend.platform_apis.rss_feed_api.rss_feed import get_company_news


def _normalize_key_value(series: pd.Series, *, low: float, high: float, invert: bool = False) -> pd.Series:
    """
    Normalize a series into [0,1] given soft bounds. Optionally invert to reward lower values.
    """
    span = max(high - low, 1e-9)
    scaled = (series - low) / span
    clipped = scaled.clip(0.0, 1.0)
    return 1.0 - clipped if invert else clipped


def _load_history(
    tickers: Optional[list[str]],
    years: int,
    *,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Shared data loader for Tiingo-backed OHLCV history.
    """
    universe = tickers or _read_default_tickers()
    if not universe:
        raise ValueError("No tickers provided or found in default list.")

    provider = TiingoMarketDataProvider()
    return provider.history(universe, period=f"{years}y", cancel_token=cancel_token)


def _to_float(value: Any) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def _maybe_cancel(cancel_token: Optional[CancelToken]) -> None:
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()


def _parse_timestamp(val: Any) -> Optional[pd.Timestamp]:
    """
    Parse a variety of timestamp formats (epoch sec/ms, ISO strings).
    """
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    if isinstance(val, (int, np.integer, float)):
        try:
            unit = "ms" if val > 1e12 else "s"
            return pd.to_datetime(val, unit=unit, utc=True)
        except Exception:
            pass
    try:
        return pd.to_datetime(val, utc=True)
    except Exception:
        return None


def _statement_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """
    Get a numeric, date-indexed series from a financial statement row.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in candidates:
        if name in df.index:
            ser = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if ser.empty:
                continue
            ser.index = pd.to_datetime(ser.index, errors="coerce")
            ser = ser.dropna().sort_index()
            if not ser.empty:
                return ser
    return pd.Series(dtype=float)


def _ttm_growth(series: pd.Series) -> float:
    """
    TTM vs prior TTM growth for quarterly data.
    """
    if series is None or series.empty or len(series) < 8:
        return np.nan
    vals = series.sort_index().values
    recent = np.nansum(vals[-4:])
    prev = np.nansum(vals[-8:-4])
    if prev == 0 or not np.isfinite(prev):
        return np.nan
    return (recent - prev) / abs(prev)


def _yoy_change(series: pd.Series, lag: int = 4) -> float:
    if series is None or series.empty or len(series) <= lag:
        return np.nan
    ser = series.sort_index()
    new = ser.iloc[-1]
    old = ser.iloc[-lag]
    if old == 0 or not np.isfinite(old):
        return np.nan
    return (new - old) / abs(old)


def _gross_margin_series(income_df: pd.DataFrame) -> pd.Series:
    gross = _statement_series(income_df, ("Gross Profit", "GrossProfit"))
    revenue = _statement_series(income_df, ("Total Revenue", "TotalRevenue", "Revenues", "Revenue"))
    if gross.empty or revenue.empty:
        return pd.Series(dtype=float)
    aligned = pd.concat([gross, revenue], axis=1, join="inner")
    aligned.columns = ["gross", "revenue"]
    aligned = aligned[(aligned["revenue"] != 0) & aligned["revenue"].notna()]
    if aligned.empty:
        return pd.Series(dtype=float)
    margins = (aligned["gross"] / aligned["revenue"]).dropna()
    return margins.sort_index()


def _fcf_series(cash_df: pd.DataFrame) -> pd.Series:
    ocf = _statement_series(cash_df, ("Operating Cash Flow", "Total Cash From Operating Activities"))
    capex = _statement_series(cash_df, ("Capital Expenditures", "Capital Expenditure"))
    if ocf.empty and capex.empty:
        return pd.Series(dtype=float)
    if ocf.empty:
        ocf = pd.Series(0.0, index=capex.index)
    if capex.empty:
        capex = pd.Series(0.0, index=ocf.index)
    capex = capex.reindex(ocf.index, fill_value=0.0)
    return (ocf + capex).sort_index()


def _shares_dilution_rate(shares_df: pd.DataFrame) -> float:
    if shares_df is None or shares_df.empty:
        return np.nan
    ser = pd.to_numeric(shares_df.iloc[:, 0], errors="coerce").dropna().sort_index()
    if ser.empty:
        return np.nan
    start = ser.iloc[0]
    end = ser.iloc[-1]
    if start <= 0:
        return np.nan
    return (end - start) / start


def _short_term_catalyst_snapshot(
    ticker: str,
    *,
    news_lookback_days: int = 3,
    upgrade_lookback_days: int = 7,
    earnings_lookahead_days: int = 2,
    news_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Lightweight news + event pulse for short-term trades.
    (RSS feed placeholder until wired.)
    """
    now = pd.Timestamp.now(tz="UTC")
    score = 0.0
    reasons: set[str] = set()

    news_items = get_company_news(ticker, limit=50)
    for row in news_items:
        if not isinstance(row, dict):
            continue
        ts = _parse_timestamp(row.get("published_at") or row.get("time") or row.get("created"))
        if ts is None or (now - ts) > pd.Timedelta(days=news_lookback_days):
            continue
        text = " ".join(
            str(row.get(k, "")) for k in ("title", "summary", "description", "content", "text")
        ).lower()
        if not text.strip():
            continue
        base = 0.15
        if any(k in text for k in ("earnings", "surprise", "guidance")):
            base += 0.25
            reasons.add("Earnings surprise/headline")
        if any(k in text for k in ("upgrade", "overweight", "initiated")):
            base += 0.30
            reasons.add("Upgrade/analyst move")
        if any(k in text for k in ("downgrade", "underweight")):
            base -= 0.20
        if any(k in text for k in ("fda", "phase", "trial", "approval", "regulator")):
            base += 0.30
            reasons.add("FDA/regulatory")
        if any(k in text for k in ("merger", "acquisition", "m&a", "buyout", "takeover", "rumor")):
            base += 0.35
            reasons.add("M&A rumor/headline")
        score += max(base, 0.0)

    upgrade_hits = 0
    downgrade_hits = 0
    earnings_within = False
    next_earnings = None
    last_surprise = np.nan

    return {
        "news_score": score,
        "has_fresh_news": score >= news_threshold,
        "upgrade_hits": upgrade_hits,
        "downgrade_hits": downgrade_hits,
        "earnings_within_window": earnings_within,
        "next_earnings_date": next_earnings,
        "last_earnings_surprise_pct": last_surprise,
        "catalyst_reasons": sorted(reasons),
    }


def _fundamental_snapshot(ticker: str) -> dict[str, Any]:
    """
    Shared fundamentals snapshot used by swing and long-term screens.
    """
    data, _ = get_edgar_fundamentals(ticker, price_hint=None, include_filings=False)
    if not data:
        return {
            "sector": "",
            "revenue_growth": np.nan,
            "eps_growth": np.nan,
            "revenue_ttm": np.nan,
            "gross_margin": np.nan,
            "gross_margin_trend": np.nan,
            "fcf_ttm": np.nan,
            "fcf_margin": np.nan,
            "fcf_yield": np.nan,
            "net_debt": np.nan,
            "net_debt_to_ebitda": np.nan,
            "ps_ratio": np.nan,
            "ev_to_ebitda": np.nan,
            "inst_percent": np.nan,
            "inst_trending_up": np.nan,
            "dilution_rate": np.nan,
        }

    inst_percent = _to_float(data.get("inst_percent"))
    dilution_rate = _to_float(data.get("dilution_rate"))
    inst_trending_up = np.nan
    if np.isfinite(inst_percent):
        inst_trending_up = inst_percent >= 0.2 and (np.isnan(dilution_rate) or dilution_rate <= 0.05)

    return {
        "sector": data.get("sector") or "",
        "revenue_growth": _to_float(data.get("revenue_growth")),
        "eps_growth": _to_float(data.get("eps_growth")),
        "revenue_ttm": _to_float(data.get("revenue_ttm")),
        "gross_margin": _to_float(data.get("gross_margin")),
        "gross_margin_trend": _to_float(data.get("gross_margin_trend")),
        "fcf_ttm": _to_float(data.get("fcf_ttm")),
        "fcf_margin": _to_float(data.get("fcf_margin")),
        "fcf_yield": _to_float(data.get("fcf_yield")),
        "net_debt": _to_float(data.get("net_debt")),
        "net_debt_to_ebitda": _to_float(data.get("net_debt_to_ebitda")),
        "ps_ratio": _to_float(data.get("ps_ratio")),
        "ev_to_ebitda": _to_float(data.get("ev_to_ebitda")),
        "inst_percent": inst_percent,
        "inst_trending_up": inst_trending_up,
        "dilution_rate": dilution_rate,
    }


def _empty_payload(horizon: str) -> pd.DataFrame:
    # Build a truly empty frame with the expected columns; avoid length mismatches.
    return pd.DataFrame({"ticker": [], "action": [], "horizon": [], "thesis": []})


def _attach_common_fields(plan: pd.DataFrame, *, horizon: str, thesis: str) -> pd.DataFrame:
    """
    Normalize the output shape and add UI-friendly columns.
    """
    if plan.empty:
        return _empty_payload(horizon)

    out = plan.copy()
    out.index.name = "ticker"
    out = out.reset_index()
    out["ticker"] = out["ticker"].fillna("").astype(str)
    out["action"] = "Buy"
    out["horizon"] = horizon
    out["thesis"] = thesis
    out["conviction"] = out.get("score", pd.Series(index=out.index, dtype=float)).apply(
        lambda s: "High" if s >= 0.75 else "Medium" if s >= 0.45 else "Low"
    )
    preferred_cols = [
        "ticker",
        "action",
        "horizon",
        "thesis",
        "entry",
        "target",
        "stop",
        "conviction",
        "score",
        "close",
    ]
    remaining = [c for c in out.columns if c not in preferred_cols]
    return out[preferred_cols + remaining]


@dataclass
class SharedSuggesterData:
    """
    Shared artifacts for running multiple horizons in one go.
    """
    history: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    fundamentals: dict[str, dict[str, Any]] = field(default_factory=dict)
    long_addons: Optional[pd.DataFrame] = None
    feature_lookback: Optional[int] = None


def build_shared_suggester_data(
    tickers: Optional[list[str]] = None,
    *,
    years: int = 5,
    feature_lookback: Optional[int] = 252,
    include_spy: bool = True,
    cancel_token: Optional[CancelToken] = None,
) -> SharedSuggesterData:
    """
    Precompute history, features, fundamentals, and long-term addons once to reuse across horizons.
    """
    base = tickers or _read_default_tickers()
    if not base:
        raise ValueError("No tickers provided or found in default list.")

    universe = set(base)
    if include_spy:
        universe.add("SPY")
    logger.debug(f"Building shared suggester data")

    _maybe_cancel(cancel_token)
    history = _load_history(sorted(universe), years=years, cancel_token=cancel_token)
    _maybe_cancel(cancel_token)
    features = compute_features(history, high_lookback=feature_lookback, cancel_token=cancel_token)
    _maybe_cancel(cancel_token)
    long_addons = _long_horizon_addons(history, cancel_token=cancel_token)
    fundamentals: dict[str, dict[str, Any]] = {}
    for t in features.index:
        _maybe_cancel(cancel_token)
        if t == "SPY":
            continue
        fundamentals[t] = _fundamental_snapshot(t)

    return SharedSuggesterData(
        history=history,
        features=features,
        fundamentals=fundamentals,
        long_addons=long_addons,
        feature_lookback=feature_lookback,
    )


@dataclass
class ShortTermConfig:
    news_lookback_days: int = 3
    upgrade_lookback_days: int = 7
    earnings_lookahead_days: int = 2
    news_threshold: float = 0.6
    min_intraday_vol_ratio: float = 1.3
    require_news: bool = True
    exclude_near_earnings: bool = True


def generate_short_term_suggestions(
    tickers: Optional[list[str]] = None,
    *,
    period: str = "1y",
    cfg_overrides: Optional[dict[str, Any]] = None,
    shared_data: Optional[SharedSuggesterData] = None,
    feature_lookback: Optional[int] = None,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Thin wrapper around the existing daily momentum suggester for 1-2 day holds.
    """
    st_cfg = ShortTermConfig()
    if cfg_overrides:
        valid = {f.name for f in fields(ShortTermConfig)}
        for k, v in cfg_overrides.items():
            if k in valid:
                setattr(st_cfg, k, v)

    _maybe_cancel(cancel_token)
    df = generate_daily_buy_suggestions(
        tickers=tickers,
        period=period,
        cfg_overrides=cfg_overrides,
        shared_data=shared_data,
        feature_lookback=feature_lookback,
        cancel_token=cancel_token,
    )
    logger.debug(f"generate_short_term_suggestions df:\n{df}")
    if df.empty:
        return df
    df = df.copy()
    extras: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        _maybe_cancel(cancel_token)
        ticker = row.get("ticker")
        if not ticker:
            continue
        snap = _short_term_catalyst_snapshot(
            ticker,
            news_lookback_days=st_cfg.news_lookback_days,
            upgrade_lookback_days=st_cfg.upgrade_lookback_days,
            earnings_lookahead_days=st_cfg.earnings_lookahead_days,
            news_threshold=st_cfg.news_threshold,
        )
        snap["ticker"] = ticker
        extras.append(snap)

    if extras:
        catalyst_df = pd.DataFrame(extras).set_index("ticker")
        df = df.set_index("ticker").join(catalyst_df, how="left").reset_index()
    df["intraday_rel_vol"] = df.get("vol_ratio20")

    if st_cfg.require_news and "has_fresh_news" in df.columns:
        df = df[df["has_fresh_news"].fillna(False)]
    if st_cfg.exclude_near_earnings and "earnings_within_window" in df.columns:
        df = df[~df["earnings_within_window"].fillna(False)]
    if st_cfg.min_intraday_vol_ratio is not None:
        df = df[df["intraday_rel_vol"].fillna(0) >= st_cfg.min_intraday_vol_ratio]

    if df.empty:
        return _empty_payload("1-2 day")

    if "news_score" in df.columns:
        df["score"] = df.get("score", pd.Series(index=df.index, dtype=float)).fillna(0.0) + 0.1 * _normalize_key_value(
            df["news_score"], low=0.0, high=2.0
        )
    df["horizon"] = "1-2 day"
    catalyst_notes = (
        df.get("catalyst_reasons", pd.Series(index=df.index, dtype=object))
        .apply(lambda x: ", ".join(x) if isinstance(x, (list, tuple, set)) else "")
        .fillna("")
    )
    df["thesis"] = "Momentum + liquidity with fresh catalysts; avoiding near-term earnings landmines."
    df.loc[catalyst_notes.str.len() > 0, "thesis"] = (
        "Momentum + liquidity with catalysts: " + catalyst_notes[catalyst_notes.str.len() > 0]
    )
    logger.debug(f"generate_short_term_suggestions final df:\n{df}")
    return df


# -----------------------------------------------------------------------------
# Swing (3-4 month) suggester
# -----------------------------------------------------------------------------

@dataclass
class SwingConfig:
    target_pct: float = 0.18
    stop_pct: float = 0.08
    max_positions: int = 12
    min_price: float = 5.0
    min_dollar_vol: float = 3_000_000.0
    near_high_thresh_pct: float = 20.0
    min_momentum_20: float = -0.02
    min_momentum_60: float = 0.05
    prefer_trend: bool = True
    min_revenue_growth: float = 0.12
    min_eps_growth: float = 0.0
    min_inst_percent: float = 0.15
    require_inst_trend: bool = True


def generate_swing_suggestions(
    tickers: Optional[list[str]] = None,
    *,
    period: str = "3y",
    cfg_overrides: Optional[dict[str, Any]] = None,
    shared_data: Optional[SharedSuggesterData] = None,
    feature_lookback: Optional[int] = None,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Medium-term (3-4 month) swing ideas that favor steady 3-6 month momentum
    and rising trends with room to run.
    """
    cfg = SwingConfig()
    if cfg_overrides:
        valid = {f.name for f in fields(SwingConfig)}
        for k, v in cfg_overrides.items():
            if k in valid:
                setattr(cfg, k, v)

    lookback = feature_lookback or (shared_data.feature_lookback if shared_data else None)
    history_years = max(int(str(period).replace("y", "") or 3), 2)
    base_universe = tickers or _read_default_tickers()
    history_universe = sorted({*base_universe, "SPY"})
    _maybe_cancel(cancel_token)
    df = (
        shared_data.history
        if shared_data and shared_data.history is not None
        else _load_history(history_universe, years=history_years, cancel_token=cancel_token)
    )
    _maybe_cancel(cancel_token)
    feats_source = shared_data.features if shared_data and shared_data.features is not None else None
    feats = (
        feats_source.copy()
        if feats_source is not None
        else compute_features(df, high_lookback=lookback, cancel_token=cancel_token)
    )
    market_feats = feats.loc["SPY"] if "SPY" in feats.index else None
    feats = feats.drop(index="SPY", errors="ignore")
    feats = feats.loc[feats.index.intersection(base_universe)]

    logger.debug(f"swing suggester feats: {feats}")

    fund_cache = shared_data.fundamentals if shared_data else None
    fundamentals: list[dict[str, Any]] = []
    for t in feats.index:
        _maybe_cancel(cancel_token)
        snap = fund_cache.get(t) if fund_cache else None
        if snap is None:
            snap = _fundamental_snapshot(t)
            if fund_cache is not None:
                fund_cache[t] = snap
        enriched = dict(snap)
        enriched["ticker"] = t
        fundamentals.append(enriched)
    if fundamentals:
        feats = feats.join(pd.DataFrame(fundamentals).set_index("ticker"), how="left")
    if "sector" not in feats.columns:
        feats["sector"] = ""
    if "inst_trending_up" in feats.columns:
        feats["inst_trending_up"] = feats["inst_trending_up"].astype("boolean")

    market_r60 = market_feats["r60"] if market_feats is not None and "r60" in market_feats else np.nan
    feats["market_rel_strength"] = feats["r60"] - market_r60 if pd.notna(market_r60) else np.nan
    sector_strength = (
        feats.groupby("sector")["r60"].transform("mean") if "sector" in feats.columns else pd.Series(np.nan, index=feats.index)
    )
    feats["sector_rel_strength"] = feats["r60"] - sector_strength

    logger.debug(f"swing suggester feats2: {feats}")
    out_dir = (Path(__file__).resolve().parent / ".." / ".." / "files" / "daily_suggestions").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(os.path.join(out_dir, f"feats_full_ex.csv"))
    candidates = feats[
        (feats["close"] >= cfg.min_price)
        & (feats["adv20_dollar"] >= cfg.min_dollar_vol)
        & (feats["r20"] >= cfg.min_momentum_20)
        & (feats["r60"] >= cfg.min_momentum_60)
    ]
    if cfg.prefer_trend:
        candidates = candidates[(candidates["sma50"] > candidates["sma200"])]
    if cfg.near_high_thresh_pct is not None:
        candidates = candidates[(candidates["prox_52w_high_pct"] <= cfg.near_high_thresh_pct)]
    if "revenue_growth" in candidates.columns and cfg.min_revenue_growth is not None:
        mask = (candidates["revenue_growth"] >= cfg.min_revenue_growth) | candidates["revenue_growth"].isna()
        candidates = candidates[mask]
    if "eps_growth" in candidates.columns and cfg.min_eps_growth is not None:
        mask = (candidates["eps_growth"] >= cfg.min_eps_growth) | candidates["eps_growth"].isna()
        candidates = candidates[mask]
    if "inst_percent" in candidates.columns and cfg.min_inst_percent is not None:
        mask = candidates["inst_percent"].isna() | (candidates["inst_percent"] >= cfg.min_inst_percent)
        candidates = candidates[mask]
    if cfg.require_inst_trend and "inst_trending_up" in candidates.columns:
        inst_trend = candidates["inst_trending_up"].astype("boolean")
        mask = inst_trend.fillna(True)
        candidates = candidates[mask]

    logger.debug(f"generate_swing_suggestions candidates:\n{candidates}")

    if candidates.empty:
        return _empty_payload("3-4 month")

    growth_score = (
        0.6
        * _normalize_key_value(
            candidates.get("revenue_growth", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=0.0,
            high=0.30,
        )
        + 0.4
        * _normalize_key_value(
            candidates.get("eps_growth", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=0.0,
            high=0.25,
        )
    )
    rel_strength_score = (
        0.6
        * _normalize_key_value(
            candidates.get("sector_rel_strength", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=-0.05,
            high=0.20,
        )
        + 0.4
        * _normalize_key_value(
            candidates.get("market_rel_strength", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=-0.05,
            high=0.20,
        )
    )
    inst_score = _normalize_key_value(
        candidates.get("inst_percent", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
        low=0.05,
        high=0.65,
    )

    candidates = candidates.copy()
    inst_trending = candidates.get("inst_trending_up", pd.Series(index=candidates.index, dtype="boolean")).astype("boolean")
    candidates["score"] = (
        0.30 * _normalize_key_value(candidates["r60"], low=-0.05, high=0.30)
        + 0.20 * _normalize_key_value(candidates["r20"], low=-0.05, high=0.20)
        + 0.10 * _normalize_key_value(candidates["prox_52w_high_pct"], low=0.0, high=max(cfg.near_high_thresh_pct, 1.0), invert=True)
        + 0.10 * (candidates["sma50"] > candidates["sma200"]).astype(float)
        + 0.15 * growth_score
        + 0.10 * rel_strength_score
        + 0.05 * inst_score
        + 0.05 * inst_trending.fillna(False).astype(float)
    )

    plan = candidates.sort_values("score", ascending=False).head(cfg.max_positions)
    plan["entry"] = plan["close"] * 1.005  # lean into strength, tiny breakout buffer
    plan["target"] = plan["entry"] * (1.0 + cfg.target_pct)
    plan["stop"] = plan["entry"] * (1.0 - cfg.stop_pct)

    return _attach_common_fields(
        plan,
        horizon="3-4 month",
        thesis="Medium-term trend with revenue/EPS growth, sector leadership, and supportive ownership.",
    )


# -----------------------------------------------------------------------------
# Long-term (1+ year) suggester
# -----------------------------------------------------------------------------

@dataclass
class LongTermConfig:
    target_pct: float = 0.35
    stop_pct: float = 0.20
    max_positions: int = 10
    min_price: float = 5.0
    min_dollar_vol: float = 1_000_000.0
    near_high_thresh_pct: float = 35.0
    min_return_1y: float = 0.05
    prefer_stability: bool = True
    max_volatility: float = 0.45  # annualized, approximate
    min_fcf_margin: float = 0.0
    min_gross_margin: float = 0.25
    min_gross_margin_trend: float = -0.02
    max_net_debt_to_ebitda: float = 4.0
    max_price_to_sales: float = 12.0
    max_ev_to_ebitda: float = 35.0
    max_dilution_rate: float = 0.20


def _long_horizon_addons(
    df: pd.DataFrame,
    *,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Compute slower-moving features (6m/12m returns, 60d vol) for long holds.
    """
    rows: list[dict[str, Any]] = []
    tickers = sorted({t for t, _ in df.columns})
    for t in tickers:
        _maybe_cancel(cancel_token)
        close = df[t]["Close"].dropna()
        if close.empty:
            continue
        r120 = close.pct_change(120).iloc[-1] if len(close) > 120 else np.nan
        r252 = close.pct_change(252).iloc[-1] if len(close) > 252 else np.nan
        vol60 = close.pct_change().rolling(60).std().iloc[-1]
        rows.append(
            {
                "ticker": t,
                "r120": r120,
                "r252": r252,
                "volatility60": vol60 * np.sqrt(252) if pd.notna(vol60) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def generate_long_term_suggestions(
    tickers: Optional[list[str]] = None,
    *,
    period: str = "5y",
    cfg_overrides: Optional[dict[str, Any]] = None,
    shared_data: Optional[SharedSuggesterData] = None,
    feature_lookback: Optional[int] = None,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Long-hold (1+ year) ideas that prioritize durable uptrends, positive 12m
    returns, and relatively lower realized volatility.
    """
    cfg = LongTermConfig()
    if cfg_overrides:
        valid = {f.name for f in fields(LongTermConfig)}
        for k, v in cfg_overrides.items():
            if k in valid:
                setattr(cfg, k, v)

    lookback = feature_lookback or (shared_data.feature_lookback if shared_data else None)
    history_years = max(int(str(period).replace("y", "") or 5), 3)
    base_universe = tickers or _read_default_tickers()
    _maybe_cancel(cancel_token)
    df = (
        shared_data.history
        if shared_data and shared_data.history is not None
        else _load_history(base_universe, years=history_years, cancel_token=cancel_token)
    )
    _maybe_cancel(cancel_token)
    feats_source = shared_data.features if shared_data and shared_data.features is not None else None
    feats = (
        feats_source.copy()
        if feats_source is not None
        else compute_features(df, high_lookback=lookback, cancel_token=cancel_token)
    )
    feats = feats.drop(index="SPY", errors="ignore")
    feats = feats.loc[feats.index.intersection(base_universe)]
    _maybe_cancel(cancel_token)
    addons = (
        shared_data.long_addons
        if shared_data and shared_data.long_addons is not None
        else _long_horizon_addons(df, cancel_token=cancel_token)
    )
    if not addons.empty:
        feats = feats.join(addons, how="left")
    for col in ("r120", "r252", "volatility60"):
        if col not in feats.columns:
            feats[col] = np.nan

    fund_cache = shared_data.fundamentals if shared_data else None
    fundamentals: list[dict[str, Any]] = []
    for t in feats.index:
        _maybe_cancel(cancel_token)
        snap = fund_cache.get(t) if fund_cache else None
        if snap is None:
            snap = _fundamental_snapshot(t)
            if fund_cache is not None:
                fund_cache[t] = snap
        enriched = dict(snap)
        enriched["ticker"] = t
        fundamentals.append(enriched)
    if fundamentals:
        feats = feats.join(pd.DataFrame(fundamentals).set_index("ticker"), how="left")
    if "sector" not in feats.columns:
        feats["sector"] = ""

    candidates = feats[
        (feats["close"] >= cfg.min_price)
        & (feats["adv20_dollar"] >= cfg.min_dollar_vol)
        & (feats["sma200"] > 0)
        & (feats["close"] > feats["sma200"])
        & (feats["r252"] >= cfg.min_return_1y)
    ]
    if cfg.prefer_stability and "volatility60" in candidates.columns:
        candidates = candidates[candidates["volatility60"] <= cfg.max_volatility]
    if cfg.near_high_thresh_pct is not None:
        candidates = candidates[candidates["prox_52w_high_pct"] <= cfg.near_high_thresh_pct]
    if "fcf_margin" in candidates.columns and cfg.min_fcf_margin is not None:
        mask = (candidates["fcf_margin"] >= cfg.min_fcf_margin) | candidates["fcf_margin"].isna()
        candidates = candidates[mask]
    if "gross_margin" in candidates.columns and cfg.min_gross_margin is not None:
        mask = (candidates["gross_margin"] >= cfg.min_gross_margin) | candidates["gross_margin"].isna()
        candidates = candidates[mask]
    if "gross_margin_trend" in candidates.columns and cfg.min_gross_margin_trend is not None:
        mask = (candidates["gross_margin_trend"] >= cfg.min_gross_margin_trend) | candidates["gross_margin_trend"].isna()
        candidates = candidates[mask]
    if "net_debt_to_ebitda" in candidates.columns and cfg.max_net_debt_to_ebitda is not None:
        debt_mask = (
            candidates["net_debt_to_ebitda"].isna()
            | (candidates["net_debt_to_ebitda"] <= cfg.max_net_debt_to_ebitda)
            | (candidates.get("net_debt", pd.Series(index=candidates.index, dtype=float)) <= 0)
        )
        candidates = candidates[debt_mask]
    if "ps_ratio" in candidates.columns and cfg.max_price_to_sales is not None:
        mask = candidates["ps_ratio"].isna() | (candidates["ps_ratio"] <= cfg.max_price_to_sales)
        candidates = candidates[mask]
    if "ev_to_ebitda" in candidates.columns and cfg.max_ev_to_ebitda is not None:
        mask = candidates["ev_to_ebitda"].isna() | (candidates["ev_to_ebitda"] <= cfg.max_ev_to_ebitda)
        candidates = candidates[mask]
    if "dilution_rate" in candidates.columns and cfg.max_dilution_rate is not None:
        mask = candidates["dilution_rate"].isna() | (candidates["dilution_rate"] <= cfg.max_dilution_rate)
        candidates = candidates[mask]

    logger.debug(f"generate_long_term_suggestions candidates:\n{candidates}")
    if candidates.empty:
        return _empty_payload("1+ year")

    fcf_score = _normalize_key_value(
        candidates.get("fcf_margin", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
        low=0.0,
        high=0.20,
    )
    margin_score = (
        0.6
        * _normalize_key_value(
            candidates.get("gross_margin", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=cfg.min_gross_margin,
            high=max(cfg.min_gross_margin + 0.25, 0.3),
        )
        + 0.4
        * _normalize_key_value(
            candidates.get("gross_margin_trend", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
            low=cfg.min_gross_margin_trend,
            high=0.10,
        )
    )
    debt_score = _normalize_key_value(
        candidates.get("net_debt_to_ebitda", pd.Series(index=candidates.index, dtype=float)).fillna(cfg.max_net_debt_to_ebitda / 2),
        low=0.0,
        high=cfg.max_net_debt_to_ebitda,
        invert=True,
    )
    val_score = (
        0.5
        * _normalize_key_value(
            candidates.get("ps_ratio", pd.Series(index=candidates.index, dtype=float)).fillna(cfg.max_price_to_sales),
            low=0.0,
            high=cfg.max_price_to_sales,
            invert=True,
        )
        + 0.5
        * _normalize_key_value(
            candidates.get("ev_to_ebitda", pd.Series(index=candidates.index, dtype=float)).fillna(cfg.max_ev_to_ebitda),
            low=0.0,
            high=cfg.max_ev_to_ebitda,
            invert=True,
        )
    )
    dilution_score = _normalize_key_value(
        candidates.get("dilution_rate", pd.Series(index=candidates.index, dtype=float)).fillna(0.0),
        low=0.0,
        high=cfg.max_dilution_rate if cfg.max_dilution_rate else 0.20,
        invert=True,
    )

    candidates = candidates.copy()
    candidates["score"] = (
        0.25 * _normalize_key_value(candidates["r252"], low=-0.05, high=0.40)
        + 0.15 * _normalize_key_value(candidates.get("r120", pd.Series(0, index=candidates.index)), low=-0.05, high=0.30)
        + 0.10 * _normalize_key_value(candidates["prox_52w_high_pct"], low=0.0, high=max(cfg.near_high_thresh_pct, 1.0), invert=True)
        + 0.08
        * _normalize_key_value(
            candidates.get("volatility60", pd.Series(np.nan, index=candidates.index)),
            low=0.0,
            high=cfg.max_volatility,
            invert=True,
        )
        + 0.18 * fcf_score
        + 0.12 * margin_score
        + 0.07 * val_score
        + 0.03 * debt_score
        + 0.02 * dilution_score
    )

    plan = candidates.sort_values("score", ascending=False).head(cfg.max_positions)
    plan["entry"] = plan["close"]
    plan["target"] = plan["entry"] * (1.0 + cfg.target_pct)
    plan["stop"] = plan["entry"] * (1.0 - cfg.stop_pct)

    return _attach_common_fields(
        plan,
        horizon="1+ year",
        thesis="Long-term compounding: trend + FCF-positive names with stable margins and sane leverage/valuations.",
    )
