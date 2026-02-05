#!/usr/bin/env python3
"""
Stock Buying Suggester (Evening Run) — Documented Version

This script scans a universe of tickers after the close and proposes buy
candidates for tomorrow based on liquidity, momentum, proximity to highs,
and a simple trend filter. It then produces breakout entries, ATR-based
stops, and position sizes using fixed risk-per-trade.

Usage (example):
    python daily_buy_suggester.py --tickers-file helper_files/mass_tickers_list.csv \
        --capital 100000 --risk-per-trade 0.005 --max-positions 8

    python daily_buy_suggester.py --tickers-file helper_files/mass_tickers_list.csv

Outputs:
    - CSV file: suggestions_YYYYMMDD.csv (ranked & sized picks)
    - Console table with the key columns

Dependencies:
    pip install pandas numpy loguru
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import math
import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, TYPE_CHECKING
from loguru import logger
from backend.core.suggester_cancel import CancelToken

# Tiingo price history
from backend.platform_apis.tiingo_api.tiingo_api import get_daily_prices

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backend.core.horizon_buy_suggesters import SharedSuggesterData


# ============================================================================
# Data providers
# ============================================================================

class DataProvider:
    """Abstract interface for fetching historical OHLCV.

    The concrete provider returns a pandas DataFrame with:
        - Index: DatetimeIndex (daily resolution)
        - Columns: MultiIndex [(ticker, field)] for field in
          {'Open','High','Low','Close','Volume'}.

    Implementations must provide:
        history(tickers: List[str], period: str = "1y") -> pd.DataFrame
    """
    def history(
        self,
        tickers: List[str],
        period: str = "1y",
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


def _to_epoch_ms(dt_val: Optional[object]) -> Optional[int]:
    """Convert a date/datetime/pandas Timestamp into epoch milliseconds."""
    if dt_val is None:
        return None
    if isinstance(dt_val, pd.Timestamp):
        dt_val = dt_val.to_pydatetime()
    if isinstance(dt_val, date) and not isinstance(dt_val, datetime):
        dt_val = datetime.combine(dt_val, datetime.min.time())
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        else:
            dt_val = dt_val.astimezone(timezone.utc)
        return int(dt_val.timestamp() * 1000)
    return None

class TiingoMarketDataProvider(DataProvider):
    """Tiingo-backed provider with the same MultiIndex OHLCV contract."""

    def __init__(self, *, use_adjusted: bool = False):
        self.use_adjusted = use_adjusted

    def _period_to_days(self, period: str | None) -> int:
        if not period:
            return 365
        text = str(period).strip().lower()
        m = re.match(r"^(\d+)\s*(y|yr|year)s?$", text)
        if m:
            return max(int(m.group(1)), 1) * 365
        m = re.match(r"^(\d+)\s*(mo|month)s?$", text)
        if m:
            return max(int(m.group(1)), 1) * 31
        m = re.match(r"^(\d+)\s*(d|day)s?$", text)
        if m:
            return max(int(m.group(1)), 1)
        return 365

    def _resolve_dates(
        self,
        period: str,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
    ) -> tuple[str | None, str | None]:
        if start_date is not None or end_date is not None:
            start = start_date
            end = end_date
        else:
            days = self._period_to_days(period)
            end = datetime.now(timezone.utc).date()
            start = end - timedelta(days=days)
        if isinstance(start, str):
            start_text = start
        else:
            start_text = start.isoformat() if hasattr(start, "isoformat") else None
        if isinstance(end, str):
            end_text = end
        else:
            end_text = end.isoformat() if hasattr(end, "isoformat") else None
        return start_text, end_text

    def _one_symbol_daily(
        self,
        symbol: str,
        *,
        period: str,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        _maybe_cancel(cancel_token)
        start_text, end_text = self._resolve_dates(period, start_date=start_date, end_date=end_date)
        try:
            payload = get_daily_prices(symbol, start_date=start_text, end_date=end_text)
        except Exception as exc:
            logger.warning(f"Tiingo history failed for {symbol}: {exc}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        frame = pd.DataFrame(payload if isinstance(payload, list) else [])
        if frame.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        date_col = "date" if "date" in frame.columns else None
        if date_col is None:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        cols = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        if self.use_adjusted and "adjClose" in frame.columns:
            cols = {
                "Open": "adjOpen" if "adjOpen" in frame.columns else "open",
                "High": "adjHigh" if "adjHigh" in frame.columns else "high",
                "Low": "adjLow" if "adjLow" in frame.columns else "low",
                "Close": "adjClose",
                "Volume": "adjVolume" if "adjVolume" in frame.columns else "volume",
            }

        out = pd.DataFrame({k: frame.get(v) for k, v in cols.items()})
        out.index = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
        out = out.dropna(subset=["Close"])
        out.index.name = None
        return out

    def history(
        self,
        tickers: List[str],
        period: str = "1y",
        *,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        frames = []
        for t in tickers:
            _maybe_cancel(cancel_token)
            sub = self._one_symbol_daily(
                t,
                period=period,
                start_date=start_date,
                end_date=end_date,
                cancel_token=cancel_token,
            )
            if sub.empty:
                continue
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)

        if not frames:
            raise ValueError("Tiingo provider: no data returned for any ticker.")

        out = pd.concat(frames, axis=1).sort_index()
        out = out.dropna(how="all", axis=0)
        return out
# ============================================================================
# Feature engineering helpers
# ============================================================================

def _sma(x: pd.Series, window: int) -> pd.Series:
    """Simple moving average of `x` over `window` bars."""
    return x.rolling(window, min_periods=window).mean()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Approximate Wilder ATR via a simple rolling mean of True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def _pct_change(x: pd.Series, periods: int) -> pd.Series:
    """Percent change over `periods` bars."""
    return x.pct_change(periods=periods)

def _rolling_avg_dollar_vol(close: pd.Series, vol: pd.Series, window: int = 20) -> pd.Series:
    """Rolling average dollar volume: mean of (close × volume) over `window`."""
    return (close * vol).rolling(window, min_periods=window).mean()

def _vol_ratio_today(vol: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume relative to the rolling `window` average volume."""
    return vol / vol.rolling(window, min_periods=window).mean()

def _proximity_to_high(
    close: pd.Series,
    high: pd.Series,
    *,
    lookback: Optional[int] = None,
) -> pd.Series:
    """Percent below the rolling `lookback` high (0% = at high; smaller is better)."""
    window = min(len(close), int(lookback)) if lookback is not None else len(close)
    if window <= 0:
        return pd.Series(index=close.index, dtype=float)
    hh = high.rolling(window, min_periods=window).max()
    return 100 * (hh - close) / hh

def _maybe_cancel(cancel_token: Optional[CancelToken]) -> None:
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()


def compute_features(
    df: pd.DataFrame,
    *,
    high_lookback: Optional[int] = None,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """Compute per-symbol, last-observation features from daily OHLCV.

    Parameters
    ----------
    df : DataFrame
        MultiIndex columns [(ticker, field)], daily rows.
    high_lookback : int, optional
        If provided, cap the proximity-to-high lookback window (in bars).

    Returns
    -------
    DataFrame
        One row per ticker (index), with the latest values of:
        - close, high, low
        - sma50, sma200
        - atr14, adv20_dollar, vol_ratio20
        - prox_52w_high_pct
        - r5, r20, r60 (momentum over 1/4/12 weeks)
    """
    feats = []
    tickers = sorted(set([t for t, _ in df.columns]))
    for t in tickers:
        _maybe_cancel(cancel_token)
        sub = df[t].dropna()
        if sub.empty:
            continue
        close = sub['Close']
        high = sub['High']
        low = sub['Low']
        vol = sub['Volume']

        sma50 = _sma(close, 50)
        sma200 = _sma(close, 200)
        atr14 = _atr(high, low, close, 14)
        adv20 = _rolling_avg_dollar_vol(close, vol, 20)
        vol_r = _vol_ratio_today(vol, 20)
        prox_high = _proximity_to_high(close, high, lookback=high_lookback)
        r5 = _pct_change(close, 5)
        r20 = _pct_change(close, 20)
        r60 = _pct_change(close, 60)

        latest = pd.DataFrame({
            'symbol': t,
            'close': close,
            'high': high,
            'low': low,
            'sma50': sma50,
            'sma200': sma200,
            'atr14': atr14,
            'adv20_dollar': adv20,
            'vol_ratio20': vol_r,
            'prox_52w_high_pct': prox_high,
            'r5': r5,
            'r20': r20,
            'r60': r60,
        }).tail(1)

        feats.append(latest)

    if not feats:
        raise ValueError("No features computed (check your tickers or data).")

    latest = pd.concat(feats, axis=0)
    latest.set_index('symbol', inplace=True)
    return latest


# ============================================================================
# Filtering, scoring, and position sizing
# ============================================================================

@dataclass
class Config:
    """Run-time knobs for filtering and position sizing."""
    min_price: float = 3.0
    max_price: float = 1000.0
    min_dollar_vol: float = 5_000_000.0
    near_high_thresh_pct: float = 5.0
    min_vol_ratio: float = 1.3
    require_trend: bool = True  # SMA50 > SMA200
    entry_buffer_bps: int = 10  # buy stop above today's high (0.10%)
    stop_atr_mult: float = 1.5
    capital: float = 5_000.0 # 100_000.0
    risk_per_trade: float = 0.005  # 0.5%
    max_positions: int = 8
    cap_by_equal_allocation: bool = True  # cap notional at capital / max_positions

def filter_candidates(latest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply hard screening rules for liquidity/price/trend/momentum."""
    filt = latest.copy()
    # Price band
    filt = filt[(filt['close'] >= cfg.min_price) & (filt['close'] <= cfg.max_price)]
    # Liquidity
    filt = filt[filt['adv20_dollar'] >= cfg.min_dollar_vol]
    # Longer-term uptrend (optional)
    if cfg.require_trend:
        filt = filt[filt['sma50'] > filt['sma200']]
    # Volume expansion today
    filt = filt[filt['vol_ratio20'] >= cfg.min_vol_ratio]
    # Near 52w highs
    filt = filt[filt['prox_52w_high_pct'] <= cfg.near_high_thresh_pct]
    # Short-term momentum confirmation
    filt = filt[(filt['r20'] > 0) & (filt['r5'] > -0.02)]
    return filt

def score_candidates(filt: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Combine normalized signals into a single ranking score [0..1]."""
    df = filt.copy()
    # Normalize components
    r20n = ((df['r20'].clip(-0.10, 0.30) + 0.10) / 0.40).fillna(0.0)  # momentum
    voln = ((df['vol_ratio20'] - 0.8 * cfg.min_vol_ratio) / (0.2 * cfg.min_vol_ratio)).clip(0.0, 1.0)  # vol surge
    proxn = (1.0 - (df['prox_52w_high_pct'] / cfg.near_high_thresh_pct)).clip(0.0, 1.0)  # closeness to highs
    trendn = (df['sma50'] > df['sma200']).astype(float)  # 1 if uptrend

    df['score'] = 0.35 * r20n + 0.25 * voln + 0.20 * proxn + 0.20 * trendn
    return df.sort_values('score', ascending=False)

def plan_entries_and_size(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create breakout entries/stops and size positions from fixed risk-per-trade.

    - Entry: today's high + `entry_buffer_bps` (buy-stop for tomorrow)
    - Stop: min( close - ATR*mult , today's low )
    - Size: floor( risk_dollars / risk_per_share ), optionally capped by equal allocation

    Returns
    -------
    DataFrame
        Top-N rows (by score) with helpful columns for orders and risk tracking.
    """
    out = scored.copy()
    # Breakout entry: tiny buffer above today's high
    out['entry'] = out['high'] * (1.0 + cfg.entry_buffer_bps / 10_000.0)
    # Initial stop: ATR-based, never above today's low
    raw_stop = out['close'] - cfg.stop_atr_mult * out['atr14']
    out['stop'] = np.minimum(raw_stop, out['low'])

    # Risk per share & sizing
    rps = (out['entry'] - out['stop']).clip(lower=0.01)  # guard against zero/negative
    dollar_risk_per_trade = cfg.capital * cfg.risk_per_trade
    qty_by_risk = np.floor(dollar_risk_per_trade / rps)

    if cfg.cap_by_equal_allocation:
        alloc_cap = cfg.capital / max(1, cfg.max_positions)
        qty_by_alloc = np.floor(alloc_cap / out['entry'])
        qty = np.minimum(qty_by_risk, qty_by_alloc)
    else:
        qty = qty_by_risk

    out['qty'] = qty.astype(int)
    out['notional'] = out['qty'] * out['entry']
    out = out[out['qty'] > 0]  # drop names that cannot be sized

    # Keep only top-N
    out = out.sort_values('score', ascending=False).head(cfg.max_positions)

    # Risk context
    out['risk_$'] = (out['entry'] - out['stop']) * out['qty']
    out['risk_%'] = 100 * (out['entry'] - out['stop']) / out['entry']
    out['atr_%'] = 100 * out['atr14'] / out['close']

    cols = [
        'close','sma50','sma200','adv20_dollar','vol_ratio20',
        'prox_52w_high_pct','r5','r20','r60','atr14','atr_%',
        'score','entry','stop','risk_%','qty','notional','risk_$'
    ]
    return out[cols].round({
        'close': 2, 'sma50': 2, 'sma200': 2, 'adv20_dollar': 0,
        'vol_ratio20': 2, 'prox_52w_high_pct': 2, 'r5': 4, 'r20': 4, 'r60': 4,
        'atr14': 3, 'atr_%': 2, 'score': 3, 'entry': 2, 'stop': 2,
        'risk_%': 2, 'notional': 0, 'risk_$': 0
    })

# ============================================================================
# API-friendly runner
# ============================================================================

def generate_daily_buy_suggestions(
    tickers: Optional[list[str]] = None,
    *,
    period: str = "1y",
    cfg_overrides: Optional[dict[str, Any]] = None,
    shared_data: Optional["SharedSuggesterData"] = None,
    feature_lookback: Optional[int] = None,
    cancel_token: Optional[CancelToken] = None,
) -> pd.DataFrame:
    """
    Run the daily buy suggester and return a DataFrame of suggestions.
    The API layer can serialize this via ``to_dict(orient="records")``.
    """
    cfg = Config()
    if cfg_overrides:
        # Only apply known fields to keep things safe
        valid = {f.name for f in fields(Config)}
        for k, v in cfg_overrides.items():
            if k in valid:
                setattr(cfg, k, v)

    _maybe_cancel(cancel_token)

    # Build universe
    if not tickers:
        tickers = _read_default_tickers()

    if not tickers:
        raise ValueError("No tickers provided or found in default list.")

    lookback = feature_lookback
    if lookback is None and shared_data is not None:
        lookback = getattr(shared_data, "feature_lookback", None)

    feats_source = shared_data.features if shared_data and getattr(shared_data, "features", None) is not None else None
    history = shared_data.history if shared_data and getattr(shared_data, "history", None) is not None else None

    _maybe_cancel(cancel_token)

    if feats_source is not None:
        feats = feats_source.copy()
    else:
        if history is None:
            provider = TiingoMarketDataProvider()
            history = provider.history(tickers, period=period, cancel_token=cancel_token)
        feats = compute_features(history, high_lookback=lookback, cancel_token=cancel_token)

    _maybe_cancel(cancel_token)
    feats = feats.loc[feats.index.intersection(tickers)]
    feats = feats.drop(index="SPY", errors="ignore")
    latest = feats.dropna()
    logger.debug(f"daily_buy_suggestions feats: {feats}")
    cand = filter_candidates(latest, cfg)
    if cand.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "horizon",
                "thesis",
            ]
        )

    scored = score_candidates(cand, cfg)
    plan = plan_entries_and_size(scored, cfg)
    
    logger.debug(f"daily_buy_suggestions plan: {plan}")
    if plan.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "horizon",
                "thesis",
            ]
        )

    # Promote index to column for JSON serialization
    plan.index.name = "ticker"
    plan = plan.reset_index()
    plan["ticker"] = plan["ticker"].fillna("").astype(str)

    # Attach human-friendly fields expected by UI
    plan["action"] = "Buy"
    plan["horizon"] = "1-2 day"
    plan["thesis"] = "Momentum + liquidity screen near highs"  # placeholder

    # Present columns in a UI-friendly order but keep any extras
    preferred_cols = [
        "ticker",
        "action",
        "horizon",
        "thesis",
        "entry",
        "stop",
        "qty",
        "notional",
        "risk_$",
        "risk_%",
        "score",
        "close",
        "vol_ratio20",
        "prox_52w_high_pct",
        "r20",
    ]
    remaining_cols = [c for c in plan.columns if c not in preferred_cols]
    plan = plan[preferred_cols + remaining_cols]

    return plan

# ============================================================================
# CLI plumbing
# ============================================================================

DEFAULT_TICKER_FILE = Path(__file__).resolve().parent / "helper_files" / "sandp_tickers_list.csv"
FALLBACK_TICKER_FILE = Path(__file__).resolve().parent / "helper_files" / "sandp_tickers_list_for_testing.csv"

def _read_default_tickers() -> List[str]:
    """Load tickers from the default CSVs sitting next to this module."""
    path = DEFAULT_TICKER_FILE if DEFAULT_TICKER_FILE.exists() else FALLBACK_TICKER_FILE
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found. Checked: {DEFAULT_TICKER_FILE} and {FALLBACK_TICKER_FILE}")
    content = path.read_text()
    return [s.strip() for s in content.split(",") if s.strip()]

def load_tickers_from_file(path: Path) -> List[str]:
    """Read a CSV/whitespace list of tickers, ignoring blank lines and comments (#)."""
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    tickers = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip().upper() for p in line.replace(',', ' ').split()]
        for p in parts:
            if p and p not in tickers:
                tickers.append(p)
    if not tickers:
        raise ValueError("No tickers found in file.")
    return tickers

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse command-line arguments for the evening run."""
    p = argparse.ArgumentParser(description="Evening stock buying suggester")
    p.add_argument('--tickers-file', type=Path, help='Path to a text/CSV file of tickers')
    p.add_argument('--capital', type=float, default=100000.0)
    p.add_argument('--risk-per-trade', type=float, default=0.005)
    p.add_argument('--max-positions', type=int, default=8)
    p.add_argument('--min-dollar-vol', type=float, default=5_000_000.0)
    p.add_argument('--min-price', type=float, default=3.0)
    p.add_argument('--max-price', type=float, default=1000.0)
    p.add_argument('--near-high-thresh-pct', type=float, default=5.0)
    p.add_argument('--min-vol-ratio', type=float, default=1.3)
    p.add_argument('--entry-buffer-bps', type=int, default=10)
    p.add_argument('--stop-atr-mult', type=float, default=1.5)
    p.add_argument('--no-equal-allocation-cap', action='store_true',
                   help='Disable per-position capital cap (default is enabled)')
    p.add_argument('--period', default='1y', help='History period for features (default 1y)')
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    """Orchestrate the evening run: fetch → feature → filter → score → size → export."""
    args = parse_args(argv)
    cfg = Config(
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        max_positions=args.max_positions,
        min_dollar_vol=args.min_dollar_vol,
        min_price=args.min_price,
        max_price=args.max_price,
        near_high_thresh_pct=args.near_high_thresh_pct,
        min_vol_ratio=args.min_vol_ratio,
        entry_buffer_bps=args.entry_buffer_bps,
        stop_atr_mult=args.stop_atr_mult,
        cap_by_equal_allocation=not args.no_equal_allocation_cap,
    )

    # Build universe and choose a provider
    # tickers = load_tickers_from_file(args.tickers_file)
    # sp_500_list = data_api.get_sp_500()
    tickers = _read_default_tickers()

    # Use Tiingo provider
    provider = TiingoMarketDataProvider()

    print(f"Loaded {len(tickers)} tickers. Fetching {args.period} daily history ...")
    df = provider.history(tickers, period=args.period)

    print(f"df:\n{df}")
    

    # Feature calc on daily bars; keep last row per ticker
    feats = compute_features(df)
    print(f"feats:\n{feats}")
    latest = feats.dropna()  # drop tickers with insufficient history
    print(f"latest:\n{latest}")
    # Hard filters → ranking → breakout plan & sizing
    cand = filter_candidates(latest, cfg)
    if cand.empty:
        print("No candidates passed filters. Consider relaxing thresholds.")
        return 0

    scored = score_candidates(cand, cfg)
    plan = plan_entries_and_size(scored, cfg)

    if plan.empty:
        print("No positions sized > 0. Try adjusting capital, risk-per-trade, or thresholds.")
        return 0

    # Export
    today = datetime.now(timezone.utc).astimezone().date().strftime('%Y%m%d')
    out_dir = (Path(__file__).resolve().parent / ".." / ".." / "files" / "daily_suggestions").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"suggestions_{today}.csv"
    plan.to_csv(out_path, index=True)
    print("\nTop suggestions (save to:", out_path, ")\n")
    display_cols = ['close','vol_ratio20','prox_52w_high_pct','r20','atr_%','score','entry','stop','risk_%','qty','notional']
    print(plan[display_cols].to_string())

    # Helpful notes
    print("\nNotes:")
    print(" - Place BUY STOP orders at 'entry' only for tomorrow. If not triggered, re-evaluate next evening.")
    print(" - Stops are initial. Trail or update per your rules.")
    print(" - Consider filtering out symbols with earnings tomorrow if your strategy requires it.")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
