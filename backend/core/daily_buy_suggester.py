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
    pip install pandas numpy yfinance loguru
    (Schwab provider requires your repo's MarketData and auth helpers)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import math
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
import pytz
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, TYPE_CHECKING
from loguru import logger
from backend.core.suggester_cancel import CancelToken

# Schwab integrations (present in your codebase)
from backend.platform_apis.schwab_api.accounts_trading import AccountsTrading
from backend.platform_apis.schwab_api.helpers import design_get_historical_price
from backend.platform_apis.schwab_api.get_refresh_token import refresh_tokens
from backend.platform_apis.schwab_api.market_data import MarketData

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

class SchwabMarketDataProvider(DataProvider):
    """Adapter that pulls daily bars from your Schwab `MarketData` client.

    Parameters
    ----------
    md : MarketData
        Your authenticated Schwab MarketData client.
    use_extended_hours : bool, optional
        Whether to include pre/after-hours in daily candles (default: False).
    years : int, optional
        Number of years of daily history to request (default: 1).

    Notes
    -----
    - Normalizes response into the common MultiIndex OHLCV shape expected
      by the downstream feature/score pipeline.
    """
    def __init__(
        self,
        md: MarketData,
        accounts: AccountsTrading | None = None,
        *,
        use_extended_hours: bool = False,
        years: int = 1,
    ):
        self.md = md
        self.accounts_trading = accounts
        self.use_extended_hours = use_extended_hours
        self.years = years

    # ------------------------------------------------------------------ #
    # Auth helpers
    # ------------------------------------------------------------------ #
    def refresh_classes_access_tokens(self) -> None:
        """
        Refresh OAuth tokens for the composed API clients during runtime.
        """
        self.md.refresh_access_token()
        try:
            if self.accounts_trading is None:
                self.accounts_trading = AccountsTrading()
            self.accounts_trading.refresh_access_token()
        except Exception as exc:
            logger.warning(f"Could not refresh AccountsTrading token: {exc}")

    def _one_symbol_daily(
        self,
        symbol: str,
        *,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        """Fetch ~1y of *daily* OHLCV for a single symbol from Schwab.

        Returns a DataFrame indexed by datetime with columns:
        ['Open','High','Low','Close','Volume'].
        """
        _maybe_cancel(cancel_token)
        start_ms = _to_epoch_ms(start_date)
        end_ms = _to_epoch_ms(end_date)
        # Build the Schwab/TDA "price history" request payload
        price_payload = design_get_historical_price(
            symbol=symbol,
            period_type="year",
            period=self.years,
            frequency_type="daily",
            frequency=1,
            start_date=start_ms,
            end_date=end_ms,
            need_extended_hours_data=self.use_extended_hours,
            need_previous_close=False,
        )

        # Call your client (returns dict-like with 'candles')
        resp = self.md.get_price_history(price_payload)
        if resp is None:
            refresh_tokens()
            self.refresh_classes_access_tokens()
            resp = self.md.get_price_history(price_payload)
        _maybe_cancel(cancel_token)

        if resp is None:
            logger.error(f"No price history returned for {symbol} (Schwab may have rate-limited the request).")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

        if isinstance(resp, pd.DataFrame):
            # logger.debug(
            #     "Schwab price history response for {}: type=DataFrame shape={} columns={}",
            #     symbol,
            #     resp.shape,
            #     list(resp.columns),
            # )
            if "candles" in resp.columns and not resp["candles"].empty:
                sample = resp["candles"].iloc[0]
                sample_len = len(sample) if hasattr(sample, "__len__") else None
                sample_keys = list(sample.keys()) if isinstance(sample, dict) else None
                # logger.debug(
                #     "Schwab price history candles column sample for {}: type={} len={} keys={}",
                #     symbol,
                #     type(sample).__name__,
                #     sample_len,
                #     sample_keys,
                # )
        elif isinstance(resp, dict):
            # logger.debug(
            #     "Schwab price history response for {}: type=dict keys={} has_candles={}",
            #     symbol,
            #     list(resp.keys()),
            #     "candles" in resp,
            # )
            if "candles" in resp:
                sample = resp.get("candles")
                sample_len = len(sample) if hasattr(sample, "__len__") else None
                first_keys = None
                if isinstance(sample, list) and sample and isinstance(sample[0], dict):
                    first_keys = list(sample[0].keys())
                elif isinstance(sample, dict):
                    first_keys = list(sample.keys())
                # logger.debug(
                #     "Schwab price history candles sample for {}: type={} len={} first_keys={}",
                #     symbol,
                #     type(sample).__name__,
                #     sample_len,
                #     first_keys,
                # )
        else:
            logger.debug(
                "Schwab price history response for {}: type={}",
                symbol,
                type(resp).__name__,
            )

        candles_raw = None
        if isinstance(resp, pd.DataFrame):
            try:
                if "candles" in resp.columns and not resp["candles"].empty:
                    col = resp["candles"]
                    if len(resp) == 1:
                        candles_raw = col.iloc[0]
                    else:
                        if col.apply(lambda x: isinstance(x, dict)).all():
                            candles_raw = col.tolist()
                        else:
                            flattened = []
                            for item in col:
                                if isinstance(item, list):
                                    flattened.extend(item)
                                elif isinstance(item, dict):
                                    flattened.append(item)
                            candles_raw = flattened if flattened else col.iloc[0]
                else:
                    candles_raw = resp.to_dict("records")
            except Exception:
                candles_raw = None
        elif isinstance(resp, dict):
            candles_raw = resp.get("candles")
        else:
            getter = getattr(resp, "get", None)
            candles_raw = getter("candles") if getter else None

        # Handle nested payloads (e.g., DataFrame row holding a dict with 'candles')
        if isinstance(candles_raw, list) and candles_raw:
            first_entry = candles_raw[0]
            if isinstance(first_entry, dict) and "candles" in first_entry:
                candles_raw = first_entry.get("candles")
        elif isinstance(candles_raw, dict):
            if "candles" in candles_raw:
                candles_raw = candles_raw.get("candles")
        elif isinstance(candles_raw, pd.Series):
            try:
                first_val = candles_raw.iloc[0]
                if isinstance(first_val, dict):
                    if "candles" in first_val:
                        candles_raw = first_val.get("candles")
                    else:
                        candles_raw = [first_val]
            except Exception:
                pass

        if not candles_raw:
            logger.error(f"No candles found in price history response for {symbol}.")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

        sample_len = len(candles_raw) if hasattr(candles_raw, "__len__") else None
        sample_keys = None
        if isinstance(candles_raw, dict):
            sample_keys = list(candles_raw.keys())
        elif isinstance(candles_raw, list) and candles_raw and isinstance(candles_raw[0], dict):
            sample_keys = list(candles_raw[0].keys())
        # logger.debug(
        #     "Schwab price history normalized candles for {}: type={} len={} sample_keys={}",
        #     symbol,
        #     type(candles_raw).__name__,
        #     sample_len,
        #     sample_keys,
        # )

        # Normalize to a DataFrame
        try:
            if isinstance(candles_raw, dict):
                candles_df = pd.DataFrame([candles_raw])
            elif isinstance(candles_raw, list):
                candles_df = pd.DataFrame(candles_raw)
            else:
                candles_df = pd.DataFrame(list(candles_raw))
        except Exception as exc:
            logger.error(f"Failed to normalize candles for {symbol}: {exc}")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

        if candles_df.empty:
            logger.error(f"Empty candles DataFrame for {symbol}. Raw candles: {candles_raw}")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

        def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
            col_lookup = {str(col).strip().lower(): col for col in df.columns}
            rename_case: dict[str, str] = {}
            for lower_name, actual_name in col_lookup.items():
                if lower_name in ("open", "high", "low", "close", "volume"):
                    rename_case[actual_name] = lower_name.capitalize()
                elif lower_name in ("datetime", "datetimems", "date", "timestamp"):
                    rename_case[actual_name] = "Datetime"
            if rename_case:
                df = df.rename(columns=rename_case)

            rename = {
                "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
                "datetime": "Datetime", "datetimeMs": "Datetime",
            }
            for k_old, k_new in rename.items():
                if k_old in df.columns:
                    df = df.rename(columns={k_old: k_new})
            return df

        candles_df = _normalize_columns(candles_df)

        # Convert ms-epoch to timezone-aware datetime index (UTC)
        if "Datetime" in candles_df.columns:
            dt_series = pd.to_datetime(candles_df["Datetime"], unit="ms", utc=True)
            candles_df.index = dt_series
            candles_df.drop(columns=["Datetime"], inplace=True, errors="ignore")
        elif "datetime" in candles_df.columns:
            dt_series = pd.to_datetime(candles_df["datetime"], errors="coerce", utc=True)
            candles_df.index = dt_series
            candles_df.drop(columns=["datetime"], inplace=True, errors="ignore")
        elif "date" in candles_df.columns:
            dt_series = pd.to_datetime(candles_df["date"], errors="coerce", utc=True)
            candles_df.index = dt_series
            candles_df.drop(columns=["date"], inplace=True, errors="ignore")

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in candles_df.columns]

        # If columns are still missing but the raw payload is a dict/list of dicts, rebuild and retry once.
        if missing and candles_raw:
            alt_df = None
            if isinstance(candles_raw, dict):
                alt_df = pd.DataFrame([candles_raw])
            elif isinstance(candles_raw, list) and all(isinstance(item, dict) for item in candles_raw):
                alt_df = pd.DataFrame(candles_raw)
            if alt_df is not None:
                candles_df = _normalize_columns(alt_df)
                missing = [c for c in required_cols if c not in candles_df.columns]

        # Keep only the OHLCV columns, as floats
        if missing:
            logger.error(
                f"Missing OHLCV columns {missing} in price history for {symbol}. Columns={list(candles_df.columns)}",
            )
            logger.debug(
                f"Raw candles sample for {symbol}: {candles_raw if isinstance(candles_raw, (list, dict)) else str(candles_raw)[:500]}"
            )
            return pd.DataFrame(columns=required_cols)
        candles_df = candles_df[required_cols].astype(float)
        candles_df.index.name = None
        return candles_df

    def history(
        self,
        tickers: List[str],
        period: str = "1y",
        *,
        start_date: Optional[object] = None,
        end_date: Optional[object] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        """Fetch and stack daily OHLCV for all `tickers` into a single MultiIndex DataFrame."""
        frames = []
        for t in tickers:
            _maybe_cancel(cancel_token)
            sub = self._one_symbol_daily(
                t,
                start_date=start_date,
                end_date=end_date,
                cancel_token=cancel_token,
            )
            if sub.empty:
                continue
            # Promote per-ticker columns to level-0 (ticker), level-1 (field)
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)

        if not frames:
            raise ValueError("Schwab provider: no data returned for any ticker.")

        out = pd.concat(frames, axis=1).sort_index()
        out = out.dropna(how="all", axis=0)  # remove completely empty rows
        return out


class YFinanceProvider(DataProvider):
    """`yfinance` fallback provider with the same MultiIndex OHLCV contract."""
    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust  # if True, OHLC are back-adjusted for dividends/splits

    def history(
        self,
        tickers: List[str],
        period: str = "1y",
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> pd.DataFrame:
        import yfinance as yf
        _maybe_cancel(cancel_token)
        data = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1d",
            auto_adjust=self.auto_adjust,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        _maybe_cancel(cancel_token)
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker shape already
            frames = []
            for t in tickers:
                if t not in data.columns.levels[0]:
                    continue
                sub = data[t][["Open", "High", "Low", "Close", "Volume"]].copy()
                sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
                frames.append(sub)
            if not frames:
                raise ValueError("No data returned for any ticker.")
            out = pd.concat(frames, axis=1).sort_index()
        else:
            # Single-ticker: lift to MultiIndex
            sub = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.columns = pd.MultiIndex.from_product([[tickers[0]], sub.columns])
            out = sub
        return out.dropna(how="all", axis=0)


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

    # Ensure Schwab session is fresh
    refresh_tokens()

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
            md = MarketData()
            provider = SchwabMarketDataProvider(md, use_extended_hours=False, years=1)
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

    # Ensure Schwab session is fresh (your helper rotates tokens/refreshes auth)
    refresh_tokens()

    # Build universe and choose a provider
    # tickers = load_tickers_from_file(args.tickers_file)
    # sp_500_list = data_api.get_sp_500()
    tickers = _read_default_tickers()

    # Example: use Schwab provider (recommended for your setup)
    md = MarketData()
    provider = SchwabMarketDataProvider(md, use_extended_hours=False, years=1)

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
