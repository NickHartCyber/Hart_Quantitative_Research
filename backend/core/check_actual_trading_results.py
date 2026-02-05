"""Minimal helpers for order history analysis."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger


def get_orders(accounts_trading: Any, *, days: int = 30) -> pd.DataFrame:
    """
    Fetch recent orders via AccountsTrading; fall back to empty DataFrame on failure.
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, int(days)))
        if hasattr(accounts_trading, "get_orders_by_date"):
            resp = accounts_trading.get_orders_by_date(start.isoformat(), end.isoformat())
        else:
            resp = accounts_trading.get_recent_orders()
        if isinstance(resp, pd.DataFrame):
            return resp
    except Exception as exc:
        logger.warning(f"Failed to fetch orders: {exc}")
    return pd.DataFrame()


def analyze_order_history(orders_df: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight summary scaffolding when detailed analytics are unavailable.
    """
    empty_df = pd.DataFrame()
    metrics = {
        "total_orders": int(len(orders_df)) if hasattr(orders_df, "__len__") else 0,
        "total_trades": 0,
        "win_rate": None,
        "net_pnl": 0.0,
    }
    return {
        "trades_df": empty_df,
        "per_symbol": empty_df,
        "equity_curve": empty_df,
        "daily_pnl": pd.Series(dtype=float),
        "metrics": metrics,
    }
