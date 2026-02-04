"""
Helpers for constructing Schwab Trader API payloads and storing/retrieving
auxiliary JSON data (latest movers news).

This module provides:
- Order payload builders (`design_order`, `design_stop_loss_order`)
- Market data payload builders (`design_get_historical_price`, `design_get_fundamentals_payload`,
  `design_get_quote_payload`)
- Convenience I/O helpers for a "latest movers news" JSON file
- Utility to normalize an HTTP response into a pandas DataFrame

Notes
-----
- These helpers only build payload dicts; they do not perform HTTP requests.
- Keep types consistent with Schwab's API expectations (e.g., numeric quantities and
  strings for enumerated fields).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger

# --------------------------------------------------------------------------- #
# Order builders
# --------------------------------------------------------------------------- #

OrderType = Literal[
    "MARKET",
    "LIMIT",
    "STOP",
    "STOP_LIMIT",
    "TRAILING_STOP",
    "MARKET_ON_CLOSE",
    "LIMIT_ON_CLOSE",
    "STOP_ON_QUOTE",
    "STOP_LIMIT_ON_QUOTE",
]
Instruction = Literal[
    "BUY",
    "SELL",
    "BUY_TO_COVER",
    "SELL_SHORT",
    "SELL_SHORT_EXEMPT",
]
OrderLegType = Literal["EQUITY", "OPTION", "MUTUAL_FUND", "CASH_EQUIVALENT", "FIXED_INCOME"]
AssetType = Literal["EQUITY", "OPTION", "MUTUAL_FUND", "CASH_EQUIVALENT", "FIXED_INCOME", "INDEX"]
Session = Literal["NORMAL", "AM", "PM", "SEAMLESS"]
Duration = Literal["DAY", "GOOD_TILL_CANCEL", "FILL_OR_KILL"]
ComplexOrderType = Literal["NONE", "COVERED", "VERTICAL", "BACK_RATIO", "CALENDAR", "DIAGONAL"]
TaxLotMethod = Literal["FIFO", "LIFO", "HIGH_COST", "LOW_COST", "AVERAGE_COST"]
PositionEffect = Literal["OPENING", "CLOSING"]
StrategyType = Literal["SINGLE", "TRIGGER", "OCO", "TRIGGER_OCO"]

def design_order(
    symbol: str,
    order_type: OrderType,
    instruction: Instruction,
    quantity: int | float,
    *,
    leg_id: int = 1,
    order_leg_type: OrderLegType = "EQUITY",
    asset_type: AssetType = "EQUITY",
    price: str | float | None = None,
    session: Session = "NORMAL",
    duration: Duration = "DAY",
    complex_order_strategy_type: ComplexOrderType = "NONE",
    tax_lot_method: TaxLotMethod = "FIFO",
    position_effect: PositionEffect = "OPENING",
    order_strategy_type: StrategyType = "SINGLE",
) -> dict[str, Any]:
    """
    Build a single-leg order payload.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g., "AAPL").
    order_type : OrderType
        Schwab order type (e.g., "MARKET", "LIMIT").
    instruction : Instruction
        Order side (e.g., "BUY", "SELL").
    quantity : int | float
        Number of shares/contracts.
    leg_id : int, default 1
        Leg identifier (for multi-leg strategies; kept here for consistency).
    order_leg_type : OrderLegType, default "EQUITY"
        Type of order leg.
    asset_type : AssetType, default "EQUITY"
        Instrument asset type.
    price : str | float | None, default None
        Limit/stop price when applicable to the order type. Sent as-is.
    session : Session, default "NORMAL"
        Trading session.
    duration : Duration, default "DAY"
        Time-in-force.
    complex_order_strategy_type : ComplexOrderType, default "NONE"
        Complex order strategy (if any).
    tax_lot_method : TaxLotMethod, default "FIFO"
        Tax-lot accounting preference.
    position_effect : PositionEffect, default "OPENING"
        Opening or closing transaction.
    order_strategy_type : StrategyType, default "SINGLE"
        Order strategy type.

    Returns
    -------
    dict
        JSON-serializable order payload.

    Raises
    ------
    ValueError
        If `quantity` is not positive.
    """
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    payload: dict[str, Any] = {
        "price": price,
        "session": session,
        "duration": duration,
        "orderType": order_type,
        "complexOrderStrategyType": complex_order_strategy_type,
        "quantity": quantity,
        "taxLotMethod": tax_lot_method,
        "orderLegCollection": [
            {
                "orderLegType": order_leg_type,
                "legId": leg_id,
                "instrument": {"symbol": symbol, "assetType": asset_type},
                "instruction": instruction,
                "positionEffect": position_effect,
                "quantity": quantity,
            }
        ],
        "orderStrategyType": order_strategy_type,
    }
    return payload


def design_stop_loss_order(
    symbol: str,
    order_type: OrderType,
    instruction: Instruction,
    quantity: int | float,
    *,
    leg_id: int = 1,
    order_leg_type: OrderLegType = "EQUITY",
    asset_type: AssetType = "EQUITY",
    stop_price: str | float | None = None,
    session: Session = "NORMAL",
    duration: Duration = "DAY",
    complex_order_strategy_type: ComplexOrderType = "NONE",
    tax_lot_method: TaxLotMethod = "FIFO",
    position_effect: PositionEffect = "OPENING",
    order_strategy_type: StrategyType = "SINGLE",
) -> dict[str, Any]:
    """
    Build a single-leg **stop** order payload.

    Parameters
    ----------
    stop_price : str | float | None, default None
        Stop trigger price.

    Other parameters
    ----------------
    See `design_order` for descriptions of the other fields.

    Returns
    -------
    dict
        JSON-serializable stop order payload.
    """
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    payload: dict[str, Any] = {
        "stopPrice": stop_price,
        "session": session,
        "duration": duration,
        "orderType": order_type,
        "complexOrderStrategyType": complex_order_strategy_type,
        "quantity": quantity,
        "taxLotMethod": tax_lot_method,
        "orderLegCollection": [
            {
                "orderLegType": order_leg_type,
                "legId": leg_id,
                "instrument": {"symbol": symbol, "assetType": asset_type},
                "instruction": instruction,
                "positionEffect": position_effect,
                "quantity": quantity,
            }
        ],
        "orderStrategyType": order_strategy_type,
    }
    return payload


# --------------------------------------------------------------------------- #
# Market data payload builders
# --------------------------------------------------------------------------- #

PeriodType = Literal["day", "month", "year", "ytd"]
FrequencyType = Literal["minute", "daily", "weekly", "monthly"]
ContractType = Literal["CALL", "PUT", "ALL"]
Strategy = Literal["SINGLE", "ANALYTICAL", "COVERED", "VERTICAL", "CALENDAR", "STRANGLE", "STRADDLE", "BUTTERFLY", "CONDOR", "DIAGONAL", "COLLAR", "ROLL"]


def design_get_historical_price(
    symbol: str,
    period_type: PeriodType,
    period: int,
    frequency_type: FrequencyType,
    frequency: int,
    *,
    start_date: int | None = None,
    end_date: int | None = None,
    need_extended_hours_data: bool = True,
    need_previous_close: bool = True,
) -> dict[str, Any]:
    """
    Build a `/pricehistory` query payload.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    period_type : {"day","month","year","ytd"}
    period : int
        Number of periods (API-dependent).
    frequency_type : {"minute","daily","weekly","monthly"}
    frequency : int
        Frequency within the `frequency_type`.
    start_date : int | None, default None
        Epoch milliseconds (optional).
    end_date : int | None, default None
        Epoch milliseconds (optional).
    need_extended_hours_data : bool, default True
        Include extended-hours data.
    need_previous_close : bool, default True
        Include previous close.

    Returns
    -------
    dict
        Dict suitable for `params=` on the pricehistory endpoint.
    """
    return {
        "symbol": symbol,
        "periodType": period_type,
        "period": period,
        "frequencyType": frequency_type,
        "frequency": frequency,
        "startDate": start_date,
        "endDate": end_date,
        "needExtendedHoursData": need_extended_hours_data,
        "needPreviousClose": need_previous_close,
    }


def design_get_fundamentals_payload(symbol: str, projection: str) -> dict[str, Any]:
    """
    Build an `/instruments` (fundamentals) query payload.

    Parameters
    ----------
    symbol : str
        Ticker symbol (or CSV string of symbols, per API).
    projection : str
        One of: "fundamental", "symbol-search", "symbol-regex",
        "desc-search", "desc-regex", "search".

    Returns
    -------
    dict
        Dict suitable for `params=` on the instruments endpoint.
    """
    return {"symbol": symbol, "projection": projection}


def design_get_quote_payload(
    symbols: Sequence[str] | str, indicative: bool = False
) -> dict[str, Any]:
    """
    Build a `/quotes` query payload.

    Parameters
    ----------
    symbols : Sequence[str] | str
        Symbols as a list/tuple OR a comma-separated string.
    indicative : bool, default False
        If True, include indicative ETF symbol quotes (e.g., $ABC.IV).

    Returns
    -------
    dict
        Dict suitable for `params=` on the quotes endpoint.
    """
    return {"symbols": symbols, "indicative": indicative}


def design_get_options_chain_payload(
    symbol: str,
    *,
    contract_type: ContractType = "ALL",
    strike_count: int | None = 8,
    include_underlying_quote: bool = True,
    strategy: Strategy = "SINGLE",
    interval: float | None = None,
    strike: float | None = None,
    strike_range: str | None = "NTM",
    from_date: str | None = None,
    to_date: str | None = None,
    volatility: float | None = None,
    underlying_price: float | None = None,
    interest_rate: float | None = None,
    days_to_expiration: int | None = None,
    exp_month: str | None = "ALL",
    option_type: str | None = None,
    entitlement: str | None = None,
) -> dict[str, Any]:
    """
    Build a `/chains` query payload.

    Parameters
    ----------
    symbol : str
        Symbol as a string like "AAPL".
    contract_type : {"CALL","PUT","ALL"}, default "ALL"
        Which side of the chain to return.
    strike_count : int | None, default 8
        Number of strikes to include above/below the at-the-money price.
    include_underlying_quote : bool, default True
        Whether to include the underlying quote in the response.
    strategy : Strategy, default "SINGLE"
        Option chain strategy for the response payload.
    interval : float | None, default None
        Strike interval for spread strategies (if applicable).
    strike : float | None, default None
        Explicit strike price filter.
    strike_range : str | None, default "NTM"
        Range filter (e.g., "ITM", "OTM", "NTM", "ALL").
    from_date, to_date : str | None, default None
        Expiration date bounds in "YYYY-MM-DD" format.
    volatility, underlying_price, interest_rate, days_to_expiration : optional
        Analytical parameters (used for certain strategies).
    exp_month : str | None, default "ALL"
        Expiration month filter.
    option_type : str | None, default None
        Additional option type filter (if supported).
    entitlement : str | None, default None
        Entitlement flag (PN/NP/PP) for retail tokens.

    Returns
    -------
    dict
        Dict suitable for `params=` on the chains endpoint with falsy/None values removed.
    """
    payload: dict[str, Any] = {
        "symbol": symbol,
        "contractType": contract_type,
        "strikeCount": strike_count,
        "includeUnderlyingQuote": include_underlying_quote,
        "strategy": strategy,
        "interval": interval,
        "strike": strike,
        "range": strike_range,
        "fromDate": from_date,
        "toDate": to_date,
        "volatility": volatility,
        "underlyingPrice": underlying_price,
        "interestRate": interest_rate,
        "daysToExpiration": days_to_expiration,
        "expMonth": exp_month,
        "optionType": option_type,
        "entitlement": entitlement,
    }

    # Strip out `None` values so the query stays minimal
    return {k: v for k, v in payload.items() if v is not None}



# --------------------------------------------------------------------------- #
# HTTP response normalization
# --------------------------------------------------------------------------- #


def return_dataframe_from_response(response: Any) -> pd.DataFrame:
    """
    Normalize a requests.Response (or lookalike) into a pandas DataFrame.

    Parameters
    ----------
    response : Any
        Expected to have `.status_code`, `.json()` attributes.

    Returns
    -------
    pandas.DataFrame
        DataFrame on 200; empty DataFrame otherwise.

    Notes
    -----
    - If the JSON body is a list of objects, returns a standard table.
    - If the JSON body is a dict, returns a 1-row DataFrame.
    """
    try:
        status = getattr(response, "status_code", None)
        if status != 200:
            logger.error(f"Non-200 response: status={status}")
            return pd.DataFrame()
        data = response.json()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        logger.error(f"Unexpected JSON type: {type(data)}")
        return pd.DataFrame()
    except Exception as exc:
        logger.error(f"Failed to normalize response: {exc}")
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Latest movers news helpers
# --------------------------------------------------------------------------- #

# Keep the JSON alongside this module in ../helper_files
_HELPER_DIR = (Path(__file__).resolve().parent / "../helper_files").resolve()
_HELPER_DIR.mkdir(parents=True, exist_ok=True)
_LATEST_NEWS_PATH = _HELPER_DIR / "latest_movers_news.json"


def save_latest_stock_news(ticker_news_frame: pd.DataFrame) -> None:
    """
    Persist the latest movers news DataFrame to JSON on disk.

    Parameters
    ----------
    ticker_news_frame : pandas.DataFrame
        DataFrame to serialize as JSON.

    Side Effects
    ------------
    - Writes to `../helper_files/latest_movers_news.json` relative to this file.
    """
    try:
        ticker_news_frame.to_json(_LATEST_NEWS_PATH)
        logger.debug("Updated latest_movers_news.json.")
    except FileNotFoundError:
        logger.error(f"File not found: {_LATEST_NEWS_PATH}")
    except (TypeError, ValueError) as exc:
        logger.error(f"Failed to serialize DataFrame to JSON: {exc}")


def get_existing_latest_stock_news() -> pd.DataFrame | None:
    """
    Load the previously saved movers news JSON (if present).

    Returns
    -------
    pandas.DataFrame | None
        DataFrame if the file exists and is valid JSON; otherwise None.
    """
    try:
        if not _LATEST_NEWS_PATH.exists():
            logger.warning(f"News file does not exist: {_LATEST_NEWS_PATH}")
            return None
        frame = pd.read_json(_LATEST_NEWS_PATH)
        logger.debug("Retrieved latest_movers_news.")
        return frame
    except FileNotFoundError:
        logger.error(f"File not found: {_LATEST_NEWS_PATH}")
        return None
    except ValueError as exc:
        logger.error(f"Invalid JSON in {_LATEST_NEWS_PATH}: {exc}")
        return None
