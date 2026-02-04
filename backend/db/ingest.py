from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import logging
import re
from datetime import datetime, date, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.db.models import (
    Asset,
    DataWatermark,
    FundamentalSnapshot,
    NewsItem,
    OptionChainSnapshot,
    OptionContract,
    OptionsSuggestionBatch,
    OptionsSuggestionItem,
    Politician,
    PoliticianFlowAgg,
    PoliticianTrade,
    PriceBar,
    SuggestionBatch,
    SuggestionItem,
)
from backend.db.session import get_session, init_db

logger = logging.getLogger(__name__)

_SUGGESTION_RE = re.compile(r"^(?P<horizon>[a-z0-9_\-]+)_suggestions_(?P<date>\d{8})\.csv$")
_LEGACY_SUGGESTION_RE = re.compile(r"^suggestions_(?P<date>\d{8})\.csv$")
_OPTIONS_RE = re.compile(r"^(?P<horizon>[a-z0-9_\-]+)_options_(?P<date>\d{8})\.json$")
_PRICE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<ticker>[A-Z0-9\.\-]+)_(?P<range>1day1minute|1year1daily)_price_history\.csv$")
_FUND_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<ticker>[A-Z0-9\.\-]+)_fundamentals\.csv$")
_NEWS_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<ticker>[A-Z0-9\.\-]+)_news\.csv$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_date_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_date_compact(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan"}:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return value.item() if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _bulk_insert_price_bars(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    engine = session.get_bind()
    dialect = engine.dialect.name if engine else ""
    if dialect == "postgresql":
        stmt = pg_insert(PriceBar).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["asset_id", "interval", "ts"])
        session.execute(stmt)
        return
    session.bulk_insert_mappings(PriceBar, rows)


def _get_or_create_asset(session, ticker: str, name: str | None = None, exchange: str | None = None) -> Asset:
    if not ticker:
        raise ValueError("Ticker is required to resolve asset.")
    asset = session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()
    if asset:
        if name and not asset.name:
            asset.name = name
        if exchange and not asset.exchange:
            asset.exchange = exchange
        return asset
    asset = Asset(ticker=ticker, name=name, exchange=exchange)
    session.add(asset)
    session.flush()
    return asset


def _find_pending_watermark(session, dataset: str, asset_id: int | None) -> DataWatermark | None:
    for obj in session.new:
        if isinstance(obj, DataWatermark) and obj.dataset == dataset and obj.asset_id == asset_id:
            return obj
    return None


def _normalize_ts(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _update_watermark(session, dataset: str, asset_id: int | None, ts: datetime | None) -> None:
    cache = session.info.setdefault("watermark_cache", {})
    key = (dataset, asset_id)
    rec = cache.get(key) or _find_pending_watermark(session, dataset, asset_id)
    if rec is None:
        rec = session.execute(
            select(DataWatermark).where(DataWatermark.dataset == dataset, DataWatermark.asset_id == asset_id)
        ).scalar_one_or_none()
    if rec is None:
        rec = DataWatermark(dataset=dataset, asset_id=asset_id, last_complete_at=ts)
        session.add(rec)
    cache[key] = rec
    ts_norm = _normalize_ts(ts)
    last_norm = _normalize_ts(rec.last_complete_at)
    if ts_norm is None:
        rec.last_complete_at = None
    elif last_norm is None or ts_norm > last_norm:
        rec.last_complete_at = ts_norm


def _normalize_suggestions_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" in df.columns:
        return df
    if "symbol" in df.columns:
        return df.rename(columns={"symbol": "ticker"})
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        return df.rename(columns={unnamed[0]: "ticker"})
    df.insert(0, "ticker", "")
    return df


def ingest_suggestions_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith(".csv"):
            continue
        match = _SUGGESTION_RE.match(path.name) or _LEGACY_SUGGESTION_RE.match(path.name)
        if not match:
            continue
        horizon = match.groupdict().get("horizon") or "short_term"
        as_of = _parse_date_compact(match.group("date"))
        df = pd.read_csv(path)
        df = _normalize_suggestions_df(df)
        columns = list(df.columns)
        batch = SuggestionBatch(
            batch_type=horizon,
            horizon=horizon,
            as_of_date=as_of,
            generated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            params={"columns": columns},
            source_file=str(path),
        )
        session.add(batch)
        session.flush()

        for _, row in df.iterrows():
            row_dict = {col: _safe_json_value(row.get(col)) for col in columns}
            ticker = str(row_dict.get("ticker") or "").strip()
            asset_id = None
            if ticker:
                asset_id = _get_or_create_asset(session, ticker).id
            item = SuggestionItem(
                batch_id=batch.id,
                asset_id=asset_id,
                ticker=ticker,
                action=row_dict.get("action"),
                horizon=row_dict.get("horizon"),
                thesis=row_dict.get("thesis"),
                entry=_safe_float(row_dict.get("entry")),
                target=_safe_float(row_dict.get("target")),
                stop=_safe_float(row_dict.get("stop")),
                conviction=row_dict.get("conviction"),
                score=_safe_float(row_dict.get("score")),
                sector=row_dict.get("sector"),
                volatility60=_safe_float(row_dict.get("volatility60")),
                adv20_dollar=_safe_float(row_dict.get("adv20_dollar")),
                features=row_dict,
            )
            session.add(item)
        _update_watermark(session, f"suggestions:{horizon}", None, batch.generated_at)
        total += len(df)
    session.commit()
    return total


def _parse_option_expiration(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _ensure_option_contract(session, ticker: str, payload: dict[str, Any]) -> OptionContract:
    symbol = payload.get("symbol")
    if not symbol:
        raise ValueError("Option symbol required.")
    existing = session.execute(select(OptionContract).where(OptionContract.occ_symbol == symbol)).scalar_one_or_none()
    if existing:
        return existing
    asset = _get_or_create_asset(session, ticker)
    contract = OptionContract(
        occ_symbol=symbol,
        underlying_asset_id=asset.id,
        put_call=payload.get("put_call") or "",
        expiration_date=_parse_option_expiration(payload.get("expiration")) or date.today(),
        strike=_safe_float(payload.get("strike")) or 0.0,
        description=payload.get("description"),
    )
    session.add(contract)
    session.flush()
    return contract


def _ensure_chain_snapshot(
    session,
    contract: OptionContract,
    payload: dict[str, Any],
    as_of_ts: datetime,
) -> OptionChainSnapshot:
    existing = session.execute(
        select(OptionChainSnapshot).where(
            OptionChainSnapshot.contract_id == contract.id,
            OptionChainSnapshot.as_of_ts == as_of_ts,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    snapshot = OptionChainSnapshot(
        contract_id=contract.id,
        as_of_ts=as_of_ts,
        bid=_safe_float(payload.get("bid")),
        ask=_safe_float(payload.get("ask")),
        mid=_safe_float(payload.get("mid")),
        mark=_safe_float(payload.get("mark")),
        last=_safe_float(payload.get("last")),
        open_interest=_safe_int(payload.get("open_interest")),
        volume=_safe_int(payload.get("volume")),
        delta=_safe_float(payload.get("delta")),
        gamma=_safe_float(payload.get("gamma")),
        theta=_safe_float(payload.get("theta")),
        vega=_safe_float(payload.get("vega")),
        rho=_safe_float(payload.get("rho")),
        implied_volatility=_safe_float(payload.get("implied_volatility")),
        in_the_money=payload.get("in_the_money"),
        spread=_safe_float(payload.get("spread")),
        spread_pct=_safe_float(payload.get("spread_pct")),
        break_even=_safe_float(payload.get("break_even")),
        moneyness_pct=_safe_float(payload.get("moneyness_pct")),
        days_to_expiration=_safe_int(payload.get("days_to_expiration")),
        target_dte=_safe_int(payload.get("target_dte")),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def ingest_options_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith(".json"):
            continue
        match = _OPTIONS_RE.match(path.name)
        if not match:
            continue
        horizon = match.group("horizon")
        as_of_date = _parse_date_compact(match.group("date"))
        as_of_ts = datetime.combine(as_of_date, datetime.min.time(), tzinfo=timezone.utc)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        options = payload.get("options", payload if isinstance(payload, list) else [])
        meta = payload.get("meta") if isinstance(payload, dict) else None
        batch = OptionsSuggestionBatch(
            horizon=horizon,
            as_of_ts=as_of_ts,
            generated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            params=meta,
            source_file=str(path),
        )
        session.add(batch)
        session.flush()

        for row in options:
            ticker = row.get("ticker") or ""
            asset_id = _get_or_create_asset(session, ticker).id if ticker else None
            call_snapshot_id = None
            put_snapshot_id = None
            best_call = row.get("best_call") or {}
            best_put = row.get("best_put") or {}
            if best_call:
                contract = _ensure_option_contract(session, ticker, best_call)
                snapshot = _ensure_chain_snapshot(session, contract, best_call, as_of_ts)
                call_snapshot_id = snapshot.id
            if best_put:
                contract = _ensure_option_contract(session, ticker, best_put)
                snapshot = _ensure_chain_snapshot(session, contract, best_put, as_of_ts)
                put_snapshot_id = snapshot.id
            item = OptionsSuggestionItem(
                batch_id=batch.id,
                asset_id=asset_id,
                ticker=ticker,
                action=row.get("action"),
                horizon=row.get("horizon"),
                thesis=row.get("thesis"),
                equity_entry=_safe_float(row.get("equity_entry")),
                underlying_price=_safe_float(row.get("underlying_price")),
                best_call_snapshot_id=call_snapshot_id,
                best_put_snapshot_id=put_snapshot_id,
                call_candidates=_safe_int(row.get("call_candidates")),
                put_candidates=_safe_int(row.get("put_candidates")),
                warnings=row.get("warnings"),
                chain_error=row.get("chain_error"),
            )
            session.add(item)
        _update_watermark(session, f"options:{horizon}", None, batch.generated_at)
        total += len(options)
    session.commit()
    return total


def _parse_trade_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _estimate_amount_min_max(trade: dict[str, Any]) -> float | None:
    amount_min = _safe_float(trade.get("amount_min"))
    amount_max = _safe_float(trade.get("amount_max"))
    if amount_min is None or amount_max is None:
        return None
    return (amount_min + amount_max) / 2.0


def _is_purchase(trade_type: str | None) -> bool:
    if not trade_type:
        return True
    lowered = trade_type.lower()
    if "sale" in lowered or "sell" in lowered:
        return False
    return True


def _aggregate_politician_flows(session, as_of_date: date, windows: Iterable[int]) -> None:
    trades = session.execute(select(PoliticianTrade)).scalars().all()
    for window in windows:
        cutoff = as_of_date.toordinal() - window
        agg: dict[int, dict[str, Any]] = {}
        for trade in trades:
            if not trade.transaction_date or not trade.asset_id:
                continue
            if trade.transaction_date.toordinal() <= cutoff:
                continue
            est = _estimate_amount_min_max(
                {
                    "amount_min": trade.amount_min,
                    "amount_max": trade.amount_max,
                    "transaction_type": trade.transaction_type,
                }
            )
            if est is None:
                continue
            is_buy = _is_purchase(trade.transaction_type)
            entry = agg.setdefault(
                trade.asset_id,
                {"net": 0.0, "buys": 0, "sells": 0, "pols": set()},
            )
            entry["net"] += est if is_buy else -est
            entry["buys"] += 1 if is_buy else 0
            entry["sells"] += 0 if is_buy else 1
            if trade.politician_id:
                entry["pols"].add(trade.politician_id)
        session.query(PoliticianFlowAgg).filter(
            PoliticianFlowAgg.as_of_date == as_of_date,
            PoliticianFlowAgg.window_days == window,
        ).delete()
        for asset_id, data in agg.items():
            session.add(
                PoliticianFlowAgg(
                    asset_id=asset_id,
                    window_days=window,
                    as_of_date=as_of_date,
                    net_amount_est=data["net"],
                    buys_count=data["buys"],
                    sells_count=data["sells"],
                    distinct_politicians=len(data["pols"]),
                )
            )


def ingest_politician_trades_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith(".json"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        trades = payload.get("trades", [])
        fetched_at = payload.get("fetched_at")
        fetched_ts = datetime.fromisoformat(fetched_at) if fetched_at else None
        for trade in trades:
            source_id = trade.get("source_id")
            if not source_id:
                continue
            exists = session.execute(
                select(PoliticianTrade).where(PoliticianTrade.source_id == source_id)
            ).scalar_one_or_none()
            if exists:
                continue
            politician_name = trade.get("politician") or "Unknown"
            politician = session.execute(
                select(Politician).where(Politician.name == politician_name)
            ).scalar_one_or_none()
            if not politician:
                politician = Politician(
                    name=politician_name,
                    chamber=trade.get("chamber"),
                    party=trade.get("party"),
                    state=trade.get("state"),
                )
                session.add(politician)
                session.flush()
            ticker = trade.get("ticker")
            asset_id = None
            if ticker:
                asset_id = _get_or_create_asset(session, ticker).id
            session.add(
                PoliticianTrade(
                    source=trade.get("source") or "",
                    source_id=source_id,
                    politician_id=politician.id,
                    transaction_date=_parse_trade_date(trade.get("transaction_date")),
                    report_date=_parse_trade_date(trade.get("report_date")),
                    filed_date=_parse_trade_date(trade.get("filed_date")),
                    asset_id=asset_id,
                    ticker=ticker,
                    asset_name=trade.get("asset_name"),
                    transaction_type=trade.get("transaction_type"),
                    owner=trade.get("owner"),
                    amount_min=_safe_float(trade.get("amount_min")),
                    amount_max=_safe_float(trade.get("amount_max")),
                    url=trade.get("url"),
                    comment=trade.get("comment"),
                )
            )
            total += 1
        if fetched_ts:
            _update_watermark(session, "politician_trades", None, fetched_ts)
            as_of_date = fetched_ts.date()
        else:
            as_of_date = date.today()
        _aggregate_politician_flows(session, as_of_date, windows=[7, 30, 90])
    session.commit()
    return total


def ingest_price_history_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith("_price_history.csv"):
            continue
        match = _PRICE_RE.match(path.name)
        if not match:
            continue
        ticker = match.group("ticker")
        range_key = match.group("range")
        interval = "1m" if range_key == "1day1minute" else "1d"
        asset = _get_or_create_asset(session, ticker)
        batch_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[int, str, datetime]] = set()
        file_count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                candle_raw = row.get("candles")
                if not candle_raw:
                    continue
                candle = ast.literal_eval(candle_raw)
                ts_ms = candle.get("datetime")
                if not ts_ms:
                    continue
                ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
                key = (asset.id, interval, ts)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                batch_rows.append(
                    {
                        "asset_id": asset.id,
                        "ticker": ticker,
                        "interval": interval,
                        "ts": ts,
                        "open": _safe_float(candle.get("open")),
                        "high": _safe_float(candle.get("high")),
                        "low": _safe_float(candle.get("low")),
                        "close": _safe_float(candle.get("close")),
                        "volume": _safe_int(candle.get("volume")),
                        "source_file": str(path),
                    }
                )
                file_count += 1
                if len(batch_rows) >= 2000:
                    _bulk_insert_price_bars(session, batch_rows)
                    batch_rows.clear()
        if batch_rows:
            _bulk_insert_price_bars(session, batch_rows)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.warning("Duplicate bars detected for %s; consider loading into an empty DB.", path.name)
        total += file_count
        _update_watermark(session, f"price_bar:{interval}", asset.id, datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    session.commit()
    return total


def ingest_fundamentals_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith("_fundamentals.csv"):
            continue
        match = _FUND_RE.match(path.name)
        if not match:
            continue
        as_of = _parse_date_ymd(match.group("date"))
        ticker = match.group("ticker")
        df = pd.read_csv(path)
        if df.empty:
            continue
        raw_row = df.iloc[0].to_dict()
        row = {k: _safe_json_value(v) for k, v in raw_row.items()}
        asset_id = _get_or_create_asset(session, ticker).id
        session.add(
            FundamentalSnapshot(
                asset_id=asset_id,
                ticker=ticker,
                as_of_date=as_of,
                data=row,
                source_file=str(path),
            )
        )
        total += 1
        _update_watermark(session, "fundamentals", asset_id, datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    session.commit()
    return total


def _parse_news_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None


def ingest_news_dir(session, dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or not path.name.endswith("_news.csv"):
            continue
        match = _NEWS_RE.match(path.name)
        if not match:
            continue
        ticker = match.group("ticker")
        asset_id = _get_or_create_asset(session, ticker).id
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                data_raw = row.get("data")
                if not data_raw:
                    continue
                data = ast.literal_eval(data_raw)
                published_at = _parse_news_datetime(data.get("date"))
                session.add(
                    NewsItem(
                        asset_id=asset_id,
                        ticker=ticker,
                        published_at=published_at,
                        title=data.get("title"),
                        url=data.get("news_url"),
                        source_name=data.get("source_name"),
                        sentiment=data.get("sentiment"),
                        data=data,
                        source_file=str(path),
                    )
                )
                total += 1
        _update_watermark(session, "news", asset_id, datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    session.commit()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Hart Quantitative Research files into the database.")
    parser.add_argument("--init-db", action="store_true", help="Create tables before ingesting.")
    parser.add_argument("--data-dir", type=str, default="", help="Root files/ directory (defaults to repo files/).")
    parser.add_argument("--suggestions", action="store_true", help="Ingest daily_suggestions CSVs.")
    parser.add_argument("--options", action="store_true", help="Ingest options_suggestions JSON.")
    parser.add_argument("--politician", action="store_true", help="Ingest politician_trades JSON.")
    parser.add_argument("--prices", action="store_true", help="Ingest price history CSVs.")
    parser.add_argument("--fundamentals", action="store_true", help="Ingest fundamentals CSVs.")
    parser.add_argument("--news", action="store_true", help="Ingest news CSVs.")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else _project_root() / "files"
    session = get_session()

    if args.init_db:
        init_db()

    run_all = not any(
        [
            args.suggestions,
            args.options,
            args.politician,
            args.prices,
            args.fundamentals,
            args.news,
        ]
    )

    if args.suggestions or run_all:
        ingest_suggestions_dir(session, data_root / "daily_suggestions")
    if args.options or run_all:
        ingest_options_dir(session, data_root / "options_suggestions")
    if args.politician or run_all:
        ingest_politician_trades_dir(session, data_root / "politician_trades")
    if args.prices or run_all:
        ingest_price_history_dir(session, data_root / "data")
    if args.fundamentals or run_all:
        ingest_fundamentals_dir(session, data_root / "data")
    if args.news or run_all:
        ingest_news_dir(session, data_root / "data")


if __name__ == "__main__":
    main()
