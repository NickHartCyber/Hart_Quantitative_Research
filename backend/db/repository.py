from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import desc, func, select

from backend.db.models import (
    Asset,
    DataWatermark,
    OptionChainSnapshot,
    OptionContract,
    OptionsSuggestionBatch,
    OptionsSuggestionItem,
    Politician,
    PoliticianTrade,
    SuggestionBatch,
    SuggestionItem,
)


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


def _get_or_create_asset(session, ticker: str, name: str | None = None, exchange: str | None = None) -> Asset:
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


def list_suggestion_batches(session, horizon: str, limit: int | None = None) -> list[dict[str, Any]]:
    count_subq = (
        select(
            SuggestionItem.batch_id.label("batch_id"),
            func.count(SuggestionItem.id).label("item_count"),
        )
        .group_by(SuggestionItem.batch_id)
        .subquery()
    )
    rows = session.execute(
        select(
            SuggestionBatch,
            func.coalesce(count_subq.c.item_count, 0).label("item_count"),
        )
        .outerjoin(count_subq, count_subq.c.batch_id == SuggestionBatch.id)
        .where(SuggestionBatch.batch_type == horizon)
    ).all()
    entries_by_date: dict[str, dict[str, Any]] = {}
    for batch, item_count in rows:
        date_str = batch.as_of_date.strftime("%Y%m%d")
        last_run = batch.generated_at or batch.created_at
        next_entry = {
            "date": date_str,
            "label": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            "batch_id": batch.id,
            "last_run": last_run.isoformat() if last_run else None,
            "count": int(item_count or 0),
            "_last_run_ts": last_run,
        }
        existing = entries_by_date.get(date_str)
        if not existing:
            entries_by_date[date_str] = next_entry
            continue
        if next_entry["count"] > existing["count"]:
            entries_by_date[date_str] = next_entry
            continue
        if next_entry["count"] == existing["count"]:
            if next_entry["_last_run_ts"] and (
                not existing.get("_last_run_ts") or next_entry["_last_run_ts"] > existing["_last_run_ts"]
            ):
                entries_by_date[date_str] = next_entry

    entries = sorted(entries_by_date.values(), key=lambda item: item["date"], reverse=True)
    if limit is not None:
        entries = entries[: max(0, limit)]
    for entry in entries:
        entry.pop("_last_run_ts", None)
    return entries


def _suggestion_items_payload(batch: SuggestionBatch, items: list[SuggestionItem]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    columns = []
    if isinstance(batch.params, dict):
        columns = batch.params.get("columns") or []
    for item in items:
        if isinstance(item.features, dict):
            record = dict(item.features)
            record.setdefault("ticker", item.ticker)
        else:
            record = {
                "ticker": item.ticker,
                "action": item.action,
                "horizon": item.horizon,
                "thesis": item.thesis,
                "entry": item.entry,
                "target": item.target,
                "stop": item.stop,
                "conviction": item.conviction,
                "score": item.score,
                "sector": item.sector,
                "volatility60": item.volatility60,
                "adv20_dollar": item.adv20_dollar,
            }
        records.append(record)
        if not columns:
            columns = list(record.keys())
    last_run = batch.generated_at or batch.created_at
    return {
        "suggestions": records,
        "columns": columns,
        "count": len(records),
        "last_run": last_run.isoformat() if last_run else None,
        "cached": True,
        "horizon": batch.horizon or batch.batch_type,
    }


def load_latest_suggestions(session, horizon: str) -> dict[str, Any] | None:
    batch = session.execute(
        select(SuggestionBatch)
        .where(SuggestionBatch.batch_type == horizon)
        .order_by(desc(SuggestionBatch.as_of_date))
        .limit(1)
    ).scalar_one_or_none()
    if not batch:
        return None
    items = session.execute(select(SuggestionItem).where(SuggestionItem.batch_id == batch.id)).scalars().all()
    return _suggestion_items_payload(batch, items)


def load_suggestions_by_date(session, horizon: str, target_date: date) -> dict[str, Any] | None:
    count_subq = (
        select(
            SuggestionItem.batch_id.label("batch_id"),
            func.count(SuggestionItem.id).label("item_count"),
        )
        .group_by(SuggestionItem.batch_id)
        .subquery()
    )
    item_count = func.coalesce(count_subq.c.item_count, 0).label("item_count")
    row = session.execute(
        select(SuggestionBatch, item_count)
        .outerjoin(count_subq, count_subq.c.batch_id == SuggestionBatch.id)
        .where(SuggestionBatch.batch_type == horizon, SuggestionBatch.as_of_date == target_date)
        .order_by(desc(item_count), desc(SuggestionBatch.generated_at), desc(SuggestionBatch.created_at))
        .limit(1)
    ).first()
    if not row:
        return None
    batch = row[0]
    items = session.execute(select(SuggestionItem).where(SuggestionItem.batch_id == batch.id)).scalars().all()
    return _suggestion_items_payload(batch, items)


def list_options_batches(session, horizon: str, limit: int | None = None) -> list[dict[str, Any]]:
    count_subq = (
        select(
            OptionsSuggestionItem.batch_id.label("batch_id"),
            func.count(OptionsSuggestionItem.id).label("item_count"),
        )
        .group_by(OptionsSuggestionItem.batch_id)
        .subquery()
    )
    rows = session.execute(
        select(
            OptionsSuggestionBatch,
            func.coalesce(count_subq.c.item_count, 0).label("item_count"),
        )
        .outerjoin(count_subq, count_subq.c.batch_id == OptionsSuggestionBatch.id)
        .where(OptionsSuggestionBatch.horizon == horizon)
    ).all()
    entries_by_date: dict[str, dict[str, Any]] = {}
    for batch, item_count in rows:
        date_str = batch.as_of_ts.strftime("%Y%m%d")
        last_run = batch.generated_at or batch.created_at
        next_entry = {
            "date": date_str,
            "label": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            "count": int(item_count or 0),
            "_last_run_ts": last_run,
        }
        existing = entries_by_date.get(date_str)
        if not existing:
            entries_by_date[date_str] = next_entry
            continue
        if next_entry["count"] > existing["count"]:
            entries_by_date[date_str] = next_entry
            continue
        if next_entry["count"] == existing["count"]:
            if next_entry["_last_run_ts"] and (
                not existing.get("_last_run_ts") or next_entry["_last_run_ts"] > existing["_last_run_ts"]
            ):
                entries_by_date[date_str] = next_entry

    entries = sorted(entries_by_date.values(), key=lambda item: item["date"], reverse=True)
    if limit is not None:
        entries = entries[: max(0, limit)]
    for entry in entries:
        entry.pop("_last_run_ts", None)
    return entries


def _snapshot_payload(snapshot: OptionChainSnapshot, contract: OptionContract) -> dict[str, Any]:
    expiration_ts = datetime.combine(contract.expiration_date, datetime.min.time(), tzinfo=timezone.utc)
    return {
        "put_call": contract.put_call,
        "symbol": contract.occ_symbol,
        "strike": contract.strike,
        "expiration": expiration_ts.isoformat(),
        "days_to_expiration": snapshot.days_to_expiration,
        "target_dte": snapshot.target_dte,
        "bid": snapshot.bid,
        "ask": snapshot.ask,
        "mid": snapshot.mid,
        "mark": snapshot.mark,
        "last": snapshot.last,
        "open_interest": snapshot.open_interest,
        "volume": snapshot.volume,
        "delta": snapshot.delta,
        "gamma": snapshot.gamma,
        "theta": snapshot.theta,
        "vega": snapshot.vega,
        "rho": snapshot.rho,
        "implied_volatility": snapshot.implied_volatility,
        "in_the_money": snapshot.in_the_money,
        "spread": snapshot.spread,
        "spread_pct": snapshot.spread_pct,
        "break_even": snapshot.break_even,
        "moneyness_pct": snapshot.moneyness_pct,
        "description": contract.description,
    }


def load_options_by_date(session, horizon: str, target_date: date) -> dict[str, Any] | None:
    count_subq = (
        select(
            OptionsSuggestionItem.batch_id.label("batch_id"),
            func.count(OptionsSuggestionItem.id).label("item_count"),
        )
        .group_by(OptionsSuggestionItem.batch_id)
        .subquery()
    )
    item_count = func.coalesce(count_subq.c.item_count, 0).label("item_count")
    row = session.execute(
        select(OptionsSuggestionBatch, item_count)
        .outerjoin(count_subq, count_subq.c.batch_id == OptionsSuggestionBatch.id)
        .where(OptionsSuggestionBatch.horizon == horizon, func.date(OptionsSuggestionBatch.as_of_ts) == target_date)
        .order_by(desc(item_count), desc(OptionsSuggestionBatch.generated_at), desc(OptionsSuggestionBatch.created_at))
        .limit(1)
    ).first()
    if not row:
        return None
    batch = row[0]
    items = session.execute(select(OptionsSuggestionItem).where(OptionsSuggestionItem.batch_id == batch.id)).scalars().all()
    snapshot_ids = {item.best_call_snapshot_id for item in items if item.best_call_snapshot_id}
    snapshot_ids.update({item.best_put_snapshot_id for item in items if item.best_put_snapshot_id})
    snapshots = (
        session.execute(select(OptionChainSnapshot).where(OptionChainSnapshot.id.in_(snapshot_ids))).scalars().all()
        if snapshot_ids
        else []
    )
    snapshot_map = {snap.id: snap for snap in snapshots}
    contract_ids = {snap.contract_id for snap in snapshots}
    contracts = (
        session.execute(select(OptionContract).where(OptionContract.id.in_(contract_ids))).scalars().all()
        if contract_ids
        else []
    )
    contract_map = {contract.id: contract for contract in contracts}

    records = []
    for item in items:
        payload = {
            "ticker": item.ticker,
            "action": item.action,
            "horizon": item.horizon,
            "thesis": item.thesis,
            "equity_entry": item.equity_entry,
            "underlying_price": item.underlying_price,
            "call_candidates": item.call_candidates,
            "put_candidates": item.put_candidates,
            "warnings": item.warnings or [],
            "chain_error": item.chain_error,
        }
        if item.best_call_snapshot_id:
            snap = snapshot_map.get(item.best_call_snapshot_id)
            if snap:
                contract = contract_map.get(snap.contract_id)
                if contract:
                    payload["best_call"] = _snapshot_payload(snap, contract)
        if item.best_put_snapshot_id:
            snap = snapshot_map.get(item.best_put_snapshot_id)
            if snap:
                contract = contract_map.get(snap.contract_id)
                if contract:
                    payload["best_put"] = _snapshot_payload(snap, contract)
        records.append(payload)
    last_run = batch.generated_at or batch.created_at
    return {
        "date": batch.as_of_ts.strftime("%Y%m%d"),
        "label": batch.as_of_ts.strftime("%Y-%m-%d"),
        "options": records,
        "meta": batch.params or {},
        "count": len(records),
        "last_run": last_run.isoformat() if last_run else None,
        "cached": True,
    }


def load_latest_options(session, horizon: str) -> dict[str, Any] | None:
    batch = session.execute(
        select(OptionsSuggestionBatch)
        .where(OptionsSuggestionBatch.horizon == horizon)
        .order_by(desc(OptionsSuggestionBatch.as_of_ts))
        .limit(1)
    ).scalar_one_or_none()
    if not batch:
        return None
    return load_options_by_date(session, horizon, batch.as_of_ts.date())


def load_politician_trades(session, days: int, limit: int) -> dict[str, Any]:
    since_date = date.today() - timedelta(days=max(1, days))
    trades = session.execute(
        select(PoliticianTrade, Politician)
        .join(Politician, Politician.id == PoliticianTrade.politician_id)
        .where(PoliticianTrade.transaction_date >= since_date)
        .order_by(desc(PoliticianTrade.transaction_date))
        .limit(limit)
    ).all()
    rows = []
    sources = set()
    for trade, pol in trades:
        sources.add(trade.source)
        rows.append(
            {
                "source": trade.source,
                "source_id": trade.source_id,
                "chamber": pol.chamber,
                "politician": pol.name,
                "party": pol.party,
                "state": pol.state,
                "transaction_date": trade.transaction_date.isoformat() if trade.transaction_date else None,
                "report_date": trade.report_date.isoformat() if trade.report_date else None,
                "filed_date": trade.filed_date.isoformat() if trade.filed_date else None,
                "ticker": trade.ticker,
                "asset_name": trade.asset_name,
                "transaction_type": trade.transaction_type,
                "owner": trade.owner,
                "amount_min": trade.amount_min,
                "amount_max": trade.amount_max,
                "comment": trade.comment,
                "url": trade.url,
            }
        )
    watermark = session.execute(
        select(DataWatermark).where(DataWatermark.dataset == "politician_trades")
    ).scalar_one_or_none()
    fetched_at = watermark.last_complete_at.isoformat() if watermark and watermark.last_complete_at else None
    return {
        "days": max(1, days),
        "fetched_at": fetched_at,
        "sources": sorted(sources),
        "count": len(rows),
        "trades": rows,
    }


def save_suggestions(session, df, horizon: str, generated_at: datetime | None = None, source_file: str | None = None) -> str:
    if hasattr(df, "to_dict"):
        records = df.to_dict(orient="records")
        columns = list(df.columns)
    else:
        records = list(df)
        columns = list(records[0].keys()) if records else []
    as_of_date = (generated_at or datetime.now(timezone.utc)).date()
    batch = SuggestionBatch(
        batch_type=horizon,
        horizon=horizon,
        as_of_date=as_of_date,
        generated_at=generated_at or datetime.now(timezone.utc),
        params={"columns": columns},
        source_file=source_file,
    )
    session.add(batch)
    session.flush()
    for record in records:
        record = {k: _safe_json_value(v) for k, v in record.items()}
        ticker = str(record.get("ticker") or "").strip()
        asset_id = None
        if ticker:
            asset_id = _get_or_create_asset(session, ticker).id
        session.add(
            SuggestionItem(
                batch_id=batch.id,
                asset_id=asset_id,
                ticker=ticker,
                action=record.get("action"),
                horizon=record.get("horizon"),
                thesis=record.get("thesis"),
                entry=_safe_float(record.get("entry")),
                target=_safe_float(record.get("target")),
                stop=_safe_float(record.get("stop")),
                conviction=record.get("conviction"),
                score=_safe_float(record.get("score")),
                sector=record.get("sector"),
                volatility60=_safe_float(record.get("volatility60")),
                adv20_dollar=_safe_float(record.get("adv20_dollar")),
                features=record,
            )
        )
    _update_watermark(session, f"suggestions:{horizon}", None, generated_at or datetime.now(timezone.utc))
    session.commit()
    return batch.id


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


def save_options(
    session,
    rows: list[dict[str, Any]],
    horizon: str,
    meta: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
    source_file: str | None = None,
) -> str:
    as_of_ts = (generated_at or datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
    batch = OptionsSuggestionBatch(
        horizon=horizon,
        as_of_ts=as_of_ts,
        generated_at=generated_at or datetime.now(timezone.utc),
        params=meta,
        source_file=source_file,
    )
    session.add(batch)
    session.flush()
    for row in rows:
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
        session.add(
            OptionsSuggestionItem(
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
        )
    _update_watermark(session, f"options:{horizon}", None, generated_at or datetime.now(timezone.utc))
    session.commit()
    return batch.id


def _parse_trade_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def save_politician_trades(session, trades: list[dict[str, Any]], fetched_at: datetime | None = None) -> int:
    inserted = 0
    for trade in trades:
        source_id = trade.get("source_id")
        if not source_id:
            continue
        exists = session.execute(select(PoliticianTrade).where(PoliticianTrade.source_id == source_id)).scalar_one_or_none()
        if exists:
            continue
        politician_name = trade.get("politician") or "Unknown"
        politician = session.execute(select(Politician).where(Politician.name == politician_name)).scalar_one_or_none()
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
        asset_id = _get_or_create_asset(session, ticker).id if ticker else None
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
        inserted += 1
    _update_watermark(session, "politician_trades", None, fetched_at)
    session.commit()
    return inserted
