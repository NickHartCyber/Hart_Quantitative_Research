from __future__ import annotations

import uuid
from datetime import datetime, date

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "user_account"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    subscription_tier: Mapped[str] = mapped_column(String(32), index=True)
    subscription_status: Mapped[str] = mapped_column(String(32), default="active")
    stripe_customer_id: Mapped[str | None] = mapped_column(String(80), unique=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(80), unique=True)
    payment_brand: Mapped[str | None] = mapped_column(String(24))
    payment_last4: Mapped[str | None] = mapped_column(String(4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Asset(Base):
    __tablename__ = "asset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200))
    exchange: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DataWatermark(Base):
    __tablename__ = "data_watermark"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(64), index=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    last_complete_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("dataset", "asset_id", name="uq_watermark_dataset_asset"),)


class SuggestionBatch(Base):
    __tablename__ = "suggestion_batch"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_type: Mapped[str] = mapped_column(String(40), index=True)
    horizon: Mapped[str | None] = mapped_column(String(40), index=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    universe: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    params: Mapped[dict | None] = mapped_column(JSON)
    source_file: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_suggestion_batch_as_of", "batch_type", "as_of_date"),)


class SuggestionItem(Base):
    __tablename__ = "suggestion_item"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("suggestion_batch.id"), index=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str | None] = mapped_column(String(16))
    horizon: Mapped[str | None] = mapped_column(String(40))
    thesis: Mapped[str | None] = mapped_column(Text)
    entry: Mapped[float | None] = mapped_column(Float)
    target: Mapped[float | None] = mapped_column(Float)
    stop: Mapped[float | None] = mapped_column(Float)
    conviction: Mapped[str | None] = mapped_column(String(24))
    score: Mapped[float | None] = mapped_column(Float)
    sector: Mapped[str | None] = mapped_column(String(64))
    volatility60: Mapped[float | None] = mapped_column(Float)
    adv20_dollar: Mapped[float | None] = mapped_column(Float)
    features: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_suggestion_item_batch_ticker", "batch_id", "ticker"),)


class OptionContract(Base):
    __tablename__ = "option_contract"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    occ_symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    underlying_asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    put_call: Mapped[str] = mapped_column(String(4))
    expiration_date: Mapped[date] = mapped_column(Date, index=True)
    strike: Mapped[float] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(String(120))


class OptionChainSnapshot(Base):
    __tablename__ = "option_chain_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contract_id: Mapped[int] = mapped_column(ForeignKey("option_contract.id"), index=True)
    as_of_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    bid: Mapped[float | None] = mapped_column(Float)
    ask: Mapped[float | None] = mapped_column(Float)
    mid: Mapped[float | None] = mapped_column(Float)
    mark: Mapped[float | None] = mapped_column(Float)
    last: Mapped[float | None] = mapped_column(Float)
    open_interest: Mapped[int | None] = mapped_column(Integer)
    volume: Mapped[int | None] = mapped_column(Integer)
    delta: Mapped[float | None] = mapped_column(Float)
    gamma: Mapped[float | None] = mapped_column(Float)
    theta: Mapped[float | None] = mapped_column(Float)
    vega: Mapped[float | None] = mapped_column(Float)
    rho: Mapped[float | None] = mapped_column(Float)
    implied_volatility: Mapped[float | None] = mapped_column(Float)
    in_the_money: Mapped[bool | None] = mapped_column(Boolean)
    spread: Mapped[float | None] = mapped_column(Float)
    spread_pct: Mapped[float | None] = mapped_column(Float)
    break_even: Mapped[float | None] = mapped_column(Float)
    moneyness_pct: Mapped[float | None] = mapped_column(Float)
    days_to_expiration: Mapped[int | None] = mapped_column(Integer)
    target_dte: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("contract_id", "as_of_ts", name="uq_option_chain_contract_as_of"),
        Index("ix_option_chain_contract_as_of", "contract_id", "as_of_ts"),
    )


class OptionsSuggestionBatch(Base):
    __tablename__ = "options_suggestion_batch"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    horizon: Mapped[str | None] = mapped_column(String(40), index=True)
    as_of_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    params: Mapped[dict | None] = mapped_column(JSON)
    source_file: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OptionsSuggestionItem(Base):
    __tablename__ = "options_suggestion_item"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("options_suggestion_batch.id"), index=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str | None] = mapped_column(String(16))
    horizon: Mapped[str | None] = mapped_column(String(40))
    thesis: Mapped[str | None] = mapped_column(Text)
    equity_entry: Mapped[float | None] = mapped_column(Float)
    underlying_price: Mapped[float | None] = mapped_column(Float)
    best_call_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("option_chain_snapshot.id"))
    best_put_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("option_chain_snapshot.id"))
    call_candidates: Mapped[int | None] = mapped_column(Integer)
    put_candidates: Mapped[int | None] = mapped_column(Integer)
    warnings: Mapped[list | None] = mapped_column(JSON)
    chain_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Politician(Base):
    __tablename__ = "politician"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    chamber: Mapped[str | None] = mapped_column(String(32))
    party: Mapped[str | None] = mapped_column(String(16))
    state: Mapped[str | None] = mapped_column(String(8))


class PoliticianTrade(Base):
    __tablename__ = "politician_trade"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(300), unique=True, index=True)
    politician_id: Mapped[int | None] = mapped_column(ForeignKey("politician.id"))
    transaction_date: Mapped[date | None] = mapped_column(Date)
    report_date: Mapped[date | None] = mapped_column(Date)
    filed_date: Mapped[date | None] = mapped_column(Date)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    ticker: Mapped[str | None] = mapped_column(String(32), index=True)
    asset_name: Mapped[str | None] = mapped_column(String(200))
    transaction_type: Mapped[str | None] = mapped_column(String(20))
    owner: Mapped[str | None] = mapped_column(String(40))
    amount_min: Mapped[float | None] = mapped_column(Float)
    amount_max: Mapped[float | None] = mapped_column(Float)
    url: Mapped[str | None] = mapped_column(String(300))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_politician_trade_asset_date", "asset_id", "transaction_date"),)


class PoliticianFlowAgg(Base):
    __tablename__ = "politician_flow_agg"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("asset.id"), index=True)
    window_days: Mapped[int] = mapped_column(Integer)
    as_of_date: Mapped[date] = mapped_column(Date)
    net_amount_est: Mapped[float | None] = mapped_column(Float)
    buys_count: Mapped[int | None] = mapped_column(Integer)
    sells_count: Mapped[int | None] = mapped_column(Integer)
    distinct_politicians: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("asset_id", "window_days", "as_of_date", name="uq_politician_flow_asset_window"),
    )


class PriceBar(Base):
    __tablename__ = "price_bar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source_file: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (
        UniqueConstraint("asset_id", "interval", "ts", name="uq_price_bar_asset_interval_ts"),
        Index("ix_price_bar_asset_interval_ts", "asset_id", "interval", "ts"),
    )


class FundamentalSnapshot(Base):
    __tablename__ = "fundamental_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    data: Mapped[dict | None] = mapped_column(JSON)
    source_file: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_fundamental_ticker_as_of", "ticker", "as_of_date"),)


class NewsItem(Base):
    __tablename__ = "news_item"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("asset.id"))
    ticker: Mapped[str | None] = mapped_column(String(32), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(300))
    source_name: Mapped[str | None] = mapped_column(String(120))
    sentiment: Mapped[str | None] = mapped_column(String(24))
    data: Mapped[dict | None] = mapped_column(JSON)
    source_file: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (Index("ix_news_ticker_published", "ticker", "published_at"),)
