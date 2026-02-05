# backend/api.py
from __future__ import annotations
import asyncio
import ast
import json
import logging
import math
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import requests
from backend.api_prices_tiingo import router as prices_router
from backend.core.auth_refresh import periodic_token_refresh
from backend.job_runner import build_default_manager
from backend.core.check_actual_trading_results import (
    get_orders,
    analyze_order_history,
)
from backend.core.horizon_buy_suggesters import (
    SharedSuggesterData,
    build_shared_suggester_data,
    generate_long_term_suggestions,
    generate_short_term_suggestions,
    generate_swing_suggestions,
)
from backend.core.suggester_cancel import CancelToken, SuggesterCancelled
from backend.core.holdings_analysis import analyze_holdings
from backend.core.scripts import get_mass_stock_list as stock_list_builder
from backend.platform_apis.schwab_api.accounts_trading import AccountsTrading
from backend.backtest_runtime import BacktestManager
from backend.platform_apis.schwab_api.get_init_auth_token import (
    construct_headers_and_payload,
    construct_init_auth_url,
    retrieve_tokens,
)
from backend.platform_apis.schwab_api.retrieve_secrets_and_tokens import (
    clear_auth_refresh,
    get_auth_status,
    store_auth_token_value,
)
from backend.platform_apis.gmail_api.gmail import Gmail
from backend.platform_apis.gmail_api.gmail_auth import HeadlessAuthRequired, finish_headless_auth
from backend.core.passwords import hash_password, verify_password
from backend.party_time_runtime import PARTY_TIME_ALGOS, PartyTimeManager
import numpy as np
import pandas as pd

try:
    from backend.db.session import get_session, is_db_enabled
    from backend.db import repository as db_repo
    from backend.db.models import UserAccount
except Exception:  # pragma: no cover - optional DB support
    db_repo = None
    UserAccount = None

    def is_db_enabled() -> bool:
        return False

STRATEGY_DIR = Path(__file__).resolve().parent / "core" / "strategies"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GMAIL_TOKEN_PATH = PROJECT_ROOT / "schwab_secrets" / "gmail_token.json"
FILES_DIR = (Path(__file__).resolve().parent / ".." / "files").resolve()
DAILY_SUGGESTIONS_DIR = FILES_DIR / "daily_suggestions"
OPTIONS_SUGGESTIONS_DIR = FILES_DIR / "options_suggestions"
BACKTEST_RUNS_DIR = FILES_DIR / "backtests" / "runs"
POLITICIAN_TRADES_DIR = FILES_DIR / "politician_trades"
LEGAL_DIR = FILES_DIR / "legal"
TERMS_PDF_PATH = LEGAL_DIR / "Terms_of_Service.pdf"
PRIVACY_PDF_PATH = LEGAL_DIR / "Privacy_Policy.pdf"
POLITICIAN_TRADES_CACHE_TTL_SEC = 15 * 60
POLITICIAN_TRADES_MAX_DAYS = 365
POLITICIAN_TRADES_MAX_LIMIT = 500
POLITICIAN_TRADES_DEFAULT_SOURCES = "house_clerk,senate_efd"
TRADES_CACHE_DIR = FILES_DIR / "trades_cache"
TRADES_CACHE_MAX_DAYS = 365
CONFIGS_DIR = Path(__file__).resolve().parent / "core" / "configs"
HELPER_FILES_DIR = Path(__file__).resolve().parent / "core" / "helper_files"
HOLDINGS_SAVE_PATH = FILES_DIR / "logs" / "holdings.json"
_CONFIG_EXTS = (".yaml", ".yml")
STOCK_LIST_SPECS = {
    "sp500": {
        "label": "S&P 500 list",
        "path": HELPER_FILES_DIR / "sandp_tickers_list.csv",
        "builder": lambda: stock_list_builder.fetch_sp500(prefer="wiki"),
    },
    "russell2000": {
        "label": "Russell 2000 list",
        "path": HELPER_FILES_DIR / "russell2000_tickers_list.csv",
        "builder": stock_list_builder.fetch_russell2000_iwm,
    },
    "mass_combo": {
        "label": "Mass combo (S&P 500 + NASDAQ-100 + Russell 2000)",
        "path": HELPER_FILES_DIR / "mass_tickers_list.csv",
        "builder": lambda: stock_list_builder.build_mass_universe(
            include_sp500=True,
            include_nasdaq100=True,
            include_russell2000=True,
            include_nasdaq_all=False,
            include_russell3000=False,
        ),
    },
}
logger = logging.getLogger(__name__)
_DAILY_SUGGESTIONS_PATTERN = re.compile(r"^suggestions_(\d{8})\.csv$")
_SUGGESTIONS_PATTERN = re.compile(r"^(?P<key>[a-z0-9_\-]+)_suggestions_(\d{8})\.csv$")
DEFAULT_SUGGESTER = "short_term"
SUGGESTER_ALIASES = {
    "short": DEFAULT_SUGGESTER,
    "short_term": DEFAULT_SUGGESTER,
    "shortterm": DEFAULT_SUGGESTER,
    "1-2_day": DEFAULT_SUGGESTER,
    "1_2_day": DEFAULT_SUGGESTER,
    "1d": DEFAULT_SUGGESTER,
    "daily": DEFAULT_SUGGESTER,
    "daily_buy_suggester": DEFAULT_SUGGESTER,
    "momentum": DEFAULT_SUGGESTER,
    "swing": "swing_term",
    "swing_term": "swing_term",
    "medium": "swing_term",
    "medium_term": "swing_term",
    "3-4_month": "swing_term",
    "3_4_month": "swing_term",
    "3-4_months": "swing_term",
    "long": "long_term",
    "long_term": "long_term",
    "longterm": "long_term",
    "long_hold": "long_term",
    "1+_year": "long_term",
    "1_year": "long_term",
    "1y": "long_term",
}
SUGGESTER_RUNNERS = {
    "short_term": generate_short_term_suggestions,
    "swing_term": generate_swing_suggestions,
    "long_term": generate_long_term_suggestions,
}
_BUY_SUGGESTER_LOCK = threading.Lock()
_BUY_SUGGESTER_TOKENS: dict[str, set[CancelToken]] = {}
_OPTIONS_SUGGESTER_LOCK = threading.Lock()
_OPTIONS_SUGGESTER_TOKENS: dict[str, set[CancelToken]] = {}
_TRIAL_DAYS = 30


def _register_buy_suggester_token(scope: str) -> CancelToken:
    token = CancelToken()
    with _BUY_SUGGESTER_LOCK:
        _BUY_SUGGESTER_TOKENS.setdefault(scope, set()).add(token)
    return token


def _release_buy_suggester_token(scope: str, token: CancelToken) -> None:
    with _BUY_SUGGESTER_LOCK:
        tokens = _BUY_SUGGESTER_TOKENS.get(scope)
        if not tokens:
            return
        tokens.discard(token)
        if not tokens:
            _BUY_SUGGESTER_TOKENS.pop(scope, None)


def _cancel_buy_suggester_tokens(scope: Optional[str] = None) -> int:
    tokens: list[CancelToken] = []
    with _BUY_SUGGESTER_LOCK:
        if scope:
            tokens = list(_BUY_SUGGESTER_TOKENS.get(scope, set()))
        else:
            for items in _BUY_SUGGESTER_TOKENS.values():
                tokens.extend(list(items))
    for token in tokens:
        token.cancel("Suggester run stopped by user.")
    return len(tokens)


def _register_options_suggester_token(scope: str) -> CancelToken:
    token = CancelToken()
    with _OPTIONS_SUGGESTER_LOCK:
        _OPTIONS_SUGGESTER_TOKENS.setdefault(scope, set()).add(token)
    return token


def _release_options_suggester_token(scope: str, token: CancelToken) -> None:
    with _OPTIONS_SUGGESTER_LOCK:
        tokens = _OPTIONS_SUGGESTER_TOKENS.get(scope)
        if not tokens:
            return
        tokens.discard(token)
        if not tokens:
            _OPTIONS_SUGGESTER_TOKENS.pop(scope, None)


def _cancel_options_suggester_tokens(scope: Optional[str] = None) -> int:
    tokens: list[CancelToken] = []
    with _OPTIONS_SUGGESTER_LOCK:
        if scope:
            tokens = list(_OPTIONS_SUGGESTER_TOKENS.get(scope, set()))
        else:
            for items in _OPTIONS_SUGGESTER_TOKENS.values():
                tokens.extend(list(items))
    for token in tokens:
        token.cancel("Suggester run stopped by user.")
    return len(tokens)


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _format_name_from_email(email: str) -> str:
    handle = (email or "").split("@")[0] or "member"
    parts = [part for part in re.split(r"[._-]+", handle) if part]
    if not parts:
        return "Member"
    return " ".join(part.capitalize() for part in parts)


def _ensure_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _trial_info(created_at: datetime | None) -> tuple[datetime | None, int | None, bool]:
    if created_at is None:
        return None, None, False
    created_at = _ensure_utc(created_at)
    if created_at is None:
        return None, None, False
    trial_end = created_at + timedelta(days=_TRIAL_DAYS)
    now = datetime.now(timezone.utc)
    remaining = trial_end - now
    seconds_left = remaining.total_seconds()
    days_left = max(0, math.ceil(seconds_left / 86400))
    return trial_end, days_left, seconds_left <= 0


def _user_payload(user: UserAccount) -> dict[str, Any]:
    created_at = _ensure_utc(user.created_at)
    subscription_tier = user.subscription_tier or "trial"
    subscription_status = user.subscription_status or "active"
    payload: dict[str, Any] = {
        "email": user.email,
        "fullName": _format_name_from_email(user.email),
        "subscriptionTier": subscription_tier,
        "subscriptionStatus": subscription_status,
        "createdAt": created_at.isoformat() if created_at else None,
    }
    if subscription_tier in {"trial", "free"}:
        trial_end, days_left, _ = _trial_info(created_at)
        payload["trialEndsAt"] = trial_end.isoformat() if trial_end else None
        payload["trialDaysLeft"] = days_left
    return payload


def _safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return _safe_json_value(obj)


def _normalize_suggester_key(raw: Optional[str]) -> str:
    cleaned = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not cleaned:
        return DEFAULT_SUGGESTER
    if cleaned in SUGGESTER_ALIASES:
        return SUGGESTER_ALIASES[cleaned]
    if cleaned in SUGGESTER_RUNNERS:
        return cleaned
    raise ValueError(f"Unknown suggester key: {raw}")


def _suggester_prefix(key: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", (key or DEFAULT_SUGGESTER).lower()).strip("_")
    return safe or DEFAULT_SUGGESTER


def _default_period_for_horizon(horizon_key: str) -> str:
    normalized = _normalize_suggester_key(horizon_key)
    if normalized == "long_term":
        return "5y"
    if normalized == "swing_term":
        return "3y"
    return "1y"


def _period_to_years(period: Optional[str], fallback: int) -> int:
    """Parse a period like '3y' into an integer year count."""
    try:
        if period is None:
            return fallback
        match = re.match(r"^\s*(\d+)", str(period))
        if match:
            return max(int(match.group(1)), 1)
    except Exception:
        pass
    return fallback


def _normalize_stock_list_id(raw: Optional[str]) -> str:
    cleaned = str(raw or "").strip().lower()
    if not cleaned:
        raise ValueError("Stock list id is required.")
    if cleaned not in STOCK_LIST_SPECS:
        raise KeyError(f"Unknown stock list id: {raw}")
    return cleaned


def _read_stock_list_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    tokens: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        tokens.extend(re.split(r"[,\s]+", cleaned))
    return stock_list_builder.normalize_and_filter(tokens)


def _stock_list_payload(
    list_id: str,
    *,
    include_tickers: bool = False,
    tickers: Optional[list[str]] = None,
) -> dict[str, Any]:
    normalized = _normalize_stock_list_id(list_id)
    spec = STOCK_LIST_SPECS[normalized]
    path = spec["path"]
    if tickers is None:
        tickers = _read_stock_list_file(path)
    else:
        tickers = stock_list_builder.normalize_and_filter(tickers)
    updated_at = None
    try:
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except Exception:
        updated_at = None
    payload: dict[str, Any] = {
        "id": normalized,
        "label": spec.get("label", normalized),
        "count": len(tickers),
        "updated_at": updated_at,
    }
    if include_tickers:
        payload["tickers"] = tickers
    return payload


def _refresh_stock_list(list_id: str, *, include_tickers: bool = False) -> dict[str, Any]:
    normalized = _normalize_stock_list_id(list_id)
    spec = STOCK_LIST_SPECS[normalized]
    builder = spec.get("builder")
    if builder is None:
        raise ValueError(f"No builder configured for stock list '{list_id}'.")
    tickers = builder() or []
    tickers = stock_list_builder.normalize_and_filter(tickers)
    stock_list_builder.write_ticker_file(spec["path"], tickers, header=spec.get("label", normalized))
    return _stock_list_payload(normalized, include_tickers=include_tickers, tickers=tickers)


def _load_stock_list(list_id: str) -> list[str]:
    normalized = _normalize_stock_list_id(list_id)
    tickers = _read_stock_list_file(STOCK_LIST_SPECS[normalized]["path"])
    if not tickers:
        raise ValueError(f"Stock list '{list_id}' is empty.")
    return tickers


def _resolve_stock_list_and_tickers(body: dict) -> tuple[Optional[list[str]], Optional[str], Optional[str]]:
    tickers = body.get("tickers")
    stock_list_raw = body.get("stock_list") or body.get("stock_list_id") or body.get("list_id")
    stock_list_id: str | None = None
    if stock_list_raw:
        stock_list_id = _normalize_stock_list_id(stock_list_raw)

    if tickers is None and stock_list_id:
        tickers = _load_stock_list(stock_list_id)

    if isinstance(tickers, str):
        tickers = [t.strip() for t in re.split(r"[,\s]+", tickers) if t.strip()]

    stock_list_label = STOCK_LIST_SPECS.get(stock_list_id, {}).get("label") if stock_list_id else None
    return tickers, stock_list_id, stock_list_label


def _extract_strategy_description(path: Path) -> str | None:
    """
    Try to pull a short, human-friendly description from a strategy file.
    For Python files we use the module docstring or first comment; for markdown
    we grab the first non-empty line/heading.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if path.suffix == ".md":
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned.lstrip("#").strip()
        return None

    if path.suffix == ".py":
        try:
            module = ast.parse(text)
            doc = ast.get_docstring(module)
            if doc:
                return doc.splitlines()[0]
        except Exception:
            # fallback to comments below
            pass

        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.startswith("#"):
                return cleaned.lstrip("#").strip()
            break

    return None


def _today_str() -> str:
    """Return local date in YYYYMMDD for naming cached suggestion files."""
    return datetime.now().astimezone().strftime("%Y%m%d")


def _suggestions_path(key: str, date_str: Optional[str] = None, *, allow_legacy: bool = False) -> Path:
    """Path to the on-disk cache for a given day and suggester (defaults to today)."""
    ds = date_str or _today_str()
    primary = DAILY_SUGGESTIONS_DIR / f"{_suggester_prefix(key)}_suggestions_{ds}.csv"
    if allow_legacy and _normalize_suggester_key(key) == DEFAULT_SUGGESTER:
        legacy = DAILY_SUGGESTIONS_DIR / f"suggestions_{ds}.csv"
        if legacy.exists() and not primary.exists():
            return legacy
    return primary


def _normalize_suggestions_date(date_str: str) -> str:
    """Accept YYYYMMDD or YYYY-MM-DD and normalize to YYYYMMDD."""
    cleaned = str(date_str or "").strip()
    if re.fullmatch(r"\d{8}", cleaned):
        return cleaned
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned.replace("-", "")
    raise ValueError("Expected date in YYYYMMDD or YYYY-MM-DD format.")


def _normalize_daily_suggestions_date(date_str: str) -> str:
    """Backward-compatible alias used by the legacy daily endpoints."""
    return _normalize_suggestions_date(date_str)


def _count_csv_rows(path: Path) -> int:
    """Count non-header rows in a CSV file without loading it into pandas."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            rows = 0
            saw_header = False
            for line in handle:
                if not line.strip():
                    continue
                if not saw_header:
                    saw_header = True
                    continue
                rows += 1
            return rows
    except Exception as exc:
        logger.warning("Failed to count rows in %s: %s", path, exc)
        return 0


def _list_suggestions_history(key: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """List available cached suggestion files for a given script key."""
    normalized_key = _normalize_suggester_key(key)
    if is_db_enabled() and db_repo:
        try:
            with get_session() as session:
                entries = db_repo.list_suggestion_batches(session, normalized_key, limit=limit)
            if entries:
                return entries
        except Exception as exc:
            logger.warning("DB suggestion history lookup failed: %s", exc)
    if not DAILY_SUGGESTIONS_DIR.exists():
        return []
    pattern = re.compile(rf"^{re.escape(_suggester_prefix(normalized_key))}_suggestions_(\d{{8}})\.csv$")

    entries: list[dict[str, Any]] = []
    for path in DAILY_SUGGESTIONS_DIR.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        legacy_match = _DAILY_SUGGESTIONS_PATTERN.match(path.name) if normalized_key == DEFAULT_SUGGESTER else None
        use_match = match or legacy_match
        if not use_match:
            continue
        date_str = use_match.group(1)
        stat = path.stat()
        entries.append(
            {
                "date": date_str,
                "label": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                "filename": path.name,
                "last_run": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "count": _count_csv_rows(path),
            }
        )

    entries.sort(key=lambda item: item["date"], reverse=True)
    if limit is not None:
        entries = entries[: max(0, limit)]
    return entries


def _list_daily_suggestions_history(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Legacy helper that targets the short-term (daily) suggester."""
    return _list_suggestions_history(DEFAULT_SUGGESTER, limit=limit)


def _load_cached_suggestions(key: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load today's cached suggestions if present, returning the dataframe and last run timestamp."""
    path = _suggestions_path(key, allow_legacy=True)
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
            if unnamed:
                df = df.rename(columns={unnamed[0]: "ticker"})
        last_run = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        return df, last_run
    except Exception as exc:
        logger.warning("Failed to load cached suggestions from %s: %s", path, exc)
        return None, None


def _load_cached_daily_suggestions() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Legacy helper targeting the short-term suggester."""
    return _load_cached_suggestions(DEFAULT_SUGGESTER)


def _persist_suggestions(df: pd.DataFrame, key: str) -> Optional[str]:
    """Write suggestions to today's cache file and return the write timestamp."""
    try:
        DAILY_SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _suggestions_path(key)
        df_to_save = df.copy()
        if "ticker" not in df_to_save.columns and df_to_save.index.name == "ticker":
            df_to_save = df_to_save.reset_index()
        df_to_save.to_csv(path, index=False)
        if is_db_enabled() and db_repo:
            try:
                with get_session() as session:
                    db_repo.save_suggestions(
                        session,
                        df_to_save,
                        _normalize_suggester_key(key),
                        generated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                        source_file=str(path),
                    )
            except Exception as exc:
                logger.warning("Failed to persist suggestions to DB: %s", exc)
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except Exception as exc:
        logger.warning("Failed to persist %s suggestions: %s", key, exc)
        return None


def _persist_daily_suggestions(df: pd.DataFrame) -> Optional[str]:
    """Legacy helper targeting the short-term suggester."""
    return _persist_suggestions(df, DEFAULT_SUGGESTER)


def _options_suggestions_path(key: str, date_str: Optional[str] = None) -> Path:
    ds = date_str or _normalize_suggestions_date(datetime.now().strftime("%Y%m%d"))
    return OPTIONS_SUGGESTIONS_DIR / f"{_suggester_prefix(key)}_options_{ds}.json"


def _persist_options_suggestions(
    rows: list[dict[str, Any]],
    key: str,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Persist options suggestions JSON for history."""
    try:
        OPTIONS_SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _options_suggestions_path(key)
        payload = {"options": rows, "meta": meta or {}}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_safe_json_value, ensure_ascii=False, indent=2)
        if is_db_enabled() and db_repo:
            try:
                with get_session() as session:
                    db_repo.save_options(
                        session,
                        rows,
                        _normalize_suggester_key(key),
                        meta=meta,
                        generated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                        source_file=str(path),
                    )
            except Exception as exc:
                logger.warning("Failed to persist options suggestions to DB: %s", exc)
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except Exception as exc:
        logger.warning("Failed to persist %s options suggestions: %s", key, exc)
        return None


def _list_options_suggestions_history(key: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """List saved options suggestion runs for a horizon."""
    key = _normalize_suggester_key(key)
    if is_db_enabled() and db_repo:
        try:
            with get_session() as session:
                entries = db_repo.list_options_batches(session, key, limit=limit)
            if entries:
                return entries
        except Exception as exc:
            logger.warning("DB options history lookup failed: %s", exc)
    if not OPTIONS_SUGGESTIONS_DIR.exists():
        return []
    pattern = re.compile(rf"^{re.escape(_suggester_prefix(key))}_options_(\d{{8}})\.json$")
    entries: list[dict[str, Any]] = []
    for path in sorted(OPTIONS_SUGGESTIONS_DIR.iterdir(), reverse=True):
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        ds = m.group(1)
        count = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                items = data.get("options")
                count = len(items) if isinstance(items, list) else None
            elif isinstance(data, list):
                count = len(data)
        except Exception:
            count = None
        entries.append(
            {
                "date": ds,
                "label": f"{ds[:4]}-{ds[4:6]}-{ds[6:]}",
                "count": count,
                "file": path.name,
            }
        )
        if limit and len(entries) >= limit:
            break
    return entries


def _load_options_suggestions_history_entry(key: str, date_str: str) -> dict[str, Any]:
    """Load a saved options suggestions run for a given date."""
    normalized = _normalize_suggestions_date(date_str)
    if is_db_enabled() and db_repo:
        try:
            with get_session() as session:
                payload = db_repo.load_options_by_date(
                    session, _normalize_suggester_key(key), datetime.strptime(normalized, "%Y%m%d").date()
                )
            if payload:
                return payload
        except Exception as exc:
            logger.warning("DB options history entry lookup failed: %s", exc)
    path = _options_suggestions_path(key, normalized)
    if not path.exists():
        raise FileNotFoundError(f"No options suggestions cached for {normalized}.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load options suggestions: {exc}")

    meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        meta = payload.get("meta") or {}
        if isinstance(payload.get("options"), list):
            records = payload["options"]
    elif isinstance(payload, list):
        records = payload
    last_run_ts = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    date_label = f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:]}"
    return {
        "date": normalized,
        "label": date_label,
        "options": records,
        "meta": meta,
        "count": len(records),
        "last_run": last_run_ts,
        "cached": True,
    }


def _load_suggestions_history_entry(key: str, date_str: str) -> dict[str, Any]:
    """Load a specific suggester run by date."""
    normalized = _normalize_suggestions_date(date_str)
    if is_db_enabled() and db_repo:
        try:
            with get_session() as session:
                payload = db_repo.load_suggestions_by_date(
                    session, _normalize_suggester_key(key), datetime.strptime(normalized, "%Y%m%d").date()
                )
            if payload:
                payload["date"] = normalized
                payload["label"] = f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:]}"
                return payload
        except Exception as exc:
            logger.warning("DB suggestion history entry lookup failed: %s", exc)
    path = _suggestions_path(key, normalized, allow_legacy=True)
    if not path.exists():
        raise FileNotFoundError(f"No suggestions cached for {normalized}.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load suggestions: {exc}")

    df = _normalize_suggestions_df(df)
    columns = list(df.columns)
    records = json.loads(df.to_json(orient="records")) if not df.empty else []
    last_run_ts = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    date_label = f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:]}"

    return {
        "date": normalized,
        "label": date_label,
        "suggestions": records,
        "columns": columns,
        "count": len(records),
        "last_run": last_run_ts,
        "cached": True,
    }


def _normalize_suggestions_df(raw: Any) -> pd.DataFrame:
    """Best-effort conversion of suggester output into a DataFrame with a ticker column."""
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        df = raw
    else:
        try:
            df = pd.DataFrame(raw)
        except Exception as exc:
            raise ValueError(f"Failed to normalize suggestions: {exc}")

    if "ticker" not in df.columns:
        if df.index.name == "ticker":
            df = df.reset_index()
        elif any(str(c).lower() == "symbol" for c in df.columns):
            df = df.rename(columns={next(c for c in df.columns if str(c).lower() == "symbol"): "ticker"})
        else:
            unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
            if unnamed:
                df = df.rename(columns={unnamed[0]: "ticker"})
            else:
                df.insert(0, "ticker", "")
    return df


def _load_cached_suggestions_payload(
    key: str,
    *,
    stock_list_id: Optional[str],
    stock_list_label: Optional[str],
) -> Optional[dict[str, Any]]:
    """Load today's cached suggestions for a horizon, returning API-shaped payload if present."""
    normalized_key = _normalize_suggester_key(key)
    if is_db_enabled() and db_repo:
        try:
            with get_session() as session:
                payload = db_repo.load_latest_suggestions(session, normalized_key)
            if payload:
                payload["stock_list"] = stock_list_id
                payload["stock_list_label"] = stock_list_label
                return payload
        except Exception as exc:
            logger.warning("DB cached suggestions lookup failed: %s", exc)
    cached_df, last_run_ts = _load_cached_suggestions(key)
    if cached_df is None:
        return None
    df = _normalize_suggestions_df(cached_df)
    if last_run_ts is None and not df.empty:
        last_run_ts = datetime.now().astimezone().isoformat()
    records = json.loads(df.to_json(orient="records")) if not df.empty else []
    return {
        "suggestions": records,
        "columns": list(df.columns),
        "count": len(records),
        "last_run": last_run_ts,
        "cached": True,
        "horizon": _normalize_suggester_key(key),
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
    }


async def _execute_suggester(
    key: str,
    *,
    tickers: Optional[list[str]] = None,
    stock_list_id: Optional[str] = None,
    period: str = "1y",
    cfg_overrides: Optional[dict[str, Any]] = None,
    prefer_cache: bool = True,
    force_rerun: bool = False,
    cache_only: bool = False,
    runner_kwargs: Optional[dict[str, Any]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> dict[str, Any]:
    """
    Shared runner that applies caching and normalizes the response shape across suggesters.
    """
    normalized_key = _normalize_suggester_key(key)
    runner = SUGGESTER_RUNNERS.get(normalized_key)
    if runner is None:
        raise ValueError(f"Unknown suggester: {key}")

    loop = asyncio.get_running_loop()
    cached_df: Optional[pd.DataFrame] = None
    last_run_ts: Optional[str] = None

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    if prefer_cache and not force_rerun:
        cached_df, last_run_ts = _load_cached_suggestions(normalized_key)

    if cached_df is not None:
        df = _normalize_suggestions_df(cached_df)
    elif cache_only:
        df = pd.DataFrame()
    else:
        try:
            run_kwargs = {"tickers": tickers, "period": period, "cfg_overrides": cfg_overrides}
            if runner_kwargs:
                run_kwargs.update(runner_kwargs)
            if cancel_token is not None:
                run_kwargs["cancel_token"] = cancel_token
            suggestions_df = await loop.run_in_executor(None, lambda: runner(**run_kwargs))
        except SuggesterCancelled:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        try:
            df = _normalize_suggestions_df(suggestions_df)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        last_run_ts = _persist_suggestions(df, normalized_key)

    if last_run_ts is None and not df.empty:
        last_run_ts = datetime.now().astimezone().isoformat()

    columns = list(df.columns)
    records = json.loads(df.to_json(orient="records")) if not df.empty else []
    stock_list_label = STOCK_LIST_SPECS.get(stock_list_id, {}).get("label") if stock_list_id else None

    return {
        "suggestions": records,
        "columns": columns,
        "count": len(records),
        "last_run": last_run_ts,
        "cached": cached_df is not None or cache_only,
        "horizon": normalized_key,
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
    }


_OPTION_TARGET_DTE = {"short_term": 14, "swing_term": 45, "long_term": 120}


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _option_mid(contract: dict[str, Any]) -> float | None:
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    if bid is not None and ask is not None and bid >= 0 and ask >= 0:
        return (bid + ask) / 2
    for key in ("mark", "last", "close", "price"):
        val = _safe_float(contract.get(key))
        if val is not None and val > 0:
            return val
    return None


def _option_spread(contract: dict[str, Any]) -> float | None:
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    if bid is None or ask is None or bid < 0 or ask < 0:
        return None
    return max(0.0, ask - bid)


def _normalize_expiration(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if ts is not None and not pd.isna(ts):
            return ts.isoformat()
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return None


def _flatten_option_map(exp_map: Any, put_call: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    if not isinstance(exp_map, dict):
        return flattened

    for exp_label, strikes in exp_map.items():
        exp_date: Optional[str] = None
        exp_days: Optional[int] = None
        if isinstance(exp_label, str):
            parts = exp_label.split(":")
            if parts:
                exp_date = parts[0]
            if len(parts) > 1:
                try:
                    exp_days = int(parts[1])
                except Exception:
                    exp_days = None
        try:
            strike_items = strikes.items()
        except Exception:
            continue
        for strike_key, contracts in strike_items:
            strike_price = _safe_float(strike_key)
            if not isinstance(contracts, (list, tuple)):
                continue
            for contract in contracts:
                if not isinstance(contract, dict):
                    continue
                enriched = dict(contract)
                if strike_price is not None and enriched.get("strikePrice") is None:
                    enriched["strikePrice"] = strike_price
                if exp_date and enriched.get("expirationDate") is None:
                    enriched["expirationDate"] = exp_date
                if exp_days is not None and enriched.get("daysToExpiration") is None:
                    enriched["daysToExpiration"] = exp_days
                enriched["putCall"] = enriched.get("putCall") or put_call
                flattened.append(enriched)
    return flattened


def _target_dte_for_horizon(horizon: str) -> int:
    return _OPTION_TARGET_DTE.get(horizon, 45)


def _summarize_option_contract(
    contract: dict[str, Any],
    *,
    put_call: str,
    underlying_price: float | None,
    target_dte: int,
) -> dict[str, Any]:
    strike = _safe_float(contract.get("strikePrice"))
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    mark = _safe_float(contract.get("mark"))
    last = _safe_float(contract.get("last"))
    mid = _option_mid(contract)
    spread = _option_spread(contract)
    spread_pct = (spread / mid * 100) if spread is not None and mid else None

    expiration_raw = contract.get("expirationDate") or contract.get("expiration")
    expiration = _normalize_expiration(expiration_raw)
    days_to_expiration = _safe_int(contract.get("daysToExpiration"))
    if days_to_expiration is None and expiration:
        try:
            exp_dt = pd.to_datetime(expiration, errors="coerce")
            if pd.notna(exp_dt):
                days_to_expiration = max(
                    (exp_dt.date() - datetime.now(timezone.utc).date()).days,
                    0,
                )
        except Exception:
            days_to_expiration = None

    break_even = None
    if mid is not None and strike is not None:
        break_even = strike + mid if put_call.upper() == "CALL" else strike - mid

    moneyness_pct = None
    if strike is not None and underlying_price not in (None, 0):
        moneyness_pct = ((strike - underlying_price) / underlying_price) * 100

    return {
        "put_call": put_call,
        "symbol": contract.get("symbol") or contract.get("description"),
        "strike": strike,
        "expiration": expiration or expiration_raw,
        "days_to_expiration": days_to_expiration,
        "target_dte": target_dte,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "mark": mark,
        "last": last,
        "open_interest": _safe_int(contract.get("openInterest")),
        "volume": _safe_int(contract.get("totalVolume") or contract.get("volume")),
        "delta": _safe_float(contract.get("delta")),
        "gamma": _safe_float(contract.get("gamma")),
        "theta": _safe_float(contract.get("theta")),
        "vega": _safe_float(contract.get("vega")),
        "rho": _safe_float(contract.get("rho")),
        "implied_volatility": _safe_float(contract.get("volatility")),
        "in_the_money": contract.get("inTheMoney"),
        "spread": spread,
        "spread_pct": spread_pct,
        "break_even": break_even,
        "moneyness_pct": moneyness_pct,
        "description": contract.get("description"),
    }


def _select_best_option_from_chain(
    chain_data: dict[str, Any],
    put_call: str,
    *,
    underlying_price: float | None,
    horizon: str,
) -> tuple[Optional[dict[str, Any]], list[str], int]:
    exp_map_key = "callExpDateMap" if put_call.upper() == "CALL" else "putExpDateMap"
    exp_map = chain_data.get(exp_map_key)
    flattened = _flatten_option_map(exp_map, put_call)
    warnings: list[str] = []
    if not flattened:
        warnings.append(f"No {put_call.lower()} contracts returned.")
        return None, warnings, 0

    target_dte = _target_dte_for_horizon(horizon)

    def score(contract: dict[str, Any]) -> tuple[float, float, float, float, float]:
        strike_val = _safe_float(contract.get("strikePrice"))
        if strike_val is None:
            return (float("inf"), float("inf"), float("inf"), 0, 0)
        dte_raw = contract.get("daysToExpiration")
        dte_val = _safe_int(dte_raw)
        if dte_val is None:
            exp_iso = _normalize_expiration(contract.get("expirationDate"))
            if exp_iso:
                try:
                    exp_dt = pd.to_datetime(exp_iso, errors="coerce")
                    if pd.notna(exp_dt):
                        dte_val = max(
                            (exp_dt.date() - datetime.now(timezone.utc).date()).days,
                            0,
                        )
                except Exception:
                    dte_val = None
        spread_val = _option_spread(contract)
        oi = _safe_int(contract.get("openInterest")) or 0
        vol = _safe_int(contract.get("totalVolume") or contract.get("volume")) or 0
        moneyness = abs(strike_val - underlying_price) if underlying_price is not None else abs(strike_val)
        dte_penalty = abs(dte_val - target_dte) if dte_val is not None else target_dte
        spread_penalty = spread_val if spread_val is not None else float("inf")
        return (moneyness, dte_penalty, spread_penalty, -oi, -vol)

    flattened.sort(key=score)
    best_contract = flattened[0]
    summary = _summarize_option_contract(
        best_contract,
        put_call=put_call.upper(),
        underlying_price=underlying_price,
        target_dte=target_dte,
    )
    return summary, warnings, len(flattened)


def _build_option_suggestions(
    suggestions: list[dict[str, Any]],
    *,
    horizon: str,
    limit: int = 10,
    strike_count: int | None = 8,
    include_puts: bool = True,
    strike_range: str | None = "NTM",
    cancel_token: Optional[CancelToken] = None,
) -> list[dict[str, Any]]:
    if limit is None or not isinstance(limit, (int, float)) or limit <= 0:
        limit = 10
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    results: list[dict[str, Any]] = []
    for row in suggestions[: int(limit)]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        underlying_price = (
            _safe_float(row.get("close"))
            or _safe_float(row.get("price"))
            or _safe_float(row.get("entry"))
        )
        chain_error = "Options chain unavailable (Schwab market data disabled)."
        warnings = [chain_error]

        results.append(
            {
                "ticker": ticker,
                "action": row.get("action") or row.get("signal"),
                "horizon": horizon,
                "thesis": row.get("thesis") or row.get("reason"),
                "equity_entry": _safe_float(row.get("entry") or row.get("close") or row.get("price")),
                "underlying_price": underlying_price,
                "best_call": None,
                "best_put": None if include_puts else None,
                "call_candidates": 0,
                "put_candidates": 0 if include_puts else 0,
                "warnings": warnings,
                "chain_error": chain_error,
            }
        )

    return results
def _load_strategy_config(path: Path) -> dict[str, Any]:
    if path.suffix not in _CONFIG_EXTS:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    try:
        import yaml  # pip install pyyaml
    except Exception as exc:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`") from exc
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path.name} must be a YAML mapping.")
    return data


def _resolve_strategy_config_path(config_id: str) -> Path:
    if not config_id:
        raise FileNotFoundError("Config id missing.")
    clean = Path(config_id).name
    stem = Path(clean).stem
    for ext in _CONFIG_EXTS:
        candidate = CONFIGS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config found for '{config_id}'.")


def _list_strategy_configs() -> list[dict[str, Any]]:
    if not CONFIGS_DIR.exists():
        return []
    configs: list[dict[str, Any]] = []
    for path in sorted(CONFIGS_DIR.iterdir()):
        if not path.is_file() or path.suffix not in _CONFIG_EXTS:
            continue
        data = _load_strategy_config(path)
        label = str(data.get("name") or path.stem)
        configs.append(
            {
                "id": path.stem,
                "label": label,
                "filename": path.name,
            }
        )
    return configs


def _serve_legal_pdf(path: Path, filename: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Legal document not found.")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


app = FastAPI(title="Hart Quantitative Research API")

# Dev CORS – tighten origins for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",  # support dev servers on any local port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Models --------------------

class AuthRequest(BaseModel):
    email: str = Field(..., description="Account email")
    password: str = Field(..., description="Account password")

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Account email")
    password: str = Field(..., description="Account password")
    accepted_terms: bool = Field(..., description="User accepted Terms of Service")
    acknowledged_disclaimer: bool = Field(..., description="User acknowledged Financial Disclaimer")


class RunRequest(BaseModel):
    job: str = Field(..., description="Registered job name")
    params: dict[str, Any] = Field(default_factory=dict)

class RunResponse(BaseModel):
    status: str
    job_id: str
    job: str

class JobStatus(BaseModel):
    id: str
    job: str
    status: str
    created_ts: float
    started_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    params: dict = Field(default_factory=dict)


class PartyTimeRunRequest(BaseModel):
    algo: str = Field(..., description="party_time algo id (e.g. 'party_time_momentum')")
    config_id: Optional[str] = Field(
        default=None,
        description="Optional config id from core/configs (e.g. 'baseline').",
    )
    config_overrides: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional overrides merged onto the base config.",
    )


class BacktestRunRequest(BaseModel):
    config_id: str = Field(..., description="Config id from core/configs (e.g. 'baseline').")
    config_overrides: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional overrides merged into the config before running.",
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Optional start date (YYYY-MM-DD) to bound the backtest window.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Optional end date (YYYY-MM-DD) to bound the backtest window.",
    )
    slippage_override: Optional[float] = Field(
        default=None,
        description="Override slippage per side passed to the tester.",
    )
    slippage_sweep: Optional[list[float]] = Field(
        default=None,
        description="Optional list of slippage values to sweep for this run.",
    )


class SchwabInitAuthRequest(BaseModel):
    returned_url: Optional[str] = Field(
        default=None,
        description="Full redirect URL received after completing Schwab login",
    )


class GmailResetRequest(BaseModel):
    returned_url: Optional[str] = Field(
        default=None,
        description="Full redirect URL from Gmail consent (contains ?code=...)",
    )


class HoldingInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol (e.g., AAPL)")
    shares: Optional[float] = Field(
        default=None,
        description="Number of shares currently owned (optional).",
    )
    cost_basis: Optional[float] = Field(
        default=None,
        description="Average cost per share in USD (optional).",
    )


class HoldingsAnalyzeRequest(BaseModel):
    holdings: list[HoldingInput] = Field(
        ...,
        description="List of holdings to analyze.",
        min_items=1,
    )
    price_period: str = Field(
        default="1y",
        description="History window to evaluate (period string, e.g., '6mo', '1y').",
    )


class HoldingsSaveRequest(BaseModel):
    holdings: list[HoldingInput] = Field(
        default_factory=list,
        description="List of holdings to persist to files/logs/holdings.json",
        min_items=0,
    )

# -------------------- Routes --------------------

@app.get("/")
def root():
    # Send folks to the Swagger UI instead of a 404
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    return JSONResponse({"status": "ok"})


@app.get("/api/legal/terms")
def legal_terms():
    return _serve_legal_pdf(TERMS_PDF_PATH, "Terms_of_Service.pdf")


@app.get("/api/legal/privacy")
def legal_privacy():
    return _serve_legal_pdf(PRIVACY_PDF_PATH, "Privacy_Policy.pdf")


@app.post("/api/auth/register")
def register_account(req: RegisterRequest):
    if not is_db_enabled() or UserAccount is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    email = _normalize_email(req.email)
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    if not req.password:
        raise HTTPException(status_code=400, detail="Password is required.")
    if not req.accepted_terms:
        raise HTTPException(status_code=400, detail="Terms of Service must be accepted.")
    if not req.acknowledged_disclaimer:
        raise HTTPException(status_code=400, detail="Financial Disclaimer must be acknowledged.")
    with get_session() as session:
        existing = session.execute(select(UserAccount).where(UserAccount.email == email)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="Account already exists.")
        try:
            password_hash = hash_password(req.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        user = UserAccount(
            email=email,
            password_hash=password_hash,
            subscription_tier="trial",
            subscription_status="trialing",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    return _user_payload(user)


@app.post("/api/auth/login")
def login_account(req: AuthRequest):
    if not is_db_enabled() or UserAccount is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    email = _normalize_email(req.email)
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    if not req.password:
        raise HTTPException(status_code=400, detail="Password is required.")
    with get_session() as session:
        user = session.execute(select(UserAccount).where(UserAccount.email == email)).scalar_one_or_none()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        if user.subscription_tier == "free":
            user.subscription_tier = "trial"
        if user.subscription_tier == "trial":
            trial_end, _, expired = _trial_info(user.created_at)
            if expired:
                user.subscription_status = "expired"
                session.commit()
                raise HTTPException(status_code=403, detail="Trial expired. Please upgrade your plan.")
            if user.subscription_status not in {"trialing", "active"}:
                user.subscription_status = "trialing"
            if trial_end is not None:
                session.commit()
        else:
            if user.subscription_status is None:
                user.subscription_status = "active"
                session.commit()
    return _user_payload(user)

@app.on_event("startup")
async def _startup():
    # background Schwab token refresh
    if not getattr(app.state, "token_refresh_task", None):
        app.state.token_refresh_task = asyncio.create_task(periodic_token_refresh())
    # in-memory job manager
    if not getattr(app.state, "job_manager", None):
        app.state.job_manager = build_default_manager()
    # party_time runtime
    if not getattr(app.state, "party_time_manager", None):
        app.state.party_time_manager = PartyTimeManager()
    # backtest runtime
    if not getattr(app.state, "backtest_manager", None):
        app.state.backtest_manager = BacktestManager(BACKTEST_RUNS_DIR)

@app.on_event("shutdown")
async def _shutdown():
    task = getattr(app.state, "token_refresh_task", None)
    if task:
        task.cancel()

# ----- Prices API from your existing router -----
app.include_router(prices_router)

# ----- Jobs API -----

@app.post("/api/run", response_model=RunResponse)
async def api_run(req: RunRequest):
    jm = app.state.job_manager
    try:
        rec = await jm.submit(req.job, req.params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RunResponse(status="queued", job_id=rec.id, job=rec.job)

@app.get("/api/jobs", response_model=list[JobStatus])
async def api_list_jobs():
    jm = app.state.job_manager
    jobs = await jm.list()
    # pydantic conversion
    return [JobStatus(**j) for j in jobs]

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def api_get_job(job_id: str):
    jm = app.state.job_manager
    j = await jm.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(**j)

@app.delete("/api/jobs/{job_id}", response_model=dict)
async def api_cancel_job(job_id: str):
    jm = app.state.job_manager
    ok = await jm.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": "cancel_requested", "job_id": job_id}


# ----- Party Time algos -----

@app.get("/api/party-time/configs")
async def api_party_configs():
    try:
        configs = _list_strategy_configs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"configs": configs}


@app.get("/api/party-time/configs/{config_id}")
async def api_party_config_detail(config_id: str):
    try:
        path = _resolve_strategy_config_path(config_id)
        config = _load_strategy_config(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"id": path.stem, "filename": path.name, "config": config}


@app.get("/api/party-time/algos")
async def api_party_algos():
    mgr: PartyTimeManager = app.state.party_time_manager
    running_algo = getattr(mgr, "algo", None)
    algos = [
        {"id": k, "label": v.get("label"), "description": v.get("description")}
        for k, v in PARTY_TIME_ALGOS.items()
    ]
    return {"algos": algos, "active": running_algo, "running": getattr(mgr, "is_running", False)}


@app.post("/api/party-time/run")
async def api_party_run(req: PartyTimeRunRequest):
    mgr: PartyTimeManager = app.state.party_time_manager
    config = None
    if req.config_id:
        try:
            path = _resolve_strategy_config_path(req.config_id)
            config = _load_strategy_config(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    if req.config_overrides:
        config = {**(config or {}), **req.config_overrides}
    try:
        await mgr.start(req.algo, config=config, config_id=req.config_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "started", "algo": req.algo, "config_id": req.config_id}


@app.post("/api/party-time/stop")
async def api_party_stop():
    mgr: PartyTimeManager = app.state.party_time_manager
    await mgr.stop()
    return {"status": "stopped"}


@app.get("/api/party-time/status")
async def api_party_status():
    mgr: PartyTimeManager = app.state.party_time_manager
    snap = await mgr.snapshot()
    return snap


# ----- Backtests (daily momentum) -----

@app.get("/api/backtests/configs")
async def api_backtest_configs():
    try:
        configs = _list_strategy_configs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"configs": configs}


@app.get("/api/backtests/configs/{config_id}")
async def api_backtest_config_detail(config_id: str):
    try:
        path = _resolve_strategy_config_path(config_id)
        config = _load_strategy_config(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"id": path.stem, "filename": path.name, "config": config}


@app.post("/api/backtests/run")
async def api_backtest_run(req: BacktestRunRequest):
    mgr: BacktestManager = app.state.backtest_manager
    try:
        snap = await mgr.start(
            config_id=req.config_id,
            config_overrides=req.config_overrides,
            start_date=req.start_date,
            end_date=req.end_date,
            slippage_override=req.slippage_override,
            slippage_sweep=req.slippage_sweep,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "started", **snap}


@app.get("/api/backtests/status")
async def api_backtest_status():
    mgr: BacktestManager = app.state.backtest_manager
    snap = await mgr.snapshot()
    return _sanitize_for_json(snap)


@app.get("/api/backtests/history")
async def api_backtest_history(limit: Optional[int] = 25):
    mgr: BacktestManager = app.state.backtest_manager
    try:
        limit_int = int(limit) if limit is not None else 25
    except Exception:
        limit_int = 25
    entries = mgr.list_history(limit=max(1, limit_int))
    return {"count": len(entries), "runs": _sanitize_for_json(entries)}


@app.get("/api/backtests/history/{run_id}")
async def api_backtest_history_detail(
    run_id: str,
    include_trades: bool = False,
    trades_limit: int = 200,
    trades_offset: int = 0,
):
    mgr: BacktestManager = app.state.backtest_manager
    try:
        detail = mgr.load_run(
            run_id,
            include_trades=include_trades,
            trades_limit=trades_limit,
            trades_offset=trades_offset,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"run": _sanitize_for_json(detail)}


@app.get("/api/profile/auth-status")
async def api_profile_auth_status():
    """
    Return last-refresh metadata for Schwab and Gmail auth flows.
    """
    return get_auth_status()


@app.post("/api/profile/schwab-init")
async def api_profile_schwab_init(req: SchwabInitAuthRequest):
    """
    Kick off or complete the Schwab OAuth flow used by get_init_auth_token.py.

    Call with no `returned_url` to fetch the login/consent URL, then call again
    with the redirected URL (containing `?code=...`) to exchange and store tokens.
    """
    try:
        app_key, app_secret, redirect_uri, auth_url = construct_init_auth_url()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Failed to build Schwab auth URL: {exc}")

    if not req.returned_url:
        return {
            "status": "auth_url",
            "auth_url": auth_url,
            "message": (
                "Open this URL in your browser, finish Schwab login, then paste the "
                "redirected URL into the form to complete setup."
            ),
        }

    try:
        headers, payload = construct_headers_and_payload(
            req.returned_url, app_key, app_secret, redirect_uri
        )
        tokens = retrieve_tokens(headers=headers, payload=payload)
        store_auth_token_value(tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        status = exc.response.status_code if exc.response is not None else 502
        mapped_status = 400 if status and status < 500 else 502
        raise HTTPException(
            status_code=mapped_status,
            detail=f"Failed to exchange authorization code (HTTP {status}): {detail[:500]}",
        )
    except Exception as exc:  # pragma: no cover - network/IO errors
        raise HTTPException(status_code=502, detail=f"Failed to exchange authorization code: {exc}")

    return {
        "status": "stored",
        "token_keys": list(tokens.keys()),
        "message": "Tokens retrieved and stored.",
    }


@app.post("/api/profile/gmail-reset")
async def api_profile_gmail_reset(req: GmailResetRequest):
    """
    Delete the Gmail token file and start a fresh Google auth flow.
    In headless environments we return the consent URL immediately so
    the UI can guide the user without blocking.
    """
    token_state: str | None = None
    try:
        if GMAIL_TOKEN_PATH.exists():
            GMAIL_TOKEN_PATH.unlink()
            token_state = "deleted"
        else:
            token_state = "not_found"
        clear_auth_refresh("gmail")
    except Exception as exc:  # pragma: no cover - filesystem failure
        token_state = f"delete_failed: {exc}"

    gmail = Gmail()

    if req.returned_url:
        try:
            finish_headless_auth(req.returned_url)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to complete Gmail auth: {exc}")

        return {
            "status": "stored",
            "token_file": token_state,
            "message": "Gmail token retrieved and stored.",
        }

    try:
        ok = gmail.connect()
    except HeadlessAuthRequired as exc:
        return {
            "status": "headless_auth",
            "auth_url": exc.auth_url,
            "token_file": token_state,
            "message": (
                "Open the Gmail consent URL in your browser, approve access, and paste the redirected URL back here."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run gmail auth flow: {exc}")

    if not ok:
        raise HTTPException(status_code=500, detail="Failed to connect to Gmail.")

    return {
        "status": "ok",
        "token_file": token_state,
        "message": "Gmail token ready.",
    }


@app.get("/api/strategies")
def api_list_strategies():
    if not STRATEGY_DIR.exists():
        raise HTTPException(status_code=404, detail="strategy folder not found")

    items = []
    for path in sorted(STRATEGY_DIR.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith("__") or path.is_dir():
            continue
        if path.suffix not in {".py", ".md"}:
            continue

        desc = _extract_strategy_description(path) or ""
        stat = path.stat()
        items.append(
            {
                "name": path.stem,
                "filename": path.name,
                "extension": path.suffix,
                "description": desc,
                "relative_path": str(path.relative_to(STRATEGY_DIR.parent)),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": stat.st_size,
            }
        )

    return {"count": len(items), "strategies": items}

# Optionally serve your built React app (enable for prod):
# app.mount("/", StaticFiles(directory="frontend/build", html=True), name="app")


def _load_cached_orders(days: int) -> tuple[pd.DataFrame | None, Path | None]:
    """
    Load the newest on-disk order_history CSV and trim it to the requested window.
    Used as a fallback when the live Schwab pull fails.
    """
    logs_dir = FILES_DIR / "logs"
    if not logs_dir.exists():
        return None, None

    candidates = sorted(
        logs_dir.glob("*_order_history.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read cached orders %s: %s", path, exc)
            continue

        ts = None
        for col in ("closeTime", "enteredTime", "time"):
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                break

        if ts is not None and ts.notna().any():
            df = df.loc[ts >= cutoff].copy()

        if df.empty:
            continue

        return df, path

    return None, None


def _trades_cache_path(days: int) -> Path:
    safe_days = max(1, min(int(days), TRADES_CACHE_MAX_DAYS))
    return TRADES_CACHE_DIR / f"trades_{safe_days}d.json"


def _coerce_trades_fetched_ts(payload: dict) -> float | None:
    fetched_ts = payload.get("fetched_ts")
    if isinstance(fetched_ts, (int, float)):
        return float(fetched_ts)
    fetched_at = payload.get("fetched_at")
    if isinstance(fetched_at, str):
        try:
            parsed = datetime.fromisoformat(fetched_at)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _load_cached_trades(days: int) -> tuple[dict | None, float | None, Path]:
    path = _trades_cache_path(days)
    if not path.exists():
        return None, None, path
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read cached trades %s: %s", path, exc)
        return None, None, path
    if not isinstance(payload, dict):
        return None, None, path
    trades = payload.get("trades")
    if not isinstance(trades, list):
        return None, None, path
    fetched_ts = _coerce_trades_fetched_ts(payload)
    return payload, fetched_ts, path


def _write_cached_trades(days: int, payload: dict) -> Path:
    path = _trades_cache_path(days)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = dict(payload)
    sanitized.pop("live_error", None)
    path.write_text(json.dumps(_sanitize_for_json(sanitized), indent=2))
    return path


def _build_trades_payload(
    orders_df: pd.DataFrame,
    *,
    source: str,
    cached_path: Path | None = None,
    live_error: Exception | None = None,
    fetched_at: datetime | None = None,
) -> dict:
    results = analyze_order_history(orders_df)

    trades_df: pd.DataFrame = results["trades_df"]
    per_symbol: pd.DataFrame = results["per_symbol"]
    equity_curve: pd.DataFrame = results["equity_curve"]
    daily_pnl: pd.Series = results["daily_pnl"]
    metrics: dict = results["metrics"]

    payload = {
        "metrics": metrics,
        "trades": trades_df.to_dict(orient="records"),
        "per_symbol": per_symbol.to_dict(orient="records"),
        "equity_curve": equity_curve.to_dict(orient="records"),
        "daily_pnl": [
            {"date": str(idx), "pnl": val}
            for idx, val in daily_pnl.items()
        ],
        "order_count": len(orders_df) if hasattr(orders_df, "__len__") else 0,
        "fetched_at": fetched_at or datetime.now(timezone.utc),
        "source": source,
        "cache_file": cached_path.name if cached_path else None,
        "live_error": str(live_error) if live_error else None,
    }
    return _sanitize_for_json(payload)


def _refresh_trades_cache(days: int, *, allow_live: bool = True) -> tuple[dict, Path]:
    safe_days = max(1, min(int(days), TRADES_CACHE_MAX_DAYS))
    live_error: Exception | None = None
    cached_path: Path | None = None
    source = "live"
    orders_df: pd.DataFrame | None = None

    if allow_live:
        try:
            at = AccountsTrading()
            orders_df = get_orders(at, days=safe_days)
        except Exception as exc:
            live_error = exc
            orders_df = None
            logger.warning("Live Schwab pull failed, will try cached orders: %s", exc)

    if orders_df is None or getattr(orders_df, "empty", False):
        cached_df, cached_path = _load_cached_orders(safe_days)
        if cached_df is None:
            raise live_error or RuntimeError("No cached order history available.")
        orders_df = cached_df
        source = f"cache:{cached_path.name if cached_path else 'unknown'}"

    payload = _build_trades_payload(
        orders_df,
        source=source,
        cached_path=cached_path,
        live_error=live_error if allow_live else None,
    )
    cache_path = _write_cached_trades(safe_days, payload)
    return payload, cache_path


def _normalize_saved_holdings(raw_holdings: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for idx, item in enumerate(raw_holdings or []):
        if not isinstance(item, dict):
            logger.warning("Skipping saved holding at index %s (not a dict)", idx)
            continue
        try:
            cleaned.append(HoldingInput(**item).dict())
        except Exception as exc:
            logger.warning("Skipping invalid saved holding at index %s: %s", idx, exc)
    return cleaned


def _read_saved_holdings() -> tuple[list[dict], str | None]:
    if not HOLDINGS_SAVE_PATH.exists():
        return [], None
    try:
        payload = json.loads(HOLDINGS_SAVE_PATH.read_text())
    except Exception as exc:
        logger.error("Failed to read saved holdings file %s: %s", HOLDINGS_SAVE_PATH, exc)
        raise

    if isinstance(payload, dict):
        raw_holdings = payload.get("holdings", [])
        saved_at = payload.get("saved_at")
    else:
        raw_holdings = payload
        saved_at = None

    if raw_holdings is None:
        raw_holdings = []

    if not isinstance(raw_holdings, list):
        raise ValueError("Saved holdings file is malformed (expected a list).")

    cleaned = _normalize_saved_holdings(raw_holdings)
    return cleaned, saved_at


def _write_saved_holdings(holdings: list[dict]) -> tuple[list[dict], str]:
    HOLDINGS_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {"holdings": holdings, "saved_at": saved_at}
    HOLDINGS_SAVE_PATH.write_text(json.dumps(payload, indent=2))
    return holdings, saved_at


def _politician_cache_path(days: int) -> Path:
    safe_days = max(1, min(int(days), POLITICIAN_TRADES_MAX_DAYS))
    return POLITICIAN_TRADES_DIR / f"politician_trades_{safe_days}d.json"


def _load_cached_politician_trades(days: int) -> tuple[dict | None, float | None, Path]:
    path = _politician_cache_path(days)
    if not path.exists():
        return None, None, path
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read cached politician trades %s: %s", path, exc)
        return None, None, path
    if not isinstance(payload, dict):
        return None, None, path
    trades = payload.get("trades")
    if not isinstance(trades, list):
        return None, None, path
    fetched_ts = _coerce_politician_fetched_ts(payload)
    return payload, fetched_ts, path


def _load_latest_cached_politician_trades() -> tuple[dict | None, float | None, Path | None]:
    if not POLITICIAN_TRADES_DIR.exists():
        return None, None, None
    paths = sorted(
        POLITICIAN_TRADES_DIR.glob("politician_trades_*d.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read cached politician trades %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        trades = payload.get("trades")
        if not isinstance(trades, list):
            continue
        fetched_ts = _coerce_politician_fetched_ts(payload)
        return payload, fetched_ts, path
    return None, None, None


def _coerce_politician_fetched_ts(payload: dict) -> float | None:
    fetched_ts = payload.get("fetched_ts")
    if isinstance(fetched_ts, (int, float)):
        return float(fetched_ts)
    fetched_at = payload.get("fetched_at")
    if isinstance(fetched_at, str):
        try:
            parsed = datetime.fromisoformat(fetched_at)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _write_cached_politician_trades(
    days: int,
    trades: list[dict],
    fetched_at: datetime,
    *,
    source: str | None = None,
    sources: list[str] | None = None,
) -> Path:
    path = _politician_cache_path(days)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "days": days,
        "fetched_at": fetched_at.isoformat(),
        "fetched_ts": fetched_at.timestamp(),
        "trades": trades,
        "source": source,
        "sources": sources or [],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


@app.get("/api/trades")
async def api_trades(days: int = 30, refresh: bool = False):
    """
    Pull real orders from Schwab for the last `days`,
    run them through the analyzer, and return JSON that the
    React dashboard can chart.
    """
    safe_days = max(1, min(int(days), TRADES_CACHE_MAX_DAYS))

    if not refresh:
        cached_payload, _cached_ts, cache_path = _load_cached_trades(safe_days)
        if cached_payload:
            response = dict(cached_payload)
            response["source"] = f"cache:{cache_path.name}"
            response.pop("live_error", None)
            return _sanitize_for_json(response)

        loop = asyncio.get_running_loop()

        def _work_cached():
            cached_df, cached_orders_path = _load_cached_orders(safe_days)
            if cached_df is None:
                raise RuntimeError(
                    "No cached trades available. Run `python -m backend.core.refresh_trades_cache`."
                )
            payload = _build_trades_payload(
                cached_df,
                source=f"cache:{cached_orders_path.name if cached_orders_path else 'unknown'}",
                cached_path=cached_orders_path,
                live_error=None,
            )
            _write_cached_trades(safe_days, payload)
            return payload

        try:
            payload = await loop.run_in_executor(None, _work_cached)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

        return payload

    loop = asyncio.get_running_loop()

    def _work_refresh():
        payload, _cache_path = _refresh_trades_cache(safe_days, allow_live=True)
        return payload

    try:
        payload = await loop.run_in_executor(None, _work_refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return payload


# ----- Politician trades -----

@app.get("/api/politician-trades")
def api_politician_trades(days: int = 14, limit: int = 200, refresh: bool = False):
    safe_days = max(1, min(int(days), POLITICIAN_TRADES_MAX_DAYS))
    safe_limit = max(1, min(int(limit), POLITICIAN_TRADES_MAX_LIMIT))
    cached_payload = None
    cached_ts = None
    cache_path = None
    source_list: list[str] = []
    source_label = "unknown"
    if not refresh:
        cached_payload, cached_ts, cache_path = _load_cached_politician_trades(safe_days)
        if cached_payload:
            source_list = cached_payload.get("sources", [])
            source_label = cached_payload.get("source") or "unknown"
    if is_db_enabled() and db_repo and not refresh:
        try:
            with get_session() as session:
                payload = db_repo.load_politician_trades(session, safe_days, safe_limit)
            trades = payload.get("trades", []) if payload else []
            if trades:
                db_ts = _coerce_politician_fetched_ts(payload)
                if cached_payload and cached_ts is not None and (db_ts is None or cached_ts > db_ts):
                    return _sanitize_for_json(
                        {
                            "trades": cached_payload.get("trades", [])[:safe_limit],
                            "count": min(len(cached_payload.get("trades", [])), safe_limit),
                            "days": safe_days,
                            "limit": safe_limit,
                            "cached": True,
                            "stale": False,
                            "fetched_at": cached_payload.get("fetched_at"),
                            "cache_file": cache_path.name if cache_path else None,
                            "source": source_label,
                            "sources": source_list,
                        }
                    )
                return _sanitize_for_json(
                    {
                        "trades": trades[:safe_limit],
                        "count": min(len(trades), safe_limit),
                        "days": safe_days,
                        "limit": safe_limit,
                        "cached": True,
                        "stale": False,
                        "fetched_at": payload.get("fetched_at"),
                        "cache_file": None,
                        "source": "db",
                        "sources": payload.get("sources", []),
                    }
                )
        except Exception as exc:
            logger.warning("DB politician trades lookup failed: %s", exc)
    if cached_payload and not refresh:
        trades = cached_payload.get("trades", [])
        return _sanitize_for_json(
            {
                "trades": trades[:safe_limit],
                "count": min(len(trades), safe_limit),
                "days": safe_days,
                "limit": safe_limit,
                "cached": True,
                "stale": False,
                "fetched_at": cached_payload.get("fetched_at"),
                "cache_file": cache_path.name,
                "source": source_label,
                "sources": source_list,
            }
        )
    if not refresh:
        fallback_payload, fallback_ts, fallback_path = _load_latest_cached_politician_trades()
        if fallback_payload and fallback_path:
            fallback_trades = fallback_payload.get("trades", [])
            fallback_days = fallback_payload.get("days", safe_days)
            fallback_sources = fallback_payload.get("sources", [])
            fallback_source_label = fallback_payload.get("source") or "unknown"
            return _sanitize_for_json(
                {
                    "trades": fallback_trades[:safe_limit],
                    "count": min(len(fallback_trades), safe_limit),
                    "days": fallback_days,
                    "limit": safe_limit,
                    "cached": True,
                    "stale": False,
                    "fetched_at": fallback_payload.get("fetched_at"),
                    "cache_file": fallback_path.name,
                    "source": fallback_source_label,
                    "sources": fallback_sources,
                    "requested_days": safe_days,
                }
            )
        raise HTTPException(
            status_code=404,
            detail="No cached politician trades available. Use refresh=true to fetch new data.",
        )

    try:
        from backend.politician_trades.collector import fetch_trades, resolve_sources

        since_date = datetime.utcnow() - timedelta(days=safe_days)
        sources_override = os.getenv("POLITICIAN_TRADES_SOURCES", "").strip() or POLITICIAN_TRADES_DEFAULT_SOURCES
        trades = fetch_trades(since_date, sources=sources_override)
        source_list = resolve_sources(sources_override)
        source_label = "+".join(source_list) if source_list else "unknown"

        def _trade_sort_key(trade: dict) -> datetime:
            for key in ("transaction_date", "report_date", "filed_date"):
                dt = trade.get(key)
                if isinstance(dt, datetime):
                    return dt
                if isinstance(dt, date):
                    return datetime.combine(dt, datetime.min.time())
            return datetime.min

        trades.sort(key=_trade_sort_key, reverse=True)
        safe_trades = _sanitize_for_json(trades)
        fetched_at = datetime.now(timezone.utc)
        cache_path = _write_cached_politician_trades(
            safe_days,
            safe_trades,
            fetched_at,
            source=source_label,
            sources=source_list,
        )
        if is_db_enabled() and db_repo:
            try:
                with get_session() as session:
                    db_repo.save_politician_trades(session, safe_trades, fetched_at=fetched_at)
            except Exception as exc:
                logger.warning("Failed to persist politician trades to DB: %s", exc)

        return _sanitize_for_json(
            {
                "trades": safe_trades[:safe_limit],
                "count": min(len(safe_trades), safe_limit),
                "days": safe_days,
                "limit": safe_limit,
                "cached": False,
                "stale": False,
                "fetched_at": fetched_at,
                "cache_file": cache_path.name,
                "source": source_label,
                "sources": source_list,
            }
        )
    except Exception as exc:
        logger.warning("Politician trades fetch failed: %s", exc)
        if cached_payload:
            trades = cached_payload.get("trades", [])
            return _sanitize_for_json(
                {
                    "trades": trades[:safe_limit],
                    "count": min(len(trades), safe_limit),
                    "days": safe_days,
                    "limit": safe_limit,
                    "cached": True,
                    "stale": True,
                    "fetched_at": cached_payload.get("fetched_at"),
                    "cache_file": cache_path.name,
                    "source": source_label,
                    "sources": source_list,
                    "live_error": str(exc),
                }
            )
        fallback_payload, fallback_ts, fallback_path = _load_latest_cached_politician_trades()
        if fallback_payload and fallback_path:
            fallback_trades = fallback_payload.get("trades", [])
            fallback_days = fallback_payload.get("days", safe_days)
            fallback_sources = fallback_payload.get("sources", [])
            fallback_source_label = fallback_payload.get("source") or "unknown"
            return _sanitize_for_json(
                {
                    "trades": fallback_trades[:safe_limit],
                    "count": min(len(fallback_trades), safe_limit),
                    "days": fallback_days,
                    "limit": safe_limit,
                    "cached": True,
                    "stale": True,
                    "fetched_at": fallback_payload.get("fetched_at"),
                    "cache_file": fallback_path.name,
                    "source": fallback_source_label,
                    "sources": fallback_sources,
                    "live_error": str(exc),
                    "requested_days": safe_days,
                }
            )
        raise HTTPException(status_code=500, detail=str(exc))


# ----- Holdings analysis -----

@app.post("/api/holdings/analyze")
def api_holdings_analyze(req: HoldingsAnalyzeRequest):
    try:
        result = analyze_holdings([h.dict() for h in req.holdings], price_period=req.price_period or "1y")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _sanitize_for_json(result)


@app.get("/api/holdings/saved")
def api_holdings_saved():
    try:
        holdings, saved_at = _read_saved_holdings()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _sanitize_for_json(
        {
            "holdings": holdings,
            "count": len(holdings),
            "saved_at": saved_at,
            "path": str(HOLDINGS_SAVE_PATH),
        }
    )


@app.post("/api/holdings/saved")
def api_holdings_save(req: HoldingsSaveRequest):
    try:
        holdings, saved_at = _write_saved_holdings([h.dict() for h in req.holdings])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _sanitize_for_json(
        {"holdings": holdings, "count": len(holdings), "saved_at": saved_at, "path": str(HOLDINGS_SAVE_PATH)}
    )


@app.delete("/api/holdings/saved")
def api_holdings_clear():
    try:
        if HOLDINGS_SAVE_PATH.exists():
            HOLDINGS_SAVE_PATH.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "cleared", "path": str(HOLDINGS_SAVE_PATH)}


@app.get("/api/stock-lists")
def api_stock_lists():
    try:
        lists = [_stock_list_payload(list_id) for list_id in STOCK_LIST_SPECS.keys()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"count": len(lists), "lists": lists}


@app.get("/api/stock-lists/{list_id}")
def api_stock_list_detail(list_id: str, with_tickers: bool = False):
    try:
        return _stock_list_payload(list_id, include_tickers=with_tickers)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/stock-lists/{list_id}/refresh")
def api_stock_list_refresh(list_id: str):
    try:
        payload = _refresh_stock_list(list_id, include_tickers=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return payload


@app.get("/api/suggest/daily-buy/history")
def api_daily_buy_history(limit: Optional[int] = None):
    entries = _list_suggestions_history(DEFAULT_SUGGESTER, limit=limit)
    return {"horizon": DEFAULT_SUGGESTER, "count": len(entries), "history": entries}


@app.get("/api/suggest/daily-buy/history/{date_str}")
def api_daily_buy_history_date(date_str: str):
    try:
        return _load_suggestions_history_entry(DEFAULT_SUGGESTER, date_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/suggest/buy-sell/history")
def api_buy_sell_history(horizon: Optional[str] = None, limit: Optional[int] = None):
    try:
        key = _normalize_suggester_key(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    entries = _list_suggestions_history(key, limit=limit)
    return {"horizon": key, "count": len(entries), "history": entries}


@app.get("/api/suggest/buy-sell/history/{date_str}")
def api_buy_sell_history_date(date_str: str, horizon: Optional[str] = None):
    try:
        key = _normalize_suggester_key(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        return _load_suggestions_history_entry(key, date_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/suggest/daily-buy")
@app.post("/api/suggest/buy-sell")
async def api_daily_buy_suggester(body: dict = Body(default_factory=dict)):
    """
    Run one of the buy/sell suggesters and return suggestions suitable for the UI tables.

    Body shape (all optional):
      - horizon/mode/script/job: selector for which suggester to run
      - tickers: list[str]
      - stock_list / stock_list_id / list_id: use a saved ticker list by id (e.g., "sp500")
      - period: str (e.g., "1y")
      - config / cfg: dict overrides for Config fields
      - force/refresh/rerun: bypass cache and run fresh
      - use_cache: default True; if False always reruns
      - cache_only: read cached file if present; do not rerun when missing
    """
    horizon_raw = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    try:
        horizon_key = _normalize_suggester_key(horizon_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        tickers, stock_list_id, stock_list_label = _resolve_stock_list_and_tickers(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    period = body.get("period") or _default_period_for_horizon(horizon_key)
    cfg_overrides = body.get("config") or body.get("cfg") or {}
    force_rerun = bool(body.get("force") or body.get("refresh") or body.get("rerun"))
    prefer_cache = bool(body.get("use_cache", True))
    cache_only = bool(body.get("cache_only", False))

    token = _register_buy_suggester_token(horizon_key)
    try:
        return await _execute_suggester(
            horizon_key,
            tickers=tickers,
            stock_list_id=stock_list_id,
            period=period,
            cfg_overrides=cfg_overrides,
            prefer_cache=prefer_cache,
            force_rerun=force_rerun,
            cache_only=cache_only,
            cancel_token=token,
        )
    except SuggesterCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        _release_buy_suggester_token(horizon_key, token)


@app.post("/api/suggest/buy-sell/stop")
async def api_stop_buy_sell_suggester(body: dict = Body(default_factory=dict)):
    """
    Stop any in-flight buy/sell suggester runs (single horizon or run-all).
    """
    horizon_raw = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    scope = None
    if isinstance(horizon_raw, str) and horizon_raw.strip():
        cleaned = horizon_raw.strip().lower()
        if cleaned not in ("all", "*"):
            try:
                scope = _normalize_suggester_key(cleaned)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
    cancelled = _cancel_buy_suggester_tokens(scope)
    return {"cancelled": cancelled, "scope": scope or "all"}


@app.post("/api/suggest/buy-sell/run-all")
async def api_buy_sell_run_all(body: dict = Body(default_factory=dict)):
    """
    Run all configured horizon buy suggesters in one request (short, swing, and long term).
    Accepts the same payload as /api/suggest/buy-sell; stock list/tickers are reused for each horizon.
    """
    default_horizons = list(SUGGESTER_RUNNERS.keys())
    horizon_param = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    horizons_raw = body.get("horizons")

    if horizons_raw is not None:
        if not isinstance(horizons_raw, (list, tuple, set)):
            raise HTTPException(status_code=400, detail="horizons must be a list of horizon ids.")
        requested = horizons_raw
    elif isinstance(horizon_param, str) and str(horizon_param).lower() not in ("all", "*"):
        requested = [horizon_param]
    else:
        requested = default_horizons

    horizons: list[str] = []
    seen: set[str] = set()
    
    logger.debug("API buy sell run all")
    for raw in requested:
        try:
            normalized = _normalize_suggester_key(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if normalized in seen:
            continue
        seen.add(normalized)
        horizons.append(normalized)
    
    logger.debug(f"api buy sell run all horizons: {horizons}")

    if not horizons:
        raise HTTPException(status_code=400, detail="No horizons requested.")

    try:
        tickers, stock_list_id, stock_list_label = _resolve_stock_list_and_tickers(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    cfg_overrides = body.get("config") or body.get("cfg") or {}
    force_rerun = bool(body.get("force") or body.get("refresh") or body.get("rerun"))
    prefer_cache = bool(body.get("use_cache", True))
    cache_only = bool(body.get("cache_only", False))
    period_override = body.get("period")
    period_years = {
        h: _period_to_years(period_override or _default_period_for_horizon(h), 1) for h in horizons
    }
    need_shared = False
    cache_hits: dict[str, bool] = {}
    if prefer_cache and not force_rerun:
        for h in horizons:
            cached_df, _ = _load_cached_suggestions(h)
            cache_hits[h] = cached_df is not None
    for h in horizons:
        if cache_only:
            continue
        if force_rerun or not prefer_cache or not cache_hits.get(h, False):
            need_shared = True
            break

    shared_data: SharedSuggesterData | None = None
    runner_kwargs: dict[str, dict[str, Any]] = {}
    token = _register_buy_suggester_token("run_all")
    try:
        if need_shared:
            try:
                max_years = max(period_years.values()) if period_years else 1
                shared_data = build_shared_suggester_data(
                    tickers=tickers,
                    years=max_years,
                    feature_lookback=252,
                    include_spy=True,
                    cancel_token=token,
                )
            except SuggesterCancelled as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to build shared suggester data: {exc}")
            runner_kwargs = {
                h: {"shared_data": shared_data, "feature_lookback": shared_data.feature_lookback}
                for h in horizons
            }

        async def _run_one(horizon_key: str, kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
            return await _execute_suggester(
                horizon_key,
                tickers=tickers,
                stock_list_id=stock_list_id,
                period=period_override or _default_period_for_horizon(horizon_key),
                cfg_overrides=cfg_overrides,
                prefer_cache=prefer_cache,
                force_rerun=force_rerun,
                cache_only=cache_only,
                runner_kwargs=kwargs,
                cancel_token=token,
            )

        results_list = await asyncio.gather(*[_run_one(h, runner_kwargs.get(h)) for h in horizons])
        results = {h: payload for h, payload in zip(horizons, results_list)}
    except SuggesterCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        _release_buy_suggester_token("run_all", token)
    total_count = sum(
        payload.get("count", 0) for payload in results.values() if isinstance(payload, dict)
    )

    response = {
        "horizons": horizons,
        "results": results,
        "count": total_count,
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
        "force_rerun": force_rerun,
        "use_cache": prefer_cache,
        "cache_only": cache_only,
    }
    return _sanitize_for_json(response)


def _options_request_settings(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize shared options suggester request settings."""
    cfg_overrides = body.get("config") or body.get("cfg") or {}
    force_rerun = bool(body.get("force") or body.get("refresh") or body.get("rerun"))
    prefer_cache = bool(body.get("use_cache", True))
    cache_only = bool(body.get("cache_only", False))
    period_override = body.get("period")

    limit_raw = body.get("limit") or body.get("top_n")
    try:
        limit = int(limit_raw) if limit_raw is not None else 10
    except Exception:
        limit = 10
    strike_count_raw = body.get("strike_count") or body.get("strikes")
    try:
        strike_count = int(strike_count_raw) if strike_count_raw is not None else 8
    except Exception:
        strike_count = 8
    include_puts = bool(body.get("include_puts", True))
    strike_range = body.get("strike_range") or body.get("range") or "NTM"

    return {
        "cfg_overrides": cfg_overrides,
        "force_rerun": force_rerun,
        "prefer_cache": prefer_cache,
        "cache_only": cache_only,
        "period_override": period_override,
        "limit": limit,
        "strike_count": strike_count,
        "include_puts": include_puts,
        "strike_range": strike_range,
    }


async def _run_options_suggester_flow(
    horizon_key: str,
    *,
    tickers: Optional[list[str]],
    stock_list_id: Optional[str],
    stock_list_label: Optional[str],
    suggestions_payload: Optional[dict[str, Any]] = None,
    cfg_overrides: dict[str, Any],
    force_rerun: bool,
    prefer_cache: bool,
    cache_only: bool,
    period_override: Optional[str],
    limit: int,
    strike_count: int,
    include_puts: bool,
    strike_range: str,
    cancel_token: Optional[CancelToken] = None,
) -> dict[str, Any]:
    """Build options suggestions for a single horizon."""
    if suggestions_payload is None:
        if not force_rerun:
            suggestions_payload = _load_cached_suggestions_payload(
                horizon_key,
                stock_list_id=stock_list_id,
                stock_list_label=stock_list_label,
            )
        if suggestions_payload is None:
            suggestions_payload = await _execute_suggester(
                horizon_key,
                tickers=tickers,
                stock_list_id=stock_list_id,
                period=period_override or _default_period_for_horizon(horizon_key),
                cfg_overrides=cfg_overrides,
                prefer_cache=prefer_cache,
                force_rerun=force_rerun,
                cache_only=cache_only,
                cancel_token=cancel_token,
            )

    suggestion_rows = suggestions_payload.get("suggestions") or []
    loop = asyncio.get_running_loop()
    try:
        options_rows = await loop.run_in_executor(
            None,
            lambda: _build_option_suggestions(
                suggestion_rows,
                horizon=horizon_key,
                limit=limit,
                strike_count=strike_count,
                include_puts=include_puts,
                strike_range=strike_range,
                cancel_token=cancel_token,
            ),
        )
    except SuggesterCancelled:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    stock_list_label = suggestions_payload.get("stock_list_label") or stock_list_label
    stock_list_id = suggestions_payload.get("stock_list") or stock_list_id

    meta = {
        "horizon": horizon_key,
        "limit": limit,
        "strike_count": strike_count,
        "include_puts": include_puts,
        "range": strike_range,
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
        "suggestions_count": suggestions_payload.get("count"),
        "generated_at": suggestions_payload.get("last_run"),
    }

    persisted_ts = _persist_options_suggestions(options_rows, horizon_key, meta=meta)
    last_run_ts = persisted_ts or suggestions_payload.get("last_run")

    response = {
        "horizon": horizon_key,
        "last_run": last_run_ts,
        "cached": suggestions_payload.get("cached"),
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
        "suggestions_count": suggestions_payload.get("count"),
        "limit": limit,
        "strike_count": strike_count,
        "include_puts": include_puts,
        "range": strike_range,
        "options": options_rows[:limit],
        "meta": meta,
    }
    return _sanitize_for_json(response)


@app.post("/api/suggest/options")
async def api_options_suggester(body: dict = Body(default_factory=dict)):
    """
    Run a horizon-specific buy suggester (reusing today's cached run when available), then suggest options contracts for each ticker.

    Body shape (all optional):
      - horizon/mode/script/job: selector for which suggester to run
      - tickers: list[str]
      - stock_list / stock_list_id / list_id: use a saved ticker list by id (e.g., "sp500")
      - period: str (e.g., "1y")
      - config / cfg: dict overrides for Config fields
      - force/refresh/rerun: bypass today's cache and run the horizon suggester fresh before building options
      - limit: number of tickers to evaluate for options (default 10)
      - strike_count / strikes: number of strikes above/below ATM to request (default 8)
      - include_puts: include put suggestions (default True)
      - range/strike_range: options chain range filter (default "NTM")
    """
    horizon_raw = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    try:
        horizon_key = _normalize_suggester_key(horizon_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    settings = _options_request_settings(body)

    try:
        tickers, stock_list_id, stock_list_label = _resolve_stock_list_and_tickers(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    token = _register_options_suggester_token(horizon_key)
    try:
        return await _run_options_suggester_flow(
            horizon_key,
            tickers=tickers,
            stock_list_id=stock_list_id,
            stock_list_label=stock_list_label,
            cfg_overrides=settings["cfg_overrides"],
            force_rerun=settings["force_rerun"],
            prefer_cache=settings["prefer_cache"],
            cache_only=settings["cache_only"],
            period_override=settings["period_override"],
            limit=settings["limit"],
            strike_count=settings["strike_count"],
            include_puts=settings["include_puts"],
            strike_range=settings["strike_range"],
            cancel_token=token,
        )
    except SuggesterCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        _release_options_suggester_token(horizon_key, token)


@app.post("/api/suggest/options/stop")
async def api_stop_options_suggester(body: dict = Body(default_factory=dict)):
    """
    Stop any in-flight options suggester runs (single horizon or run-all).
    """
    horizon_raw = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    scope = None
    if isinstance(horizon_raw, str) and horizon_raw.strip():
        cleaned = horizon_raw.strip().lower()
        if cleaned not in ("all", "*"):
            try:
                scope = _normalize_suggester_key(cleaned)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
    cancelled = _cancel_options_suggester_tokens(scope)
    return {"cancelled": cancelled, "scope": scope or "all"}


@app.post("/api/suggest/options/run-all")
async def api_options_suggester_run_all(body: dict = Body(default_factory=dict)):
    """
    Run the options suggester for all configured horizons in one request.
    Accepts the same payload as /api/suggest/options; you can pass `horizons`
    as a list to limit which modes run or set `horizon` to "all".
    """
    default_horizons = list(SUGGESTER_RUNNERS.keys())
    horizon_param = body.get("horizon") or body.get("mode") or body.get("script") or body.get("job")
    horizons_raw = body.get("horizons")

    if horizons_raw is not None:
        if not isinstance(horizons_raw, (list, tuple, set)):
            raise HTTPException(status_code=400, detail="horizons must be a list of horizon ids.")
        requested = horizons_raw
    elif isinstance(horizon_param, str) and str(horizon_param).lower() not in ("all", "*"):
        requested = [horizon_param]
    else:
        requested = default_horizons

    horizons: list[str] = []
    seen: set[str] = set()
    for raw in requested:
        try:
            normalized = _normalize_suggester_key(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if normalized in seen:
            continue
        seen.add(normalized)
        horizons.append(normalized)

    if not horizons:
        raise HTTPException(status_code=400, detail="No horizons requested.")

    try:
        tickers, stock_list_id, stock_list_label = _resolve_stock_list_and_tickers(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    settings = _options_request_settings(body)

    period_override = settings["period_override"]
    period_years = {
        h: _period_to_years(period_override or _default_period_for_horizon(h), 1) for h in horizons
    }
    need_shared = False
    cache_hits: dict[str, bool] = {}
    if settings["prefer_cache"] and not settings["force_rerun"]:
        for h in horizons:
            cached_df, _ = _load_cached_suggestions(h)
            cache_hits[h] = cached_df is not None
    for h in horizons:
        if settings["cache_only"]:
            continue
        if settings["force_rerun"] or not settings["prefer_cache"] or not cache_hits.get(h, False):
            need_shared = True
            break

    results: list[dict[str, Any]] = []
    token = _register_options_suggester_token("run_all")
    try:
        suggestions_by_horizon: dict[str, dict[str, Any]] = {}
        suggestion_errors: dict[str, str] = {}
        shared_data: SharedSuggesterData | None = None
        runner_kwargs: dict[str, dict[str, Any]] = {}
        if need_shared:
            try:
                max_years = max(period_years.values()) if period_years else 1
                shared_data = build_shared_suggester_data(
                    tickers=tickers,
                    years=max_years,
                    feature_lookback=252,
                    include_spy=True,
                    cancel_token=token,
                )
            except SuggesterCancelled as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:
                suggestion_errors = {
                    h: f"Failed to build shared suggester data: {exc}" for h in horizons
                }
            else:
                runner_kwargs = {
                    h: {"shared_data": shared_data, "feature_lookback": shared_data.feature_lookback}
                    for h in horizons
                }

        async def _run_one(horizon_key: str, kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
            return await _execute_suggester(
                horizon_key,
                tickers=tickers,
                stock_list_id=stock_list_id,
                period=period_override or _default_period_for_horizon(horizon_key),
                cfg_overrides=settings["cfg_overrides"],
                prefer_cache=settings["prefer_cache"],
                force_rerun=settings["force_rerun"],
                cache_only=settings["cache_only"],
                runner_kwargs=kwargs,
                cancel_token=token,
            )

        results_list = await asyncio.gather(
            *[_run_one(h, runner_kwargs.get(h)) for h in horizons],
            return_exceptions=True,
        )
        for horizon_key, payload in zip(horizons, results_list):
            if isinstance(payload, Exception):
                if isinstance(payload, SuggesterCancelled):
                    raise HTTPException(status_code=409, detail=str(payload))
                if isinstance(payload, HTTPException):
                    suggestion_errors[horizon_key] = str(payload.detail)
                else:
                    suggestion_errors[horizon_key] = str(payload)
            elif isinstance(payload, dict):
                suggestions_by_horizon[horizon_key] = payload
            else:
                suggestion_errors[horizon_key] = "Unexpected suggester response."

        for horizon_key in horizons:
            if token.is_cancelled():
                raise SuggesterCancelled(token.reason)
            suggestions_payload = suggestions_by_horizon.get(horizon_key)
            if suggestions_payload is None:
                error = suggestion_errors.get(horizon_key, "Missing buy suggester payload.")
                results.append(
                    {
                        "horizon": horizon_key,
                        "ok": False,
                        "error": f"Buy suggester failed: {error}",
                        "status": 500,
                    }
                )
                continue
            try:
                outcome = await _run_options_suggester_flow(
                    horizon_key,
                    tickers=tickers,
                    stock_list_id=stock_list_id,
                    stock_list_label=stock_list_label,
                    suggestions_payload=suggestions_payload,
                    cfg_overrides=settings["cfg_overrides"],
                    force_rerun=settings["force_rerun"],
                    prefer_cache=settings["prefer_cache"],
                    cache_only=settings["cache_only"],
                    period_override=period_override,
                    limit=settings["limit"],
                    strike_count=settings["strike_count"],
                    include_puts=settings["include_puts"],
                    strike_range=settings["strike_range"],
                    cancel_token=token,
                )
                results.append({"horizon": horizon_key, "ok": True, "result": outcome})
            except HTTPException as exc:
                results.append({"horizon": horizon_key, "ok": False, "error": exc.detail, "status": exc.status_code})
            except Exception as exc:
                results.append({"horizon": horizon_key, "ok": False, "error": str(exc), "status": 500})
    except SuggesterCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        _release_options_suggester_token("run_all", token)

    succeeded = sum(1 for entry in results if entry.get("ok"))
    payload = {
        "horizons": horizons,
        "results": results,
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "stock_list": stock_list_id,
        "stock_list_label": stock_list_label,
        "limit": settings["limit"],
        "strike_count": settings["strike_count"],
        "include_puts": settings["include_puts"],
        "range": settings["strike_range"],
    }
    return _sanitize_for_json(payload)


@app.get("/api/suggest/options/history")
def api_options_history(horizon: Optional[str] = None, limit: Optional[int] = None):
    try:
        key = _normalize_suggester_key(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        entries = _list_options_suggestions_history(key, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"horizon": key, "count": len(entries), "history": entries}


@app.get("/api/suggest/options/history/{date_str}")
def api_options_history_entry(
    date_str: str,
    horizon: Optional[str] = None,
    limit: Optional[int] = None,
    strike_count: Optional[int] = None,
    include_puts: Optional[bool] = None,
    strike_range: Optional[str] = None,
):
    try:
        key = _normalize_suggester_key(horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        entry = _load_options_suggestions_history_entry(key, date_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    meta = entry.get("meta") or {}
    try:
        limit_val = int(limit) if limit is not None else int(meta.get("limit", 10))
    except Exception:
        limit_val = 10
    try:
        strike_count_val = int(strike_count) if strike_count is not None else int(meta.get("strike_count", 8))
    except Exception:
        strike_count_val = 8
    if include_puts is None:
        include_puts_val = bool(meta.get("include_puts", True))
    else:
        include_puts_val = bool(include_puts)
    strike_range_val = strike_range or meta.get("range") or entry.get("range") or "NTM"

    options_rows_raw = entry.get("options") or []
    if options_rows_raw and isinstance(options_rows_raw, list):
        options_rows = options_rows_raw
    else:
        options_rows = []

    response = {
        "horizon": key,
        "date": entry.get("date"),
        "label": entry.get("label"),
        "last_run": entry.get("last_run"),
        "limit": limit_val,
        "strike_count": strike_count_val,
        "include_puts": include_puts_val,
        "range": strike_range_val,
        "count": len(options_rows),
        "options": options_rows[:limit_val],
        "source": "history",
        "meta": meta,
    }
    return _sanitize_for_json(response)
