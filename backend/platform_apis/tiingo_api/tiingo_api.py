from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping

import requests

from .secrets import resolve_tiingo_token

TIINGO_API_BASE = os.getenv("TIINGO_API_BASE", "https://api.tiingo.com")
DEFAULT_TIMEOUT = float(os.getenv("TIINGO_API_TIMEOUT", "30"))
REQUEST_MIN_INTERVAL = float(os.getenv("TIINGO_API_MIN_INTERVAL", "0.12"))

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

log = logging.getLogger("tiingo_api")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_LAST_REQUEST_TS = 0.0
_SESSION = requests.Session()


def _normalize_date(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if len(text) >= 10:
        return text[:10]
    return text


def _to_csv(values: Iterable[str] | str | None) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values.strip() or None
    items = [str(v).strip() for v in values if str(v).strip()]
    return ",".join(items) if items else None


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    return {k: v for k, v in params.items() if v is not None}


def _throttle() -> None:
    global _LAST_REQUEST_TS
    if REQUEST_MIN_INTERVAL <= 0:
        return
    now = time.monotonic()
    wait_s = REQUEST_MIN_INTERVAL - (now - _LAST_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_REQUEST_TS = time.monotonic()


def _sleep_with_jitter(base_s: float) -> None:
    time.sleep(base_s + random.random() * 0.35)


def _build_headers(token: str | None, extra_headers: Mapping[str, str] | None = None) -> dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    if token:
        headers["Authorization"] = f"Token {token}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    max_tries: int = 4,
    session: requests.Session | None = None,
    base_url: str | None = None,
) -> requests.Response:
    url = _join_url(base_url or TIINGO_API_BASE, path)
    timeout = timeout or DEFAULT_TIMEOUT
    token_value = resolve_tiingo_token(token, token_path=token_path)
    headers = _build_headers(token_value)
    payload = dict(payload) if payload else None
    params = _clean_params(params)
    last_exc: Exception | None = None

    for attempt in range(1, max_tries + 1):
        _throttle()
        try:
            sess = session or _SESSION
            if method.upper() == "GET":
                resp = sess.get(url, params=params, headers=headers, timeout=timeout)
            else:
                resp = sess.post(url, params=params, json=payload, headers=headers, timeout=timeout)

            if resp.status_code in RETRYABLE_STATUS:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(float(retry_after))
                else:
                    _sleep_with_jitter(0.5 * attempt)
                continue

            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_tries:
                break
            _sleep_with_jitter(0.5 * attempt)

    if last_exc:
        log.error("Tiingo request failed: %s", last_exc)
        raise last_exc
    raise RuntimeError("Tiingo request failed with no response.")


def request_raw(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    max_tries: int = 4,
    session: requests.Session | None = None,
    base_url: str | None = None,
) -> requests.Response:
    return _request(
        method,
        path,
        params=params,
        payload=payload,
        token=token,
        token_path=token_path,
        timeout=timeout,
        max_tries=max_tries,
        session=session,
        base_url=base_url,
    )


def request_json(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    max_tries: int = 4,
    session: requests.Session | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | list[Any]:
    resp = _request(
        method,
        path,
        params=params,
        payload=payload,
        token=token,
        token_path=token_path,
        timeout=timeout,
        max_tries=max_tries,
        session=session,
        base_url=base_url,
    )
    if resp.status_code == 204:
        return {}
    return resp.json()


def request_text(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    max_tries: int = 4,
    session: requests.Session | None = None,
    base_url: str | None = None,
) -> str:
    resp = _request(
        method,
        path,
        params=params,
        payload=payload,
        token=token,
        token_path=token_path,
        timeout=timeout,
        max_tries=max_tries,
        session=session,
        base_url=base_url,
    )
    return resp.text


# --------------------------------------------------------------------------- #
# Daily prices and metadata
# --------------------------------------------------------------------------- #


def get_ticker_metadata(
    ticker: str,
    *,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | list[Any]:
    return request_json(
        "GET",
        f"/tiingo/daily/{ticker}",
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


def list_tickers(
    *,
    tickers: Iterable[str] | str | None = None,
    exchange: str | None = None,
    asset_type: str | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    is_active: bool | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    params = {
        "tickers": _to_csv(tickers),
        "exchange": exchange,
        "assetType": asset_type,
        "startDate": _normalize_date(start_date),
        "endDate": _normalize_date(end_date),
        "isActive": is_active,
    }
    if extra_params:
        params.update(extra_params)

    paths = ("/tiingo/daily", "/tiingo/daily/")
    base_urls = (None, f"{TIINGO_API_BASE.rstrip('/')}/api")
    last_exc: Exception | None = None

    for base in base_urls:
        for path in paths:
            try:
                return request_json(
                    "GET",
                    path,
                    params=params,
                    token=token,
                    token_path=token_path,
                    timeout=timeout,
                    session=session,
                    base_url=base,
                )
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status == 404:
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                continue

    if last_exc:
        raise last_exc
    return {}


def get_daily_prices(
    ticker: str,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    frequency: str | None = "daily",
    resample_freq: str | None = None,
    columns: Iterable[str] | str | None = None,
    fmt: str | None = "json",
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | list[Any] | str:
    params = {
        "startDate": _normalize_date(start_date),
        "endDate": _normalize_date(end_date),
        "frequency": frequency,
        "resampleFreq": resample_freq,
        "columns": _to_csv(columns),
        "format": fmt,
    }
    if fmt and fmt.lower() == "csv":
        return request_text(
            "GET",
            f"/tiingo/daily/{ticker}/prices",
            params=params,
            token=token,
            token_path=token_path,
            timeout=timeout,
            session=session,
        )
    return request_json(
        "GET",
        f"/tiingo/daily/{ticker}/prices",
        params=params,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


def get_latest_prices(
    tickers: Iterable[str] | str,
    *,
    columns: Iterable[str] | str | None = None,
    fmt: str | None = "json",
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | list[Any] | str:
    params = {
        "tickers": _to_csv(tickers),
        "columns": _to_csv(columns),
        "format": fmt,
    }
    if fmt and fmt.lower() == "csv":
        return request_text(
            "GET",
            "/tiingo/daily/prices",
            params=params,
            token=token,
            token_path=token_path,
            timeout=timeout,
            session=session,
        )
    return request_json(
        "GET",
        "/tiingo/daily/prices",
        params=params,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


# --------------------------------------------------------------------------- #
# Intraday (IEX) prices
# --------------------------------------------------------------------------- #


def get_iex_prices(
    ticker: str,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    resample_freq: str | None = None,
    columns: Iterable[str] | str | None = None,
    fmt: str | None = "json",
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | list[Any] | str:
    params = {
        "startDate": _normalize_date(start_date),
        "endDate": _normalize_date(end_date),
        "resampleFreq": resample_freq,
        "columns": _to_csv(columns),
        "format": fmt,
    }
    if fmt and fmt.lower() == "csv":
        return request_text(
            "GET",
            f"/iex/{ticker}/prices",
            params=params,
            token=token,
            token_path=token_path,
            timeout=timeout,
            session=session,
        )
    return request_json(
        "GET",
        f"/iex/{ticker}/prices",
        params=params,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


def get_iex_latest(
    tickers: Iterable[str] | str,
    *,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    params = {"tickers": _to_csv(tickers)}
    if extra_params:
        params.update(extra_params)
    return request_json(
        "GET",
        "/iex",
        params=params,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


# --------------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------------- #


def get_fundamentals(
    ticker: str,
    endpoint: str,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    payload = {
        "startDate": _normalize_date(start_date),
        "endDate": _normalize_date(end_date),
    }
    if params:
        payload.update(params)
    return request_json(
        "GET",
        f"/tiingo/fundamentals/{ticker}/{endpoint}",
        params=payload,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
    )


def get_fundamentals_daily(
    ticker: str,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    return get_fundamentals(
        ticker,
        "daily",
        start_date=start_date,
        end_date=end_date,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
        params=params,
    )


def get_fundamentals_statements(
    ticker: str,
    *,
    statement_type: str | None = None,
    as_reported: bool | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    extra: dict[str, Any] = {}
    if statement_type is not None:
        extra["statementType"] = statement_type
    if as_reported is not None:
        extra["asReported"] = as_reported
    if params:
        extra.update(params)

    return get_fundamentals(
        ticker,
        "statements",
        start_date=start_date,
        end_date=end_date,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
        params=extra,
    )


def get_fundamentals_metrics(
    ticker: str,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    token: str | None = None,
    token_path: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    return get_fundamentals(
        ticker,
        "metrics",
        start_date=start_date,
        end_date=end_date,
        token=token,
        token_path=token_path,
        timeout=timeout,
        session=session,
        params=params,
    )


# --------------------------------------------------------------------------- #
# Convenience client
# --------------------------------------------------------------------------- #


@dataclass
class TiingoClient:
    api_token: str | None = None
    token_path: str | None = None
    base_url: str = TIINGO_API_BASE
    timeout: float = DEFAULT_TIMEOUT
    session: requests.Session = field(default_factory=requests.Session)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        max_tries: int = 4,
    ) -> dict[str, Any] | list[Any]:
        return request_json(
            method,
            path,
            params=params,
            payload=payload,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            max_tries=max_tries,
            session=self.session,
            base_url=self.base_url,
        )

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        max_tries: int = 4,
    ) -> str:
        return request_text(
            method,
            path,
            params=params,
            payload=payload,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            max_tries=max_tries,
            session=self.session,
            base_url=self.base_url,
        )

    def get_daily_prices(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any] | str:
        return get_daily_prices(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_latest_prices(self, tickers: Iterable[str] | str, **kwargs: Any) -> dict[str, Any] | list[Any] | str:
        return get_latest_prices(
            tickers,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_iex_prices(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any] | str:
        return get_iex_prices(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_iex_latest(self, tickers: Iterable[str] | str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return get_iex_latest(
            tickers,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_ticker_metadata(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return get_ticker_metadata(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def list_tickers(self, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return list_tickers(
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_fundamentals_daily(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return get_fundamentals_daily(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_fundamentals_statements(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return get_fundamentals_statements(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )

    def get_fundamentals_metrics(self, ticker: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return get_fundamentals_metrics(
            ticker,
            token=self.api_token,
            token_path=self.token_path,
            timeout=self.timeout,
            session=self.session,
            **kwargs,
        )
