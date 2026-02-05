from __future__ import annotations

import logging
import os
import random
import time
from datetime import date, datetime
from typing import Any, Iterable

import requests

SEC_API_BASE = "https://data.sec.gov"
SEC_SEARCH_BASE = "https://efts.sec.gov/LATEST"
SEC_TICKER_CIK_URL = os.getenv(
    "SEC_TICKER_CIK_URL",
    "https://www.sec.gov/files/company_tickers.json",
)

DEFAULT_TIMEOUT = float(os.getenv("SEC_API_TIMEOUT", "30"))
REQUEST_MIN_INTERVAL = float(os.getenv("SEC_API_MIN_INTERVAL", "0.12"))
TICKER_CIK_CACHE_TTL = float(os.getenv("SEC_TICKER_CIK_TTL", "86400"))
USER_AGENT = (
    os.getenv("SEC_USER_AGENT")
    or os.getenv("EDGAR_USER_AGENT")
    or "HartQuantitativeResearch/1.0"
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

FUNDAMENTAL_FORM_BASES = (
    "10-K",
    "10-Q",
    "8-K",
    "20-F",
    "40-F",
    "6-K",
)

log = logging.getLogger("edgar_api")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_LAST_REQUEST_TS = 0.0
_TICKER_CIK_CACHE: dict[str, str] = {}
_TICKER_CIK_CACHE_TS = 0.0


def _normalize_cik(cik: str | int) -> str:
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise ValueError("CIK must contain digits.")
    return digits.zfill(10)


def _normalize_cik_int(cik: str | int) -> str:
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise ValueError("CIK must contain digits.")
    return str(int(digits))


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


def _request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_tries: int = 4,
) -> dict[str, Any]:
    timeout = timeout or DEFAULT_TIMEOUT
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        _throttle()
        try:
            if method == "GET":
                resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout)
            else:
                resp = requests.post(url, json=payload, headers=REQUEST_HEADERS, timeout=timeout)

            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(float(retry_after))
                else:
                    _sleep_with_jitter(0.5 * attempt)
                continue

            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_tries:
                break
            _sleep_with_jitter(0.5 * attempt)

    if last_exc:
        log.error("SEC request failed: %s", last_exc)
        raise last_exc
    return {}


def _to_list(values: Iterable[str | int] | None) -> list[str]:
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def get_ticker_cik_map(
    *,
    force_refresh: bool = False,
    timeout: float | None = None,
) -> dict[str, str]:
    global _TICKER_CIK_CACHE_TS

    now = time.monotonic()
    if (
        not force_refresh
        and _TICKER_CIK_CACHE
        and TICKER_CIK_CACHE_TTL > 0
        and (now - _TICKER_CIK_CACHE_TS) < TICKER_CIK_CACHE_TTL
    ):
        return dict(_TICKER_CIK_CACHE)

    payload = _request_json("GET", SEC_TICKER_CIK_URL, timeout=timeout)
    mapping: dict[str, str] = {}
    if isinstance(payload, dict):
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker") or "").strip().upper()
            cik_val = entry.get("cik_str")
            if not ticker or cik_val is None:
                continue
            try:
                cik_norm = _normalize_cik(cik_val)
            except ValueError:
                continue
            mapping[ticker] = cik_norm

    if mapping:
        _TICKER_CIK_CACHE.clear()
        _TICKER_CIK_CACHE.update(mapping)
        _TICKER_CIK_CACHE_TS = now
    return dict(mapping)


def resolve_cik_from_ticker(
    ticker: str,
    *,
    refresh: bool = False,
    timeout: float | None = None,
) -> str | None:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    mapping = get_ticker_cik_map(force_refresh=refresh, timeout=timeout)
    if symbol in mapping:
        return mapping[symbol]
    if "." in symbol:
        alt = symbol.replace(".", "-")
        if alt in mapping:
            return mapping[alt]
    if "-" in symbol:
        alt = symbol.replace("-", ".")
        if alt in mapping:
            return mapping[alt]
    return None


def _normalize_form(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().upper()


def _normalize_forms(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [v for v in (_normalize_form(item) for item in values) if v]


def is_form_variant(form: str | None, base_forms: Iterable[str]) -> bool:
    form_norm = _normalize_form(form)
    if not form_norm:
        return False
    for base in _normalize_forms(base_forms):
        if form_norm == base or form_norm.startswith(base):
            return True
    return False


def extract_company_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    filings = submissions.get("filings")
    recent: Any = None
    if isinstance(filings, dict):
        recent = filings.get("recent")
    if isinstance(recent, list):
        return [item for item in recent if isinstance(item, dict)]
    if not isinstance(recent, dict):
        return []

    lengths = [len(values) for values in recent.values() if isinstance(values, list)]
    if not lengths:
        return []
    count = max(lengths)

    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row: dict[str, Any] = {}
        for key, values in recent.items():
            if not isinstance(values, list):
                continue
            if idx < len(values):
                row[key] = values[idx]
        if row:
            rows.append(row)
    return rows


def get_company_submissions_file(file_name: str, *, timeout: float | None = None) -> dict[str, Any]:
    name = str(file_name).strip()
    if not name:
        raise ValueError("file_name is required.")
    url = f"{SEC_API_BASE}/submissions/{name}"
    return _request_json("GET", url, timeout=timeout)


def get_company_filings(
    cik: str | int,
    *,
    include_historical: bool = False,
    max_historical_files: int | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    submissions = get_company_submissions(cik, timeout=timeout)
    filings = extract_company_filings(submissions)

    if include_historical:
        files = submissions.get("filings", {}).get("files")
        if isinstance(files, list):
            for idx, file_meta in enumerate(files):
                if max_historical_files is not None and idx >= max_historical_files:
                    break
                if not isinstance(file_meta, dict):
                    continue
                name = file_meta.get("name")
                if not name:
                    continue
                data = get_company_submissions_file(str(name), timeout=timeout)
                filings.extend(extract_company_filings(data))

    return filings


def get_company_fundamental_filings(
    cik: str | int,
    *,
    forms: Iterable[str] | None = None,
    include_variants: bool = True,
    include_historical: bool = False,
    max_historical_files: int | None = None,
    sort_by: str | None = "filingDate",
    limit: int | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    base_forms = _normalize_forms(forms) or list(FUNDAMENTAL_FORM_BASES)
    filings = get_company_filings(
        cik,
        include_historical=include_historical,
        max_historical_files=max_historical_files,
        timeout=timeout,
    )

    if include_variants:
        filtered = [item for item in filings if is_form_variant(item.get("form"), base_forms)]
    else:
        allowed = set(base_forms)
        filtered = [item for item in filings if _normalize_form(item.get("form")) in allowed]

    if sort_by:
        filtered.sort(key=lambda row: str(row.get(sort_by) or ""), reverse=True)
    if limit is not None:
        filtered = filtered[: max(0, int(limit))]
    return filtered


def search_filings(
    query: str,
    *,
    start: int = 0,
    size: int = 50,
    form_types: Iterable[str] | None = None,
    ciks: Iterable[str | int] | None = None,
    from_date: str | date | datetime | None = None,
    to_date: str | date | datetime | None = None,
    category: str = "custom",
    sort: str = "filedAt",
    order: str = "desc",
    timeout: float | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Query the SEC full-text search index for filings.

    See: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
    """
    payload: dict[str, Any] = {
        "keys": query,
        "start": max(0, int(start)),
        "size": max(1, int(size)),
        "category": category,
        "sort": sort,
        "order": order,
    }
    if form_types:
        payload["forms"] = _to_list(form_types)
    if ciks:
        payload["ciks"] = [_normalize_cik(c) for c in ciks]
    from_norm = _normalize_date(from_date)
    to_norm = _normalize_date(to_date)
    if from_norm:
        payload["from"] = from_norm
    if to_norm:
        payload["to"] = to_norm
    if extra_payload:
        payload.update(extra_payload)

    return _request_json("POST", f"{SEC_SEARCH_BASE}/search-index", payload=payload, timeout=timeout)


def extract_search_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("hits")
    if isinstance(hits, dict):
        inner = hits.get("hits")
        if isinstance(inner, list):
            out: list[dict[str, Any]] = []
            for item in inner:
                if isinstance(item, dict):
                    source = item.get("_source")
                    out.append(source if isinstance(source, dict) else item)
            return out
    results = payload.get("results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    return []


def get_company_submissions(cik: str | int, *, timeout: float | None = None) -> dict[str, Any]:
    cik_norm = _normalize_cik(cik)
    url = f"{SEC_API_BASE}/submissions/CIK{cik_norm}.json"
    return _request_json("GET", url, timeout=timeout)


def get_company_facts(cik: str | int, *, timeout: float | None = None) -> dict[str, Any]:
    cik_norm = _normalize_cik(cik)
    url = f"{SEC_API_BASE}/api/xbrl/companyfacts/CIK{cik_norm}.json"
    return _request_json("GET", url, timeout=timeout)


def get_company_concept(
    cik: str | int,
    taxonomy: str,
    tag: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    cik_norm = _normalize_cik(cik)
    url = f"{SEC_API_BASE}/api/xbrl/companyconcept/CIK{cik_norm}/{taxonomy}/{tag}.json"
    return _request_json("GET", url, timeout=timeout)


def get_frames(
    taxonomy: str,
    tag: str,
    unit: str,
    period: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    url = f"{SEC_API_BASE}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
    return _request_json("GET", url, timeout=timeout)


def get_filing_index(
    cik: str | int,
    accession_no: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    cik_num = _normalize_cik_int(cik)
    acc_no = str(accession_no).strip()
    acc_no_nodash = acc_no.replace("-", "")
    url = f"{SEC_API_BASE}/Archives/edgar/data/{cik_num}/{acc_no_nodash}/index.json"
    return _request_json("GET", url, timeout=timeout)
