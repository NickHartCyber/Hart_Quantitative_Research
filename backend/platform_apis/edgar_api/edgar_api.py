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

DEFAULT_TIMEOUT = float(os.getenv("SEC_API_TIMEOUT", "30"))
REQUEST_MIN_INTERVAL = float(os.getenv("SEC_API_MIN_INTERVAL", "0.12"))
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

log = logging.getLogger("edgar_api")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_LAST_REQUEST_TS = 0.0


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
