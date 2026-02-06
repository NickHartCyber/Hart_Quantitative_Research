"""RSS news integration for ticker/ETF headlines."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import datetime as dt
from html import unescape
import logging
import os
import random
import re
import time
from typing import Any, Iterable
from urllib.parse import urlencode

import requests
from lxml import etree

log = logging.getLogger("rss_feed")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

REQUEST_TIMEOUT = float(os.getenv("RSS_FEED_TIMEOUT", "12"))
REQUEST_MIN_INTERVAL = float(os.getenv("RSS_FEED_MIN_INTERVAL", "0.12"))
CACHE_TTL = float(os.getenv("RSS_FEED_CACHE_TTL", "600"))
CACHE_MAX = int(os.getenv("RSS_FEED_CACHE_MAX", "128"))
RSS_LANG = os.getenv("RSS_FEED_LANG", "en-US")
RSS_REGION = os.getenv("RSS_FEED_REGION", "US")
RSS_SOURCES = os.getenv("RSS_FEED_SOURCES", "yahoo,google")
RSS_EXTRA_URLS = os.getenv("RSS_FEED_EXTRA_URLS", "")
RSS_MAX_FETCH = int(os.getenv("RSS_FEED_MAX_FETCH", "80"))
USER_AGENT = os.getenv("RSS_FEED_USER_AGENT", "HartQuantitativeResearch/1.0")

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
    "Accept-Encoding": "gzip, deflate",
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_SESSION = requests.Session()
_LAST_REQUEST_TS = 0.0
_CACHE: OrderedDict[str, tuple[float, list[dict[str, Any]], str]] = OrderedDict()

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DOWNGRADE_RE = re.compile(r"\b(downgrade|downgraded|cuts? to|lowered to|reduced to|trimmed to)\b", re.I)
_UPGRADE_RE = re.compile(r"\b(upgrade|upgraded|raised to|raises to|boosted to|hiked to)\b", re.I)
_INIT_RE = re.compile(r"\b(initiated|initiation|coverage initiated|starts coverage|started coverage|resumes coverage)\b", re.I)
_PRICE_TARGET_RE = re.compile(r"\b(price target|pt)\b", re.I)
_MAINTAIN_RE = re.compile(r"\b(reiterates|maintains|reaffirmed)\b", re.I)


@dataclass(frozen=True)
class FeedSource:
    key: str
    description: str


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


def _request_feed(url: str, *, timeout: float | None = None, max_tries: int = 4) -> bytes:
    timeout = timeout or REQUEST_TIMEOUT
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        _throttle()
        try:
            resp = _SESSION.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code in RETRYABLE_STATUS:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(float(retry_after))
                else:
                    _sleep_with_jitter(0.5 * attempt)
                continue
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_tries:
                break
            _sleep_with_jitter(0.5 * attempt)
    if last_exc:
        log.warning("RSS fetch failed for %s: %s", url, last_exc)
        raise last_exc
    return b""


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _HTML_TAG_RE.sub(" ", value)
    cleaned = unescape(cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _text_from_node(node) -> str | None:
    if node is None:
        return None
    text = "".join(node.itertext()).strip()
    return text or None


def _find_text(elem, names: Iterable[str]) -> str | None:
    for name in names:
        matches = elem.xpath(f".//*[local-name()='{name}']")
        for match in matches:
            text = _text_from_node(match)
            if text:
                return text
    return None


def _find_link(elem) -> str | None:
    for link_node in elem.xpath(".//*[local-name()='link']"):
        href = link_node.get("href")
        text = _text_from_node(link_node)
        candidate = (href or text or "").strip()
        if not candidate:
            continue
        rel = (link_node.get("rel") or "").lower()
        if rel in ("", "alternate", "related"):
            return candidate
    return None


def _find_author(elem) -> str | None:
    for author_node in elem.xpath(".//*[local-name()='author']"):
        name_node = author_node.xpath(".//*[local-name()='name']")
        if name_node:
            text = _text_from_node(name_node[0])
        else:
            text = _text_from_node(author_node)
        if text:
            return text
    return None


def _coerce_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        pass
    try:
        cleaned = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = dt.datetime.fromisoformat(cleaned)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _classify_headline(title: str) -> str | None:
    if not title:
        return None
    if _DOWNGRADE_RE.search(title):
        return "downgrade"
    if _UPGRADE_RE.search(title):
        return "upgrade"
    if _INIT_RE.search(title):
        return "initiation"
    if _PRICE_TARGET_RE.search(title):
        return "price_target"
    if _MAINTAIN_RE.search(title):
        return "rating_maintained"
    return None


def _build_google_news_url(query: str) -> str:
    lang = RSS_LANG or "en-US"
    region = RSS_REGION or "US"
    lang_code = lang.split("-")[0]
    ceid = f"{region}:{lang_code}"
    params = {
        "q": query,
        "hl": lang,
        "gl": region,
        "ceid": ceid,
    }
    return f"https://news.google.com/rss/search?{urlencode(params)}"


def _build_yahoo_url(symbol: str) -> str:
    params = {"s": symbol, "region": RSS_REGION or "US", "lang": RSS_LANG or "en-US"}
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?{urlencode(params)}"


def _build_finviz_url(symbol: str) -> str:
    params = {"t": symbol}
    return f"https://finviz.com/rss.ashx?{urlencode(params)}"


def _parse_feed(content: bytes, *, source: str, symbol: str | None = None) -> list[dict[str, Any]]:
    if not content:
        return []
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(content, parser=parser)
    except Exception as exc:
        log.warning("Failed to parse RSS feed from %s: %s", source, exc)
        return []

    entries = root.xpath(".//*[local-name()='item']")
    if not entries:
        entries = root.xpath(".//*[local-name()='entry']")

    results: list[dict[str, Any]] = []
    for entry in entries:
        title = _find_text(entry, ["title"])
        if not title:
            continue
        link = _find_link(entry)
        guid = _find_text(entry, ["guid", "id"])
        description = _find_text(entry, ["description", "summary", "content"])
        summary = _strip_html(description)
        publisher = _find_text(entry, ["source"])
        if not publisher:
            publisher = _find_author(entry)

        published_raw = _find_text(entry, ["pubDate", "published", "updated", "dc:date", "date"])
        published_dt = _coerce_datetime(published_raw)
        published_at = published_dt.isoformat() if published_dt else None

        item: dict[str, Any] = {
            "id": guid,
            "title": title,
            "link": link,
            "publisher": publisher,
            "published_at": published_at,
            "summary": summary,
            "source": source,
        }
        if symbol:
            item["related_tickers"] = [symbol]

        headline_type = _classify_headline(title)
        if headline_type:
            item["headline_type"] = headline_type
            item["tags"] = [headline_type, "rating_action"]

        results.append(item)
    return results


def _resolve_sources() -> list[FeedSource]:
    raw = RSS_SOURCES or ""
    keys = [s.strip().lower() for s in raw.split(",") if s.strip()]
    resolved: list[FeedSource] = []
    for key in keys:
        if key == "google":
            resolved.append(FeedSource(key="google", description="Google News"))
        elif key == "yahoo":
            resolved.append(FeedSource(key="yahoo", description="Yahoo Finance"))
        elif key == "finviz":
            resolved.append(FeedSource(key="finviz", description="Finviz"))
        else:
            log.info("Unknown RSS source '%s' ignored.", key)
    return resolved


def _expand_extra_urls(symbol: str, query: str) -> list[str]:
    raw = RSS_EXTRA_URLS or ""
    if not raw.strip():
        return []
    urls: list[str] = []
    for value in raw.split(","):
        url = value.strip()
        if not url:
            continue
        url = url.replace("{symbol}", symbol).replace("{query}", query)
        urls.append(url)
    return urls


def _build_queries(symbol: str) -> list[str]:
    base = symbol.strip()
    if not base:
        return []
    queries = [base]
    if " " not in base and base.isalnum():
        queries.append(f"{base} stock")
        queries.append(f"{base} ETF")
    return list(dict.fromkeys(queries))


def _cache_get(key: str) -> tuple[list[dict[str, Any]], str] | None:
    if CACHE_TTL <= 0:
        return None
    payload = _CACHE.get(key)
    if not payload:
        return None
    ts, items, source = payload
    if (time.monotonic() - ts) > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return list(items), source


def _cache_set(key: str, items: list[dict[str, Any]], source: str) -> None:
    if CACHE_TTL <= 0:
        return
    _CACHE[key] = (time.monotonic(), list(items), source)
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)


def _dedupe_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        key = (title, link)
        if not title or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def get_company_news(ticker: str, *, limit: int = 30) -> tuple[list[dict[str, Any]], str]:
    """
    Fetch RSS headlines for a ticker or ETF symbol.
    Returns a tuple of (items, source_label).
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return [], "rss"

    cache_key = symbol
    cached = _cache_get(cache_key)
    if cached:
        items, source = cached
        return items[:limit], source

    queries = _build_queries(symbol)
    primary_query = queries[0] if queries else symbol
    sources = _resolve_sources()
    source_labels: list[str] = []
    collected: list[dict[str, Any]] = []

    for source in sources:
        try:
            if source.key == "google":
                for query in queries[:2]:
                    url = _build_google_news_url(query)
                    content = _request_feed(url)
                    collected.extend(_parse_feed(content, source=source.key, symbol=symbol))
                source_labels.append(source.key)
            elif source.key == "yahoo":
                url = _build_yahoo_url(symbol)
                content = _request_feed(url)
                collected.extend(_parse_feed(content, source=source.key, symbol=symbol))
                source_labels.append(source.key)
            elif source.key == "finviz":
                url = _build_finviz_url(symbol)
                content = _request_feed(url)
                collected.extend(_parse_feed(content, source=source.key, symbol=symbol))
                source_labels.append(source.key)
        except Exception as exc:
            log.info("RSS source %s failed for %s: %s", source.key, symbol, exc)

    for extra_url in _expand_extra_urls(symbol, primary_query):
        try:
            content = _request_feed(extra_url)
            collected.extend(_parse_feed(content, source="extra", symbol=symbol))
            source_labels.append("extra")
        except Exception as exc:
            log.info("RSS extra feed failed for %s: %s", symbol, exc)

    deduped = _dedupe_items(collected)
    deduped.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    trimmed = deduped[: max(limit, 0)] if limit else deduped
    source_label = "rss:" + ",".join(sorted(set(source_labels))) if source_labels else "rss"
    _cache_set(cache_key, deduped[:RSS_MAX_FETCH], source_label)
    return trimmed, source_label
