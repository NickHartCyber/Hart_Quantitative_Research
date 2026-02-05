from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from backend.platform_apis.tiingo_api.tiingo_api import get_ticker_metadata, list_tickers


def _to_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [row for row in payload["data"] if isinstance(row, dict)]
        if isinstance(payload.get("results"), list):
            return [row for row in payload["results"] if isinstance(row, dict)]
    return []


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    contains: str | None = None,
) -> list[dict[str, Any]]:
    if not contains:
        return rows
    needle = contains.strip().lower()
    if not needle:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        text = " ".join(
            str(row.get(key, "")).lower()
            for key in ("ticker", "symbol", "name", "description")
        )
        if needle in text:
            out.append(row)
    return out


def _select_fields(rows: list[dict[str, Any]], fields: Iterable[str]) -> list[dict[str, Any]]:
    keys = [f for f in fields if f]
    if not keys:
        return rows
    out = []
    for row in rows:
        out.append({k: row.get(k) for k in keys})
    return out


def _print_json(payload: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(payload, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Tiingo supported tickers list and optionally filter.",
    )
    parser.add_argument("--tickers", help="Comma-separated tickers to query directly.")
    parser.add_argument("--check", help="Comma-separated tickers to check; returns found/missing.")
    parser.add_argument("--exchange", help="Exchange filter (e.g., NYSE, NASDAQ).")
    parser.add_argument("--asset-type", help="Asset type filter (e.g., Stock, ETF).")
    parser.add_argument("--active", help="Filter by active status (true/false).")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD).")
    parser.add_argument("--contains", help="Substring filter across ticker/name/description.")
    parser.add_argument("--fields", help="Comma-separated fields to keep in output.")
    parser.add_argument("--limit", type=int, help="Limit output rows.")
    parser.add_argument("--output", help="Write JSON output to file instead of stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    args = parser.parse_args()

    tickers = _to_list(args.tickers)
    check = _to_list(args.check)
    if tickers and check:
        raise SystemExit("Use --tickers or --check, not both.")

    is_active = _parse_bool(args.active) if args.active else None
    rows: list[dict[str, Any]] = []
    if check:
        # For specific tickers, hit the per-ticker metadata endpoint.
        for ticker in check:
            try:
                payload = get_ticker_metadata(ticker)
                if isinstance(payload, dict):
                    rows.append(payload)
                elif isinstance(payload, list):
                    rows.extend([row for row in payload if isinstance(row, dict)])
            except Exception:
                continue
    else:
        payload = list_tickers(
            tickers=tickers or None,
            exchange=args.exchange,
            asset_type=args.asset_type,
            start_date=args.start_date,
            end_date=args.end_date,
            is_active=is_active,
        )
        rows = _extract_rows(payload)
    rows = _filter_rows(rows, contains=args.contains)
    if args.fields:
        rows = _select_fields(rows, _to_list(args.fields))
    if args.limit:
        rows = rows[: max(args.limit, 0)]

    if check:
        found = {str(row.get("ticker") or row.get("symbol") or "").upper() for row in rows}
        requested = {t.upper() for t in check}
        output = {
            "requested": sorted(requested),
            "found": sorted(found & requested),
            "missing": sorted(requested - found),
            "count": len(rows),
            "results": rows,
        }
    else:
        output = {"count": len(rows), "results": rows}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2 if args.pretty else None, default=str)
    else:
        _print_json(output, pretty=args.pretty)


if __name__ == "__main__":
    main()
