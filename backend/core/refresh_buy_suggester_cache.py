from __future__ import annotations

import argparse
import asyncio

from backend.api import (
    STOCK_LIST_SPECS,
    SUGGESTER_RUNNERS,
    _default_period_for_horizon,
    _execute_suggester,
    _load_cached_suggestions,
    _normalize_suggester_key,
    _normalize_stock_list_id,
    _period_to_years,
    _resolve_stock_list_and_tickers,
    _run_options_suggester_flow,
)
from backend.core.horizon_buy_suggesters import build_shared_suggester_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh cached buy suggester results for all horizons.",
    )
    parser.add_argument(
        "--stock-list",
        default="mass_combo",
        help="Stock list id to use (default: mass_combo).",
    )
    parser.add_argument(
        "--stock-lists",
        nargs="*",
        help="Optional list of stock list ids to refresh (use 'all' for every list).",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        help="Optional horizon keys/aliases to refresh (defaults to all).",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Override history period for all horizons (e.g., 1y, 3y, 5y).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use cached results when available instead of forcing a fresh run.",
    )
    parser.add_argument(
        "--refresh-options",
        action="store_true",
        default=False,
        help="Also refresh options suggester cache for the selected horizons.",
    )
    parser.add_argument(
        "--options-limit",
        type=int,
        default=10,
        help="Number of tickers to evaluate for options per horizon.",
    )
    parser.add_argument(
        "--options-strike-count",
        type=int,
        default=8,
        help="Number of strikes above/below ATM to evaluate.",
    )
    parser.add_argument(
        "--options-strike-range",
        default="NTM",
        help="Options chain range filter (e.g., NTM, ITM).",
    )
    parser.add_argument(
        "--options-include-puts",
        action="store_true",
        default=True,
        help="Include put ideas in options suggestions (default).",
    )
    parser.add_argument(
        "--options-no-puts",
        dest="options_include_puts",
        action="store_false",
        help="Exclude put ideas from options suggestions.",
    )
    return parser


def _normalize_horizons(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(SUGGESTER_RUNNERS.keys())
    horizons: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        normalized = _normalize_suggester_key(raw)
        if normalized in seen:
            continue
        seen.add(normalized)
        horizons.append(normalized)
    if not horizons:
        raise ValueError("No valid horizons provided.")
    return horizons


def _normalize_stock_lists(raw_values: list[str] | None, fallback: str | None) -> list[str | None]:
    if raw_values:
        lowered = {str(val).strip().lower() for val in raw_values if str(val).strip()}
        if "all" in lowered or "*" in lowered:
            return list(STOCK_LIST_SPECS.keys())
        stock_lists: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            normalized = _normalize_stock_list_id(raw)
            if normalized in seen:
                continue
            seen.add(normalized)
            stock_lists.append(normalized)
        if stock_lists:
            return stock_lists
    if isinstance(fallback, str) and fallback.strip():
        return [_normalize_stock_list_id(fallback)]
    return [None]


async def _refresh_all(
    *,
    horizons: list[str],
    stock_list_id: str | None,
    period_override: str | None,
    use_cache: bool,
) -> tuple[list[object], list[str], str | None, str | None]:
    body: dict[str, object] = {}
    if stock_list_id:
        body["stock_list"] = stock_list_id
    tickers, resolved_list_id, stock_list_label = _resolve_stock_list_and_tickers(body)

    force_rerun = not use_cache
    prefer_cache = use_cache
    cache_hits: dict[str, bool] = {}
    need_shared = False
    if prefer_cache and not force_rerun:
        for horizon in horizons:
            cached_df, _ = _load_cached_suggestions(horizon)
            cache_hits[horizon] = cached_df is not None
    for horizon in horizons:
        if force_rerun or not prefer_cache or not cache_hits.get(horizon, False):
            need_shared = True
            break

    runner_kwargs: dict[str, dict[str, object]] = {}
    if need_shared:
        period_years = {
            h: _period_to_years(period_override or _default_period_for_horizon(h), 1) for h in horizons
        }
        max_years = max(period_years.values()) if period_years else 1
        shared_data = build_shared_suggester_data(
            tickers=tickers,
            years=max_years,
            feature_lookback=252,
            include_spy=True,
        )
        runner_kwargs = {
            h: {"shared_data": shared_data, "feature_lookback": shared_data.feature_lookback}
            for h in horizons
        }

    async def _run_one(horizon_key: str) -> dict[str, object]:
        return await _execute_suggester(
            horizon_key,
            tickers=tickers,
            stock_list_id=resolved_list_id,
            period=period_override or _default_period_for_horizon(horizon_key),
            cfg_overrides=None,
            prefer_cache=prefer_cache,
            force_rerun=force_rerun,
            cache_only=False,
            runner_kwargs=runner_kwargs.get(horizon_key),
        )

    results = await asyncio.gather(*[_run_one(h) for h in horizons], return_exceptions=True)
    return results, tickers, resolved_list_id, stock_list_label


async def _refresh_options(
    *,
    horizons: list[str],
    tickers: list[str] | None,
    stock_list_id: str | None,
    stock_list_label: str | None,
    suggestions_by_horizon: dict[str, dict[str, object]] | None,
    period_override: str | None,
    options_limit: int,
    options_strike_count: int,
    options_include_puts: bool,
    options_strike_range: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for horizon in horizons:
        try:
            suggestions_payload = None
            cache_only = False
            if suggestions_by_horizon:
                suggestions_payload = suggestions_by_horizon.get(horizon)
                if suggestions_payload is None:
                    cache_only = True
            outcome = await _run_options_suggester_flow(
                horizon,
                tickers=tickers,
                stock_list_id=stock_list_id,
                stock_list_label=stock_list_label,
                suggestions_payload=suggestions_payload,
                cfg_overrides={},
                force_rerun=False,
                prefer_cache=True,
                cache_only=cache_only,
                period_override=period_override,
                limit=options_limit,
                strike_count=options_strike_count,
                include_puts=options_include_puts,
                strike_range=options_strike_range,
            )
            results.append({"horizon": horizon, "ok": True, "result": outcome})
        except Exception as exc:
            results.append({"horizon": horizon, "ok": False, "error": str(exc)})
    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    horizons = _normalize_horizons(args.horizons)
    stock_lists = _normalize_stock_lists(args.stock_lists, args.stock_list)

    for stock_list_id in stock_lists:
        label = STOCK_LIST_SPECS.get(stock_list_id, {}).get("label") if stock_list_id else "default"
        print(f"Stock list: {label} ({stock_list_id or 'default'})")
        results, tickers, resolved_list_id, stock_list_label = asyncio.run(
            _refresh_all(
                horizons=horizons,
                stock_list_id=stock_list_id,
                period_override=args.period,
                use_cache=args.use_cache,
            )
        )
        for horizon, payload in zip(horizons, results):
            if isinstance(payload, Exception):
                error = getattr(payload, "detail", None) or str(payload)
                print(f"{horizon}: failed ({error})")
                continue
            if not isinstance(payload, dict):
                print(f"{horizon}: failed (unexpected suggester result)")
                continue
            count = payload.get("count", 0)
            last_run = payload.get("last_run") or "unknown"
            cached = payload.get("cached")
            status = "cached" if cached else "refreshed"
            print(f"{horizon}: {status} ({count} ideas, last_run={last_run})")

        suggestions_by_horizon = {
            horizon: payload for horizon, payload in zip(horizons, results) if isinstance(payload, dict)
        }

        if args.refresh_options:
            options_results = asyncio.run(
                _refresh_options(
                    horizons=horizons,
                    tickers=tickers,
                    stock_list_id=resolved_list_id,
                    stock_list_label=stock_list_label,
                    suggestions_by_horizon=suggestions_by_horizon,
                    period_override=args.period,
                    options_limit=args.options_limit,
                    options_strike_count=args.options_strike_count,
                    options_include_puts=args.options_include_puts,
                    options_strike_range=args.options_strike_range,
                )
            )
            for entry in options_results:
                horizon = entry.get("horizon", "unknown")
                if entry.get("ok") and isinstance(entry.get("result"), dict):
                    result = entry["result"]
                    last_run = result.get("last_run") or "unknown"
                    count = len(result.get("options", []) or [])
                    print(f"{horizon} options: refreshed ({count} ideas, last_run={last_run})")
                else:
                    error = entry.get("error") or "unknown error"
                    print(f"{horizon} options: failed ({error})")


if __name__ == "__main__":
    main()
