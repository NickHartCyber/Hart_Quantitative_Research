from __future__ import annotations

import argparse

from backend.api import _refresh_trades_cache
from backend.platform_apis.schwab_api.get_refresh_token import refresh_tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh cached trades payload used by the Nick's Trades dashboard.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of orders to pull into the cache.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    refresh_tokens()
    payload, cache_path = _refresh_trades_cache(args.days, allow_live=True)
    fetched_at = payload.get("fetched_at") or "unknown"
    print(f"Cached trades to {cache_path} (fetched_at={fetched_at}).")


if __name__ == "__main__":
    main()
