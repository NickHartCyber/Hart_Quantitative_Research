from __future__ import annotations

import argparse

from backend.api_prices_tiingo import refresh_market_indices_cache


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh cached market indices payload used by the Home page."
    )
    parser.add_argument(
        "--analysis-tf",
        default="6M",
        help="Timeframe used for AI analysis and trend stats.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=5,
        help="Number of headlines to pull per index.",
    )
    parser.add_argument(
        "--include-ai",
        action="store_true",
        default=True,
        help="Include OpenAI bull/bear commentary.",
    )
    parser.add_argument(
        "--no-ai",
        dest="include_ai",
        action="store_false",
        help="Disable AI commentary.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        default=True,
        help="Include pre/post market in 1D series when available.",
    )
    parser.add_argument(
        "--no-extended",
        dest="extended",
        action="store_false",
        help="Exclude pre/post market in 1D series.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload, cache_path = refresh_market_indices_cache(
        analysis_tf=args.analysis_tf,
        news_limit=args.news_limit,
        include_ai=args.include_ai,
        extended=args.extended,
    )
    as_of = payload.get("as_of") or payload.get("cached_at") or "unknown"
    print(f"Cached market indices to {cache_path} (as_of={as_of}).")


if __name__ == "__main__":
    main()
