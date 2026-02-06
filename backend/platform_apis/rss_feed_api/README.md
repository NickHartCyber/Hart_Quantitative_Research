# RSS Feed for News

Get company news via RSS feeds (Yahoo Finance, Google News, optional extras).

## Configuration
- `RSS_FEED_SOURCES`: Comma-separated sources. Default: `yahoo,google`. Options: `yahoo`, `google`, `finviz`.
- `RSS_FEED_EXTRA_URLS`: Comma-separated RSS URLs (supports `{symbol}` and `{query}` placeholders).
- `RSS_FEED_TIMEOUT`: Request timeout seconds. Default: `12`.
- `RSS_FEED_CACHE_TTL`: In-memory cache TTL seconds. Default: `600`.
- `RSS_FEED_CACHE_MAX`: Max cached symbols. Default: `128`.
- `RSS_FEED_LANG`: Language, e.g., `en-US`.
- `RSS_FEED_REGION`: Region, e.g., `US`.

