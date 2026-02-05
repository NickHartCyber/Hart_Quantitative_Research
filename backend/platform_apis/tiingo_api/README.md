# Tiingo API

Used for:
- Daily OHLCV bars
- Adjusted prices
- Limited intraday (if needed later)
- Basic fundamentals (backup)

Token loading:
- Preferred: `TIINGO_API_TOKEN` or `TIINGO_API_KEY` env var
- Fallback: `quant_secrets/tiingo_api_token.json` (sibling of repo root)

Usage (module functions):
```python
from backend.platform_apis.tiingo_api.tiingo_api import (
    get_daily_prices,
    get_iex_prices,
    get_fundamentals_daily,
)

bars = get_daily_prices("AAPL", start_date="2024-01-01", end_date="2024-01-31")
intraday = get_iex_prices("AAPL", start_date="2024-01-02", resample_freq="1min")
fundamentals = get_fundamentals_daily("AAPL")
```

Usage (client):
```python
from backend.platform_apis.tiingo_api.tiingo_api import TiingoClient

client = TiingoClient()
latest = client.get_latest_prices(["AAPL", "MSFT"])
```

