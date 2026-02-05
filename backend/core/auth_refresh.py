"""Background token refresh task."""
from __future__ import annotations

import asyncio
import os

from loguru import logger

from backend.platform_apis.schwab_api.get_refresh_token import refresh_tokens


async def periodic_token_refresh(interval_seconds: int | None = None) -> None:
    """
    Periodically refresh Schwab access tokens.

    This is a best-effort loop; failures are logged and retried.
    """
    if interval_seconds is None:
        interval_seconds = int(os.getenv("HART_QUANT_REFRESH_SECONDS", "1500"))

    while True:
        try:
            refresh_tokens()
        except Exception as exc:
            logger.warning(f"Token refresh failed: {exc}")
        await asyncio.sleep(max(60, interval_seconds))
