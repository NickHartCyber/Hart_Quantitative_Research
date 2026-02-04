"""
Token refresh utility for Charles Schwab Trader API.

This script exchanges a long-lived refresh token for a new access token
(and optionally a new refresh token). Tokens are then stored securely
for downstream API calls.

References
----------
- Schwab API Docs: https://developer.schwab.com/products/trader-api--individual

Notes
-----
- Schwab access tokens expire quickly (≈30 minutes).
- Refresh tokens typically last longer (≈7 days), but you must
  periodically re-run the *initial authorization flow* to get a new one.
- This module should be scheduled (e.g., cron, task runner) to refresh
  every ~29 minutes to avoid access token expiry.
"""

from __future__ import annotations

import sys

import requests
from loguru import logger

from backend.core.core_helpers import basic_auth_header
from .retrieve_secrets_and_tokens import (
    retrieve_auth_token_value,
    retrieve_schwab_secret_dict,
    store_auth_token_value,
)

TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"


def refresh_tokens() -> str | None:
    """
    Refresh Schwab API tokens using the stored refresh token.

    Returns
    -------
    str | None
        "Done!" if successful, otherwise None on failure.

    Raises
    ------
    requests.RequestException
        For network/connection errors.
    """
    logger.info("Starting token refresh...")

    secrets = retrieve_schwab_secret_dict()
    if not isinstance(secrets, dict):
        logger.error("Schwab secrets file missing or invalid; cannot refresh tokens.")
        return None
    app_key = secrets.get("app_key")
    app_secret = secrets.get("app_secret")
    if not app_key or not app_secret:
        logger.error("Schwab secrets missing app_key/app_secret; cannot refresh tokens.")
        return None

    stored_token = retrieve_auth_token_value()
    if not isinstance(stored_token, dict):
        logger.error("Stored Schwab auth token is missing/invalid JSON; cannot refresh.")
        return None

    refresh_token_value = stored_token.get("refresh_token")
    if not refresh_token_value:
        logger.error("No refresh_token found in stored credentials.")
        return None

    headers = {
        "Authorization": basic_auth_header(app_key, app_secret),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token_value,
    }

    try:
        resp = requests.post(url=TOKEN_URL, headers=headers, data=payload, timeout=30)
        logger.debug(f"Refresh token response status={resp.status_code}")
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to refresh token: {exc}")
        return None

    new_tokens: dict = resp.json()
    # Automatic refreshes should not update the manual auth timestamp.
    store_auth_token_value(new_tokens, record_refresh_time=False)

    # Avoid printing tokens — just confirm.
    logger.info("Access/refresh tokens successfully refreshed and stored.")

    return "Done!"


if __name__ == "__main__":
    # Schwab requires refresh every ~30 minutes.
    try:
        sys.exit(0 if refresh_tokens() == "Done!" else 1)
    except Exception:
        logger.exception("Token refresh failed unexpectedly.")
        sys.exit(1)
