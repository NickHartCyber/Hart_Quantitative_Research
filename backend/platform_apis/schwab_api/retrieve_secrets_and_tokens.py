"""
Utility functions for securely retrieving and storing secrets and tokens
for the Schwab API and Stock News API.

Secrets are expected to be stored in JSON files inside the `schwab_secrets/` folder.
All functions handle FileNotFoundError and JSONDecodeError gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# --------------------------------------------------------------------------- #
# Base path
# --------------------------------------------------------------------------- #

SECRETS_DIR = Path(__file__).resolve().parents[4] / "schwab_secrets"
AUTH_STATUS_FILE = SECRETS_DIR / "auth_status.json"
AUTH_REFRESH_INTERVAL_DAYS = 7


def _load_json(file_path: Path) -> dict[str, Any] | None:
    """
    Load JSON content from a file.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file.

    Returns
    -------
    dict | None
        Parsed JSON as a dict, or None on failure.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"Loaded secrets from {file_path.name}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    return None


def _save_json(file_path: Path, value: dict[str, Any]) -> bool:
    """
    Save a dict as JSON to a file.

    Parameters
    ----------
    file_path : Path
        Path to the file to save.
    value : dict
        Data to write.

    Returns
    -------
    bool
        True if write succeeded, False otherwise.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
        logger.debug(f"Updated {file_path.name} successfully.")
        return True
    except Exception as exc:
        logger.error(f"Failed to save {file_path}: {exc}")
        return False


def _load_auth_status() -> dict[str, Any]:
    """
    Load the auth status JSON (or return an empty dict if missing/invalid).
    """
    data = _load_json(AUTH_STATUS_FILE)
    return data if isinstance(data, dict) else {}


def _parse_iso_ts(value: str) -> datetime | None:
    """
    Best-effort ISO8601 parser that tolerates trailing 'Z'.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"Could not parse datetime string: {value}")
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def record_auth_refresh(service: str, *, when: datetime | None = None) -> bool:
    """
    Record the timestamp of the latest auth refresh for a given service.
    """
    ts = (when or datetime.now(timezone.utc)).isoformat()
    data = _load_auth_status()
    data[service] = {"last_refresh": ts}
    return _save_json(AUTH_STATUS_FILE, data)


def clear_auth_refresh(service: str) -> bool:
    """
    Remove the stored refresh timestamp for a given service.
    """
    data = _load_auth_status()
    if service in data:
        data.pop(service, None)
        return _save_json(AUTH_STATUS_FILE, data)
    return True


def get_auth_status(
    now: datetime | None = None, refresh_interval_days: int = AUTH_REFRESH_INTERVAL_DAYS
) -> dict[str, dict[str, Any]]:
    """
    Summarize last-refresh metadata for supported auth flows.
    """
    data = _load_auth_status()
    now_ts = now or datetime.now(timezone.utc)

    def _build(service: str) -> dict[str, Any]:
        entry = data.get(service) or {}
        raw_ts = entry.get("last_refresh")
        last_refresh_dt = _parse_iso_ts(raw_ts) if raw_ts else None
        age_days: float | None = None
        stale = True

        if last_refresh_dt:
            delta = now_ts - last_refresh_dt
            age_days = delta.total_seconds() / 86400
            stale = age_days >= refresh_interval_days

        return {
            "last_refresh": last_refresh_dt.isoformat() if last_refresh_dt else None,
            "age_days": age_days,
            "stale": stale,
            "refresh_interval_days": refresh_interval_days,
        }

    return {"schwab": _build("schwab"), "gmail": _build("gmail")}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def retrieve_schwab_secret_dict() -> dict[str, Any] | None:
    """
    Retrieve Schwab secrets (app_key, app_secret, etc.).

    Returns
    -------
    dict | None
        Dict with secret values or None if unavailable.
    """
    return _load_json(SECRETS_DIR / "schwab_secrets.json")


def retrieve_auth_token_value() -> dict[str, Any] | None:
    """
    Retrieve the stored Schwab OAuth token dict.

    Returns
    -------
    dict | None
        Dict with auth token fields (access_token, refresh_token, etc.), or None.
    """
    return _load_json(SECRETS_DIR / "schwab_auth_token.json")


def store_auth_token_value(value: dict[str, Any], *, record_refresh_time: bool = True) -> bool:
    """
    Store Schwab OAuth tokens to disk.

    Parameters
    ----------
    value : dict
        Token dict to save.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    ok = _save_json(SECRETS_DIR / "schwab_auth_token.json", value)
    if ok and record_refresh_time:
        record_auth_refresh("schwab")
    return ok


def retrieve_news_api_token() -> dict[str, Any] | None:
    """
    Retrieve Stock News API token.

    Returns
    -------
    dict | None
        Dict with API token, or None if unavailable.
    """
    return _load_json(SECRETS_DIR / "stock_news_api_token.json")
