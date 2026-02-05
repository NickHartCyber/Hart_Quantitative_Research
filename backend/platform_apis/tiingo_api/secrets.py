from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("tiingo_api.secrets")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

SECRETS_DIR = Path(__file__).resolve().parents[4] / "quant_secrets"
DEFAULT_TOKEN_PATH = SECRETS_DIR / "tiingo_api_token.json"

TOKEN_ENV_VARS = ("TIINGO_API_TOKEN", "TIINGO_API_KEY", "TIINGO_TOKEN")
TOKEN_JSON_KEYS = (
    "api_token",
    "apiKey",
    "api_key",
    "token",
    "key",
    "tiingo_api_token",
)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        log.error("Tiingo token file not found: %s", path)
    except json.JSONDecodeError:
        log.error("Tiingo token file is not valid JSON: %s", path)
    return None


def resolve_tiingo_token(
    token: str | None = None,
    *,
    token_path: str | Path | None = None,
    env_vars: tuple[str, ...] = TOKEN_ENV_VARS,
) -> str:
    """
    Resolve a Tiingo API token from (in order):
    1) explicit argument
    2) environment variables
    3) JSON file in quant_secrets/tiingo_api_token.json
    """
    if token and str(token).strip():
        return str(token).strip()

    for env_var in env_vars:
        value = os.getenv(env_var)
        if value and value.strip():
            return value.strip()

    path = Path(token_path) if token_path else Path(os.getenv("TIINGO_API_TOKEN_PATH", DEFAULT_TOKEN_PATH))
    payload = _load_json(path)
    if isinstance(payload, dict):
        for key in TOKEN_JSON_KEYS:
            value = payload.get(key)
            if value and str(value).strip():
                return str(value).strip()
    elif isinstance(payload, str) and payload.strip():
        return payload.strip()

    raise FileNotFoundError(
        "Tiingo API token not found. Set TIINGO_API_TOKEN (or TIINGO_API_KEY) "
        "or add it to quant_secrets/tiingo_api_token.json."
    )
