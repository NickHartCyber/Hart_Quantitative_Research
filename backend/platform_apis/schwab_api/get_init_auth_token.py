"""
Initial OAuth authorization flow for Charles Schwab Trader API.

This script:
1) Builds an authorization URL using client credentials.
2) Opens the user's browser to authorize the app.
3) Prompts the user to paste the redirected URL (containing ?code=...).
4) Exchanges the authorization code for tokens and stores them.

References
----------
Unofficial guide:
https://medium.com/@carstensavage/the-unofficial-guide-to-charles-schwabs-trader-apis-14c1f5bc1d57

Notes
-----
- The redirect URI must match exactly what you registered in the Schwab app portal.
- Never log or print raw tokens or client secrets.
- Tokens typically expire; you should implement refresh logic separately.
"""

from __future__ import annotations

import sys
import webbrowser
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from loguru import logger

from backend.core.core_helpers import basic_auth_header

# Support running both as a script and as an imported module.
try:
    from .retrieve_secrets_and_tokens import (  # type: ignore
        retrieve_schwab_secret_dict,
        store_auth_token_value,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from retrieve_secrets_and_tokens import (
        retrieve_schwab_secret_dict,
        store_auth_token_value,
    )

AUTH_BASE = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
DEFAULT_REDIRECT_URI = "https://127.0.0.1"  # Must match the app settings exactly


def _load_auth_settings() -> tuple[str, str, str, str | None]:
    """
    Load Schwab OAuth settings from disk (app_key, app_secret, optional redirect_uri/scope).

    Returns
    -------
    (app_key, app_secret, redirect_uri, scope)
    """
    secrets = retrieve_schwab_secret_dict()
    if not isinstance(secrets, dict):
        raise ValueError("Schwab secrets file missing or invalid JSON.")

    app_key = str(secrets.get("app_key", "")).strip()
    app_secret = str(secrets.get("app_secret", "")).strip()
    if not app_key or not app_secret:
        raise ValueError("Schwab secrets missing app_key/app_secret.")

    redirect_uri = str(secrets.get("redirect_uri") or DEFAULT_REDIRECT_URI).strip()
    scope_val = secrets.get("scope") or secrets.get("scopes")
    if isinstance(scope_val, (list, tuple)):
        scope_val = " ".join(str(s).strip() for s in scope_val if str(s).strip())
    scope = str(scope_val).strip() if scope_val else None

    return app_key, app_secret, redirect_uri, scope


def construct_init_auth_url() -> tuple[str, str, str, str]:
    """
    Build the OAuth authorization URL and return (app_key, app_secret, redirect_uri, auth_url).

    Returns
    -------
    tuple[str, str, str]
        (app_key, app_secret, redirect_uri, authorization_url)

    Notes
    -----
    The `auth_url` should be opened in a browser. After consenting, the
    provider will redirect to your configured redirect URI with a `?code=...` parameter.
    """
    app_key, app_secret, redirect_uri, scope = _load_auth_settings()

    params: dict[str, str] = {
        "client_id": app_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }
    if scope:
        params["scope"] = scope

    auth_url = f"{AUTH_BASE}?{urlencode(params)}"

    logger.info("Open this URL to authenticate:")
    logger.info(auth_url)

    return app_key, app_secret, redirect_uri, auth_url


def construct_headers_and_payload(
    returned_url: str, app_key: str, app_secret: str, redirect_uri: str
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Construct the headers and form payload to exchange an authorization code for tokens.

    Parameters
    ----------
    returned_url : str
        The full URL the browser was redirected to after consent (contains `?code=...`).
    app_key : str
        Schwab application client ID.
    app_secret : str
        Schwab application client secret.

    Returns
    -------
    (headers, payload) : (dict, dict)
        Headers including Basic Auth, and the x-www-form-urlencoded payload.

    Raises
    ------
    ValueError
        If an authorization `code` cannot be found in the returned URL.
    """
    # Parse ?code=... from the returned URL robustly.
    parsed = urlparse(returned_url)
    query = parse_qs(parsed.query)
    codes = query.get("code", [])
    if not codes or not codes[0]:
        raise ValueError("Authorization code not found in returned URL. Expected '?code=...'.")
    auth_code = codes[0]

    returned_redirect = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    expected_redirect = redirect_uri.rstrip("/")
    if returned_redirect and returned_redirect != expected_redirect:
        logger.warning(
            "Redirect URI mismatch. Returned {} but configured {}. Continuing with configured value.",
            returned_redirect,
            expected_redirect,
        )

    headers = {
        "Authorization": basic_auth_header(app_key, app_secret),
        "Content-Type": "application/x-www-form-urlencoded",
    }

    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }

    return headers, payload


def retrieve_tokens(headers: dict[str, str], payload: dict[str, str]) -> dict:
    """
    Exchange authorization code for tokens.

    Parameters
    ----------
    headers : dict
        Request headers (must include Basic Auth).
    payload : dict
        x-www-form-urlencoded body with grant_type, code, redirect_uri.

    Returns
    -------
    dict
        Parsed token response as a dictionary.

    Raises
    ------
    requests.HTTPError
        If the HTTP response is not 2xx.
    requests.RequestException
        For network/connection errors.
    """
    resp: requests.Response | None = None
    try:
        resp = requests.post(url=TOKEN_URL, headers=headers, data=payload, timeout=30)
        logger.debug(f"Token exchange response status={resp.status_code}")
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        status = exc.response.status_code if exc.response is not None else "unknown"
        logger.error(f"Token request failed with status {status}: {body[:500]}")
        raise
    except requests.RequestException as exc:
        logger.error(f"Token request failed: {exc}")
        raise

    tokens = resp.json()
    # Avoid logging full tokens; log only keys present.
    logger.info(f"Received token payload keys: {list(tokens.keys())}")
    return tokens


def main() -> str:
    """
    Interactive launcher for the initial OAuth flow.

    Steps
    -----
    1) Build the auth URL and open it in the default web browser.
    2) Ask the user to paste the redirected URL containing `?code=...`.
    3) Exchange the code for tokens and store them securely.

    Returns
    -------
    str
        "Done!" on success.

    Raises
    ------
    ValueError
        If the pasted URL does not contain an authorization code.
    requests.RequestException
        If the token exchange fails.
    """
    app_key, app_secret, redirect_uri, auth_url = construct_init_auth_url()
    try:
        webbrowser.open(auth_url)
    except Exception as exc:
        logger.warning(
            f"Could not open browser automatically: {exc}. Please open the URL manually."
        )

    logger.info("Paste the full returned URL (the one your browser was redirected to):")
    returned_url = input("> ").strip()

    headers, payload = construct_headers_and_payload(returned_url, app_key, app_secret, redirect_uri)
    tokens = retrieve_tokens(headers=headers, payload=payload)
    store_auth_token_value(tokens)

    logger.info("Initial tokens stored successfully.")
    return "Done!"


if __name__ == "__main__":
    # Schwab requires re-authorization periodically; schedule accordingly.
    try:
        sys.exit(0 if main() == "Done!" else 1)
    except Exception:
        logger.exception("Authorization flow failed.")
        sys.exit(1)
