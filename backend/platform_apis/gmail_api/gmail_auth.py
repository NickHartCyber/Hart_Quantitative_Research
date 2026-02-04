"""
Auth utilities for Gmail OAuth tokens.

Responsibilities
----------------
- Handle installed-app OAuth flow (interactive or headless).
- Persist and refresh gmail_token.json alongside gmail_credentials.json.
- Record refresh timestamps via retrieve_secrets_and_tokens.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import json
import os
import sys
import urllib.parse
import webbrowser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from loguru import logger

try:
    from backend.platform_apis.schwab_api.retrieve_secrets_and_tokens import record_auth_refresh
except ImportError:  # pragma: no cover - allow direct script execution
    sys.path.append(str(Path(__file__).resolve().parent.parent / "schwab_api"))
    from retrieve_secrets_and_tokens import record_auth_refresh  # type: ignore

# Directory holding token/credentials. Keep this out of version control.
THIS_FOLDER = Path(__file__).resolve().parent / "../../../../schwab_secrets"
AUTH_STATE_FILE = THIS_FOLDER / "gmail_auth_state.json"
HEADLESS_REDIRECT_URI = "http://localhost:8765/"

# OAuth scope: full Gmail (adjust to least privilege if possible)
SCOPES = ["https://mail.google.com/"]


class HeadlessAuthRequired(Exception):
    """Raised when a browser cannot be opened and manual auth is required."""

    def __init__(self, auth_url: str):
        super().__init__("Headless environment requires manual authorization.")
        self.auth_url = auth_url


def _save_state(state: str) -> None:
    AUTH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_STATE_FILE.write_text(json.dumps({"state": state}), encoding="utf-8")


def _load_state() -> str | None:
    try:
        data = json.loads(AUTH_STATE_FILE.read_text(encoding="utf-8"))
        return data.get("state")
    except Exception:
        return None


def start_headless_auth(creds_dir: Path = THIS_FOLDER, scopes: Iterable[str] = SCOPES) -> str:
    """Produce an authorization URL and save state for the follow-up token exchange."""
    creds_dir = Path(creds_dir)
    creds_file = creds_dir / "gmail_credentials.json"
    if not creds_file.exists():
        raise FileNotFoundError(
            f"Missing credentials file: {creds_file}. "
            "Download OAuth client secrets (JSON) from Google Cloud Console."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(creds_file),
        scopes,
    )
    flow.redirect_uri = HEADLESS_REDIRECT_URI
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent select_account",
        include_granted_scopes="true",
    )
    _save_state(state)
    return auth_url


def finish_headless_auth(
    returned_url: str,
    creds_dir: Path = THIS_FOLDER,
    scopes: Iterable[str] = SCOPES,
) -> Credentials:
    """
    Complete the OAuth flow given the full redirect URL returned by Google.
    """
    creds_dir = Path(creds_dir)
    creds_file = creds_dir / "gmail_credentials.json"
    token_file = creds_dir / "gmail_token.json"
    if not creds_file.exists():
        raise FileNotFoundError(
            f"Missing credentials file: {creds_file}. "
            "Download OAuth client secrets (JSON) from Google Cloud Console."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(creds_file),
        scopes,
    )
    flow.redirect_uri = HEADLESS_REDIRECT_URI
    saved_state = _load_state()
    if saved_state:
        # Ensure state matches the authorization response
        flow.oauth2session.state = saved_state

    parsed = urllib.parse.urlparse(returned_url)
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    flow.fetch_token(authorization_response=returned_url)
    creds = flow.credentials
    token_file.write_text(creds.to_json(), encoding="utf-8")
    if not record_auth_refresh("gmail"):
        logger.warning("Failed to record Gmail auth refresh timestamp.")

    # Best-effort cleanup of saved state
    try:
        AUTH_STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    return creds


def get_gmail_credentials(
    creds_dir: Path = THIS_FOLDER,
    scopes: Iterable[str] = SCOPES,
) -> Credentials:
    """
    Load Gmail OAuth credentials, refreshing or launching auth when needed.
    """
    creds_dir = Path(creds_dir)
    creds_dir.mkdir(parents=True, exist_ok=True)
    token_file = creds_dir / "gmail_token.json"
    creds_file = creds_dir / "gmail_credentials.json"

    creds: Credentials | None = None
    performed_manual_auth = False

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if not creds:
        if not creds_file.exists():
            raise FileNotFoundError(
                f"Missing credentials file: {creds_file}. "
                "Download OAuth client secrets (JSON) from Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
        try:
            # Attempt to launch browser automatically (best UX when available)
            creds = flow.run_local_server(port=0)
            performed_manual_auth = True
        except webbrowser.Error as exc:
            auth_url = start_headless_auth(creds_dir, scopes)
            logger.warning(
                f"Browser launch failed ({exc}); running headless auth. "
                "Copy the authorization URL into a browser from the UI."
            )
            raise HeadlessAuthRequired(auth_url)
    elif creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Automatic refreshes keep tokens valid but should not update manual auth status.
        performed_manual_auth = False

    if creds:
        token_file.write_text(creds.to_json(), encoding="utf-8")
        if performed_manual_auth and not record_auth_refresh("gmail"):
            logger.warning("Failed to record Gmail auth refresh timestamp.")
        return creds

    raise RuntimeError("Credentials not available after auth flow.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gmail OAuth helper.")
    parser.add_argument(
        "--creds-dir",
        type=Path,
        default=THIS_FOLDER,
        help="Directory containing gmail_credentials.json and gmail_token.json.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--start-headless",
        action="store_true",
        help="Print an authorization URL for manual/headless environments.",
    )
    group.add_argument(
        "--finish-headless",
        metavar="RETURN_URL",
        help="Complete headless auth using the redirected URL.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    creds_dir = Path(args.creds_dir)
    token_file = creds_dir / "gmail_token.json"

    try:
        if args.start_headless:
            auth_url = start_headless_auth(creds_dir, SCOPES)
            logger.info("Open this URL in a browser to authorize Gmail:\n{}", auth_url)
            return 0

        if args.finish_headless:
            finish_headless_auth(args.finish_headless, creds_dir, SCOPES)
            logger.info("Headless Gmail auth complete. Token saved to {}", token_file)
            return 0

        get_gmail_credentials(creds_dir, SCOPES)
        logger.info("Gmail credentials ready. Token stored at {}", token_file)
        return 0

    except HeadlessAuthRequired as exc:
        logger.warning(
            "Browser launch failed. Use --start-headless then --finish-headless. Auth URL: {}",
            exc.auth_url,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
    except Exception:
        logger.exception("Gmail auth flow failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
