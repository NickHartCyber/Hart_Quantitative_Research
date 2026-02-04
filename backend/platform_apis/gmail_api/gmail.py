"""
Gmail helper for reading subjects and extracting symbols.

Features
--------
- Connects to Gmail API using OAuth installed-app flow.
- Stores/refreshes credentials in a local directory (`THIS_FOLDER`).
- Lists messages (paginated), reads Subject headers, and (optionally) trashes them.
- Date-aware fetching: keep today's emails for reprocessing; auto-trash older ones.
- Extracts ticker symbols from subject lines using a simple pattern heuristic.
- Parses Thinkorswim-style option tickers (utility).

Security
--------
- Be cautious with `trash_on_read=True`: use a Gmail query to limit scope (e.g., label).
- Token files contain credentials; ensure the directory is not committed to VCS.

Setup
-----
- Place `gmail_credentials.json` (OAuth client secrets) in `THIS_FOLDER`.
- First run opens a browser to authorize; subsequent runs use `gmail_token.json`.

Reference
---------
- Gmail API Python Quickstart: https://developers.google.com/gmail/api/quickstart/python
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from google.oauth2.credentials import Credentials
from loguru import logger
from backend.platform_apis.gmail_api.gmail_auth import (
    SCOPES,
    THIS_FOLDER,
    HeadlessAuthRequired,
    get_gmail_credentials,
)


class Gmail:
    """
    Minimal Gmail API client for reading subjects and extracting symbols.
    """

    def __init__(
        self,
        creds_dir: Path = THIS_FOLDER,
        scopes: Iterable[str] = SCOPES,
    ) -> None:
        """
        Parameters
        ----------
        creds_dir : Path
            Directory where `gmail_credentials.json` and `gmail_token.json` live.
        scopes : Iterable[str]
            OAuth scopes for the Gmail API.
        """
        self.creds_dir = Path(creds_dir)
        self.scopes = list(scopes)

        self.creds: Credentials | None = None
        self.service = None  # Gmail Resource object

    # --------------------------------------------------------------------- #
    # Auth / Connection
    # --------------------------------------------------------------------- #
    def connect(self) -> bool:
        """
        Authenticate and build the Gmail API service.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        try:
            logger.info("Connecting to Gmail...")
            self.creds = get_gmail_credentials(self.creds_dir, self.scopes)
            self.service = build("gmail", "v1", credentials=self.creds)
            logger.info("Connected to Gmail.")
            return True

        except HeadlessAuthRequired:
            # Allow caller to handle manual auth flow.
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Gmail: {e}")
            return False

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def parse_option_symbol(symbol: str) -> tuple[str, str, datetime, str]:
        """
        Parse a Thinkorswim-style option symbol.

        Examples
        --------
        ".AA201211C5.5" -> ("AA", "AA_121120C5.5", datetime(2020, 12, 11), "CALL")

        Parameters
        ----------
        symbol : str
            Raw option symbol text.

        Returns
        -------
        tuple
            (root, normalized, expiry_date, option_type)

        Notes
        -----
        Heuristic parser; adjust for your broker’s exact format.
        """
        s = symbol.replace(".", "", 1).strip()
        ending_index = 0
        int_found = False

        for idx, ch in enumerate(s):
            if ch.isdigit():
                int_found = True
            elif not int_found:
                ending_index = idx

        exp = s[ending_index + 1 :]
        if len(exp) < 7:
            # Not enough info for YYMMDD+type+strike
            raise ValueError(f"Cannot parse option symbol: {symbol}")

        year = exp[:2]
        month = exp[2:4]
        day = exp[4:6]
        option_type = "CALL" if "C" in exp else "PUT"

        pre_symbol = f"{s[:ending_index + 1]}_{month}{day}{year}{exp[6:]}"
        return (
            s[: ending_index + 1],
            pre_symbol,
            datetime.strptime(f"{year}-{month}-{day}", "%y-%m-%d"),
            option_type,
        )

    @staticmethod
    def _subject_from_message(msg: dict) -> str | None:
        """
        Extract the Subject header from a Gmail message resource.

        Parameters
        ----------
        msg : dict
            Gmail message resource (format='metadata' with headers).

        Returns
        -------
        str | None
            Subject text if present.
        """
        try:
            headers = msg.get("payload", {}).get("headers", [])
            for h in headers:
                if h.get("name") == "Subject":
                    return (h.get("value") or "").strip()
        except Exception:
            pass
        return None

    def _modify_labels(self, msg_id: str, remove: list[str] | None = None, add: list[str] | None = None) -> dict:
        """Wrapper around users.messages.modify to add/remove labels safely."""
        remove = remove or []
        add = add or []
        return (
            self.service.users()
            .messages()
            .modify(userId="me", id=msg_id, body={"removeLabelIds": remove, "addLabelIds": add})
            .execute()
        )

    @staticmethod
    def _message_date(msg: dict) -> date | None:
        """Return the local date of the message using Gmail's internalDate (ms since epoch)."""
        try:
            internal_ms = int(msg.get("internalDate"))
            return datetime.fromtimestamp(internal_ms / 1000).date()
        except Exception:
            return None

    # --------------------------------------------------------------------- #
    # Symbol extraction
    # --------------------------------------------------------------------- #
    @staticmethod
    def extract_symbols_from_subjects(subjects: Iterable[str]) -> list[str]:
        """
        Extract ticker symbols from a collection of subject lines.

        Heuristic logic based on expected phrases like
        `"SYMBOLS: AAA, BBB were added to ..."`. Adjust for your exact email format.

        Parameters
        ----------
        subjects : Iterable[str]
            Subject strings.

        Returns
        -------
        list[str]
            Unique, whitespace-trimmed symbols in order of appearance.
        """
        symbols: list[str] = []
        for subj in subjects:
            if not subj:
                continue
            try:
                parts = subj.split(":")
                if len(parts) <= 2:
                    continue

                tail = parts[2]
                markers = ["were added to", "was added to"]
                for marker in markers:
                    if marker in tail:
                        head = tail.split(marker)[0]
                        for sym in head.split(","):
                            s = (sym or "").strip()
                            # logger.debug(f"gmail extract_symbols_from_subjects s: {s}")
                            if s and (s not in symbols):
                                # Check to make sure no / in symbol str
                                if "/" not in s:
                                    symbols.append(s)
                        break
                logger.info(f"New Email Subject: {subj}")
            except Exception as e:
                logger.warning(f"{Gmail.__name__} - subject parse issue: {e} | subj={subj!r}")
        return symbols

    # --------------------------------------------------------------------- #
    # Gmail operations
    # --------------------------------------------------------------------- #
    def fetch_subjects(
        self,
        query: str = "in:inbox",  # process all inbox mail regardless of read state
        max_messages: int = 200,
        trash_on_read: bool = False,
        mark_read_on_read: bool = False,  # keep state unchanged so restarts still process messages
    ) -> list[str]:
        """
        Fetch message subjects (optionally filter with a Gmail search query) with
        date-aware handling:
        - Collect symbols only from messages dated today.
        - Move messages from prior days to Trash.

        Parameters
        ----------
        query : str, default "in:inbox"
            Gmail search query. Avoid `is:unread` filters so restarts still see messages.
        max_messages : int, default 200
            Safety cap to avoid reading/trashing an entire inbox inadvertently.
        trash_on_read : bool, default False
            Kept for backward compatibility; today's messages are never trashed.
        mark_read_on_read : bool, default False
            Optional; mark today's messages as read after processing.

        Returns
        -------
        list[str]
            Subject lines found.
        """
        if not self.service:
            raise RuntimeError("Not connected. Call connect() first.")

        # Parameter kept for API compatibility; today's messages are never trashed.
        _ = trash_on_read

        subjects: list[str] = []
        try:
            user_id = "me"
            next_page_token: str | None = None
            today = datetime.now().date()

            while True:
                result = (
                    self.service.users()
                    .messages()
                    .list(userId=user_id, q=query, pageToken=next_page_token, maxResults=min(100, max_messages))
                    .execute()
                )
                next_page_token = result.get("nextPageToken")
                msgs = result.get("messages", [])
                if not msgs:
                    break

                for m in msgs:
                    if len(subjects) >= max_messages:
                        break

                    msg = (
                        self.service.users()
                        .messages()
                        .get(userId=user_id, id=m["id"], format="metadata", metadataHeaders=["Subject"])
                        .execute()
                    )
                    msg_date = self._message_date(msg)
                    if msg_date and msg_date < today:
                        # Delete older messages to keep inbox clean.
                        try:
                            self._modify_labels(m["id"], remove=["INBOX", "UNREAD"], add=["TRASH"])
                        except Exception as e:
                            logger.error(f"Failed to trash old message {m['id']}: {e}")
                        continue

                    subj = self._subject_from_message(msg)
                    if msg_date and msg_date == today and subj:
                        subjects.append(subj)

                    # Mark as read? (optional; default False so restarts still see the email)
                    if msg_date and msg_date == today and mark_read_on_read:
                        try:
                            self._modify_labels(m["id"], remove=["UNREAD"])
                        except Exception as e:
                            logger.warning(f"Failed to mark read for {m['id']}: {e}")

                    # Never delete today's messages so they can be reprocessed on restart.

                if len(subjects) >= max_messages or not next_page_token:
                    break

        except HttpError as he:
            logger.error(f"Gmail API error: {he}")
        except Exception as e:
            logger.error(f"{Gmail.__name__} - unexpected error: {e}")

        return subjects

    def get_email_symbols(
        self,
        query: str = "in:inbox",
        max_messages: int = 200,
        trash_on_read: bool = False,  # ignored for today's messages; older ones are always trashed
        mark_read_on_read: bool = False,
    ) -> list[str]:
        """
        Convenience method: fetch today's subjects and extract symbols while
        auto-trashing messages from prior days.

        Parameters
        ----------
        query : str, default "in:inbox"
            Gmail query to limit which messages are read.
        max_messages : int, default 200
            Maximum messages to process.
        trash_on_read : bool, default False
            Kept for backward compatibility; today's messages are preserved, older ones are auto-trashed.
        mark_read_on_read : bool, default False
            Optional; mark today's messages as read after processing.

        Returns
        -------
        list[str]
            Unique symbols extracted from subjects.
        """
        subjects = self.fetch_subjects(
            query=query,
            max_messages=max_messages,
            trash_on_read=trash_on_read,
            mark_read_on_read=mark_read_on_read,
        )
        return self.extract_symbols_from_subjects(subjects)

if __name__ == "__main__":
    gmail = Gmail()
    if gmail.connect():
        # Example: only read recent messages with a specific label or phrase
        query = "in:inbox"
        emails = gmail.get_email_symbols(query=query, max_messages=200, trash_on_read=False)
        logger.debug(f"symbols: {emails}")
