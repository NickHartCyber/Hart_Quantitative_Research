from __future__ import annotations

from threading import Event
from typing import Optional


class SuggesterCancelled(RuntimeError):
    """Raised when a running suggester is stopped by the user."""


class CancelToken:
    def __init__(self, reason: Optional[str] = None) -> None:
        self._event = Event()
        self._reason = reason or "Suggester run cancelled."

    def cancel(self, reason: Optional[str] = None) -> None:
        if reason:
            self._reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self, *, message: Optional[str] = None) -> None:
        if self._event.is_set():
            raise SuggesterCancelled(message or self._reason)

    @property
    def reason(self) -> str:
        return self._reason
