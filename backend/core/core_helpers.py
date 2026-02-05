"""Small shared helpers used by auth/token workflows."""
from __future__ import annotations

import base64


def basic_auth_header(client_id: str, client_secret: str) -> str:
    """
    Build a Basic authorization header value for OAuth token requests.

    Returns
    -------
    str
        Header value like "Basic <base64(client_id:client_secret)>".
    """
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"Basic {encoded}"
