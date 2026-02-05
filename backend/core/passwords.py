"""Password hashing utilities for lightweight auth."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os


_ALGO = "pbkdf2_sha256"
_DEFAULT_ITERATIONS = 200_000


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(raw: str) -> bytes:
    padded = raw + "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def hash_password(password: str, *, iterations: int = _DEFAULT_ITERATIONS, salt_bytes: int = 16) -> str:
    if not password:
        raise ValueError("Password is required.")
    salt = os.urandom(salt_bytes)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGO}${iterations}${_b64encode(salt)}${_b64encode(dk)}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not password or not stored_hash:
        return False
    try:
        algo, iter_str, salt_b64, hash_b64 = stored_hash.split("$", 3)
        if algo != _ALGO:
            return False
        iterations = int(iter_str)
        salt = _b64decode(salt_b64)
        expected = _b64decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)
