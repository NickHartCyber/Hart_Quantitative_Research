"""Minimal Party Time runtime to satisfy API endpoints."""
from __future__ import annotations

import time
from typing import Any


PARTY_TIME_ALGOS: dict[str, dict[str, str]] = {
    "party_time_momentum": {
        "label": "Momentum Breakout",
        "description": "Queues momentum breakout candidates and streams live updates.",
    },
    "party_time_swing": {
        "label": "Swing Trend",
        "description": "Longer-horizon swing candidates with trend confirmation.",
    },
}


class PartyTimeManager:
    """In-memory stub manager for party-time runs."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.algo: str | None = None
        self.status: str = "idle"
        self.started_at: float | None = None
        self.last_update: float | None = None
        self.last_error: str | None = None
        self.config: dict[str, Any] | None = None
        self.config_id: str | None = None
        self.rows: list[dict[str, Any]] = []

    async def start(
        self,
        algo: str,
        *,
        config: dict[str, Any] | None = None,
        config_id: str | None = None,
    ) -> dict[str, Any]:
        if algo not in PARTY_TIME_ALGOS:
            raise ValueError(f"Unknown party-time algo: {algo}")
        now = time.time()
        self.is_running = True
        self.status = "running"
        self.algo = algo
        self.config = config or {}
        self.config_id = config_id
        self.started_at = now
        self.last_update = now
        self.last_error = None
        self.rows = []
        return await self.snapshot()

    async def stop(self) -> dict[str, Any]:
        if self.is_running:
            self.is_running = False
            self.status = "stopped"
            self.last_update = time.time()
        return await self.snapshot()

    async def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "running": self.is_running,
            "algo": self.algo,
            "started_at": self.started_at,
            "last_update": self.last_update,
            "last_error": self.last_error,
            "config": self.config,
            "config_id": self.config_id,
            "count": len(self.rows),
            "rows": list(self.rows),
        }
