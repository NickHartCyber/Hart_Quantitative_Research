"""Minimal backtest manager to satisfy API endpoints."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any


class BacktestManager:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    async def start(
        self,
        *,
        config_id: str,
        config_overrides: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        slippage_override: float | None = None,
        slippage_sweep: list[float] | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        payload = {
            "id": run_id,
            "status": "finished",
            "config_id": config_id,
            "config_overrides": config_overrides or {},
            "start_date": start_date,
            "end_date": end_date,
            "slippage_override": slippage_override,
            "slippage_sweep": slippage_sweep or [],
            "created_ts": time.time(),
            "finished_ts": time.time(),
            "result": {"message": "Backtest stub completed."},
        }
        self._current = payload
        self._history.insert(0, payload)
        return payload

    async def snapshot(self) -> dict[str, Any]:
        return self._current or {"status": "idle"}

    def list_history(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return self._history[: max(1, int(limit))]

    def load_run(
        self,
        run_id: str,
        *,
        include_trades: bool = False,
        trades_limit: int = 200,
        trades_offset: int = 0,
    ) -> dict[str, Any]:
        for rec in self._history:
            if rec.get("id") == run_id:
                if include_trades:
                    rec = dict(rec)
                    rec["trades"] = []
                    rec["trades_limit"] = trades_limit
                    rec["trades_offset"] = trades_offset
                return rec
        raise FileNotFoundError(f"Backtest run '{run_id}' not found.")
