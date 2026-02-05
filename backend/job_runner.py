"""In-memory job manager for lightweight task execution."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Awaitable, Callable


@dataclass
class JobRecord:
    id: str
    job: str
    status: str
    created_ts: float
    started_ts: float | None = None
    finished_ts: float | None = None
    result: dict | None = None
    error: str | None = None
    params: dict = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["params"] = self.params or {}
        return data


JobFunc = Callable[[dict], dict | Awaitable[dict]]


class InMemoryJobManager:
    def __init__(self, registry: dict[str, JobFunc] | None = None) -> None:
        self._registry = registry or {}
        self._jobs: dict[str, JobRecord] = {}

    async def submit(self, job: str, params: dict | None = None) -> JobRecord:
        record = JobRecord(
            id=str(uuid.uuid4()),
            job=job,
            status="queued",
            created_ts=time.time(),
            params=params or {},
        )
        self._jobs[record.id] = record
        await self._run_job(record)
        return record

    async def list(self) -> list[dict]:
        return [rec.to_dict() for rec in self._jobs.values()]

    async def get(self, job_id: str) -> dict | None:
        rec = self._jobs.get(job_id)
        return rec.to_dict() if rec else None

    async def cancel(self, job_id: str) -> dict | None:
        rec = self._jobs.get(job_id)
        if not rec:
            return None
        if rec.status in {"finished", "error"}:
            return rec.to_dict()
        rec.status = "cancelled"
        rec.finished_ts = time.time()
        return rec.to_dict()

    async def _run_job(self, record: JobRecord) -> None:
        record.status = "running"
        record.started_ts = time.time()
        func = self._registry.get(record.job)
        if func is None:
            record.status = "finished"
            record.result = {"message": "Job queued (no-op)", "job": record.job}
            record.finished_ts = time.time()
            return
        try:
            result = func(record.params)
            if asyncio.iscoroutine(result):
                result = await result
            record.result = result if isinstance(result, dict) else {"result": result}
            record.status = "finished"
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
        record.finished_ts = time.time()


def build_default_manager() -> InMemoryJobManager:
    return InMemoryJobManager()
