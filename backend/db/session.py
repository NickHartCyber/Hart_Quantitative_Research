from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base

_ENGINE = None
_SESSION_FACTORY: sessionmaker | None = None


def _database_url() -> Optional[str]:
    url = os.getenv("HART_QUANTITATIVE_RESEARCH_DATABASE_URL", "").strip()
    return url or None


def is_db_enabled() -> bool:
    return _database_url() is not None


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    url = _database_url()
    if not url:
        return None
    echo = os.getenv("HART_QUANTITATIVE_RESEARCH_DB_ECHO", "").strip().lower() in {"1", "true", "yes"}
    _ENGINE = create_engine(url, echo=echo, future=True)
    return _ENGINE


def init_db(engine=None) -> None:
    engine = engine or get_engine()
    if engine is None:
        raise RuntimeError("Database URL not configured (set HART_QUANTITATIVE_RESEARCH_DATABASE_URL).")
    Base.metadata.create_all(engine)


def get_session() -> Session:
    global _SESSION_FACTORY
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database URL not configured (set HART_QUANTITATIVE_RESEARCH_DATABASE_URL).")
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return _SESSION_FACTORY()
