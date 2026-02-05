"""Configuration helpers for file-backed data paths."""
from __future__ import annotations

import os
from pathlib import Path


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "files" / "data"


DATA_DIR = Path(os.getenv("HART_QUANTITATIVE_RESEARCH_DATA_DIR", _default_data_dir())).expanduser().resolve()


def data_path(*parts: str) -> Path:
    """
    Return an absolute path under the configured data directory.
    Creates parent directories if needed.
    """
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
