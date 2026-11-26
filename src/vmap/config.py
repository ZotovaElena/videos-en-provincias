"""Project paths, settings and secrets."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("VMAP_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANUAL = DATA / "manual"
RUN_LOG = PROCESSED / "run_log.jsonl"

load_dotenv(ROOT / ".env")


@lru_cache
def settings() -> dict:
    with open(CONFIG / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def secret(name: str, required: bool = True) -> str | None:
    value = os.environ.get(name, "").strip() or None
    if required and not value:
        raise RuntimeError(f"{name} is not set. Add it to {ROOT / '.env'} (see .env.example).")
    return value
