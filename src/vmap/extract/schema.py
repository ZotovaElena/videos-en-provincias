"""Evidence record schema and JSONL storage.

One file per source under data/interim/evidence/<source>.jsonl; each stage rewrites only its own file,
so stages stay idempotent and independent. `load_all()` concatenates them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vmap import config

SourceType = Literal["wikidata", "wikipedia", "press", "film_commission", "manual"]
EVIDENCE_DIR = config.INTERIM / "evidence"


def evidence_id(source_url: str, evidence_text: str, extra: str = "") -> str:
    return hashlib.sha1(f"{source_url}|{evidence_text}|{extra}".encode()).hexdigest()


class Evidence(BaseModel):
    evidence_id: str
    source_type: SourceType
    source_url: str
    source_name: str
    retrieved_at: str
    lang: str | None = None
    song_title: str | None = None
    artist: str | None = None
    location_mentions: list[str] = Field(default_factory=list)
    partial: bool | None = None
    evidence_text: str
    extractor: str = "none"           # "none" = raw passage, not yet through an extractor
    confidence: float | None = None
    # source-specific context (optional)
    page_title: str | None = None
    page_kind: str | None = None       # song | artist | album | other (Wikipedia pages)
    section: str | None = None
    wikidata_qid: str | None = None
    location_qids: list[str] = Field(default_factory=list)
    coords: list[tuple[float, float]] = Field(default_factory=list)   # (lat, lon)
    youtube_ids: list[str] = Field(default_factory=list)
    query: str | None = None           # search query that found the source (press)
    # filled by the extraction stage: one dict per place mention (literal mention, field, gazetteer
    # candidates, trigger word, ambiguity flags, confidence)
    places: list[dict] = Field(default_factory=list)


def write(source: str, records: list[Evidence]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{source}.jsonl"
    seen, out = set(), []
    for r in records:
        if r.evidence_id not in seen:
            seen.add(r.evidence_id)
            out.append(r)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    tmp.replace(path)
    return path


def load(source: str) -> list[Evidence]:
    path = EVIDENCE_DIR / f"{source}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [Evidence.model_validate(json.loads(line)) for line in f if line.strip()]


def load_all() -> list[Evidence]:
    return [e for p in sorted(EVIDENCE_DIR.glob("*.jsonl")) for e in load(p.stem)]
