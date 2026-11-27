"""Hand-added evidence: data/manual/manual_evidence.csv → evidence/manual.jsonl.

Required per row: source_url, snippet (literal text from the source), location, and song_title or
youtube_id. Several locations can be separated with ";". Each location must appear literally in the
snippet (after whitespace normalisation); rows that fail are reported and skipped.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from vmap.extract.schema import Evidence, evidence_id
from vmap.sources.seed import MANUAL_CSV


def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def load_rows(path=MANUAL_CSV) -> tuple[list[Evidence], list[str]]:
    if not path.exists():
        return [], []
    df = pd.read_csv(path, dtype=str).fillna("")
    out, errors = [], []
    for i, r in df.iterrows():
        line = i + 2  # header is line 1
        snippet, url = _ws(r.get("snippet", "")), r.get("source_url", "").strip()
        locs = [_ws(x) for x in r.get("location", "").split(";") if _ws(x)]
        if not url.startswith("http"):
            errors.append(f"line {line}: source_url missing or not a URL")
            continue
        if not snippet:
            errors.append(f"line {line}: snippet missing")
            continue
        if not locs:
            errors.append(f"line {line}: location missing")
            continue
        if not (r.get("song_title") or r.get("youtube_id")):
            errors.append(f"line {line}: song_title or youtube_id required")
            continue
        missing = [loc for loc in locs if loc not in snippet]
        if missing:
            errors.append(f"line {line}: location(s) not found literally in snippet: {missing}")
            continue
        partial = r.get("partial", "").strip().lower() in ("1", "true", "yes", "si", "sí", "x")
        out.append(Evidence(
            evidence_id=evidence_id(url, snippet, r.get("youtube_id", "")), source_type="manual",
            source_url=url, source_name=re.sub(r"^https?://(www\.)?([^/]+).*", r"\2", url),
            retrieved_at=r.get("retrieved_at") or date.today().isoformat(),
            song_title=r.get("song_title") or None, artist=r.get("artist") or None,
            location_mentions=locs, partial=partial or len(locs) > 1, evidence_text=snippet,
            extractor="manual", youtube_ids=[r["youtube_id"]] if r.get("youtube_id") else [],
        ))
    return out, errors
