"""Phase 5: extracted records → clips + places → status per (clip, unit).

Outputs (data/interim/):
  location_evidence.csv  one row per (record, place mention, unit): the detailed trail behind everything
  clip_locations.csv     one row per (clip, unit): n_sources, domains, reference press, partial, shared
                         area, status (confirmed | probable | rejected) and the reasons when not confirmed
  clips.csv              one row per clip: artist, song, linked YouTube candidates
  review_queue.csv       places / links needing a human decision, and near-duplicate clips (never merged)

Status of a (record, place): `confirmed` when the place resolves to a unit with no doubt, the extractor is
confident (≥ CONFIDENT), and the record is linked to a clip; `rejected` when the text says it did not happen
there or the place is outside Spain; `probable` otherwise. A (clip, unit) is confirmed if any record confirms it.
"""
from __future__ import annotations

import json
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import yaml

from vmap import config
from vmap.extract import rules
from vmap.resolve import clips as cl
from vmap.resolve.geo import GeoResolver
from vmap.resolve.link import STRONG, Linker, near_duplicates

OUT = config.INTERIM
CONFIDENT = 0.7
REJECT = {"not_actual", "foreign_place", "outside_spain"}


def domain(url: str) -> str:
    d = urlsplit(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def reference_domains() -> list[str]:
    with open(config.CONFIG / "reference_press.yaml", encoding="utf-8") as f:
        return [d.lower() for d in yaml.safe_load(f)["domains"]]


def is_reference(d: str, refs: list[str]) -> bool:
    return any(d == r or d.endswith("." + r) for r in refs)


def query_ine() -> dict[str, str]:
    path = OUT / "press_results.jsonl"
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["query"]] = r["ine"]
    return out


def context_provinces(ev, q2ine: dict) -> set[str]:
    """Provinces the record points to on its own: unambiguous one-province places + the press query town."""
    ctx = set()
    for p in ev.places:
        provs = {c for x in p.get("candidates", []) for c in x["provinces"]}
        if len(provs) == 1 and not p.get("ambiguous"):
            ctx |= provs
    if ev.query and ev.query in q2ine:
        ctx.add(str(q2ine[ev.query]).zfill(5)[:2])
    return ctx


def record_rows(ev, ids, method, clip, geo: GeoResolver, ctx: set[str]) -> list[dict]:
    rows = []
    for p in ev.places:
        res = geo.resolve(p, ctx)
        reasons = list(p.get("ambiguous") or []) + res["reasons"]
        if (p.get("confidence") or 0) < CONFIDENT:
            reasons.append("low_confidence")
        if method == "unlinked":
            reasons.append("no_clip")
        elif method not in STRONG and method != "text_only":
            reasons.append(f"link_{method}")
        reasons = list(dict.fromkeys(reasons))
        status = "rejected" if REJECT & set(reasons) else ("probable" if reasons else "confirmed")
        for unit in res["unit_ids"] or [None]:
            rows.append({
                "evidence_id": ev.evidence_id, "source_type": ev.source_type, "source_url": ev.source_url,
                "domain": domain(ev.source_url), "clip_id": clip["clip_id"] if clip else None,
                "artist": clip["artist"] if clip else ev.artist, "song_title": clip["song_title"] if clip else ev.song_title,
                "link_method": method, "youtube_ids": " ".join(ids), "location_mention": p["mention"],
                "field": p.get("field"), "unit_id": unit, "ine": res.get("ine"), "lat": res.get("lat"),
                "lon": res.get("lon"), "precision": res.get("precision"), "shared_area": res["shared_area"],
                "geocode_name": res.get("geocode_name"), "partial": bool(ev.partial),
                "confidence": p.get("confidence"), "status": status, "reasons": " ".join(reasons),
                "evidence_text": ev.evidence_text[:500],
            })
    return rows


def aggregate(ev_rows: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    rank = {"confirmed": 0, "probable": 1, "rejected": 2}
    rows = []
    for (clip, unit), g in ev_rows.dropna(subset=["clip_id", "unit_id"]).groupby(["clip_id", "unit_id"]):
        best = min(g["status"], key=rank.get)
        kept = g[g["status"] == best]
        point = g.dropna(subset=["lat"]).sort_values("precision", key=lambda s: s != "municipality")
        doms = sorted(set(kept["domain"]))
        rows.append({
            "clip_id": clip, "unit_id": unit, "artist": g["artist"].iloc[0], "song_title": g["song_title"].iloc[0],
            "location_mention": "; ".join(dict.fromkeys(kept["location_mention"])),
            "lat": point["lat"].iloc[0] if len(point) else None, "lon": point["lon"].iloc[0] if len(point) else None,
            "ine_municipality": next((i for i in kept["ine"] if isinstance(i, str)), None),
            "n_sources": kept["source_url"].nunique(), "n_independent_domains": len(doms),
            "has_reference_press": any(is_reference(d, refs) for d in doms), "domains": " ".join(doms),
            "partial": bool(kept["partial"].any()), "shared_area": bool(kept["shared_area"].any()),
            "status": best,
            "reasons": "" if best == "confirmed" else " ".join(sorted({r for x in kept["reasons"] for r in x.split()})),
            "evidence_ids": " ".join(dict.fromkeys(kept["evidence_id"])),
        })
    return pd.DataFrame(rows)


def clip_table(ev_rows: pd.DataFrame, linker: Linker) -> pd.DataFrame:
    rows = []
    for clip, g in ev_rows.dropna(subset=["clip_id"]).groupby("clip_id"):
        ids = sorted({y for s in g["youtube_ids"] for y in s.split() if y})
        top = linker.top.loc[ids].sort_values("view_count", ascending=False) if ids else None
        rows.append({
            "clip_id": clip, "artist": g["artist"].iloc[0], "song_title": g["song_title"].iloc[0],
            "youtube_ids": " ".join(ids), "main_youtube_id": top.index[0] if ids else None,
            "youtube_title": top["title"].iloc[0] if ids else None,
            "channel_title": top["channel_title"].iloc[0] if ids else None,
            "view_count": int(top["view_count"].iloc[0]) if ids else None,
            "views_snapshot_date": top["views_snapshot_date"].iloc[0] if ids else None,
            "link_methods": " ".join(sorted(set(g["link_method"]))), "n_records": g["evidence_id"].nunique(),
        })
    return pd.DataFrame(rows)


def run(log=print) -> dict:
    from vmap.sources import seed

    records = rules.load()
    linker = Linker(pd.read_csv(seed.TOP_CSV, dtype={"youtube_id": str}))
    geo, q2ine, refs = GeoResolver(), query_ine(), reference_domains()
    rows, methods = [], Counter()
    for i, ev in enumerate(records, 1):
        ids, method = linker.link(ev)
        methods[method] += 1
        clip = linker.clip_for(ev, ids) if method != "unlinked" else None
        rows += record_rows(ev, ids, method, clip, geo, context_provinces(ev, q2ine))
        if i % 200 == 0:
            log(f"  {i}/{len(records)} records")
    ev_rows = pd.DataFrame(rows)
    locs = aggregate(ev_rows, refs)
    clips = clip_table(ev_rows, linker)
    dups = near_duplicates(clips) if len(clips) else pd.DataFrame()

    review = ev_rows[ev_rows["status"] == "probable"][
        ["evidence_id", "source_url", "artist", "song_title", "location_mention", "unit_id", "reasons",
         "evidence_text"]].assign(type="place_or_link")
    review = pd.concat([review, dups.assign(type="clip_merge")], ignore_index=True)

    ev_rows.to_csv(OUT / "location_evidence.csv", index=False)
    locs.to_csv(OUT / "clip_locations.csv", index=False)
    clips.to_csv(OUT / "clips.csv", index=False)
    review.to_csv(OUT / "review_queue.csv", index=False)
    reasons = Counter(r for x in ev_rows["reasons"] for r in x.split())
    return {"records": len(records), "link_methods": dict(methods), "place_rows": len(ev_rows),
            "row_status": ev_rows["status"].value_counts().to_dict(), "reasons": dict(reasons.most_common()),
            "clips": len(clips), "clips_with_youtube": int(clips["main_youtube_id"].notna().sum()) if len(clips) else 0,
            "clip_units": len(locs), "clip_unit_status": locs["status"].value_counts().to_dict() if len(locs) else {},
            "units_with_confirmed": int(locs.loc[locs["status"] == "confirmed", "unit_id"].nunique()) if len(locs) else 0,
            "near_duplicates": len(dups), "geocode_errors": geo.geocode_errors}
