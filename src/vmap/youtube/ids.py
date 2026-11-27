"""Phase 6: YouTube video for every clip with a place (confirmed or probable), plus fresh statistics.

ID sources, in order:
  seed     the clip is already linked to official-channel videos from the top-down stage (Phase 5)
  serper   Google results for "{artist} {song} site:youtube.com" (Serper, shares the press search budget):
           every YouTube link found is a candidate
  search   YouTube search.list (own small daily budget) for clips Serper did not resolve
Every candidate is checked with the official API (videos.list): the official-video filter of the seed
stage (no Topic channel, no lyric/live/visualizer/..., 1–10 min, music categories, not made for kids;
no minimum views), the song title in the video title, and the artist in the channel name or the title.
The candidate whose channel matches the artist wins; then the most viewed.

Statistics: one `videos.list` refresh of all chosen videos at the end, so the map has one snapshot date
(YouTube policy: stored statistics older than 29 days must be refreshed before use).

Output data/interim/youtube_videos.csv: one row per clip, with `youtube_url` (watch link used by the map
and the explorer) and the raw `view_count` as returned by the API.
"""
from __future__ import annotations

import json
import re

import pandas as pd
from rapidfuzz import fuzz

from vmap import config
from vmap.http import Client, HttpError
from vmap.resolve import clips as cl
from vmap.sources import press, seed
from vmap.youtube.api import QuotaExceeded, YouTube

VIDEOS_CSV = config.INTERIM / "youtube_videos.csv"
NOT_FOUND_CSV = config.INTERIM / "youtube_not_found.csv"
CANDIDATES_JSONL = config.INTERIM / "youtube_candidates.jsonl"
YT_ID = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^#\s]*&)?v=|embed/|v/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})")
CHANNEL_NOISE = re.compile(r"(?i)\s*(vevo|official|oficial|music|tv|channel|canal)\s*$")


def watch_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def ids_from_urls(urls: list[str]) -> list[str]:
    """YouTube video IDs in result links, in rank order (Shorts and channel pages ignored)."""
    out = []
    for u in urls:
        if "/shorts/" in u:
            continue
        m = YT_ID.search(u)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _channel_name(title: str) -> str:
    t = title or ""
    for _ in range(2):
        t = CHANNEL_NOISE.sub("", t)
    return t


def clean_artist(artist: str) -> str:
    """'Gisela (cantante)' → 'Gisela' (Wikipedia disambiguation in names taken from page titles)."""
    return re.sub(r"\s*\([^)]*\)", "", artist or "").strip()


def _squash(s: str) -> str:
    return cl.norm(s).replace(" ", "")


def channel_is_artist(artist: str, channel_title: str) -> bool:
    """Strict: the channel name is the artist's name (after removing VEVO / Official / TV ...).
    'KelisVEVO', 'GoldfrappTV', 'Sergio Contreras' match; 'THALIASCHILE', 'ESC Stuff' do not."""
    chan = _squash(_channel_name(channel_title))
    return bool(chan) and any(fuzz.ratio(_squash(a), chan) >= 90 for a in cl.split_artists(artist))


def check(row: dict, artist: str, song: str, settings: dict) -> tuple[bool, bool, str]:
    """(acceptable, channel_matches_artist, reason) for one candidate video."""
    reason = seed.exclusion_reason(row, {**settings, "min_views": 0})
    if reason:
        return False, False, reason
    artist = clean_artist(artist)
    title = cl.norm_title(row["title"])                  # "ft. Tomatito" removed before matching the song
    s = cl.norm_title(song)
    if not s or not (s in title or fuzz.partial_ratio(s, title) >= 90):
        return False, False, "song_not_in_title"
    if channel_is_artist(artist, row["channel_title"]):
        return True, True, ""
    # a collaborator's own channel: ROSALÍA's channel for "ROSALÍA & Travis Scott - TKN"
    chan = _squash(_channel_name(row["channel_title"]))
    if cl.artist_match(artist, row["title"]) and len(chan) >= 3 and chan in _squash(row["title"]):
        return True, False, ""
    return False, False, "not_artist_channel"


def choose(rows: list[dict], artist: str, song: str, settings: dict) -> tuple[dict | None, list[dict]]:
    checked = []
    for r in rows:
        ok, chan, reason = check(r, artist, song, settings)
        checked.append({**r, "ok": ok, "channel_match": chan, "reject_reason": reason})
    good = [r for r in checked if r["ok"]]
    if not good:
        return None, checked
    best = max(good, key=lambda r: (r["channel_match"], r["view_count"] or 0))
    return best, checked


def clips_to_resolve() -> pd.DataFrame:
    """Clips with a non-rejected place, in priority order: confirmed clips in units with no video yet,
    other confirmed clips, then probable ones."""
    locs = pd.read_csv(config.INTERIM / "clip_locations.csv", dtype={"unit_id": str})
    clips = pd.read_csv(config.INTERIM / "clips.csv", dtype={"main_youtube_id": str})
    locs = locs[locs["status"] != "rejected"].merge(clips[["clip_id", "main_youtube_id"]], on="clip_id")
    covered = set(locs.loc[(locs["status"] == "confirmed") & locs["main_youtube_id"].notna(), "unit_id"])
    pri = locs.assign(p=locs.apply(lambda r: 0 if r["status"] == "confirmed" and r["unit_id"] not in covered
                                   else (1 if r["status"] == "confirmed" else 2), axis=1))
    pri = pri.groupby("clip_id")["p"].min()
    out = clips[clips["clip_id"].isin(pri.index)].assign(priority=lambda d: d["clip_id"].map(pri))
    return out.sort_values(["priority", "clip_id"]).reset_index(drop=True)


def run(max_queries: int | None = None, youtube_search: bool = True, refresh: bool = True, log=print) -> dict:
    settings = config.settings()["youtube"]
    pcfg = config.search()["press"]
    client = Client(stage="youtube_ids")
    yt = YouTube(client=client)
    serper = press.make_provider(client, pcfg)
    budget = press.DailyBudget(min(max_queries if max_queries is not None else pcfg["max_queries_per_day"],
                                   pcfg["max_queries_per_day"]), pcfg.get("max_queries_total"))
    todo = clips_to_resolve()
    chosen: dict[str, dict] = {}
    not_found, audit, stats = [], [], {"seed": 0, "serper": 0, "search": 0, "not_found": 0, "skipped": 0}
    stopped = None

    def verify(ids: list[str], artist: str, song: str):
        return choose(seed.video_stats(yt, ids, log=log), artist, song, settings)

    for c in todo.itertuples(index=False):
        artist, song = c.artist, c.song_title
        if isinstance(c.main_youtube_id, str) and c.main_youtube_id:
            chosen[c.clip_id] = {"youtube_id": c.main_youtube_id, "id_method": "seed",
                                 "candidate_ids": c.youtube_ids}
            stats["seed"] += 1
            continue
        if not isinstance(artist, str) or not isinstance(song, str):
            stats["skipped"] += 1
            continue
        found = None
        # 1. Serper (Google) results restricted to youtube.com
        q = f"{artist} {song} site:youtube.com"
        if stopped is None:
            try:
                if not serper.is_cached(q):
                    if budget.remaining() <= 0:
                        stopped = "serper budget"
                    else:
                        budget.spend()
                if stopped is None:
                    ids = ids_from_urls([h["url"] for h in serper.search(q)])[:10]
                    if ids:
                        best, checked = verify(ids, artist, song)
                        audit.append({"clip_id": c.clip_id, "method": "serper", "query": q, "candidates": checked})
                        if best:
                            found = {"youtube_id": best["youtube_id"], "id_method": "serper"}
            except (HttpError, QuotaExceeded) as e:
                log(f"  serper/verify stopped: {e}")
                stopped = "error"
        # 2. YouTube search.list (own daily budget) when Serper found nothing
        if not found and youtube_search:
            try:
                d, _ = yt.call("search", {"part": "snippet", "q": f"{artist} {song}", "type": "video",
                                          "maxResults": 10, "regionCode": "ES"})
                ids = [i["id"]["videoId"] for i in d.get("items", []) if i.get("id", {}).get("videoId")]
                if ids:
                    best, checked = verify(ids, artist, song)
                    audit.append({"clip_id": c.clip_id, "method": "search", "query": f"{artist} {song}",
                                  "candidates": checked})
                    if best:
                        found = {"youtube_id": best["youtube_id"], "id_method": "search"}
            except QuotaExceeded:
                youtube_search = False
                log("  YouTube search budget used up for today; continuing with Serper only")
        if found:
            chosen[c.clip_id] = found
            stats[found["id_method"]] += 1
        else:
            # best rejected candidate (e.g. an unofficial upload) to help a manual decision
            cand = next((x for a in reversed(audit) if a["clip_id"] == c.clip_id
                         for x in sorted(a["candidates"], key=lambda r: -(r.get("view_count") or 0))
                         if x.get("reject_reason") == "not_artist_channel"), None)
            not_found.append({"clip_id": c.clip_id, "artist": artist, "song_title": song, "priority": c.priority,
                              "searched": stopped is None,
                              "candidate_url": watch_url(cand["youtube_id"]) if cand else None,
                              "candidate_title": cand["title"] if cand else None,
                              "candidate_channel": cand["channel_title"] if cand else None})
            stats["not_found"] += 1

    # 3. one refresh of all chosen videos → one snapshot date
    ids = sorted({v["youtube_id"] for v in chosen.values()})
    fresh = {r["youtube_id"]: r for r in seed.video_stats(yt, ids, log=log, refresh=refresh)}
    rows = []
    meta = todo.set_index("clip_id")
    for clip_id, v in chosen.items():
        r = fresh.get(v["youtube_id"])
        if r is None:                                   # video deleted / private since the seed run
            not_found.append({"clip_id": clip_id, "artist": meta.at[clip_id, "artist"],
                              "song_title": meta.at[clip_id, "song_title"], "priority": meta.at[clip_id, "priority"],
                              "searched": True, "note": "video unavailable"})
            continue
        rows.append({"clip_id": clip_id, "artist": meta.at[clip_id, "artist"],
                     "song_title": meta.at[clip_id, "song_title"], "youtube_id": r["youtube_id"],
                     "youtube_url": watch_url(r["youtube_id"]), "title": r["title"],
                     "channel_id": r["channel_id"], "channel_title": r["channel_title"],
                     "channel_url": f"https://www.youtube.com/channel/{r['channel_id']}",
                     "published_at": r["published_at"], "duration_s": r["duration_s"],
                     "view_count": r["view_count"], "views_snapshot_date": r["views_snapshot_date"],
                     "id_method": v["id_method"], "priority": meta.at[clip_id, "priority"]})
    out = pd.DataFrame(rows)
    out.to_csv(VIDEOS_CSV, index=False)
    pd.DataFrame(not_found).to_csv(NOT_FOUND_CSV, index=False)
    with open(CANDIDATES_JSONL, "w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    snaps = sorted({str(s)[:10] for s in out["views_snapshot_date"]}) if len(out) else []
    return {"clips": len(todo), "with_video": len(out), "by_method": stats, "snapshot_dates": snaps,
            "stopped": stopped, "serper_left_today": budget.remaining(), "youtube_quota": yt.budget.used()}
