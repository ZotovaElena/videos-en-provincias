"""Top-down candidates: official channels of seed artists → high-view official music videos.

1. Resolve seed artists (config/seed_artists.yaml + Wikipedia "most viewed" lists) to Wikidata
   items and their YouTube channel IDs (P2397).              → data/interim/seed_channels.csv
2. channels.list → uploads playlist → playlistItems.list → videos.list (no search.list).
                                                              → data/interim/channel_videos.csv
3. Official-music-video filter + view threshold.              → data/interim/top_videos.csv
4. Location evidence lookup per top video: existing evidence (by YouTube ID, or song + artist),
   then the song's Wikipedia article (es/en).                 → evidence/wikipedia_seed.jsonl
   Videos without filming information → data/processed/needs_location_research.csv
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import yaml

from vmap import config, wd
from vmap.extract import schema
from vmap.http import Client, HttpError, retrieved_at
from vmap.resolve.clips import artist_match, norm, norm_title, song_from_youtube_title
from vmap.sources import wikipedia as wp
from vmap.youtube.api import QuotaExceeded, YouTube

SEED_FILE = config.CONFIG / "seed_artists.yaml"
CHANNELS_CSV = config.INTERIM / "seed_channels.csv"
VIDEOS_CSV = config.INTERIM / "channel_videos.csv"
TOP_CSV = config.INTERIM / "top_videos.csv"
CHANNEL_STATUS_CSV = config.INTERIM / "seed_channel_status.csv"
RESEARCH_CSV = config.PROCESSED / "needs_location_research.csv"
MANUAL_CSV = config.MANUAL / "manual_evidence.csv"
MANUAL_COLUMNS = ["song_title", "artist", "youtube_id", "location", "source_url", "snippet",
                  "partial", "retrieved_at", "note"]

FILMING_RE = re.compile(
    r"(?i)\b(rodad[oa]s?|rod[óo]|rodaje|filmad[oa]s?|film[óo]|filmed|filming|shot (?:in|at|on)|"
    r"grabad[oa]s? en|se grab[óo] en|locations?\b|localizaci[óo]n)")
_DURATION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def iso_duration_s(d: str | None) -> int | None:
    m = _DURATION.fullmatch(d or "")
    if not m or not d:
        return None
    days, h, mi, s = (int(x or 0) for x in m.groups())
    return days * 86400 + h * 3600 + mi * 60 + s


# ---------------------------------------------------------------- 1. artists → channels
@dataclass
class Artist:
    name: str
    source: str
    qid: str | None = None
    label: str | None = None
    channel_ids: list[str] = field(default_factory=list)
    status: str = "ok"


def _resolve_name(client: Client, name: str) -> Artist:
    ids = wd.search(client, name)
    ents = wd.entities(client, ids)
    with_channel = [q for q in ids if q in ents and wd.claim_values(ents[q], "P2397")]
    artists = [q for q in with_channel if set(wd.claim_values(ents[q], "P31")) & wd.ARTIST_CLASSES]
    if not with_channel:
        return Artist(name, "yaml", status="no_channel_on_wikidata")
    q = artists[0] if artists else with_channel[0]
    return Artist(name, "yaml", q, wd.label(ents[q], q), wd.claim_values(ents[q], "P2397"),
                  "ok" if artists else "check_not_artist_class")


def parse_most_viewed_list(wikitext: str) -> tuple[list[str], list[str]]:
    """Linked page titles of the first two columns (video, uploader) and YouTube IDs in table rows."""
    titles, yt_ids = [], []
    for row in re.split(r"\n\|-", wikitext):
        cells = re.split(r"\n\||\|\|", row)
        for cell in cells[1:3]:
            m = re.search(r"\[\[([^\]|#]+)", cell)
            if m:
                titles.append(m.group(1).strip())
        yt_ids += wp.YOUTUBE_RE.findall(row)
    return list(dict.fromkeys(titles)), list(dict.fromkeys(yt_ids))


def _list_artists(client: Client, lang: str, title: str) -> tuple[list[Artist], list[str]]:
    d = client.get_json(f"https://{lang}.wikipedia.org/w/api.php", {
        "action": "parse", "page": title, "prop": "wikitext", "format": "json", "formatversion": 2})
    titles, yt_ids = parse_most_viewed_list(d["parse"]["wikitext"])
    qids = []
    for i in range(0, len(titles), 50):
        q = client.get_json(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "pageprops",
            "ppprop": "wikibase_item", "redirects": 1, "format": "json", "formatversion": 2})
        qids += [p["pageprops"]["wikibase_item"] for p in q["query"]["pages"]
                 if p.get("pageprops", {}).get("wikibase_item")]
    ents = wd.entities(client, qids)
    artist_qids = set()
    for q, e in ents.items():
        if wd.claim_values(e, "P2397") and set(wd.claim_values(e, "P31")) & wd.ARTIST_CLASSES:
            artist_qids.add(q)                                   # uploader is an artist
        artist_qids.update(wd.claim_values(e, "P175"))           # performers of the song
    arts = wd.entities(client, artist_qids)
    out = [Artist(wd.label(e, q), f"list:{title}", q, wd.label(e, q), wd.claim_values(e, "P2397"),
                  "ok" if wd.claim_values(e, "P2397") else "no_channel_on_wikidata")
           for q, e in arts.items()]
    return out, yt_ids


EXPANSION_QUERY = """
SELECT ?a ?aLabel ?ch WHERE {
  { ?a wdt:P31 wd:Q5 ; wdt:P27 wd:Q29 ; wdt:P106 ?occ .
    VALUES ?occ { wd:Q177220 wd:Q639669 wd:Q2252262 wd:Q488205 wd:Q130857 wd:Q36834
                  wd:Q753110 wd:Q855091 wd:Q386854 wd:Q1259917 } }
  UNION
  { ?a wdt:P31 ?cls ; wdt:P495 wd:Q29 .
    VALUES ?cls { wd:Q215380 wd:Q5741069 wd:Q2088357 wd:Q9212979 wd:Q281643 wd:Q216337 } }
  ?a wdt:P2397 ?ch .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "es,mul,en". }
}
"""


def expansion_artists(client: Client) -> list[Artist]:
    """Spanish musicians and bands with a YouTube channel ID on Wikidata."""
    d = client.get_json(config.search()["wikidata"]["endpoint"],
                        {"query": EXPANSION_QUERY, "format": "json"},
                        headers={"Accept": "application/sparql-results+json"})
    arts: dict[str, Artist] = {}
    for b in d["results"]["bindings"]:
        q = b["a"]["value"].rsplit("/", 1)[-1]
        a = arts.setdefault(q, Artist(b["aLabel"]["value"], "wikidata_expansion", q, b["aLabel"]["value"]))
        a.channel_ids.append(b["ch"]["value"])
    return list(arts.values())


def resolve_artists(client: Client) -> tuple[list[Artist], list[str]]:
    seed = yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))
    artists: list[Artist] = []
    for entry in seed.get("artists", []):
        if isinstance(entry, str):
            artists.append(_resolve_name(client, entry))
        elif entry.get("channel_ids"):
            artists.append(Artist(entry["name"], "yaml", entry.get("qid"), entry["name"],
                                  list(entry["channel_ids"]), "pinned_channels"))
        elif entry.get("qid"):
            e = wd.entities(client, [entry["qid"]]).get(entry["qid"], {})
            ch = wd.claim_values(e, "P2397")
            artists.append(Artist(entry["name"], "yaml", entry["qid"], wd.label(e, entry["name"]), ch,
                                  "pinned_qid" if ch else "no_channel_on_wikidata"))
    extra_ids: list[str] = []
    for lst in seed.get("wikipedia_lists", []):
        arts, ids = _list_artists(client, lst["lang"], lst["title"])
        artists += arts
        extra_ids += ids
    if (seed.get("wikidata_expansion") or {}).get("enabled"):
        artists += expansion_artists(client)
    # dedupe by Wikidata item (yaml entries first)
    seen, out = set(), []
    for a in artists:
        key = a.qid or a.name
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out, list(dict.fromkeys(extra_ids))


def search_channel(yt: YouTube, name: str) -> str | None:
    """YouTube channel search (search bucket, 1 call): best non-Topic channel whose title matches the name."""
    from rapidfuzz import fuzz

    d, _ = yt.call("search", {"part": "snippet", "q": name, "type": "channel", "maxResults": 5})
    best, best_score = None, 0
    for it in d.get("items", []):
        title = it["snippet"]["channelTitle"]
        if title.endswith(" - Topic"):
            continue
        score = fuzz.ratio(norm(title.removesuffix("VEVO").removesuffix("Official")), norm(name))
        if score > best_score:
            best, best_score = it["snippet"]["channelId"], score
    return best if best_score >= 85 else None


def fill_missing_channels(yt: YouTube, artists: list[Artist], chans: dict, log=print) -> list[str]:
    """For hand-listed artists without a usable (non-Topic) channel, search YouTube. Returns new channel IDs."""
    new = []
    for a in artists:
        if a.source != "yaml" or a.status.startswith("pinned"):
            continue
        usable = [c for c in a.channel_ids if c in chans and not chans[c]["title"].endswith(" - Topic")]
        if usable:
            continue
        try:
            ch = search_channel(yt, a.name)
        except QuotaExceeded as e:
            log(f"  channel search stopped: {e}")
            break
        if ch:
            a.channel_ids.append(ch)
            a.status = "channel_from_youtube_search"
            new.append(ch)
        else:
            a.status = "no_channel_found"
    return new


# ---------------------------------------------------------------- 2. channels → videos
def channel_info(yt: YouTube, channel_ids: list[str]) -> dict[str, dict]:
    info = {}
    ids = sorted(set(channel_ids))
    for i in range(0, len(ids), 50):
        d, _ = yt.call("channels", {"part": "snippet,contentDetails,statistics",
                                    "id": ",".join(ids[i:i + 50]), "maxResults": 50})
        for c in d.get("items", []):
            info[c["id"]] = {"title": c["snippet"]["title"],
                             "uploads": c["contentDetails"]["relatedPlaylists"].get("uploads"),
                             "video_count": int(c["statistics"].get("videoCount", 0)),
                             "subscribers": None if c["statistics"].get("hiddenSubscriberCount")
                             else int(c["statistics"].get("subscriberCount", 0))}
    return info


def uploads(yt: YouTube, playlist_id: str) -> list[str]:
    ids, token = [], None
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if token:
            params["pageToken"] = token
        try:
            d, _ = yt.call("playlistItems", params)
        except HttpError as e:
            if e.status == 404:          # channel without a public uploads playlist
                return ids
            raise
        ids += [it["contentDetails"]["videoId"] for it in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token:
            return ids


def expansion_min_subs() -> int:
    seed = yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))
    return int((seed.get("wikidata_expansion") or {}).get("min_subscribers", 0))


def _uploads_first_page(yt: YouTube, c: dict) -> tuple[str, dict]:
    from vmap.youtube.api import BASE
    return BASE + "playlistItems", {"part": "contentDetails", "playlistId": c["uploads"],
                                    "maxResults": 50, "key": yt.key}


def plan_channels(chans: dict, channel_artist: dict, s: dict, min_subs: int, cached=lambda c: False) -> dict:
    """Status per channel, in paging order: hand-picked and list artists first, then expansion
    artists by subscribers (largest first)."""
    def prio(item):
        ch, c = item
        a = channel_artist.get(ch)
        return (a is not None and a.source == "wikidata_expansion", -(c.get("subscribers") or 0))

    out = {}
    for ch, c in sorted(chans.items(), key=prio):
        a = channel_artist.get(ch)
        st = {"title": c["title"], "artist": a.label if a else None, "source": a.source if a else None,
              "subscribers": c.get("subscribers"), "video_count": c["video_count"], "cached": False}
        if c["title"].endswith(" - Topic") or not c["uploads"]:
            st["status"] = "skipped_topic_or_no_uploads"
        elif c["video_count"] > s["max_channel_videos"]:
            st["status"] = "skipped_too_many_uploads"
        elif a and a.source == "wikidata_expansion" and c.get("subscribers") is not None \
                and c["subscribers"] < min_subs:
            st["status"] = "skipped_few_subscribers"
        else:
            st["status"] = "to_page"
            st["cached"] = cached(c)
        out[ch] = st
    return out


def video_stats(yt: YouTube, video_ids: list[str], log=print, refresh: bool = False) -> list[dict]:
    rows = []
    ids = list(dict.fromkeys(video_ids))
    for i in range(0, len(ids), 50):
        try:
            d, r = yt.call("videos", {"part": "snippet,statistics,contentDetails,status",
                                      "id": ",".join(ids[i:i + 50]), "maxResults": 50}, refresh=refresh)
        except QuotaExceeded as e:
            log(f"  STOPPED fetching views at {i}/{len(ids)} videos: {e}")
            break
        snap = retrieved_at(r)
        for v in d.get("items", []):
            sn, st = v["snippet"], v.get("statistics", {})
            rows.append({
                "youtube_id": v["id"], "title": sn["title"], "channel_id": sn["channelId"],
                "channel_title": sn["channelTitle"], "published_at": sn.get("publishedAt"),
                "category_id": sn.get("categoryId"),
                "duration_s": iso_duration_s(v.get("contentDetails", {}).get("duration")),
                "view_count": int(st["viewCount"]) if "viewCount" in st else None,
                "made_for_kids": bool(v.get("status", {}).get("madeForKids", False)),
                "views_snapshot_date": snap,
            })
    return rows


def exclusion_reason(row: dict, s: dict) -> str:
    title = row["title"] or ""
    if (row["channel_title"] or "").endswith(" - Topic"):
        return "topic_channel"
    if row["view_count"] is None:
        return "no_view_count"
    if row.get("made_for_kids"):
        return "made_for_kids"
    if row["view_count"] < s["min_views"]:
        return "below_min_views"
    if str(row["category_id"]) not in set(map(str, s["allowed_categories"])):
        return f"category_{row['category_id']}"
    lo, hi = s["duration_s"]
    if row["duration_s"] is None or not lo <= row["duration_s"] <= hi:
        return "duration"
    if re.search(s["exclude_title_regex"], title):
        return "title_not_music_video"
    if re.search(s["exclude_live_regex"], title):
        return "title_live"
    return ""


# ---------------------------------------------------------------- 4. evidence lookup
def _ev_song(ev: schema.Evidence) -> str | None:
    return ev.song_title or (wp.clean_title(ev.page_title) if ev.page_kind == "song" else None)


class EvidenceIndex:
    """Find evidence for a video by YouTube ID, or by normalised song title + matching artist."""

    def __init__(self, evidence: list[schema.Evidence]):
        self.by_yt: dict[str, list] = {}
        self.by_title: dict[str, list] = {}
        for e in evidence:
            for y in e.youtube_ids:
                self.by_yt.setdefault(y, []).append(e)
            if (song := _ev_song(e)):
                self.by_title.setdefault(norm_title(song), []).append(e)

    def find(self, video: dict) -> list[schema.Evidence]:
        out = {e.evidence_id: e for e in self.by_yt.get(video["youtube_id"], [])}
        for e in self.by_title.get(norm_title(video["song_title_guess"]), []):
            if artist_match(e.artist, video["artist"]) or artist_match(e.artist, video["channel_title"]):
                out[e.evidence_id] = e
        return list(out.values())


def _status(evs: list[schema.Evidence]) -> str:
    if any(e.source_type == "manual" for e in evs):
        return "manual_evidence"
    if any(e.source_type == "wikidata" for e in evs):
        return "wikidata_location"
    if any(FILMING_RE.search(e.evidence_text) for e in evs):
        return "filming_info"
    if evs:
        return "article_no_filming_info"
    return "no_article"


def find_song_pages(client: Client, videos: list[dict], langs: list[str], log=print) -> list[schema.Evidence]:
    """Search es/en Wikipedia for each video's song article; keep exact-title pages that name the artist."""
    cfg = config.search()["wikipedia"]
    records: list[schema.Evidence] = []
    for lang in langs:
        pageids: list[int] = []
        for n, v in enumerate(videos, 1):
            q = f'"{v["song_title_guess"]}" {v["artist"]}'
            d = client.get_json(f"https://{lang}.wikipedia.org/w/api.php", {
                "action": "query", "list": "search", "srsearch": q, "srnamespace": 0, "srlimit": 3,
                "srprop": "", "format": "json", "formatversion": 2})
            pageids += [h["pageid"] for h in d.get("query", {}).get("search", [])]
            if n % 200 == 0:
                log(f"  [{lang}] searched {n}/{len(videos)}")
        pages = wp.fetch_pages(client, lang, pageids=list(dict.fromkeys(pageids)))
        wanted = {}
        for v in videos:
            wanted.setdefault(norm_title(v["song_title_guess"]), []).append(v)
        keep = []
        for p in pages:
            cands = wanted.get(norm_title(wp.clean_title(p["title"])), [])
            text = norm(p["wikitext"][:20000])
            if any(norm(v["artist"]) and norm(v["artist"]) in text for v in cands):
                keep.append(p)
        info = wp.item_info(client, [p["qid"] for p in keep if p["qid"]])
        records += [e for p in keep for e in wp.page_to_evidence(p, info, cfg)]
        log(f"  [{lang}] {len(keep)} song articles kept of {len(pages)} search results")
    return records


# ---------------------------------------------------------------- pipeline
def run(lookup: bool = True, log=print) -> dict:
    s = config.settings()["youtube"]
    client = Client(stage="seed")
    yt = YouTube(Client(stage="seed:youtube"))

    artists, extra_ids = resolve_artists(client)
    pd.DataFrame([{**a.__dict__, "channel_ids": " ".join(a.channel_ids)} for a in artists]) \
        .to_csv(CHANNELS_CSV, index=False)
    log(f"artists: {len(artists)} ({sum(1 for a in artists if a.channel_ids)} with channels); "
        f"{len(extra_ids)} video IDs from lists")

    channel_artist: dict[str, Artist] = {}
    for a in artists:
        for ch in a.channel_ids:
            channel_artist.setdefault(ch, a)
    chans = channel_info(yt, list(channel_artist))
    found = fill_missing_channels(yt, artists, chans, log)
    if found:
        for a in artists:
            for ch in a.channel_ids:
                channel_artist.setdefault(ch, a)
        chans.update(channel_info(yt, found))
        log(f"  {len(found)} channels found by YouTube channel search")
    pd.DataFrame([{**a.__dict__, "channel_ids": " ".join(a.channel_ids)} for a in artists]) \
        .to_csv(CHANNELS_CSV, index=False)
    statuses = plan_channels(chans, channel_artist, s, expansion_min_subs(),
                             cached=lambda c: yt.client.cached(*_uploads_first_page(yt, c)) is not None)
    video_ids: list[str] = list(extra_ids)
    for ch, st in statuses.items():
        if st["status"] != "to_page":
            continue
        c = chans[ch]
        pages = -(-c["video_count"] // 50)
        # keep enough budget for videos.list on everything collected so far plus this channel
        reserve = -(-(len(video_ids) + c["video_count"]) // 50) + 5
        if yt.budget.remaining() < pages + reserve and not st["cached"]:
            st["status"] = "pending_quota"
            continue
        try:
            video_ids += uploads(yt, c["uploads"])
        except QuotaExceeded:
            st["status"] = "pending_quota"
            continue
        st["status"] = "paged"
    pd.DataFrame([{"channel_id": ch, **st} for ch, st in statuses.items()]) \
        .to_csv(CHANNEL_STATUS_CSV, index=False)
    counts = pd.Series([st["status"] for st in statuses.values()]).value_counts().to_dict()
    log(f"channels: {len(chans)} found; {counts}; {len(set(video_ids))} video IDs")
    if counts.get("pending_quota"):
        log(f"  {counts['pending_quota']} channels left for the next run (daily YouTube budget)")

    rows = video_stats(yt, video_ids, log)
    for r in rows:
        a = channel_artist.get(r["channel_id"])
        r["artist"] = (a.label or a.name) if a else r["channel_title"].removesuffix("VEVO").strip()
        r["artist_qid"] = a.qid if a else None
        r["song_title_guess"] = song_from_youtube_title(r["title"], r["artist"])
        r["excluded_reason"] = exclusion_reason(r, s)
    videos = pd.DataFrame(rows).sort_values("view_count", ascending=False)
    videos.to_csv(VIDEOS_CSV, index=False)
    top = videos[videos["excluded_reason"] == ""].copy()

    # 4. evidence lookup
    index = EvidenceIndex(schema.load_all())
    top_rows = top.to_dict("records")
    lookup_min = s.get("lookup_min_views", 0)
    unmatched = [v for v in top_rows if v["view_count"] >= lookup_min and not index.find(v)]
    if lookup and unmatched:
        log(f"looking up Wikipedia song articles for {len(unmatched)} videos without evidence")
        found = find_song_pages(client, unmatched, config.search()["wikipedia"]["langs"], log)
        old = schema.load("wikipedia_seed")
        schema.write("wikipedia_seed", old + found)
        index = EvidenceIndex(schema.load_all())
    for v in top_rows:
        evs = index.find(v)
        v["evidence_status"] = _status(evs)
        v["n_evidence"] = len(evs)
        v["evidence_urls"] = " ".join(dict.fromkeys(e.source_url for e in evs))
    top = pd.DataFrame(top_rows)
    top.to_csv(TOP_CSV, index=False)

    research = top[top["evidence_status"].isin(["no_article", "article_no_filming_info"])
                   & (top["view_count"] >= lookup_min)].copy()
    research["youtube_url"] = "https://www.youtube.com/watch?v=" + research["youtube_id"]
    research[["youtube_url", "song_title_guess", "artist", "title", "channel_title", "view_count",
              "views_snapshot_date", "evidence_status", "evidence_urls"]] \
        .to_csv(RESEARCH_CSV, index=False)
    if not MANUAL_CSV.exists():
        MANUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(MANUAL_CSV, index=False)

    return {"artists": len(artists), "channels": len(chans), "videos": len(videos),
            "excluded": videos["excluded_reason"].value_counts().to_dict(),
            "top": len(top), "status": top["evidence_status"].value_counts().to_dict(),
            "research": len(research), "quota": yt.budget.used()}
