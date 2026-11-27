"""Evidence record → clip (song + artist), and → YouTube candidate videos from the top-down stage.

Link methods, strongest first:
  youtube_id       a YouTube ID in the source (Wikipedia external link, Wikidata P1651) is a candidate video
  article_url      the source page was already linked to the video by the seed stage (song article), and the
                   song matches (or the page is about one song only)
  artist_title     artist and song title of the record match a candidate (fuzzy, ≥ TITLE_MIN)
  title_in_text    artist matches and the candidate's song title appears word-for-word in the passage
  text_only        no candidate video: the clip is identified by the record's own song + artist
A record with no usable link is `unlinked` (reported, not used).
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from vmap.extract.schema import Evidence
from vmap.resolve import clips as cl

TITLE_MIN = 90
STRONG = {"youtube_id", "article_url", "artist_title"}


def clip_id(artist: str, title: str, label_note: str = "") -> str:
    return hashlib.sha1(f"{cl.norm_artist(artist)}|{cl.norm_title(title)}|{label_note}".encode()).hexdigest()


def _words_in(needle: str, hay: str) -> bool:
    return bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


class Linker:
    def __init__(self, top: pd.DataFrame):
        top = top.copy()
        top["song"] = top["song_title_guess"].fillna("").astype(str)
        top["t_norm"] = top["song"].map(cl.norm_title)
        self.top = top.set_index("youtube_id", drop=False)
        self.by_url: dict[str, list[str]] = defaultdict(list)
        for yid, urls in zip(top["youtube_id"], top["evidence_urls"].fillna("")):
            for u in str(urls).split():
                self.by_url[u].append(yid)
        self.by_artist: dict[str, list[str]] = defaultdict(list)
        for yid, a in zip(top["youtube_id"], top["artist"]):
            for part in cl.split_artists(a):
                self.by_artist[part].append(yid)

    def _artist_candidates(self, artist: str | None) -> list[str]:
        ids = {y for part in cl.split_artists(artist) for y in self.by_artist.get(part, [])}
        return [y for y in ids if cl.artist_match(artist, self.top.at[y, "artist"])]

    def link(self, ev: Evidence) -> tuple[list[str], str]:
        """(candidate YouTube IDs, method). Empty list with method text_only / unlinked otherwise."""
        ids = [y for y in ev.youtube_ids if y in self.top.index]
        t = cl.norm_title(ev.song_title)
        if t:      # the page names its song: other songs' videos linked from the page do not count
            ids = [y for y in ids if fuzz.ratio(self.top.at[y, "t_norm"], t) >= TITLE_MIN]
        elif len({self.top.at[y, "t_norm"] for y in ids}) > 1:
            ids = []
        if ids:
            return ids, "youtube_id"

        text = cl.norm(ev.evidence_text + " " + (ev.page_title or ""))
        by_page = self.by_url.get(ev.source_url, [])
        if by_page:
            if ev.song_title:
                hit = [y for y in by_page
                       if fuzz.ratio(self.top.at[y, "t_norm"], cl.norm_title(ev.song_title)) >= TITLE_MIN]
                if hit:
                    return hit, "article_url"
            if ev.page_kind == "song" and len({self.top.at[y, "t_norm"] for y in by_page}) == 1:
                return by_page, "article_url"
            hit = [y for y in by_page if len(self.top.at[y, "t_norm"]) >= 4
                   and _words_in(self.top.at[y, "t_norm"], text)]
            if len({self.top.at[y, "t_norm"] for y in hit}) == 1:
                return hit, "article_url"

        cands = self._artist_candidates(ev.artist) if ev.artist else []
        if cands and ev.song_title:
            t = cl.norm_title(ev.song_title)
            hit = [y for y in cands if fuzz.ratio(self.top.at[y, "t_norm"], t) >= TITLE_MIN]
            if hit:
                return hit, "artist_title"
        if cands:
            hit = [y for y in cands if len(self.top.at[y, "t_norm"]) >= 4 and _words_in(self.top.at[y, "t_norm"], text)]
            titles = {self.top.at[y, "t_norm"] for y in hit}
            if len(titles) == 1:
                return hit, "title_in_text"
        if ev.song_title and ev.artist:
            return [], "text_only"
        return [], "unlinked"

    def clip_for(self, ev: Evidence, ids: list[str]) -> dict:
        """Clip identity: from the most-viewed linked video, else from the record itself."""
        if ids:
            v = self.top.loc[ids].sort_values("view_count", ascending=False).iloc[0]
            artist, song = v["artist"], v["song"]
        else:
            artist, song = ev.artist, ev.song_title
        return {"clip_id": clip_id(artist, song), "artist": artist, "song_title": song}


def near_duplicates(clips: pd.DataFrame, cutoff: int = TITLE_MIN) -> pd.DataFrame:
    """Pairs of clips by the same artist with very similar (not equal) titles → review, never merged."""
    rows = []
    clips = clips.assign(a=clips["artist"].map(cl.norm_artist), t=clips["song_title"].map(cl.norm_title))
    for _, grp in clips.groupby("a"):
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                s = fuzz.ratio(recs[i]["t"], recs[j]["t"])
                if recs[i]["t"] != recs[j]["t"] and s >= cutoff:
                    rows.append({"clip_id_a": recs[i]["clip_id"], "clip_id_b": recs[j]["clip_id"],
                                 "artist": recs[i]["artist"], "title_a": recs[i]["song_title"],
                                 "title_b": recs[j]["song_title"], "score": round(s, 1)})
    return pd.DataFrame(rows, columns=["clip_id_a", "clip_id_b", "artist", "title_a", "title_b", "score"])
